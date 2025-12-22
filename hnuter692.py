import numpy as np
import mujoco as mj
import mujoco.viewer as viewer
import time
import math
import csv
import os
from typing import Tuple, List, Optional
from datetime import datetime

class HnuterController:
    def __init__(self, model_path: str = "scene.xml"):
        # 加载MuJoCo模型
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        
        # 打印模型诊断信息
        self._print_model_diagnostics()
        
        # ===================== 1. 核心物理参数 =====================
        self.dt = self.model.opt.timestep
        self.gravity = 9.81
        self.mass = 4.2  # 主机身质量 + 旋翼机构质量 4.2kg
        self.J = np.diag([0.08, 0.12, 0.1])  # 惯量矩阵 (kg·m²)
        self.J_inv = np.linalg.inv(self.J)    # 惯量矩阵逆
        
        # 旋翼布局参数（机体坐标系下）
        self.l1 = 0.3  # 前旋翼组Y向距离(m)
        self.l2 = 0.5  # 尾部推进器X向距离(m)
        self.rotor_positions = np.array([
            [0, self.l1, 0],   # 旋翼1 (左前)
            [0, -self.l1, 0],  # 旋翼2 (右前)
            [self.l2, 0, 0]    # 旋翼3 (尾部)
        ])  # 3个旋翼组的位置矢量 (机体系)
        
        # ===================== 2. 坐标系与符号定义 =====================
        # 惯性系 (I): 世界固定坐标系 (ENU)
        # 机体系 (B): 固连无人机中心 (x:机头, y:右机臂, z:机腹上)
        # 虚拟倾转系 (V): 绕B系x轴旋转平均倾转角得到
        
        # 虚拟系相关变量
        self.avg_tilt_angle = 0.0  # 平均倾转角 (θ_avg)
        self.R_BV = np.eye(3)      # 机体系→虚拟系旋转矩阵
        self.R_VB = np.eye(3)      # 虚拟系→机体系旋转矩阵
        
        # ===================== 3. 增益参数 (动态增益矩阵用) =====================
        # 推力差控制高增益 (快响应轴)
        self.K_high = np.diag([3.0, 2.5, 0.8])  # [roll, pitch, yaw]
        # 舵机控制低增益 (慢响应轴)
        self.K_low = np.diag([0.8, 0.6, 0.5])   # [roll, pitch, yaw]
        # 位置控制器增益
        self.Kp_pos = np.diag([6.0, 6.0, 6.0])  # 位置比例增益
        self.Kd_pos = np.diag([5.0, 5.0, 5.0])  # 速度阻尼增益
        
        # ===================== 4. WLS分配参数 =====================
        # 权重矩阵 (优先保证Z力和俯仰力矩)
        self.W = np.diag([10.0, 1.0, 10.0, 1.0, 10.0, 1.0])  # [Fx, Fy, Fz, τx, τy, τz]
        self.W_inv = np.linalg.inv(self.W)
        # 推力饱和限制
        self.T_min = 0.0   # 最小推力 (N)
        self.T_max = 60.0  # 最大推力 (N)
        self.T_rear_min = -15.0  # 尾部推进器最小推力 (可反向)
        
        # ===================== 5. 状态与目标变量 =====================
        # 惯性系状态
        self.pos_I = np.zeros(3)       # 位置 (m)
        self.vel_I = np.zeros(3)       # 速度 (m/s)
        self.acc_I = np.zeros(3)       # 加速度 (m/s²)
        # 机体系状态
        self.R_BI = np.eye(3)          # 机体系→惯性系旋转矩阵
        self.omega_B = np.zeros(3)     # 机体角速度 (rad/s)
        # 关节反馈 (实测倾转角)
        self.tilt_angles_meas = np.zeros((3, 2))  # [roll_tilt, pitch_tilt] per rotor
        # 目标状态
        self.pos_I_des = np.array([0.0, 0.0, 2.0])  # 目标位置
        self.vel_I_des = np.zeros(3)                # 目标速度
        self.acc_I_des = np.zeros(3)                # 目标加速度
        self.yaw_des = 0.0                          # 目标偏航角 (rad)
        
        # ===================== 6. 控制输出变量 =====================
        self.F_des_B = np.zeros(3)    # 期望机体力 (N)
        self.tau_des_B = np.zeros(3)  # 期望机体力矩 (N·m)
        self.T_des = np.zeros(3)      # 期望推力 (N) [T1, T2, T3]
        self.tilt_des = np.zeros((3, 2))  # 期望倾转角 [roll, pitch] per rotor
        
        # ===================== 7. 执行器/传感器ID映射 =====================
        self._get_actuator_ids()
        self._get_sensor_ids()
        
        # ===================== 8. 日志与轨迹控制 =====================
        self._create_log_file()
        # 90°大角度轨迹控制
        self.trajectory_phase = 0
        self.attitude_target_rad = np.pi*2/5  # 72°目标角
        self.phase_start_time = 0.0
        self._last_print_time = 0.0
        
        print("几何-动力学混合控制器初始化完成")
        print("核心架构：虚拟标架解耦 + 瞬时雅可比WLS分配 + 快慢回路分离")
    
    # ===================== 基础工具函数 =====================
    def _print_model_diagnostics(self):
        """打印模型诊断信息"""
        print("\n=== 模型诊断信息 ===")
        print(f"广义坐标数量 (nq): {self.model.nq}")
        print(f"速度自由度 (nv): {self.model.nv}")
        print(f"执行器数量 (nu): {self.model.nu}")
        print(f"身体数量: {self.model.nbody}")
        print(f"关节数量: {self.model.njnt}")
        
        # 检查drone主体
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'drone')
        if body_id != -1:
            print(f"Drone主体ID: {body_id}")
    
    def _create_log_file(self):
        """创建日志文件"""
        if not os.path.exists('logs'):
            os.makedirs('logs')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f'logs/hybrid_controller_{timestamp}.csv'
        
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'time', 'pos_x', 'pos_y', 'pos_z', 
                'roll', 'pitch', 'yaw',
                'tilt1_roll', 'tilt1_pitch', 'tilt2_roll', 'tilt2_pitch', 'tilt3_roll', 'tilt3_pitch',
                'T1', 'T2', 'T3',
                'F_des_x', 'F_des_y', 'F_des_z',
                'tau_des_x', 'tau_des_y', 'tau_des_z',
                'avg_tilt_angle', 'trajectory_phase'
            ])
    
    def _get_actuator_ids(self):
        """获取执行器ID映射"""
        self.actuator_ids = {}
        # 倾转舵机 (慢回路)
        tilt_actuators = [
            'tilt_roll_left', 'tilt_pitch_left',
            'tilt_roll_right', 'tilt_pitch_right',
            'tilt_roll_rear', 'tilt_pitch_rear'
        ]
        # 推力电机 (快回路)
        thrust_actuators = [
            'motor_l_upper', 'motor_l_lower',
            'motor_r_upper', 'motor_r_lower',
            'motor_rear_upper'
        ]
        
        for name in tilt_actuators + thrust_actuators:
            try:
                self.actuator_ids[name] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
            except:
                print(f"警告：未找到执行器 {name}，使用默认ID")
                self.actuator_ids[name] = -1
        
        print("执行器ID映射:", {k: v for k, v in self.actuator_ids.items() if v != -1})
    
    def _get_sensor_ids(self):
        """获取传感器ID映射 (关节角/姿态/速度)"""
        self.sensor_ids = {}
        sensor_names = [
            'drone_pos', 'drone_quat', 'body_vel', 'body_gyro',
            'tilt_roll_left_pos', 'tilt_pitch_left_pos',
            'tilt_roll_right_pos', 'tilt_pitch_right_pos',
            'tilt_roll_rear_pos', 'tilt_pitch_rear_pos'
        ]
        
        for name in sensor_names:
            try:
                self.sensor_ids[name] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, name)
            except:
                self.sensor_ids[name] = -1
        
        print("传感器ID映射:", {k: v for k, v in self.sensor_ids.items() if v != -1})
    
    # ===================== 坐标系转换核心函数 =====================
    def quat_to_rot_mat(self, quat: np.ndarray) -> np.ndarray:
        """四元数转旋转矩阵 (w, x, y, z) → R"""
        w, x, y, z = quat
        R = np.array([
            [1-2*y²-2*z², 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x²-2*z², 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x²-2*y²]
        ])
        return R
    
    def rot_mat_to_euler(self, R: np.ndarray) -> np.ndarray:
        """旋转矩阵转欧拉角 (roll, pitch, yaw)"""
        roll = np.arctan2(R[2,1], R[2,2])
        pitch = np.arcsin(-R[2,0])
        yaw = np.arctan2(R[1,0], R[0,0])
        return np.array([roll, pitch, yaw])
    
    def update_virtual_frame(self):
        """更新虚拟倾转系 (V) 与机体系 (B) 的旋转映射"""
        # 计算平均倾转角 (所有旋翼俯仰倾转角的均值)
        pitch_tilts = self.tilt_angles_meas[:, 1]  # 所有旋翼的俯仰倾转角
        self.avg_tilt_angle = np.mean(pitch_tilts)
        
        # 构建机体系→虚拟系旋转矩阵 (绕x轴旋转avg_tilt_angle)
        c = np.cos(self.avg_tilt_angle)
        s = np.sin(self.avg_tilt_angle)
        self.R_BV = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
        self.R_VB = self.R_BV.T  # 逆旋转 = 转置
    
    def compute_dynamic_gain(self) -> np.ndarray:
        """
        计算动态增益矩阵 K_B (机体系)
        公式：K_B = R_VB · K_V · R_BV
        其中 K_V 是虚拟系下的对角增益矩阵 (快轴高增益，慢轴低增益)
        """
        # 虚拟系下的增益矩阵 (快轴:垂直旋翼轴, 慢轴:平行旋翼轴)
        K_V = (1 - np.abs(np.sin(self.avg_tilt_angle))) * self.K_high + \
              np.abs(np.sin(self.avg_tilt_angle)) * self.K_low
        
        # 映射回机体系
        K_B = self.R_VB @ K_V @ self.R_BV
        return K_B
    
    # ===================== SE3几何控制器 (核心) =====================
    def se3_geometric_control(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        SE(3)几何控制器：计算期望机体力和力矩
        输出: F_des_B (机体系期望力), tau_des_B (机体系期望力矩)
        """
        # 1. 位置误差与期望加速度
        pos_error = self.pos_I_des - self.pos_I
        vel_error = self.vel_I_des - self.vel_I
        acc_des_I = self.acc_I_des + self.Kp_pos @ pos_error + self.Kd_pos @ vel_error
        
        # 2. 计算期望机体力 (惯性系→机体系)
        F_des_I = self.mass * (acc_des_I + np.array([0, 0, self.gravity]))
        F_des_B = self.R_BI @ F_des_I  # 惯性系→机体系
        
        # 3. 姿态误差计算 (基于旋转矩阵)
        # 目标旋转矩阵 (仅跟踪偏航角，roll/pitch由力分配决定)
        yaw = self.yaw_des
        R_des = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        # 旋转误差 (李代数)
        e_R = 0.5 * self.vee_map(R_des.T @ self.R_BI - self.R_BI.T @ R_des)
        
        # 4. 角速度误差
        omega_des_B = np.zeros(3)  # 目标角速度
        e_omega = self.omega_B - omega_des_B
        
        # 5. 动态增益矩阵
        K_B = self.compute_dynamic_gain()
        D_B = np.diag([1.2, 0.8, 1.5])  # 阻尼矩阵
        
        # 6. 计算期望力矩 (含陀螺项补偿)
        tau_des_B = -K_B @ e_R - D_B @ e_omega + np.cross(self.omega_B, self.J @ self.omega_B)
        
        return F_des_B, tau_des_B
    
    # ===================== 雅可比矩阵与WLS分配 =====================
    def rotor_thrust_vector(self, rotor_idx: int) -> np.ndarray:
        """
        计算单个旋翼的推力单位矢量 (机体系)
        输入: rotor_idx - 旋翼索引 (0,1,2)
        输出: u_i - 推力单位矢量 (机体系)
        """
        roll_tilt, pitch_tilt = self.tilt_angles_meas[rotor_idx]
        
        # 旋转矩阵：先绕y轴(roll)，再绕x轴(pitch)
        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll_tilt), -np.sin(roll_tilt)],
            [0, np.sin(roll_tilt), np.cos(roll_tilt)]
        ])
        R_pitch = np.array([
            [np.cos(pitch_tilt), 0, np.sin(pitch_tilt)],
            [0, 1, 0],
            [-np.sin(pitch_tilt), 0, np.cos(pitch_tilt)]
        ])
        R_tilt = R_pitch @ R_roll
        
        # 初始推力方向 (z轴) → 倾转后方向
        u_i = R_tilt @ np.array([0, 0, 1])
        return u_i
    
    def construct_jacobian(self) -> np.ndarray:
        """
        构造瞬时雅可比矩阵 J (6x3)
        行: [Fx, Fy, Fz, τx, τy, τz]
        列: [旋翼1, 旋翼2, 旋翼3]
        """
        J = np.zeros((6, 3))
        
        for i in range(3):
            # 第i个旋翼的推力矢量
            u_i = self.rotor_thrust_vector(i)
            # 位置矢量 (机体系)
            r_i = self.rotor_positions[i]
            # 力矩贡献 (r_i × u_i)
            tau_i = np.cross(r_i, u_i)
            
            # 填充雅可比矩阵
            J[:3, i] = u_i
            J[3:, i] = tau_i
        
        return J
    
    def wls_thrust_allocation(self, F_des: np.ndarray, tau_des: np.ndarray) -> np.ndarray:
        """
        加权最小二乘推力分配
        输入: F_des - 期望力, tau_des - 期望力矩
        输出: T_des - 各旋翼期望推力 [T1, T2, T3]
        """
        # 构造期望旋量
        wrench_des = np.concatenate([F_des, tau_des])
        
        # 构造瞬时雅可比矩阵
        J = self.construct_jacobian()
        
        # WLS求解 (公式: T = (J^T W J)^-1 J^T W wrench_des)
        J_T_W = J.T @ self.W
        try:
            T_des = np.linalg.inv(J_T_W @ J) @ J_T_W @ wrench_des
        except np.linalg.LinAlgError:
            # 奇异时用伪逆
            T_des = np.linalg.pinv(J) @ wrench_des
        
        # 推力饱和限制
        T_des[0] = np.clip(T_des[0], self.T_min, self.T_max)  # 左前旋翼
        T_des[1] = np.clip(T_des[1], self.T_min, self.T_max)  # 右前旋翼
        T_des[2] = np.clip(T_des[2], self.T_rear_min, self.T_max)  # 尾部旋翼
        
        return T_des
    
    # ===================== 慢回路：舵机角度分配 =====================
    def tilt_angle_scheduling(self, F_des: np.ndarray) -> np.ndarray:
        """
        慢回路：计算期望倾转角 (几何前馈)
        输入: F_des - 期望机体力
        输出: tilt_des - 各旋翼期望倾转角 [ (roll1,pitch1), (roll2,pitch2), (roll3,pitch3) ]
        """
        tilt_des = np.zeros((3, 2))
        
        # 期望力方向 (单位矢量)
        F_dir = F_des / np.linalg.norm(F_des) if np.linalg.norm(F_des) > 1e-6 else np.array([0,0,1])
        
        # 计算期望俯仰/滚转倾转角
        pitch_des = np.arcsin(F_dir[0])  # 沿x轴的俯仰角
        roll_des = np.arctan2(F_dir[1], F_dir[2])  # 沿y轴的滚转角
        
        # 分配到各旋翼 (简化版：前两个旋翼反向roll，尾部仅pitch)
        tilt_des[0] = [roll_des, pitch_des]    # 左前旋翼
        tilt_des[1] = [-roll_des, pitch_des]   # 右前旋翼
        tilt_des[2] = [0.0, pitch_des]         # 尾部旋翼
        
        # 角度限制 (±90°)
        tilt_des = np.clip(tilt_des, -np.pi/2, np.pi/2)
        
        return tilt_des
    
    # ===================== 状态估计 =====================
    def estimate_state(self):
        """状态估计：更新位置、速度、姿态、关节角反馈"""
        # 1. 机体位置与速度 (惯性系)
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'drone')
        if body_id != -1:
            self.pos_I = self.data.xpos[body_id].copy()
            self.vel_I = self.data.cvel[body_id][3:6].copy()
            # 旋转矩阵 (机体系→惯性系)
            quat = self.data.xquat[body_id].copy()
            self.R_BI = self.quat_to_rot_mat(quat)
            # 机体角速度 (机体系)
            self.omega_B = self.data.cvel[body_id][0:3].copy()
        
        # 2. 关节角反馈 (实测倾转角)
        # 左前旋翼
        if self.sensor_ids['tilt_roll_left_pos'] != -1:
            self.tilt_angles_meas[0, 0] = self.data.sensordata[self.sensor_ids['tilt_roll_left_pos']]
        if self.sensor_ids['tilt_pitch_left_pos'] != -1:
            self.tilt_angles_meas[0, 1] = self.data.sensordata[self.sensor_ids['tilt_pitch_left_pos']]
        
        # 右前旋翼
        if self.sensor_ids['tilt_roll_right_pos'] != -1:
            self.tilt_angles_meas[1, 0] = self.data.sensordata[self.sensor_ids['tilt_roll_right_pos']]
        if self.sensor_ids['tilt_pitch_right_pos'] != -1:
            self.tilt_angles_meas[1, 1] = self.data.sensordata[self.sensor_ids['tilt_pitch_right_pos']]
        
        # 尾部旋翼
        if self.sensor_ids['tilt_roll_rear_pos'] != -1:
            self.tilt_angles_meas[2, 0] = self.data.sensordata[self.sensor_ids['tilt_roll_rear_pos']]
        if self.sensor_ids['tilt_pitch_rear_pos'] != -1:
            self.tilt_angles_meas[2, 1] = self.data.sensordata[self.sensor_ids['tilt_pitch_rear_pos']]
        
        # 3. 更新虚拟标架
        self.update_virtual_frame()
    
    # ===================== 控制执行 =====================
    def update_control(self):
        """完整控制回路：状态估计→SE3控制→快慢回路分配→执行器输出"""
        try:
            # 1. 状态估计
            self.estimate_state()
            
            # 2. SE3几何控制：计算期望力和力矩
            self.F_des_B, self.tau_des_B = self.se3_geometric_control()
            
            # 3. 慢回路：舵机角度分配 (几何前馈)
            self.tilt_des = self.tilt_angle_scheduling(self.F_des_B)
            
            # 4. 快回路：推力分配 (WLS + 瞬时雅可比)
            self.T_des = self.wls_thrust_allocation(self.F_des_B, self.tau_des_B)
            
            # 5. 执行器输出
            self.set_actuators()
            
            # 6. 日志记录
            self.log_status()
            
            return True
        except Exception as e:
            print(f"控制更新失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def set_actuators(self):
        """设置执行器指令：慢回路(舵机) + 快回路(推力)"""
        # 慢回路：舵机角度指令
        # 左前旋翼
        if self.actuator_ids['tilt_roll_left'] != -1:
            self.data.ctrl[self.actuator_ids['tilt_roll_left']] = self.tilt_des[0, 0]
        if self.actuator_ids['tilt_pitch_left'] != -1:
            self.data.ctrl[self.actuator_ids['tilt_pitch_left']] = self.tilt_des[0, 1]
        
        # 右前旋翼
        if self.actuator_ids['tilt_roll_right'] != -1:
            self.data.ctrl[self.actuator_ids['tilt_roll_right']] = self.tilt_des[1, 0]
        if self.actuator_ids['tilt_pitch_right'] != -1:
            self.data.ctrl[self.actuator_ids['tilt_pitch_right']] = self.tilt_des[1, 1]
        
        # 尾部旋翼
        if self.actuator_ids['tilt_roll_rear'] != -1:
            self.data.ctrl[self.actuator_ids['tilt_roll_rear']] = self.tilt_des[2, 0]
        if self.actuator_ids['tilt_pitch_rear'] != -1:
            self.data.ctrl[self.actuator_ids['tilt_pitch_rear']] = self.tilt_des[2, 1]
        
        # 快回路：推力指令 (分配到上下电机)
        # 左前旋翼 (两个电机均分)
        T1 = self.T_des[0] / 2
        if self.actuator_ids['motor_l_upper'] != -1:
            self.data.ctrl[self.actuator_ids['motor_l_upper']] = T1
        if self.actuator_ids['motor_l_lower'] != -1:
            self.data.ctrl[self.actuator_ids['motor_l_lower']] = T1
        
        # 右前旋翼 (两个电机均分)
        T2 = self.T_des[1] / 2
        if self.actuator_ids['motor_r_upper'] != -1:
            self.data.ctrl[self.actuator_ids['motor_r_upper']] = T2
        if self.actuator_ids['motor_r_lower'] != -1:
            self.data.ctrl[self.actuator_ids['motor_r_lower']] = T2
        
        # 尾部旋翼
        T3 = self.T_des[2]
        if self.actuator_ids['motor_rear_upper'] != -1:
            self.data.ctrl[self.actuator_ids['motor_rear_upper']] = T3
    
    # ===================== 辅助函数 =====================
    def vee_map(self, S: np.ndarray) -> np.ndarray:
        """反对称矩阵→向量 (vee映射)"""
        return np.array([S[2,1], S[0,2], S[1,0]])
    
    def hat_map(self, v: np.ndarray) -> np.ndarray:
        """向量→反对称矩阵 (hat映射)"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    def log_status(self):
        """记录状态日志"""
        euler = self.rot_mat_to_euler(self.R_BI)
        timestamp = time.time()
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                self.pos_I[0], self.pos_I[1], self.pos_I[2],
                np.degrees(euler[0]), np.degrees(euler[1]), np.degrees(euler[2]),
                np.degrees(self.tilt_angles_meas[0,0]), np.degrees(self.tilt_angles_meas[0,1]),
                np.degrees(self.tilt_angles_meas[1,0]), np.degrees(self.tilt_angles_meas[1,1]),
                np.degrees(self.tilt_angles_meas[2,0]), np.degrees(self.tilt_angles_meas[2,1]),
                self.T_des[0], self.T_des[1], self.T_des[2],
                self.F_des_B[0], self.F_des_B[1], self.F_des_B[2],
                self.tau_des_B[0], self.tau_des_B[1], self.tau_des_B[2],
                np.degrees(self.avg_tilt_angle), self.trajectory_phase
            ])
    
    def print_status(self, current_time: float):
        """打印实时状态"""
        if current_time - self._last_print_time < 1.0:
            return
        self._last_print_time = current_time
        
        euler = self.rot_mat_to_euler(self.R_BI)
        print(f"\n=== 混合控制器状态 (t={current_time:.1f}s) ===")
        print(f"位置: X={self.pos_I[0]:.2f} Y={self.pos_I[1]:.2f} Z={self.pos_I[2]:.2f} m")
        print(f"姿态: Roll={np.degrees(euler[0]):.1f}° Pitch={np.degrees(euler[1]):.1f}° Yaw={np.degrees(euler[2]):.1f}°")
        print(f"平均倾转角: {np.degrees(self.avg_tilt_angle):.1f}°")
        print(f"期望力: Fx={self.F_des_B[0]:.1f} Fy={self.F_des_B[1]:.1f} Fz={self.F_des_B[2]:.1f} N")
        print(f"期望力矩: τx={self.tau_des_B[0]:.1f} τy={self.tau_des_B[1]:.1f} τz={self.tau_des_B[2]:.1f} N·m")
        print(f"推力指令: T1={self.T_des[0]:.1f} T2={self.T_des[1]:.1f} T3={self.T_des[2]:.1f} N")
        print("--------------------------------------------------")
    
    def update_trajectory(self, current_time: float):
        """90°大角度轨迹规划 (保持原有逻辑)"""
        if self.trajectory_phase == 0 and self.phase_start_time == 0.0:
            self.phase_start_time = current_time
        
        # 阶段时长配置
        phase_durations = {
            0:  6.0,   # 起飞悬停
            1:  12.0,  # Roll转动(0→72°)
            2:  20.0,  # Roll保持
            3:  6.0,   # Roll恢复
            4:  15.0,  # Roll后等待
            5:  12.0,  # Pitch转动(0→72°)
            6:  20.0,  # Pitch保持
            7:  6.0,   # Pitch恢复
            8:  15.0,  # Pitch后等待
            9:  12.0,  # Yaw转动(0→72°)
            10: 20.0,  # Yaw保持
            11: 6.0,   # Yaw恢复
            12: 15.0,  # Yaw后等待
            13: float('inf')  # 最终悬停
        }
        
        phase_elapsed = current_time - self.phase_start_time
        
        if phase_elapsed > phase_durations[self.trajectory_phase]:
            self.trajectory_phase += 1
            self.phase_start_time = current_time
            print(f"\n🔄 轨迹阶段切换: {self.trajectory_phase-1} → {self.trajectory_phase}")
        
        # 各阶段目标状态
        self.pos_I_des = np.array([0.0, 0.0, 2.0])  # 固定高度
        self.vel_I_des = np.zeros(3)
        self.acc_I_des = np.zeros(3)
        
        if self.trajectory_phase == 0:
            self.yaw_des = 0.0
        elif self.trajectory_phase == 1:
            progress = np.clip(phase_elapsed / phase_durations[1], 0, 1)
            self.yaw_des = progress * self.attitude_target_rad
        elif self.trajectory_phase == 2:
            self.yaw_des = self.attitude_target_rad
        elif self.trajectory_phase == 3:
            progress = np.clip(phase_elapsed / phase_durations[3], 0, 1)
            self.yaw_des = (1 - progress) * self.attitude_target_rad
        elif self.trajectory_phase == 4:
            self.yaw_des = 0.0
        elif self.trajectory_phase == 5:
            progress = np.clip(phase_elapsed / phase_durations[5], 0, 1)
            self.yaw_des = progress * self.attitude_target_rad
        elif self.trajectory_phase == 6:
            self.yaw_des = self.attitude_target_rad
        elif self.trajectory_phase == 7:
            progress = np.clip(phase_elapsed / phase_durations[7], 0, 1)
            self.yaw_des = (1 - progress) * self.attitude_target_rad
        elif self.trajectory_phase == 8:
            self.yaw_des = 0.0
        elif self.trajectory_phase == 9:
            progress = np.clip(phase_elapsed / phase_durations[9], 0, 1)
            self.yaw_des = progress * self.attitude_target_rad
        elif self.trajectory_phase == 10:
            self.yaw_des = self.attitude_target_rad
        elif self.trajectory_phase == 11:
            progress = np.clip(phase_elapsed / phase_durations[11], 0, 1)
            self.yaw_des = (1 - progress) * self.attitude_target_rad
        elif self.trajectory_phase >= 12:
            self.yaw_des = 0.0

# ===================== 主函数 =====================
def main():
    print("=== 倾转旋翼无人机 几何-动力学混合控制器 ===")
    print("核心特性：虚拟标架解耦 + 瞬时雅可比WLS + 快慢回路分离")
    print("轨迹逻辑：90°大角度姿态跟踪 + 长等待时间")
    
    try:
        controller = HnuterController("hnuter201.xml")
        controller.attitude_target_rad = np.pi*2/5  # 72°目标角
        
        with viewer.launch_passive(controller.model, controller.data) as v:
            print(f"\n仿真启动：日志文件路径: {controller.log_file}")
            print("按 Ctrl+C 终止仿真")
            
            start_time = time.time()
            count = 0
            
            try:
                while v.is_running():
                    current_time = time.time() - start_time
                    
                    # 更新轨迹
                    controller.update_trajectory(current_time)
                    # 更新控制
                    controller.update_control()
                    # 打印状态
                    controller.print_status(current_time)

                    # 仿真步进
                    count += 1
                    if count % 1 == 0:
                        mj.mj_step(controller.model, controller.data)
                    
                    v.sync()
                    time.sleep(0.001)

            except KeyboardInterrupt:
                print("\n仿真被用户中断")
            
            print("仿真结束")
            final_euler = controller.rot_mat_to_euler(controller.R_BI)
            print(f"最终位置: ({controller.pos_I[0]:.2f}, {controller.pos_I[1]:.2f}, {controller.pos_I[2]:.2f})m")
            print(f"最终姿态: Roll={np.degrees(final_euler[0]):.2f}°, Pitch={np.degrees(final_euler[1]):.2f}°, Yaw={np.degrees(final_euler[2]):.2f}°")

    except Exception as e:
        print(f"仿真主循环失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
