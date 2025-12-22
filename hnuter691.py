import numpy as np
import mujoco as mj
import mujoco.viewer as viewer
import time
import math
import csv
import os
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from datetime import datetime

class HnuterController:
    def __init__(self, model_path: str = "scene.xml"):
        # 加载MuJoCo模型
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        
        # 打印模型诊断信息
        self._print_model_diagnostics()
        
        # 物理参数
        self.dt = self.model.opt.timestep
        self.gravity = 9.81
        self.mass = 4.2  # 主机身质量 + 旋翼机构质量 4.2kg
        self.J = np.diag([0.08, 0.12, 0.1])  # 惯量矩阵
        
        # 旋翼布局参数
        self.l1 = 0.3  # 前旋翼组Y向距离(m)
        self.l2 = 0.5  # 尾部推进器X向距离(m)
        self.k_d = 8.1e-8  # 尾部反扭矩系数
        
        # ========== 新增：俯仰角阈值参数 ==========
        self.pitch_threshold_deg = 70.0  # 俯仰角阈值（度）
        self.pitch_threshold_rad = np.radians(self.pitch_threshold_deg)  # 转换为弧度
        self.is_pitch_exceed = False  # 标记是否超过阈值
        self._pitch_warned = False  # 避免重复打印警告
        
        # 几何控制器增益（针对90°大角度微调）
        self.Kp = np.diag([6, 6, 6])  # 位置增益适度提高
        self.Dp = np.diag([5, 5, 5])  # 速度阻尼
        # self.KR = np.array([3, 2.0, 0.3])   # 姿态增益适度提高，增强大角度跟踪
        # self.Domega = np.array([0.9, 0.6, 0.6])  # 角速度阻尼适度提高
        self.KR = np.array([3, 2.0, 0.3])   # 姿态增益适度提高，增强大角度跟踪
        self.Domega = np.array([0.9, 0.6, 0.6])  # 角速度阻尼适度提高

        # 控制量
        self.f_c_body = np.zeros(3)  # 机体坐标系下的控制力
        self.f_c_world = np.zeros(3)  # 世界坐标系下的控制力
        self.tau_c = np.zeros(3)     # 控制力矩
        self.u = np.zeros(7)         # 控制输入向量

        # 分配矩阵 (根据模型结构更新)
        self.A = np.array([
            [1, 0,  0, 1, 0,  0, 0,],   # X力分配 
            [0, 0, 1, 0, 0, 1, 0],   # Y力分配
            [0, 1, 0, 0, 1, 0, 1],
            [0, self.l1, 0, 0, -self.l1, 0, 0],   # 滚转力矩
            [0, 0, 0, 0, 0, 0, self.l2],  # 俯仰力矩
            [-self.l1, 0, 0, self.l1, 0, 0, 0]  # 偏航力矩
        ])
        
        # 分配矩阵的伪逆 (用于奇异情况)
        self.A_pinv = np.linalg.pinv(self.A)

        # 目标状态
        self.target_position = np.array([0.0, 0.0, 0.3])  # 初始目标高度
        self.target_velocity = np.array([0.0, 0.0, 0.0])
        self.target_acceleration = np.array([0.0, 0.0, 0.0])
        self.target_attitude = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
        self.target_attitude_rate = np.array([0.0, 0.0, 0.0])
        self.target_attitude_acceleration = np.array([0.0, 0.0, 0.0])
        
        # 倾转状态
        self.alpha1 = 0.0  # roll右倾角
        self.alpha2 = 0.0  # roll左倾角
        self.theta1 = 0.0  # pitch右倾角
        self.theta2 = 0.0  # pitch左倾角
        self.T12 = 0.0  # 前左旋翼组推力
        self.T34 = 0.0  # 前右旋翼组推力
        self.T5 = 0.0   # 尾部推进器推力
        
        # 添加角度连续性处理参数
        self.last_alpha1 = 0
        self.last_alpha2 = 0
        self.last_theta1 = 0
        self.last_theta2 = 0

        # 执行器名称映射
        self._get_actuator_ids()
        self._get_sensor_ids()
        
        # 创建日志文件
        self._create_log_file()

        # ========== 核心修改：90°大角度轨迹控制 ==========
        self.trajectory_phase = 0  # 阶段划分更细致
        self.attitude_target_rad = np.pi*2/5  # 目标姿态角度（90度转弧度，核心修改）
        self.phase_start_time = 0.0  # 各阶段起始时间
        self.attitude_tolerance = 0.08  # 90°大角度下适度放宽tolerance（弧度）

        print("倾转旋翼控制器初始化完成（适配90°大角度姿态跟踪）")
        print(f"⚠️  俯仰角超过{self.pitch_threshold_deg}°时将自动置零横滚/偏航力矩 ⚠️")
    
    def _print_model_diagnostics(self):
        """打印模型诊断信息"""
        print("\n=== 模型诊断信息 ===")
        print(f"广义坐标数量 (nq): {self.model.nq}")
        print(f"速度自由度 (nv): {self.model.nv}")
        print(f"执行器数量 (nu): {self.model.nu}")
        print(f"身体数量: {self.model.nbody}")
        print(f"关节数量: {self.model.njnt}")
        print(f"几何体数量: {self.model.ngeom}")
        
        # 检查身体信息
        print("\n=== 身体列表 ===")
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            print(f"身体 {i}: {name}")
        
        # 检查关节信息
        print("\n=== 关节列表 ===")
        for i in range(self.model.njnt):
            jnt_type = self.model.jnt_type[i]
            jnt_name = self.model.jnt(i).name
            print(f"关节 {i}: {jnt_name}, 类型: {jnt_type}")
        
        # 检查执行器信息
        print("\n=== 执行器列表 ===")
        for i in range(self.model.nu):
            act_name = self.model.name_actuatoradr[i]
            print(f"执行器 {i}: {act_name}")
       
    def _create_log_file(self):
        """创建日志文件并写入表头"""
        # 确保logs目录存在
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # 创建带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f'logs/drone_log_90deg_{timestamp}.csv'  # 标注90度日志
        
        # 写入CSV表头（新增俯仰角超限标记）
        with open(self.log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'timestamp', 'pos_x', 'pos_y', 'pos_z', 
                'target_x', 'target_y', 'target_z',
                'roll', 'pitch', 'yaw',
                'target_roll', 'target_pitch', 'target_yaw',
                'curr_quat_w', 'curr_quat_x', 'curr_quat_y', 'curr_quat_z',
                'target_quat_w', 'target_quat_x', 'target_quat_y', 'target_quat_z',
                'vel_x', 'vel_y', 'vel_z',
                'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
                'accel_x', 'accel_y', 'accel_z',
                'f_world_x', 'f_world_y', 'f_world_z',
                'f_body_x', 'f_body_y', 'f_body_z',
                'tau_x', 'tau_y', 'tau_z',
                'u1', 'u2', 'u3', 'u4', 'u5',
                'T12', 'T34', 'T5',
                'alpha1', 'alpha2',
                'theta1', 'theta2',
                'trajectory_phase',
                'is_pitch_exceed'  # 新增：俯仰角超限标记
            ])
        
        print(f"已创建90°姿态跟踪日志文件: {self.log_file}")
    
    def log_status(self, state: dict):
        """记录状态到日志文件"""
        timestamp = time.time()
        position = state.get('position', np.zeros(3))
        velocity = state.get('velocity', np.zeros(3))
        angular_velocity = state.get('angular_velocity', np.zeros(3))
        acceleration = state.get('acceleration', np.zeros(3))
        euler = state.get('euler', np.zeros(3))
        current_quat = state.get('quaternion', np.array([1.0, 0.0, 0.0, 0.0]))
        target_quat = self._euler_to_quaternion(self.target_attitude)
        is_pitch_exceed = state.get('is_pitch_exceed', False)
        
        with open(self.log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                timestamp,
                position[0], position[1], position[2],
                self.target_position[0], self.target_position[1], self.target_position[2],
                euler[0], euler[1], euler[2],
                self.target_attitude[0], self.target_attitude[1], self.target_attitude[2],
                current_quat[0], current_quat[1], current_quat[2], current_quat[3],
                target_quat[0], target_quat[1], target_quat[2], target_quat[3],
                velocity[0], velocity[1], velocity[2],
                angular_velocity[0], angular_velocity[1], angular_velocity[2],
                acceleration[0], acceleration[1], acceleration[2],
                self.f_c_world[0], self.f_c_world[1], self.f_c_world[2],
                self.f_c_body[0], self.f_c_body[1], self.f_c_body[2],
                self.tau_c[0], self.tau_c[1], self.tau_c[2],
                self.u[0], self.u[1], self.u[2], self.u[3], self.u[4],
                self.T12, self.T34, self.T5,
                self.alpha1, self.alpha2,
                self.theta1, self.theta2,
                self.trajectory_phase,
                int(is_pitch_exceed)  # 记录是否超限（0/1）
            ])
    
    def _get_actuator_ids(self):
        """获取执行器ID"""
        self.actuator_ids = {}
        
        try:
            # 机臂偏航执行器
            self.actuator_ids['arm_pitch_right'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_pitch_right')
            self.actuator_ids['arm_pitch_left'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_pitch_left')
            
            # 螺旋桨倾转执行器
            self.actuator_ids['prop_tilt_right'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_roll_right')
            self.actuator_ids['prop_tilt_left'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_roll_left')
            
            # 推力执行器
            thrust_actuators = [
                'motor_r_upper', 'motor_r_lower', 
                'motor_l_upper', 'motor_l_lower', 
                'motor_rear_upper'
            ]
            for name in thrust_actuators:
                self.actuator_ids[name] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
            
            print("执行器ID映射:", self.actuator_ids)
        except Exception as e:
            print(f"获取执行器ID失败: {e}")
            self.actuator_ids = {
                'arm_pitch_right': 0,
                'arm_pitch_left': 1,
                'prop_tilt_right': 2,
                'prop_tilt_left': 3,
                'motor_r_upper': 4,
                'motor_r_lower': 5,
                'motor_l_upper': 6,
                'motor_l_lower': 7,
                'motor_rear_upper': 8
            }
            print("使用默认执行器ID映射")
    
    def _get_sensor_ids(self):
        """获取传感器ID"""
        self.sensor_ids = {}
        
        try:
            # 位置和姿态传感器
            self.sensor_ids['drone_pos'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, 'drone_pos')
            self.sensor_ids['drone_quat'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, 'drone_quat')
            
            # 速度传感器
            self.sensor_ids['body_vel'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, 'body_vel')
            self.sensor_ids['body_gyro'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, 'body_gyro')
            self.sensor_ids['body_acc'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, 'body_acc')
            
            # 螺旋桨速度传感器
            propeller_sensors = [
                'prop_r_upper_vel', 'prop_r_lower_vel',
                'prop_l_upper_vel', 'prop_l_lower_vel',
                'prop_rear_upper_vel'
            ]
            for name in propeller_sensors:
                self.sensor_ids[name] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, name)
            
            # 倾转角度传感器
            tilt_sensors = [
                'arm_pitch_right_pos', 'arm_pitch_left_pos',
                'prop_tilt_right_pos', 'prop_tilt_left_pos'
            ]
            for name in tilt_sensors:
                self.sensor_ids[name] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, name)
            
            print("传感器ID映射:", self.sensor_ids)
        except Exception as e:
            print(f"获取传感器ID失败: {e}")
            self.sensor_ids = {
                'drone_pos': 0,
                'drone_quat': 1,
                'body_vel': 2,
                'body_gyro': 3,
                'body_acc': 4
            }
            print("使用默认传感器ID映射")
    
    def get_state(self) -> dict:
        """获取无人机当前状态（新增俯仰角超限判断）"""
        state = {
            'position': np.zeros(3),
            'quaternion': np.array([1.0, 0.0, 0.0, 0.0]),
            'rotation_matrix': np.eye(3),
            'velocity': np.zeros(3),
            'angular_velocity': np.zeros(3),
            'acceleration': np.zeros(3),
            'euler': np.zeros(3),
            'is_pitch_exceed': False  # 新增：俯仰角超限标记
        }
        
        try:
            body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'drone')
            if body_id != -1:
                state['position'] = self.data.xpos[body_id].copy()
                state['quaternion'] = self.data.xquat[body_id].copy()
                state['velocity'] = self.data.cvel[body_id][3:6].copy()
                state['angular_velocity'] = self.data.cvel[body_id][0:3].copy()
            
            state['rotation_matrix'] = self._quat_to_rotation_matrix(state['quaternion'])
            state['euler'] = self._quat_to_euler(state['quaternion'])
            
            # ========== 核心修改：判断俯仰角是否超限 ==========
            self.is_pitch_exceed = abs(state['euler'][1]) > self.pitch_threshold_rad
            state['is_pitch_exceed'] = self.is_pitch_exceed
            
            # 打印超限警告（仅首次超限/恢复时）
            if self.is_pitch_exceed and not self._pitch_warned:
                pitch_deg = np.degrees(state['euler'][1])
                print(f"\n⚠️ 警告：俯仰角 {pitch_deg:.1f}° 超过 {self.pitch_threshold_deg}°，已置零横滚/偏航力矩！")
                self._pitch_warned = True
            elif not self.is_pitch_exceed and self._pitch_warned:
                pitch_deg = np.degrees(state['euler'][1])
                print(f"\n✅ 恢复：俯仰角 {pitch_deg:.1f}° 低于 {self.pitch_threshold_deg}°，恢复横滚/偏航力矩控制！")
                self._pitch_warned = False
            
            if np.any(np.isnan(state['position'])):
                print("警告: 位置数据包含NaN，使用零值")
                state['position'] = np.zeros(3)
                
            return state
        except Exception as e:
            print(f"状态获取错误: {e}")
            return state

    def _quat_to_rotation_matrix(self, quat: np.ndarray) -> np.ndarray:
        """四元数转旋转矩阵"""
        w, x, y, z = quat
        
        R11 = 1 - 2 * (y * y + z * z)
        R12 = 2 * (x * y - w * z)
        R13 = 2 * (x * z + w * y)
        
        R21 = 2 * (x * y + w * z)
        R22 = 1 - 2 * (x * x + z * z)
        R23 = 2 * (y * z - w * x)
        
        R31 = 2 * (x * z - w * y)
        R32 = 2 * (y * z + w * x)
        R33 = 1 - 2 * (x * x + y * y)
        
        return np.array([
            [R11, R12, R13],
            [R21, R22, R23],
            [R31, R32, R33]
        ])

    def _quat_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """四元数转欧拉角 (roll, pitch, yaw)"""
        w, x, y, z = quat
        
        # Roll (x轴旋转)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y轴旋转)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z轴旋转)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def _euler_to_quaternion(self, euler: np.ndarray) -> np.ndarray:
        """欧拉角转四元数 [w, x, y, z]"""
        roll, pitch, yaw = euler
        
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])
    
    def vee_map(self, S: np.ndarray) -> np.ndarray:
        """反对称矩阵的vee映射"""
        return np.array([S[2, 1], S[0, 2], S[1, 0]])

    def hat_map(self, v: np.ndarray) -> np.ndarray:
        """向量的hat映射（叉乘矩阵）"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def compute_control_wrench(self, state: dict) -> Tuple[np.ndarray, np.ndarray]:
        """计算控制力矩和力（基于几何控制器）"""
        position = state['position']
        velocity = state['velocity']
        
        # 位置误差和速度误差
        pos_error = self.target_position - position
        vel_error = self.target_velocity - velocity
        
        # 期望加速度（PD控制）
        acc_des = self.target_acceleration + self.Kp @ pos_error + self.Dp @ vel_error
        
        # 世界坐标系下的控制力
        f_c_world = self.mass * (acc_des + np.array([0, 0, self.gravity]))
        
        # 姿态误差计算
        R = state['rotation_matrix']
        angular_velocity = state['angular_velocity']
        R_des = self._euler_to_rotation_matrix(self.target_attitude)
        e_R = 0.5 * self.vee_map(R_des.T @ R - R.T @ R_des)
        omega_error = angular_velocity - R.T @ R_des @ self.target_attitude_rate
        
        # 控制力矩
        tau_c = -self.KR * e_R - self.Domega * omega_error
        
        # ========== 核心修改：俯仰角超限时置零横滚/偏航力矩 ==========
        # if state['is_pitch_exceed']:
            # tau_c[0] = 0.0  # 横滚力矩置零
            # tau_c[2] = 0.0  # 偏航力矩置零
        # 转换到机体坐标系
        f_c_body = R.T @ f_c_world
        
        # 更新类成员变量
        self.f_c_body = f_c_body
        self.f_c_world = f_c_world
        self.tau_c = tau_c
        
        return f_c_body, tau_c
    
    def _euler_to_rotation_matrix(self, euler: np.ndarray) -> np.ndarray:
        """将欧拉角转换为旋转矩阵（RPY顺序）"""
        roll, pitch, yaw = euler
        
        R_x = np.array([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]
        ])
        
        R_y = np.array([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])
        
        R_z = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        return R_z @ R_y @ R_x

    def inverse_nonlinear_mapping(self, W, state):
        """修正后的代数逆映射函数（适配90°大角度）"""
        # 尾部推力 (由俯仰力矩确定)
        u7 = (2/1) * W[4]                     
        
        # 左/右旋翼的 X轴分力 (由总Fx和偏航力矩Tz确定)
        u1 = W[0]/2 - (10/3)*W[5]              
        u4 = W[0]/2 + (10/3)*W[5]              
        
        # 左/右旋翼的 Z轴分力 (由总Fz和滚转力矩Tx确定)
        Fz_front = W[2]
        u2 = Fz_front/2 - (10/3)*W[3]  
        u5 = Fz_front/2 + (10/3)*W[3]  

        # 侧向分力均分
        target_Fy = W[1]
        u3 = -target_Fy / 2.0
        u6 = -target_Fy / 2.0
        
        # 计算推力和角度（增加90°大角度保护）
        F1 = np.sqrt(u1**2 + u2**2 + u3**2)
        F2 = np.sqrt(u4**2 + u5**2 + u6**2)
        F3 = u7
        
        # 防止除零保护（90°大角度下更严格）
        eps = 1e-8
        F1_safe = F1 if F1 > eps else eps
        F2_safe = F2 if F2 > eps else eps

        # 求解倾转角度（增加数值稳定性）
        alpha1 = np.arctan2(u1, u2)  
        alpha2 = np.arctan2(u4, u5)
        
        val1 = np.clip(u3 / F1_safe, -1.0 + eps, 1.0 - eps)  # 避免arcsin(±1)的数值问题
        val2 = np.clip(u6 / F2_safe, -1.0 + eps, 1.0 - eps)
        
        theta1 = np.arcsin(val1)
        theta2 = np.arcsin(val2)
        
        return np.array([F1, F2, F3, alpha1, alpha2, theta1, theta2])

    def allocate_actuators(self, f_c_body: np.ndarray, tau_c: np.ndarray, state: dict):
        """
        基于带宽感知的混合几何分配算法 (Scheme 2 + Scheme 4)
        输入:
            f_c_body: 期望机体三维力 [Fx, Fy, Fz]
            tau_c:    期望机体三维力矩 [Tx, Ty, Tz]
            state:    当前状态字典 (包含传感器反馈的实际关节角)
        """
        # 1. 获取几何参数 (根据你的XML模型硬编码，或在init中加载)
        L_arm = 0.3   # 前臂总长 (0.1 + 0.2)
        L_rear = 0.5  # 后臂总长 (0.1 + 0.4)
        
        # 2. 读取当前舵机实际角度 (用于动力学代偿)
        # 注意：XML中 joint 定义可能是弧度
        alpha_L_act = state.get('arm_pitch_left_pos', 0.0)  # 绕Y轴旋转
        alpha_R_act = state.get('arm_pitch_right_pos', 0.0)
        theta_L_act = state.get('prop_tilt_left_pos', 0.0)  # 绕X轴旋转
        theta_R_act = state.get('prop_tilt_right_pos', 0.0)
        
        # ---------------------------------------------------------
        # 第一阶段：计算舵机目标角度 (慢回路 - 几何解耦)
        # ---------------------------------------------------------
        # 逻辑：舵机的任务是让推力矢量尽可能对齐期望的力方向
        # 前旋翼主要负责产生 Fx 和 Fz
        
        # 计算前旋翼的理想倾转角 (Alpha) - 也就是 Pitch 倾转
        # 简单的几何：Alpha = atan2(Fx, Fz)
        # 注意：这里假设后旋翼不产生 X 向力，所以前旋翼要承担全部 Fx
        F_xy_plane = np.sqrt(f_c_body[0]**2 + f_c_body[2]**2) + 1e-6
        alpha_cmd = np.arctan2(f_c_body[0], f_c_body[2])
        
        # 计算侧倾角 (Theta) - 也就是 Roll 倾转
        # 用于产生侧向力 Fy
        theta_cmd = np.arcsin(np.clip(-f_c_body[1] / 60.0, -0.5, 0.5)) # 假设最大推力归一化防饱和
        
        # ---------------------------------------------------------
        # 第二阶段：计算电机推力 (快回路 - 动力学代偿)
        # ---------------------------------------------------------
        # 核心：使用 alpha_ACTUAL (实际值) 构建雅可比矩阵
        # 这意味着：如果舵机还没转到位，矩阵会准确描述当前的"错误"几何
        # 求解器会自动计算出需要多大的推力差来修正这个"错误"
        
        # 构造雅可比矩阵 J (6x3): [F; Tau] = J * [TL, TR, TRear]
        # TL: 左前组总推力, TR: 右前组总推力, TRear: 尾部推力
        
        # 预计算左/右旋翼的单位推力矢量 (在机体坐标系下)
        # 旋转顺序：先 Pitch(Y) 后 Roll(X) -> 取决于你的机械结构，假设是标准的
        # XML: arm_pitch (Y) -> prop_tilt (X) -> Propeller (Z)
        
        def get_thrust_vector(alpha, theta):
            # R_y(alpha) @ R_x(theta) @ [0,0,1]
            sa, ca = np.sin(alpha), np.cos(alpha)
            st, ct = np.sin(theta), np.cos(theta)
            # 矢量: [cos(theta)sin(alpha), -sin(theta), cos(theta)cos(alpha)]
            return np.array([ct*sa, -st, ct*ca])

        # 获得当前的实际推力方向
        vec_L = get_thrust_vector(alpha_L_act, theta_L_act)
        vec_R = get_thrust_vector(alpha_R_act, theta_R_act)
        vec_Rear = np.array([0.0, 0.0, 1.0]) # 后旋翼不倾转，始终朝上
        
        # 力臂矢量
        pos_L = np.array([0.0, L_arm, 0.0])   # 左臂坐标 (+Y) [XML中 left_arm_base 是 pos 0 -0.1? 需确认方向]
        # 修正：根据MuJoCo标准，通常左是+Y。但你的XML中 left_arm_base pos="0 -0.1"。
        # 我们以XML为准：Left = -Y, Right = +Y 
        # (通常 right_arm_base pos="0 0.1" 是正Y)
        pos_L_xml = np.array([0.0, -L_arm, 0.0]) 
        pos_R_xml = np.array([0.0, L_arm, 0.0])
        pos_Rear_xml = np.array([-L_rear, 0.0, 0.0])

        # 计算力矩贡献 (r x F)
        mom_L = np.cross(pos_L_xml, vec_L)
        mom_R = np.cross(pos_R_xml, vec_R)
        mom_Rear = np.cross(pos_Rear_xml, vec_Rear) # [-L_rear*1, 0, 0] -> Pitch力矩
        
        # 组装矩阵 J (这里只取我们关心的维度，简化计算)
        # 我们关心: Fz (升力), Tx (滚转), Ty (俯仰), Tz (偏航)
        # Fx 由 alpha 控制，暂时不通过推力强行解(避免对抗)
        
        # J_reduced: 4行 x 3列
        # Row 0: Fz
        # Row 1: Tx (Roll)
        # Row 2: Ty (Pitch)
        # Row 3: Tz (Yaw)
        
        J = np.zeros((4, 3))
        
        # Column 0 (Left Motor)
        J[0,0] = vec_L[2] # Fz
        J[1,0] = mom_L[0] # Tx
        J[2,0] = mom_L[1] # Ty
        J[3,0] = mom_L[2] # Tz
        
        # Column 1 (Right Motor)
        J[0,1] = vec_R[2]
        J[1,1] = mom_R[0]
        J[2,1] = mom_R[1]
        J[3,1] = mom_R[2]
        
        # Column 2 (Rear Motor)
        J[0,2] = vec_Rear[2]
        J[1,2] = mom_Rear[0]
        J[2,2] = mom_Rear[1] # 主要贡献
        J[3,2] = mom_Rear[2]
        
        # 目标向量 b
        b = np.array([
            f_c_body[2], # Fz
            tau_c[0],    # Tx
            tau_c[1],    # Ty
            tau_c[2]     # Tz
        ])
        
        # ---------------------------------------------------------
        # 求解线性方程 J * T = b
        # ---------------------------------------------------------
        # 由于是超定或欠定混合，使用加权最小二乘法 (Weighted Least Squares)
        # 权重矩阵 W: 优先保证 Pitch 和 Fz (生存)，其次 Roll，Yaw 最次
        weights = np.diag([10.0, 5.0, 20.0, 1.0]) 
        
        # WLS 解: T = (J.T * W * J)^-1 * J.T * W * b
        # 为了数值稳定，使用 np.linalg.pinv 或 lstsq
        
        J_weighted = weights @ J
        b_weighted = weights @ b
        
        T_sol, residuals, rank, s = np.linalg.lstsq(J_weighted, b_weighted, rcond=None)
        
        # ---------------------------------------------------------
        # 输出限幅与分配
        # ---------------------------------------------------------
        # 限制推力为正 (不可反转) 且不超过最大值
        T_L_total = np.clip(T_sol[0], 0.1, 50.0)
        T_R_total = np.clip(T_sol[1], 0.1, 50.0)
        T_Rear    = np.clip(T_sol[2], 0.0, 20.0)
        
        # 更新类成员变量用于日志
        self.T12 = T_L_total
        self.T34 = T_R_total
        self.T5 = T_Rear
        self.alpha1 = alpha_cmd
        self.alpha2 = alpha_cmd # 简化：左右 tilt 相同
        self.theta1 = theta_cmd
        self.theta2 = theta_cmd
        
        # 返回指令
        return T_L_total, T_R_total, T_Rear, alpha_cmd, alpha_cmd, theta_cmd, theta_cmd
    
    def _handle_angle_continuity(self, current: float, last: float) -> float:
        """处理角度连续性，避免跳变"""
        diff = current - last
        if diff > np.pi:
            return current - 2 * np.pi
        elif diff < -np.pi:
            return current + 2 * np.pi
        return current
    
    def set_actuators(self, T12: float, T34: float, T5: float, alpha1: float, alpha2: float, theta1: float, theta2: float):
        """应用控制命令到执行器"""
        try:            
            # 设置机臂偏航角度 (alpha)
            if 'arm_pitch_right' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['arm_pitch_right']] = alpha2
            
            if 'arm_pitch_left' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['arm_pitch_left']] = alpha1
            
            # 设置螺旋桨倾转角度 (theta)
            if 'prop_tilt_right' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['prop_tilt_right']] = theta1
            
            if 'prop_tilt_left' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['prop_tilt_left']] = theta2
            
            # 设置推力（左右旋翼组均分推力）
            if 'motor_r_upper' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['motor_r_upper']] = T34 / 2
            
            if 'motor_r_lower' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['motor_r_lower']] = T34 / 2
            
            if 'motor_l_upper' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['motor_l_upper']] = T12 / 2
            
            if 'motor_l_lower' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['motor_l_lower']] = T12 / 2
            
            # 尾部推进器
            if 'motor_rear_upper' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['motor_rear_upper']] = T5
                
        except Exception as e:
            print(f"设置执行器失败: {e}")
    
    def update_control(self):
        """更新控制量"""
        try:
            # 获取当前状态
            state = self.get_state()

            # 计算控制力矩和力
            f_c_body, tau_c = self.compute_control_wrench(state)
            
            # 分配执行器命令
            T12, T34, T5, alpha1, alpha2, theta1, theta2 = self.allocate_actuators(f_c_body, tau_c, state)
            
            # 应用控制
            self.set_actuators(T12, T34, T5, alpha1, alpha2, theta1, theta2)
            
            # 记录状态
            self.log_status(state)
            
            return True
        except Exception as e:
            print(f"控制更新失败: {e}")
            return False
    
    def print_status(self):
        """打印当前状态信息（含90°大角度标注+俯仰角超限提示）"""
        try:
            state = self.get_state()
            pos = state['position']
            vel = state['velocity']
            accel = state['acceleration']
            euler_deg = np.degrees(state['euler'])
            target_euler_deg = np.degrees(self.target_attitude)
            current_quat = state['quaternion']
            target_quat = self._euler_to_quaternion(self.target_attitude)
            
            # 阶段名称映射（更新为90°标注）
            phase_names = {
                0: "起飞悬停",
                1: "Roll转动(0°→90°)",
                2: "Roll保持(90°，稳定5s)",
                3: "Roll恢复(90°→0°)",
                4: "Pitch转动(0°→90°)",
                5: "Pitch保持(90°，稳定5s)",
                6: "Pitch恢复(90°→0°)",
                7: "Yaw转动(0°→90°)",
                8: "Yaw保持(90°，稳定5s)",
                9: "Yaw恢复(90°→0°)",
                10: "最终悬停"
            }
            phase_name = phase_names.get(self.trajectory_phase, "未知阶段")
            
            print(f"\n=== 轨迹阶段: {self.trajectory_phase} ({phase_name}) ===")
            print(f"位置: X={pos[0]:.8f}m, Y={pos[1]:.8f}m, Z={pos[2]:.8f}m")
            print(f"目标位置: X={self.target_position[0]:.8f}m, Y={self.target_position[1]:.8f}m, Z={self.target_position[2]:.8f}m")
            print(f"姿态: Roll={euler_deg[0]:.2f}°, Pitch={euler_deg[1]:.2f}°, Yaw={euler_deg[2]:.2f}°")  
            print(f"控制力矩: X={self.tau_c[0]:.4f}Nm, Y={self.tau_c[1]:.4f}Nm, Z={self.tau_c[2]:.4f}Nm")
            print(f"目标姿态: Roll={target_euler_deg[0]:.1f}°, Pitch={target_euler_deg[1]:.1f}°, Yaw={target_euler_deg[2]:.1f}°") 
            print(f"角速度: Roll={np.degrees(state['angular_velocity'][0]):.2f}°/s, Pitch={np.degrees(state['angular_velocity'][1]):.2f}°/s, Yaw={np.degrees(state['angular_velocity'][2]):.2f}°/s")
            print(f"执行器状态: T12={self.T12:.2f}N, T34={self.T34:.2f}N, T5={self.T5:.2f}N, α1={math.degrees(self.alpha1):.2f}°, α2={math.degrees(self.alpha2):.2f}°, θ1={math.degrees(self.theta1):.2f}°, θ2={math.degrees(self.theta2):.2f}°")
            # ========== 新增：打印俯仰角超限状态 ==========
            print(f"俯仰角限制: {'超限(横滚/偏航力矩已置零)' if self.is_pitch_exceed else '正常'} (阈值: {self.pitch_threshold_deg}°)")
            print("--------------------------------------------------")
        except Exception as e:
            print(f"状态打印失败: {e}")
    
    def update_trajectory(self, current_time: float):
        """
        适配90°大角度的轨迹发布器（延长时间确保稳定）
        阶段划分（总时长~70秒）：
        0: 0~6s    - 起飞悬停（升到2m高度，姿态归零，确保稳定）
        1: 6~18s   - Roll缓慢转动（12秒从0°→90°，角速度≈7.5°/s）
        2: 18~23s  - Roll保持（5秒，稳定在90°）
        3: 23~29s  - Roll恢复（90°→0°）
        4: 29~41s  - Pitch缓慢转动（12秒从0°→90°）
        5: 41~46s  - Pitch保持（5秒，稳定在90°）
        6: 46~52s  - Pitch恢复（90°→0°）
        7: 52~64s  - Yaw缓慢转动（12秒从0°→90°）
        8: 64~69s  - Yaw保持（5秒，稳定在90°）
        9: 69~75s  - Yaw恢复（90°→0°）
        10: 75s~   - 最终悬停（姿态归零，高度2m）
        """
        # 初始化阶段起始时间
        if self.trajectory_phase == 0 and self.phase_start_time == 0.0:
            self.phase_start_time = current_time
        
        # 阶段时长配置（90°大角度专属）
        phase_durations = {
            0: 6.0,    # 起飞悬停（延长到6秒，确保高度稳定）
            1: 12.0,   # Roll转动（12秒，90°大角度缓慢变化）
            2: 5.0,    # Roll保持（5秒，稳定90°姿态）
            3: 6.0,    # Roll恢复（6秒，平稳回零）
            4: 12.0,   # Pitch转动（12秒）
            5: 5.0,    # Pitch保持（5秒）
            6: 6.0,    # Pitch恢复（6秒）
            7: 12.0,   # Yaw转动（12秒）
            8: 5.0,    # Yaw保持（5秒）
            9: 6.0,    # Yaw恢复（6秒）
            10: float('inf')  # 最终悬停
        }
        
        # 计算当前阶段已运行时间
        phase_elapsed = current_time - self.phase_start_time
        
        # 阶段切换判断
        if phase_elapsed > phase_durations[self.trajectory_phase]:
            self.trajectory_phase += 1
            self.phase_start_time = current_time  # 重置阶段起始时间
            print(f"\n🔄 轨迹阶段切换: {self.trajectory_phase-1} → {self.trajectory_phase}")
        
        # 各阶段轨迹逻辑（所有阶段保持高度2m，只变化姿态）
        if self.trajectory_phase == 0:
            # 阶段0：起飞悬停（高度稳定在2m，姿态归零）
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([0.0, 0.0, 0.0])
            
        elif self.trajectory_phase == 1:
            # 阶段1：Roll缓慢转动（0°→90°，线性插值）
            progress = phase_elapsed / phase_durations[1]  # 0~1
            progress = np.clip(progress, 0.0, 1.0)
            roll_target = progress * self.attitude_target_rad
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([roll_target, 0.0, 0.0])
            
        elif self.trajectory_phase == 2:
            # 阶段2：Roll保持（稳定在90°）
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([self.attitude_target_rad, 0.0, 0.0])
            
        elif self.trajectory_phase == 3:
            # 阶段3：Roll恢复（90°→0°，线性插值）
            progress = phase_elapsed / phase_durations[3]  # 0~1
            progress = np.clip(progress, 0.0, 1.0)
            roll_target = (1 - progress) * self.attitude_target_rad
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([roll_target, 0.0, 0.0])
            
        elif self.trajectory_phase == 4:
            # 阶段4：Pitch缓慢转动（0°→90°）
            progress = phase_elapsed / phase_durations[4]  # 0~1
            progress = np.clip(progress, 0.0, 1.0)
            pitch_target = progress * self.attitude_target_rad
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([0.0, pitch_target, 0.0])
            
        elif self.trajectory_phase == 5:
            # 阶段5：Pitch保持（稳定在90°）
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([0.0, self.attitude_target_rad, 0.0])
            
        elif self.trajectory_phase == 6:
            # 阶段6：Pitch恢复（90°→0°）
            progress = phase_elapsed / phase_durations[6]  # 0~1
            progress = np.clip(progress, 0.0, 1.0)
            pitch_target = (1 - progress) * self.attitude_target_rad
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([0.0, pitch_target, 0.0])
            
        elif self.trajectory_phase == 7:
            # 阶段7：Yaw缓慢转动（0°→90°）
            progress = phase_elapsed / phase_durations[7]  # 0~1
            progress = np.clip(progress, 0.0, 1.0)
            yaw_target = progress * self.attitude_target_rad
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([0.0, 0.0, yaw_target])
            
        elif self.trajectory_phase == 8:
            # 阶段8：Yaw保持（稳定在90°）
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([0.0, 0.0, self.attitude_target_rad])
            
        elif self.trajectory_phase == 9:
            # 阶段9：Yaw恢复（90°→0°）
            progress = phase_elapsed / phase_durations[9]  # 0~1
            progress = np.clip(progress, 0.0, 1.0)
            yaw_target = (1 - progress) * self.attitude_target_rad
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([0.0, 0.0, yaw_target])
            
        else:
            # 阶段10：最终悬停（姿态归零，高度稳定）
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([0.0, 0.0, 0.0])
        
        # 速度/加速度归零（悬停状态，避免位置漂移）
        self.target_velocity = np.zeros(3)
        self.target_acceleration = np.zeros(3)
        self.target_attitude_rate = np.zeros(3)
        self.target_attitude_acceleration = np.zeros(3)
    

def main():
    """主函数 - 启动90°大角度姿态跟踪仿真"""
    print("=== 倾转旋翼无人机90°大角度姿态跟踪仿真 ===")
    print("核心优化：适配90°大角度，延长转动/保持/恢复时间，提高控制器增益")
    print("安全限制：俯仰角超过70°时自动置零横滚/偏航力矩")
    print("轨迹逻辑：起飞悬停→Roll90°(保持5s)→恢复→Pitch90°(保持5s)→恢复→Yaw90°(保持5s)→恢复→悬停")
    
    try:
        # 初始化控制器
        controller = HnuterController("hnuter201.xml")
        
        # 初始目标（会被update_trajectory覆盖）
        controller.target_position = np.array([0.0, 0.0, 2.0])
        controller.target_attitude = np.array([0.0, 0.0, 0.0])
        
        # 启动 Viewer
        with viewer.launch_passive(controller.model, controller.data) as v:
            print("\n仿真启动：")
            print(f"90°姿态跟踪日志文件路径: {controller.log_file}")
            print("按 Ctrl+C 终止仿真")
            
            start_time = time.time()
            last_print_time = 0
            print_interval = 1.0  # 90°大角度下延长打印间隔，便于观察
            count = 0
            
            try:
                while v.is_running():
                    current_time = time.time() - start_time
                    
                    # 启用轨迹更新（核心）
                    # controller.update_trajectory(current_time)
                    
                    # 更新控制
                    controller.update_control()

                    count += 1
                    if count % 1 == 0:
                        # 仿真步进（保持与模型步长一致）
                        mj.mj_step(controller.model, controller.data)
                    
                    # 同步可视化
                    v.sync()
                    
                    # 定期打印状态
                    if current_time - last_print_time > print_interval:
                        controller.print_status()
                        last_print_time = current_time

                    # 控制仿真速率（避免过快）
                    time.sleep(0.001)

            except KeyboardInterrupt:
                print("\n仿真被用户中断")
            
            print("仿真结束")
            final_state = controller.get_state()
            print(f"最终位置: ({final_state['position'][0]:.2f}, {final_state['position'][1]:.2f}, {final_state['position'][2]:.2f})m")
            print(f"最终姿态: Roll={np.degrees(final_state['euler'][0]):.2f}°, Pitch={np.degrees(final_state['euler'][1]):.2f}°, Yaw={np.degrees(final_state['euler'][2]):.2f}°")

    except Exception as e:
        print(f"仿真主循环失败: {e}")
        import traceback
        traceback.print_exc()
    

if __name__ == "__main__":
    main()
