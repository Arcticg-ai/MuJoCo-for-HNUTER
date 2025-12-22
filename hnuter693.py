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
        
        # 旋翼布局参数（根据实际模型调整）
        self.l1 = 0.2  # 前旋翼组Y向距离(m) - 从模型看是0.1+0.1=0.2m
        self.l2 = 0.4  # 尾部推进器X向距离(m) - 从模型看是0.4m
        self.k_d = 8.1e-8  # 尾部反扭矩系数
        
        # ========== 新增：俯仰角阈值参数 ==========
        self.pitch_threshold_deg = 70.0  # 俯仰角阈值（度）
        self.pitch_threshold_rad = np.radians(self.pitch_threshold_deg)  # 转换为弧度
        self.is_pitch_exceed = False  # 标记是否超过阈值
        self._pitch_warned = False  # 避免重复打印警告
        
        # 几何控制器增益（针对90°大角度微调）
        self.Kp = np.diag([8, 8, 12])  # 位置增益适度提高，Z轴增益更高
        self.Dp = np.diag([6, 6, 8])  # 速度阻尼
        
        # ========== 新增：基于俯仰角的增益调度 ==========
        self.KR_fast = np.array([8.0, 6.0, 4.0])   # 快轴增益（横滚/偏航）
        self.KR_slow = np.array([2.0, 6.0, 1.5])   # 慢轴增益（偏航/横滚）
        self.Domega = np.array([2.5, 2.0, 1.5])  # 角速度阻尼适度提高
        
        # 控制量
        self.f_c_body = np.zeros(3)  # 机体坐标系下的控制力
        self.f_c_world = np.zeros(3)  # 世界坐标系下的控制力
        self.tau_c = np.zeros(3)     # 控制力矩
        self.u = np.zeros(7)         # 控制输入向量

        # ========== 新增：几何解耦控制参数 ==========
        # 虚拟坐标系
        self.virtual_R = np.eye(3)  # 虚拟坐标系旋转矩阵
        self.current_pitch = 0.0    # 当前俯仰角
        
        # 舵机动态模型参数
        self.servo_time_constants = {
            'alpha': 0.05,  # 机臂偏航响应时间常数(s)
            'theta': 0.03   # 螺旋桨倾转响应时间常数(s)
        }
        
        # 状态预测器
        self.predicted_alpha1 = 0.0
        self.predicted_alpha2 = 0.0
        self.predicted_theta1 = 0.0
        self.predicted_theta2 = 0.0
        
        # 实际倾转角度（从传感器）
        self.alpha1_actual = 0.0
        self.alpha2_actual = 0.0
        self.theta1_actual = 0.0
        self.theta2_actual = 0.0

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
        self.attitude_target_rad = np.pi/2  # 目标姿态角度（90度转弧度，核心修改）
        self.phase_start_time = 0.0  # 各阶段起始时间
        self.attitude_tolerance = 0.08  # 90°大角度下适度放宽tolerance（弧度）

        print("倾转旋翼控制器初始化完成（适配90°大角度姿态跟踪）")
        print("⚠️  采用基于倾转预测的几何解耦控制方案 ⚠️")
        print(f"俯仰角阈值: {self.pitch_threshold_deg}°")
    
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
        self.log_file = f'logs/drone_log_geometric_decoupled_{timestamp}.csv'
        
        # 写入CSV表头（新增几何解耦相关字段）
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
                'f_world_x', 'f_world_y', 'f_world_z',
                'f_body_x', 'f_body_y', 'f_body_z',
                'tau_x', 'tau_y', 'tau_z',
                'T12', 'T34', 'T5',
                'alpha1_cmd', 'alpha2_cmd', 'alpha1_actual', 'alpha2_actual',
                'theta1_cmd', 'theta2_cmd', 'theta1_actual', 'theta2_actual',
                'trajectory_phase',
                'is_pitch_exceed',
                'axis_type_roll', 'axis_type_pitch', 'axis_type_yaw',
                'KR_roll', 'KR_pitch', 'KR_yaw'
            ])
        
        print(f"已创建几何解耦控制日志文件: {self.log_file}")
    
    def log_status(self, state: dict):
        """记录状态到日志文件"""
        timestamp = time.time()
        position = state.get('position', np.zeros(3))
        euler = state.get('euler', np.zeros(3))
        current_quat = state.get('quaternion', np.array([1.0, 0.0, 0.0, 0.0]))
        target_quat = self._euler_to_quaternion(self.target_attitude)
        is_pitch_exceed = state.get('is_pitch_exceed', False)
        
        # 获取当前轴类型
        axis_types = self._get_all_axis_types(euler[1])
        
        # 获取当前增益
        KR_current = self._get_scheduled_gains(euler[1])
        
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
                state.get('velocity', [0,0,0])[0], state.get('velocity', [0,0,0])[1], state.get('velocity', [0,0,0])[2],
                state.get('angular_velocity', [0,0,0])[0], state.get('angular_velocity', [0,0,0])[1], state.get('angular_velocity', [0,0,0])[2],
                self.f_c_world[0], self.f_c_world[1], self.f_c_world[2],
                self.f_c_body[0], self.f_c_body[1], self.f_c_body[2],
                self.tau_c[0], self.tau_c[1], self.tau_c[2],
                self.T12, self.T34, self.T5,
                self.alpha1, self.alpha2, self.alpha1_actual, self.alpha2_actual,
                self.theta1, self.theta2, self.theta1_actual, self.theta2_actual,
                self.trajectory_phase,
                int(is_pitch_exceed),
                axis_types[0], axis_types[1], axis_types[2],
                KR_current[0], KR_current[1], KR_current[2]
            ])
    
    def _get_actuator_ids(self):
        """获取执行器ID"""
        self.actuator_ids = {}
        
        try:
            # 机臂偏航执行器
            self.actuator_ids['tilt_pitch_left'] = mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_pitch_left')
            self.actuator_ids['tilt_pitch_right'] = mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_pitch_right')
            
            # 螺旋桨倾转执行器
            self.actuator_ids['tilt_roll_left'] = mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_roll_left')
            self.actuator_ids['tilt_roll_right'] = mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_roll_right')
            
            # 推力执行器
            thrust_actuators = [
                'motor_r_upper', 'motor_r_lower', 
                'motor_l_upper', 'motor_l_lower', 
                'motor_rear_upper'
            ]
            for name in thrust_actuators:
                self.actuator_ids[name] = mj.mj_name2id(
                    self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
            
            print("执行器ID映射:", self.actuator_ids)
            
        except Exception as e:
            print(f"获取执行器ID失败: {e}")
            # 使用备用方案：直接按顺序获取
            self.actuator_ids = {}
            for i in range(self.model.nu):
                act_name = self.model.name_actuatoradr[i]
                if act_name:
                    self.actuator_ids[act_name] = i
            print("顺序执行器ID映射:", self.actuator_ids)
    
    def _get_sensor_ids(self):
        """获取传感器ID"""
        self.sensor_ids = {}
        
        try:
            # 位置和姿态传感器
            self.sensor_ids['drone_pos'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, 'drone_pos')
            self.sensor_ids['drone_quat'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, 'drone_quat')
            
            # 倾转角度传感器
            tilt_sensors = [
                'arm_pitch_left_pos', 'arm_pitch_right_pos',
                'prop_tilt_left_pos', 'prop_tilt_right_pos'
            ]
            for name in tilt_sensors:
                self.sensor_ids[name] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, name)
            
            print("传感器ID映射:", self.sensor_ids)
            
        except Exception as e:
            print(f"获取传感器ID失败: {e}")
            # 创建默认映射
            self.sensor_ids = {}
            for i in range(self.model.nsensor):
                sensor_name = self.model.name_sensoradr[i]
                if sensor_name:
                    self.sensor_ids[sensor_name] = i
            print("顺序传感器ID映射:", self.sensor_ids)
    
    def get_state(self) -> dict:
        """获取无人机当前状态（新增俯仰角超限判断和实际倾转角度）"""
        state = {
            'position': np.zeros(3),
            'quaternion': np.array([1.0, 0.0, 0.0, 0.0]),
            'rotation_matrix': np.eye(3),
            'velocity': np.zeros(3),
            'angular_velocity': np.zeros(3),
            'acceleration': np.zeros(3),
            'euler': np.zeros(3),
            'is_pitch_exceed': False
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
            self.current_pitch = state['euler'][1]  # 更新当前俯仰角
            
            # ========== 核心修改：判断俯仰角是否超限 ==========
            self.is_pitch_exceed = abs(state['euler'][1]) > self.pitch_threshold_rad
            state['is_pitch_exceed'] = self.is_pitch_exceed
            
            # 打印超限警告（仅首次超限/恢复时）
            if self.is_pitch_exceed and not self._pitch_warned:
                pitch_deg = np.degrees(state['euler'][1])
                print(f"\n⚠️ 警告：俯仰角 {pitch_deg:.1f}° 超过 {self.pitch_threshold_deg}°，启用几何解耦控制！")
                self._pitch_warned = True
            elif not self.is_pitch_exceed and self._pitch_warned:
                pitch_deg = np.degrees(state['euler'][1])
                print(f"\n✅ 恢复：俯仰角 {pitch_deg:.1f}° 低于 {self.pitch_threshold_deg}°，恢复正常控制！")
                self._pitch_warned = False
            
            # ========== 获取实际倾转角度 ==========
            try:
                if 'arm_pitch_left_pos' in self.sensor_ids:
                    self.alpha1_actual = self.data.sensordata[self.sensor_ids['arm_pitch_left_pos']]
                if 'arm_pitch_right_pos' in self.sensor_ids:
                    self.alpha2_actual = self.data.sensordata[self.sensor_ids['arm_pitch_right_pos']]
                if 'prop_tilt_left_pos' in self.sensor_ids:
                    self.theta1_actual = self.data.sensordata[self.sensor_ids['prop_tilt_left_pos']]
                if 'prop_tilt_right_pos' in self.sensor_ids:
                    self.theta2_actual = self.data.sensordata[self.sensor_ids['prop_tilt_right_pos']]
            except:
                pass
            
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
        """计算控制力矩和力（基于几何解耦控制器）"""
        position = state['position']
        velocity = state['velocity']
        
        # 位置误差和速度误差
        pos_error = self.target_position - position
        vel_error = self.target_velocity - velocity
        
        # 期望加速度（PD控制）
        acc_des = self.target_acceleration + self.Kp @ pos_error + self.Dp @ vel_error
        
        # 世界坐标系下的控制力
        f_c_world = self.mass * (acc_des + np.array([0, 0, self.gravity]))
        
        # ========== 核心修改：基于虚拟坐标系的姿态控制 ==========
        pitch = state['euler'][1]
        
        # 计算虚拟坐标系
        self.virtual_R = self._compute_virtual_frame(pitch)
        
        # 姿态误差计算（在虚拟坐标系中）
        R = state['rotation_matrix']
        angular_velocity = state['angular_velocity']
        R_des = self._euler_to_rotation_matrix(self.target_attitude)
        
        # 转换到虚拟坐标系
        R_virtual = self.virtual_R.T @ R
        R_des_virtual = self.virtual_R.T @ R_des
        
        # 在虚拟坐标系中计算姿态误差
        e_R = 0.5 * self.vee_map(R_des_virtual.T @ R_virtual - R_virtual.T @ R_des_virtual)
        omega_error = angular_velocity - R.T @ R_des @ self.target_attitude_rate
        
        # ========== 基于俯仰角的增益调度 ==========
        KR_current = self._get_scheduled_gains(pitch)
        
        # 控制力矩（在虚拟坐标系中计算，然后转换回机体坐标系）
        tau_c_virtual = -KR_current * e_R - self.Domega * omega_error
        tau_c = self.virtual_R @ tau_c_virtual
        
        # ========== 俯仰角超限时的特殊处理 ==========
        if state['is_pitch_exceed']:
            # 当俯仰角超限时，减小横滚和偏航力矩增益
            tau_c[0] *= 0.3  # 横滚力矩减小
            tau_c[2] *= 0.3  # 偏航力矩减小
        
        # 转换到机体坐标系
        f_c_body = R.T @ f_c_world
        
        # 更新类成员变量
        self.f_c_body = f_c_body
        self.f_c_world = f_c_world
        self.tau_c = tau_c
        
        return f_c_body, tau_c
    
    def _compute_virtual_frame(self, pitch: float) -> np.ndarray:
        """计算虚拟坐标系"""
        # 简单实现：根据俯仰角混合两个坐标系
        mix_factor = abs(np.sin(pitch))**2  # 0-1混合因子
        
        # 当pitch=0时，虚拟坐标系与机体坐标系对齐
        # 当pitch=90°时，虚拟坐标系旋转，使快慢轴交换
        
        # 创建混合旋转矩阵
        if mix_factor < 0.01:
            return np.eye(3)
        else:
            # 创建绕Y轴旋转的矩阵，交换X和Z轴
            angle = mix_factor * np.pi/2
            c, s = np.cos(angle), np.sin(angle)
            R_mix = np.array([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
            ])
            return R_mix
    
    def _get_scheduled_gains(self, pitch: float) -> np.ndarray:
        """根据俯仰角获取调度后的增益"""
        pitch_abs = abs(pitch)
        pitch_deg = np.degrees(pitch_abs)
        
        # 混合因子：0°时=0，90°时=1
        mix_factor = min(1.0, pitch_deg / 90.0)
        
        # 插值增益
        KR_roll = self.KR_fast[0] * (1 - mix_factor) + self.KR_slow[0] * mix_factor
        KR_pitch = self.KR_fast[1]  # 俯仰增益保持中等
        KR_yaw = self.KR_slow[2] * (1 - mix_factor) + self.KR_fast[2] * mix_factor
        
        return np.array([KR_roll, KR_pitch, KR_yaw])
    
    def _get_axis_type(self, axis_idx: int, pitch: float) -> str:
        """确定当前轴的响应类型"""
        pitch_deg = abs(np.degrees(pitch))
        
        if axis_idx == 0:  # 横滚轴
            if pitch_deg < 45:
                return 'fast'  # 水平时横滚是快轴
            else:
                return 'slow'  # 直立时横滚变慢轴
        elif axis_idx == 2:  # 偏航轴
            if pitch_deg < 45:
                return 'slow'  # 水平时偏航是慢轴
            else:
                return 'fast'  # 直立时偏航变快轴
        else:  # 俯仰轴
            return 'medium'  # 俯仰轴始终是中速
    
    def _get_all_axis_types(self, pitch: float) -> List[str]:
        """获取所有轴的响应类型"""
        return [
            self._get_axis_type(0, pitch),
            self._get_axis_type(1, pitch),
            self._get_axis_type(2, pitch)
        ]
    
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

    def predict_servo_states(self):
        """预测舵机状态（一阶惯性模型）"""
        dt = self.dt
        
        # 预测机臂偏航角度
        alpha1_error = self.alpha1 - self.predicted_alpha1
        alpha2_error = self.alpha2 - self.predicted_alpha2
        
        tau_alpha = self.servo_time_constants['alpha']
        self.predicted_alpha1 += (alpha1_error / tau_alpha) * dt
        self.predicted_alpha2 += (alpha2_error / tau_alpha) * dt
        
        # 预测螺旋桨倾转角度
        theta1_error = self.theta1 - self.predicted_theta1
        theta2_error = self.theta2 - self.predicted_theta2
        
        tau_theta = self.servo_time_constants['theta']
        self.predicted_theta1 += (theta1_error / tau_theta) * dt
        self.predicted_theta2 += (theta2_error / tau_theta) * dt
        
        return self.predicted_alpha1, self.predicted_alpha2, self.predicted_theta1, self.predicted_theta2
    
    def _build_allocation_matrix(self, alpha1: float, alpha2: float, 
                                theta1: float, theta2: float) -> np.ndarray:
        """构建分配矩阵（考虑倾转角）"""
        # 每个机臂的推力方向（使用预测角度）
        dir1 = self._rotation_z(alpha1) @ self._rotation_x(theta1) @ np.array([0, 0, 1])
        dir2 = self._rotation_z(alpha2) @ self._rotation_x(theta2) @ np.array([0, 0, 1])
        dir3 = np.array([0, 0, 1])  # 尾部推进器方向固定
        
        # 位置向量（从机身中心到各推力点）
        r1 = np.array([0, self.l1, 0])   # 左机臂
        r2 = np.array([0, -self.l1, 0])  # 右机臂
        r3 = np.array([-self.l2, 0, 0])  # 尾部
        
        # 构建分配矩阵 (6x7)
        A = np.zeros((6, 7))
        
        # 左机臂贡献
        A[0:3, 0] = dir1  # 力
        A[3:6, 0] = np.cross(r1, dir1)  # 力矩
        
        # 右机臂贡献
        A[0:3, 1] = dir2  # 力
        A[3:6, 1] = np.cross(r2, dir2)  # 力矩
        
        # 尾部推进器贡献
        A[0:3, 2] = dir3  # 力
        A[3:6, 2] = np.cross(r3, dir3)  # 力矩
        
        return A
    
    def _rotation_x(self, angle: float) -> np.ndarray:
        """绕X轴旋转矩阵"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    
    def _rotation_z(self, angle: float) -> np.ndarray:
        """绕Z轴旋转矩阵"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

    def inverse_nonlinear_mapping_with_delay(self, W, state):
        """带延迟补偿的非线性逆映射"""
        # 预测舵机状态
        pred_alpha1, pred_alpha2, pred_theta1, pred_theta2 = self.predict_servo_states()
        
        # 使用预测状态构建分配矩阵
        A_pred = self._build_allocation_matrix(pred_alpha1, pred_alpha2, pred_theta1, pred_theta2)
        
        try:
            # 尝试求解控制输入
            u_pred = np.linalg.lstsq(A_pred, W, rcond=None)[0]
            
            # 提取控制量
            T12 = max(0, u_pred[0])
            T34 = max(0, u_pred[1])
            T5 = u_pred[2]
            
            # 角度命令（基于实际动力学计算）
            # 使用预测误差来补偿延迟
            alpha1_error = pred_alpha1 - self.alpha1_actual
            alpha2_error = pred_alpha2 - self.alpha2_actual
            theta1_error = pred_theta1 - self.theta1_actual
            theta2_error = pred_theta2 - self.theta2_actual
            
            # 计算角度命令，包含延迟补偿
            alpha1_cmd = pred_alpha1 - 0.5 * alpha1_error
            alpha2_cmd = pred_alpha2 - 0.5 * alpha2_error
            theta1_cmd = pred_theta1 - 0.5 * theta1_error
            theta2_cmd = pred_theta2 - 0.5 * theta2_error
            
            return np.array([T12, T34, T5, alpha1_cmd, alpha2_cmd, theta1_cmd, theta2_cmd])
            
        except:
            # 如果求解失败，使用简化方法
            return self.inverse_nonlinear_mapping_simple(W, state)
    
    def inverse_nonlinear_mapping_simple(self, W, state):
        """简化的非线性逆映射（备选方案）"""
        # 提取控制向量
        Fx, Fy, Fz, Tx, Ty, Tz = W
        
        # 尾部推力主要提供俯仰力矩
        T5 = Ty / self.l2 if abs(self.l2) > 1e-6 else 0
        
        # 左右机臂总推力提供升力
        total_front_thrust = max(0, Fz - T5)
        
        # 根据滚转力矩分配左右推力
        T12 = total_front_thrust/2 + Tx/(2*self.l1)
        T34 = total_front_thrust/2 - Tx/(2*self.l1)
        
        # 根据X方向力和偏航力矩计算机臂偏航角
        if abs(T12) > 1e-6:
            alpha1 = np.arctan2(Fx/2 - Tz/(2*self.l1), T12)
        else:
            alpha1 = 0.0
            
        if abs(T34) > 1e-6:
            alpha2 = np.arctan2(Fx/2 + Tz/(2*self.l1), T34)
        else:
            alpha2 = 0.0
        
        # 根据Y方向力计算螺旋桨倾转角
        if abs(T12) > 1e-6:
            theta1 = np.arcsin(Fy/(2*T12))
        else:
            theta1 = 0.0
            
        if abs(T34) > 1e-6:
            theta2 = np.arcsin(Fy/(2*T34))
        else:
            theta2 = 0.0
        
        return np.array([T12, T34, T5, alpha1, alpha2, theta1, theta2])

    def allocate_actuators(self, f_c_body: np.ndarray, tau_c: np.ndarray, state: dict):
        """分配执行器命令（使用带延迟补偿的非线性逆映射）"""
        # 构造控制向量W
        W = np.array([
            f_c_body[0],    # X力
            f_c_body[1],    # Y力
            f_c_body[2],    # Z力
            tau_c[0],       # 滚转力矩
            tau_c[1],       # 俯仰力矩
            tau_c[2]        # 偏航力矩
        ])
        
        # 带延迟补偿的非线性逆映射
        uu = self.inverse_nonlinear_mapping_with_delay(W, state)
        
        # 提取参数
        T12 = uu[0]  # 前左组推力
        T34 = uu[1]  # 前右组推力
        T5 = uu[2]   # 尾部推进器推力
        alpha1 = uu[3]  # roll左倾角
        alpha2 = uu[4]  # roll右倾角
        theta1 = uu[5]  # pitch左倾角
        theta2 = uu[6]  # pitch右倾角
        
        # 推力限制
        T_max = 60
        T12 = np.clip(T12, 0, T_max)
        T34 = np.clip(T34, 0, T_max)
        T5 = np.clip(T5, -15, 15)
        
        # 角度限制
        alpha_max = np.radians(90)
        alpha1 = np.clip(alpha1, -alpha_max, alpha_max)
        alpha2 = np.clip(alpha2, -alpha_max, alpha_max)
        theta_max = np.radians(90)
        theta1 = np.clip(theta1, -theta_max, theta_max)
        theta2 = np.clip(theta2, -theta_max, theta_max)
        
        # 更新状态
        self.T12 = T12
        self.T34 = T34
        self.T5 = T5
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.theta1 = theta1
        self.theta2 = theta2
        
        # 存储控制输入向量
        self.u = np.array([T12, T34, T5, alpha1, alpha2, theta1, theta2])
        
        return T12, T34, T5, alpha1, alpha2, theta1, theta2
    
    def _handle_angle_continuity(self, current: float, last: float) -> float:
        """处理角度连续性，避免跳变"""
        diff = current - last
        if diff > np.pi:
            return current - 2 * np.pi
        elif diff < -np.pi:
            return current + 2 * np.pi
        return current
    
    def set_actuators(self, T12: float, T34: float, T5: float, 
                     alpha1: float, alpha2: float, theta1: float, theta2: float):
        """应用控制命令到执行器"""
        try:            
            # 设置机臂偏航角度 (alpha)
            if 'tilt_pitch_left' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['tilt_pitch_left']] = alpha1
            
            if 'tilt_pitch_right' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['tilt_pitch_right']] = alpha2
            
            # 设置螺旋桨倾转角度 (theta)
            if 'tilt_roll_left' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['tilt_roll_left']] = theta1
            
            if 'tilt_roll_right' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['tilt_roll_right']] = theta2
            
            # 设置推力（左右机臂各有两个螺旋桨）
            if 'motor_l_upper' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['motor_l_upper']] = T12 / 2
            
            if 'motor_l_lower' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['motor_l_lower']] = T12 / 2
            
            if 'motor_r_upper' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['motor_r_upper']] = T34 / 2
            
            if 'motor_r_lower' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['motor_r_lower']] = T34 / 2
            
            # 尾部推进器
            if 'motor_rear_upper' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['motor_rear_upper']] = T5
                
        except Exception as e:
            print(f"设置执行器失败: {e}")
    
    def update_control(self):
        """更新控制量（使用几何解耦控制）"""
        try:
            # 获取当前状态
            state = self.get_state()

            # 计算控制力矩和力（使用几何解耦控制）
            f_c_body, tau_c = self.compute_control_wrench(state)
            
            # 分配执行器命令（带延迟补偿）
            T12, T34, T5, alpha1, alpha2, theta1, theta2 = self.allocate_actuators(f_c_body, tau_c, state)
            
            # 应用控制
            self.set_actuators(T12, T34, T5, alpha1, alpha2, theta1, theta2)
            
            # 记录状态
            self.log_status(state)
            
            return True
        except Exception as e:
            print(f"控制更新失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_status(self):
        """打印当前状态信息"""
        try:
            state = self.get_state()
            pos = state['position']
            euler_deg = np.degrees(state['euler'])
            target_euler_deg = np.degrees(self.target_attitude)
            
            # 获取当前轴类型
            axis_types = self._get_all_axis_types(state['euler'][1])
            
            # 获取当前增益
            KR_current = self._get_scheduled_gains(state['euler'][1])
            
            # 阶段名称映射
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
            print(f"位置: X={pos[0]:.3f}m, Y={pos[1]:.3f}m, Z={pos[2]:.3f}m")
            print(f"目标位置: X={self.target_position[0]:.3f}m, Y={self.target_position[1]:.3f}m, Z={self.target_position[2]:.3f}m")
            print(f"姿态: Roll={euler_deg[0]:.2f}°, Pitch={euler_deg[1]:.2f}°, Yaw={euler_deg[2]:.2f}°")  
            print(f"目标姿态: Roll={target_euler_deg[0]:.1f}°, Pitch={target_euler_deg[1]:.1f}°, Yaw={target_euler_deg[2]:.1f}°")
            print(f"控制力矩: X={self.tau_c[0]:.3f}Nm, Y={self.tau_c[1]:.3f}Nm, Z={self.tau_c[2]:.3f}Nm")
            print(f"执行器状态: T12={self.T12:.2f}N, T34={self.T34:.2f}N, T5={self.T5:.2f}N")
            print(f"机臂偏航: α1={math.degrees(self.alpha1):.2f}°(实际{math.degrees(self.alpha1_actual):.2f}°), "
                  f"α2={math.degrees(self.alpha2):.2f}°(实际{math.degrees(self.alpha2_actual):.2f}°)")
            print(f"螺旋桨倾转: θ1={math.degrees(self.theta1):.2f}°(实际{math.degrees(self.theta1_actual):.2f}°), "
                  f"θ2={math.degrees(self.theta2):.2f}°(实际{math.degrees(self.theta2_actual):.2f}°)")
            print(f"轴类型: Roll={axis_types[0]}, Pitch={axis_types[1]}, Yaw={axis_types[2]}")
            print(f"控制增益: KR=[{KR_current[0]:.2f}, {KR_current[1]:.2f}, {KR_current[2]:.2f}]")
            print(f"俯仰角限制: {'超限' if self.is_pitch_exceed else '正常'} (阈值: {self.pitch_threshold_deg}°)")
            print("--------------------------------------------------")
        except Exception as e:
            print(f"状态打印失败: {e}")
    
    def update_trajectory(self, current_time: float):
        """
        适配90°大角度的轨迹发布器
        """
        # 初始化阶段起始时间
        if self.trajectory_phase == 0 and self.phase_start_time == 0.0:
            self.phase_start_time = current_time
        
        # 阶段时长配置
        phase_durations = {
            0: 6.0,    # 起飞悬停
            1: 12.0,   # Roll转动
            2: 5.0,    # Roll保持
            3: 6.0,    # Roll恢复
            4: 12.0,   # Pitch转动
            5: 5.0,    # Pitch保持
            6: 6.0,    # Pitch恢复
            7: 12.0,   # Yaw转动
            8: 5.0,    # Yaw保持
            9: 6.0,    # Yaw恢复
            10: float('inf')  # 最终悬停
        }
        
        # 计算当前阶段已运行时间
        phase_elapsed = current_time - self.phase_start_time
        
        # 阶段切换判断
        if phase_elapsed > phase_durations[self.trajectory_phase]:
            self.trajectory_phase += 1
            self.phase_start_time = current_time
            print(f"\n🔄 轨迹阶段切换: {self.trajectory_phase-1} → {self.trajectory_phase}")
        
        # 各阶段轨迹逻辑
        if self.trajectory_phase == 0:
            # 阶段0：起飞悬停
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([0.0, 0.0, 0.0])
            
        elif self.trajectory_phase == 1:
            # 阶段1：Roll缓慢转动（0°→90°）
            progress = phase_elapsed / phase_durations[1]
            progress = np.clip(progress, 0.0, 1.0)
            roll_target = progress * self.attitude_target_rad
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([roll_target, 0.0, 0.0])
            
        elif self.trajectory_phase == 2:
            # 阶段2：Roll保持
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([self.attitude_target_rad, 0.0, 0.0])
            
        elif self.trajectory_phase == 3:
            # 阶段3：Roll恢复
            progress = phase_elapsed / phase_durations[3]
            progress = np.clip(progress, 0.0, 1.0)
            roll_target = (1 - progress) * self.attitude_target_rad
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([roll_target, 0.0, 0.0])
            
        elif self.trajectory_phase == 4:
            # 阶段4：Pitch缓慢转动
            progress = phase_elapsed / phase_durations[4]
            progress = np.clip(progress, 0.0, 1.0)
            pitch_target = progress * self.attitude_target_rad
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([0.0, pitch_target, 0.0])
            
        elif self.trajectory_phase == 5:
            # 阶段5：Pitch保持
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([0.0, self.attitude_target_rad, 0.0])
            
        elif self.trajectory_phase == 6:
            # 阶段6：Pitch恢复
            progress = phase_elapsed / phase_durations[6]
            progress = np.clip(progress, 0.0, 1.0)
            pitch_target = (1 - progress) * self.attitude_target_rad
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([0.0, pitch_target, 0.0])
            
        elif self.trajectory_phase == 7:
            # 阶段7：Yaw缓慢转动
            progress = phase_elapsed / phase_durations[7]
            progress = np.clip(progress, 0.0, 1.0)
            yaw_target = progress * self.attitude_target_rad
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([0.0, 0.0, yaw_target])
            
        elif self.trajectory_phase == 8:
            # 阶段8：Yaw保持
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([0.0, 0.0, self.attitude_target_rad])
            
        elif self.trajectory_phase == 9:
            # 阶段9：Yaw恢复
            progress = phase_elapsed / phase_durations[9]
            progress = np.clip(progress, 0.0, 1.0)
            yaw_target = (1 - progress) * self.attitude_target_rad
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([0.0, 0.0, yaw_target])
            
        else:
            # 阶段10：最终悬停
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([0.0, 0.0, 0.0])
        
        # 速度/加速度归零
        self.target_velocity = np.zeros(3)
        self.target_acceleration = np.zeros(3)
        self.target_attitude_rate = np.zeros(3)
        self.target_attitude_acceleration = np.zeros(3)
    

def main():
    """主函数 - 启动几何解耦控制仿真"""
    print("=== 倾转旋翼无人机几何解耦控制仿真 ===")
    print("核心优化：基于倾转预测的几何解耦控制方案")
    print("方案特点：")
    print("  1. 虚拟坐标系解耦快慢响应轴")
    print("  2. 基于俯仰角的增益调度")
    print("  3. 舵机动态延迟补偿")
    print("  4. 自适应轴类型切换")
    
    try:
        # 初始化控制器
        controller = HnuterController("hnuter201.xml")
        
        # 初始目标
        controller.target_position = np.array([0.0, 0.0, 2.0])
        controller.target_attitude = np.array([0.0, 0.0, 0.0])
        
        # 启动 Viewer
        with viewer.launch_passive(controller.model, controller.data) as v:
            print(f"\n仿真启动：")
            print(f"日志文件: {controller.log_file}")
            print("控制指令:")
            print("  r - 重置仿真")
            print("  p - 暂停/继续")
            print("  q - 退出")
            print("按 Ctrl+C 终止仿真")
            
            start_time = time.time()
            last_print_time = 0
            print_interval = 1.0
            paused = False
            
            try:
                while v.is_running():
                    current_time = time.time() - start_time
                    
                    # 检查键盘输入
                    key = v.get_key() if hasattr(v, 'get_key') else None
                    if key == 'r':  # 重置
                        mj.mj_resetData(controller.model, controller.data)
                        start_time = time.time()
                        controller.trajectory_phase = 0
                        controller.phase_start_time = 0.0
                        print("仿真已重置")
                    elif key == 'p':  # 暂停
                        paused = not paused
                        print("暂停" if paused else "继续")
                    elif key == 'q':  # 退出
                        break
                    
                    if not paused:
                        # 更新轨迹
                        controller.update_trajectory(current_time)
                        
                        # 更新控制
                        controller.update_control()
                        
                        # 仿真步进
                        mj.mj_step(controller.model, controller.data)
                    
                    # 同步可视化
                    v.sync()
                    
                    # 定期打印状态
                    if current_time - last_print_time > print_interval:
                        controller.print_status()
                        last_print_time = current_time
                    
                    # 控制仿真速率
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