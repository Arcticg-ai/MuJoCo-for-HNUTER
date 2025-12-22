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
        
        # 几何控制器增益 (根据论文设置)
        self.Kp = np.diag([5, 5, 5])  # 位置增益
        self.Dp = np.diag([10, 10, 10])  # 速度阻尼
        self.KR = np.array([0.5, 0.5, 0.5])   # 姿态增益
        self.Domega = np.array([0.05, 0.05, 0.05])  # 角速度阻尼

        # 控制量
        self.f_c_body = np.zeros(3)  # 机体坐标系下的控制力
        self.f_c_world = np.zeros(3)  # 世界坐标系下的控制力
        self.tau_c = np.zeros(3)     # 控制力矩
        self.u = np.zeros(7)         # 控制输入向量


        # 积分分离和抗饱和机制相关属性
        self.integral_separation_threshold = 0.2  # 积分分离阈值（弧度）
        self.integral_error = np.zeros(3)  # 积分误差累积
        self.KI = np.array([0.01, 0.01, 0.005])  # 积分增益
        self.integral_limit = 1.0  # 积分限幅

        # 分配矩阵 (根据模型结构更新)
        self.A = np.array([
            [1, 0,  0, 1, 0,  0, 0,],   # X力分配 
            [0, 0, -1, 0, 0, -1, 0],   # Y力分配
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
        # print(self._get_actuator_ids)
        self._get_sensor_ids()
        
        # 创建日志文件
        self._create_log_file()
              
        print("倾转旋翼控制器初始化完成（含几何控制器）")
    
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
        self.log_file = f'logs/drone_log_{timestamp}.csv'
        
        # 写入CSV表头
        with open(self.log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'timestamp', 'pos_x', 'pos_y', 'pos_z', 
                'target_x', 'target_y', 'target_z',
                'roll', 'pitch', 'yaw',
                'vel_x', 'vel_y', 'vel_z',
                'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
                'accel_x', 'accel_y', 'accel_z',
                'f_world_x', 'f_world_y', 'f_world_z',
                'f_body_x', 'f_body_y', 'f_body_z',
                'tau_x', 'tau_y', 'tau_z',
                'u1', 'u2', 'u3', 'u4', 'u5',
                'T12', 'T34', 'T5',
                'alpha1', 'alpha2'
                'theta1', 'theta2'
            ])
        
        print(f"已创建日志文件: {self.log_file}")
    
    def log_status(self, state: dict):
        """记录状态到日志文件"""
        timestamp = time.time()
        # 确保状态字典中有所有必要的键
        position = state.get('position', np.zeros(3))
        velocity = state.get('velocity', np.zeros(3))
        angular_velocity = state.get('angular_velocity', np.zeros(3))
        acceleration = state.get('acceleration', np.zeros(3))
        euler = state.get('euler', np.zeros(3))
        
        with open(self.log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                timestamp,
                position[0], position[1], position[2],
                self.target_position[0], self.target_position[1], self.target_position[2],
                euler[0], euler[1], euler[2],
                velocity[0], velocity[1], velocity[2],
                angular_velocity[0], angular_velocity[1], angular_velocity[2],
                acceleration[0], acceleration[1], acceleration[2],
                self.f_c_world[0], self.f_c_world[1], self.f_c_world[2],
                self.f_c_body[0], self.f_c_body[1], self.f_c_body[2],
                self.tau_c[0], self.tau_c[1], self.tau_c[2],
                self.u[0], self.u[1], self.u[2], self.u[3], self.u[4],
                self.T12, self.T34, self.T5,
                self.alpha1, self.alpha2,
                self.theta1, self.theta2
                
            ])
    
    def _get_actuator_ids(self):
        """获取执行器ID"""
        self.actuator_ids = {}
        
        # 尝试获取执行器ID，处理可能的异常
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
            # 创建默认映射
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
        
        sensor_mappings = {
            'drone_pos': 'drone_pos',
            'drone_quat': 'drone_quat', 
            'body_vel': 'body_vel',
            'body_gyro': 'body_gyro',
            'body_acc': 'body_acc'
        }



        # 尝试获取传感器ID，处理可能的异常
        try:
            # 位置和姿态传感器
            self.sensor_ids['drone_pos'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, 'drone_pos')
            self.sensor_ids['drone_quat'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, 'drone_quat')
            
            # 速度传感器
            self.sensor_ids['body_vel'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, 'body_vel')
            
            # 角速度传感器
            self.sensor_ids['body_gyro'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, 'body_gyro')
            
            # 加速度传感器
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
            # 创建默认映射
            self.sensor_ids = {
                'drone_pos': 0,
                'drone_quat': 1,
                'body_vel': 2,
                'body_gyro': 3,
                'body_acc': 4
            }
            print("使用默认传感器ID映射")
    
    def get_state(self) -> dict:
        """获取无人机当前状态"""
        state = {
            'position': np.zeros(3),
            'quaternion': np.array([1.0, 0.0, 0.0, 0.0]),
            'rotation_matrix': np.eye(3),
            'velocity': np.zeros(3),
            'angular_velocity': np.zeros(3),
            'acceleration': np.zeros(3),
            'euler': np.zeros(3)
        }
        
        try:
            # 直接从body获取位置和姿态（更可靠）
            body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'drone')
            if body_id != -1:
                # 位置
                state['position'] = self.data.xpos[body_id].copy()
                # 四元数姿态
                state['quaternion'] = self.data.xquat[body_id].copy()
                # 速度（物体坐标系下的线速度）
                state['velocity'] = self.data.cvel[body_id][3:6].copy()  # 线性速度部分
                # 角速度（物体坐标系下）
                state['angular_velocity'] = self.data.cvel[body_id][0:3].copy()  # 角速度部分
                
            # 转换为旋转矩阵和欧拉角
            state['rotation_matrix'] = self._quat_to_rotation_matrix(state['quaternion'])
            state['euler'] = self._quat_to_euler(state['quaternion'])
            
            # 验证数据合理性
            if np.any(np.isnan(state['position'])):
                print("警告: 位置数据包含NaN，使用零值")
                state['position'] = np.zeros(3)
                
            return state
        except Exception as e:
            print(f"状态获取错误: {e}")
            return state

    def _quat_to_rotation_matrix(self, quat: np.ndarray) -> np.ndarray:
        """
        四元数转旋转矩阵
        """
        w, x, y, z = quat
        
        # 计算旋转矩阵的各个元素
        R11 = 1 - 2 * (y * y + z * z)
        R12 = 2 * (x * y - w * z)
        R13 = 2 * (x * z + w * y)
        
        R21 = 2 * (x * y + w * z)
        R22 = 1 - 2 * (x * x + z * z)
        R23 = 2 * (y * z - w * x)
        
        R31 = 2 * (x * z - w * y)
        R32 = 2 * (y * z + w * x)
        R33 = 1 - 2 * (x * x + y * y)
        
        # 构造旋转矩阵
        rotation_matrix = np.array([
            [R11, R12, R13],
            [R21, R22, R23],
            [R31, R32, R33]
        ])
        
        return rotation_matrix

    def _quat_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """
        四元数转欧拉角 (roll, pitch, yaw)
        """
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
    
    def _limit_angle_rate(self, last_angle: float, target_angle: float, max_rate: float) -> float:
        """限制角度变化速率，防止角度突变
        
        参数:
            last_angle: 上一时刻的角度
            target_angle: 当前计算的目标角度
            max_rate: 最大允许的角度变化率（弧度/秒）
            
        返回:
            受速率限制后的角度
        """
        # 计算角度差（考虑角度周期性，确保选择最小旋转方向）
        angle_diff = target_angle - last_angle
        angle_diff = ((angle_diff + np.pi) % (2 * np.pi)) - np.pi
        
        # 计算最大允许的角度变化
        max_angle_diff = max_rate * self.dt
        
        # 限制角度变化
        if abs(angle_diff) > max_angle_diff:
            return last_angle + np.sign(angle_diff) * max_angle_diff
        else:
            return target_angle
    
    def compute_control_wrench(self, state: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        几何控制器核心计算
        返回：f_c_body (机体系控制力), tau_c (机体系控制力矩)
        """
        # 位置控制器（在惯性系中计算）
        e_x = state['position'] - self.target_position
        e_v = state['velocity'] - self.target_velocity
        
        # 位置控制律
        f_c_world = self.mass * self.gravity * np.array([0, 0, 1]) + \
                    self.mass * self.target_acceleration - \
                    self.Kp @ e_x - \
                    self.Dp @ e_v
        
        # 转换到机体系
        f_c_body = state['rotation_matrix'].T @ f_c_world
        
        # 姿态控制器（在机体系中计算）
        R = state['rotation_matrix']
        R_d = self._euler_to_rotation_matrix(self.target_attitude)
        Omega = state['angular_velocity']
        Omega_d = self.target_attitude_rate
        
        # 姿态误差计算
        # 使用对数映射计算姿态误差
        # e_R_matrix = R_d.T @ R
        # e_R = self.vee_map(0.5 * (e_R_matrix - e_R_matrix.T))
        e_R_matrix = 0.5 * (R_d.T @ R - R.T @ R_d)  # 反对称矩阵表示姿态误差
        e_R = self.vee_map(e_R_matrix)  # vee映射得到3维向量
        
        # 角速度误差
        e_Omega = Omega - R.T @ R_d @ Omega_d
        
        # # *** 积分分离逻辑 ***
        # # 计算当前误差的范数（幅度）
        # e_R_norm = np.linalg.norm(e_R)
        # # 决定是否启用积分
        # if e_R_norm < self.integral_separation_threshold:
        #     # 误差小，启用积分：累加误差
        #     self.integral_error += e_R * self.dt
        # else:
        #     # 误差大，关闭积分：清除积分项，防止饱和
        #     self.integral_error.fill(0)

        # # 积分项也要限制幅度，防止无限增大（简单的积分限幅）
        # integral_limit = 1.0  # 根据实际情况调整
        # self.integral_error = np.clip(self.integral_error, -integral_limit, integral_limit)

        # # 新的控制律，包含比例、微分、积分（条件性启用）项
        # tau_c = -self.KR * e_R - self.Domega * e_Omega + self.KI * self.integral_error


        # 姿态控制律
        tau_c = -self.KR * e_R - \
                self.Domega * e_Omega + \
                np.cross(Omega, self.J @ Omega) - \
                self.J @ (self.hat_map(Omega) @ R.T @ R_d @ Omega_d - R.T @ R_d @ self.target_attitude_acceleration)
        # tau_c = -self.KR * e_R - self.Domega * e_Omega + np.cross(Omega, self.J @ Omega)

        return f_c_body, tau_c
    
    def _euler_to_rotation_matrix(self, euler: np.ndarray) -> np.ndarray:
        """
        欧拉角转旋转矩阵 (roll, pitch, yaw)
        """
        roll, pitch, yaw = euler
        
        # 绕X轴旋转 (roll)
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # 绕Y轴旋转 (pitch)
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # 绕Z轴旋转 (yaw)
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # 组合旋转 (Z-Y-X顺序)
        return R_z @ R_y @ R_x
    
    def inverse_nonlinear_mapping(self, W):
        """
        重新设计的逆映射函数 - 使用更直接、更稳定的计算方法
        输入：W（6×1向量）[Fx, Fy, Fz, τx, τy, τz]
        输出：uu = [F1, F2, F3, alpha1, alpha2, theta1, theta2]
        """
        # 提取控制需求
        Fx = W[0]  # X方向力
        Fy = W[1]  # Y方向力
        Fz = W[2]  # Z方向力（主要升力）
        tau_x = W[3]  # 滚转力矩
        tau_y = W[4]  # 俯仰力矩
        tau_z = W[5]  # 偏航力矩
        
        # 简化模型：假设左右两组旋翼对称工作，这样可以大大简化计算
        # 设置尾部推进器推力
        F3 = (5/3) * tau_y  # 尾部推进器主要负责俯仰力矩
        
        # 计算前旋翼组总升力需求
        # 主要升力来自于Z方向，需要平衡重力
        total_lift_needed = max(Fz, 10.0)  # 确保有足够的升力，避免过小
        
        # 分配到左右两组旋翼（基本对称，根据偏航力矩微调）
        yaw_adjustment = (5/4) * tau_z  # 偏航力矩引起的推力差异
        F1 = total_lift_needed / 2 - yaw_adjustment  # 左组推力
        F2 = total_lift_needed / 2 + yaw_adjustment  # 右组推力
        
        # 确保推力为正
        F1 = max(F1, 1.0)  # 最小推力，避免接近零
        F2 = max(F2, 1.0)  # 最小推力，避免接近零
        
        # 计算倾转角度theta
        # theta主要控制X方向的力
        # 基于简单的物理模型：Fx = F1*sin(theta) + F2*sin(theta) = (F1+F2)*sin(theta)
        total_forward_force_capability = F1 + F2
        
        # 计算所需的theta角度
        if total_forward_force_capability > 0:
            # 限制最大前向力比例，避免过大的theta
            max_forward_force_ratio = 0.5  # 最大使用50%的力产生前向推力
            max_Fx = max_forward_force_ratio * total_forward_force_capability
            limited_Fx = np.clip(Fx, -max_Fx, max_Fx)
            
            # 计算theta角度
            sin_theta = limited_Fx / total_forward_force_capability
            sin_theta = np.clip(sin_theta, -0.999, 0.999)  # 避免arcsin边界问题
            theta = np.arcsin(sin_theta)
        else:
            theta = 0.0  # 默认值
        
        # 应用角度连续性 - 基于当前角度进行平滑过渡
        current_avg_theta = (self.last_theta1 + self.last_theta2) / 2
        # 使用低通滤波平滑角度变化
        smooth_factor = 0.3  # 平滑因子，值越小变化越平滑
        theta = smooth_factor * theta + (1 - smooth_factor) * current_avg_theta
        
        # 计算roll角度alpha1和alpha2
        # alpha主要控制Y方向的力和滚转力矩
        # 假设alpha1和alpha2大小相等，方向相反，以产生滚转力矩
        # 基于滚转力矩计算alpha差异
        alpha_diff = tau_x / (self.l1 * (F1 + F2))  # 滚转力矩与alpha差异的关系
        alpha_diff = np.clip(alpha_diff, -np.radians(15), np.radians(15))  # 限制最大差异
        
        # 计算Y方向力所需的平均alpha
        if (F1 + F2) * np.cos(theta) > 0:
            sin_alpha_avg = Fy / ((F1 + F2) * np.cos(theta))
            sin_alpha_avg = np.clip(sin_alpha_avg, -0.999, 0.999)  # 避免边界问题
            alpha_avg = np.arcsin(sin_alpha_avg)
        else:
            alpha_avg = 0.0
        
        # 计算最终的alpha1和alpha2
        alpha1 = alpha_avg - alpha_diff / 2  # 左倾角度
        alpha2 = alpha_avg + alpha_diff / 2  # 右倾角度
        
        # 限制所有角度在合理范围内
        max_alpha = np.radians(30)  # 减小最大alpha角度以提高稳定性
        alpha1 = np.clip(alpha1, -max_alpha, max_alpha)
        alpha2 = np.clip(alpha2, -max_alpha, max_alpha)
        
        max_theta = np.radians(30)  # 进一步减小最大theta角度
        theta = np.clip(theta, -max_theta, max_theta)
        
        # 确保theta1和theta2相等（对称）
        theta1 = theta
        theta2 = theta
        
        # 组合输出
        uu = np.array([F1, F2, F3, alpha1, alpha2, theta1, theta2])
        return uu
    
    def allocate_actuators(self, f_c_body: np.ndarray, tau_c: np.ndarray, state: dict):
        """增强的执行器分配函数，强化稳定机制，解决theta持续增大问题"""
        # 初始化趋势检测变量（如果不存在）
        if not hasattr(self, 'theta_trend'):
            self.theta_trend = False
            self.theta_trend_count = 0
        if not hasattr(self, 'last_alpha1'):
            self.last_alpha1 = 0.0
            self.last_alpha2 = 0.0
        
        # 加强物理约束检查，避免过大的控制力
        max_force = 50  # 降低最大允许力
        max_torque = 5  # 降低最大允许力矩
        
        # 更严格地限制控制力和力矩
        f_c_body_clamped = np.clip(f_c_body, -max_force, max_force)
        tau_c_clamped = np.clip(tau_c, -max_torque, max_torque)
        
        # 构造控制向量W（6×1向量）
        # W = [Fx, Fy, Fz, τx, τy, τz]
        W = np.array([
            f_c_body_clamped[0],    # X力
            f_c_body_clamped[1],    # Y力
            f_c_body_clamped[2],    # Z力
            tau_c_clamped[0],       # 滚转力矩
            tau_c_clamped[1],       # 俯仰力矩
            tau_c_clamped[2]        # 偏航力矩
        ])
        
        # 使用非线性逆映射
        uu = self.inverse_nonlinear_mapping(W)
        
        # 提取参数
        F1 = uu[0]  # 前左组推力
        F2 = uu[1]  # 前右组推力
        F3 = uu[2]  # 尾部推进器推力
        alpha1 = uu[3]  # roll左倾角
        alpha2 = uu[4]  # roll右倾角
        theta1 = uu[5]  # pitch左倾角
        theta2 = uu[6]  # pitch右倾角
        
        # 1. 更严格的推力限制，加入最小推力保证稳定性
        T_max = 30  # 降低最大推力
        min_thrust = 5.0  # 设定最小推力，防止过低导致不稳定
        F1 = np.clip(F1, min_thrust, T_max)
        F2 = np.clip(F2, min_thrust, T_max)
        F3 = np.clip(F3, -5, 5)  # 更严格限制尾部推力
        
        # 2. 强制对称性 - 确保左右旋翼工作对称
        avg_theta = (theta1 + theta2) / 2
        # 强制theta1和theta2完全相同
        theta1 = avg_theta
        theta2 = avg_theta
        
        # 3. 大幅收紧角度限制，提高稳定性
        alpha_max = np.radians(20)  # 减小到20度
        theta_max = np.radians(15)  # 减小到15度
        alpha1 = np.clip(alpha1, -alpha_max, alpha_max)
        alpha2 = np.clip(alpha2, -alpha_max, alpha_max)
        theta1 = np.clip(theta1, -theta_max, theta_max)
        theta2 = np.clip(theta2, -theta_max, theta_max)
        
        # 4. 角度速率限制 - 更严格
        max_theta_rate = np.radians(10)  # 减小到每秒10度
        max_alpha_rate = np.radians(15)  # 减小到每秒15度
        theta1 = self._limit_angle_rate(self.last_theta1, theta1, max_theta_rate)
        theta2 = theta1  # 保持theta1和theta2一致
        alpha1 = self._limit_angle_rate(self.last_alpha1, alpha1, max_alpha_rate)
        alpha2 = self._limit_angle_rate(self.last_alpha2, alpha2, max_alpha_rate)
        
        # 5. 强中心化机制 - 主动拉向零位
        # 无论theta多小，都施加轻微的中心化力
        centering_factor = 0.05 * self.dt  # 基于时间步长的中心化因子
        theta1 = theta1 * (1 - centering_factor)
        theta2 = theta1
        
        # 6. 趋势检测与抑制 - 彻底解决theta持续增大问题
        theta_abs_prev = abs(self.last_theta1)
        theta_abs_curr = abs(theta1)
        
        # 更新趋势检测
        if theta_abs_curr > theta_abs_prev * 1.05:  # 增长超过5%
            self.theta_trend_count += 1
            if self.theta_trend_count > 3:  # 连续3步都在增长
                # 强力抑制增长趋势
                theta1 = theta1 * 0.8  # 立即减少20%
                theta2 = theta1
                self.theta_trend_count = 0  # 重置计数器
        else:
            # 没有增长趋势，递减计数器
            self.theta_trend_count = max(0, self.theta_trend_count - 1)
        
        # 7. 当theta超过阈值时，施加更强的恢复力
        theta_threshold = np.radians(5)  # 降低阈值到5度
        recovery_factor = 0.2  # 增大恢复因子到20%
        if abs(theta1) > theta_threshold:
            # 与角度大小成比例的恢复力，角度越大恢复越强
            recovery_strength = recovery_factor * (abs(theta1) / theta_threshold)
            theta1 = theta1 * (1 - recovery_strength)
            theta2 = theta1
        
        # 8. 保存当前角度用于下次角度速率限制
        self.last_theta1 = theta1
        self.last_theta2 = theta2
        self.last_alpha1 = alpha1
        self.last_alpha2 = alpha2
        
        # 更新状态
        self.T12 = F1
        self.T34 = F2
        self.T5 = F3
        self.alpha1 = alpha1  # roll右倾角
        self.alpha2 = alpha2  # roll左倾角
        self.theta1 = theta1
        self.theta2 = theta2
        
        # 存储控制输入向量（用于日志记录）
        self.u = np.array([F1, F2, F3, alpha1, alpha2, theta1, theta2])
        
        return F1, F2, F3, alpha1, alpha2, theta1, theta2

    def set_actuators(self, T12: float, T34: float, T5: float, alpha1: float, alpha2: float, theta1: float, theta2: float):
        """应用控制命令到执行器"""
        try:
            # 设置机臂偏航角度（保持水平）
            if 'arm_pitch_right' in self.actuator_ids:
                arm_pitch_right_id = self.actuator_ids['arm_pitch_right']
                self.data.ctrl[arm_pitch_right_id] = alpha1  # 保持水平
            
            if 'arm_pitch_left' in self.actuator_ids:
                arm_pitch_left_id = self.actuator_ids['arm_pitch_left']
                self.data.ctrl[arm_pitch_left_id] = alpha2  # 保持水平
            
            # 设置螺旋桨倾转角度
            if 'prop_tilt_right' in self.actuator_ids:
                tilt_right_id = self.actuator_ids['prop_tilt_right']
                self.data.ctrl[tilt_right_id] = theta1  # 右侧倾角
            
            if 'prop_tilt_left' in self.actuator_ids:
                tilt_left_id = self.actuator_ids['prop_tilt_left']
                self.data.ctrl[tilt_left_id] = theta2  # 左侧倾角
            
            # 设置推力
            # 右侧两个螺旋桨（每个推力为总推力的一半）
            if 'motor_r_upper' in self.actuator_ids:
                thrust_rt_id = self.actuator_ids['motor_r_upper']
                self.data.ctrl[thrust_rt_id] = T34 / 2
            
            if 'motor_r_lower' in self.actuator_ids:
                thrust_rb_id = self.actuator_ids['motor_r_lower']
                self.data.ctrl[thrust_rb_id] = T34 / 2
            
            # 左侧两个螺旋桨
            if 'motor_l_upper' in self.actuator_ids:
                thrust_lt_id = self.actuator_ids['motor_l_upper']
                self.data.ctrl[thrust_lt_id] = T12 / 2
            
            if 'motor_l_lower' in self.actuator_ids:
                thrust_lb_id = self.actuator_ids['motor_l_lower']
                self.data.ctrl[thrust_lb_id] = T12 / 2
            
            # 尾部推进器
            if 'motor_rear_upper' in self.actuator_ids:
                thrust_tail_id = self.actuator_ids['motor_rear_upper']
                self.data.ctrl[thrust_tail_id] = T5
                
            # print(self.data.ctrl)
        except Exception as e:
            print(f"设置执行器失败: {e}")
    
    def update_control(self):
        """更新控制命令（整合几何控制器和改进的逆映射）"""
        # 初始化控制状态变量（如果不存在）
        if not hasattr(self, 'theta_control_state'):
            # 创建一个专用的控制状态对象来跟踪和限制theta
            self.theta_control_state = {
                'last_theta': 0.0,
                'theta_integral': 0.0,
                'theta_peak': 0.0,
                'recovery_engaged': False,
                'steps_since_growth': 0,
                'total_recovery_attempts': 0
            }
        
        # 初始化调试计数
        if not hasattr(self, 'debug_count'):
            self.debug_count = 0
            self.theta_history = []  # 用于记录theta角度历史
            self.last_log_time = 0
        
        try:
            # 获取当前状态
            state = self.get_state()

            # 记录状态到日志
            self.log_status(state)
            
            # 计算控制力和力矩 - 增强版几何控制器
            f_c_body, tau_c = self.compute_control_wrench(state)
            
            # 1. 整合控制需求为W向量 - 确保使用正确的力/力矩
            W = np.array([
                f_c_body[0],  # X方向力
                f_c_body[1],  # Y方向力
                f_c_body[2],  # Z方向力
                tau_c[0],     # 滚转力矩
                tau_c[1],     # 俯仰力矩
                tau_c[2]      # 偏航力矩
            ])
            
            # 2. 使用我们重写的逆映射方法计算控制参数
            uu = self.inverse_nonlinear_mapping(W)
            
            # 3. 提取控制参数
            F1, F2, F3, alpha1, alpha2, theta1, theta2 = uu
            
            # 4. 关键保护措施：强制theta1和theta2完全一致
            avg_theta = (theta1 + theta2) / 2
            theta1 = avg_theta
            theta2 = avg_theta
            
            # 5. 强力稳定化：专用的theta控制算法
            current_theta = avg_theta
            last_theta = self.theta_control_state['last_theta']
            
            # 检测异常增长
            theta_abs = abs(current_theta)
            last_theta_abs = abs(last_theta)
            
            if theta_abs > last_theta_abs * 1.03:  # 增长超过3%
                self.theta_control_state['steps_since_growth'] = 0
                self.theta_control_state['theta_peak'] = max(self.theta_control_state['theta_peak'], theta_abs)
                # 如果持续增长且超过阈值，启动紧急恢复
                if self.theta_control_state['theta_peak'] > np.radians(10):
                    self.theta_control_state['recovery_engaged'] = True
            else:
                # 没有持续增长
                self.theta_control_state['steps_since_growth'] += 1
                # 如果一段时间没有增长，重置峰值和恢复状态
                if self.theta_control_state['steps_since_growth'] > 20:  # 20个时间步没有增长
                    self.theta_control_state['theta_peak'] = 0.0
                    self.theta_control_state['recovery_engaged'] = False
            
            # 应用恢复控制
            if self.theta_control_state['recovery_engaged']:
                self.theta_control_state['total_recovery_attempts'] += 1
                # 比例+积分控制，将theta拉回零位
                recovery_gain = 0.3  # 比例增益
                integral_gain = 0.05  # 积分增益
                
                # 更新积分项
                self.theta_control_state['theta_integral'] += current_theta * self.dt
                # 积分饱和限制
                max_integral = np.radians(5)
                self.theta_control_state['theta_integral'] = np.clip(
                    self.theta_control_state['theta_integral'], 
                    -max_integral, 
                    max_integral
                )
                
                # 计算恢复控制
                recovery_control = -(recovery_gain * current_theta + integral_gain * self.theta_control_state['theta_integral'])
                
                # 应用恢复控制
                theta1 += recovery_control
                theta2 += recovery_control
            
            # 6. 最终限制 - 确保角度在安全范围内
            max_theta = np.radians(15)  # 最终的严格限制
            theta1 = np.clip(theta1, -max_theta, max_theta)
            theta2 = np.clip(theta2, -max_theta, max_theta)
            
            # 7. 强制对称的alpha角度
            avg_alpha = (alpha1 + alpha2) / 2
            alpha_diff = (alpha2 - alpha1) / 2  # 保持滚转差异
            alpha1 = avg_alpha - alpha_diff
            alpha2 = avg_alpha + alpha_diff
            
            # 8. 限制alpha角度
            max_alpha = np.radians(20)
            alpha1 = np.clip(alpha1, -max_alpha, max_alpha)
            alpha2 = np.clip(alpha2, -max_alpha, max_alpha)
            
            # 9. 限制推力
            max_F = 30.0
            min_F = 5.0
            F1 = np.clip(F1, min_F, max_F)
            F2 = np.clip(F2, min_F, max_F)
            F3 = np.clip(F3, -10.0, 10.0)
            
            # 10. 更新状态
            self.theta_control_state['last_theta'] = current_theta
            
            # 保存控制量用于日志
            self.f_c_body = f_c_body
            self.tau_c = tau_c
            self.f_c_world = state['rotation_matrix'] @ f_c_body
            self.T12 = F1
            self.T34 = F2
            self.T5 = F3
            self.alpha1 = alpha1
            self.alpha2 = alpha2
            self.theta1 = theta1
            self.theta2 = theta2
            self.u = np.array([F1, F2, F3, alpha1, alpha2, theta1, theta2])
            
            # 11. 设置执行器
            self.set_actuators(F1, F2, F3, alpha1, alpha2, theta1, theta2)
            
            # 12. 调试信息
            if hasattr(self, 'debug_count') and self.debug_count % 100 == 0:
                if self.theta_control_state['recovery_engaged']:
                    print(f"调试: Theta恢复控制已启动 - 当前角度: {np.degrees(theta1):.2f}度")
                if self.theta_control_state['total_recovery_attempts'] > 100:
                    print(f"调试: 系统可能存在持续不稳定 - 恢复尝试次数: {self.theta_control_state['total_recovery_attempts']}")
            
            return True
        except Exception as e:
            print(f"控制更新失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_status(self):
        """打印当前状态信息（包含控制信息）"""
        try:
            state = self.get_state()
            pos = state['position']
            vel = state['velocity']
            accel = state['acceleration']
            euler_deg = np.degrees(state['euler'])
            target_euler_deg = np.degrees(self.target_attitude)
            
            print(f"位置: X={pos[0]:.2f}m, Y={pos[1]:.2f}m, Z={pos[2]:.2f}m")
            print(f"目标位置: X={self.target_position[0]:.2f}m, Y={self.target_position[1]:.2f}m, Z={self.target_position[2]:.2f}m")
            print(f"姿态: Roll={euler_deg[0]:.6f}°, Pitch={euler_deg[1]:.6f}°, Yaw={euler_deg[2]:.6f}°")  
            print(f"目标姿态: Roll={target_euler_deg[0]:.1f}°, Pitch={target_euler_deg[1]:.1f}°, Yaw={target_euler_deg[2]:.1f}°") 
            print(f"速度: X={vel[0]:.2f}m/s, Y={vel[1]:.2f}m/s, Z={vel[2]:.2f}m/s")
            print(f"加速度: X={accel[0]:.2f}m/s², Y={accel[1]:.2f}m/s², Z={accel[2]:.2f}m/s²")
            print(f"控制力: X={self.f_c_body[0]:.2f}N, Y={self.f_c_body[1]:.2f}N, Z={self.f_c_body[2]:.2f}N")
            print(f"控制力矩: X={self.tau_c[0]:.6f}Nm, Y={self.tau_c[1]:.6f}Nm, Z={self.tau_c[2]:.6f}Nm")
            print(f"执行器状态: T12={self.T12:.2f}N, T34={self.T34:.2f}N, T5={self.T5:.2f}N, α1={math.degrees(self.alpha1):.6f}°, α2={math.degrees(self.alpha2):.6f}°, theta1={math.degrees(self.theta1):.6f}°, theta2={math.degrees(self.theta2):.6f}°")
            print("--------------------------------------------------")
        except Exception as e:
            print(f"状态打印失败: {e}")


def main():
    """主函数 - 启动仿真"""
    print("=== 倾转旋翼无人机状态监控系统 ===")
    
    try:
        # 初始化控制器
        controller = HnuterController("hnuter201.xml")
        
        # 设置目标轨迹（简单悬停）
        controller.target_position = np.array([3.0, 0.0, 1.5])  # 目标高度1.5米
        controller.target_velocity = np.zeros(3)
        controller.target_acceleration = np.zeros(3)
        controller.target_attitude = np.zeros(3)  # 水平姿态
        controller.target_attitude_rate = np.zeros(3)
        controller.target_attitude_acceleration = np.zeros(3)
        
        # 启动 Viewer
        with viewer.launch_passive(controller.model, controller.data) as v:
            print("\n仿真启动：")
            print("按 Ctrl+C 终止仿真")
            print(f"日志文件路径: {controller.log_file}")
            
            start_time = time.time()
            last_print_time = 0
            last_plot_update = 0

            print_interval = 0.5  # 打印间隔 (秒)
            plot_update_interval = 0.1  # 绘图更新间隔
            count = 0
            
            try:
                while v.is_running():
                    current_time = time.time() - start_time

                    # 更新控制
                    controller.update_control()
                    
                    # controller.set_actuators(20.5,20.5,0,0.0,0.0,0.0,0.0)

                    count = count + 1
                    if count % 3 == 0:
                        # 仿真步进
                        mj.mj_step(controller.model, controller.data)
                    
                    # 同步可视化
                    v.sync()
                    
                    # 定期打印状态
                    if current_time - last_print_time > print_interval:
                        controller.print_status()
                        last_print_time = current_time

                    time.sleep(0.001)

            except KeyboardInterrupt:
                print("\n仿真中断")
            
            print("仿真结束")
            plt.savefig(controller.log_file.replace('.csv', '.png'))
            plt.close()
            print(f"图表保存至: {controller.log_file.replace('.csv', '.png')}")

            final_state = controller.get_state()
            print(f"最终位置: ({final_state['position'][0]:.2f}, {final_state['position'][1]:.2f}, {final_state['position'][2]:.2f})")
    
    except Exception as e:
        print(f"仿真主循环失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()