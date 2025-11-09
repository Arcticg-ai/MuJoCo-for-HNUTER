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
        非线性逆映射函数
        输入：W（6×1向量）
        输出：uu = [F1, F2, F3, alpha1, alpha2, theta1, theta2]，其中alpha1 = alpha2
        """
        # 步骤1：计算u中与t无关的分量（u1, u2, u4, u5, u7）
        u7 = (5/3) * W[4]                     # 由W(5)确定u7，注意Python索引从0开始
        u1 = W[0]/2 - (5/4)*W[5]              # 由W(1)和W(6)确定u1
        u4 = W[0]/2 + (5/4)*W[5]              # 由W(1)和W(6)确定u4
        u2 = (W[2] - (5/3)*W[4])/2 + (5/4)*W[3]  # 由W(3), W(4), W(5)确定u2
        u5 = (W[2] - (5/3)*W[4])/2 - (5/4)*W[3]  # 由W(3), W(4), W(5)确定u5
        
        # 步骤2：利用alpha1 = alpha2约束求解u3 = t
        C1 = u1**2 + u2**2  # F1² = C1 + t²（不含t的常数项）
        C2 = u4**2 + u5**2  # F2² = C2 + (W2 + t)²（不含t的常数项，W2 = W(2)）
        W2 = W[1]
        
        # 由alpha1 = alpha2推导的t的两个可能解（线性方程解）
        sqrtC1 = np.sqrt(C1)
        sqrtC2 = np.sqrt(C2)
        
        # 避免分母为0（物理意义上C1、C2通常非零，因u1,u2等不全为0）
        if abs(sqrtC2 - sqrtC1) > 1e-10:
            t1 = (W2 * sqrtC1) / (sqrtC2 - sqrtC1)  # 解1
        else:
            t1 = np.nan  # 解1无效
        
        t2 = (-W2 * sqrtC1) / (sqrtC2 + sqrtC1)      # 解2（分母恒正，有效）
        
        # 选择合理的t（优先保证F1、F2非负且角度在合理范围，此处选t2，可根据场景调整）
        t_candidates = [t1, t2]
        # 剔除无效解（NaN）
        t_candidates = [t for t in t_candidates if not np.isnan(t)]
        
        # 验证候选t，选择使alpha1=alpha2且F1、F2非负的解
        t_selected = None
        for t_test in t_candidates:
            u3_test = t_test
            u6_test = -W2 - t_test
            
            F1_test = np.sqrt(C1 + u3_test**2)
            F2_test = np.sqrt(C2 + u6_test**2)
            
            # 避免除以0（物理上力不为0）
            if F1_test < 1e-10 or F2_test < 1e-10:
                continue
            
            # 验证sin(theta1)是否等于sin(theta2)（考虑数值误差）
            sin_theta1 = u3_test / F1_test
            sin_theta2 = u6_test / F2_test
            if abs(sin_theta1 - sin_theta2) < 1e-6:
                t_selected = t_test
                break
        
        # 若未找到有效解，默认用t2（工程上通常有效）
        if t_selected is None:
            t_selected = t2
        
        # 步骤3：确定u3和u6
        u3 = t_selected
        u6 = -W2 - u3
        
        # 步骤4：反推uu的7个参数（保证alpha1=alpha2）
        # F3
        F3 = u7
        
        # F1, theta1, alpha1
        F1 = np.sqrt(C1 + u3**2)
        alpha1 = np.arctan2(u1, u2)  # 角度范围[-π, π]，使用arctan2函数
        theta1 = np.arcsin(u3 / F1)   # theta1 = theta2
        
        # F2, theta2, alpha2
        F2 = np.sqrt(C2 + u6**2)
        alpha2 = np.arctan2(u4, u5)  # 角度范围[-π, π]
        theta2 = np.arcsin(u3 / F1)   # theta1 = theta2

        # 组合输出
        uu = np.array([F1, F2, F3, alpha1, alpha2, theta1, theta2])
        return uu
    
    def allocate_actuators(self, f_c_body: np.ndarray, tau_c: np.ndarray, state: dict):
        """分配执行器命令（使用非线性逆映射）"""
        # 构造控制向量W（6×1向量）
        # W = [Fx, Fy, Fz, τx, τy, τz]
        W = np.array([
            f_c_body[0],    # X力
            f_c_body[1],    # Y力（通常为0，但保留）
            f_c_body[2],    # Z力
            tau_c[0],       # 滚转力矩
            tau_c[1],       # 俯仰力矩
            tau_c[2]        # 偏航力矩
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
        
        # 推力限制
        T_max = 50
        F1 = np.clip(F1, 0, T_max)
        F2 = np.clip(F2, 0, T_max)
        F3 = np.clip(F3, -10, 10)
        
        # 角度限制（±85度）
        alpha_max = np.radians(85)
        alpha1 = np.clip(alpha1, -alpha_max, alpha_max)
        alpha2 = np.clip(alpha2, -alpha_max, alpha_max)
        theta_max = np.radians(85)
        theta1 = np.clip(theta1, -theta_max, theta_max)
        theta2 = np.clip(theta2, -theta_max, theta_max)
        
        # 更新状态
        self.T12 = F1
        self.T34 = F2
        self.T5 = F3
        self.alpha1 = alpha1  # roll右倾角
        self.alpha2 = alpha2  # roll左倾角（与alpha1相同）
        self.theta1 = theta1
        self.theta2 = theta2
        
        # 存储控制输入向量（用于日志记录）
        self.u = np.array([F1, F2, F3, alpha1, alpha2, theta2, theta2])
        
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
        """更新控制命令（整合几何控制器）"""
        try:
            # 获取当前状态
            state = self.get_state()

            # 记录状态到日志
            self.log_status(state)
            
            # 计算控制力和力矩
            f_c_body, tau_c = self.compute_control_wrench(state)
            
            # 保存控制量用于日志
            self.f_c_body = f_c_body
            self.tau_c = tau_c
            self.f_c_world = state['rotation_matrix'] @ f_c_body
            
            # 分配执行器命令
            T12, T34, T5, alpha1, alpha2, theta1, theta2 = self.allocate_actuators(f_c_body, tau_c, state)
            
            # 应用控制
            self.set_actuators(T12, T34, T5, alpha1, alpha2, 0, 0)
            
            return True
        except Exception as e:
            print(f"控制更新失败: {e}")
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