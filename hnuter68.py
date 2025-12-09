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
        
        # 几何控制器增益（适配45°姿态限制）
        self.Kp = np.diag([6, 6, 6])  # 位置增益适度提高
        self.Dp = np.diag([5, 5, 5])  # 速度阻尼
        self.KR = np.array([3, 0.8, 0.3])   # 姿态增益适度提高，增强大角度跟踪
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

        # ========== 核心修改：限制姿态角≤±45° ==========
        self.trajectory_phase = 0  # 轨迹阶段
        self.phase_start_time = 0.0  # 各阶段起始时间
        self.max_attitude_angle = np.pi/4  # 最大姿态角（45°，核心限制）
        
        # 复杂轨迹参数配置（姿态幅值≤45°）
        self.trajectory_params = {
            # 圆形轨迹参数
            "circle_radius": 2.0,          # 圆半径(m)
            "circle_omega": 0.3,           # 圆周运动角速度(rad/s)
            "circle_z": 2.0,               # 圆形轨迹高度(m)
            
            # 8字形轨迹参数 (李萨如曲线)
            "figure8_a": 1.5,              # 8字x轴幅值
            "figure8_b": 1.0,              # 8字y轴幅值
            "figure8_omega": 0.25,         # 8字运动角速度
            "figure8_z_base": 2.5,         # 8字基础高度
            "figure8_z_amp": 0.5,          # 8字高度波动幅值
            
            # 螺旋轨迹参数
            "spiral_radius_start": 1.0,    # 螺旋起始半径
            "spiral_radius_end": 3.0,      # 螺旋终止半径
            "spiral_omega": 0.4,           # 螺旋角速度
            "spiral_z_start": 2.0,         # 螺旋起始高度
            "spiral_z_end": 3.0,           # 螺旋终止高度
            "spiral_duration": 20.0,       # 螺旋运动时长
            
            # 姿态随轨迹变化参数（所有轴≤45°）
            "attitude_amp_roll": np.pi/6,  # 滚转最大幅值(30°，≤45°)
            "attitude_amp_pitch": np.pi/8, # 俯仰最大幅值(22.5°，≤45°)
            "attitude_amp_yaw": np.pi/4,   # 偏航最大幅值(45°，上限)
            "attitude_omega": 0.5,         # 姿态变化角速度
        }

        print("倾转旋翼控制器初始化完成（姿态角限制≤±45°）")
    
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
        self.log_file = f'logs/drone_log_complex_trajectory_45deg_{timestamp}.csv'  # 标注45°限制
        
        # 写入CSV表头（新增轨迹阶段和轨迹类型）
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
                'trajectory_phase', 'trajectory_type'
            ])
        
        print(f"已创建45°限制复杂轨迹日志文件: {self.log_file}")
    
    def log_status(self, state: dict):
        """记录状态到日志文件（新增轨迹类型）"""
        timestamp = time.time()
        position = state.get('position', np.zeros(3))
        velocity = state.get('velocity', np.zeros(3))
        angular_velocity = state.get('angular_velocity', np.zeros(3))
        acceleration = state.get('acceleration', np.zeros(3))
        euler = state.get('euler', np.zeros(3))
        current_quat = state.get('quaternion', np.array([1.0, 0.0, 0.0, 0.0]))
        target_quat = self._euler_to_quaternion(self.target_attitude)
        
        # 轨迹类型映射
        phase_to_type = {
            0: "takeoff_hover",
            1: "circular_trajectory",
            2: "figure8_trajectory",
            3: "spiral_trajectory",
            4: "return_hover",
            5: "complex_attitude_maneuver",
            6: "final_hover"
        }
        trajectory_type = phase_to_type.get(self.trajectory_phase, "unknown")
        
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
                self.trajectory_phase, trajectory_type
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
            body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'drone')
            if body_id != -1:
                state['position'] = self.data.xpos[body_id].copy()
                state['quaternion'] = self.data.xquat[body_id].copy()
                state['velocity'] = self.data.cvel[body_id][3:6].copy()
                state['angular_velocity'] = self.data.cvel[body_id][0:3].copy()
            
            state['rotation_matrix'] = self._quat_to_rotation_matrix(state['quaternion'])
            state['euler'] = self._quat_to_euler(state['quaternion'])
            
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

    def inverse_nonlinear_mapping(self, W):
        """修正后的代数逆映射函数（适配45°姿态限制）"""
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
        
        # 计算推力和角度（增加数值稳定性）
        F1 = np.sqrt(u1**2 + u2**2 + u3**2)
        F2 = np.sqrt(u4**2 + u5**2 + u6**2)
        F3 = u7
        
        # 防止除零保护
        eps = 1e-8
        F1_safe = F1 if F1 > eps else eps
        F2_safe = F2 if F2 > eps else eps

        # 求解倾转角度
        alpha1 = np.arctan2(u1, u2)  
        alpha2 = np.arctan2(u4, u5)
        
        val1 = np.clip(u3 / F1_safe, -1.0 + eps, 1.0 - eps)
        val2 = np.clip(u6 / F2_safe, -1.0 + eps, 1.0 - eps)
        
        theta1 = np.arcsin(val1)
        theta2 = np.arcsin(val2)
        
        return np.array([F1, F2, F3, alpha1, alpha2, theta1, theta2])

    def allocate_actuators(self, f_c_body: np.ndarray, tau_c: np.ndarray, state: dict):
        """分配执行器命令（使用非线性逆映射）"""
        # 构造控制向量W
        W = np.array([
            f_c_body[0],    # X力
            f_c_body[1],    # Y力
            f_c_body[2],    # Z力
            tau_c[0],       # 滚转力矩
            tau_c[1],       # 俯仰力矩
            tau_c[2]        # 偏航力矩
        ])
        
        # 非线性逆映射
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
        T_max = 80
        F1 = np.clip(F1, 0, T_max)
        F2 = np.clip(F2, 0, T_max)
        F3 = np.clip(F3, -20, 20)
        
        # 角度限制（适配45°姿态限制，倾转角度也适度减小）
        alpha_max = np.radians(60)  # 倾转角度≤60°（配合姿态限制）
        alpha1 = np.clip(alpha1, -alpha_max, alpha_max)
        alpha2 = np.clip(alpha2, -alpha_max, alpha_max)
        theta_max = np.radians(60)
        theta1 = np.clip(theta1, -theta_max, theta_max)
        theta2 = np.clip(theta2, -theta_max, theta_max)
        
        # 更新状态
        self.last_alpha1 = alpha1
        self.last_alpha2 = alpha2
        self.last_theta1 = theta1
        self.last_theta2 = theta2
        
        self.T12 = F1
        self.T34 = F2
        self.T5 = F3
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.theta1 = theta1
        self.theta2 = theta2
        
        # 存储控制输入向量
        self.u = np.array([F1, F2, F3, alpha1, alpha2, theta2, theta2])
        
        return F1, F2, F3, alpha1, alpha2, theta1, theta2
    
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
        """打印当前状态信息（含45°限制标注）"""
        try:
            state = self.get_state()
            pos = state['position']
            vel = state['velocity']
            accel = state['acceleration']
            euler_deg = np.degrees(state['euler'])
            target_euler_deg = np.degrees(self.target_attitude)
            current_quat = state['quaternion']
            target_quat = self._euler_to_quaternion(self.target_attitude)
            
            # 阶段名称映射（复杂轨迹+45°限制）
            phase_names = {
                0: "起飞悬停",
                1: "圆形轨迹 (半径2m，高度2m)",
                2: "8字形轨迹 (李萨如曲线，高度2.0-3.0m)",
                3: "螺旋上升轨迹 (半径1-3m，高度2-4m)",
                4: "返回原点",
                5: "复杂姿态机动 (≤45°)",
                6: "最终悬停"
            }
            phase_name = phase_names.get(self.trajectory_phase, "未知阶段")
            
            print(f"\n=== 轨迹阶段: {self.trajectory_phase} ({phase_name}) ===")
            print(f"位置: X={pos[0]:.8f}m, Y={pos[1]:.8f}m, Z={pos[2]:.8f}m")
            print(f"目标位置: X={self.target_position[0]:.8f}m, Y={self.target_position[1]:.8f}m, Z={self.target_position[2]:.8f}m")
            print(f"姿态: Roll={euler_deg[0]:.2f}°, Pitch={euler_deg[1]:.2f}°, Yaw={euler_deg[2]:.2f}° (限制≤±45°)")  
            print(f"控制力矩: X={self.tau_c[0]:.2f}Nm, Y={self.tau_c[1]:.2f}Nm, Z={self.tau_c[2]:.2f}Nm")
            print(f"目标姿态: Roll={target_euler_deg[0]:.1f}°, Pitch={target_euler_deg[1]:.1f}°, Yaw={target_euler_deg[2]:.1f}°") 
            print(f"角速度: Roll={np.degrees(state['angular_velocity'][0]):.2f}°/s, Pitch={np.degrees(state['angular_velocity'][1]):.2f}°/s, Yaw={np.degrees(state['angular_velocity'][2]):.2f}°/s")
            print(f"执行器状态: T12={self.T12:.2f}N, T34={self.T34:.2f}N, T5={self.T5:.2f}N, α1={math.degrees(self.alpha1):.2f}°, α2={math.degrees(self.alpha2):.2f}°, θ1={math.degrees(self.theta1):.2f}°, θ2={math.degrees(self.theta2):.2f}°")
            print("--------------------------------------------------")
        except Exception as e:
            print(f"状态打印失败: {e}")
    
    def update_trajectory(self, current_time: float):
        """
        核心修改：复杂轨迹生成器（姿态角≤±45°）
        阶段划分（总时长~120秒）：
        0: 0~8s    - 起飞悬停（升到2m高度，稳定）
        1: 8~38s   - 圆形轨迹（半径2m，高度2m，姿态≤30°）
        2: 38~68s  - 8字形轨迹（李萨如曲线，高度2.0-3.0m，姿态≤30°）
        3: 68~98s  - 螺旋上升轨迹（半径1→3m，高度2→4m，姿态≤30°）
        4: 98~108s - 返回原点（10秒）
        5: 108~118s- 复杂姿态机动（Roll≤45°/Pitch≤30°/Yaw≤45°）
        6: 118s~   - 最终悬停
        """
        # 初始化阶段起始时间
        if self.trajectory_phase == 0 and self.phase_start_time == 0.0:
            self.phase_start_time = current_time
        
        # 阶段时长配置
        phase_durations = {
            0: 8.0,     # 起飞悬停
            1: 30.0,    # 圆形轨迹
            2: 30.0,    # 8字形轨迹
            3: 30.0,    # 螺旋轨迹
            4: 10.0,    # 返回原点
            5: 10.0,    # 复杂姿态机动
            6: float('inf')  # 最终悬停
        }
        
        # 计算当前阶段已运行时间
        phase_elapsed = current_time - self.phase_start_time
        
        # 阶段切换判断
        if phase_elapsed > phase_durations[self.trajectory_phase]:
            self.trajectory_phase += 1
            self.phase_start_time = current_time  # 重置阶段起始时间
            print(f"\n🔄 轨迹阶段切换: {self.trajectory_phase-1} → {self.trajectory_phase}")
        
        # ========== 各阶段复杂轨迹生成（姿态≤45°） ==========
        if self.trajectory_phase == 0:
            # 阶段0：起飞悬停
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([0.0, 0.0, 0.0])
            self.target_velocity = np.zeros(3)
            self.target_acceleration = np.zeros(3)
            
        elif self.trajectory_phase == 1:
            # 阶段1：圆形轨迹 (x=Rcosωt, y=Rsinωt, z=恒定)
            t = phase_elapsed
            R = self.trajectory_params["circle_radius"]
            omega = self.trajectory_params["circle_omega"]
            z = self.trajectory_params["circle_z"]
            
            # 位置
            x = R * np.cos(omega * t)
            y = R * np.sin(omega * t)
            self.target_position = np.array([x, y, z])
            
            # 速度（一阶导数）
            vx = -R * omega * np.sin(omega * t)
            vy = R * omega * np.cos(omega * t)
            vz = 0.0
            self.target_velocity = np.array([vx, vy, vz])
            
            # 加速度（二阶导数）
            ax = -R * omega**2 * np.cos(omega * t)
            ay = -R * omega**2 * np.sin(omega * t)
            az = 0.0
            self.target_acceleration = np.array([ax, ay, az])
            
            # 姿态随轨迹变化（≤30°，Yaw跟踪切线方向）
            yaw = np.arctan2(vy, vx)  # 航向跟踪速度方向（自然≤45°）
            roll = self.trajectory_params["attitude_amp_roll"] * np.sin(2 * omega * t)  # 30°
            pitch = self.trajectory_params["attitude_amp_pitch"] * np.cos(2 * omega * t)  # 22.5°
            # 最终clip到±45°（双重保险）
            roll = np.clip(roll, -self.max_attitude_angle, self.max_attitude_angle)
            pitch = np.clip(pitch, -self.max_attitude_angle, self.max_attitude_angle)
            yaw = np.clip(yaw, -self.max_attitude_angle, self.max_attitude_angle)
            self.target_attitude = np.array([roll, pitch, yaw])
            
            # 姿态角速度
            self.target_attitude_rate = np.array([
                2 * omega * self.trajectory_params["attitude_amp_roll"] * np.cos(2 * omega * t),
                -2 * omega * self.trajectory_params["attitude_amp_pitch"] * np.sin(2 * omega * t),
                omega  # yaw角速度等于圆周运动角速度
            ])
            
        elif self.trajectory_phase == 2:
            # 阶段2：8字形轨迹 (李萨如曲线: x=Asinωt, y=Bsin2ωt, z=z0+A_z*sinωt)
            t = phase_elapsed
            A = self.trajectory_params["figure8_a"]
            B = self.trajectory_params["figure8_b"]
            omega = self.trajectory_params["figure8_omega"]
            z_base = self.trajectory_params["figure8_z_base"]
            z_amp = self.trajectory_params["figure8_z_amp"]
            
            # 位置（李萨如曲线）
            x = A * np.sin(omega * t)
            y = B * np.sin(2 * omega * t)
            z = z_base + z_amp * np.sin(omega * t)
            self.target_position = np.array([x, y, z])
            
            # 速度
            vx = A * omega * np.cos(omega * t)
            vy = 2 * B * omega * np.cos(2 * omega * t)
            vz = z_amp * omega * np.cos(omega * t)
            self.target_velocity = np.array([vx, vy, vz])
            
            # 加速度
            ax = -A * omega**2 * np.sin(omega * t)
            ay = -4 * B * omega**2 * np.sin(2 * omega * t)
            az = -z_amp * omega**2 * np.sin(omega * t)
            self.target_acceleration = np.array([ax, ay, az])
            
            # 姿态随轨迹变化（≤30°）
            yaw = np.arctan2(vy, vx) if np.sqrt(vx**2 + vy**2) > 0.01 else 0.0
            roll = self.trajectory_params["attitude_amp_roll"] * np.sin(3 * omega * t)  # 30°
            pitch = self.trajectory_params["attitude_amp_pitch"] * np.sin(omega * t)  # 22.5°
            # clip到±45°
            roll = np.clip(roll, -self.max_attitude_angle, self.max_attitude_angle)
            pitch = np.clip(pitch, -self.max_attitude_angle, self.max_attitude_angle)
            yaw = np.clip(yaw, -self.max_attitude_angle, self.max_attitude_angle)
            self.target_attitude = np.array([roll, pitch, yaw])
            
        elif self.trajectory_phase == 3:
            # 阶段3：螺旋上升轨迹 (半径线性增加，高度线性增加，圆周运动)
            t = phase_elapsed
            t_total = self.trajectory_params["spiral_duration"]
            r_start = self.trajectory_params["spiral_radius_start"]
            r_end = self.trajectory_params["spiral_radius_end"]
            omega = self.trajectory_params["spiral_omega"]
            z_start = self.trajectory_params["spiral_z_start"]
            z_end = self.trajectory_params["spiral_z_end"]
            
            # 半径线性插值
            r = r_start + (r_end - r_start) * (t / t_total)
            # 高度线性插值
            z = z_start + (z_end - z_start) * (t / t_total)
            
            # 位置
            x = r * np.cos(omega * t)
            y = r * np.sin(omega * t)
            self.target_position = np.array([x, y, z])
            
            # 速度（包含半径变化和高度变化的贡献）
            dr_dt = (r_end - r_start) / t_total
            dz_dt = (z_end - z_start) / t_total
            vx = dr_dt * np.cos(omega * t) - r * omega * np.sin(omega * t)
            vy = dr_dt * np.sin(omega * t) + r * omega * np.cos(omega * t)
            vz = dz_dt
            self.target_velocity = np.array([vx, vy, vz])
            
            # 加速度
            d2r_dt2 = 0.0  # 匀加速半径变化
            ax = d2r_dt2 * np.cos(omega * t) - 2 * dr_dt * omega * np.sin(omega * t) - r * omega**2 * np.cos(omega * t)
            ay = d2r_dt2 * np.sin(omega * t) + 2 * dr_dt * omega * np.cos(omega * t) - r * omega**2 * np.sin(omega * t)
            az = 0.0
            self.target_acceleration = np.array([ax, ay, az])
            
            # 姿态随螺旋变化（≤30°）
            yaw = np.arctan2(vy, vx) if np.sqrt(vx**2 + vy**2) > 0.01 else 0.0
            roll = 0.0  # 0→30°
            pitch = self.trajectory_params["attitude_amp_pitch"] * (r - r_start) / (r_end - r_start)  # 0→22.5°
            # clip到±45°
            roll = np.clip(roll, -self.max_attitude_angle, self.max_attitude_angle)
            pitch = np.clip(pitch, -self.max_attitude_angle, self.max_attitude_angle)
            yaw = np.clip(yaw, -self.max_attitude_angle, self.max_attitude_angle)
            self.target_attitude = np.array([roll, pitch, yaw])
            
        elif self.trajectory_phase == 4:
            # 阶段4：返回原点（线性插值）
            t = phase_elapsed
            t_total = phase_durations[4]
            progress = np.clip(t / t_total, 0.0, 1.0)
            
            # 当前位置到原点的插值
            current_target_pos = self.target_position
            target_pos = current_target_pos * (1 - progress)
            self.target_position = target_pos
            
            # 速度和加速度线性减小到零
            self.target_velocity = self.target_velocity * (1 - progress)
            self.target_acceleration = np.zeros(3)
            
            # 姿态归零
            self.target_attitude = self.target_attitude * (1 - progress)
            self.target_attitude_rate = np.zeros(3)
            
        elif self.trajectory_phase == 5:
            # 阶段5：复杂姿态机动（严格≤45°）
            t = phase_elapsed
            omega = np.pi / phase_durations[5]  # 10秒完成一个周期
            
            # Roll最大45°，Pitch最大30°，Yaw最大45°
            roll = self.trajectory_params["attitude_amp_yaw"] * np.sin(omega * t)  # 45°
            pitch = self.trajectory_params["attitude_amp_roll"] * np.cos(omega * t)  # 30°
            yaw = self.trajectory_params["attitude_amp_yaw"] * (2 * (t / phase_durations[5]) - 1)  # -45°到+45°
            
            # 最终clip到±45°（三重保险）
            roll = np.clip(roll, -self.max_attitude_angle, self.max_attitude_angle)
            pitch = np.clip(pitch, -self.max_attitude_angle, self.max_attitude_angle)
            yaw = np.clip(yaw, -self.max_attitude_angle, self.max_attitude_angle)
            
            self.target_position = np.array([0.0, 0.0, 2.0])  # 保持位置
            self.target_attitude = np.array([roll, pitch, yaw])
            self.target_velocity = np.zeros(3)
            self.target_acceleration = np.zeros(3)
            
            # 姿态角速度
            self.target_attitude_rate = np.array([
                self.trajectory_params["attitude_amp_yaw"] * omega * np.cos(omega * t),
                -self.trajectory_params["attitude_amp_roll"] * omega * np.sin(omega * t),
                self.trajectory_params["attitude_amp_yaw"] * 2 / phase_durations[5]
            ])
            
        else:
            # 阶段6：最终悬停
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([0.0, 0.0, 0.0])
            self.target_velocity = np.zeros(3)
            self.target_acceleration = np.zeros(3)
            self.target_attitude_rate = np.zeros(3)
            self.target_attitude_acceleration = np.zeros(3)


def main():
    """主函数 - 启动45°限制复杂轨迹跟踪仿真"""
    print("=== 倾转旋翼无人机复杂轨迹跟踪仿真（姿态≤±45°）===")
    print("轨迹规划：起飞悬停→圆形轨迹→8字形轨迹→螺旋上升→返回原点→复杂姿态机动→悬停")
    print("核心特性：姿态角严格限制≤±45°，位置轨迹复杂，平滑过渡")
    
    try:
        # 初始化控制器
        controller = HnuterController("hnuter201.xml")
        
        # 设置目标轨迹（简单悬停）
        controller.target_position = np.array([0.0, 0.0, 2.0])  # 目标高度1.5米
        controller.target_velocity = np.zeros(3)
        controller.target_acceleration = np.zeros(3)
        controller.target_attitude = np.array([0.8, 0.0, 0.0])  # 水平姿态
        
        controller.target_attitude_rate = np.zeros(3)
        controller.target_attitude_acceleration = np.zeros(3)

        # 启动 Viewer
        with viewer.launch_passive(controller.model, controller.data) as v:
            print("\n仿真启动：")
            print(f"45°限制日志文件路径: {controller.log_file}")
            print("按 Ctrl+C 终止仿真")
            
            start_time = time.time()
            last_print_time = 0
            print_interval = 1.0
            count = 0
            
            try:
                while v.is_running():
                    current_time = time.time() - start_time
                    
                    # 更新复杂轨迹（核心）
                    # controller.update_trajectory(current_time)
                    
                    # 更新控制
                    controller.update_control()

                    count += 1
                    if count % 1 == 0:
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
