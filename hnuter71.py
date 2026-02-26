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
        
        # ========== 俯仰角阈值参数 ==========
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
        self.target_rotation_matrix = np.eye(3)  # 目标旋转矩阵（替代欧拉角）
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

        # ========== 交互式控制参数 ==========
        self.control_mode = 1  # 1:悬停, 2:姿态跟踪, 3:环形轨迹, 4:竖直方形轨迹
        self.trajectory_start_time = 0.0  # 轨迹起始时间
        self.trajectory_duration = 20.0  # 轨迹持续时间（秒）
        self.circle_radius = 1.0  # 环形轨迹半径
        self.square_size = 2.0  # 方形轨迹边长

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
        self.log_file = f'logs/drone_log_interactive_{timestamp}.csv'

        # 写入CSV表头
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
                'control_mode',
                'is_pitch_exceed'
            ])

        print(f"已创建交互式控制日志文件: {self.log_file}")
    
    def log_status(self, state: dict):
        """记录状态到日志文件"""
        timestamp = time.time()
        position = state.get('position', np.zeros(3))
        velocity = state.get('velocity', np.zeros(3))
        angular_velocity = state.get('angular_velocity', np.zeros(3))
        acceleration = state.get('acceleration', np.zeros(3))
        euler = state.get('euler', np.zeros(3))
        current_quat = state.get('quaternion', np.array([1.0, 0.0, 0.0, 0.0]))

        # 从目标旋转矩阵提取四元数
        from scipy.spatial.transform import Rotation as R_scipy
        target_quat_obj = R_scipy.from_matrix(self.target_rotation_matrix)
        target_quat = target_quat_obj.as_quat()  # [x, y, z, w]
        target_euler = target_quat_obj.as_euler('xyz', degrees=False)

        is_pitch_exceed = state.get('is_pitch_exceed', False)
        
        with open(self.log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                timestamp,
                position[0], position[1], position[2],
                self.target_position[0], self.target_position[1], self.target_position[2],
                euler[0], euler[1], euler[2],
                target_euler[0], target_euler[1], target_euler[2],
                current_quat[0], current_quat[1], current_quat[2], current_quat[3],
                target_quat[3], target_quat[0], target_quat[1], target_quat[2],  # [w, x, y, z]
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
                self.control_mode,
                int(is_pitch_exceed)
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
        """计算控制力矩和力（基于几何控制器，使用旋转矩阵）"""
        position = state['position']
        velocity = state['velocity']

        # 位置误差和速度误差
        pos_error = self.target_position - position
        vel_error = self.target_velocity - velocity

        # 期望加速度（PD控制）
        acc_des = self.target_acceleration + self.Kp @ pos_error + self.Dp @ vel_error

        # 世界坐标系下的控制力
        f_c_world = self.mass * (acc_des + np.array([0, 0, self.gravity]))

        # 姿态误差计算（使用旋转矩阵）
        R = state['rotation_matrix']
        angular_velocity = state['angular_velocity']
        R_des = self.target_rotation_matrix
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

    def inverse_nonlinear_mapping(self, W, state):
        """修正后的代数逆映射函数（适配90°大角度）"""
        # W 0 fx 1 fy 2 fz 3 tx 4 ty 5 tz
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
        uu = self.inverse_nonlinear_mapping(W,state)
        
        # 提取参数
        F1 = uu[0]  # 前左组推力
        F2 = uu[1]  # 前右组推力
        F3 = uu[2]  # 尾部推进器推力
        alpha1 = uu[3]  # roll左倾角
        alpha2 = uu[4]  # roll右倾角
        theta1 = uu[5]  # pitch左倾角
        theta2 = uu[6]  # pitch右倾角
        
        # 推力限制（90°大角度下适度提高上限）
        T_max = 60
        F1 = np.clip(F1, 0, T_max)
        F2 = np.clip(F2, 0, T_max)
        F3 = np.clip(F3, -15, 15)
        
        # 角度限制（90°大角度，匹配目标）
        alpha_max = np.radians(200)  # 略大于90°，留有余量
        alpha1 = np.clip(alpha1, -alpha_max, alpha_max)
        alpha2 = np.clip(alpha2, -alpha_max, alpha_max)
        theta_max = np.radians(200)
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
        self.u = np.array([F1, F2, F3, alpha1, alpha2, theta1, theta2])
        
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
        """打印当前状态信息"""
        try:
            state = self.get_state()
            pos = state['position']
            vel = state['velocity']
            accel = state['acceleration']
            euler_deg = np.degrees(state['euler'])

            # 从目标旋转矩阵提取欧拉角用于显示
            from scipy.spatial.transform import Rotation as R_scipy
            target_euler = R_scipy.from_matrix(self.target_rotation_matrix).as_euler('xyz', degrees=True)

            # 控制模式名称
            mode_names = {
                1: "悬停",
                2: "姿态跟踪",
                3: "环形轨迹",
                4: "竖直方形轨迹"
            }
            mode_name = mode_names.get(self.control_mode, "未知模式")

            print(f"\n=== 控制模式: {mode_name} ===")
            print(f"位置: X={pos[0]:.4f}m, Y={pos[1]:.4f}m, Z={pos[2]:.4f}m")
            print(f"目标位置: X={self.target_position[0]:.4f}m, Y={self.target_position[1]:.4f}m, Z={self.target_position[2]:.4f}m")
            print(f"姿态: Roll={euler_deg[0]:.2f}°, Pitch={euler_deg[1]:.2f}°, Yaw={euler_deg[2]:.2f}°")
            print(f"目标姿态: Roll={target_euler[0]:.2f}°, Pitch={target_euler[1]:.2f}°, Yaw={target_euler[2]:.2f}°")
            print(f"控制力矩: X={self.tau_c[0]:.4f}Nm, Y={self.tau_c[1]:.4f}Nm, Z={self.tau_c[2]:.4f}Nm")
            print(f"执行器: T12={self.T12:.2f}N, T34={self.T34:.2f}N, T5={self.T5:.2f}N")
            print(f"倾转角: α1={math.degrees(self.alpha1):.2f}°, α2={math.degrees(self.alpha2):.2f}°, θ1={math.degrees(self.theta1):.2f}°, θ2={math.degrees(self.theta2):.2f}°")
            print("--------------------------------------------------")
        except Exception as e:
            print(f"状态打印失败: {e}")
    
    def update_trajectory(self, current_time: float):
        """
        交互式轨迹更新
        根据control_mode更新目标位置和姿态
        """
        if self.control_mode == 1:
            # 模式1：悬停
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_rotation_matrix = np.eye(3)
            self.target_velocity = np.zeros(3)
            self.target_acceleration = np.zeros(3)

        elif self.control_mode == 2:
            # 模式2：姿态跟踪（保持当前设置的target_rotation_matrix）
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_velocity = np.zeros(3)
            self.target_acceleration = np.zeros(3)

        elif self.control_mode == 3:
            # 模式3：环形轨迹
            t = current_time - self.trajectory_start_time
            omega = 2 * np.pi / self.trajectory_duration  # 角频率

            # 水平圆形轨迹
            x = self.circle_radius * np.cos(omega * t)
            y = self.circle_radius * np.sin(omega * t)
            z = 2.0

            # 速度
            vx = -self.circle_radius * omega * np.sin(omega * t)
            vy = self.circle_radius * omega * np.cos(omega * t)
            vz = 0.0

            # 加速度
            ax = -self.circle_radius * omega**2 * np.cos(omega * t)
            ay = -self.circle_radius * omega**2 * np.sin(omega * t)
            az = 0.0

            self.target_position = np.array([x, y, z])
            self.target_velocity = np.array([vx, vy, vz])
            self.target_acceleration = np.array([ax, ay, az])

            # 姿态：偏航角跟随运动方向
            yaw = omega * t + np.pi / 2
            self.target_rotation_matrix = self._euler_to_rotation_matrix(np.array([0.0, 0.0, yaw]))

        elif self.control_mode == 4:
            # 模式4：竖直方形轨迹
            t = current_time - self.trajectory_start_time
            period = self.trajectory_duration
            segment_time = period / 4  # 每条边的时间

            # 确定当前在哪条边
            t_mod = t % period
            segment = int(t_mod / segment_time)
            t_seg = t_mod - segment * segment_time
            progress = t_seg / segment_time

            half_size = self.square_size / 2

            if segment == 0:
                # 边1：从下到上 (x=0, z: 1.0→3.0)
                x, y = 0.0, 0.0
                z = 1.0 + progress * 2.0
                vx, vy = 0.0, 0.0
                vz = 2.0 / segment_time
                ax, ay, az = 0.0, 0.0, 0.0
            elif segment == 1:
                # 边2：从左到右 (x: 0→2, z=3.0)
                x = progress * 2.0
                y, z = 0.0, 3.0
                vx = 2.0 / segment_time
                vy, vz = 0.0, 0.0
                ax, ay, az = 0.0, 0.0, 0.0
            elif segment == 2:
                # 边3：从上到下 (x=2, z: 3.0→1.0)
                x, y = 2.0, 0.0
                z = 3.0 - progress * 2.0
                vx, vy = 0.0, 0.0
                vz = -2.0 / segment_time
                ax, ay, az = 0.0, 0.0, 0.0
            else:
                # 边4：从右到左 (x: 2→0, z=1.0)
                x = 2.0 - progress * 2.0
                y, z = 0.0, 1.0
                vx = -2.0 / segment_time
                vy, vz = 0.0, 0.0
                ax, ay, az = 0.0, 0.0, 0.0

            self.target_position = np.array([x, y, z])
            self.target_velocity = np.array([vx, vy, vz])
            self.target_acceleration = np.array([ax, ay, az])
            self.target_rotation_matrix = np.eye(3)

        self.target_attitude_rate = np.zeros(3)
        self.target_attitude_acceleration = np.zeros(3)

    def set_attitude_target(self, axis: str, angle_deg: float):
        """
        设置姿态目标（用于模式2）
        axis: 'roll', 'pitch', 或 'yaw'
        angle_deg: 目标角度（度）
        """
        angle_rad = np.radians(angle_deg)

        if axis.lower() == 'roll':
            euler = np.array([angle_rad, 0.0, 0.0])
        elif axis.lower() == 'pitch':
            euler = np.array([0.0, angle_rad, 0.0])
        elif axis.lower() == 'yaw':
            euler = np.array([0.0, 0.0, angle_rad])
        else:
            print(f"未知轴: {axis}")
            return

        self.target_rotation_matrix = self._euler_to_rotation_matrix(euler)
        print(f"姿态目标已设置: {axis} = {angle_deg}°")
    

def main():
    """主函数 - 交互式控制"""
    print("=== 倾转旋翼无人机交互式控制仿真 ===")
    print("控制模式：")
    print("  1 - 悬停")
    print("  2 - 姿态跟踪")
    print("  3 - 环形轨迹")
    print("  4 - 竖直方形轨迹")
    print("  q - 退出")

    try:
        # 初始化控制器
        controller = HnuterController("hnuter201.xml")

        # 初始目标
        controller.target_position = np.array([0.0, 0.0, 2.0])
        controller.target_rotation_matrix = np.eye(3)

        # 启动 Viewer
        with viewer.launch_passive(controller.model, controller.data) as v:
            print("\n仿真启动：")
            print(f"日志文件路径: {controller.log_file}")
            print("按数字键切换模式，按 q 退出")

            start_time = time.time()
            last_print_time = 0
            print_interval = 1.0
            count = 0

            # 非阻塞输入设置
            import sys
            import select
            import termios
            import tty

            # 保存终端设置
            old_settings = termios.tcgetattr(sys.stdin)

            try:
                # 设置为非阻塞模式
                tty.setcbreak(sys.stdin.fileno())

                while v.is_running():
                    current_time = time.time() - start_time

                    # 检查用户输入（非阻塞）
                    if select.select([sys.stdin], [], [], 0)[0]:
                        key = sys.stdin.read(1)

                        if key == 'q':
                            print("\n退出仿真...")
                            break
                        elif key == '1':
                            controller.control_mode = 1
                            print("\n切换到模式1: 悬停")
                        elif key == '2':
                            controller.control_mode = 2
                            print("\n切换到模式2: 姿态跟踪")
                            print("请输入轴 (roll/pitch/yaw): ", end='', flush=True)

                            # 恢复终端设置以读取输入
                            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                            axis = input().strip()

                            print("请输入角度 (度): ", end='', flush=True)
                            try:
                                angle = float(input().strip())
                                controller.set_attitude_target(axis, angle)
                                controller.trajectory_start_time = current_time
                            except ValueError:
                                print("无效的角度输入")

                            # 重新设置为非阻塞模式
                            tty.setcbreak(sys.stdin.fileno())

                        elif key == '3':
                            controller.control_mode = 3
                            controller.trajectory_start_time = current_time
                            print("\n切换到模式3: 环形轨迹")
                        elif key == '4':
                            controller.control_mode = 4
                            controller.trajectory_start_time = current_time
                            print("\n切换到模式4: 竖直方形轨迹")

                    # 更新轨迹
                    controller.update_trajectory(current_time)

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
            finally:
                # 恢复终端设置
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

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
