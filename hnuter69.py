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
from scipy.spatial.transform import Rotation as R

class HnuterController:
    def __init__(self, model_path: str = "hnuter201.xml"):
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
        
        # ========== 核心修改：水平状态时俯仰角为90度的旋转矩阵 ==========
        # 定义从标准坐标系到新坐标系的旋转矩阵
        # 当无人机水平放置时，我们希望俯仰角为90度
        self.R_horizontal = self._get_horizontal_rotation_matrix()
        
        # 几何自适应控制器增益 [3](@ref)
        self.Kp = np.diag([8, 8, 8])  # 位置增益
        self.Dp = np.diag([6, 6, 6])  # 速度阻尼
        self.KR = np.array([4, 1.5, 0.4])   # 姿态增益
        self.Domega = np.array([1.0, 0.5, 0.7])  # 角速度阻尼
        
        # 滑模控制参数 [4](@ref)
        self.lambda_smc = np.array([2.0, 2.0, 2.0])  # 滑模面参数
        self.epsilon = np.array([0.6, 0.6, 0.6])     # 趋近律参数
        self.k_smc = np.array([1.8, 1.8, 1.8])       # 滑模增益
        
        # 扩张状态观测器参数 [4](@ref)
        self.eso_beta = np.array([100, 300, 500])  # ESO增益
        self.disturbance_estimate = np.zeros(3)     # 扰动估计
        self.observer_state = np.zeros(6)  # 补充ESO状态初始化（原代码缺失）

        # ========== 新增：俯仰角限制参数 ==========
        self.pitch_threshold_deg = 70.0  # 俯仰角阈值（度）
        self.pitch_threshold_rad = np.radians(self.pitch_threshold_deg)  # 转换为弧度
        self.is_pitch_exceed = False  # 标记是否超过俯仰角阈值
        self._pitch_warned = False  # 避免重复打印警告

        # 控制量
        self.f_c_body = np.zeros(3)  # 机体坐标系下的控制力
        self.f_c_world = np.zeros(3)  # 世界坐标系下的控制力
        self.tau_c = np.zeros(3)     # 控制力矩
        self.u = np.zeros(7)         # 控制输入向量

        # 分配矩阵
        self.A = np.array([
            [1, 0,  0, 1, 0,  0, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, self.l1, 0, 0, -self.l1, 0, 0],
            [0, 0, 0, 0, 0, 0, self.l2],
            [-self.l1, 0, 0, self.l1, 0, 0, 0]
        ])
        
        self.A_pinv = np.linalg.pinv(self.A)

        # 目标状态
        self.target_position = np.array([0.0, 0.0, 2.0])  # 初始目标高度
        self.target_velocity = np.array([0.0, 0.0, 0.0])
        self.target_acceleration = np.array([0.0, 0.0, 0.0])
        self.target_attitude = np.array([0.0, np.pi/2, 0.0])  # 水平状态时俯仰角为90度
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
        
        # 角度连续性处理
        self.last_alpha1 = 0
        self.last_alpha2 = 0
        self.last_theta1 = 0
        self.last_theta2 = 0

        # 执行器名称映射
        self._get_actuator_ids()
        self._get_sensor_ids()
        
        # 创建日志文件
        self._create_log_file()

        # 轨迹控制参数
        self.trajectory_phase = 0
        self.attitude_target_rad = np.pi/2  # 90度目标
        self.phase_start_time = 0.0
        self.attitude_tolerance = 0.08

        print("倾转旋翼控制器初始化完成（水平状态俯仰角90度）")

    def _get_horizontal_rotation_matrix(self):
        """获取水平状态为俯仰90度的旋转矩阵 [4](@ref)"""
        # 绕Y轴旋转-90度，使水平状态时俯仰角为90度
        pitch_correction = -np.pi / 2
        cos_pitch = np.cos(pitch_correction)
        sin_pitch = np.sin(pitch_correction)
        
        R_y = np.array([
            [cos_pitch, 0, sin_pitch],
            [0, 1, 0],
            [-sin_pitch, 0, cos_pitch]
        ])
        
        return R_y

    def apply_horizontal_rotation(self, state: dict) -> dict:
        """应用水平旋转变换到状态量"""
        # 变换旋转矩阵
        state['rotation_matrix'] = state['rotation_matrix'] @ self.R_horizontal.T
        
        # 变换欧拉角：在新坐标系中，水平状态俯仰角为0度
        if 'euler' in state:
            # 将原始俯仰角减去90度
            state['euler'][1] -= np.pi / 2
            
            # 角度规范化到[-π, π]
            state['euler'][1] = (state['euler'][1] + np.pi) % (2 * np.pi) - np.pi
        
        return state

    def _print_model_diagnostics(self):
        """打印模型诊断信息"""
        print("\n=== 模型诊断信息 ===")
        print(f"广义坐标数量 (nq): {self.model.nq}")
        print(f"速度自由度 (nv): {self.model.nv}")
        print(f"执行器数量 (nu): {self.model.nu}")
        
        # 检查身体信息
        print("\n=== 身体列表 ===")
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            print(f"身体 {i}: {name}")

    def _create_log_file(self):
        """创建日志文件"""
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f'logs/drone_log_horizontal_{timestamp}.csv'
        
        with open(self.log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'timestamp', 'pos_x', 'pos_y', 'pos_z', 
                'target_x', 'target_y', 'target_z',
                'roll', 'pitch', 'yaw',
                'target_roll', 'target_pitch', 'target_yaw',
                'vel_x', 'vel_y', 'vel_z',
                'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
                'f_body_x', 'f_body_y', 'f_body_z',
                'tau_x', 'tau_y', 'tau_z',
                'T12', 'T34', 'T5',
                'alpha1', 'alpha2', 'theta1', 'theta2',
                'trajectory_phase', 'disturbance_est_x', 'disturbance_est_y', 'disturbance_est_z',
                'is_pitch_exceed'  # 新增：俯仰角超限标记
            ])

    def log_status(self, state: dict):
        """记录状态到日志文件"""
        timestamp = time.time()
        position = state.get('position', np.zeros(3))
        velocity = state.get('velocity', np.zeros(3))
        angular_velocity = state.get('angular_velocity', np.zeros(3))
        euler = state.get('euler', np.zeros(3))
        is_pitch_exceed = state.get('is_pitch_exceed', False)
        
        with open(self.log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                timestamp,
                position[0], position[1], position[2],
                self.target_position[0], self.target_position[1], self.target_position[2],
                euler[0], euler[1], euler[2],
                self.target_attitude[0], self.target_attitude[1], self.target_attitude[2],
                velocity[0], velocity[1], velocity[2],
                angular_velocity[0], angular_velocity[1], angular_velocity[2],
                self.f_c_body[0], self.f_c_body[1], self.f_c_body[2],
                self.tau_c[0], self.tau_c[1], self.tau_c[2],
                self.T12, self.T34, self.T5,
                self.alpha1, self.alpha2, self.theta1, self.theta2,
                self.trajectory_phase,
                self.disturbance_estimate[0], self.disturbance_estimate[1], self.disturbance_estimate[2],
                int(is_pitch_exceed)  # 记录是否超限（0/1）
            ])

    def _get_actuator_ids(self):
        """获取执行器ID"""
        self.actuator_ids = {}
        
        actuator_names = [
            'tilt_pitch_right', 'tilt_pitch_left',
            'tilt_roll_right', 'tilt_roll_left',
            'motor_r_upper', 'motor_r_lower', 
            'motor_l_upper', 'motor_l_lower', 
            'motor_rear_upper'
        ]
        
        for name in actuator_names:
            act_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id != -1:
                self.actuator_ids[name] = act_id
            else:
                print(f"警告: 未找到执行器 {name}")

    def _get_sensor_ids(self):
        """获取传感器ID"""
        self.sensor_ids = {}
        
        sensor_names = [
            'drone_pos', 'drone_quat', 'body_vel', 'body_gyro'
        ]
        
        for name in sensor_names:
            sensor_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, name)
            if sensor_id != -1:
                self.sensor_ids[name] = sensor_id

    def get_state(self) -> dict:
        """获取无人机当前状态（应用水平旋转变换）"""
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
            
            # 计算原始旋转矩阵和欧拉角
            state['rotation_matrix'] = self._quat_to_rotation_matrix(state['quaternion'])
            state['euler'] = self._quat_to_euler(state['quaternion'])
            
            # 应用水平旋转变换
            state = self.apply_horizontal_rotation(state)
            
            # ========== 核心修改：判断俯仰角是否超限 ==========
            self.is_pitch_exceed = abs(state['euler'][1]) > self.pitch_threshold_rad
            state['is_pitch_exceed'] = self.is_pitch_exceed
            
            # 打印超限警告（仅首次超限/恢复时）
            if self.is_pitch_exceed and not self._pitch_warned:
                pitch_deg = np.degrees(state['euler'][1])
                print(f"\n⚠️ 警告：俯仰角 {pitch_deg:.1f}° 超过 {self.pitch_threshold_deg}°，禁用横滚/偏航控制！")
                self._pitch_warned = True
            elif not self.is_pitch_exceed and self._pitch_warned:
                pitch_deg = np.degrees(state['euler'][1])
                print(f"\n✅ 恢复：俯仰角 {pitch_deg:.1f}° 低于 {self.pitch_threshold_deg}°，恢复横滚/偏航控制！")
                self._pitch_warned = False
            
            if np.any(np.isnan(state['position'])):
                state['position'] = np.zeros(3)
                
            return state
        except Exception as e:
            print(f"状态获取错误: {e}")
            return state

    def _quat_to_rotation_matrix(self, quat: np.ndarray) -> np.ndarray:
        """四元数转旋转矩阵"""
        return R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()

    def _quat_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """四元数转欧拉角 (roll, pitch, yaw)"""
        # 使用scipy的Rotation类避免万向锁问题
        rotation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        euler = rotation.as_euler('xyz', degrees=False)
        return euler

    def _euler_to_quaternion(self, euler: np.ndarray) -> np.ndarray:
        """欧拉角转四元数 [w, x, y, z]"""
        rotation = R.from_euler('xyz', euler)
        quat = rotation.as_quat()
        return np.array([quat[3], quat[0], quat[1], quat[2]])  # w, x, y, z

    def _euler_to_rotation_matrix(self, euler: np.ndarray) -> np.ndarray:
        """欧拉角转旋转矩阵"""
        return R.from_euler('xyz', euler).as_matrix()

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

    def expansion_state_observer(self, state: dict, control_input: np.ndarray):
        """扩张状态观测器估计扰动 [4](@ref)"""
        # 简化的ESO实现
        angular_velocity = state['angular_velocity']
        euler_angles = state['euler']
        
        # 系统动力学模型
        omega_dot = np.linalg.inv(self.J) @ (control_input - np.cross(angular_velocity, self.J @ angular_velocity))
        
        # 观测器更新
        estimation_error = angular_velocity - self.observer_state[:3]
        self.observer_state[:3] += self.dt * (omega_dot + self.eso_beta[0] * estimation_error)
        self.observer_state[3:] += self.dt * (self.eso_beta[1] * estimation_error)
        
        # 扰动估计
        self.disturbance_estimate = self.observer_state[3:]
        
        return self.disturbance_estimate

    def compute_sliding_mode_control(self, state: dict, target: dict) -> np.ndarray:
        """分数阶滑模控制 [4](@ref)"""
        # 姿态误差
        e_R = target['attitude'] - state['euler']
        e_omega = target['attitude_rate'] - state['angular_velocity']
        
        # 滑模面设计
        s = e_omega + self.lambda_smc * e_R
        
        # ========== 核心修改：俯仰角超限时禁用横滚/偏航滑模控制 ==========
        if state['is_pitch_exceed']:
            s[0] = 0.0  # 横滚滑模面置零
            s[2] = 0.0  # 偏航滑模面置零
        
        # 快速趋近律
        sign_s = np.sign(s)
        alpha = 0.8
        reaching_law = -self.epsilon * (np.abs(s)**alpha * sign_s + self.k_smc * s)
        
        # 控制力矩
        inertia_term = self.J @ target['attitude_acceleration']
        coriolis_term = np.cross(state['angular_velocity'], self.J @ state['angular_velocity'])
        disturbance_compensation = self.disturbance_estimate
        
        tau = inertia_term + coriolis_term + disturbance_compensation + reaching_law
        
        # ========== 核心修改：俯仰角超限时置零横滚/偏航控制力矩 ==========
        if state['is_pitch_exceed']:
            tau[0] = 0.0  # 横滚力矩置零
            tau[2] = 0.0  # 偏航力矩置零
        
        return tau

    def compute_geometric_adaptive_control(self, state: dict, target: dict) -> Tuple[np.ndarray, np.ndarray]:
        """几何自适应控制 [3](@ref)"""
        position = state['position']
        velocity = state['velocity']
        R = state['rotation_matrix']
        
        # 位置误差
        pos_error = target['position'] - position
        vel_error = target['velocity'] - velocity
        
        # ========== 核心修改：俯仰角超限时仅保留Z轴位置控制 ==========
        if state['is_pitch_exceed']:
            pos_error[0] = 0.0  # X轴位置误差置零
            pos_error[1] = 0.0  # Y轴位置误差置零
            vel_error[0] = 0.0  # X轴速度误差置零
            vel_error[1] = 0.0  # Y轴速度误差置零
        
        # 期望加速度
        acc_des = target['acceleration'] + self.Kp @ pos_error + self.Dp @ vel_error
        
        # 世界坐标系下的控制力
        f_c_world = self.mass * (acc_des + np.array([0, 0, self.gravity]))
        
        # 姿态误差计算
        R_des = self._euler_to_rotation_matrix(target['attitude'])
        e_R = 0.5 * self.vee_map(R_des.T @ R - R.T @ R_des)
        
        # ========== 核心修改：俯仰角超限时禁用横滚/偏航姿态误差 ==========
        if state['is_pitch_exceed']:
            e_R[0] = 0.0  # 横滚姿态误差置零
            e_R[2] = 0.0  # 偏航姿态误差置零
        
        # 角速度误差
        omega_error = state['angular_velocity'] - R.T @ R_des @ target['attitude_rate']
        
        # ========== 核心修改：俯仰角超限时禁用横滚/偏航角速度误差 ==========
        if state['is_pitch_exceed']:
            omega_error[0] = 0.0  # 横滚角速度误差置零
            omega_error[2] = 0.0  # 偏航角速度误差置零
        
        # 控制力矩（几何自适应）
        tau_c = -self.KR * e_R - self.Domega * omega_error + np.cross(state['angular_velocity'], self.J @ state['angular_velocity'])
        
        # 转换到机体坐标系
        f_c_body = R.T @ f_c_world
        
        return f_c_body, tau_c

    def compute_control_wrench(self, state: dict) -> Tuple[np.ndarray, np.ndarray]:
        """计算控制力矩和力（混合控制策略）"""
        target = {
            'position': self.target_position,
            'velocity': self.target_velocity,
            'acceleration': self.target_acceleration,
            'attitude': self.target_attitude,
            'attitude_rate': self.target_attitude_rate,
            'attitude_acceleration': self.target_attitude_acceleration
        }
        
        # 使用几何自适应控制计算主要控制量
        f_c_body, tau_c_geo = self.compute_geometric_adaptive_control(state, target)
        
        # 使用滑模控制增强鲁棒性
        tau_c_smc = self.compute_sliding_mode_control(state, target)
        
        # 混合控制：几何控制为主，滑模控制为补偿
        alpha = 0.7  # 几何控制权重
        tau_c = alpha * tau_c_geo + (1 - alpha) * tau_c_smc
        
        # ========== 最终防护：确保俯仰角超限时横滚/偏航力矩为0 ==========
        if state['is_pitch_exceed']:
            tau_c[0] = 0.0
            tau_c[2] = 0.0
        
        # 更新类成员变量
        self.f_c_body = f_c_body
        self.f_c_world = state['rotation_matrix'] @ f_c_body
        self.tau_c = tau_c
        
        return f_c_body, tau_c

    def inverse_nonlinear_mapping(self, W):
        """修正后的代数逆映射函数"""
        # 尾部推力
        u7 = (2/1) * W[4]                     
        
        # 左/右旋翼的 X轴分力
        u1 = W[0]/2 - (10/3)*W[5]              
        u4 = W[0]/2 + (10/3)*W[5]              
        
        # 左/右旋翼的 Z轴分力
        Fz_front = W[2]
        u2 = Fz_front/2 - (10/3)*W[3]  
        u5 = Fz_front/2 + (10/3)*W[3]  

        # 侧向分力均分
        target_Fy = W[1]
        u3 = -target_Fy / 2.0
        u6 = -target_Fy / 2.0
        
        # 计算推力和角度
        F1 = np.sqrt(u1**2 + u2**2 + u3**2)
        F2 = np.sqrt(u4**2 + u5**2 + u6**2)
        F3 = u7
        
        # 防止除零保护
        eps = 1e-8
        F1_safe = max(F1, eps)
        F2_safe = max(F2, eps)

        # 求解倾转角度
        alpha1 = np.arctan2(u1, u2)  
        alpha2 = np.arctan2(u4, u5)
        
        val1 = np.clip(u3 / F1_safe, -1.0 + eps, 1.0 - eps)
        val2 = np.clip(u6 / F2_safe, -1.0 + eps, 1.0 - eps)
        
        theta1 = np.arcsin(val1)
        theta2 = np.arcsin(val2)
        
        return np.array([F1, F2, F3, alpha1, alpha2, theta1, theta2])

    def allocate_actuators(self, f_c_body: np.ndarray, tau_c: np.ndarray, state: dict):
        """分配执行器命令"""
        # ========== 核心修改：俯仰角超限时置零横滚/偏航力矩 ==========
        if state['is_pitch_exceed']:
            tau_c[0] = 0.0  # 横滚力矩置零
            tau_c[2] = 0.0  # 偏航力矩置零
            f_c_body[0] = 0.0  # X轴力置零
            f_c_body[1] = 0.0  # Y轴力置零
        
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
        F1, F2, F3, alpha1, alpha2, theta1, theta2 = uu
        
        # ========== 核心修改：俯仰角超限时置零横滚/偏航倾转角度 ==========
        if state['is_pitch_exceed']:
            alpha1 = 0.0
            alpha2 = 0.0
            theta1 = 0.0
            theta2 = 0.0
        
        # 推力限制
        T_max = 60
        F1 = np.clip(F1, 0, T_max)
        F2 = np.clip(F2, 0, T_max)
        F3 = np.clip(F3, -15, 15)
        
        # 角度限制
        alpha_max = np.radians(95)
        alpha1 = np.clip(alpha1, -alpha_max, alpha_max)
        alpha2 = np.clip(alpha2, -alpha_max, alpha_max)
        theta_max = np.radians(95)
        theta1 = np.clip(theta1, -theta_max, theta_max)
        theta2 = np.clip(theta2, -theta_max, theta_max)
        
        # 更新状态
        self.T12 = F1
        self.T34 = F2
        self.T5 = F3
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.theta1 = theta1
        self.theta2 = theta2
        
        self.u = np.array([F1, F2, F3, alpha1, alpha2, theta1, theta2])
        
        return F1, F2, F3, alpha1, alpha2, theta1, theta2

    def set_actuators(self, T12: float, T34: float, T5: float, alpha1: float, alpha2: float, theta1: float, theta2: float):
        """应用控制命令到执行器"""
        try:
            # ========== 核心修改：俯仰角超限时强制置零横滚/偏航倾转角度 ==========
            if self.is_pitch_exceed:
                alpha1 = 0.0
                alpha2 = 0.0
                theta1 = 0.0
                theta2 = 0.0
            
            # 设置机臂偏航角度
            if 'tilt_pitch_right' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['tilt_pitch_right']] = alpha2
            if 'tilt_pitch_left' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['tilt_pitch_left']] = alpha1
            
            # 设置螺旋桨倾转角度
            if 'tilt_roll_right' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['tilt_roll_right']] = theta1
            if 'tilt_roll_left' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['tilt_roll_left']] = theta2
            
            # 设置推力
            thrust_actuators = ['motor_r_upper', 'motor_r_lower', 'motor_l_upper', 'motor_l_lower', 'motor_rear_upper']
            if all(act in self.actuator_ids for act in thrust_actuators[:4]):
                self.data.ctrl[self.actuator_ids['motor_r_upper']] = T34 / 2
                self.data.ctrl[self.actuator_ids['motor_r_lower']] = T34 / 2
                self.data.ctrl[self.actuator_ids['motor_l_upper']] = T12 / 2
                self.data.ctrl[self.actuator_ids['motor_l_lower']] = T12 / 2
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

    def update_trajectory(self, current_time: float):
        """更新轨迹（适配新坐标系）"""
        if self.trajectory_phase == 0 and self.phase_start_time == 0.0:
            self.phase_start_time = current_time
        
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
        
        phase_elapsed = current_time - self.phase_start_time
        
        if phase_elapsed > phase_durations.get(self.trajectory_phase, 0):
            self.trajectory_phase += 1
            self.phase_start_time = current_time
            print(f"\n轨迹阶段切换: {self.trajectory_phase-1} → {self.trajectory_phase}")
        
        # 在新坐标系中，水平状态对应俯仰角为0度
        if self.trajectory_phase == 0:
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([0.0, 0.0, 0.0])  # 水平状态
            
        elif self.trajectory_phase == 1:
            progress = phase_elapsed / phase_durations[1]
            progress = np.clip(progress, 0.0, 1.0)
            roll_target = progress * self.attitude_target_rad
            self.target_position = np.array([0.0, 0.0, 2.0])
            self.target_attitude = np.array([roll_target, 0.0, 0.0])
            
        # 其他阶段类似实现...
        
        # 速度和加速度归零
        self.target_velocity = np.zeros(3)
        self.target_acceleration = np.zeros(3)
        self.target_attitude_rate = np.zeros(3)
        self.target_attitude_acceleration = np.zeros(3)

    def print_status(self):
        """打印当前状态信息"""
        try:
            state = self.get_state()
            pos = state['position']
            euler_deg = np.degrees(state['euler'])
            target_euler_deg = np.degrees(self.target_attitude)
            
            phase_names = {
                0: "起飞悬停", 1: "Roll转动", 2: "Roll保持",
                3: "Roll恢复", 4: "Pitch转动", 5: "Pitch保持",
                6: "Pitch恢复", 7: "Yaw转动", 8: "Yaw保持",
                9: "Yaw恢复", 10: "最终悬停"
            }
            phase_name = phase_names.get(self.trajectory_phase, "未知阶段")
            
            print(f"\n=== 轨迹阶段: {self.trajectory_phase} ({phase_name}) ===")
            print(f"位置: X={pos[0]:.3f}m, Y={pos[1]:.3f}m, Z={pos[2]:.3f}m")
            print(f"姿态: Roll={euler_deg[0]:.1f}°, Pitch={euler_deg[1]:.1f}°, Yaw={euler_deg[2]:.1f}°")
            print(f"目标姿态: Roll={target_euler_deg[0]:.1f}°, Pitch={target_euler_deg[1]:.1f}°, Yaw={target_euler_deg[2]:.1f}°")
            print(f"执行器状态: T12={self.T12:.2f}N, T34={self.T34:.2f}N, T5={self.T5:.2f}N")
            # ========== 新增：打印俯仰角超限状态 ==========
            print(f"俯仰角限制: {'超限(禁用横滚/偏航)' if self.is_pitch_exceed else '正常'} (阈值: {self.pitch_threshold_deg}°)")
            
        except Exception as e:
            print(f"状态打印失败: {e}")


def main():
    """主函数 - 启动仿真"""
    print("=== 倾转旋翼无人机控制系统（水平状态俯仰角90度）===")
    print(f"⚠️  俯仰角超过70度时将自动禁用横滚/偏航控制 ⚠️\n")
    
    try:
        controller = HnuterController("hnuter201.xml")
        # 设置目标轨迹（简单悬停）
        controller.target_position = np.array([0.0, 0.0, 2.0])  # 目标高度2米
        controller.target_velocity = np.zeros(3)
        controller.target_acceleration = np.zeros(3)
        controller.target_attitude = np.array([0.0, 0.0, 0.0])  # 水平姿态
        controller.target_attitude_rate = np.zeros(3)
        controller.target_attitude_acceleration = np.zeros(3)

        with viewer.launch_passive(controller.model, controller.data) as v:
            print("仿真启动，按 Ctrl+C 终止")
            
            start_time = time.time()
            last_print_time = 0
            print_interval = 1.0
            
            try:
                while v.is_running():
                    current_time = time.time() - start_time
                    
                    controller.update_trajectory(current_time)
                    controller.update_control()

                    mj.mj_step(controller.model, controller.data)
                    v.sync()
                    
                    if current_time - last_print_time > print_interval:
                        controller.print_status()
                        last_print_time = current_time

                    time.sleep(0.001)

            except KeyboardInterrupt:
                print("\n仿真被用户中断")
            
            print("仿真结束")

    except Exception as e:
        print(f"仿真失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
