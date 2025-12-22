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
    def __init__(self, model_path: str = "scene.xml"):
        # 加载MuJoCo模型
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        
        # 打印模型诊断信息
        self._print_model_diagnostics()
        
        # 物理参数（对齐Matlab设置）
        self.dt = self.model.opt.timestep
        self.gravity = 9.81
        self.mass = 1.0  # 对齐Matlab的mass=1.0
        self.J = np.diag([0.1, 0.1, 0.1])  # 对齐Matlab的惯性张量
        self.J_inv = np.linalg.inv(self.J)    # 惯量矩阵逆
        
        # 质心偏移
        self.r_cm = np.array([0.0, 0.0, 0.0])
        
        # 旋翼布局参数
        self.l1 = 0.3  
        self.l2 = 0.5  
        self.k_d = 8.1e-8  
        
        # 位置控制器增益（保留）
        self.Kp_pos = np.diag([8.0, 8.0, 8.0])    
        self.Dp_pos = np.diag([6.0, 6.0, 6.0])    
        
        # ========== Matlab几何控制器核心参数 ==========
        self.Kp_att = 8.0  # 姿态比例增益（Matlab的Kp=8.0）
        self.Kd_att = 2.5  # 角速度微分增益（Matlab的Kd=2.5）
        self.filter_alpha = 0.8  # 低通滤波器系数（Matlab的filter_alpha=0.8）
        
        # 噪声参数（对齐Matlab，减小后的噪声）
        self.gyro_bias = np.array([0.0001, 0.0001, 0.0001])  # 陀螺仪偏置
        self.gyro_noise_std = 0.0005  # 陀螺仪噪声标准差 (rad/s)
        self.attitude_noise_std = 0.001  # 姿态测量噪声标准差
        
        # 外部扰动参数（对齐Matlab）
        self.disturbance_torque_std = 0.000  # 随机扰动标准差
        self.periodic_disturbance_freq = 0.5  # 周期性扰动频率 (Hz)
        self.periodic_disturbance_amp = 0.00  # 周期性扰动幅度
        self.pulse_disturbance_iter = [150, 300]  # 脉冲扰动迭代步
        self.pulse_disturbance_val = np.array([0.00, -0.00, 0.00])  # 脉冲扰动值
        
        # 控制量
        self.f_c_body = np.zeros(3)  
        self.f_c_world = np.zeros(3)  
        self.tau_c = np.zeros(3)     # 控制力矩（Matlab的tau_control）
        self.u = np.zeros(7)         
        
        # 分配矩阵 
        self.A = np.array([
            [1, 0,  0, 1, 0,  0, 0],   # X力分配 
            [0, 0, 1, 0, 0, 1, 0],   # Y力分配
            [0, 1, 0, 0, 1, 0, 1],
            [0, self.l1, 0, 0, -self.l1, 0, 0],   # 滚转力矩
            [0, 0, 0, 0, 0, 0, self.l2],  # 俯仰力矩
            [-self.l1, 0, 0, self.l1, 0, 0, 0]  # 偏航力矩
        ])
        self.A_pinv = np.linalg.pinv(self.A)

        # 目标状态（三阶段轨迹，对齐Matlab的90度俯仰轨迹）
        self.target_position = np.array([0.0, 0.0, 2.0])  
        self.target_attitude_phases = [
            np.array([0, np.pi/2, 0]),   # 阶段1：俯仰90度
            np.array([0, np.pi/2, 0]),   # 阶段2：保持90度
            np.array([0, 0, 0])          # 阶段3：转回0度
        ]
        self.current_phase = 0
        self.phase_reached = False
        self.phase_start_iter = 0
        self.settle_threshold = 0.01  # 稳定阈值（Matlab的0.01）
        
        # 轨迹参数（对齐Matlab）
        self.transition_duration = 10.0  # 过渡阶段时长
        self.stay_duration = 5.0         # 保持时长
        
        # 倾转状态
        self.alpha1 = 0.0  
        self.alpha2 = 0.0  
        self.theta1 = 0.0  
        self.theta2 = 0.0  
        self.T12 = 0.0  
        self.T34 = 0.0  
        self.T5 = 0.0  
        
        # 角度连续性处理
        self.last_alpha1 = 0
        self.last_alpha2 = 0
        self.last_theta1 = 0
        self.last_theta2 = 0

        # 执行器/传感器ID
        self._get_actuator_ids()
        self._get_sensor_ids()
        
        # 日志文件
        self._create_log_file()

        # ========== 控制器状态缓存（Matlab控制器专用） ==========
        self.filtered_omega_prev = np.zeros(3)  # 滤波前的角速度
        self.omega_measured = np.zeros(3)       # 测量角速度（带噪声）
        self.coupling_torque = np.zeros(3)      # 耦合力矩
        self.total_disturbance = np.zeros(3)    # 总扰动力矩
        self.q_measured = np.array([1.0, 0.0, 0.0, 0.0])  # 带噪声的姿态
        
        # 误差记录（对齐Matlab）
        self.error_history = {
            'roll_error': [],
            'pitch_error': [],
            'yaw_error': [],
            'quat_norm_error': [],
            'iterations': [],
            'disturbance': [],
            'coupling_torque': [],
            'measured_error': []
        }
        
        # 迭代计数
        self.iteration = 0
        
        print("倾转旋翼控制器初始化完成（移植Matlab几何姿态控制器）")
    
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
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f'logs/drone_log_matlab_controller_{timestamp}.csv'
        
        # 写入CSV表头（包含Matlab控制器变量）
        with open(self.log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'timestamp', 'pos_x', 'pos_y', 'pos_z', 
                'target_x', 'target_y', 'target_z',
                'roll', 'pitch', 'yaw',
                'target_roll', 'target_pitch', 'target_yaw',
                'curr_quat_w', 'curr_quat_x', 'curr_quat_y', 'curr_quat_z',
                'meas_quat_w', 'meas_quat_x', 'meas_quat_y', 'meas_quat_z',
                'target_quat_w', 'target_quat_x', 'target_quat_y', 'target_quat_z',
                'vel_x', 'vel_y', 'vel_z',
                'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
                'meas_angular_vel_x', 'meas_angular_vel_y', 'meas_angular_vel_z',
                'f_world_x', 'f_world_y', 'f_world_z',
                'f_body_x', 'f_body_y', 'f_body_z',
                'tau_x', 'tau_y', 'tau_z',
                'coupling_tau_x', 'coupling_tau_y', 'coupling_tau_z',
                'disturbance_x', 'disturbance_y', 'disturbance_z',
                'roll_error', 'pitch_error', 'yaw_error',
                'quat_error_norm', 'iteration',
                'T12', 'T34', 'T5',
                'alpha1', 'alpha2', 'theta1', 'theta2',
                'phase'
            ])
        
        print(f"已创建Matlab控制器日志文件: {self.log_file}")
    
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
    
    # ========== Matlab控制器核心函数移植 ==========
    def quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """四元数乘法（Matlab的quaternion_multiply）"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def quaternion_error(self, q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
        """计算四元数误差（Matlab的quaternion_error）"""
        q_conj = np.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])
        q_err = self.quaternion_multiply(q_target, q_conj)
        
        # 确保标量部分为正
        if q_err[0] < 0:
            q_err = -q_err
        return q_err
    
    def vec2skew(self, v: np.ndarray) -> np.ndarray:
        """向量转斜对称矩阵（Matlab的vec2skew）"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    def quat2rotm(self, q: np.ndarray) -> np.ndarray:
        """四元数转旋转矩阵（Matlab的quat2rotm）"""
        w, x, y, z = q
        return np.array([
            [1-2*y**2-2*z**2,   2*x*y-2*z*w,     2*x*z+2*y*w],
            [2*x*y+2*z*w,     1-2*x**2-2*z**2,   2*y*z-2*x*w],
            [2*x*z-2*y*w,     2*y*z+2*x*w,     1-2*x**2-2*y**2]
        ])
    
    def quat2euler(self, q: np.ndarray) -> np.ndarray:
        """四元数转欧拉角（ZYX顺序，Matlab的quat2euler）"""
        w, x, y, z = q
        
        # 滚转 (x轴)
        sinr_cosp = 2 * (w*x + y*z)
        cosr_cosp = 1 - 2 * (x*x + y*y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # 俯仰 (y轴)
        sinp = 2 * (w*y - z*x)
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi/2
        else:
            pitch = np.arcsin(sinp)
        
        # 偏航 (z轴)
        siny_cosp = 2 * (w*z + x*y)
        cosy_cosp = 1 - 2 * (y*y + z*z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def euler2quat(self, euler: np.ndarray) -> np.ndarray:
        """欧拉角转四元数（ZYX顺序）"""
        roll, pitch, yaw = euler
        
        cy = np.cos(yaw/2)
        sy = np.sin(yaw/2)
        cp = np.cos(pitch/2)
        sp = np.sin(pitch/2)
        cr = np.cos(roll/2)
        sr = np.sin(roll/2)
        
        return np.array([
            cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy
        ])
    
    def geometric_attitude_controller(self, q_current: np.ndarray, q_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        移植Matlab的geometric_attitude_controller函数
        返回：控制力矩tau_control, 滤波后的角速度filtered_omega
        """
        # 步骤1：四元数转旋转矩阵
        R_current = self.quat2rotm(q_current)
        R_target = self.quat2rotm(q_target)
        
        # 步骤2：计算姿态误差旋转矩阵
        R_err = R_target.T @ R_current
        
        # 步骤3：计算姿态误差向量（轴角表示）
        tr_err = np.trace(R_err)
        cos_theta = np.clip((tr_err - 1) / 2, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        if theta < 1e-6:
            e_R = np.zeros(3)
        else:
            e_hat = (1 / (2 * np.sin(theta))) * np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1]
            ])
            e_R = theta * e_hat
        
        # 步骤4：计算角速度误差
        omega_desired = np.zeros(3)
        e_omega = self.omega_measured - omega_desired
        
        # 步骤5：计算耦合力矩（科里奥利力）
        omega_vec = self.omega_measured.reshape(3, 1)
        self.coupling_torque = -self.vec2skew(omega_vec).dot(self.J).dot(omega_vec).flatten()
        
        # 步骤6：Matlab原版控制律
        tau_control = -self.Kp_att * e_R - self.Kd_att * e_omega + self.coupling_torque
        
        # 步骤7：低通滤波
        if self.iteration == 1:
            filtered_omega = tau_control
        else:
            filtered_omega = self.filter_alpha * self.filtered_omega_prev + (1 - self.filter_alpha) * tau_control
        
        # 更新缓存
        self.filtered_omega_prev = filtered_omega
        
        return tau_control, filtered_omega
    
    # ========== 噪声和扰动模拟（对齐Matlab） ==========
    def simulate_noise(self, q_current: np.ndarray) -> None:
        """模拟传感器噪声（陀螺仪+姿态）"""
        # 陀螺仪噪声：偏置 + 高斯白噪声
        gyro_noise = self.gyro_bias + self.gyro_noise_std * np.random.randn(3)
        
        # 姿态测量噪声
        attitude_noise = self.attitude_noise_std * np.random.randn(4)
        self.q_measured = q_current + attitude_noise
        self.q_measured = self.q_measured / np.linalg.norm(self.q_measured)
        
        # 更新测量角速度（带噪声）
        self.omega_measured = self.filtered_omega_prev + gyro_noise
    
    def simulate_disturbance(self) -> None:
        """模拟外部扰动（随机+周期性+脉冲）"""
        t_current = self.iteration * self.dt
        
        # 随机扰动
        random_disturbance = self.disturbance_torque_std * np.random.randn(3)
        
        # 周期性扰动
        periodic_disturbance = self.periodic_disturbance_amp * np.array([
            np.sin(2*np.pi*self.periodic_disturbance_freq*t_current),
            np.cos(2*np.pi*self.periodic_disturbance_freq*t_current),
            0.5*np.sin(4*np.pi*self.periodic_disturbance_freq*t_current)
        ])
        
        # 脉冲扰动
        pulse_disturbance = np.zeros(3)
        if self.iteration in self.pulse_disturbance_iter:
            pulse_disturbance = self.pulse_disturbance_val
            print(f'    迭代步 {self.iteration}: 检测到脉冲扰动')
        
        # 总扰动
        self.total_disturbance = random_disturbance + periodic_disturbance + pulse_disturbance
    
    # ========== 核心控制逻辑 ==========
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
            
            state['rotation_matrix'] = self.quat2rotm(state['quaternion'])
            state['euler'] = self.quat2euler(state['quaternion'])
            
            # 计算机体加速度
            if hasattr(self, '_last_vel'):
                state['acceleration'] = (state['velocity'] - self._last_vel) / self.dt
            self._last_vel = state['velocity'].copy()
            
            if np.any(np.isnan(state['position'])):
                print("警告: 位置数据包含NaN，使用零值")
                state['position'] = np.zeros(3)
                
            return state
        except Exception as e:
            print(f"状态获取错误: {e}")
            return state
    
    def compute_control_wrench(self, state: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        整合Matlab几何控制器的控制力/力矩计算
        """
        position = state['position']
        velocity = state['velocity']
        q_current = state['quaternion']
        omega_current = state['angular_velocity']
        
        # ========== 1. 噪声和扰动模拟 ==========
        self.simulate_noise(q_current)
        self.simulate_disturbance()
        
        # ========== 2. 位置控制器（保留原有逻辑） ==========
        pos_error = self.target_position - position
        vel_error = np.zeros(3) - velocity  # 期望速度为0
        
        acc_des = self.Kp_pos @ pos_error + self.Dp_pos @ vel_error
        f_c_world = self.mass * (acc_des + np.array([0, 0, self.gravity]))
        
        # ========== 3. Matlab几何姿态控制器（核心替换） ==========
        # 当前阶段的目标姿态
        target_euler = self.target_attitude_phases[self.current_phase]
        q_target = self.euler2quat(target_euler)
        
        # 调用Matlab移植的控制器
        self.tau_c, filtered_omega = self.geometric_attitude_controller(q_current, q_target)
        
        # 应用扰动到控制力矩
        self.tau_c += self.total_disturbance
        
        # ========== 4. 坐标变换 ==========
        R = state['rotation_matrix']
        f_c_body = R.T @ f_c_world
        
        # 更新类变量
        self.f_c_body = f_c_body
        self.f_c_world = f_c_world
        
        return f_c_body, self.tau_c
    
    def update_trajectory_phase(self, state: dict) -> None:
        """三阶段轨迹管理（对齐Matlab的90度俯仰轨迹）"""
        q_current = state['quaternion']
        target_euler = self.target_attitude_phases[self.current_phase]
        q_target = self.euler2quat(target_euler)
        
        # 计算四元数误差
        q_err = self.quaternion_error(q_current, q_target)
        err_norm = np.linalg.norm(q_err[1:4])
        
        # 记录误差
        current_euler = self.quat2euler(q_current)
        target_euler_deg = np.degrees(target_euler)
        current_euler_deg = np.degrees(current_euler)
        
        self.error_history['roll_error'].append(current_euler[0] - target_euler[0])
        self.error_history['pitch_error'].append(current_euler[1] - target_euler[1])
        self.error_history['yaw_error'].append(current_euler[2] - target_euler[2])
        self.error_history['quat_norm_error'].append(err_norm)
        self.error_history['iterations'].append(self.iteration)
        self.error_history['disturbance'].append(self.total_disturbance.copy())
        self.error_history['coupling_torque'].append(np.linalg.norm(self.coupling_torque))
        self.error_history['measured_error'].append(err_norm)
        
        # 阶段切换逻辑
        if not self.phase_reached and err_norm < self.settle_threshold:
            self.phase_reached = True
            phase_duration = (self.iteration - self.phase_start_iter) * self.dt
            print(f"阶段 {self.current_phase+1} 到达目标姿态，持续时间: {phase_duration:.2f} 秒")
            print(f"    当前姿态: 滚转={current_euler_deg[0]:.1f}°, 俯仰={current_euler_deg[1]:.1f}°, 偏航={current_euler_deg[2]:.1f}°")
            
            # 阶段2需要保持指定时间
            if self.current_phase == 1:
                # 检查保持时间
                hold_time = (self.iteration - self.phase_start_iter) * self.dt
                if hold_time >= self.stay_duration:
                    self.current_phase += 1
                    self.phase_reached = False
                    self.phase_start_iter = self.iteration
                    print(f"\n停留时间已到 ({self.stay_duration:.1f} 秒)，进入阶段 {self.current_phase+1}")
            elif self.current_phase < 2:
                self.current_phase += 1
                self.phase_reached = False
                self.phase_start_iter = self.iteration
                next_target = np.degrees(self.target_attitude_phases[self.current_phase])
                print(f"\n进入阶段 {self.current_phase+1}，目标姿态: 滚转={next_target[0]:.1f}°, 俯仰={next_target[1]:.1f}°, 偏航={next_target[2]:.1f}°")
    
    # ========== 原有框架函数（适配新控制器） ==========
    def inverse_nonlinear_mapping(self, W):
        """修正后的代数逆映射函数"""
        u7 = (2/1) * W[4]                     
        u1 = W[0]/2 - (10/3)*W[5]              
        u4 = W[0]/2 + (10/3)*W[5]              
        Fz_front = W[2]
        u2 = Fz_front/2 - (10/3)*W[3]  
        u5 = Fz_front/2 + (10/3)*W[3]  

        target_Fy = W[1]
        u3 = -target_Fy / 2.0
        u6 = -target_Fy / 2.0
        
        F1 = np.sqrt(u1**2 + u2**2 + u3**2)
        F2 = np.sqrt(u4**2 + u5**2 + u6**2)
        F3 = u7
        
        eps = 1e-8
        F1_safe = F1 if F1 > eps else eps
        F2_safe = F2 if F2 > eps else eps

        alpha1 = np.arctan2(u1, u2)  
        alpha2 = np.arctan2(u4, u5)
        
        val1 = np.clip(u3 / F1_safe, -1.0 + eps, 1.0 - eps)
        val2 = np.clip(u6 / F2_safe, -1.0 + eps, 1.0 - eps)
        
        theta1 = np.arcsin(val1)
        theta2 = np.arcsin(val2)
        
        return np.array([F1, F2, F3, alpha1, alpha2, theta1, theta2])

    def allocate_actuators(self, f_c_body: np.ndarray, tau_c: np.ndarray, state: dict):
        """分配执行器命令"""
        W = np.array([
            f_c_body[0],    # X力
            f_c_body[1],    # Y力
            f_c_body[2],    # Z力
            tau_c[0],       # 滚转力矩
            tau_c[1],       # 俯仰力矩
            tau_c[2]        # 偏航力矩
        ])
        
        uu = self.inverse_nonlinear_mapping(W)
        
        F1 = uu[0]
        F2 = uu[1]
        F3 = uu[2]
        alpha1 = uu[3]
        alpha2 = uu[4]
        theta1 = uu[5]
        theta2 = uu[6]
        
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
        
        self.u = np.array([F1, F2, F3, alpha1, alpha2, theta1, theta2])
        
        return F1, F2, F3, alpha1, alpha2, theta1, theta2
    
    def set_actuators(self, T12: float, T34: float, T5: float, alpha1: float, alpha2: float, theta1: float, theta2: float):
        """应用控制命令到执行器"""
        try:
            if 'arm_pitch_right' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['arm_pitch_right']] = alpha2
            
            if 'arm_pitch_left' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['arm_pitch_left']] = alpha1
            
            if 'prop_tilt_right' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['prop_tilt_right']] = theta1
            
            if 'prop_tilt_left' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['prop_tilt_left']] = theta2
            
            if 'motor_r_upper' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['motor_r_upper']] = T34 / 2
            
            if 'motor_r_lower' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['motor_r_lower']] = T34 / 2
            
            if 'motor_l_upper' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['motor_l_upper']] = T12 / 2
            
            if 'motor_l_lower' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['motor_l_lower']] = T12 / 2
            
            if 'motor_rear_upper' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['motor_rear_upper']] = T5
                
        except Exception as e:
            print(f"设置执行器失败: {e}")
    
    def update_control(self):
        """更新控制量（整合Matlab控制器）"""
        try:
            self.iteration += 1
            state = self.get_state()

            # 计算控制力和力矩（Matlab几何控制器）
            f_c_body, tau_c = self.compute_control_wrench(state)
            
            # 分配执行器命令
            T12, T34, T5, alpha1, alpha2, theta1, theta2 = self.allocate_actuators(f_c_body, tau_c, state)
            
            # 应用控制
            self.set_actuators(T12, T34, T5, alpha1, alpha2, theta1, theta2)
            
            # 更新轨迹阶段
            self.update_trajectory_phase(state)
            
            # 记录状态
            self.log_status(state)
            
            return True
        except Exception as e:
            print(f"控制更新失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def log_status(self, state: dict):
        """记录状态到日志文件"""
        timestamp = time.time()
        position = state.get('position', np.zeros(3))
        velocity = state.get('velocity', np.zeros(3))
        angular_velocity = state.get('angular_velocity', np.zeros(3))
        euler = state.get('euler', np.zeros(3))
        current_quat = state.get('quaternion', np.array([1.0, 0.0, 0.0, 0.0]))
        
        # 当前阶段目标姿态
        target_euler = self.target_attitude_phases[self.current_phase]
        target_quat = self.euler2quat(target_euler)
        
        with open(self.log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                timestamp,
                position[0], position[1], position[2],
                self.target_position[0], self.target_position[1], self.target_position[2],
                euler[0]*180/np.pi, euler[1]*180/np.pi, euler[2]*180/np.pi,
                target_euler[0]*180/np.pi, target_euler[1]*180/np.pi, target_euler[2]*180/np.pi,
                current_quat[0], current_quat[1], current_quat[2], current_quat[3],
                self.q_measured[0], self.q_measured[1], self.q_measured[2], self.q_measured[3],
                target_quat[0], target_quat[1], target_quat[2], target_quat[3],
                velocity[0], velocity[1], velocity[2],
                angular_velocity[0], angular_velocity[1], angular_velocity[2],
                self.omega_measured[0], self.omega_measured[1], self.omega_measured[2],
                self.f_c_world[0], self.f_c_world[1], self.f_c_world[2],
                self.f_c_body[0], self.f_c_body[1], self.f_c_body[2],
                self.tau_c[0], self.tau_c[1], self.tau_c[2],
                self.coupling_torque[0], self.coupling_torque[1], self.coupling_torque[2],
                self.total_disturbance[0], self.total_disturbance[1], self.total_disturbance[2],
                self.error_history['roll_error'][-1]*180/np.pi if self.error_history['roll_error'] else 0,
                self.error_history['pitch_error'][-1]*180/np.pi if self.error_history['pitch_error'] else 0,
                self.error_history['yaw_error'][-1]*180/np.pi if self.error_history['yaw_error'] else 0,
                self.error_history['quat_norm_error'][-1] if self.error_history['quat_norm_error'] else 0,
                self.iteration,
                self.T12, self.T34, self.T5,
                self.alpha1*180/np.pi, self.alpha2*180/np.pi, self.theta1*180/np.pi, self.theta2*180/np.pi,
                self.current_phase
            ])
    
    def print_status(self):
        """打印当前状态信息"""
        try:
            state = self.get_state()
            pos = state['position']
            euler_deg = np.degrees(state['euler'])
            target_euler = self.target_attitude_phases[self.current_phase]
            target_euler_deg = np.degrees(target_euler)
            
            phase_names = {
                0: "转到俯仰90度",
                1: "保持俯仰90度",
                2: "转回俯仰0度"
            }
            phase_name = phase_names.get(self.current_phase, "未知阶段")
            
            print(f"\n=== 轨迹阶段: {self.current_phase+1} ({phase_name}) | 迭代步: {self.iteration} ===")
            print(f"位置: X={pos[0]:.8f}m, Y={pos[1]:.8f}m, Z={pos[2]:.8f}m")
            print(f"目标位置: X={self.target_position[0]:.8f}m, Y={self.target_position[1]:.8f}m, Z={self.target_position[2]:.8f}m")
            print(f"姿态: Roll={euler_deg[0]:.2f}°, Pitch={euler_deg[1]:.2f}°, Yaw={euler_deg[2]:.2f}°")  
            print(f"目标姿态: Roll={target_euler_deg[0]:.1f}°, Pitch={target_euler_deg[1]:.1f}°, Yaw={target_euler_deg[2]:.1f}°") 
            print(f"测量角速度: Roll={np.degrees(self.omega_measured[0]):.2f}°/s, Pitch={np.degrees(self.omega_measured[1]):.2f}°/s, Yaw={np.degrees(self.omega_measured[2]):.2f}°/s")
            print(f"控制力矩: X={self.tau_c[0]:.4f}Nm, Y={self.tau_c[1]:.4f}Nm, Z={self.tau_c[2]:.4f}Nm")
            print(f"耦合力矩: X={self.coupling_torque[0]:.4f}Nm, Y={self.coupling_torque[1]:.4f}Nm, Z={self.coupling_torque[2]:.4f}Nm")
            print(f"当前扰动: X={self.total_disturbance[0]:.4f}Nm, Y={self.total_disturbance[1]:.4f}Nm, Z={self.total_disturbance[2]:.4f}Nm")
            print(f"执行器状态: T12={self.T12:.2f}N, T34={self.T34:.2f}N, T5={self.T5:.2f}N")
            print("--------------------------------------------------")
        except Exception as e:
            print(f"状态打印失败: {e}")
    
    def plot_error_analysis(self):
        """绘制误差分析曲线（对齐Matlab的plot_attitude_errors_with_noise）"""
        fig = plt.figure(figsize=(14, 10))
        
        if not self.error_history['iterations']:
            print("无误差数据可绘制")
            return
        
        iterations = np.array(self.error_history['iterations'])
        roll_error = np.array(self.error_history['roll_error']) * 180/np.pi
        pitch_error = np.array(self.error_history['pitch_error']) * 180/np.pi
        yaw_error = np.array(self.error_history['yaw_error']) * 180/np.pi
        quat_error = np.array(self.error_history['quat_norm_error'])
        
        # 1. 滚转误差
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(iterations, roll_error, 'r-', linewidth=2, label='滚转误差')
        ax1.set_title('滚转轴误差跟踪曲线')
        ax1.set_xlabel('迭代步数')
        ax1.set_ylabel('滚转误差 (度)')
        ax1.grid(True)
        ax1.legend()
        
        # 2. 俯仰误差
        ax2 = plt.subplot(3, 3, 4)
        ax2.plot(iterations, pitch_error, 'g-', linewidth=2, label='俯仰误差')
        ax2.set_title('俯仰轴误差跟踪曲线')
        ax2.set_xlabel('迭代步数')
        ax2.set_ylabel('俯仰误差 (度)')
        ax2.grid(True)
        ax2.legend()
        
        # 3. 偏航误差
        ax3 = plt.subplot(3, 3, 7)
        ax3.plot(iterations, yaw_error, 'b-', linewidth=2, label='偏航误差')
        ax3.set_title('偏航轴误差跟踪曲线')
        ax3.set_xlabel('迭代步数')
        ax3.set_ylabel('偏航误差 (度)')
        ax3.grid(True)
        ax3.legend()
        
        # 4. 四元数误差范数
        ax4 = plt.subplot(3, 3, 2)
        ax4.plot(iterations, quat_error, 'm-', linewidth=2, label='四元数误差')
        ax4.set_title('四元数误差范数')
        ax4.set_xlabel('迭代步数')
        ax4.set_ylabel('误差范数')
        ax4.grid(True)
        ax4.legend()
        
        # 5. 误差收敛速度（对数坐标）
        ax5 = plt.subplot(3, 3, 5)
        ax5.semilogy(iterations, np.abs(roll_error), 'r-', linewidth=1, label='滚转误差')
        ax5.semilogy(iterations, np.abs(pitch_error), 'g-', linewidth=1, label='俯仰误差')
        ax5.semilogy(iterations, np.abs(yaw_error), 'b-', linewidth=1, label='偏航误差')
        ax5.set_title('误差收敛速度（对数坐标）')
        ax5.set_xlabel('迭代步数')
        ax5.set_ylabel('绝对误差 (度)')
        ax5.grid(True)
        ax5.legend()
        
        # 6. 扰动力矩历史
        ax6 = plt.subplot(3, 3, 3)
        if self.error_history['disturbance']:
            disturbance = np.array(self.error_history['disturbance']).T
            ax6.plot(iterations, disturbance[0], 'r-', linewidth=1, label='X轴')
            ax6.plot(iterations, disturbance[1], 'g-', linewidth=1, label='Y轴')
            ax6.plot(iterations, disturbance[2], 'b-', linewidth=1, label='Z轴')
            ax6.set_title('扰动力矩历史')
            ax6.set_xlabel('迭代步数')
            ax6.set_ylabel('扰动力矩 (N·m)')
            ax6.legend()
            ax6.grid(True)
        
        # 7. 误差分布直方图
        ax7 = plt.subplot(3, 3, 6)
        all_errors = np.concatenate([roll_error, pitch_error, yaw_error])
        ax7.hist(all_errors, bins=20, facecolor='blue', edgecolor='black', alpha=0.7)
        ax7.set_title('误差分布直方图')
        ax7.set_xlabel('误差值 (度)')
        ax7.set_ylabel('频次')
        ax7.grid(True)
        
        # 8. 耦合力矩历史
        ax8 = plt.subplot(3, 3, 8)
        if self.error_history['coupling_torque']:
            coupling = np.array(self.error_history['coupling_torque'])
            ax8.plot(iterations, coupling, 'm-', linewidth=1)
            ax8.set_title('耦合力矩大小历史')
            ax8.set_xlabel('迭代步数')
            ax8.set_ylabel('耦合力矩大小 (N·m)')
            ax8.grid(True)
        
        # 9. 性能统计
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        final_roll = roll_error[-1] if len(roll_error) > 0 else 0
        final_pitch = pitch_error[-1] if len(pitch_error) > 0 else 0
        final_yaw = yaw_error[-1] if len(yaw_error) > 0 else 0
        error_std = np.std(all_errors) if len(all_errors) > 0 else 0
        
        stats_text = f"""性能统计：
最终误差：
  滚转: {final_roll:.3f}°
  俯仰: {final_pitch:.3f}°
  偏航: {final_yaw:.3f}°

收敛步数: {len(iterations)}
误差标准差: {error_std:.3f}°

阶段完成情况：
  阶段1: {"✓" if self.current_phase >= 1 else "✗"}
  阶段2: {"✓" if self.current_phase >= 2 else "✗"}
  阶段3: {"✓" if self.current_phase >= 3 else "✗"}
"""
        ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('刚体姿态控制误差分析（移植Matlab控制器）', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'logs/error_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.show()
        
        # 打印性能统计
        print("\n=== 控制器性能统计分析 ===")
        print(f"最终误差[滚转:{final_roll:.3f}°, 俯仰:{final_pitch:.3f}°, 偏航:{final_yaw:.3f}°]")
        print(f"收敛步数:{len(iterations)}, 误差标准差:{error_std:.3f}°")
        
        # 扰动统计
        if self.error_history['disturbance']:
            disturbance = np.array(self.error_history['disturbance'])
            mean_dist = np.mean(disturbance, axis=0)
            max_dist = np.max(np.abs(disturbance), axis=0)
            print(f"\n扰动统计:")
            print(f"    平均扰动: [%.4f, %.4f, %.4f] N·m" % (mean_dist[0], mean_dist[1], mean_dist[2]))
            print(f"    最大扰动: [%.4f, %.4f, %.4f] N·m" % (max_dist[0], max_dist[1], max_dist[2]))

def main():
    """主函数 - 启动Matlab控制器移植版仿真"""
    print("=== 倾转旋翼无人机90°俯仰姿态跟踪仿真（移植Matlab几何控制器）===")
    print("核心算法：Matlab几何姿态控制器（轴角误差+耦合力矩补偿+噪声/扰动）")
    print("轨迹逻辑：阶段1→俯仰90° | 阶段2→保持5s | 阶段3→转回0°")
    
    try:
        # 初始化控制器
        controller = HnuterController("hnuter201.xml")
        
        # 初始阶段设置
        controller.current_phase = 0
        controller.phase_start_iter = 1
        print("\n初始目标：俯仰90度")
        
        # 启动 Viewer
        with viewer.launch_passive(controller.model, controller.data) as v:
            print(f"\n仿真启动：日志文件路径: {controller.log_file}")
            print("按 Ctrl+C 终止仿真")
            
            start_time = time.time()
            last_print_time = 0
            print_interval = 1.0
            
            try:
                while v.is_running():
                    current_time = time.time() - start_time
                    
                    # 更新控制（Matlab几何控制器）
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
                    
                    # 检查是否完成所有阶段
                    if controller.current_phase >= 3:
                        print("\n所有轨迹阶段执行完成！")
                        break

            except KeyboardInterrupt:
                print("\n仿真被用户中断")
            
            print("\n仿真结束")
            final_state = controller.get_state()
            final_euler = np.degrees(final_state['euler'])
            print(f"最终位置: ({final_state['position'][0]:.2f}, {final_state['position'][1]:.2f}, {final_state['position'][2]:.2f})m")
            print(f"最终姿态: Roll={final_euler[0]:.2f}°, Pitch={final_euler[1]:.2f}°, Yaw={final_euler[2]:.2f}°")
            
            # 绘制误差分析曲线
            controller.plot_error_analysis()

    except Exception as e:
        print(f"仿真主循环失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
