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
        # ... (前面的代码保持不变) ...
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        self._print_model_diagnostics()
        
        # 物理参数
        self.dt = self.model.opt.timestep
        self.gravity = 9.81
        self.mass = 4.2
        self.J = np.diag([0.08, 0.12, 0.1])
        
        self.l1 = 0.3
        self.l2 = 0.5
        self.k_d = 8.1e-8
        
        # --- 修改 1: 增加 KI (积分增益) ---
        self.Kp = np.diag([8, 8, 8])      # 位置增益略微调大
        self.Dp = np.diag([6, 6, 6])      # 速度阻尼
        
        # 姿态 PID 参数
        self.KR = np.array([8.0, 8.0, 2.0])    # 比例增益
        self.KI = np.array([1.0, 2.0, 0.5])    # 积分增益 <--- 新增 (Pitch轴给大一点)
        self.Domega = np.array([0.6, 0.6, 0.6]) # 微分/角速度阻尼
        
        # 积分累加器
        self.integral_e_R = np.zeros(3)
        self.integral_limit = 2.0 # 防止积分饱和（Windup）

        # 控制量
        self.f_c_body = np.zeros(3)
        self.f_c_world = np.zeros(3)
        self.tau_c = np.zeros(3)
        self.u = np.zeros(7)

        # ... (分配矩阵等代码保持不变) ...
        self.A = np.array([
            [1, 0,  0, 1, 0,  0, 0,],   
            [0, 0, -1, 0, 0, -1, 0],   
            [0, 1, 0, 0, 1, 0, 1],
            [0, self.l1, 0, 0, -self.l1, 0, 0],   
            [0, 0, 0, 0, 0, 0, self.l2],  
            [-self.l1, 0, 0, self.l1, 0, 0, 0]  
        ])
        
        self.A_pinv = np.linalg.pinv(self.A)

        # 目标状态
        self.target_position = np.array([0.0, 0.0, 1.5])
        self.target_velocity = np.zeros(3)
        self.target_acceleration = np.zeros(3)
        self.target_attitude = np.zeros(3)
        self.target_attitude_rate = np.zeros(3)
        self.target_attitude_acceleration = np.zeros(3)
        
        # 倾转状态初始化
        self.alpha1 = 0.0
        self.alpha2 = 0.0
        self.theta1 = 0.0
        self.theta2 = 0.0
        self.T12 = 0.0
        self.T34 = 0.0
        self.T5 = 0.0
        
        self.last_alpha1 = 0
        self.last_alpha2 = 0
        self.last_theta1 = 0
        self.last_theta2 = 0

        self._get_actuator_ids()
        self._get_sensor_ids()
        self._create_log_file()
        print("初始化完成：已启用PID姿态控制")

    # ... (中间的辅助函数保持不变: _print_model_diagnostics, _create_log_file, log_status, _get_actuator_ids, _get_sensor_ids, get_state, _quat_to_rotation_matrix, _quat_to_euler, vee_map, hat_map) ...
    # 为了节省篇幅，这里省略重复代码，请保留你原有的辅助函数
    
    # 必须保留的辅助函数
    def _print_model_diagnostics(self):
        pass # 请保留原代码
    def _create_log_file(self):
        pass # 请保留原代码
    def log_status(self, state):
        pass # 请保留原代码
    def _get_actuator_ids(self):
        # 这里的代码完全复用原来的
        self.actuator_ids = {}
        try:
            self.actuator_ids['arm_pitch_right'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_pitch_right')
            self.actuator_ids['arm_pitch_left'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_pitch_left')
            self.actuator_ids['prop_tilt_right'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_roll_right')
            self.actuator_ids['prop_tilt_left'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_roll_left')
            thrust_actuators = ['motor_r_upper', 'motor_r_lower', 'motor_l_upper', 'motor_l_lower', 'motor_rear_upper']
            for name in thrust_actuators:
                self.actuator_ids[name] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
        except:
             pass # 简化显示，实际运行请用你原来的代码
    def _get_sensor_ids(self):
        pass # 请保留原代码
    def get_state(self) -> dict:
        # 请完全保留你原来的 get_state 实现
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
            return state
        except:
            return state

    def _quat_to_rotation_matrix(self, quat):
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
        return np.array([[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]])

    def _quat_to_euler(self, quat):
        w, x, y, z = quat
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw])

    def vee_map(self, S):
        return np.array([S[2, 1], S[0, 2], S[1, 0]])
        
    def _euler_to_rotation_matrix(self, euler):
        roll, pitch, yaw = euler
        R_x = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        return R_z @ R_y @ R_x

    # --- 修改 2: 更新 compute_control_wrench 加入积分项 ---
    def compute_control_wrench(self, state: dict) -> Tuple[np.ndarray, np.ndarray]:
        """计算控制力矩和力（几何控制器 + PID）"""
        position = state['position']
        velocity = state['velocity']
        
        pos_error = self.target_position - position
        vel_error = self.target_velocity - velocity
        
        acc_des = self.target_acceleration + self.Kp @ pos_error + self.Dp @ vel_error
        f_c_world = self.mass * (acc_des + np.array([0, 0, self.gravity]))
        
        R = state['rotation_matrix']
        angular_velocity = state['angular_velocity']
        R_des = self._euler_to_rotation_matrix(self.target_attitude)
        
        # 几何姿态误差
        e_R = 0.5 * self.vee_map(R_des.T @ R - R.T @ R_des)
        omega_error = angular_velocity - R.T @ R_des @ self.target_attitude_rate
        
        # --- 积分项更新 ---
        # 简单的矩形积分法
        self.integral_e_R += e_R * self.dt
        
        # 积分限幅 (Anti-windup)
        # 防止因长时间大误差导致积分项过大，系统超调
        self.integral_e_R = np.clip(self.integral_e_R, -self.integral_limit, self.integral_limit)
        
        # --- PID 控制律 ---
        # tau = -Kp*e - Ki*int(e) - Kd*w
        tau_c = -self.KR * e_R - self.KI * self.integral_e_R - self.Domega * omega_error
        
        f_c_body = R.T @ f_c_world
        
        self.f_c_body = f_c_body
        self.f_c_world = f_c_world
        self.tau_c = tau_c
        
        return f_c_body, tau_c

    # ... (inverse_nonlinear_mapping 保持不变) ...
    def inverse_nonlinear_mapping(self, W):
        # 你的原代码逻辑是正确的，不需要改
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
        eps = 1e-9
        F1_safe = F1 if F1 > eps else eps
        F2_safe = F2 if F2 > eps else eps
        alpha1 = np.arctan2(u1, u2)  
        alpha2 = np.arctan2(u4, u5)
        val1 = np.clip(u3 / F1_safe, -1.0, 1.0)
        val2 = np.clip(u6 / F2_safe, -1.0, 1.0)
        theta1 = np.arcsin(val1)
        theta2 = np.arcsin(val2)
        return np.array([F1, F2, F3, alpha1, alpha2, theta1, theta2])

    # ... (allocate_actuators, _handle_angle_continuity, set_actuators, update_control 保持不变) ...
    def allocate_actuators(self, f_c_body, tau_c, state):
        return super().allocate_actuators(f_c_body, tau_c, state) if hasattr(super(), 'allocate_actuators') else self._original_allocate(f_c_body, tau_c)
    
    # 为了简化，这里直接复制你的 allocate_actuators 实现
    def allocate_actuators(self, f_c_body: np.ndarray, tau_c: np.ndarray, state: dict):
        W = np.array([f_c_body[0], f_c_body[1], f_c_body[2], tau_c[0], tau_c[1], tau_c[2]])
        uu = self.inverse_nonlinear_mapping(W)
        F1, F2, F3 = uu[0], uu[1], uu[2]
        alpha1, alpha2 = uu[3], uu[4]
        theta1, theta2 = uu[5], uu[6]
        
        T_max = 50
        F1 = np.clip(F1, 0, T_max)
        F2 = np.clip(F2, 0, T_max)
        F3 = np.clip(F3, -10, 10)
        
        alpha_max = np.radians(100)
        alpha1 = np.clip(alpha1, -alpha_max, alpha_max)
        alpha2 = np.clip(alpha2, -alpha_max, alpha_max)
        theta_max = np.radians(100)
        theta1 = np.clip(theta1, -theta_max, theta_max)
        theta2 = np.clip(theta2, -theta_max, theta_max)
        
        alpha1 = self._handle_angle_continuity(alpha1, self.last_alpha1)
        alpha2 = self._handle_angle_continuity(alpha2, self.last_alpha2)
        theta1 = self._handle_angle_continuity(theta1, self.last_theta1)
        theta2 = self._handle_angle_continuity(theta2, self.last_theta2)
        
        self.last_alpha1 = alpha1
        self.last_alpha2 = alpha2
        self.last_theta1 = theta1
        self.last_theta2 = theta2
        
        self.T12, self.T34, self.T5 = F1, F2, F3
        self.alpha1, self.alpha2 = alpha1, alpha2
        self.theta1, self.theta2 = theta1, theta2
        self.u = np.array([F1, F2, F3, alpha1, alpha2, theta2, theta2])
        return F1, F2, F3, alpha1, alpha2, theta1, theta2

    def _handle_angle_continuity(self, current, last):
        diff = current - last
        if diff > np.pi: return current - 2 * np.pi
        elif diff < -np.pi: return current + 2 * np.pi
        return current

    def set_actuators(self, T12, T34, T5, alpha1, alpha2, theta1, theta2):
        # 复制你的实现
        try:
            if 'arm_pitch_right' in self.actuator_ids: self.data.ctrl[self.actuator_ids['arm_pitch_right']] = alpha2
            if 'arm_pitch_left' in self.actuator_ids: self.data.ctrl[self.actuator_ids['arm_pitch_left']] = alpha1
            if 'prop_tilt_right' in self.actuator_ids: self.data.ctrl[self.actuator_ids['prop_tilt_right']] = theta1
            if 'prop_tilt_left' in self.actuator_ids: self.data.ctrl[self.actuator_ids['prop_tilt_left']] = theta2
            if 'motor_r_upper' in self.actuator_ids: self.data.ctrl[self.actuator_ids['motor_r_upper']] = T34 / 2
            if 'motor_r_lower' in self.actuator_ids: self.data.ctrl[self.actuator_ids['motor_r_lower']] = T34 / 2
            if 'motor_l_upper' in self.actuator_ids: self.data.ctrl[self.actuator_ids['motor_l_upper']] = T12 / 2
            if 'motor_l_lower' in self.actuator_ids: self.data.ctrl[self.actuator_ids['motor_l_lower']] = T12 / 2
            if 'motor_rear_upper' in self.actuator_ids: self.data.ctrl[self.actuator_ids['motor_rear_upper']] = T5
        except Exception as e:
            print(f"设置执行器失败: {e}")

    def update_control(self):
        try:
            state = self.get_state()
            f_c_body, tau_c = self.compute_control_wrench(state)
            T12, T34, T5, alpha1, alpha2, theta1, theta2 = self.allocate_actuators(f_c_body, tau_c, state)
            self.set_actuators(T12, T34, T5, alpha1, alpha2, theta1, theta2)
            self.log_status(state)
            return True
        except Exception as e:
            print(f"控制更新失败: {e}")
            return False

    def print_status(self):
        # 复制你的实现
        try:
            state = self.get_state()
            pos = state['position']
            euler_deg = np.degrees(state['euler'])
            target_euler_deg = np.degrees(self.target_attitude)
            
            print(f"位置: {pos}")
            # 注意：这里的打印可能会在 Pitch=90 时显示乱跳，这是正常的
            print(f"实际姿态(Deg): Roll={euler_deg[0]:.1f}, Pitch={euler_deg[1]:.1f}, Yaw={euler_deg[2]:.1f}")  
            print(f"目标姿态(Deg): Roll={target_euler_deg[0]:.1f}, Pitch={target_euler_deg[1]:.1f}, Yaw={target_euler_deg[2]:.1f}") 
            print(f"积分误差: {self.integral_e_R}") # 打印积分项观察是否在工作
            print("--------------------------------------------------")
        except Exception as e:
            pass

def main():
    print("=== 倾转旋翼无人机 (PID优化版) ===")
    controller = HnuterController("hnuter201.xml")
    
    # 设定一个大角度测试：Pitch = -85度 (-1.48弧度)
    # 注意：不要设为正好 -1.57 (-90度)，因为数学奇点会导致显示很难看
    controller.target_attitude = np.array([0.0, -1.05, 0.0]) 
    
    with viewer.launch_passive(controller.model, controller.data) as v:
        start_time = time.time()
        while v.is_running():
            step_start = time.time()
            
            controller.update_control()
            mj.mj_step(controller.model, controller.data)
            v.sync()
            
            if time.time() - start_time > 1.0 and time.time() - start_time < 1.1:
                 # 1秒后打印一次状态
                 controller.print_status()

            # 简单的帧率控制
            time_until_next_step = controller.dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()