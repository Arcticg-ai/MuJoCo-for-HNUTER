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
        
        # -------------------------------------------------------------
        # [NEW] 几何配置 (用于确定性分配 DFV-CA)
        # -------------------------------------------------------------
        self.l1 = 0.3  # 前旋翼组Y向距离(m)
        self.l2 = 0.5  # 尾部推进器X向距离(m)
        
        # 定义动力单元在机体坐标系下的位置
        # Left(1), Right(2), Tail(3)
        self.pos_L = np.array([0.0, self.l1, 0.0])
        self.pos_R = np.array([0.0, -self.l1, 0.0])
        self.pos_Tail = np.array([-self.l2, 0.0, 0.0]) 
        
        # [关键修正] 初始化几何矩阵并求伪逆 (pinv)
        self._init_allocation_matrix()
        
        # -------------------------------------------------------------
        # 几何控制器增益 (SE(3) Control Gains)
        # -------------------------------------------------------------
        self.Kp = np.array([12.0, 12.0, 15.0])    # 位置 P
        self.Dp = np.array([6.0, 6.0, 8.0])       # 速度 D
        self.KR = np.array([5.0, 5.0, 1.0])       # 姿态 P
        self.Domega = np.array([0.5, 0.5, 0.5])   # 角速度 D

        # 控制量
        self.f_c_body = np.zeros(3)  # 机体坐标系下的控制力
        self.f_c_world = np.zeros(3)  # 世界坐标系下的控制力
        self.tau_c = np.zeros(3)     # 控制力矩
        self.u = np.zeros(7)         # 控制输入向量

        # 目标状态
        self.target_position = np.array([0.0, 0.0, 0.3])  # 初始目标高度
        self.target_velocity = np.array([0.0, 0.0, 0.0])
        self.target_acceleration = np.array([0.0, 0.0, 0.0])
        self.target_yaw = 0.0  # 单独控制航向
        
        # 倾转状态
        self.alpha1 = 0.0  # roll右倾角
        self.alpha2 = 0.0  # roll左倾角
        self.theta1 = 0.0  # pitch右倾角
        self.theta2 = 0.0  # pitch左倾角
        self.T12 = 0.0  # 前左旋翼组推力
        self.T34 = 0.0  # 前右旋翼组推力
        self.T5 = 0.0   # 尾部推进器推力
        
        # 执行器名称映射
        self._get_actuator_ids()
        self._get_sensor_ids()
        
        # 创建日志文件
        self._create_log_file()
              
        print("倾转旋翼控制器初始化完成（DFV-CA 确定性分配模式 - 修正版）")
    
    def _init_allocation_matrix(self):
        """[NEW] 初始化几何矩阵并求伪逆"""
        # 构造反对称矩阵函数
        def skew(v):
            return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

        # 构造 M_geo (6x6)
        # 第一行块: 力平衡 (Identity)
        # 第二行块: 力矩平衡 (Cross Product)
        top = np.hstack((np.eye(3), np.eye(3)))
        bot = np.hstack((skew(self.pos_L), skew(self.pos_R)))
        M_geo = np.vstack((top, bot))
        
        # [关键修正] 使用 pinv (伪逆) 而不是 inv
        # 原因: 左右旋翼在 Y 轴上无法产生 My (Pitch力矩)，导致 M_geo 第5行全为0，矩阵奇异。
        # pinv 会自动处理这个问题，求解其他5个自由度的最小范数解。
        self.M_inv = np.linalg.pinv(M_geo)
        print("分配矩阵伪逆计算完成。")

    def _solve_allocation_dfv(self, F_des_body, M_des_body):
        """
        [NEW] 确定性力矢量控制分配算法
        """
        # 1. 约束注入 (Constraint Injection) - 计算尾桨推力
        # 尾桨主要负责 My (Pitch)。
        # 假设尾桨推力 T5 正方向为垂直向上(+Z)
        
        d_tail_x = self.pos_Tail[0] # -0.5
        
        # 动态分配系数
        k_pitch = 0.95 # 95% 的俯仰力矩由尾桨承担
        k_lift = 0.2   # 20% 的升力由尾桨承担
        
        # 计算尾桨推力 T5
        # 力矩公式: My = -x_tail * T5 (若T5向上)
        if abs(d_tail_x) > 0.01:
            T5_req = (M_des_body[1] * k_pitch) / (-d_tail_x) + (F_des_body[2] * k_lift) / 3.0
        else:
            T5_req = 0.0
            
        T5 = np.clip(T5_req, -30.0, 30.0) # 饱和限制
        
        # 2. 计算剩余控制量 (Residual Wrench)
        f_tail_vec = np.array([0, 0, T5])
        m_tail_vec = np.cross(self.pos_Tail, f_tail_vec)
        
        F_rem = F_des_body - f_tail_vec
        M_rem = M_des_body - m_tail_vec
        W_rem = np.hstack((F_rem, M_rem))
        
        # 3. 线性求解左右力矢量 (Linear Solve)
        # [f_L, f_R] = pinv(M_geo) * W_rem
        force_vecs = self.M_inv @ W_rem
        f_L = force_vecs[0:3]
        f_R = force_vecs[3:6]
        
        # 4. 几何反解 (Geometric Unwrapping)
        # 将力矢量分解为 推力(T), 两个倾转角
        
        def solve_single_arm(f):
            T = np.linalg.norm(f)
            if T < 0.01: return 0.0, 0.0, 0.0
            
            # 归一化
            n = f / T 
            
            # 解 theta (Roll tilt / prop_tilt): 对应绕X轴旋转
            # ny = -sin(theta)
            sin_theta = -n[1]
            sin_theta = np.clip(sin_theta, -1.0, 1.0)
            theta = np.arcsin(sin_theta)
            
            # 解 alpha (Pitch tilt / arm_pitch): 对应绕Y轴旋转
            # nx = sin(alpha)*cos(theta), nz = cos(alpha)*cos(theta)
            # nx/nz = tan(alpha)
            alpha = np.arctan2(n[0], n[2])
            
            return T, alpha, theta

        T_L, alpha_L, theta_L = solve_single_arm(f_L)
        T_R, alpha_R, theta_R = solve_single_arm(f_R)
        
        return T_L, T_R, T5, alpha_L, alpha_R, theta_L, theta_R

    def update_control(self):
        """更新控制量 - 集成 SE(3) 控制与 DFV-CA 分配"""
        try:
            # 1. 获取状态
            state = self.get_state()
            pos = state['position']
            vel = state['velocity']
            R = state['rotation_matrix']
            omega = state['angular_velocity']
            
            # 2. 几何位置控制 (SE(3) Position Control)
            e_x = pos - self.target_position
            e_v = vel - self.target_velocity
            
            # 期望力 F_des_world
            F_des_world = -self.Kp * e_x - self.Dp * e_v
            F_des_world[2] += self.mass * self.gravity
            
            # 转换到机体坐标系: F_des_body = R^T * F_des_world
            F_des_body = R.T @ F_des_world
            
            # 3. 几何姿态控制 (SE(3) Attitude Control)
            # 构建 R_des (Yaw 控制)
            cy = math.cos(self.target_yaw)
            sy = math.sin(self.target_yaw)
            R_des = np.array([
                [cy, -sy, 0],
                [sy,  cy, 0],
                [ 0,   0, 1]
            ])
            
            # 姿态误差
            R_err_mat = R_des.T @ R - R.T @ R_des
            e_R = 0.5 * self.vee_map(R_err_mat)
            
            # 期望力矩 M_des_body
            M_des_body = -self.KR * e_R - self.Domega * omega
            
            # 记录供日志使用
            self.f_c_body = F_des_body
            self.f_c_world = F_des_world
            self.tau_c = M_des_body
            
            # 4. 调用确定性控制分配 (DFV-CA)
            (T_L, T_R, T_Tail, 
             alpha_L, alpha_R, 
             theta_L, theta_R) = self._solve_allocation_dfv(F_des_body, M_des_body)
            
            # 更新内部状态用于日志
            self.T12 = T_L; self.T34 = T_R; self.T5 = T_Tail
            self.alpha1 = alpha_R; self.alpha2 = alpha_L # 注意左右对应
            self.theta1 = theta_R; self.theta2 = theta_L
            self.u = np.array([T_L, alpha_R, theta_R, T_R, alpha_L, theta_L, T_Tail])

            # 5. 应用控制 (Mapping to Actuators)
            # 对应关系：alpha -> arm_pitch, theta -> prop_tilt
            self.set_actuators(
                T_L,      # T12
                T_R,      # T34
                T_Tail,   # T5
                alpha_R,  # alpha1 (右侧 arm_pitch)
                alpha_L,  # alpha2 (左侧 arm_pitch)
                theta_R,  # theta1 (右侧 prop_tilt)
                theta_L   # theta2 (左侧 prop_tilt)
            )
            
            # 记录状态
            self.log_status(state)
            
            return True
        except Exception as e:
            print(f"控制更新失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    # ------------------------------------------------------------------
    # 以下方法严格保持原样 (ID获取、日志、辅助函数等)
    # ------------------------------------------------------------------
    
    def _print_model_diagnostics(self):
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
        self.log_file = f'logs/drone_log_{timestamp}.csv'
        
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
                'alpha1', 'alpha2',
                'theta1', 'theta2'
            ])
        print(f"已创建日志文件: {self.log_file}")
    
    def log_status(self, state: dict):
        timestamp = time.time()
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
        self.actuator_ids = {}
        try:
            self.actuator_ids['arm_pitch_right'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_pitch_right')
            self.actuator_ids['arm_pitch_left'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_pitch_left')
            self.actuator_ids['prop_tilt_right'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_roll_right')
            self.actuator_ids['prop_tilt_left'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_roll_left')
            
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
                'arm_pitch_right': 0, 'arm_pitch_left': 1,
                'prop_tilt_right': 2, 'prop_tilt_left': 3,
                'motor_r_upper': 4, 'motor_r_lower': 5,
                'motor_l_upper': 6, 'motor_l_lower': 7,
                'motor_rear_upper': 8
            }
            print("使用默认执行器ID映射")

    def _get_sensor_ids(self):
        self.sensor_ids = {}
        try:
            self.sensor_ids['drone_pos'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, 'drone_pos')
            self.sensor_ids['drone_quat'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, 'drone_quat')
            self.sensor_ids['body_vel'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, 'body_vel')
            self.sensor_ids['body_gyro'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, 'body_gyro')
            self.sensor_ids['body_acc'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, 'body_acc')
            
            propeller_sensors = [
                'prop_r_upper_vel', 'prop_r_lower_vel',
                'prop_l_upper_vel', 'prop_l_lower_vel',
                'prop_rear_upper_vel'
            ]
            for name in propeller_sensors:
                self.sensor_ids[name] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, name)
            
            tilt_sensors = [
                'arm_pitch_right_pos', 'arm_pitch_left_pos',
                'prop_tilt_right_pos', 'prop_tilt_left_pos'
            ]
            for name in tilt_sensors:
                self.sensor_ids[name] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, name)
            
            print("传感器ID映射:", self.sensor_ids)
        except Exception as e:
            print(f"获取传感器ID失败: {e}")
            self.sensor_ids = {'drone_pos': 0, 'drone_quat': 1, 'body_vel': 2, 'body_gyro': 3, 'body_acc': 4}
            print("使用默认传感器ID映射")
    
    def get_state(self) -> dict:
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
            if np.any(np.isnan(state['position'])): state['position'] = np.zeros(3)
            return state
        except Exception as e:
            return state

    def _quat_to_rotation_matrix(self, quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])

    def _quat_to_euler(self, quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw])
    
    def vee_map(self, S: np.ndarray) -> np.ndarray:
        return np.array([S[2, 1], S[0, 2], S[1, 0]])

    def set_actuators(self, T12: float, T34: float, T5: float, alpha1: float, alpha2: float, theta1: float, theta2: float):
        try:
            # 兼容性映射：alpha对应arm_pitch, theta对应prop_tilt
            if 'arm_pitch_right' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['arm_pitch_right']] = alpha1
            if 'arm_pitch_left' in self.actuator_ids:
                self.data.ctrl[self.actuator_ids['arm_pitch_left']] = alpha2
            
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
    
    def print_status(self):
        try:
            state = self.get_state()
            pos = state['position']
            vel = state['velocity']
            accel = state['acceleration']
            euler_deg = np.degrees(state['euler'])
            
            print(f"位置: X={pos[0]:.2f}m, Y={pos[1]:.2f}m, Z={pos[2]:.2f}m")
            print(f"目标: X={self.target_position[0]:.2f}m, Y={self.target_position[1]:.2f}m, Z={self.target_position[2]:.2f}m")
            print(f"姿态: R={euler_deg[0]:.1f}, P={euler_deg[1]:.1f}, Y={euler_deg[2]:.1f}")  
            print(f"控制: T_L={self.T12:.2f}, T_R={self.T34:.2f}, T_Tail={self.T5:.2f}")
            print(f"倾转: aR={math.degrees(self.alpha1):.1f}, aL={math.degrees(self.alpha2):.1f}")
            print("--------------------------------------------------")
        except Exception as e:
            print(f"状态打印失败: {e}")

def main():
    print("=== 倾转旋翼无人机状态监控系统 ===")
    try:
        controller = HnuterController("hnuter201.xml")
        
        # 目标: 悬停在 1.5m
        controller.target_position = np.array([0.0, 0.0, 1.5])
        
        with viewer.launch_passive(controller.model, controller.data) as v:
            print("\n仿真启动...")
            start_time = time.time()
            last_print_time = 0
            
            count = 0
            while v.is_running():
                current_time = time.time() - start_time

                # 更新控制
                controller.update_control()
                
                count += 1
                if count % 1 == 0:
                    mj.mj_step(controller.model, controller.data)
                
                v.sync()
                
                if current_time - last_print_time > 0.5:
                    controller.print_status()
                    last_print_time = current_time

                time.sleep(0.001)

    except Exception as e:
        print(f"仿真主循环失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()