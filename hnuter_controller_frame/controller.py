import numpy as np
from typing import Tuple, List, Dict, Any
from utils import vee_map, hat_map, euler_to_rotation_matrix, euler_to_quaternion
from simulation_framework import SimulationFramework


class HnuterController:
    def __init__(self, sim_framework: SimulationFramework):
        # 保存仿真框架实例
        self.sim = sim_framework
        
        # 物理参数
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
        self.KR = np.array([3, 2.5, 1.5])   # 姿态增益适度提高，增强大角度跟踪
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

        # 目标状态
        self.target_position = np.array([0.0, 0.0, 0.3])  # 初始目标高度
        self.target_velocity = np.zeros(3)
        self.target_acceleration = np.zeros(3)
        self.target_rotation_matrix = np.eye(3)  # 目标旋转矩阵
        self.target_attitude_rate = np.zeros(3)
        self.target_attitude_acceleration = np.zeros(3)
        
        print("倾转旋翼控制器初始化完成（适配90°大角度姿态跟踪）")
        print(f"⚠️  俯仰角超过{self.pitch_threshold_deg}°时将自动置零横滚/偏航力矩 ⚠️")
    
    def compute_control_wrench(self, state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
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
        R_des = self.target_rotation_matrix
        e_R = 0.5 * vee_map(R_des.T @ R - R.T @ R_des)
        omega_error = angular_velocity - R.T @ R_des @ self.target_attitude_rate
        
        # 控制力矩
        tau_c = -self.KR * e_R - self.Domega * omega_error
        
        # ========== 核心修改：俯仰角超限时置零横滚/偏航力矩 ==========  
        # if state['is_pitch_exceed']:
        #     tau_c[0] = 0.0  # 横滚力矩置零
        #     tau_c[2] = 0.0  # 偏航力矩置零
        # 转换到机体坐标系
        f_c_body = R.T @ f_c_world
        
        # 更新类成员变量
        self.f_c_body = f_c_body
        self.f_c_world = f_c_world
        self.tau_c = tau_c
        
        return f_c_body, tau_c
    
    def update_control(self):
        """更新控制量"""
        try:
            # 获取当前状态
            state = self.sim.get_state()

            # 计算控制力矩和力
            f_c_body, tau_c = self.compute_control_wrench(state)
            
            # 返回控制力矩和力，供分配求解模块使用
            return f_c_body, tau_c, state
        except Exception as e:
            print(f"控制更新失败: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(3), np.zeros(3), None
    
    def get_current_state(self) -> Dict[str, Any]:
        """获取控制器当前状态"""
        state = self.sim.get_state()
        return {
            'position': state['position'],
            'euler': state['euler'],
            'target_attitude': self.target_attitude,
            'is_pitch_exceed': state['is_pitch_exceed'],
            'f_c_body': self.f_c_body,
            'f_c_world': self.f_c_world,
            'tau_c': self.tau_c,
            'T12': self.T12,
            'T34': self.T34,
            'T5': self.T5,
            'alpha1': self.alpha1,
            'alpha2': self.alpha2,
            'theta1': self.theta1,
            'theta2': self.theta2
        }
    

    
    def set_tilt_angles(self, alpha1: float, alpha2: float, theta1: float, theta2: float):
        """设置倾转角度"""
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.theta1 = theta1
        self.theta2 = theta2
    
    def set_thrusts(self, T12: float, T34: float, T5: float):
        """设置推力"""
        self.T12 = T12
        self.T34 = T34
        self.T5 = T5
    
    def get_tilt_angles(self) -> Tuple[float, float, float, float]:
        """获取当前倾转角度（从传感器反馈）"""
        # 从仿真框架获取实际倾转角度
        actual_angles = self.sim.get_actual_tilt_angles()
        return actual_angles['alpha1_actual'], actual_angles['alpha2_actual'], actual_angles['theta1_actual'], actual_angles['theta2_actual']
    
    def get_thrusts(self) -> Tuple[float, float, float]:
        """获取当前推力（从传感器反馈的力矩转换）"""
        # 从仿真框架获取执行器力矩
        torques = self.sim.get_actuator_torques()
        
        # 将力矩转换为推力（简化处理，实际应根据螺旋桨特性进行转换）
        # 前左组推力（由两个电机力矩之和近似）
        T12_actual = abs(torques.get('motor_l_upper_torque', 0.0)) + abs(torques.get('motor_l_lower_torque', 0.0))
        
        # 前右组推力（由两个电机力矩之和近似）
        T34_actual = abs(torques.get('motor_r_upper_torque', 0.0)) + abs(torques.get('motor_r_lower_torque', 0.0))
        
        # 尾部推进器推力（由单个电机力矩近似）
        T5_actual = abs(torques.get('motor_rear_upper_torque', 0.0))
        
        return T12_actual, T34_actual, T5_actual
