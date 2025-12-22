import numpy as np
from typing import Tuple, List, Dict, Any
from utils import vee_map, hat_map, euler_to_rotation_matrix, get_all_axis_types, euler_to_quaternion
from simulation_framework import SimulationFramework


class HnuterController:
    def __init__(self, sim_framework: SimulationFramework):
        # 保存仿真框架实例
        self.sim = sim_framework
        
        # 物理参数
        self.gravity = 9.81
        self.mass = 4.2  # 主机身质量 + 旋翼机构质量 4.2kg
        self.J = np.diag([0.08, 0.12, 0.1])  # 惯量矩阵
        
        # 旋翼布局参数（根据实际模型调整）
        self.l1 = 0.2  # 前旋翼组Y向距离(m) - 从模型看是0.1+0.1=0.2m
        self.l2 = 0.4  # 尾部推进器X向距离(m) - 从模型看是0.4m
        self.k_d = 8.1e-8  # 尾部反扭矩系数
        
        # 几何控制器增益（针对90°大角度微调）
        self.Kp = np.diag([8, 8, 12])  # 位置增益适度提高，Z轴增益更高
        self.Dp = np.diag([6, 6, 8])  # 速度阻尼
        
        # ========== 新增：基于俯仰角的增益调度 ==========  
        self.KR_fast = np.array([8.0, 6.0, 4.0])   # 快轴增益（横滚/偏航）
        self.KR_slow = np.array([0.2, 0.3, 0.5])   # 慢轴增益（偏航/横滚）
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
        
        # 倾转状态
        self.alpha1 = 0.0  # roll右倾角
        self.alpha2 = 0.0  # roll左倾角
        self.theta1 = 0.0  # pitch右倾角
        self.theta2 = 0.0  # pitch左倾角
        self.T12 = 0.0  # 前左旋翼组推力
        self.T34 = 0.0  # 前右旋翼组推力
        self.T5 = 0.0   # 尾部推进器推力
        
        # 目标状态
        self.target_position = np.array([0.0, 0.0, 0.3])  # 初始目标高度
        self.target_velocity = np.zeros(3)
        self.target_acceleration = np.zeros(3)
        self.target_attitude = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
        self.target_attitude_rate = np.zeros(3)
        self.target_attitude_acceleration = np.zeros(3)
        
        print("倾转旋翼控制器初始化完成（适配90°大角度姿态跟踪）")
        print("⚠️  采用基于倾转预测的几何解耦控制方案 ⚠️")
        print(f"俯仰角阈值: {self.sim.pitch_threshold_deg}°")
    
    def compute_control_wrench(self, state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
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
        R_des = euler_to_rotation_matrix(self.target_attitude)
        
        # 转换到虚拟坐标系
        R_virtual = self.virtual_R.T @ R
        R_des_virtual = self.virtual_R.T @ R_des
        
        # 在虚拟坐标系中计算姿态误差
        e_R = 0.5 * vee_map(R_des_virtual.T @ R_virtual - R_virtual.T @ R_des_virtual)
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
    
    def predict_servo_states(self):
        """预测舵机状态（一阶惯性模型）"""
        dt = self.sim.dt
        
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
    

    
    def update_control(self):
        """更新控制量（使用几何解耦控制）"""
        try:
            # 获取当前状态
            state = self.sim.get_state()

            # 计算控制力矩和力（使用几何解耦控制）
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
    
    def get_predicted_tilt_angles(self) -> Tuple[float, float, float, float]:
        """获取预测的倾转角度"""
        return self.predict_servo_states()
    
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
        """获取当前倾转角度"""
        return self.alpha1, self.alpha2, self.theta1, self.theta2
    
    def get_thrusts(self) -> Tuple[float, float, float]:
        """获取当前推力"""
        return self.T12, self.T34, self.T5
