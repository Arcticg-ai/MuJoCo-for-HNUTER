import numpy as np
from typing import Tuple, List, Dict, Any
from utils import rotation_x, rotation_z
from controller import HnuterController
from simulation_framework import SimulationFramework


class ActuatorAllocation:
    def __init__(self, controller: HnuterController, sim_framework: SimulationFramework):
        # 保存控制器和仿真框架实例
        self.controller = controller
        self.sim = sim_framework
        
        # 旋翼布局参数（从控制器获取）
        self.l1 = controller.l1
        self.l2 = controller.l2
        
        # 角度限制（90°大角度，匹配目标）
        self.alpha_max = np.radians(200)  # 略大于90°，留有余量
        self.theta_max = np.radians(200)  # 略大于90°，留有余量
        
        # 推力限制（90°大角度下适度提高上限）
        self.T_max = 60  # 前旋翼组最大推力
        self.T5_max = 15  # 尾部推进器最大推力
        
        print("执行器分配模块初始化完成 (Optimization-based Null-space Allocation Enabled)")
    
    def inverse_nonlinear_mapping(self, W, state):
        """
        修正后的代数逆映射函数（引入零空间优化分配）
        """
        Fx, Fy, Fz, Tx, Ty, Tz = W
        
        # 几何参数 (从self.l1, self.l2获取，避免硬编码)
        # 避免除零
        l1 = self.l1 if self.l1 > 1e-3 else 0.15
        l2 = self.l2 if self.l2 > 1e-3 else 0.5
        
        kappa1 = 1.0 / (2.0 * l1)
        kappa2 = 1.0 / (2.0 * l1)
        kappa3 = 1.0 / l2
        
        # 1. 基础解 (Base Solution, lambda=0)
        # 尾部推力 (由俯仰力矩Ty确定)
        u7 = kappa3 * Ty
        
        # 左/右旋翼的 X轴分力 (由总Fx和偏航力矩Tz确定)
        # u1: Left X, u4: Right X
        u1 = 0.5 * Fx - kappa1 * Tz
        u4 = 0.5 * Fx + kappa1 * Tz
        
        # 左/右旋翼的 Z轴分力 (由总Fz和滚转力矩Tx确定)
        # u2: Left Z, u5: Right Z
        u2 = 0.5 * Fz - kappa2 * Tx
        u5 = 0.5 * Fz + kappa2 * Tx
        
        # 侧向分力基础解 (均分)
        # u3: Left Y, u6: Right Y
        # 注意：u3_base = -0.5 * Fy, u6_base = -0.5 * Fy
        u3_base = -0.5 * Fy
        u6_base = -0.5 * Fy
        
        # 2. 零空间优化 (Null-space Optimization)
        # 目标：利用冗余自由度 lambda 重新分配侧向力，防止单侧饱和
        # u3 = u3_base + lambda
        # u6 = u6_base - lambda
        
        # 计算剩余推力容量平方 (Residual Capacity Squared)
        T_max_sq = self.T_max ** 2
        
        term_left = u1**2 + u2**2
        term_right = u4**2 + u5**2
        
        R_left_sq = T_max_sq - term_left
        R_right_sq = T_max_sq - term_right
        
        # 如果基础分量已经导致过载，则无法通过lambda完全修复，但仍可优化
        # 这里为了数值稳定性，取max(0, ...)
        R_left = np.sqrt(np.maximum(0, R_left_sq))
        R_right = np.sqrt(np.maximum(0, R_right_sq))
        
        # 确定 lambda 的可行范围
        # -R_left <= u3_base + lambda <= R_left
        # -R_right <= u6_base - lambda <= R_right  =>  lambda <= u6_base + R_right  AND  lambda >= u6_base - R_right
        
        lambda_min_left = -R_left - u3_base
        lambda_max_left = R_left - u3_base
        
        lambda_min_right = u6_base - R_right
        lambda_max_right = u6_base + R_right
        
        # 求交集
        lambda_min = np.maximum(lambda_min_left, lambda_min_right)
        lambda_max = np.minimum(lambda_max_left, lambda_max_right)
        
        # 选择最优 lambda
        # 策略：优先选 0 (能量最优)，如果 0 不在范围内，则选边界
        optimal_lambda = 0.0
        
        if lambda_min > lambda_max:
            # 无解情况（两边都严重饱和），取中间值或者保持0
            optimal_lambda = 0.0 
        else:
            if optimal_lambda < lambda_min:
                optimal_lambda = lambda_min
            elif optimal_lambda > lambda_max:
                optimal_lambda = lambda_max
            else:
                optimal_lambda = 0.0
        
        # 应用优化后的 lambda
        u3 = u3_base + optimal_lambda
        u6 = u6_base - optimal_lambda
        
        # 3. 计算推力和角度
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
    
    def allocate_actuators(self, f_c_body: np.ndarray, tau_c: np.ndarray, state: Dict[str, Any]) -> Tuple[float, float, float, float, float, float, float]:
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
        uu = self.inverse_nonlinear_mapping(W, state)
        
        # 提取参数
        T12 = uu[0]  # 前左组推力
        T34 = uu[1]  # 前右组推力
        T5 = uu[2]   # 尾部推进器推力
        alpha1 = uu[3]  # roll左倾角
        alpha2 = uu[4]  # roll右倾角
        theta1 = uu[5]  # pitch左倾角
        theta2 = uu[6]  # pitch右倾角
        
        # 推力限制
        T12 = np.clip(T12, 0, self.T_max)
        T34 = np.clip(T34, 0, self.T_max)
        T5 = np.clip(T5, -self.T5_max, self.T5_max)
        
        # 处理角度连续性，避免跳变
        def handle_angle_continuity(current, last):
            diff = current - last
            if diff > np.pi:
                return current - 2 * np.pi
            elif diff < -np.pi:
                return current + 2 * np.pi
            return current
        
        # 使用上一次的角度值处理连续性
        alpha1 = handle_angle_continuity(alpha1, self.controller.last_alpha1)
        alpha2 = handle_angle_continuity(alpha2, self.controller.last_alpha2)
        theta1 = handle_angle_continuity(theta1, self.controller.last_theta1)
        theta2 = handle_angle_continuity(theta2, self.controller.last_theta2)
        
        # 角度限制
        alpha1 = np.clip(alpha1, -self.alpha_max, self.alpha_max)
        alpha2 = np.clip(alpha2, -self.alpha_max, self.alpha_max)
        theta1 = np.clip(theta1, -self.theta_max, self.theta_max)
        theta2 = np.clip(theta2, -self.theta_max, self.theta_max)
        
        # 更新状态
        self.controller.last_alpha1 = alpha1
        self.controller.last_alpha2 = alpha2
        self.controller.last_theta1 = theta1
        self.controller.last_theta2 = theta2
        
        # 更新控制器状态
        self.controller.set_tilt_angles(alpha1, alpha2, theta1, theta2)
        self.controller.set_thrusts(T12, T34, T5)
        
        # 存储控制输入向量
        self.controller.u = np.array([T12, T34, T5, alpha1, alpha2, theta1, theta2])
        
        return T12, T34, T5, alpha1, alpha2, theta1, theta2
    
    def set_actuators(self, T12: float, T34: float, T5: float, 
                     alpha1: float, alpha2: float, theta1: float, theta2: float):
        """应用控制命令到执行器"""
        try:            
            # 设置机臂偏航角度 (alpha)
            ctrl_values = {
                'tilt_pitch_left': alpha1,
                'tilt_pitch_right': alpha2,
                'tilt_roll_right': theta2,  # 右组侧向倾角 theta2
                'tilt_roll_left': theta1,   # 左组侧向倾角 theta1
                'motor_l_upper': T12 / 2,
                'motor_l_lower': T12 / 2,
                'motor_r_upper': T34 / 2,
                'motor_r_lower': T34 / 2,
                'motor_rear_upper': T5
            }
            
            # 应用控制值到仿真框架
            self.sim.set_actuators(ctrl_values)
            
        except Exception as e:
            print(f"设置执行器失败: {e}")
    
    def allocate_and_apply(self, f_c_body: np.ndarray, tau_c: np.ndarray, state: Dict[str, Any]) -> bool:
        """完整的分配和应用流程"""
        try:
            # 分配执行器命令
            T12, T34, T5, alpha1, alpha2, theta1, theta2 = self.allocate_actuators(f_c_body, tau_c, state)
            
            # 应用控制命令到执行器
            self.set_actuators(T12, T34, T5, alpha1, alpha2, theta1, theta2)
            
            return True
        except Exception as e:
            print(f"分配和应用失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_allocation_results(self) -> Dict[str, float]:
        """获取分配结果"""
        return {
            'T12': self.controller.T12,
            'T34': self.controller.T34,
            'T5': self.controller.T5,
            'alpha1': self.controller.alpha1,
            'alpha2': self.controller.alpha2,
            'theta1': self.controller.theta1,
            'theta2': self.controller.theta2
        }
    
    def get_actuator_limits(self) -> Dict[str, float]:
        """获取执行器限制"""
        return {
            'T_max': self.T_max,
            'T5_max': self.T5_max,
            'alpha_max_deg': np.degrees(self.alpha_max),
            'theta_max_deg': np.degrees(self.theta_max)
        }