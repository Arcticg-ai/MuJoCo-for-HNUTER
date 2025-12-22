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
        
        # 角度限制
        self.alpha_max = np.radians(90)  # 机臂偏航最大角度
        self.theta_max = np.radians(90)  # 螺旋桨倾转最大角度
        
        # 推力限制
        self.T_max = 60  # 前旋翼组最大推力
        self.T5_max = 15  # 尾部推进器最大推力
        
        print("执行器分配模块初始化完成")
    
    def _build_allocation_matrix(self, alpha1: float, alpha2: float, 
                                theta1: float, theta2: float) -> np.ndarray:
        """构建分配矩阵（考虑倾转角）"""
        # 每个机臂的推力方向（使用预测角度）
        dir1 = rotation_z(alpha1) @ rotation_x(theta1) @ np.array([0, 0, 1])
        dir2 = rotation_z(alpha2) @ rotation_x(theta2) @ np.array([0, 0, 1])
        dir3 = np.array([0, 0, 1])  # 尾部推进器方向固定
        
        # 位置向量（从机身中心到各推力点）
        r1 = np.array([0, self.l1, 0])   # 左机臂
        r2 = np.array([0, -self.l1, 0])  # 右机臂
        r3 = np.array([-self.l2, 0, 0])  # 尾部
        
        # 构建分配矩阵 (6x7)
        A = np.zeros((6, 7))
        
        # 左机臂贡献
        A[0:3, 0] = dir1  # 力
        A[3:6, 0] = np.cross(r1, dir1)  # 力矩
        
        # 右机臂贡献
        A[0:3, 1] = dir2  # 力
        A[3:6, 1] = np.cross(r2, dir2)  # 力矩
        
        # 尾部推进器贡献
        A[0:3, 2] = dir3  # 力
        A[3:6, 2] = np.cross(r3, dir3)  # 力矩
        
        return A
    
    def inverse_nonlinear_mapping_with_delay(self, W: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
        """带延迟补偿的非线性逆映射"""
        # 预测舵机状态
        pred_alpha1, pred_alpha2, pred_theta1, pred_theta2 = self.controller.get_predicted_tilt_angles()
        
        # 使用预测状态构建分配矩阵
        A_pred = self._build_allocation_matrix(pred_alpha1, pred_alpha2, pred_theta1, pred_theta2)
        
        try:
            # 尝试求解控制输入
            u_pred = np.linalg.lstsq(A_pred, W, rcond=None)[0]
            
            # 提取控制量
            T12 = max(0, u_pred[0])
            T34 = max(0, u_pred[1])
            T5 = u_pred[2]
            
            # 角度命令（基于实际动力学计算）
            # 使用预测误差来补偿延迟
            actual_angles = self.sim.get_actual_tilt_angles()
            alpha1_error = pred_alpha1 - actual_angles['alpha1_actual']
            alpha2_error = pred_alpha2 - actual_angles['alpha2_actual']
            theta1_error = pred_theta1 - actual_angles['theta1_actual']
            theta2_error = pred_theta2 - actual_angles['theta2_actual']
            
            # 计算角度命令，包含延迟补偿
            alpha1_cmd = pred_alpha1 - 0.5 * alpha1_error
            alpha2_cmd = pred_alpha2 - 0.5 * alpha2_error
            theta1_cmd = pred_theta1 - 0.5 * theta1_error
            theta2_cmd = pred_theta2 - 0.5 * theta2_error
            
            return np.array([T12, T34, T5, alpha1_cmd, alpha2_cmd, theta1_cmd, theta2_cmd])
            
        except:
            # 如果求解失败，使用简化方法
            return self.inverse_nonlinear_mapping_simple(W, state)
    
    def inverse_nonlinear_mapping_simple(self, W: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
        """简化的非线性逆映射（备选方案）"""
        # 提取控制向量
        Fx, Fy, Fz, Tx, Ty, Tz = W
        
        # 尾部推力主要提供俯仰力矩
        T5 = Ty / self.l2 if abs(self.l2) > 1e-6 else 0
        
        # 左右机臂总推力提供升力
        total_front_thrust = max(0, Fz - T5)
        
        # 根据滚转力矩分配左右推力
        T12 = total_front_thrust/2 + Tx/(2*self.l1)
        T34 = total_front_thrust/2 - Tx/(2*self.l1)
        
        # 根据X方向力和偏航力矩计算机臂偏航角
        if abs(T12) > 1e-6:
            alpha1 = np.arctan2(Fx/2 - Tz/(2*self.l1), T12)
        else:
            alpha1 = 0.0
            
        if abs(T34) > 1e-6:
            alpha2 = np.arctan2(Fx/2 + Tz/(2*self.l1), T34)
        else:
            alpha2 = 0.0
        
        # 根据Y方向力计算螺旋桨倾转角
        if abs(T12) > 1e-6:
            theta1 = np.arcsin(Fy/(2*T12))
        else:
            theta1 = 0.0
            
        if abs(T34) > 1e-6:
            theta2 = np.arcsin(Fy/(2*T34))
        else:
            theta2 = 0.0
        
        return np.array([T12, T34, T5, alpha1, alpha2, theta1, theta2])
    
    def allocate_actuators(self, f_c_body: np.ndarray, tau_c: np.ndarray, state: Dict[str, Any]) -> Tuple[float, float, float, float, float, float, float]:
        """分配执行器命令（使用带延迟补偿的非线性逆映射）"""
        # 构造控制向量W
        W = np.array([
            f_c_body[0],    # X力
            f_c_body[1],    # Y力
            f_c_body[2],    # Z力
            tau_c[0],       # 滚转力矩
            tau_c[1],       # 俯仰力矩
            tau_c[2]        # 偏航力矩
        ])
        
        # 带延迟补偿的非线性逆映射
        uu = self.inverse_nonlinear_mapping_with_delay(W, state)
        
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
        
        # 角度限制
        alpha1 = np.clip(alpha1, -self.alpha_max, self.alpha_max)
        alpha2 = np.clip(alpha2, -self.alpha_max, self.alpha_max)
        theta1 = np.clip(theta1, -self.theta_max, self.theta_max)
        theta2 = np.clip(theta2, -self.theta_max, self.theta_max)
        
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
                'tilt_roll_left': theta1,
                'tilt_roll_right': theta2,
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
