import numpy as np
import math

class HnuterController:
    def __init__(self):
        # 物理参数（仅保留必要的参数）
        self.l1 = 0.3  # 前旋翼组Y向距离(m)
        self.l2 = 0.5  # 尾部推进器X向距离(m)
        
        # 倾转状态
        self.alpha1 = 0.0  # roll右倾角
        self.alpha2 = 0.0  # roll左倾角
        self.theta1 = 0.0  # pitch右倾角
        self.theta2 = 0.0  # pitch左倾角
        self.T12 = 0.0  # 前左旋翼组推力
        self.T34 = 0.0  # 前右旋翼组推力
        self.T5 = 0.0   # 尾部推进器推力
        self.u = np.zeros(7)  # 控制输入向量
        
        # 执行器名称映射（简化版本）
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
        
        # 模拟data结构用于测试
        self.data = type('Data', (), {'ctrl': np.zeros(9)})()
        
        print("简化版倾转旋翼控制器初始化完成")
    
    def inverse_nonlinear_mapping(self, W):
        """
        非线性逆映射函数
        输入：W（6×1向量）
        输出：uu = [F1, F2, F3, alpha1, alpha2, theta1, theta2]，其中alpha1 = alpha2
        """
        # 步骤1：计算u中与t无关的分量（u1, u2, u4, u5, u7）
        # 根据物理模型调整计算系数，使其更符合实际系统
        u7 = (1/self.l2) * W[4]               # 由俯仰力矩确定尾部推进器力
        u1 = W[0]/2 - (1/(2*self.l1))*W[5]    # 由X力和偏航力矩确定
        u4 = W[0]/2 + (1/(2*self.l1))*W[5]    # 由X力和偏航力矩确定
        u2 = (W[2]/2) + (1/(2*self.l1))*W[3]  # 由Z力和滚转力矩确定
        u5 = (W[2]/2) - (1/(2*self.l1))*W[3]  # 由Z力和滚转力矩确定

        # 步骤2：利用alpha1 = alpha2约束求解u3 = t
        C1 = u1**2 + u2**2  # F1² = C1 + t²（不含t的常数项）
        C2 = u4**2 + u5**2  # F2² = C2 + (W2 + t)²（不含t的常数项，W2 = W(2)）
        W2 = W[1]
        
        # 由alpha1 = alpha2推导的t的两个可能解（线性方程解）
        sqrtC1 = np.sqrt(C1)
        sqrtC2 = np.sqrt(C2)
        
        # 避免分母为0（物理意义上C1、C2通常非零，因u1,u2等不全为0）
        if abs(sqrtC2 - sqrtC1) > 1e-10:
            t1 = (W2 * sqrtC1) / (sqrtC2 - sqrtC1)  # 解1
        else:
            t1 = np.nan  # 解1无效
        
        t2 = (-W2 * sqrtC1) / (sqrtC2 + sqrtC1)      # 解2（分母恒正，有效）
        
        # 选择合理的t（优先保证F1、F2非负且角度在合理范围，此处选t2，可根据场景调整）
        t_candidates = [t1, t2]
        # 剔除无效解（NaN）
        t_candidates = [t for t in t_candidates if not np.isnan(t)]
        
        # 验证候选t，选择使alpha1=alpha2且F1、F2非负的解
        t_selected = None
        for t_test in t_candidates:
            u3_test = t_test
            u6_test = -W2 - t_test
            
            F1_test = np.sqrt(C1 + u3_test**2)
            F2_test = np.sqrt(C2 + u6_test**2)
            
            # 避免除以0（物理上力不为0）
            if F1_test < 1e-10 or F2_test < 1e-10:
                continue
            
            # 验证sin(theta1)是否等于sin(theta2)（考虑数值误差）
            sin_theta1 = u3_test / F1_test
            sin_theta2 = u6_test / F2_test
            if abs(sin_theta1 - sin_theta2) < 1e-6:
                t_selected = t_test
                break
        
        # 若未找到有效解，默认用t2（工程上通常有效）
        if t_selected is None:
            t_selected = t2
        
        # 步骤3：确定u3和u6
        u3 = t_selected
        u6 = -W2 - u3
        
        # 步骤4：反推uu的7个参数（保证alpha1=alpha2）
        # F3
        F3 = u7
        
        # F1, theta1, alpha1
        F1 = np.sqrt(C1 + u3**2)
        alpha1 = np.arctan2(u1, u2)  # 角度范围[-π, π]，使用arctan2函数
        theta1 = np.arcsin(u3 / F1)   # theta1 = theta2
        
        # F2, theta2, alpha2
        F2 = np.sqrt(C2 + u6**2)
        alpha2 = np.arctan2(u4, u5)  # 角度范围[-π, π]
        theta2 = np.arcsin(u3 / F1)   # theta1 = theta2
        
        # 组合输出
        uu = np.array([F1, F2, F3, alpha1, alpha2, theta1, theta2])
        return uu
    
    def allocate_actuators(self, f_c_body: np.ndarray, tau_c: np.ndarray, state: dict = None):
        """分配执行器命令（使用非线性逆映射）"""
        # 构造控制向量W（6×1向量）
        # W = [Fx, Fy, Fz, τx, τy, τz]
        W = np.array([
            f_c_body[0],    # X力
            f_c_body[1],    # Y力（通常为0，但保留）
            f_c_body[2],    # Z力
            tau_c[0],       # 滚转力矩
            tau_c[1],       # 俯仰力矩
            tau_c[2]        # 偏航力矩
        ])
        
        # 使用非线性逆映射
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
        T_max = 50
        F1 = np.clip(F1, 0, T_max)
        F2 = np.clip(F2, 0, T_max)
        F3 = np.clip(F3, -10, 10)
        
        # 角度限制（±85度）
        alpha_max = np.radians(90)
        alpha1 = np.clip(alpha1, -alpha_max, alpha_max)
        alpha2 = np.clip(alpha2, -alpha_max, alpha_max)
        theta_max = np.radians(90)
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
        
        # 存储控制输入向量
        self.u = np.array([F1, F2, F3, alpha1, alpha2, theta2, theta2])
        
        return F1, F2, F3, alpha1, alpha2, theta1, theta2
    
    def set_actuators(self, T12: float, T34: float, T5: float, alpha1: float, alpha2: float, theta1: float, theta2: float):
        """应用控制命令到执行器"""
        try:
            # 设置机臂偏航角度
            if 'arm_pitch_right' in self.actuator_ids:
                arm_pitch_right_id = self.actuator_ids['arm_pitch_right']
                self.data.ctrl[arm_pitch_right_id] = alpha1
            
            if 'arm_pitch_left' in self.actuator_ids:
                arm_pitch_left_id = self.actuator_ids['arm_pitch_left']
                self.data.ctrl[arm_pitch_left_id] = alpha2
            
            # 设置螺旋桨倾转角度
            if 'prop_tilt_right' in self.actuator_ids:
                tilt_right_id = self.actuator_ids['prop_tilt_right']
                self.data.ctrl[tilt_right_id] = theta1
            
            if 'prop_tilt_left' in self.actuator_ids:
                tilt_left_id = self.actuator_ids['prop_tilt_left']
                self.data.ctrl[tilt_left_id] = theta2
            
            # 设置推力
            # 右侧两个螺旋桨（每个推力为总推力的一半）
            if 'motor_r_upper' in self.actuator_ids:
                thrust_rt_id = self.actuator_ids['motor_r_upper']
                self.data.ctrl[thrust_rt_id] = T34 / 2
            
            if 'motor_r_lower' in self.actuator_ids:
                thrust_rb_id = self.actuator_ids['motor_r_lower']
                self.data.ctrl[thrust_rb_id] = T34 / 2
            
            # 左侧两个螺旋桨
            if 'motor_l_upper' in self.actuator_ids:
                thrust_lt_id = self.actuator_ids['motor_l_upper']
                self.data.ctrl[thrust_lt_id] = T12 / 2
            
            if 'motor_l_lower' in self.actuator_ids:
                thrust_lb_id = self.actuator_ids['motor_l_lower']
                self.data.ctrl[thrust_lb_id] = T12 / 2
            
            # 尾部推进器
            if 'motor_rear_upper' in self.actuator_ids:
                thrust_tail_id = self.actuator_ids['motor_rear_upper']
                self.data.ctrl[thrust_tail_id] = T5
                
        except Exception as e:
            print(f"设置执行器失败: {e}")
    
    def update_control(self, f_c_body=None, tau_c=None):
        """简化的控制更新函数，直接接受期望力和力矩"""
        try:
            # 如果未提供力和力矩，使用默认值
            if f_c_body is None:
                f_c_body = np.array([0.0, 0.0, 40.0])  # 默认Z方向力
            if tau_c is None:
                tau_c = np.zeros(3)  # 默认零力矩
            
            # 分配执行器命令
            T12, T34, T5, alpha1, alpha2, theta1, theta2 = self.allocate_actuators(f_c_body, tau_c)
            
            # 应用控制
            self.set_actuators(T12, T34, T5, alpha1, alpha2, theta1, theta2)
            
            return True
        except Exception as e:
            print(f"控制更新失败: {e}")
            return False
    
    def print_status(self):
        """简化的状态打印函数，只显示执行器相关信息"""
        try:
            print(f"执行器状态: T12={self.T12:.2f}N, T34={self.T34:.2f}N, T5={self.T5:.2f}N")
            print(f"倾转角度: α1={math.degrees(self.alpha1):.2f}°, α2={math.degrees(self.alpha2):.2f}°, θ1={math.degrees(self.theta1):.2f}°, θ2={math.degrees(self.theta2):.2f}°")
            print(f"执行器控制信号: {self.data.ctrl}")
            print("--------------------------------------------------")
        except Exception as e:
            print(f"状态打印失败: {e}")

def test_force_to_actuators():
    """测试函数：将期望力和力矩转换为执行器控制量"""
    print("=== 期望力和力矩到执行器控制量测试 ===")
    
    # 初始化控制器
    controller = HnuterController()
    
    # 测试用例1: 悬停状态
    print("\n测试用例1: 悬停状态")
    f_c_body = np.array([0.0, 0.0, 20.0])  # Z方向力
    tau_c = np.zeros(3)  # 零力矩
    controller.update_control(f_c_body, tau_c)
    controller.print_status()
    
    # 测试用例2: 前进运动
    print("\n测试用例2: 前进运动")
    f_c_body = np.array([5.0, 0.0, 20.0])  # X方向有向前力
    tau_c = np.zeros(3)
    controller.update_control(f_c_body, tau_c)
    controller.print_status()
    
    # 测试用例3: 滚转运动
    print("\n测试用例3: 滚转运动")
    f_c_body = np.array([0.0, 0.0, 0.0])
    tau_c = np.array([1.0, 0.0, 0.0])  # 滚转力矩
    controller.update_control(f_c_body, tau_c)
    controller.print_status()
    
    # 测试用例4: 偏航运动
    print("\n测试用例4: 偏航运动")
    f_c_body = np.array([0.0, 0.0, 0.0])
    tau_c = np.array([0.0, 0.0, 0.5])  # 偏航力矩
    controller.update_control(f_c_body, tau_c)
    controller.print_status()

    # 测试用例5: 俯仰运动
    print("\n测试用例5: 俯仰运动")
    f_c_body = np.array([0.0, 0.0, 0.0])
    tau_c = np.array([0.0, 0.5, 0.0])  # 偏航力矩
    controller.update_control(f_c_body, tau_c)
    controller.print_status()

    # 测试用例5: 复杂运动
    print("\n测试用例6: 复杂运动")
    f_c_body = np.array([3.0, 2.0, 25.0])
    tau_c = np.array([0.5, 0.3, 0.2])
    controller.update_control(f_c_body, tau_c)
    controller.print_status()

if __name__ == "__main__":
    # 运行测试函数
    test_force_to_actuators()