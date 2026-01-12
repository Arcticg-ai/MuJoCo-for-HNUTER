import numpy as np
import csv
import os
from datetime import datetime
from typing import Dict, Any, List
from controller import HnuterController
from utils import euler_to_quaternion


class DroneLogger:
    def __init__(self, controller: HnuterController):
        # 保存控制器实例
        self.controller = controller
        
        # 日志文件路径
        self.log_file: str = ""
        
        # 创建日志文件
        self._create_log_file()
        
        print("日志记录模块初始化完成")
    
    def _create_log_file(self):
        """创建日志文件并写入表头"""
        # 确保logs目录存在
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # 创建带时间戳的文件名，与hnuter69区分
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f'logs/drone_log_controller_frame_{timestamp}.csv'
        
        # 写入CSV表头（新增几何解耦相关字段）
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
                'f_world_x', 'f_world_y', 'f_world_z',
                'f_body_x', 'f_body_y', 'f_body_z',
                'tau_x', 'tau_y', 'tau_z',
                'T12', 'T34', 'T5',
                'alpha1_cmd', 'alpha2_cmd', 'alpha1_actual', 'alpha2_actual',
                'theta1_cmd', 'theta2_cmd', 'theta1_actual', 'theta2_actual',
                'trajectory_phase',
                'is_pitch_exceed',
                'axis_type_roll', 'axis_type_pitch', 'axis_type_yaw',
                'KR_roll', 'KR_pitch', 'KR_yaw'
            ])
        
        print(f"已创建几何解耦控制日志文件: {self.log_file}")
    
    def log_status(self, state: Dict[str, Any], trajectory_phase: int = 0):
        """记录状态到日志文件"""
        import time
        timestamp = time.time()
        position = state.get('position', np.zeros(3))
        euler = state.get('euler', np.zeros(3))
        current_quat = state.get('quaternion', np.array([1.0, 0.0, 0.0, 0.0]))
        # 从旋转矩阵转换到四元数
        from scipy.spatial.transform import Rotation as R
        target_rot = R.from_matrix(self.controller.target_rotation_matrix)
        target_quat = target_rot.as_quat()
        # 从旋转矩阵转换到欧拉角（用于日志记录）
        target_euler = target_rot.as_euler('xyz', degrees=False)
        is_pitch_exceed = state.get('is_pitch_exceed', False)
        
        # 获取实际倾转角度
        # 检查sim是否有get_actual_tilt_angles方法
        actual_angles = {}
        if hasattr(self.controller.sim, 'get_actual_tilt_angles'):
            actual_angles = self.controller.sim.get_actual_tilt_angles()
        actual_angles.setdefault('alpha1_actual', 0.0)
        actual_angles.setdefault('alpha2_actual', 0.0)
        actual_angles.setdefault('theta1_actual', 0.0)
        actual_angles.setdefault('theta2_actual', 0.0)
        
        # 使用控制器的固定增益
        KR_current = self.controller.KR
        
        with open(self.log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                timestamp,
                position[0], position[1], position[2],
                self.controller.target_position[0], self.controller.target_position[1], self.controller.target_position[2],
                euler[0], euler[1], euler[2],
                target_euler[0], target_euler[1], target_euler[2],
                current_quat[0], current_quat[1], current_quat[2], current_quat[3],
                target_quat[0], target_quat[1], target_quat[2], target_quat[3],
                state.get('velocity', [0,0,0])[0], state.get('velocity', [0,0,0])[1], state.get('velocity', [0,0,0])[2],
                state.get('angular_velocity', [0,0,0])[0], state.get('angular_velocity', [0,0,0])[1], state.get('angular_velocity', [0,0,0])[2],
                self.controller.f_c_world[0], self.controller.f_c_world[1], self.controller.f_c_world[2],
                self.controller.f_c_body[0], self.controller.f_c_body[1], self.controller.f_c_body[2],
                self.controller.tau_c[0], self.controller.tau_c[1], self.controller.tau_c[2],
                self.controller.T12, self.controller.T34, self.controller.T5,
                self.controller.alpha1, self.controller.alpha2, actual_angles['alpha1_actual'], actual_angles['alpha2_actual'],
                self.controller.theta1, self.controller.theta2, actual_angles['theta1_actual'], actual_angles['theta2_actual'],
                trajectory_phase,
                int(is_pitch_exceed),
                0, 0, 0,  # 占位符，不再使用轴类型
                KR_current[0], KR_current[1], KR_current[2]
            ])
    
    def print_status(self, trajectory_phase: int = 0):
        """打印当前状态信息"""
        try:
            state = self.controller.sim.get_state()
            pos = state['position']
            euler_deg = np.degrees(state['euler'])
            # 从旋转矩阵转换到欧拉角
            from scipy.spatial.transform import Rotation as R
            target_rot = R.from_matrix(self.controller.target_rotation_matrix)
            target_euler = target_rot.as_euler('xyz', degrees=False)
            target_euler_deg = np.degrees(target_euler)
            
            # 使用默认轴类型（不再使用动态轴类型）
            axis_types = ["fast", "fast", "fast"]
            
            # 使用控制器的固定增益
            KR_current = self.controller.KR
            
            # 阶段名称映射
            phase_names = {
                0: "起飞悬停",
                1: "Roll转动(0°→90°)",
                2: "Roll保持(90°，稳定5s)",
                3: "Roll恢复(90°→0°)",
                4: "Pitch转动(0°→90°)",
                5: "Pitch保持(90°，稳定5s)",
                6: "Pitch恢复(90°→0°)",
                7: "Yaw转动(0°→90°)",
                8: "Yaw保持(90°，稳定5s)",
                9: "Yaw恢复(90°→0°)",
                10: "最终悬停"
            }
            phase_name = phase_names.get(trajectory_phase, "未知阶段")
            
            # 获取实际倾转角度
            actual_angles = self.controller.sim.get_actual_tilt_angles()
            
            print(f"\n=== 轨迹阶段: {trajectory_phase} ({phase_name}) ===")
            print(f"位置: X={pos[0]:.3f}m, Y={pos[1]:.3f}m, Z={pos[2]:.3f}m")
            print(f"目标位置: X={self.controller.target_position[0]:.3f}m, Y={self.controller.target_position[1]:.3f}m, Z={self.controller.target_position[2]:.3f}m")
            print(f"姿态: Roll={euler_deg[0]:.2f}°, Pitch={euler_deg[1]:.2f}°, Yaw={euler_deg[2]:.2f}°")  
            print(f"目标姿态: Roll={target_euler_deg[0]:.1f}°, Pitch={target_euler_deg[1]:.1f}°, Yaw={target_euler_deg[2]:.1f}°")
            print(f"控制力矩: X={self.controller.tau_c[0]:.3f}Nm, Y={self.controller.tau_c[1]:.3f}Nm, Z={self.controller.tau_c[2]:.3f}Nm")
            print(f"执行器状态: T12={self.controller.T12:.2f}N, T34={self.controller.T34:.2f}N, T5={self.controller.T5:.2f}N")
            print(f"机臂偏航: α1={np.degrees(self.controller.alpha1):.2f}°(实际{np.degrees(actual_angles['alpha1_actual']):.2f}°), α2={np.degrees(self.controller.alpha2):.2f}°(实际{np.degrees(actual_angles['alpha2_actual']):.2f}°)")
            print(f"螺旋桨倾转: θ1={np.degrees(self.controller.theta1):.2f}°(实际{np.degrees(actual_angles['theta1_actual']):.2f}°), θ2={np.degrees(self.controller.theta2):.2f}°(实际{np.degrees(actual_angles['theta2_actual']):.2f}°)")
            print(f"轴类型: Roll={axis_types[0]}, Pitch={axis_types[1]}, Yaw={axis_types[2]}")
            print(f"控制增益: KR=[{KR_current[0]:.2f}, {KR_current[1]:.2f}, {KR_current[2]:.2f}]")
            print(f"俯仰角限制: {'超限' if self.controller.sim.is_pitch_exceed else '正常'} (阈值: {self.controller.sim.pitch_threshold_deg}°)")
            print("--------------------------------------------------")
        except Exception as e:
            print(f"状态打印失败: {e}")
    
    def get_log_file_path(self) -> str:
        """获取日志文件路径"""
        return self.log_file
    
    def print_summary(self, final_state: Dict[str, Any]):
        """打印仿真总结"""
        print("\n=== 仿真总结 ===")
        print(f"最终位置: ({final_state['position'][0]:.2f}, {final_state['position'][1]:.2f}, {final_state['position'][2]:.2f})m")
        print(f"最终姿态: Roll={np.degrees(final_state['euler'][0]):.2f}°, Pitch={np.degrees(final_state['euler'][1]):.2f}°, Yaw={np.degrees(final_state['euler'][2]):.2f}°")
        print(f"日志文件: {self.log_file}")
        print(f"轨迹阶段: {self.controller.trajectory_phase}")
        print("=================")
    
    def log_error(self, error_message: str):
        """记录错误信息到日志文件"""
        import time
        timestamp = time.time()
        with open(self.log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([timestamp, f"ERROR: {error_message}"] + ["" for _ in range(59)])
        print(f"错误已记录: {error_message}")
