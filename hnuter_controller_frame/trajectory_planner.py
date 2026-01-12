import numpy as np
import sys
import os
from typing import Dict, Any, Tuple

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class TrajectoryPlanner:
    def __init__(self):
        # ========== 核心参数：90°大角度轨迹控制 ==========  
        self.trajectory_phase = 0  # 阶段划分更细致
        self.attitude_target_rad = np.pi/2  # 目标姿态角度（90度转弧度，与hnuter69一致）
        self.phase_start_time = 0.0  # 各阶段起始时间
        self.attitude_tolerance = 0.08  # 90°大角度下适度放宽tolerance（弧度）
        
        # 初始化旋转矩阵变量
        self.R_des_prev = np.eye(3)  # 上一时刻的目标旋转矩阵
        
        # 阶段时长配置（90°大角度专属）
        self.phase_durations = {
            0: 6.0,    # 起飞悬停（延长到6秒，确保高度稳定）
            1: 12.0,   # Roll转动（12秒，90°大角度缓慢变化）
            2: 5.0,    # Roll保持（5秒，稳定90°姿态）
            3: 6.0,    # Roll恢复（6秒，平稳回零）
            4: 12.0,   # Pitch转动（12秒）
            5: 5.0,    # Pitch保持（5秒）
            6: 6.0,    # Pitch恢复（6秒）
            7: 12.0,   # Yaw转动（12秒）
            8: 5.0,    # Yaw保持（5秒）
            9: 6.0,    # Yaw恢复（6秒）
            10: float('inf')  # 最终悬停
        }
        
        # 阶段名称映射
        self.phase_names = {
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
        
        print("轨迹规划模块初始化完成")
    
    def update_trajectory(self, current_time: float) -> Dict[str, Any]:
        """
        适配90°大角度的轨迹发布器 - 使用旋转矩阵进行轨迹生成和插值
        
        Args:
            current_time: 当前仿真时间（秒）
            
        Returns:
            包含目标状态的字典，使用旋转矩阵表示目标姿态
        """
        # 初始化阶段起始时间
        if self.trajectory_phase == 0 and self.phase_start_time == 0.0:
            self.phase_start_time = current_time
        
        # 计算当前阶段已运行时间
        phase_elapsed = current_time - self.phase_start_time
        
        # 阶段切换判断
        if phase_elapsed > self.phase_durations[self.trajectory_phase]:
            self.trajectory_phase += 1
            self.phase_start_time = current_time
            phase_name = self.phase_names.get(self.trajectory_phase, "未知阶段")
            print(f"\n🔄 轨迹阶段切换: {self.trajectory_phase-1} → {self.trajectory_phase} ({phase_name})")
            # 关键修复：阶段切换后重新计算当前阶段已运行时间
            phase_elapsed = current_time - self.phase_start_time
        
        # 目标状态
        target_position = np.array([0.0, 0.0, 2.0])
        
        # 导入必要的函数
        from utils import rotation_matrix_roll, rotation_matrix_pitch, rotation_matrix_yaw
        
        # 各阶段轨迹逻辑 - 使用旋转矩阵
        if self.trajectory_phase == 0:
            # 阶段0：起飞悬停
            target_position = np.array([0.0, 0.0, 2.0])
            # 悬停状态，目标姿态为单位矩阵
            R_des = np.eye(3)
            
        elif self.trajectory_phase == 1:
            # 阶段1：Roll缓慢转动（0°→90°）
            progress = phase_elapsed / self.phase_durations[1]
            progress = np.clip(progress, 0.0, 1.0)
            target_position = np.array([0.0, 0.0, 2.0])
            # 使用旋转矩阵表示Roll转动
            roll_angle = progress * self.attitude_target_rad * 0.8
            R_des = rotation_matrix_roll(roll_angle)
            
        elif self.trajectory_phase == 2:
            # 阶段2：Roll保持（稳定90°姿态）
            target_position = np.array([0.0, 0.0, 2.0])
            # 保持稳定Roll姿态
            R_des = rotation_matrix_roll(self.attitude_target_rad * 0.8)
            
        elif self.trajectory_phase == 3:
            # 阶段3：Roll恢复（90°→0°）
            progress = phase_elapsed / self.phase_durations[3]
            progress = np.clip(progress, 0.0, 1.0)
            target_position = np.array([0.0, 0.0, 2.0])
            # 使用旋转矩阵表示Roll恢复
            roll_angle = (1 - progress) * self.attitude_target_rad  * 0.8
            R_des = rotation_matrix_roll(roll_angle)
            
        elif self.trajectory_phase == 4:
            # 阶段4：Pitch缓慢转动（0°→90°）
            progress = phase_elapsed / self.phase_durations[4]
            progress = np.clip(progress, 0.0, 1.0)
            target_position = np.array([0.0, 0.0, 2.0])
            # 使用旋转矩阵表示Pitch转动
            pitch_angle = progress * self.attitude_target_rad
            R_des = rotation_matrix_pitch(pitch_angle)
            
        elif self.trajectory_phase == 5:
            # 阶段5：Pitch保持（稳定90°姿态）
            target_position = np.array([0.0, 0.0, 2.0])
            # 保持稳定Pitch姿态
            R_des = rotation_matrix_pitch(self.attitude_target_rad)
            
        elif self.trajectory_phase == 6:
            # 阶段6：Pitch恢复（90°→0°）
            progress = phase_elapsed / self.phase_durations[6]
            progress = np.clip(progress, 0.0, 1.0)
            target_position = np.array([0.0, 0.0, 2.0])
            # 使用旋转矩阵表示Pitch恢复
            pitch_angle = (1 - progress) * self.attitude_target_rad
            R_des = rotation_matrix_pitch(pitch_angle)
            
        elif self.trajectory_phase == 7:
            # 阶段7：Yaw缓慢转动（0°→90°）
            progress = phase_elapsed / self.phase_durations[7]
            progress = np.clip(progress, 0.0, 1.0)
            target_position = np.array([0.0, 0.0, 2.0])
            # 使用旋转矩阵表示Yaw转动
            yaw_angle = progress * self.attitude_target_rad
            R_des = rotation_matrix_yaw(yaw_angle)
            
        elif self.trajectory_phase == 8:
            # 阶段8：Yaw保持（稳定90°姿态）
            target_position = np.array([0.0, 0.0, 2.0])
            # 保持稳定Yaw姿态
            R_des = rotation_matrix_yaw(self.attitude_target_rad)
            
        elif self.trajectory_phase == 9:
            # 阶段9：Yaw恢复（90°→0°）
            progress = phase_elapsed / self.phase_durations[9]
            progress = np.clip(progress, 0.0, 1.0)
            target_position = np.array([0.0, 0.0, 2.0])
            # 使用旋转矩阵表示Yaw恢复
            yaw_angle = (1 - progress) * self.attitude_target_rad
            R_des = rotation_matrix_yaw(yaw_angle)
            
        else:
            # 阶段10：最终悬停
            target_position = np.array([0.0, 0.0, 2.0])
            # 最终悬停状态，目标姿态为单位矩阵
            R_des = np.eye(3)
        
        # 平滑插值：使用Slerp（球面线性插值）确保旋转矩阵平滑过渡
        from utils import slerp
        # 使用自定义的slerp函数进行插值，t=1.0表示直接使用当前目标
        R_des = slerp(self.R_des_prev, R_des, 1.0)
        
        # 更新上一时刻的目标旋转矩阵
        self.R_des_prev = R_des.copy()
        
        # 返回目标状态，使用旋转矩阵表示目标姿态
        return {
            'target_position': target_position,
            'target_rotation_matrix': R_des,
            'target_velocity': np.zeros(3),
            'target_acceleration': np.zeros(3),
            'target_attitude_rate': np.zeros(3),
            'target_attitude_acceleration': np.zeros(3),
            'trajectory_phase': self.trajectory_phase
        }
    
    def reset_trajectory(self):
        """重置轨迹规划器"""
        self.trajectory_phase = 0
        self.phase_start_time = 0.0
        print("轨迹已重置")
    
    def get_current_phase(self) -> Tuple[int, str]:
        """获取当前轨迹阶段和名称"""
        phase_name = self.phase_names.get(self.trajectory_phase, "未知阶段")
        return self.trajectory_phase, phase_name
    
    def get_phase_info(self, phase: int) -> Dict[str, Any]:
        """获取指定阶段的信息"""
        return {
            'duration': self.phase_durations.get(phase, 0.0),
            'name': self.phase_names.get(phase, "未知阶段")
        }
    
    def is_trajectory_complete(self) -> bool:
        """判断轨迹是否完成"""
        return self.trajectory_phase >= 10
    
    def set_attitude_target(self, target_angle_deg: float):
        """设置目标姿态角度（度）"""
        self.attitude_target_rad = np.radians(target_angle_deg)
        print(f"目标姿态角度已设置为: {target_angle_deg}°")
    
    def set_phase_duration(self, phase: int, duration: float):
        """设置指定阶段的持续时间"""
        if phase in self.phase_durations:
            self.phase_durations[phase] = duration
            print(f"阶段 {phase} 的持续时间已设置为: {duration}s")
        else:
            print(f"阶段 {phase} 不存在")
