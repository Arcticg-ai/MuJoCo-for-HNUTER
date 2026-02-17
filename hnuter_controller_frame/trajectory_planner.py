import numpy as np
import sys
import os
from typing import Dict, Any, Tuple

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class TrajectoryPlanner:
    def __init__(self, trajectory_type="default"):
        # ========== 轨迹类型选择 ==========
        self.trajectory_type = trajectory_type  # 轨迹类型："default" 或 "lissajous"
        
        # ========== 核心参数：90°大角度轨迹控制 ==========  
        self.trajectory_phase = 0  # 阶段划分更细致
        self.attitude_target_rad = np.pi/2  # 目标姿态角度
        self.phase_start_time = 0.0  # 各阶段起始时间
        self.attitude_tolerance = 0.08  # 90°大角度下适度放宽tolerance（弧度）
        
        # 初始化旋转矩阵变量
        self.R_des_prev = np.eye(3)  # 上一时刻的目标旋转矩阵
        
        # ========== 轨迹参数配置 ==========
        # 李萨如曲线参数（调整为更平缓的设置）
        self.lissajous_params = {
            'A_x': 2.0,    # X方向振幅（增大）
            'A_y': 2.0,    # Y方向振幅（增大）
            'A_z': 0.5,    # Z方向振幅（增大）
            'a': 1,        # X方向频率（降低）
            'b': 1,        # Y方向频率（降低）
            'c': 0.3,      # Z方向频率（降低）
            'phi': 0,      # X方向相位
            'psi': np.pi/2, # Y方向相位
            'omega': 0,    # Z方向相位
            'speed': 0.15,  # 曲线跟踪速度（降低）
            'pitch_amplitude': np.pi/12,  # 俯仰跟随曲线变化的振幅（15度，减小）
            'yaw_rate': 0.1  # 偏航角速度（降低）
        }
        
        # 螺旋上升轨迹参数
        self.spiral_params = {
            'radius': 3.0,     # 螺旋半径（增大）
            'ascend_rate': 0.1, # 上升速度（m/s，降低）
            'angular_speed': 0.2, # 角速度（rad/s，降低）
            'pitch_amplitude': np.pi/12,  # 俯仰振幅（15度）
            'yaw_rate': 0.1   # 偏航角速度（降低）
        }
        
        # 环形轨迹参数
        self.ring_params = {
            'radius': 3.0,     # 环半径（增大）
            'angular_speed': 0.2, # 角速度（rad/s，降低）
            'pitch_amplitude': np.pi/12,  # 俯仰振幅（15度）
            'yaw_rate': 0.1   # 偏航角速度（降低）
        }
        
        # 矩形轨迹参数（无俯仰变化，逐圈加速版本）
        self.rectangle_params = {
            'width': 16.0,      # 矩形宽度（X方向）
            'height': 16.0,     # 矩形高度（Y方向）
            'base_speed': 2.0,   # 基础速度（初始速度）
            'speed_increment': 0.5,  # 每圈速度增量
            'max_speed': 5.0,    # 最大速度限制
            'yaw_rate': 0.5,    # 偏航角速度（度/s）
            'continuous_flight': True  # 持续飞行，不停歇
        }
        
        # 加速测试轨迹参数
        self.acceleration_params = {
            'takeoff_hover_time': 5.0,  # 起飞悬停时间（秒）
            'max_forward_acceleration': 2.0,  # 最大前向加速度（m/s²）
            'max_backward_acceleration': -2.5,  # 最大反向加速度（m/s²，负数表示减速/反推）
            'target_speed': 20.0,  # 目标速度（m/s）
            'height': 2.0  # 飞行高度（m）
        }
        
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
        # 导入必要的函数
        from utils import rotation_matrix_roll, rotation_matrix_pitch, rotation_matrix_yaw
        
        # 目标状态初始化
        target_position = np.array([0.0, 0.0, 2.0])
        R_des = np.eye(3)
        target_velocity = np.zeros(3)
        target_acceleration = np.zeros(3)
        target_attitude_rate = np.zeros(3)
        target_attitude_acceleration = np.zeros(3)
        
        # 起飞悬停阶段设置
        takeoff_hover_duration = 5.0  # 起飞悬停时间（秒）
        
        # 轨迹生成逻辑
        if current_time < takeoff_hover_duration:
            # 起飞悬停阶段：保持位置在原点上方2米，姿态为水平
            target_position = np.array([0.0, 0.0, 2.0])
            R_des = np.eye(3)  # 水平姿态
            target_velocity = np.zeros(3)
            target_acceleration = np.zeros(3)
        elif self.trajectory_type == "lissajous":
            # 李萨如曲线轨迹（平滑版本）
            params = self.lissajous_params
            
            # 计算时间参数（起飞悬停后开始计算轨迹时间）
            t = (current_time - takeoff_hover_duration) * params['speed']
            
            # 生成位置
            target_position = np.array([
                params['A_x'] * np.sin(params['a'] * t + params['phi']),
                params['A_y'] * np.sin(params['b'] * t + params['psi']),
                2.0 + params['A_z'] * np.sin(params['c'] * t + params['omega'])
            ])
            
            # 计算期望速度（平滑变化）
            target_velocity = np.array([
                params['A_x'] * params['a'] * params['speed'] * np.cos(params['a'] * t + params['phi']),
                params['A_y'] * params['b'] * params['speed'] * np.cos(params['b'] * t + params['psi']),
                params['A_z'] * params['c'] * params['speed'] * np.cos(params['c'] * t + params['omega'])
            ])
            
            # 计算期望加速度（平滑变化）
            target_acceleration = np.array([
                -params['A_x'] * params['a']**2 * params['speed']**2 * np.sin(params['a'] * t + params['phi']),
                -params['A_y'] * params['b']**2 * params['speed']**2 * np.sin(params['b'] * t + params['psi']),
                -params['A_z'] * params['c']**2 * params['speed']**2 * np.sin(params['c'] * t + params['omega'])
            ])
            
            # 计算平滑的俯仰角变化
            pitch_angle = params['pitch_amplitude'] * np.sin(params['a'] * t + params['phi'])
            
            # 计算平滑的偏航角变化
            yaw_angle = params['yaw_rate'] * t
            
            # 生成旋转矩阵：先偏航，再俯仰，最后横滚（横滚为0）
            R_yaw = rotation_matrix_yaw(yaw_angle)
            R_pitch = rotation_matrix_pitch(pitch_angle)
            R_roll = rotation_matrix_roll(0.0)
            
            # 组合旋转矩阵
            R_des = R_yaw @ R_pitch @ R_roll
            
        elif self.trajectory_type == "spiral":
            # 螺旋上升轨迹
            params = self.spiral_params
            
            # 计算时间参数（起飞悬停后开始计算轨迹时间）
            t = current_time - takeoff_hover_duration
            
            # 生成螺旋位置
            # 水平平面上的圆运动
            horizontal_angle = params['angular_speed'] * t
            target_position = np.array([
                params['radius'] * np.cos(horizontal_angle),
                params['radius'] * np.sin(horizontal_angle),
                2.0 + params['ascend_rate'] * t  # 缓慢上升
            ])
            
            # 计算期望速度（螺旋运动的速度分量）
            target_velocity = np.array([
                -params['radius'] * params['angular_speed'] * np.sin(horizontal_angle),
                params['radius'] * params['angular_speed'] * np.cos(horizontal_angle),
                params['ascend_rate']
            ])
            
            # 计算期望加速度
            target_acceleration = np.array([
                -params['radius'] * params['angular_speed']**2 * np.cos(horizontal_angle),
                -params['radius'] * params['angular_speed']**2 * np.sin(horizontal_angle),
                0.0
            ])
            
            # 平滑的姿态变化
            # 螺旋飞行时，俯仰角保持较小值，偏航角跟随螺旋方向
            pitch_angle = params['pitch_amplitude'] * np.sin(params['angular_speed'] * t / 2)
            yaw_angle = horizontal_angle  # 偏航角跟随螺旋方向
            
            # 生成旋转矩阵：先偏航，再俯仰，最后横滚（横滚为0）
            R_yaw = rotation_matrix_yaw(yaw_angle)
            R_pitch = rotation_matrix_pitch(pitch_angle)
            R_roll = rotation_matrix_roll(0.0)
            
            # 组合旋转矩阵
            R_des = R_yaw @ R_pitch @ R_roll
            
        elif self.trajectory_type == "ring":
            # 环形轨迹
            params = self.ring_params
            
            # 计算时间参数（起飞悬停后开始计算轨迹时间）
            t = current_time - takeoff_hover_duration
            
            # 生成环形位置（水平平面上的圆运动，高度保持不变）
            ring_angle = params['angular_speed'] * t
            target_position = np.array([
                params['radius'] * np.cos(ring_angle),  # 水平方向X轴
                params['radius'] * np.sin(ring_angle),  # 水平方向Y轴
                2.0  # 保持高度不变
            ])
            
            # 计算期望速度（环形运动的切线速度）
            target_velocity = np.array([
                -params['radius'] * params['angular_speed'] * np.sin(ring_angle),
                params['radius'] * params['angular_speed'] * np.cos(ring_angle),
                0.0
            ])
            
            # 计算期望加速度（向心加速度）
            target_acceleration = np.array([
                -params['radius'] * params['angular_speed']**2 * np.cos(ring_angle),
                -params['radius'] * params['angular_speed']**2 * np.sin(ring_angle),
                0.0
            ])
            
            # 平滑的姿态变化
            # 环形飞行时，俯仰角保持较小值，偏航角跟随环形方向
            pitch_angle = params['pitch_amplitude'] * np.sin(params['angular_speed'] * t / 2)
            yaw_angle = ring_angle  # 偏航角跟随环形方向
            
            # 生成旋转矩阵：先偏航，再俯仰，最后横滚（横滚为0）
            R_yaw = rotation_matrix_yaw(yaw_angle)
            R_pitch = rotation_matrix_pitch(pitch_angle)
            R_roll = rotation_matrix_roll(0.0)
            
            # 组合旋转矩阵
            R_des = R_yaw @ R_pitch @ R_roll
            
        elif self.trajectory_type == "rectangle":
            # 矩形轨迹：起飞→逐圈加速飞水平矩形（无俯仰变化，不停歇）
            params = self.rectangle_params
            
            # 计算各个阶段的时间点
            t_takeoff = takeoff_hover_duration
            
            # 根据当前时间确定阶段
            if current_time < t_takeoff:
                # 起飞悬停阶段：保持位置在原点上方2米，姿态为水平
                target_position = np.array([0.0, 0.0, 2.0])
                R_des = np.eye(3)  # 水平姿态
                target_velocity = np.zeros(3)
                target_acceleration = np.zeros(3)
            else:
                # 飞逐圈加速水平矩形阶段（持续飞行，不停歇）
                # 计算矩形飞行时间
                rectangle_time = current_time - t_takeoff
                
                # 矩形周长
                rectangle_perimeter = 2 * (params['width'] + params['height'])
                
                # 计算当前速度（逐圈加速）
                # 计算当前圈数
                # 注意：这里需要动态计算当前圈数，因为速度在变化
                # 我们需要模拟每圈的速度变化，计算累计飞行距离和时间
                accumulated_distance = 0.0
                accumulated_time = 0.0
                current_speed = params['base_speed']
                current_cycle = 0
                
                # 模拟每圈的飞行，直到找到当前所在的圈数和速度
                while True:
                    # 当前圈的飞行时间
                    cycle_flight_time = rectangle_perimeter / current_speed
                    # 如果当前时间在这个圈内
                    if accumulated_time + cycle_flight_time >= rectangle_time:
                        break
                    # 否则累加距离和时间，速度增加
                    accumulated_distance += rectangle_perimeter
                    accumulated_time += cycle_flight_time
                    current_cycle += 1
                    # 计算下一圈的速度，不超过最大速度
                    current_speed = min(
                        params['base_speed'] + params['speed_increment'] * current_cycle,
                        params['max_speed']
                    )
                
                # 计算当前圈的相对时间
                cycle_time = rectangle_time - accumulated_time
                
                # 矩形飞行路径：
                # 1. 从原点沿X轴飞行width/2
                # 2. 沿Y轴飞行height
                # 3. 沿-X轴飞行width
                # 4. 沿-Y轴飞行height
                # 5. 沿X轴飞行width/2，回到原点
                
                # 计算当前位置
                segment_times = []
                segment_distances = [
                    params['width'] / 2,  # 阶段1：X轴正方向
                    params['height'],      # 阶段2：Y轴正方向
                    params['width'],       # 阶段3：X轴负方向
                    params['height'],      # 阶段4：Y轴负方向
                    params['width'] / 2     # 阶段5：X轴正方向回到原点
                ]
                
                # 计算每个阶段的飞行时间
                for distance in segment_distances:
                    segment_times.append(distance / current_speed)
                
                # 计算当前所在的阶段
                cumulative_time = 0.0
                current_segment = 0
                for i, segment_time in enumerate(segment_times):
                    if cumulative_time + segment_time > cycle_time:
                        current_segment = i
                        break
                    cumulative_time += segment_time
                
                # 计算当前阶段的相对时间
                segment_relative_time = cycle_time - cumulative_time
                
                # 计算当前位置和速度
                if current_segment == 0:
                    # 阶段1：沿X轴正方向飞行width/2
                    x = current_speed * segment_relative_time
                    y = 0.0
                    target_velocity = np.array([current_speed, 0.0, 0.0])
                elif current_segment == 1:
                    # 阶段2：沿Y轴正方向飞行height
                    x = params['width'] / 2
                    y = current_speed * segment_relative_time
                    target_velocity = np.array([0.0, current_speed, 0.0])
                elif current_segment == 2:
                    # 阶段3：沿X轴负方向飞行width
                    x = params['width'] / 2 - current_speed * segment_relative_time
                    y = params['height']
                    target_velocity = np.array([-current_speed, 0.0, 0.0])
                elif current_segment == 3:
                    # 阶段4：沿Y轴负方向飞行height
                    x = -params['width'] / 2
                    y = params['height'] - current_speed * segment_relative_time
                    target_velocity = np.array([0.0, -current_speed, 0.0])
                else:  # current_segment == 4
                    # 阶段5：沿X轴正方向飞行width/2，回到原点
                    x = -params['width'] / 2 + current_speed * segment_relative_time
                    y = 0.0
                    target_velocity = np.array([current_speed, 0.0, 0.0])
                
                target_position = np.array([x, y, 2.0])  # 保持高度不变
                R_des = np.eye(3)  # 保持水平姿态，无俯仰变化
                target_acceleration = np.zeros(3)  # 假设匀速飞行
            
        elif self.trajectory_type == "acceleration":
            # 加速测试轨迹：起飞→加速到30m/s→反推减速→反向加速→减速回到原点
            params = self.acceleration_params
            
            # 计算各个阶段的时间点和位置
            # 阶段1：起飞悬停
            t_stage1_end = takeoff_hover_duration
            
            # 阶段2：前向加速到目标速度
            # 加速时间 = 目标速度 / 最大加速度
            t_accel = params['target_speed'] / params['max_forward_acceleration']
            t_stage2_end = t_stage1_end + t_accel
            # 加速阶段结束时的位置
            x_stage2_end = 0.5 * params['max_forward_acceleration'] * t_accel**2
            
            # 阶段3：反推减速到0
            # 减速时间 = 当前速度 / 减速度大小
            t_decel = params['target_speed'] / abs(params['max_backward_acceleration'])
            t_stage3_end = t_stage2_end + t_decel
            # 减速阶段结束时的位置
            x_stage3_end = x_stage2_end + params['target_speed'] * t_decel + 0.5 * params['max_backward_acceleration'] * t_decel**2
            
            # 阶段4：反向加速
            # 反向加速到与前向相同的最大速度
            reverse_speed = params['target_speed'] * 0.8  # 反向速度设为前向的80%
            t_reverse_accel = reverse_speed / abs(params['max_backward_acceleration'])
            t_stage4_end = t_stage3_end + t_reverse_accel
            # 反向加速阶段结束时的位置
            x_stage4_end = x_stage3_end + 0.5 * params['max_backward_acceleration'] * t_reverse_accel**2
            
            # 阶段5：减速回到原点
            # 计算需要的减速距离
            distance_to_origin = abs(x_stage4_end)
            # 计算减速所需的时间
            # 使用公式：distance = v0*t + 0.5*a*t²，其中v0是反向速度，a是正加速度（减速）
            # 解方程：distance_to_origin = reverse_speed*t + 0.5*max_forward_acceleration*t²
            # 使用求根公式：t = [-b + sqrt(b² + 2*a*c)] / a，其中a=max_forward_acceleration, b=reverse_speed, c=distance_to_origin
            a = params['max_forward_acceleration']
            b = reverse_speed
            c = distance_to_origin
            t_reverse_decel = (-b + np.sqrt(b**2 + 2*a*c)) / a
            t_stage5_end = t_stage4_end + t_reverse_decel
            
            # 根据当前时间确定阶段
            if current_time < t_stage1_end:
                # 阶段1：起飞悬停
                target_position = np.array([0.0, 0.0, params['height']])
                R_des = np.eye(3)  # 水平姿态
                target_velocity = np.zeros(3)
                target_acceleration = np.zeros(3)
            elif current_time < t_stage2_end:
                # 阶段2：前向加速到30m/s
                t = current_time - t_stage1_end
                # 位置：x = 0.5*a*t²
                x = 0.5 * params['max_forward_acceleration'] * t**2
                # 速度：v = a*t
                v = params['max_forward_acceleration'] * t
                # 加速度：保持最大前向加速度
                a_target = params['max_forward_acceleration']
                
                target_position = np.array([x, 0.0, params['height']])
                target_velocity = np.array([v, 0.0, 0.0])
                target_acceleration = np.array([a_target, 0.0, 0.0])
                R_des = np.eye(3)  # 水平姿态
            elif current_time < t_stage3_end:
                # 阶段3：反推减速到0
                t = current_time - t_stage2_end
                # 初始速度：v0 = 30m/s
                v0 = params['target_speed']
                # 减速度：a = max_backward_acceleration（负数）
                a = params['max_backward_acceleration']
                # 位置：x = x_stage2_end + v0*t + 0.5*a*t²
                x = x_stage2_end + v0 * t + 0.5 * a * t**2
                # 速度：v = v0 + a*t
                v = v0 + a * t
                # 加速度：保持最大反向加速度
                a_target = a
                
                target_position = np.array([x, 0.0, params['height']])
                target_velocity = np.array([v, 0.0, 0.0])
                target_acceleration = np.array([a_target, 0.0, 0.0])
                R_des = np.eye(3)  # 水平姿态
            elif current_time < t_stage4_end:
                # 阶段4：反向加速
                t = current_time - t_stage3_end
                # 初始速度：v0 = 0
                v0 = 0.0
                # 加速度：a = max_backward_acceleration（负数，反向加速）
                a = params['max_backward_acceleration']
                # 位置：x = x_stage3_end + v0*t + 0.5*a*t²
                x = x_stage3_end + v0 * t + 0.5 * a * t**2
                # 速度：v = v0 + a*t
                v = v0 + a * t
                # 加速度：保持最大反向加速度
                a_target = a
                
                target_position = np.array([x, 0.0, params['height']])
                target_velocity = np.array([v, 0.0, 0.0])
                target_acceleration = np.array([a_target, 0.0, 0.0])
                R_des = np.eye(3)  # 水平姿态
            elif current_time < t_stage5_end:
                # 阶段5：减速回到原点
                t = current_time - t_stage4_end
                # 初始速度：v0 = 反向速度
                v0 = -reverse_speed  # 负号表示反向
                # 加速度：a = max_forward_acceleration（正数，减速）
                a = params['max_forward_acceleration']
                # 位置：x = x_stage4_end + v0*t + 0.5*a*t²
                x = x_stage4_end + v0 * t + 0.5 * a * t**2
                # 速度：v = v0 + a*t
                v = v0 + a * t
                # 加速度：保持最大前向加速度
                a_target = a
                
                target_position = np.array([x, 0.0, params['height']])
                target_velocity = np.array([v, 0.0, 0.0])
                target_acceleration = np.array([a_target, 0.0, 0.0])
                R_des = np.eye(3)  # 水平姿态
            else:
                # 阶段6：保持在原点
                target_position = np.array([0.0, 0.0, params['height']])
                target_velocity = np.zeros(3)
                target_acceleration = np.zeros(3)
                R_des = np.eye(3)  # 水平姿态
            
        else:
            # 默认轨迹逻辑（90°大角度姿态跟踪）
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
        
        # 平滑插值：使用 Slerp 确保旋转矩阵平滑过渡（每步向目标靠拢 50%，收敛速度与步长有关）
        from utils import slerp
        R_des = slerp(self.R_des_prev, R_des, 0.5)
        
        # 更新上一时刻的目标旋转矩阵
        self.R_des_prev = R_des.copy()
        
        # 返回目标状态，使用旋转矩阵表示目标姿态
        return {
            'target_position': target_position,
            'target_rotation_matrix': R_des,
            'target_velocity': target_velocity,
            'target_acceleration': target_acceleration,
            'target_attitude_rate': target_attitude_rate,
            'target_attitude_acceleration': target_attitude_acceleration,
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
