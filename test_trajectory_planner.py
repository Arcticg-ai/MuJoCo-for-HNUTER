import numpy as np
from hnuter_controller_frame.trajectory_planner import TrajectoryPlanner

# 创建轨迹规划器实例
trajectory_planner = TrajectoryPlanner()

# 测试初始状态
print("测试初始状态：")
target_state = trajectory_planner.update_trajectory(0.0)
print(f"初始旋转矩阵：\n{target_state['target_rotation_matrix']}")

# 测试阶段1（Roll转动）
print("\n测试阶段1（Roll转动）：")
target_state = trajectory_planner.update_trajectory(7.0)  # 进入阶段1
print(f"Roll转动旋转矩阵：\n{target_state['target_rotation_matrix']}")

# 测试阶段2（Roll保持）
print("\n测试阶段2（Roll保持）：")
target_state = trajectory_planner.update_trajectory(20.0)  # 进入阶段2
print(f"Roll保持旋转矩阵：\n{target_state['target_rotation_matrix']}")

# 测试阶段3（Roll恢复）
print("\n测试阶段3（Roll恢复）：")
target_state = trajectory_planner.update_trajectory(26.0)  # 进入阶段3
print(f"Roll恢复旋转矩阵：\n{target_state['target_rotation_matrix']}")

print("\n✅ 轨迹规划器测试完成，没有出现AttributeError错误！")
