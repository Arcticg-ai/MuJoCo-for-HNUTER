import numpy as np
import mujoco.viewer as viewer
import time
from simulation_framework import SimulationFramework
from controller import HnuterController
from allocation import ActuatorAllocation
from logger import DroneLogger
from trajectory_planner import TrajectoryPlanner


def main():
    """主函数 - 启动几何解耦控制仿真"""
    print("=== 倾转旋翼无人机几何解耦控制仿真 ===")
    print("核心优化：基于倾转预测的几何解耦控制方案")
    print("方案特点：")
    print("  1. 虚拟坐标系解耦快慢响应轴")
    print("  2. 基于俯仰角的增益调度")
    print("  3. 舵机动态延迟补偿")
    print("  4. 自适应轴类型切换")
    
    try:
        # 初始化仿真框架
        sim = SimulationFramework("hnuter201.xml")
        
        # 初始化控制器
        controller = HnuterController(sim)
        
        # 初始目标
        controller.target_position = np.array([0.0, 0.0, 2.0])
        controller.target_attitude = np.array([0.0, 0.0, 0.0])
        
        # 初始化执行器分配模块
        actuator_allocation = ActuatorAllocation(controller, sim)
        
        # 初始化日志记录模块
        logger = DroneLogger(controller)
        
        # 初始化轨迹规划器
        trajectory_planner = TrajectoryPlanner()
        
        # 启动 Viewer
        with viewer.launch_passive(sim.model, sim.data) as v:
            print(f"\n仿真启动：")
            print(f"日志文件: {logger.log_file}")
            print("控制指令:")
            print("  r - 重置仿真")
            print("  p - 暂停/继续")
            print("  q - 退出")
            print("按 Ctrl+C 终止仿真")
            
            start_time = time.time()
            last_print_time = 0
            print_interval = 1.0
            paused = False
            
            try:
                while v.is_running():
                    current_time = time.time() - start_time
                    
                    # 检查键盘输入
                    key = v.get_key() if hasattr(v, 'get_key') else None
                    if key == 'r':  # 重置
                        sim.reset()
                        start_time = time.time()
                        trajectory_planner.reset_trajectory()
                        print("仿真已重置")
                    elif key == 'p':  # 暂停
                        paused = not paused
                        print("暂停" if paused else "继续")
                    elif key == 'q':  # 退出
                        break
                    
                    if not paused:
                        # 更新轨迹
                        target_state = trajectory_planner.update_trajectory(current_time)
                        
                        # 将目标状态传递给控制器
                        controller.target_position = target_state['target_position']
                        controller.target_attitude = target_state['target_attitude']
                        controller.target_velocity = target_state['target_velocity']
                        controller.target_acceleration = target_state['target_acceleration']
                        controller.target_attitude_rate = target_state['target_attitude_rate']
                        controller.target_attitude_acceleration = target_state['target_attitude_acceleration']
                        
                        # 更新控制
                        f_c_body, tau_c, state = controller.update_control()
                        
                        if state is not None:
                            # 分配执行器命令并应用
                            actuator_allocation.allocate_and_apply(f_c_body, tau_c, state)
                            
                            # 记录状态，传递轨迹阶段
                            logger.log_status(state, target_state['trajectory_phase'])
                        
                        # 执行一次仿真步进
                        sim.step()
                    
                    # 同步可视化
                    v.sync()
                    
                    # 定期打印状态
                    if current_time - last_print_time > print_interval:
                        logger.print_status(trajectory_phase=target_state['trajectory_phase'])
                        last_print_time = current_time
                    
                    # 控制仿真速率
                    time.sleep(0.001)

            except KeyboardInterrupt:
                print("\n仿真被用户中断")
            
            # 打印仿真总结
            final_state = sim.get_state()
            logger.print_summary(final_state)
            print("仿真结束")
    
    except Exception as e:
        print(f"仿真主循环失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
