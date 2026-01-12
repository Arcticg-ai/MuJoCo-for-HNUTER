import numpy as np
import mujoco.viewer as viewer
import time
from simulation_framework import SimulationFramework
from controller import HnuterController
from allocation import ActuatorAllocation
from logger import DroneLogger
from trajectory_planner import TrajectoryPlanner


def main():
    """主函数 - 启动90°大角度姿态跟踪仿真"""
    print("=== 倾转旋翼无人机90°大角度姿态跟踪仿真 ===")
    print("核心优化：适配90°大角度，延长转动/保持/恢复时间，提高控制器增益")
    print("安全限制：俯仰角超过70°时自动置零横滚/偏航力矩")
    print("轨迹逻辑：起飞悬停→Roll90°(保持5s)→恢复→Pitch90°(保持5s)→恢复→Yaw90°(保持5s)→恢复→悬停")
    
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
                        controller.target_rotation_matrix = target_state['target_rotation_matrix']
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
            
            # 生成飞行数据分析图
            print("\n=== 生成飞行数据分析图 ===")
            try:
                from plotter import find_latest_log_file, load_log_data, plot_drone_data_and_save
                
                # 获取最新日志文件
                log_file = find_latest_log_file()
                
                # 加载日志数据
                df = load_log_data(log_file)
                
                # 生成并保存绘图
                plot_drone_data_and_save(df)
                print("飞行数据分析图生成成功!")
            except Exception as e:
                print(f"生成飞行数据分析图失败: {e}")
                import traceback
                traceback.print_exc()
            
            print("仿真结束")
    
    except Exception as e:
        print(f"仿真主循环失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
