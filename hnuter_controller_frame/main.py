import numpy as np
import mujoco.viewer as viewer
import time
import argparse
from simulation_framework import SimulationFramework
from controller import HnuterController
from allocation import ActuatorAllocation
from logger import DroneLogger
from trajectory_planner import TrajectoryPlanner


def main():
    """主函数 - 启动90°大角度姿态跟踪仿真"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='倾转旋翼无人机仿真')
    parser.add_argument('--trajectory', '-t', type=str, default='default', choices=['default', 'lissajous', 'spiral', 'ring', 'rectangle', 'acceleration'],
                        help='轨迹类型：default（默认90°姿态跟踪）、lissajous（李萨如曲线）、spiral（螺旋上升）、ring（环形）、rectangle（矩形）或acceleration（加速测试）')
    args = parser.parse_args()
    
    # 选择轨迹类型
    trajectory_type = args.trajectory
    
    if trajectory_type == 'lissajous':
        print("=== 倾转旋翼无人机李萨如曲线跟踪仿真 ===")
        print("核心特点：平滑跟踪李萨如曲线，机体俯仰跟随曲线变化")
        print("轨迹逻辑：生成平滑李萨如曲线，位置和俯仰角随曲线缓慢变化")
    elif trajectory_type == 'spiral':
        print("=== 倾转旋翼无人机螺旋上升轨迹仿真 ===")
        print("核心特点：平滑螺旋上升，姿态变化平稳")
        print("轨迹逻辑：水平平面圆周运动+缓慢上升，形成螺旋轨迹")
    elif trajectory_type == 'ring':
        print("=== 倾转旋翼无人机环形轨迹仿真 ===")
        print("核心特点：平滑环形飞行，高度保持不变，姿态变化平稳")
        print("轨迹逻辑：水平平面圆周运动，形成环形轨迹")
    elif trajectory_type == 'rectangle':
        print("=== 倾转旋翼无人机逐圈加速矩形轨迹仿真 ===")
        print("核心特点：起飞→逐圈加速飞水平矩形（无俯仰变化，不停歇）")
        print("轨迹逻辑：先悬停，然后以1.5m/s为初始速度持续飞行矩形，每圈加速0.2m/s，最大速度5.0m/s")
    elif trajectory_type == 'acceleration':
        print("=== 倾转旋翼无人机加速测试轨迹仿真 ===")
        print("核心特点：起飞→加速到30m/s→反推减速→反向加速→减速回到原点")
        print("轨迹逻辑：测试无人机的加速和减速性能，使用最大推力加速和反推减速")
        print("关键参数：最大前向加速度2.0m/s²，最大反向加速度-2.5m/s²，目标速度30.0m/s")
    else:
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
        
        # 初始化执行器分配模块
        actuator_allocation = ActuatorAllocation(controller, sim)
        
        # 初始化日志记录模块
        logger = DroneLogger(controller)
        
        # 初始化轨迹规划器，指定轨迹类型
        trajectory_planner = TrajectoryPlanner(trajectory_type)
        
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
                trajectory_complete = False
                trajectory_complete_time = 0.0
                wait_time_after_complete = 5.0  # 轨迹完成后等待5秒
                
                while v.is_running():
                    current_time = time.time() - start_time
                    
                    # 检查键盘输入
                    key = v.get_key() if hasattr(v, 'get_key') else None
                    if key == 'r':  # 重置
                        sim.reset()
                        start_time = time.time()
                        trajectory_planner.reset_trajectory()
                        trajectory_complete = False
                        trajectory_complete_time = 0.0
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
                    
                    # 检查轨迹是否完成
                    if not trajectory_complete and trajectory_planner.is_trajectory_complete():
                        trajectory_complete = True
                        trajectory_complete_time = current_time
                        print(f"\n✅ 轨迹执行完成！将在{wait_time_after_complete}秒后结束仿真")
                    
                    # 如果轨迹完成且等待时间已到，退出仿真
                    if trajectory_complete and (current_time - trajectory_complete_time) > wait_time_after_complete:
                        print(f"\n⏱️  等待时间已到，结束仿真")
                        break
                    
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
                from plotter import find_latest_log_file, load_log_data, plot_drone_data_and_save_all
                
                # 获取最新日志文件
                log_file = find_latest_log_file()
                
                # 加载日志数据
                df = load_log_data(log_file)
                
                # 生成并保存所有绘图
                plot_drone_data_and_save_all(df)
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
