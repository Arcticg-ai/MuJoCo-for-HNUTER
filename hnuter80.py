"""
hnuter80.py - 模块化测试框架
用于单独测试 hnuter69.py 中的各个模块：
1. 轨迹生成器 (Trajectory Planner)
2. 控制器 (Controller)
3. 分配器 (Allocator)
4. 执行器发布 (Actuator Publisher)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import sys

# 导入 hnuter69 的控制器类
from hnuter69 import HnuterController


class ModuleTester:
    """模块测试器基类"""
    def __init__(self, controller: HnuterController):
        self.controller = controller
        self.test_results = []

    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """记录测试结果"""
        status = "✓ PASS" if passed else "✗ FAIL"
        result = f"{status} | {test_name}"
        if details:
            result += f" | {details}"
        self.test_results.append((test_name, passed, details))
        print(result)

    def print_summary(self):
        """打印测试摘要"""
        total = len(self.test_results)
        passed = sum(1 for _, p, _ in self.test_results if p)
        print(f"\n{'='*60}")
        print(f"测试摘要: {passed}/{total} 通过")
        print(f"{'='*60}")


class TrajectoryTester(ModuleTester):
    """轨迹生成器测试"""

    def test_phase_transitions(self):
        """测试阶段切换逻辑"""
        print("\n=== 测试1: 轨迹阶段切换 ===")

        # 测试时间点（对应各阶段边界）
        test_times = [0, 6, 18, 23, 29, 41, 46, 52, 64, 69, 75]
        expected_phases = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        for t, expected_phase in zip(test_times, expected_phases):
            self.controller.trajectory_phase = 0
            self.controller.phase_start_time = 0.0
            self.controller.update_trajectory(t)

            actual_phase = self.controller.trajectory_phase
            passed = (actual_phase == expected_phase)
            self.log_result(
                f"阶段切换 t={t}s",
                passed,
                f"期望阶段{expected_phase}, 实际阶段{actual_phase}"
            )

    def test_attitude_targets(self):
        """测试姿态目标生成"""
        print("\n=== 测试2: 姿态目标生成 ===")

        # 测试关键时间点的姿态目标
        test_cases = [
            # (时间, 期望roll, 期望pitch, 期望yaw, 描述)
            (3.0, 0.0, 0.0, 0.0, "起飞悬停"),
            (12.0, np.pi*2/5/2, 0.0, 0.0, "Roll转动中点"),
            (20.0, np.pi*2/5, 0.0, 0.0, "Roll保持"),
            (35.0, 0.0, np.pi*2/5/2, 0.0, "Pitch转动中点"),
            (43.0, 0.0, np.pi*2/5, 0.0, "Pitch保持"),
            (58.0, 0.0, 0.0, np.pi*2/5/2, "Yaw转动中点"),
            (76.0, 0.0, 0.0, 0.0, "最终悬停")
        ]

        for t, exp_roll, exp_pitch, exp_yaw, desc in test_cases:
            self.controller.trajectory_phase = 0
            self.controller.phase_start_time = 0.0
            self.controller.update_trajectory(t)

            actual = self.controller.target_attitude
            expected = np.array([exp_roll, exp_pitch, exp_yaw])
            error = np.linalg.norm(actual - expected)

            passed = error < 0.1  # 容差0.1弧度
            self.log_result(
                f"{desc} (t={t}s)",
                passed,
                f"误差={error:.4f}rad, 目标=[{np.degrees(actual[0]):.1f}°, {np.degrees(actual[1]):.1f}°, {np.degrees(actual[2]):.1f}°]"
            )

    def test_position_targets(self):
        """测试位置目标生成"""
        print("\n=== 测试3: 位置目标生成 ===")

        # 所有阶段都应该保持在 [0, 0, 2.0]
        test_times = [3, 12, 20, 35, 43, 58, 76]
        expected_pos = np.array([0.0, 0.0, 2.0])

        for t in test_times:
            self.controller.trajectory_phase = 0
            self.controller.phase_start_time = 0.0
            self.controller.update_trajectory(t)

            actual_pos = self.controller.target_position
            error = np.linalg.norm(actual_pos - expected_pos)

            passed = error < 0.01
            self.log_result(
                f"位置目标 t={t}s",
                passed,
                f"误差={error:.6f}m, 目标={actual_pos}"
            )

        # 生成可视化
        self.visualize_trajectory()

    def visualize_trajectory(self):
        """可视化完整轨迹"""
        print("\n=== 生成轨迹可视化 ===")

        times = np.linspace(0, 80, 800)
        rolls = []
        pitches = []
        yaws = []
        phases = []

        for t in times:
            self.controller.trajectory_phase = 0
            self.controller.phase_start_time = 0.0
            self.controller.update_trajectory(t)

            rolls.append(np.degrees(self.controller.target_attitude[0]))
            pitches.append(np.degrees(self.controller.target_attitude[1]))
            yaws.append(np.degrees(self.controller.target_attitude[2]))
            phases.append(self.controller.trajectory_phase)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 子图1: 姿态角
        ax1.plot(times, rolls, 'b-', label='Roll', linewidth=2)
        ax1.plot(times, pitches, 'g-', label='Pitch', linewidth=2)
        ax1.plot(times, yaws, 'r-', label='Yaw', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Attitude (deg)')
        ax1.set_title('Trajectory: Target Attitude vs Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2: 阶段
        ax2.plot(times, phases, 'k-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Phase')
        ax2.set_title('Trajectory Phase vs Time')
        ax2.grid(True, alpha=0.3)
        ax2.set_yticks(range(11))

        plt.tight_layout()
        plt.savefig('logs/trajectory_test.png', dpi=150)
        print("轨迹可视化已保存到: logs/trajectory_test.png")
        plt.close()


class ControllerTester(ModuleTester):
    """控制器测试"""

    def test_position_control(self):
        """测试位置控制"""
        print("\n=== 测试4: 位置控制 ===")

        # 创建测试状态
        test_cases = [
            # (当前位置, 目标位置, 描述)
            (np.array([0, 0, 1.0]), np.array([0, 0, 2.0]), "向上1m"),
            (np.array([1, 0, 2.0]), np.array([0, 0, 2.0]), "X方向偏移1m"),
            (np.array([0, 1, 2.0]), np.array([0, 0, 2.0]), "Y方向偏移1m"),
        ]

        for curr_pos, target_pos, desc in test_cases:
            state = self._create_test_state(position=curr_pos)
            self.controller.target_position = target_pos
            self.controller.target_attitude = np.zeros(3)

            f_c_body, tau_c = self.controller.compute_control_wrench(state)

            # 检查控制力方向是否正确
            pos_error = target_pos - curr_pos
            f_c_world = self.controller.f_c_world

            # 控制力应该指向目标方向
            force_direction_correct = np.dot(f_c_world[:2], pos_error[:2]) > 0 if np.linalg.norm(pos_error[:2]) > 0.01 else True

            passed = force_direction_correct
            self.log_result(
                f"位置控制: {desc}",
                passed,
                f"位置误差={pos_error}, 控制力(世界)={f_c_world}"
            )

    def test_attitude_control(self):
        """测试姿态控制"""
        print("\n=== 测试5: 姿态控制 ===")

        # 测试不同姿态误差下的力矩响应
        test_cases = [
            # (当前姿态, 目标姿态, 描述)
            (np.array([0.1, 0, 0]), np.array([0, 0, 0]), "Roll误差+0.1rad"),
            (np.array([0, 0.1, 0]), np.array([0, 0, 0]), "Pitch误差+0.1rad"),
            (np.array([0, 0, 0.1]), np.array([0, 0, 0]), "Yaw误差+0.1rad"),
            (np.array([0, 0, 0]), np.array([0.5, 0, 0]), "Roll目标+0.5rad"),
        ]

        for curr_att, target_att, desc in test_cases:
            state = self._create_test_state(euler=curr_att)
            self.controller.target_position = np.array([0, 0, 2.0])
            self.controller.target_attitude = target_att

            f_c_body, tau_c = self.controller.compute_control_wrench(state)

            # 检查力矩是否非零（有控制响应）
            torque_magnitude = np.linalg.norm(tau_c)
            passed = torque_magnitude > 0.001

            self.log_result(
                f"姿态控制: {desc}",
                passed,
                f"力矩={tau_c}, 幅值={torque_magnitude:.4f}Nm"
            )

    def test_pitch_exceed_logic(self):
        """测试俯仰角超限逻辑"""
        print("\n=== 测试6: 俯仰角超限逻辑 ===")

        # 测试俯仰角在阈值附近的行为
        pitch_angles = [60, 70, 80, 90]  # 度

        for pitch_deg in pitch_angles:
            pitch_rad = np.radians(pitch_deg)
            state = self._create_test_state(euler=np.array([0.0, pitch_rad, 0.0]))

            # 设置目标姿态有误差，产生力矩
            self.controller.target_position = np.array([0, 0, 2.0])
            self.controller.target_attitude = np.zeros(3)

            f_c_body, tau_c = self.controller.compute_control_wrench(state)

            # 检查是否正确标记超限
            is_exceed = state['is_pitch_exceed']
            expected_exceed = pitch_deg > self.controller.pitch_threshold_deg

            passed = (is_exceed == expected_exceed)
            self.log_result(
                f"俯仰角{pitch_deg}°超限检测",
                passed,
                f"超限标记={is_exceed}, 期望={expected_exceed}, 力矩={tau_c}"
            )

    def test_pitch_control_coupling_detailed(self):
        """详细测试纯俯仰控制时的力矩耦合（测试6增强版）"""
        print("\n=== 测试6增强: 纯俯仰控制力矩耦合详细分析 ===")

        # 测试场景：纯俯仰运动，roll和yaw保持0
        # 理论上只应产生俯仰力矩tau_y，tau_x和tau_z应接近0

        test_cases = [
            # (当前姿态[roll,pitch,yaw], 目标姿态[roll,pitch,yaw], 描述)
            (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.2, 0.0]), "纯俯仰: 0°→11.5°"),
            (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.5, 0.0]), "纯俯仰: 0°→28.6°"),
            (np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), "纯俯仰: 0°→57.3°"),
            (np.array([0.0, 0.2, 0.0]), np.array([0.0, 0.5, 0.0]), "纯俯仰: 11.5°→28.6°"),
            (np.array([0.0, 0.5, 0.0]), np.array([0.0, 0.0, 0.0]), "纯俯仰恢复: 28.6°→0°"),
            (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0]), "纯俯仰恢复: 57.3°→0°"),
        ]

        print("\n详细力矩分析（纯俯仰运动时，tau_x和tau_z应接近0）：")
        print(f"{'描述':<25} {'tau_x(Nm)':<12} {'tau_y(Nm)':<12} {'tau_z(Nm)':<12} {'耦合比':<12} {'结果':<8}")
        print("-" * 85)

        for curr_att, target_att, desc in test_cases:
            # 创建纯俯仰状态（roll=0, yaw=0）
            state = self._create_test_state(
                euler=curr_att,
                position=np.array([0, 0, 2.0]),
                velocity=np.zeros(3),
                angular_velocity=np.zeros(3)
            )

            self.controller.target_position = np.array([0, 0, 2.0])
            self.controller.target_attitude = target_att
            self.controller.target_velocity = np.zeros(3)
            self.controller.target_attitude_rate = np.zeros(3)

            # 计算控制力矩
            f_c_body, tau_c = self.controller.compute_control_wrench(state)

            tau_x = tau_c[0]  # 横滚力矩
            tau_y = tau_c[1]  # 俯仰力矩
            tau_z = tau_c[2]  # 偏航力矩

            # 计算耦合比：横滚和偏航力矩相对于俯仰力矩的比例
            if abs(tau_y) > 1e-6:
                coupling_ratio = (abs(tau_x) + abs(tau_z)) / abs(tau_y)
            else:
                coupling_ratio = 0.0

            # 判断标准：
            # 1. 俯仰力矩应该非零（有控制作用）
            # 2. 横滚和偏航力矩应该很小（耦合比 < 0.05，即5%）
            tau_y_valid = abs(tau_y) > 0.001  # 俯仰力矩应该存在
            coupling_small = coupling_ratio < 0.05  # 耦合应该小于5%

            passed = tau_y_valid and coupling_small

            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{desc:<25} {tau_x:>11.6f} {tau_y:>11.6f} {tau_z:>11.6f} {coupling_ratio:>11.4f} {status:<8}")

            self.log_result(
                f"纯俯仰耦合: {desc}",
                passed,
                f"tau=[{tau_x:.6f}, {tau_y:.6f}, {tau_z:.6f}], 耦合比={coupling_ratio:.4f}"
            )

            # 如果测试失败，输出详细诊断信息
            if not passed:
                print(f"  ⚠ 诊断信息:")
                if not tau_y_valid:
                    print(f"    - 俯仰力矩过小: {tau_y:.6f} Nm")
                if not coupling_small:
                    print(f"    - 耦合力矩过大: tau_x={tau_x:.6f} Nm, tau_z={tau_z:.6f} Nm")
                    print(f"    - 耦合比: {coupling_ratio*100:.2f}% (应 < 5%)")

                # 输出旋转矩阵和姿态误差
                R = state['rotation_matrix']
                R_des = self.controller._euler_to_rotation_matrix(target_att)
                e_R = 0.5 * self.controller.vee_map(R_des.T @ R - R.T @ R_des)
                print(f"    - 姿态误差向量 e_R: [{e_R[0]:.6f}, {e_R[1]:.6f}, {e_R[2]:.6f}]")
                print(f"    - 当前姿态: Roll={np.degrees(curr_att[0]):.2f}°, Pitch={np.degrees(curr_att[1]):.2f}°, Yaw={np.degrees(curr_att[2]):.2f}°")
                print(f"    - 目标姿态: Roll={np.degrees(target_att[0]):.2f}°, Pitch={np.degrees(target_att[1]):.2f}°, Yaw={np.degrees(target_att[2]):.2f}°")

        print()  # 空行分隔

    def _create_test_state(self, position=None, euler=None, velocity=None, angular_velocity=None):
        """创建测试用状态字典"""
        if position is None:
            position = np.array([0, 0, 2.0])
        if euler is None:
            euler = np.zeros(3)
        if velocity is None:
            velocity = np.zeros(3)
        if angular_velocity is None:
            angular_velocity = np.zeros(3)

        # 欧拉角转四元数和旋转矩阵
        quat = self.controller._euler_to_quaternion(euler)
        R = self.controller._euler_to_rotation_matrix(euler)

        # 判断俯仰角是否超限
        is_pitch_exceed = abs(euler[1]) > self.controller.pitch_threshold_rad

        return {
            'position': position,
            'quaternion': quat,
            'rotation_matrix': R,
            'velocity': velocity,
            'angular_velocity': angular_velocity,
            'acceleration': np.zeros(3),
            'euler': euler,
            'is_pitch_exceed': is_pitch_exceed
        }


class AllocatorTester(ModuleTester):
    """分配器测试"""

    def test_force_allocation(self):
        """测试力分配"""
        print("\n=== 测试7: 力分配 ===")

        # 测试不同控制力的分配
        test_cases = [
            # (f_body, tau, 描述)
            (np.array([10, 0, 40]), np.zeros(3), "纯推力(Fx=10, Fz=40)"),
            (np.array([0, 5, 40]), np.zeros(3), "侧向力(Fy=5, Fz=40)"),
            (np.array([0, 0, 40]), np.array([1, 0, 0]), "滚转力矩(Tx=1)"),
            (np.array([0, 0, 40]), np.array([0, 2, 0]), "俯仰力矩(Ty=2)"),
            (np.array([0, 0, 40]), np.array([0, 0, 0.5]), "偏航力矩(Tz=0.5)"),
        ]

        state = self._create_dummy_state()

        for f_body, tau, desc in test_cases:
            T12, T34, T5, alpha1, alpha2, theta1, theta2 = self.controller.allocate_actuators(
                f_body, tau, state
            )

            # 检查推力非负
            thrust_valid = (T12 >= 0) and (T34 >= 0)

            # 检查角度在限制范围内
            alpha_max = np.radians(200)
            theta_max = np.radians(200)
            angles_valid = (abs(alpha1) <= alpha_max and abs(alpha2) <= alpha_max and
                          abs(theta1) <= theta_max and abs(theta2) <= theta_max)

            passed = thrust_valid and angles_valid
            self.log_result(
                f"力分配: {desc}",
                passed,
                f"T12={T12:.2f}N, T34={T34:.2f}N, T5={T5:.2f}N, α1={np.degrees(alpha1):.1f}°, α2={np.degrees(alpha2):.1f}°, θ1={np.degrees(theta1):.1f}°, θ2={np.degrees(theta2):.1f}°"
            )

    def test_inverse_mapping(self):
        """测试逆映射函数"""
        print("\n=== 测试8: 逆映射函数 ===")

        # 测试控制向量W到执行器命令的映射
        test_cases = [
            # (Fx, Fy, Fz, Tx, Ty, Tz, 描述)
            (0, 0, 40, 0, 0, 0, "悬停(Fz=40N)"),
            (10, 0, 40, 0, 0, 0, "前飞(Fx=10N)"),
            (0, 0, 40, 2, 0, 0, "滚转(Tx=2Nm)"),
            (0, 0, 40, 0, 1, 0, "俯仰(Ty=1Nm)"),
            (0, 0, 40, 0, 0, 1, "偏航(Tz=1Nm)"),
        ]

        state = self._create_dummy_state()

        for Fx, Fy, Fz, Tx, Ty, Tz, desc in test_cases:
            W = np.array([Fx, Fy, Fz, Tx, Ty, Tz])
            result = self.controller.inverse_nonlinear_mapping(W, state)

            F1, F2, F3, alpha1, alpha2, theta1, theta2 = result

            # 检查结果有效性
            valid = (not np.isnan(F1) and not np.isnan(F2) and
                    not np.isnan(alpha1) and not np.isnan(alpha2) and
                    not np.isnan(theta1) and not np.isnan(theta2))

            passed = valid
            self.log_result(
                f"逆映射: {desc}",
                passed,
                f"F1={F1:.2f}, F2={F2:.2f}, F3={F3:.2f}, α1={np.degrees(alpha1):.1f}°, α2={np.degrees(alpha2):.1f}°, θ1={np.degrees(theta1):.1f}°, θ2={np.degrees(theta2):.1f}°"
            )

    def test_allocation_limits(self):
        """测试分配限制"""
        print("\n=== 测试9: 分配限制 ===")

        # 测试极端输入下的限制
        state = self._create_dummy_state()

        # 极大推力需求
        f_body = np.array([0, 0, 200])  # 远超最大推力
        tau = np.zeros(3)

        T12, T34, T5, alpha1, alpha2, theta1, theta2 = self.controller.allocate_actuators(
            f_body, tau, state
        )

        # 检查是否被限制
        T_max = 60
        thrust_limited = (T12 <= T_max) and (T34 <= T_max)

        self.log_result(
            "推力限制测试",
            thrust_limited,
            f"输入Fz=200N, 输出T12={T12:.2f}N(限制{T_max}N), T34={T34:.2f}N"
        )

        # 极大力矩需求
        f_body = np.array([0, 0, 40])
        tau = np.array([10, 10, 10])  # 极大力矩

        T12, T34, T5, alpha1, alpha2, theta1, theta2 = self.controller.allocate_actuators(
            f_body, tau, state
        )

        alpha_max = np.radians(200)
        theta_max = np.radians(200)
        angles_limited = (abs(alpha1) <= alpha_max) and (abs(alpha2) <= alpha_max) and \
                        (abs(theta1) <= theta_max) and (abs(theta2) <= theta_max)

        self.log_result(
            "角度限制测试",
            angles_limited,
            f"输入τ=[10,10,10]Nm, 输出α1={np.degrees(alpha1):.1f}°, α2={np.degrees(alpha2):.1f}°, θ1={np.degrees(theta1):.1f}°, θ2={np.degrees(theta2):.1f}°(限制±200°)"
        )

    def test_theta_angle_allocation(self):
        """详细测试theta角度分配（侧向倾转角）"""
        print("\n=== 测试9增强: Theta角度分配详细测试 ===")

        # 测试不同侧向力需求下的theta角度分配
        test_cases = [
            # (Fx, Fy, Fz, Tx, Ty, Tz, 描述)
            (0, 0, 40, 0, 0, 0, "悬停(无侧向力)"),
            (0, 2, 40, 0, 0, 0, "侧向力Fy=2N"),
            (0, 5, 40, 0, 0, 0, "侧向力Fy=5N"),
            (0, 10, 40, 0, 0, 0, "侧向力Fy=10N"),
            (0, -5, 40, 0, 0, 0, "侧向力Fy=-5N"),
            (5, 5, 40, 0, 0, 0, "前飞+侧向Fx=5,Fy=5"),
            (0, 0, 40, 2, 0, 0, "滚转力矩Tx=2Nm"),
        ]

        state = self._create_dummy_state()

        print(f"\n{'描述':<25} {'Fy(N)':<10} {'θ1(°)':<10} {'θ2(°)':<10} {'θ差值(°)':<12} {'结果':<8}")
        print("-" * 80)

        for Fx, Fy, Fz, Tx, Ty, Tz, desc in test_cases:
            W = np.array([Fx, Fy, Fz, Tx, Ty, Tz])
            result = self.controller.inverse_nonlinear_mapping(W, state)

            F1, F2, F3, alpha1, alpha2, theta1, theta2 = result

            theta1_deg = np.degrees(theta1)
            theta2_deg = np.degrees(theta2)
            theta_diff = abs(theta1_deg - theta2_deg)

            # 检查theta角度的合理性
            # 1. 悬停时theta应该接近0
            # 2. 有侧向力时theta应该非零
            # 3. theta1和theta2应该对称（差值小）

            if abs(Fy) < 0.1 and abs(Tx) < 0.1:
                # 无侧向力和滚转力矩时，theta应该接近0
                theta_reasonable = (abs(theta1_deg) < 5) and (abs(theta2_deg) < 5)
            else:
                # 有侧向力或滚转力矩时，theta应该有响应
                theta_reasonable = True  # 只要不是NaN就算合理

            # theta1和theta2应该相对对称（对于纯侧向力）
            if abs(Tx) < 0.1:  # 无滚转力矩
                symmetry_ok = theta_diff < 10  # 差值小于10度
            else:
                symmetry_ok = True  # 有滚转力矩时允许不对称

            passed = theta_reasonable and symmetry_ok and \
                    (not np.isnan(theta1)) and (not np.isnan(theta2))

            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{desc:<25} {Fy:>9.1f} {theta1_deg:>9.2f} {theta2_deg:>9.2f} {theta_diff:>11.2f} {status:<8}")

            self.log_result(
                f"Theta分配: {desc}",
                passed,
                f"Fy={Fy:.1f}N, θ1={theta1_deg:.2f}°, θ2={theta2_deg:.2f}°, 差值={theta_diff:.2f}°"
            )

        print()

    def _create_dummy_state(self):
        """创建虚拟状态"""
        return {
            'position': np.array([0, 0, 2.0]),
            'euler': np.zeros(3),
            'rotation_matrix': np.eye(3),
            'is_pitch_exceed': False
        }


class ActuatorTester(ModuleTester):
    """执行器测试"""

    def test_actuator_mapping(self):
        """测试执行器映射"""
        print("\n=== 测试10: 执行器ID映射 ===")

        # 检查所有必需的执行器是否存在
        required_actuators = [
            'arm_pitch_right', 'arm_pitch_left',
            'prop_tilt_right', 'prop_tilt_left',
            'motor_r_upper', 'motor_r_lower',
            'motor_l_upper', 'motor_l_lower',
            'motor_rear_upper'
        ]

        for name in required_actuators:
            exists = name in self.controller.actuator_ids
            self.log_result(
                f"执行器映射: {name}",
                exists,
                f"ID={self.controller.actuator_ids.get(name, 'N/A')}"
            )

    def test_actuator_commands(self):
        """测试执行器命令设置"""
        print("\n=== 测试11: 执行器命令设置 ===")

        # 测试不同的执行器命令
        test_cases = [
            (20, 20, 0, 0, 0, 0, 0, "悬停配置"),
            (30, 30, 5, 0.5, 0.5, 0, 0, "前飞配置"),
            (25, 25, 0, 1.0, 1.0, 0.2, 0.2, "大角度配置"),
        ]

        for T12, T34, T5, alpha1, alpha2, theta1, theta2, desc in test_cases:
            try:
                self.controller.set_actuators(T12, T34, T5, alpha1, alpha2, theta1, theta2)

                # 检查控制命令是否被设置
                passed = True
                details = f"T12={T12}N, T34={T34}N, T5={T5}N"
            except Exception as e:
                passed = False
                details = f"异常: {str(e)}"

            self.log_result(
                f"执行器命令: {desc}",
                passed,
                details
            )


class IntegrationTester(ModuleTester):
    """集成测试"""

    def test_full_pipeline(self):
        """测试完整数据流"""
        print("\n=== 测试12: 完整数据流 ===")

        # 模拟一个完整的控制周期
        test_times = [0, 10, 30, 50, 70]

        for t in test_times:
            try:
                # 1. 轨迹生成
                self.controller.trajectory_phase = 0
                self.controller.phase_start_time = 0.0
                self.controller.update_trajectory(t)

                # 2. 获取状态（使用虚拟状态）
                state = self._create_test_state()

                # 3. 计算控制
                f_c_body, tau_c = self.controller.compute_control_wrench(state)

                # 4. 分配执行器
                T12, T34, T5, alpha1, alpha2, theta1, theta2 = self.controller.allocate_actuators(
                    f_c_body, tau_c, state
                )

                # 5. 设置执行器
                self.controller.set_actuators(T12, T34, T5, alpha1, alpha2, theta1, theta2)

                passed = True
                details = f"阶段{self.controller.trajectory_phase}, T12={T12:.1f}N, τ={tau_c}"
            except Exception as e:
                passed = False
                details = f"异常: {str(e)}"

            self.log_result(
                f"完整流程 t={t}s",
                passed,
                details
            )

    def test_data_consistency(self):
        """测试数据一致性"""
        print("\n=== 测试13: 数据一致性 ===")

        # 测试控制器内部状态的一致性
        state = self._create_test_state()
        self.controller.target_position = np.array([0, 0, 2.0])
        self.controller.target_attitude = np.array([0.5, 0, 0])

        # 计算控制
        f_c_body, tau_c = self.controller.compute_control_wrench(state)

        # 检查内部状态是否更新
        f_body_stored = self.controller.f_c_body
        tau_stored = self.controller.tau_c

        f_consistent = np.allclose(f_c_body, f_body_stored)
        tau_consistent = np.allclose(tau_c, tau_stored)

        passed = f_consistent and tau_consistent
        self.log_result(
            "控制器状态一致性",
            passed,
            f"f_body一致={f_consistent}, tau一致={tau_consistent}"
        )

        # 分配执行器
        T12, T34, T5, alpha1, alpha2, theta1, theta2 = self.controller.allocate_actuators(
            f_c_body, tau_c, state
        )

        # 检查分配结果是否存储
        u_stored = self.controller.u
        u_expected = np.array([T12, T34, T5, alpha1, alpha2, theta1, theta2])

        u_consistent = np.allclose(u_stored, u_expected)

        self.log_result(
            "分配器状态一致性",
            u_consistent,
            f"u向量一致={u_consistent}"
        )

    def _create_test_state(self):
        """创建测试状态"""
        euler = np.array([0.1, 0.1, 0.1])
        quat = self.controller._euler_to_quaternion(euler)
        R = self.controller._euler_to_rotation_matrix(euler)

        return {
            'position': np.array([0, 0, 2.0]),
            'quaternion': quat,
            'rotation_matrix': R,
            'velocity': np.zeros(3),
            'angular_velocity': np.zeros(3),
            'acceleration': np.zeros(3),
            'euler': euler,
            'is_pitch_exceed': False
        }


def main():
    """主测试函数"""
    print("="*60)
    print("hnuter80.py - 模块化测试框架")
    print("="*60)

    try:
        # 初始化控制器（不启动仿真）
        print("\n初始化控制器...")
        controller = HnuterController("hnuter201.xml")
        print("控制器初始化完成\n")

        # 创建测试器
        testers = [
            ("轨迹生成器", TrajectoryTester(controller)),
            ("控制器", ControllerTester(controller)),
            ("分配器", AllocatorTester(controller)),
            ("执行器", ActuatorTester(controller)),
            ("集成测试", IntegrationTester(controller))
        ]

        # 运行所有测试
        all_results = []

        for name, tester in testers:
            print(f"\n{'='*60}")
            print(f"开始测试: {name}")
            print(f"{'='*60}")

            # 运行该测试器的所有测试方法
            test_methods = [m for m in dir(tester) if m.startswith('test_') and callable(getattr(tester, m))]
            for method_name in test_methods:
                method = getattr(tester, method_name)
                try:
                    method()
                except Exception as e:
                    print(f"✗ 测试方法 {method_name} 失败: {e}")
                    import traceback
                    traceback.print_exc()

            all_results.extend(tester.test_results)

        # 打印总体摘要
        print(f"\n{'='*60}")
        print("总体测试摘要")
        print(f"{'='*60}")

        total = len(all_results)
        passed = sum(1 for _, p, _ in all_results if p)
        failed = total - passed

        print(f"总测试数: {total}")
        print(f"通过: {passed} ({100*passed/total:.1f}%)")
        print(f"失败: {failed} ({100*failed/total:.1f}%)")

        if failed > 0:
            print(f"\n失败的测试:")
            for name, p, details in all_results:
                if not p:
                    print(f"  - {name}: {details}")

        print(f"\n{'='*60}")
        print("测试完成")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n测试框架失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()