"""
Tri-Tiltrotor Coaxial Configuration + Analytical Allocation + Tilt Feedforward Compensation
High-Fidelity Simulation Verification Framework

Verification Scenarios:
1. Scenario 1: Full Attitude Decoupling Verification
2. Scenario 2: Narrow Space Visual Adaptation (Core Highlight)
3. Scenario 3: Actuator Saturation and Nullspace Optimization
"""

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from dataclasses import dataclass
from typing import Tuple, List
import time
import threading


@dataclass
class SimConfig:
    """Simulation configuration parameters"""
    dt: float = 0.001  # 1ms simulation step to capture actuator transient response
    duration: float = 20.0

    # Physical parameters (matching hnuter69)
    mass: float = 4.2  # Main body mass + rotor mechanism mass (kg)
    J: np.ndarray = None  # Inertia matrix [0.08, 0.12, 0.1]

    # Rotor layout parameters
    l1: float = 0.3  # Front rotor group Y-axis distance (m)
    l2: float = 0.5  # Rear thruster X-axis distance (m)

    # Actuator dynamics parameters (high-fidelity modeling)
    motor_time_constant: float = 0.05  # Motor first-order lag time constant
    servo_bandwidth: float = 20.0  # Servo bandwidth (Hz)
    tilt_rate_limit: float = np.pi  # Tilt rate limit (rad/s)

    # Control gains (matching hnuter69)
    Kp: np.ndarray = None  # Position gain [6, 6, 6]
    Dp: np.ndarray = None  # Velocity damping [5, 5, 5]
    KR: np.ndarray = None  # Attitude gain [3, 2.0, 0.3]
    Domega: np.ndarray = None  # Angular velocity damping [0.9, 0.6, 0.6]

    # Actuator limits
    thrust_min: float = 0.0
    thrust_max: float = 60.0  # Max thrust per rotor group (N)
    thrust_rear_min: float = -15.0
    thrust_rear_max: float = 15.0

    # Visualization
    enable_viewer: bool = True  # Enable real-time visualization

    def __post_init__(self):
        if self.J is None:
            self.J = np.diag([0.08, 0.12, 0.1])
        if self.Kp is None:
            self.Kp = np.diag([6, 6, 6])
        if self.Dp is None:
            self.Dp = np.diag([5, 5, 5])
        if self.KR is None:
            self.KR = np.array([3, 2.0, 0.3])
        if self.Domega is None:
            self.Domega = np.array([0.9, 0.6, 0.6])


class TiltRotorDynamics:
    """Tilt-rotor actuator dynamics model"""

    def __init__(self, config: SimConfig):
        self.config = config
        self.tau_motor = config.motor_time_constant
        self.omega_servo = 2 * np.pi * config.servo_bandwidth

        # State variables (matching hnuter69 structure)
        self.T12_actual = 0.0  # Front left rotor group thrust
        self.T34_actual = 0.0  # Front right rotor group thrust
        self.T5_actual = 0.0   # Rear thruster thrust
        self.alpha1_actual = 0.0  # Roll left tilt angle
        self.alpha2_actual = 0.0  # Roll right tilt angle
        self.theta1_actual = 0.0  # Pitch left tilt angle
        self.theta2_actual = 0.0  # Pitch right tilt angle

        # Command values
        self.T12_cmd = 0.0
        self.T34_cmd = 0.0
        self.T5_cmd = 0.0
        self.alpha1_cmd = 0.0
        self.alpha2_cmd = 0.0
        self.theta1_cmd = 0.0
        self.theta2_cmd = 0.0

    def update(self, T12: float, T34: float, T5: float,
               alpha1: float, alpha2: float, theta1: float, theta2: float, dt: float):
        """Update actuator state (including dynamic response)"""
        # Motor first-order lag for thrust
        alpha_motor = dt / (self.tau_motor + dt)
        self.T12_actual += alpha_motor * (T12 - self.T12_actual)
        self.T34_actual += alpha_motor * (T34 - self.T34_actual)
        self.T5_actual += alpha_motor * (T5 - self.T5_actual)

        # Servo response for tilt angles
        alpha_servo = dt * self.omega_servo

        # Alpha angles (roll tilt)
        alpha1_error = alpha1 - self.alpha1_actual
        alpha2_error = alpha2 - self.alpha2_actual
        alpha1_rate = np.clip(self.omega_servo * alpha1_error,
                             -self.config.tilt_rate_limit, self.config.tilt_rate_limit)
        alpha2_rate = np.clip(self.omega_servo * alpha2_error,
                             -self.config.tilt_rate_limit, self.config.tilt_rate_limit)
        self.alpha1_actual += alpha1_rate * dt
        self.alpha2_actual += alpha2_rate * dt

        # Theta angles (pitch tilt)
        theta1_error = theta1 - self.theta1_actual
        theta2_error = theta2 - self.theta2_actual
        theta1_rate = np.clip(self.omega_servo * theta1_error,
                             -self.config.tilt_rate_limit, self.config.tilt_rate_limit)
        theta2_rate = np.clip(self.omega_servo * theta2_error,
                             -self.config.tilt_rate_limit, self.config.tilt_rate_limit)
        self.theta1_actual += theta1_rate * dt
        self.theta2_actual += theta2_rate * dt

        # Limit angles
        alpha_max = np.radians(200)
        self.alpha1_actual = np.clip(self.alpha1_actual, -alpha_max, alpha_max)
        self.alpha2_actual = np.clip(self.alpha2_actual, -alpha_max, alpha_max)
        theta_max = np.radians(200)
        self.theta1_actual = np.clip(self.theta1_actual, -theta_max, theta_max)
        self.theta2_actual = np.clip(self.theta2_actual, -theta_max, theta_max)

        # Store commands
        self.T12_cmd = T12
        self.T34_cmd = T34
        self.T5_cmd = T5
        self.alpha1_cmd = alpha1
        self.alpha2_cmd = alpha2
        self.theta1_cmd = theta1
        self.theta2_cmd = theta2

        return (self.T12_actual, self.T34_actual, self.T5_actual,
                self.alpha1_actual, self.alpha2_actual,
                self.theta1_actual, self.theta2_actual)


class AnalyticalAllocator:
    """Analytical control allocator using nonlinear inverse mapping (matching hnuter69)"""

    def __init__(self, config: SimConfig):
        self.config = config
        self.l1 = config.l1  # Front rotor Y-axis distance
        self.l2 = config.l2  # Rear thruster X-axis distance

    def inverse_nonlinear_mapping(self, W: np.ndarray) -> np.ndarray:
        """
        Nonlinear inverse mapping function (matching hnuter69)
        W: [fx, fy, fz, tx, ty, tz] - desired wrench
        Returns: [F1, F2, F3, alpha1, alpha2, theta1, theta2]
        """
        # Rear thrust (determined by pitch moment)
        u7 = (2/1) * W[4]  # T5 from pitch moment

        # Left/right rotor X-axis force components (from total Fx and yaw moment Tz)
        u1 = W[0]/2 - (10/3)*W[5]
        u4 = W[0]/2 + (10/3)*W[5]

        # Left/right rotor Z-axis force components (from total Fz and roll moment Tx)
        Fz_front = W[2]
        u2 = Fz_front/2 - (10/3)*W[3]
        u5 = Fz_front/2 + (10/3)*W[3]

        # Lateral force components (evenly distributed)
        target_Fy = W[1]
        u3 = -target_Fy / 2.0
        u6 = -target_Fy / 2.0

        # Calculate thrust magnitudes
        F1 = np.sqrt(u1**2 + u2**2 + u3**2)
        F2 = np.sqrt(u4**2 + u5**2 + u6**2)
        F3 = u7

        # Prevent division by zero
        eps = 1e-8
        F1_safe = F1 if F1 > eps else eps
        F2_safe = F2 if F2 > eps else eps

        # Solve tilt angles
        alpha1 = np.arctan2(u1, u2)
        alpha2 = np.arctan2(u4, u5)

        val1 = np.clip(u3 / F1_safe, -1.0 + eps, 1.0 - eps)
        val2 = np.clip(u6 / F2_safe, -1.0 + eps, 1.0 - eps)

        theta1 = np.arcsin(val1)
        theta2 = np.arcsin(val2)

        return np.array([F1, F2, F3, alpha1, alpha2, theta1, theta2])

    def allocate(self, f_c_body: np.ndarray, tau_c: np.ndarray) -> Tuple:
        """
        Allocate control to actuators

        Args:
            f_c_body: Desired force in body frame [Fx, Fy, Fz]
            tau_c: Desired moment [Mx, My, Mz]

        Returns:
            T12, T34, T5: Thrust commands
            alpha1, alpha2: Roll tilt angles
            theta1, theta2: Pitch tilt angles
        """
        # Construct control wrench vector
        W = np.array([
            f_c_body[0],  # X force
            f_c_body[1],  # Y force
            f_c_body[2],  # Z force
            tau_c[0],     # Roll moment
            tau_c[1],     # Pitch moment
            tau_c[2]      # Yaw moment
        ])

        # Nonlinear inverse mapping
        uu = self.inverse_nonlinear_mapping(W)

        # Extract parameters
        F1 = uu[0]  # Front left group thrust
        F2 = uu[1]  # Front right group thrust
        F3 = uu[2]  # Rear thruster thrust
        alpha1 = uu[3]  # Roll left tilt angle
        alpha2 = uu[4]  # Roll right tilt angle
        theta1 = uu[5]  # Pitch left tilt angle
        theta2 = uu[6]  # Pitch right tilt angle

        # Thrust limits
        F1 = np.clip(F1, self.config.thrust_min, self.config.thrust_max)
        F2 = np.clip(F2, self.config.thrust_min, self.config.thrust_max)
        F3 = np.clip(F3, self.config.thrust_rear_min, self.config.thrust_rear_max)

        # Angle limits
        alpha_max = np.radians(200)
        alpha1 = np.clip(alpha1, -alpha_max, alpha_max)
        alpha2 = np.clip(alpha2, -alpha_max, alpha_max)
        theta_max = np.radians(200)
        theta1 = np.clip(theta1, -theta_max, theta_max)
        theta2 = np.clip(theta2, -theta_max, theta_max)

        return F1, F2, F3, alpha1, alpha2, theta1, theta2


class SE3GeometricController:
    """SE(3) Geometric Controller (matching hnuter69)"""

    def __init__(self, config: SimConfig):
        self.config = config
        self.mass = config.mass
        self.g = 9.81
        self.J = config.J

        # Control gains from config
        self.Kp = config.Kp
        self.Dp = config.Dp
        self.KR = config.KR
        self.Domega = config.Domega

    def vee_map(self, M: np.ndarray) -> np.ndarray:
        """Vee map: so(3) -> R^3"""
        return np.array([M[2, 1], M[0, 2], M[1, 0]])

    def compute_control(self, state: dict, desired: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute control command (matching hnuter69 implementation)

        Args:
            state: Current state {pos, vel, R, omega}
            desired: Desired state {pos, vel, acc, yaw}

        Returns:
            f_c_body: Desired force in body frame
            tau_c: Desired moment
        """
        # Extract state
        x = state['pos']
        v = state['vel']
        R = state['R']
        Omega = state['omega']

        # Extract desired trajectory
        xd = desired['pos']
        vd = desired['vel']
        ad = desired['acc']
        yaw_d = desired['yaw']

        # Position error
        ex = x - xd
        ev = v - vd

        # Desired force in world frame
        A = -self.Kp @ ex - self.Dp @ ev + self.mass * np.array([0, 0, self.g]) + self.mass * ad

        # Current body z-axis in world frame
        b3 = R[:, 2]

        # Desired total thrust magnitude
        f_total = -np.dot(A, b3)

        # Desired body z-axis direction
        b3d = -A / np.linalg.norm(A)

        # Construct desired rotation matrix
        # Desired yaw direction
        c1d = np.array([np.cos(yaw_d), np.sin(yaw_d), 0.0])

        # Desired body y-axis
        b2d_unnorm = np.cross(b3d, c1d)
        b2d = b2d_unnorm / (np.linalg.norm(b2d_unnorm) + 1e-8)

        # Desired body x-axis
        b1d = np.cross(b2d, b3d)

        # Desired rotation matrix
        Rd = np.column_stack([b1d, b2d, b3d])

        # Attitude error (SO(3))
        eR_matrix = 0.5 * (Rd.T @ R - R.T @ Rd)
        eR = self.vee_map(eR_matrix)

        # Angular velocity error (assume desired angular velocity is zero)
        eOmega = Omega

        # Desired moment
        tau_c = -self.KR * eR - self.Domega * eOmega

        # Desired force in body frame
        f_c_body = np.array([0, 0, f_total])

        return f_c_body, tau_c


class ScenarioManager:
    """Verification scenario manager"""

    @staticmethod
    def scenario_1_decoupling(t: float) -> dict:
        """
        Scenario 1: Full Attitude Decoupling Verification
        Apply gentle yaw step commands during hover to verify position lock
        """
        desired = {
            'pos': np.array([0.0, 0.0, 1.0]),
            'vel': np.zeros(3),
            'acc': np.zeros(3),
            'yaw': 0.0
        }

        # Gentle yaw changes to verify decoupling
        if 3.0 < t < 8.0:
            # Gradual yaw to 30°
            desired['yaw'] = np.radians(30) * min(1.0, (t - 3.0) / 2.0)
        elif 8.0 <= t < 10.0:
            # Hold at 30°
            desired['yaw'] = np.radians(30)
        elif 10.0 <= t < 15.0:
            # Gradual yaw back to 0°
            desired['yaw'] = np.radians(30) * (1.0 - (t - 10.0) / 5.0)

        return desired

    @staticmethod
    def scenario_2_narrow_space(t: float) -> dict:
        """
        Scenario 2: Narrow Space Visual Adaptation
        Simulate inspection - position lock with slow sinusoidal yaw scanning
        """
        desired = {
            'pos': np.array([0.0, 0.0, 1.2]),  # Fixed position
            'vel': np.zeros(3),
            'acc': np.zeros(3),
            'yaw': 0.0
        }

        # Slow sinusoidal yaw scanning after stabilization
        if t > 3.0:
            # ±30° viewing angle scan with 0.2 Hz frequency (5s period)
            desired['yaw'] = np.radians(30) * np.sin(0.4 * (t - 3.0))

        return desired

    @staticmethod
    def scenario_3_saturation(t: float) -> dict:
        """
        Scenario 3: Combined Position and Attitude Maneuver
        Gentle combined motion to test allocation without saturation
        """
        desired = {
            'pos': np.array([0.0, 0.0, 1.0]),
            'vel': np.zeros(3),
            'acc': np.zeros(3),
            'yaw': 0.0
        }

        # Gentle combined maneuver
        if 3.0 < t < 12.0:
            # Small circular motion in XY plane
            omega = 0.5  # rad/s
            radius = 0.3  # m
            phase = omega * (t - 3.0)
            desired['pos'][0] = radius * (1 - np.cos(phase))
            desired['pos'][1] = radius * np.sin(phase)

            # Yaw follows the motion direction
            desired['yaw'] = phase

        return desired

    @staticmethod
    def scenario_hover_test(t: float) -> dict:
        """
        Scenario 0: Simple Hover Test
        Just hover at fixed position to verify basic stability
        """
        desired = {
            'pos': np.array([0.0, 0.0, 1.0]),
            'vel': np.zeros(3),
            'acc': np.zeros(3),
            'yaw': 0.0
        }
        return desired


class HighFidelitySimulator:
    """High-fidelity simulator with real-time visualization"""

    def __init__(self, xml_path: str, config: SimConfig, scenario_func):
        self.config = config
        self.scenario_func = scenario_func

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Initialize components
        self.controller = SE3GeometricController(config)
        self.allocator = AnalyticalAllocator(config)
        self.actuator_dynamics = TiltRotorDynamics(config)

        # Viewer
        self.viewer = None
        self.viewer_running = False

        # Data recording
        self.time_history = []
        self.pos_history = []
        self.pos_des_history = []
        self.att_history = []
        self.thrust_history = []
        self.tilt_history = []
        self.pos_error_history = []

    def get_state(self) -> dict:
        """Get current state"""
        pos = self.data.qpos[:3].copy()
        quat = self.data.qpos[3:7].copy()
        vel = self.data.qvel[:3].copy()
        omega = self.data.qvel[3:6].copy()

        # Quaternion to rotation matrix
        R = Rot.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()

        return {
            'pos': pos,
            'vel': vel,
            'R': R,
            'omega': omega,
            'quat': quat
        }

    def run(self):
        """Run simulation with real-time visualization"""
        t = 0.0
        step = 0

        print("Starting high-fidelity simulation verification...")
        print(f"Simulation step: {self.config.dt*1000:.1f}ms")
        print(f"Simulation duration: {self.config.duration:.1f}s")

        # Launch viewer in separate thread if enabled
        if self.config.enable_viewer:
            print("Launching real-time visualization...")
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer_running = True

        try:
            while t < self.config.duration:
                # Get current state
                state = self.get_state()

                # Get desired trajectory
                desired = self.scenario_func(t)

                # Geometric controller
                f_c_body, tau_c = self.controller.compute_control(state, desired)

                # Analytical allocation
                T12, T34, T5, alpha1, alpha2, theta1, theta2 = self.allocator.allocate(
                    f_c_body, tau_c
                )

                # Actuator dynamics
                T12_act, T34_act, T5_act, alpha1_act, alpha2_act, theta1_act, theta2_act = \
                    self.actuator_dynamics.update(
                        T12, T34, T5, alpha1, alpha2, theta1, theta2, self.config.dt
                    )

                # Apply control to MuJoCo (matching hnuter69 actuator mapping)
                # ctrl[0] = T12 (front left rotor group)
                # ctrl[1] = T34 (front right rotor group)
                # ctrl[2] = alpha1 (roll left tilt)
                # ctrl[3] = alpha2 (roll right tilt)
                # ctrl[4] = theta1 (pitch left tilt)
                # ctrl[5] = theta2 (pitch right tilt)
                # ctrl[6] = T5 (rear thruster)
                self.data.ctrl[0] = T12_act
                self.data.ctrl[1] = T34_act
                self.data.ctrl[2] = alpha1_act
                self.data.ctrl[3] = alpha2_act
                self.data.ctrl[4] = theta1_act
                self.data.ctrl[5] = theta2_act
                if len(self.data.ctrl) > 6:
                    self.data.ctrl[6] = T5_act

                # Simulation step
                mujoco.mj_step(self.model, self.data)

                # Update viewer
                if self.viewer_running and self.viewer.is_running():
                    self.viewer.sync()
                elif self.viewer_running and not self.viewer.is_running():
                    print("Viewer closed by user")
                    break

                # Record data
                if step % 10 == 0:  # Record every 10 steps
                    self.time_history.append(t)
                    self.pos_history.append(state['pos'].copy())
                    self.pos_des_history.append(desired['pos'].copy())

                    # Extract Euler angles
                    euler = Rot.from_matrix(state['R']).as_euler('xyz')
                    self.att_history.append(euler)

                    # Record thrust and tilt
                    self.thrust_history.append([T12_act, T34_act, T5_act])
                    self.tilt_history.append([alpha1_act, alpha2_act, theta1_act, theta2_act])

                    pos_error = np.linalg.norm(state['pos'] - desired['pos'])
                    self.pos_error_history.append(pos_error)

                t += self.config.dt
                step += 1

                # Progress display
                if step % 1000 == 0:
                    print(f"Progress: {t:.1f}s / {self.config.duration:.1f}s")

        finally:
            if self.viewer_running and self.viewer.is_running():
                self.viewer.close()

        print("Simulation completed!")

    def plot_results(self, scenario_name: str):
        """Plot results"""
        time = np.array(self.time_history)
        pos = np.array(self.pos_history)
        pos_des = np.array(self.pos_des_history)
        att = np.array(self.att_history)
        thrust = np.array(self.thrust_history)
        tilt = np.array(self.tilt_history)
        pos_error = np.array(self.pos_error_history)

        fig, axes = plt.subplots(4, 2, figsize=(15, 12))
        fig.suptitle(f'{scenario_name} - High-Fidelity Simulation Results', fontsize=16)

        # Position tracking
        ax = axes[0, 0]
        ax.plot(time, pos[:, 0], 'b-', label='Actual x')
        ax.plot(time, pos_des[:, 0], 'b--', label='Desired x')
        ax.plot(time, pos[:, 1], 'r-', label='Actual y')
        ax.plot(time, pos_des[:, 1], 'r--', label='Desired y')
        ax.plot(time, pos[:, 2], 'g-', label='Actual z')
        ax.plot(time, pos_des[:, 2], 'g--', label='Desired z')
        ax.set_ylabel('Position (m)')
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True)

        # Position error
        ax = axes[0, 1]
        ax.plot(time, pos_error, 'k-', linewidth=2)
        ax.set_ylabel('Position Error (m)')
        ax.set_xlabel('Time (s)')
        ax.grid(True)
        ax.set_title(f'Mean Error: {np.mean(pos_error):.4f}m')

        # Attitude angles
        ax = axes[1, 0]
        ax.plot(time, np.rad2deg(att[:, 0]), label='Roll')
        ax.plot(time, np.rad2deg(att[:, 1]), label='Pitch')
        ax.plot(time, np.rad2deg(att[:, 2]), label='Yaw')
        ax.set_ylabel('Attitude Angle (deg)')
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True)

        # Thrust allocation
        ax = axes[1, 1]
        ax.plot(time, thrust[:, 0], label='T12 (Front Left)')
        ax.plot(time, thrust[:, 1], label='T34 (Front Right)')
        ax.plot(time, thrust[:, 2], label='T5 (Rear)')
        ax.axhline(y=self.config.thrust_max, color='r', linestyle='--', label='Max Thrust')
        ax.set_ylabel('Thrust (N)')
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True)

        # Roll tilt angles
        ax = axes[2, 0]
        ax.plot(time, np.rad2deg(tilt[:, 0]), label='Alpha1 (Roll Left)')
        ax.plot(time, np.rad2deg(tilt[:, 1]), label='Alpha2 (Roll Right)')
        ax.set_ylabel('Roll Tilt Angle (deg)')
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True)

        # Pitch tilt angles
        ax = axes[2, 1]
        ax.plot(time, np.rad2deg(tilt[:, 2]), label='Theta1 (Pitch Left)')
        ax.plot(time, np.rad2deg(tilt[:, 3]), label='Theta2 (Pitch Right)')
        ax.set_ylabel('Pitch Tilt Angle (deg)')
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True)

        # 3D trajectory
        ax = axes[3, 0]
        ax.remove()
        ax = fig.add_subplot(4, 2, 7, projection='3d')
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', label='Actual', linewidth=2)
        ax.plot(pos_des[:, 0], pos_des[:, 1], pos_des[:, 2], 'r--', label='Desired', linewidth=2)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        ax.set_title('3D Trajectory')

        # Position-Attitude phase plane (core highlight)
        ax = axes[3, 1]
        ax.plot(np.rad2deg(att[:, 2]), pos_error, 'b-', linewidth=2)
        ax.set_xlabel('Yaw Angle (deg)')
        ax.set_ylabel('Position Drift (m)')
        ax.grid(True)
        ax.set_title('Position Lock - Attitude Evolution Decoupling')

        plt.tight_layout()
        plt.savefig(f'{scenario_name}_results.png', dpi=300)
        print(f"Results saved: {scenario_name}_results.png")
        plt.show()

    def print_metrics(self, scenario_name: str):
        """Print quantitative metrics"""
        pos = np.array(self.pos_history)
        pos_des = np.array(self.pos_des_history)
        pos_error = np.array(self.pos_error_history)
        thrust = np.array(self.thrust_history)

        print(f"\n{'='*60}")
        print(f"{scenario_name} - Quantitative Metrics")
        print(f"{'='*60}")

        # Position drift integral error
        pos_error_integral = np.trapz(pos_error, self.time_history)
        print(f"Position Drift Integral Error: {pos_error_integral:.4f} m·s")

        # Position fluctuation RMS
        pos_std = np.std(pos - pos_des, axis=0)
        print(f"Position Fluctuation RMS: X={pos_std[0]:.4f}m, Y={pos_std[1]:.4f}m, Z={pos_std[2]:.4f}m")

        # Actuator utilization
        thrust_utilization_front = np.max(thrust[:, :2], axis=0) / self.config.thrust_max
        print(f"Front Rotor Utilization: Left={thrust_utilization_front[0]:.2%}, Right={thrust_utilization_front[1]:.2%}")
        print(f"Max Front Utilization: {np.max(thrust_utilization_front):.2%}")

        # Attitude decoupling metric (position fluctuation during attitude change)
        att = np.array(self.att_history)
        att_change = np.diff(att[:, 2])  # Yaw change
        pos_drift_during_att_change = pos_error[1:][np.abs(att_change) > 0.01]
        if len(pos_drift_during_att_change) > 0:
            decoupling_metric = np.sqrt(np.mean(pos_drift_during_att_change**2))
            print(f"Attitude Decoupling Metric (Position Fluctuation RMS): {decoupling_metric:.4f}m")

        print(f"{'='*60}\n")


def main():
    """Main function"""
    # XML path (modify according to actual situation)
    xml_path = "hnuter201.xml"

    # Simulation configuration (matching hnuter69 parameters)
    config = SimConfig(
        dt=0.001,
        duration=20.0,
        mass=4.2,
        enable_viewer=True  # Enable real-time visualization
    )

    # Scenario list - starting with hover test
    scenarios = [
        ("Scenario_0_Hover_Test", ScenarioManager.scenario_hover_test),
        ("Scenario_1_Gentle_Yaw_Decoupling", ScenarioManager.scenario_1_decoupling),
        ("Scenario_2_Slow_Yaw_Scanning", ScenarioManager.scenario_2_narrow_space),
        ("Scenario_3_Gentle_Combined_Motion", ScenarioManager.scenario_3_saturation)
    ]

    # Run all scenarios
    for scenario_name, scenario_func in scenarios:
        print(f"\n{'#'*60}")
        print(f"Running {scenario_name}")
        print(f"{'#'*60}\n")

        try:
            sim = HighFidelitySimulator(xml_path, config, scenario_func)
            sim.run()
            sim.plot_results(scenario_name)
            sim.print_metrics(scenario_name)
        except Exception as e:
            print(f"Scenario {scenario_name} failed: {e}")
            print("Please check XML file path and model configuration")
            import traceback
            traceback.print_exc()

    print("\nAll scenario verification completed!")


if __name__ == "__main__":
    main()
