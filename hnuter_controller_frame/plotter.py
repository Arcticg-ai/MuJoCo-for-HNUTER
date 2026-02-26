import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Ensure proper display of negative signs
plt.rcParams['axes.unicode_minus'] = False    # Correct negative sign display

def find_latest_log_file(log_dir: str = 'logs') -> str:
    """
    Find the latest log file in the logs directory
    """
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"Log directory {log_dir} does not exist")
    
    # Get all drone_log csv files
    log_files = [f for f in os.listdir(log_dir) 
                 if f.startswith('drone_log_') and f.endswith('.csv')]
    
    if not log_files:
        raise FileNotFoundError("No log files found")
    
    # Sort by modification time (newest first)
    log_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
    latest_file = os.path.join(log_dir, log_files[0])
    
    print(f"Found latest log file: {latest_file}")
    return latest_file

def load_log_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess log data
    """
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Preprocess timestamp: convert to relative time (start from 0)
    df['relative_time'] = df['timestamp'] - df['timestamp'].iloc[0]
    
    # Convert radians to degrees (attitude angles)
    attitude_cols = ['roll', 'pitch', 'yaw', 
                  'target_roll', 'target_pitch', 'target_yaw']
    
    for col in attitude_cols:
        if col in df.columns:
            df[col + '_deg'] = np.degrees(df[col])
    
    # 处理倾转角度：使用实际角度或命令角度，如果都不存在则使用默认值
    tilt_cols = ['alpha1', 'alpha2', 'theta1', 'theta2']
    tilt_cmd_cols = ['alpha1_cmd', 'alpha2_cmd', 'theta1_cmd', 'theta2_cmd']
    tilt_actual_cols = ['alpha1_actual', 'alpha2_actual', 'theta1_actual', 'theta2_actual']
    
    # 为每个倾转角度列设置值
    for i, col in enumerate(tilt_cols):
        cmd_col = tilt_cmd_cols[i]
        actual_col = tilt_actual_cols[i]

        if actual_col in df.columns:
            # 使用实际角度
            df[col] = df[actual_col]
            df[col + '_deg'] = np.degrees(df[col])
        elif cmd_col in df.columns:
            # 使用命令角度
            df[col] = df[cmd_col]
            df[col + '_deg'] = np.degrees(df[col])
        else:
            # 使用默认值
            print(f"Warning: Columns '{actual_col}' and '{cmd_col}' not found in log file, using default value 0.0 for '{col}'")
            df[col] = 0.0
            df[col + '_deg'] = 0.0

        # 同时保留期望倾转角的度数列（用于与实际值对比绘图）
        if cmd_col in df.columns:
            df[cmd_col + '_deg'] = np.degrees(df[cmd_col])
    
    # Check required columns (added T12/T34/T5 thrust columns)
    # 核心必需列，这些列必须存在
    core_required_cols = [
        'relative_time', 'pos_x', 'pos_y', 'pos_z',
        'target_x', 'target_y', 'target_z',
        'roll', 'pitch', 'yaw',
        'target_roll', 'target_pitch', 'target_yaw',
        'f_body_x', 'f_body_y', 'f_body_z',  # Thrust in body frame
        'T12', 'T34', 'T5'  # Propeller thrust values
    ]
    
    # 检查核心必需列
    core_missing_cols = [col for col in core_required_cols if col not in df.columns]
    if core_missing_cols:
        raise ValueError(f"Log file missing required core columns: {core_missing_cols}")
    
    # 检查alpha1, alpha2, theta1, theta2列，如果不存在，添加默认值
    tilt_cols = ['alpha1', 'alpha2', 'theta1', 'theta2']
    for col in tilt_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in log file, using default value 0.0")
            df[col] = 0.0
            df[col + '_deg'] = 0.0
    
    # 检查力矩列，如果不存在，添加默认值
    torque_cols = ['tau_x', 'tau_y', 'tau_z']
    for col in torque_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in log file, using default value 0.0")
            df[col] = 0.0
    
    print(f"Successfully loaded log data with {len(df)} records")
    print(f"Time range: {df['relative_time'].min():.2f} ~ {df['relative_time'].max():.2f} seconds")
    
    return df

def plot_drone_data(df: pd.DataFrame, save_path: str = None):
    """
    Plot drone data (5 subplots: position, attitude, tilt angles, body thrust, propeller thrust)
    """
    # Create figure with adjusted size for 5 subplots
    fig = plt.figure(figsize=(18, 25))
    fig.suptitle('Tilt-Rotor UAV Flight Data Visualization', fontsize=16, fontweight='bold')
    
    # -------------------------- Subplot 1: 3-Axis Position (Actual vs Desired) --------------------------
    ax1 = plt.subplot(5, 1, 1)
    
    # X position
    ax1.plot(df['relative_time'], df['pos_x'], 'b-', linewidth=2, label='Actual X Position (m)')
    ax1.plot(df['relative_time'], df['target_x'], 'b--', linewidth=1.5, label='Desired X Position (m)')
    
    # Y position
    ax1.plot(df['relative_time'], df['pos_y'], 'g-', linewidth=2, label='Actual Y Position (m)')
    ax1.plot(df['relative_time'], df['target_y'], 'g--', linewidth=1.5, label='Desired Y Position (m)')
    
    # Z position
    ax1.plot(df['relative_time'], df['pos_z'], 'r-', linewidth=2, label='Actual Z Position (m)')
    ax1.plot(df['relative_time'], df['target_z'], 'r--', linewidth=1.5, label='Desired Z Position (m)')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.set_title('3-Axis Position Tracking (Actual vs Desired)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_xlim(0, df['relative_time'].max())
    
    # -------------------------- Subplot 2: 3-Axis Attitude (Actual vs Desired) --------------------------
    ax2 = plt.subplot(5, 1, 2)
    
    # Roll angle
    ax2.plot(df['relative_time'], df['roll_deg'], 'b-', linewidth=2, label='Actual Roll (°)')
    ax2.plot(df['relative_time'], df['target_roll_deg'], 'b--', linewidth=1.5, label='Desired Roll (°)')
    
    # Pitch angle
    ax2.plot(df['relative_time'], df['pitch_deg'], 'g-', linewidth=2, label='Actual Pitch (°)')
    ax2.plot(df['relative_time'], df['target_pitch_deg'], 'g--', linewidth=1.5, label='Desired Pitch (°)')
    
    # Yaw angle
    ax2.plot(df['relative_time'], df['yaw_deg'], 'r-', linewidth=2, label='Actual Yaw (°)')
    ax2.plot(df['relative_time'], df['target_yaw_deg'], 'r--', linewidth=1.5, label='Desired Yaw (°)')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Attitude Angle (°)')
    ax2.set_title('3-Axis Attitude Tracking (Actual vs Desired)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    ax2.set_xlim(0, df['relative_time'].max())
    
    # -------------------------- Subplot 3: Four Tilt Angles --------------------------
    ax3 = plt.subplot(5, 1, 3)
    
    # Alpha1 (roll right tilt)
    ax3.plot(df['relative_time'], df['alpha1_deg'], 'b-', linewidth=2, label='Alpha1 (Roll Right Tilt) (°)')
    
    # Alpha2 (roll left tilt)
    ax3.plot(df['relative_time'], df['alpha2_deg'], 'g-', linewidth=2, label='Alpha2 (Roll Left Tilt) (°)')
    
    # Theta1 (pitch right tilt)
    ax3.plot(df['relative_time'], df['theta1_deg'], 'r-', linewidth=2, label='Theta1 (Pitch Right Tilt) (°)')
    
    # Theta2 (pitch left tilt)
    ax3.plot(df['relative_time'], df['theta2_deg'], 'm-', linewidth=2, label='Theta2 (Pitch Left Tilt) (°)')
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Tilt Angle (°)')
    ax3.set_title('Four Tilt Angles Variation', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')
    ax3.set_xlim(0, df['relative_time'].max())
    
    # -------------------------- Subplot 4: 3-Axis Thrust (Body Frame) --------------------------
    ax4 = plt.subplot(5, 1, 4)
    
    # X-axis thrust (body frame)
    ax4.plot(df['relative_time'], df['f_body_x'], 'b-', linewidth=2, label='Thrust X (Body Frame) (N)')
    
    # Y-axis thrust (body frame)
    ax4.plot(df['relative_time'], df['f_body_y'], 'g-', linewidth=2, label='Thrust Y (Body Frame) (N)')
    
    # Z-axis thrust (body frame)
    ax4.plot(df['relative_time'], df['f_body_z'], 'r-', linewidth=2, label='Thrust Z (Body Frame) (N)')
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Thrust (N)')
    ax4.set_title('3-Axis Thrust in Body Frame', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best')
    ax4.set_xlim(0, df['relative_time'].max())
    
    # -------------------------- Subplot 5: Propeller Thrust (T12/T34/T5) --------------------------
    ax5 = plt.subplot(5, 1, 5)
    
    # T12 (Front-left rotor group thrust)
    ax5.plot(df['relative_time'], df['T12'], 'b-', linewidth=2, label='T12 (Front-left Rotor Group) (N)')
    
    # T34 (Front-right rotor group thrust)
    ax5.plot(df['relative_time'], df['T34'], 'g-', linewidth=2, label='T34 (Front-right Rotor Group) (N)')
    
    # T5 (Rear propeller thrust)
    ax5.plot(df['relative_time'], df['T5'], 'r-', linewidth=2, label='T5 (Rear Propeller) (N)')
    
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Thrust (N)')
    ax5.set_title('Propeller Group Thrust (T12/T34/T5)', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='best')
    ax5.set_xlim(0, df['relative_time'].max())
    
    return df

def plot_drone_data_and_save(df: pd.DataFrame, save_path: str = None):
    """
    Plot drone data and save the figure
    """
    # Create figure with adjusted size for 5 subplots
    fig = plt.figure(figsize=(18, 25))
    fig.suptitle('Tilt-Rotor UAV Flight Data Visualization', fontsize=16, fontweight='bold')
    
    # -------------------------- Subplot 1: 3-Axis Position (Actual vs Desired) --------------------------
    ax1 = plt.subplot(5, 1, 1)
    
    # X position
    ax1.plot(df['relative_time'], df['pos_x'], 'b-', linewidth=2, label='Actual X Position (m)')
    ax1.plot(df['relative_time'], df['target_x'], 'b--', linewidth=1.5, label='Desired X Position (m)')
    
    # Y position
    ax1.plot(df['relative_time'], df['pos_y'], 'g-', linewidth=2, label='Actual Y Position (m)')
    ax1.plot(df['relative_time'], df['target_y'], 'g--', linewidth=1.5, label='Desired Y Position (m)')
    
    # Z position
    ax1.plot(df['relative_time'], df['pos_z'], 'r-', linewidth=2, label='Actual Z Position (m)')
    ax1.plot(df['relative_time'], df['target_z'], 'r--', linewidth=1.5, label='Desired Z Position (m)')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.set_title('3-Axis Position Tracking (Actual vs Desired)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_xlim(0, df['relative_time'].max())
    
    # -------------------------- Subplot 2: 3-Axis Attitude (Actual vs Desired) --------------------------
    ax2 = plt.subplot(5, 1, 2)
    
    # Roll angle
    ax2.plot(df['relative_time'], df['roll_deg'], 'b-', linewidth=2, label='Actual Roll (°)')
    ax2.plot(df['relative_time'], df['target_roll_deg'], 'b--', linewidth=1.5, label='Desired Roll (°)')
    
    # Pitch angle
    ax2.plot(df['relative_time'], df['pitch_deg'], 'g-', linewidth=2, label='Actual Pitch (°)')
    ax2.plot(df['relative_time'], df['target_pitch_deg'], 'g--', linewidth=1.5, label='Desired Pitch (°)')
    
    # Yaw angle
    ax2.plot(df['relative_time'], df['yaw_deg'], 'r-', linewidth=2, label='Actual Yaw (°)')
    ax2.plot(df['relative_time'], df['target_yaw_deg'], 'r--', linewidth=1.5, label='Desired Yaw (°)')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Attitude Angle (°)')
    ax2.set_title('3-Axis Attitude Tracking (Actual vs Desired)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    ax2.set_xlim(0, df['relative_time'].max())
    
    # -------------------------- Subplot 3: Four Tilt Angles --------------------------
    ax3 = plt.subplot(5, 1, 3)
    
    # Alpha1 (roll right tilt)
    ax3.plot(df['relative_time'], df['alpha1_deg'], 'b-', linewidth=2, label='Alpha1 (Roll Right Tilt) (°)')
    
    # Alpha2 (roll left tilt)
    ax3.plot(df['relative_time'], df['alpha2_deg'], 'g-', linewidth=2, label='Alpha2 (Roll Left Tilt) (°)')
    
    # Theta1 (pitch right tilt)
    ax3.plot(df['relative_time'], df['theta1_deg'], 'r-', linewidth=2, label='Theta1 (Pitch Right Tilt) (°)')
    
    # Theta2 (pitch left tilt)
    ax3.plot(df['relative_time'], df['theta2_deg'], 'm-', linewidth=2, label='Theta2 (Pitch Left Tilt) (°)')
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Tilt Angle (°)')
    ax3.set_title('Four Tilt Angles Variation', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')
    ax3.set_xlim(0, df['relative_time'].max())
    
    # -------------------------- Subplot 4: 3-Axis Thrust (Body Frame) --------------------------
    ax4 = plt.subplot(5, 1, 4)
    
    # X-axis thrust (body frame)
    ax4.plot(df['relative_time'], df['f_body_x'], 'b-', linewidth=2, label='Thrust X (Body Frame) (N)')
    
    # Y-axis thrust (body frame)
    ax4.plot(df['relative_time'], df['f_body_y'], 'g-', linewidth=2, label='Thrust Y (Body Frame) (N)')
    
    # Z-axis thrust (body frame)
    ax4.plot(df['relative_time'], df['f_body_z'], 'r-', linewidth=2, label='Thrust Z (Body Frame) (N)')
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Thrust (N)')
    ax4.set_title('3-Axis Thrust in Body Frame', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best')
    ax4.set_xlim(0, df['relative_time'].max())
    
    # -------------------------- Subplot 5: Propeller Thrust (T12/T34/T5) --------------------------
    ax5 = plt.subplot(5, 1, 5)
    
    # T12 (Front-left rotor group thrust)
    ax5.plot(df['relative_time'], df['T12'], 'b-', linewidth=2, label='T12 (Front-left Rotor Group) (N)')
    
    # T34 (Front-right rotor group thrust)
    ax5.plot(df['relative_time'], df['T34'], 'g-', linewidth=2, label='T34 (Front-right Rotor Group) (N)')
    
    # T5 (Rear propeller thrust)
    ax5.plot(df['relative_time'], df['T5'], 'r-', linewidth=2, label='T5 (Rear Propeller) (N)')
    
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Thrust (N)')
    ax5.set_title('Propeller Group Thrust (T12/T34/T5)', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='best')
    ax5.set_xlim(0, df['relative_time'].max())
    
    # Adjust subplot spacing to prevent overlap
    plt.tight_layout()
    
    # Save plot with high resolution
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        # Auto-generate save path with timestamp, using different name pattern
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'logs/frame_analysis_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Show interactive plot
    plt.show()


def plot_torque_data(df: pd.DataFrame, save_path: str = None):
    """
    Plot torque data
    """
    # Create figure
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Tilt-Rotor UAV Torque Data', fontsize=16, fontweight='bold')
    
    # -------------------------- Subplot 1: 3-Axis Torque --------------------------
    ax1 = plt.subplot(1, 1, 1)
    
    # X-axis torque
    ax1.plot(df['relative_time'], df['tau_x'], 'b-', linewidth=2, label='Torque X (Nm)')
    
    # Y-axis torque
    ax1.plot(df['relative_time'], df['tau_y'], 'g-', linewidth=2, label='Torque Y (Nm)')
    
    # Z-axis torque
    ax1.plot(df['relative_time'], df['tau_z'], 'r-', linewidth=2, label='Torque Z (Nm)')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Torque (Nm)')
    ax1.set_title('3-Axis Control Torque', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_xlim(0, df['relative_time'].max())
    
    # Adjust subplot spacing to prevent overlap
    plt.tight_layout()
    
    # Save plot with high resolution
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Torque plot saved to: {save_path}")
    else:
        # Auto-generate save path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'logs/torque_analysis_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Torque plot saved to: {save_path}")
    
    # Close the plot to free memory
    plt.close()


def plot_rotation_matrix_data(df: pd.DataFrame, save_path: str = None):
    """
    Plot rotation matrix data
    """
    # Check if rotation matrix columns exist
    has_rotation_matrix = all(col in df.columns for col in ['R11', 'R12', 'R13', 'R21', 'R22', 'R23', 'R31', 'R32', 'R33'])
    has_target_rotation_matrix = all(col in df.columns for col in ['R_des11', 'R_des12', 'R_des13', 'R_des21', 'R_des22', 'R_des23', 'R_des31', 'R_des32', 'R_des33'])
    
    if not (has_rotation_matrix or has_target_rotation_matrix):
        print("Warning: No rotation matrix data found in log file, skipping rotation matrix plot.")
        return
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(18, 15))
    fig.suptitle('Tilt-Rotor UAV Rotation Matrix Data', fontsize=16, fontweight='bold')
    
    # -------------------------- Subplot 1: Current Rotation Matrix --------------------------
    ax1 = plt.subplot(2, 1, 1)
    
    if has_rotation_matrix:
        # R11, R12, R13 (first row)
        ax1.plot(df['relative_time'], df['R11'], 'b-', linewidth=2, label='R11')
        ax1.plot(df['relative_time'], df['R12'], 'g-', linewidth=2, label='R12')
        ax1.plot(df['relative_time'], df['R13'], 'r-', linewidth=2, label='R13')
        
        # R21, R22, R23 (second row)
        ax1.plot(df['relative_time'], df['R21'], 'c-', linewidth=2, label='R21')
        ax1.plot(df['relative_time'], df['R22'], 'm-', linewidth=2, label='R22')
        ax1.plot(df['relative_time'], df['R23'], 'y-', linewidth=2, label='R23')
        
        # R31, R32, R33 (third row)
        ax1.plot(df['relative_time'], df['R31'], 'k-', linewidth=2, label='R31')
        ax1.plot(df['relative_time'], df['R32'], 'b--', linewidth=2, label='R32')
        ax1.plot(df['relative_time'], df['R33'], 'g--', linewidth=2, label='R33')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Rotation Matrix Elements')
    ax1.set_title('Current Rotation Matrix Elements', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_xlim(0, df['relative_time'].max())
    
    # -------------------------- Subplot 2: Target Rotation Matrix --------------------------
    ax2 = plt.subplot(2, 1, 2)
    
    if has_target_rotation_matrix:
        # R_des11, R_des12, R_des13 (first row)
        ax2.plot(df['relative_time'], df['R_des11'], 'b-', linewidth=2, label='R_des11')
        ax2.plot(df['relative_time'], df['R_des12'], 'g-', linewidth=2, label='R_des12')
        ax2.plot(df['relative_time'], df['R_des13'], 'r-', linewidth=2, label='R_des13')
        
        # R_des21, R_des22, R_des23 (second row)
        ax2.plot(df['relative_time'], df['R_des21'], 'c-', linewidth=2, label='R_des21')
        ax2.plot(df['relative_time'], df['R_des22'], 'm-', linewidth=2, label='R_des22')
        ax2.plot(df['relative_time'], df['R_des23'], 'y-', linewidth=2, label='R_des23')
        
        # R_des31, R_des32, R_des33 (third row)
        ax2.plot(df['relative_time'], df['R_des31'], 'k-', linewidth=2, label='R_des31')
        ax2.plot(df['relative_time'], df['R_des32'], 'b--', linewidth=2, label='R_des32')
        ax2.plot(df['relative_time'], df['R_des33'], 'g--', linewidth=2, label='R_des33')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Rotation Matrix Elements')
    ax2.set_title('Target Rotation Matrix Elements', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    ax2.set_xlim(0, df['relative_time'].max())
    
    # Adjust subplot spacing to prevent overlap
    plt.tight_layout()
    
    # Save plot with high resolution
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Rotation matrix plot saved to: {save_path}")
    else:
        # Auto-generate save path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'logs/rotation_matrix_analysis_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Rotation matrix plot saved to: {save_path}")
    
    # Close the plot to free memory
    plt.close()


def plot_u_values_data(df: pd.DataFrame, save_path: str = None):
    """
    Plot u1 to u7 values
    """
    # Check if u1 to u7 columns exist
    has_u_values = all(col in df.columns for col in ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7'])
    
    if not has_u_values:
        print("Warning: No u1 to u7 data found in log file, skipping u values plot.")
        return
    
    # Create figure
    fig = plt.figure(figsize=(18, 15))
    fig.suptitle('Tilt-Rotor UAV u1 to u7 Values', fontsize=16, fontweight='bold')
    
    # -------------------------- Subplot 1: u1 to u7 Values --------------------------
    ax1 = plt.subplot(1, 1, 1)
    
    # Plot u1 to u7
    ax1.plot(df['relative_time'], df['u1'], 'b-', linewidth=2, label='u1')
    ax1.plot(df['relative_time'], df['u2'], 'g-', linewidth=2, label='u2')
    ax1.plot(df['relative_time'], df['u3'], 'r-', linewidth=2, label='u3')
    ax1.plot(df['relative_time'], df['u4'], 'c-', linewidth=2, label='u4')
    ax1.plot(df['relative_time'], df['u5'], 'm-', linewidth=2, label='u5')
    ax1.plot(df['relative_time'], df['u6'], 'y-', linewidth=2, label='u6')
    ax1.plot(df['relative_time'], df['u7'], 'k-', linewidth=2, label='u7')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('u Values')
    ax1.set_title('u1 to u7 Control Values', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_xlim(0, df['relative_time'].max())
    
    # Adjust subplot spacing to prevent overlap
    plt.tight_layout()
    
    # Save plot with high resolution
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"u values plot saved to: {save_path}")
    else:
        # Auto-generate save path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'logs/u_values_analysis_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"u values plot saved to: {save_path}")
    
    # Close the plot to free memory
    plt.close()


def main():
    """
    Main function - Tilt-Rotor UAV Log Analysis
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tilt-Rotor UAV Log Data Analysis Tool')
    parser.add_argument('--file', '-f', type=str, help='Path to log file (optional, uses latest log by default)')
    
    args = parser.parse_args()
    
    try:
        # Get log file path
        if args.file:
            log_file = args.file
            if not os.path.exists(log_file):
                raise FileNotFoundError(f"Specified log file does not exist: {log_file}")
            print(f"Using specified log file: {log_file}")
        else:
            log_file = find_latest_log_file()
        
        # Load and preprocess log data
        df = load_log_data(log_file)
        
        # Generate and save three PNG files according to requirements
        plot_drone_data_and_save_all(df)
        
        print("\nData analysis completed successfully! Generated three PNG files:")
        print("1. Position, attitude angles, and rotation matrix tracking")
        print("2. Force and torque output")
        print("3. Tilt angles, T12/T34/T5, and u1-u7 tracking")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


def plot_position_attitude_rotation(df: pd.DataFrame, save_path: str = None):
    """
    Plot position, attitude angles, and rotation matrix elements
    """
    # Create figure
    fig = plt.figure(figsize=(18, 25))
    fig.suptitle('Tilt-Rotor UAV Position, Attitude, and Rotation Matrix', fontsize=16, fontweight='bold')
    
    # -------------------------- Subplot 1: 3-Axis Position (Actual vs Desired) --------------------------
    ax1 = plt.subplot(4, 1, 1)
    
    # X position
    ax1.plot(df['relative_time'], df['pos_x'], 'b-', linewidth=2, label='Actual X Position (m)')
    ax1.plot(df['relative_time'], df['target_x'], 'b--', linewidth=1.5, label='Desired X Position (m)')
    
    # Y position
    ax1.plot(df['relative_time'], df['pos_y'], 'g-', linewidth=2, label='Actual Y Position (m)')
    ax1.plot(df['relative_time'], df['target_y'], 'g--', linewidth=1.5, label='Desired Y Position (m)')
    
    # Z position
    ax1.plot(df['relative_time'], df['pos_z'], 'r-', linewidth=2, label='Actual Z Position (m)')
    ax1.plot(df['relative_time'], df['target_z'], 'r--', linewidth=1.5, label='Desired Z Position (m)')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.set_title('3-Axis Position Tracking (Actual vs Desired)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_xlim(0, df['relative_time'].max())
    
    # -------------------------- Subplot 2: 3-Axis Attitude (Actual vs Desired) --------------------------
    ax2 = plt.subplot(4, 1, 2)
    
    # Roll angle
    ax2.plot(df['relative_time'], df['roll_deg'], 'b-', linewidth=2, label='Actual Roll (°)')
    ax2.plot(df['relative_time'], df['target_roll_deg'], 'b--', linewidth=1.5, label='Desired Roll (°)')
    
    # Pitch angle
    ax2.plot(df['relative_time'], df['pitch_deg'], 'g-', linewidth=2, label='Actual Pitch (°)')
    ax2.plot(df['relative_time'], df['target_pitch_deg'], 'g--', linewidth=1.5, label='Desired Pitch (°)')
    
    # Yaw angle
    ax2.plot(df['relative_time'], df['yaw_deg'], 'r-', linewidth=2, label='Actual Yaw (°)')
    ax2.plot(df['relative_time'], df['target_yaw_deg'], 'r--', linewidth=1.5, label='Desired Yaw (°)')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Attitude Angle (°)')
    ax2.set_title('3-Axis Attitude Tracking (Actual vs Desired)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    ax2.set_xlim(0, df['relative_time'].max())
    
    # -------------------------- Subplot 3: Current Rotation Matrix Elements --------------------------
    ax3 = plt.subplot(4, 1, 3)
    
    # Check if rotation matrix columns exist
    has_rotation_matrix = all(col in df.columns for col in ['R11', 'R12', 'R13', 'R21', 'R22', 'R23', 'R31', 'R32', 'R33'])
    
    if has_rotation_matrix:
        # R11, R12, R13 (first row)
        ax3.plot(df['relative_time'], df['R11'], 'b-', linewidth=2, label='R11')
        ax3.plot(df['relative_time'], df['R12'], 'g-', linewidth=2, label='R12')
        ax3.plot(df['relative_time'], df['R13'], 'r-', linewidth=2, label='R13')
        
        # R21, R22, R23 (second row)
        ax3.plot(df['relative_time'], df['R21'], 'c-', linewidth=2, label='R21')
        ax3.plot(df['relative_time'], df['R22'], 'm-', linewidth=2, label='R22')
        ax3.plot(df['relative_time'], df['R23'], 'y-', linewidth=2, label='R23')
        
        # R31, R32, R33 (third row)
        ax3.plot(df['relative_time'], df['R31'], 'k-', linewidth=2, label='R31')
        ax3.plot(df['relative_time'], df['R32'], 'b--', linewidth=2, label='R32')
        ax3.plot(df['relative_time'], df['R33'], 'g--', linewidth=2, label='R33')
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Rotation Matrix Elements')
    ax3.set_title('Current Rotation Matrix Elements', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')
    ax3.set_xlim(0, df['relative_time'].max())
    
    # -------------------------- Subplot 4: Target Rotation Matrix Elements --------------------------
    ax4 = plt.subplot(4, 1, 4)
    
    # Check if target rotation matrix columns exist
    has_target_rotation_matrix = all(col in df.columns for col in ['R_des11', 'R_des12', 'R_des13', 'R_des21', 'R_des22', 'R_des23', 'R_des31', 'R_des32', 'R_des33'])
    
    if has_target_rotation_matrix:
        # R_des11, R_des12, R_des13 (first row)
        ax4.plot(df['relative_time'], df['R_des11'], 'b-', linewidth=2, label='R_des11')
        ax4.plot(df['relative_time'], df['R_des12'], 'g-', linewidth=2, label='R_des12')
        ax4.plot(df['relative_time'], df['R_des13'], 'r-', linewidth=2, label='R_des13')
        
        # R_des21, R_des22, R_des23 (second row)
        ax4.plot(df['relative_time'], df['R_des21'], 'c-', linewidth=2, label='R_des21')
        ax4.plot(df['relative_time'], df['R_des22'], 'm-', linewidth=2, label='R_des22')
        ax4.plot(df['relative_time'], df['R_des23'], 'y-', linewidth=2, label='R_des23')
        
        # R_des31, R_des32, R_des33 (third row)
        ax4.plot(df['relative_time'], df['R_des31'], 'k-', linewidth=2, label='R_des31')
        ax4.plot(df['relative_time'], df['R_des32'], 'b--', linewidth=2, label='R_des32')
        ax4.plot(df['relative_time'], df['R_des33'], 'g--', linewidth=2, label='R_des33')
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Rotation Matrix Elements')
    ax4.set_title('Target Rotation Matrix Elements', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best')
    ax4.set_xlim(0, df['relative_time'].max())
    
    # Adjust subplot spacing to prevent overlap
    plt.tight_layout()
    
    # Save plot with high resolution
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Position, attitude, and rotation matrix plot saved to: {save_path}")
    else:
        # Auto-generate save path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'logs/position_attitude_rotation_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Position, attitude, and rotation matrix plot saved to: {save_path}")
    
    # Close the plot to free memory
    plt.close()


def plot_force_torque(df: pd.DataFrame, save_path: str = None):
    """
    Plot force and torque data
    """
    # Create figure
    fig = plt.figure(figsize=(18, 20))
    fig.suptitle('Tilt-Rotor UAV Force and Torque', fontsize=16, fontweight='bold')
    
    # -------------------------- Subplot 1: 3-Axis Thrust (Body Frame) --------------------------
    ax1 = plt.subplot(3, 1, 1)
    
    # X-axis thrust (body frame)
    ax1.plot(df['relative_time'], df['f_body_x'], 'b-', linewidth=2, label='Thrust X (Body Frame) (N)')
    
    # Y-axis thrust (body frame)
    ax1.plot(df['relative_time'], df['f_body_y'], 'g-', linewidth=2, label='Thrust Y (Body Frame) (N)')
    
    # Z-axis thrust (body frame)
    ax1.plot(df['relative_time'], df['f_body_z'], 'r-', linewidth=2, label='Thrust Z (Body Frame) (N)')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Thrust (N)')
    ax1.set_title('3-Axis Thrust in Body Frame', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_xlim(0, df['relative_time'].max())
    
    # -------------------------- Subplot 2: 3-Axis Torque --------------------------
    ax2 = plt.subplot(3, 1, 2)
    
    # X-axis torque
    ax2.plot(df['relative_time'], df['tau_x'], 'b-', linewidth=2, label='Torque X (Nm)')
    
    # Y-axis torque
    ax2.plot(df['relative_time'], df['tau_y'], 'g-', linewidth=2, label='Torque Y (Nm)')
    
    # Z-axis torque
    ax2.plot(df['relative_time'], df['tau_z'], 'r-', linewidth=2, label='Torque Z (Nm)')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Torque (Nm)')
    ax2.set_title('3-Axis Control Torque', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    ax2.set_xlim(0, df['relative_time'].max())
    
    # -------------------------- Subplot 3: 3-Axis Force (World Frame) --------------------------
    ax3 = plt.subplot(3, 1, 3)
    
    # X-axis force (world frame)
    ax3.plot(df['relative_time'], df['f_world_x'], 'b-', linewidth=2, label='Force X (World Frame) (N)')
    
    # Y-axis force (world frame)
    ax3.plot(df['relative_time'], df['f_world_y'], 'g-', linewidth=2, label='Force Y (World Frame) (N)')
    
    # Z-axis force (world frame)
    ax3.plot(df['relative_time'], df['f_world_z'], 'r-', linewidth=2, label='Force Z (World Frame) (N)')
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Force (N)')
    ax3.set_title('3-Axis Force in World Frame', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')
    ax3.set_xlim(0, df['relative_time'].max())
    
    # Adjust subplot spacing to prevent overlap
    plt.tight_layout()
    
    # Save plot with high resolution
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Force and torque plot saved to: {save_path}")
    else:
        # Auto-generate save path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'logs/force_torque_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Force and torque plot saved to: {save_path}")
    
    # Close the plot to free memory
    plt.close()


def plot_tilt_thrust_u_values(df: pd.DataFrame, save_path: str = None):
    """
    Plot tilt angles, thrust values, and u1 to u7
    """
    # Create figure
    fig = plt.figure(figsize=(18, 25))
    fig.suptitle('Tilt-Rotor UAV Tilt Angles, Thrust, and u1-u7 Values', fontsize=16, fontweight='bold')
    
    # -------------------------- Subplot 1: Four Tilt Angles --------------------------
    ax1 = plt.subplot(4, 1, 1)
    
    # Alpha1 (roll right tilt)
    ax1.plot(df['relative_time'], df['alpha1_deg'], 'b-', linewidth=2, label='Alpha1 (Roll Right Tilt) (°)')
    
    # Alpha2 (roll left tilt)
    ax1.plot(df['relative_time'], df['alpha2_deg'], 'g-', linewidth=2, label='Alpha2 (Roll Left Tilt) (°)')
    
    # Theta1 (pitch right tilt)
    ax1.plot(df['relative_time'], df['theta1_deg'], 'r-', linewidth=2, label='Theta1 (Pitch Right Tilt) (°)')
    
    # Theta2 (pitch left tilt)
    ax1.plot(df['relative_time'], df['theta2_deg'], 'm-', linewidth=2, label='Theta2 (Pitch Left Tilt) (°)')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Tilt Angle (°)')
    ax1.set_title('Four Tilt Angles Variation', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_xlim(0, df['relative_time'].max())
    
    # -------------------------- Subplot 2: Propeller Thrust (T12/T34/T5) --------------------------
    ax2 = plt.subplot(4, 1, 2)
    
    # T12 (Front-left rotor group thrust)
    ax2.plot(df['relative_time'], df['T12'], 'b-', linewidth=2, label='T12 (Front-left Rotor Group) (N)')
    
    # T34 (Front-right rotor group thrust)
    ax2.plot(df['relative_time'], df['T34'], 'g-', linewidth=2, label='T34 (Front-right Rotor Group) (N)')
    
    # T5 (Rear propeller thrust)
    ax2.plot(df['relative_time'], df['T5'], 'r-', linewidth=2, label='T5 (Rear Propeller) (N)')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Thrust (N)')
    ax2.set_title('Propeller Group Thrust (T12/T34/T5)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    ax2.set_xlim(0, df['relative_time'].max())
    
    # -------------------------- Subplot 3: u1 to u7 Values --------------------------
    ax3 = plt.subplot(4, 1, 3)
    
    # Check if u1 to u7 columns exist
    has_u_values = all(col in df.columns for col in ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7'])
    
    if has_u_values:
        # Plot u1 to u7
        ax3.plot(df['relative_time'], df['u1'], 'b-', linewidth=2, label='u1')
        ax3.plot(df['relative_time'], df['u2'], 'g-', linewidth=2, label='u2')
        ax3.plot(df['relative_time'], df['u3'], 'r-', linewidth=2, label='u3')
        ax3.plot(df['relative_time'], df['u4'], 'c-', linewidth=2, label='u4')
        ax3.plot(df['relative_time'], df['u5'], 'm-', linewidth=2, label='u5')
        ax3.plot(df['relative_time'], df['u6'], 'y-', linewidth=2, label='u6')
        ax3.plot(df['relative_time'], df['u7'], 'k-', linewidth=2, label='u7')
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('u Values')
    ax3.set_title('u1 to u7 Control Values', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')
    ax3.set_xlim(0, df['relative_time'].max())
    
    # -------------------------- Subplot 4: Control Input Vector u --------------------------
    ax4 = plt.subplot(4, 1, 4)
    
    # Check if u control vector columns exist
    has_u_vector = all(col in df.columns for col in ['u_T12', 'u_T34', 'u_T5', 'u_alpha1', 'u_alpha2', 'u_theta1', 'u_theta2'])
    
    if has_u_vector:
        # Plot u control vector components
        ax4.plot(df['relative_time'], df['u_T12'], 'b-', linewidth=2, label='u_T12')
        ax4.plot(df['relative_time'], df['u_T34'], 'g-', linewidth=2, label='u_T34')
        ax4.plot(df['relative_time'], df['u_T5'], 'r-', linewidth=2, label='u_T5')
        ax4.plot(df['relative_time'], np.degrees(df['u_alpha1']), 'c-', linewidth=2, label='u_alpha1 (°)')
        ax4.plot(df['relative_time'], np.degrees(df['u_alpha2']), 'm-', linewidth=2, label='u_alpha2 (°)')
        ax4.plot(df['relative_time'], np.degrees(df['u_theta1']), 'y-', linewidth=2, label='u_theta1 (°)')
        ax4.plot(df['relative_time'], np.degrees(df['u_theta2']), 'k-', linewidth=2, label='u_theta2 (°)')
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Control Input Values')
    ax4.set_title('Control Input Vector u Components', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best')
    ax4.set_xlim(0, df['relative_time'].max())
    
    # Adjust subplot spacing to prevent overlap
    plt.tight_layout()
    
    # Save plot with high resolution
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Tilt angles, thrust, and u values plot saved to: {save_path}")
    else:
        # Auto-generate save path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'logs/tilt_thrust_u_values_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Tilt angles, thrust, and u values plot saved to: {save_path}")
    
    # Close the plot to free memory
    plt.close()


def plot_force_torque_tilt_attitude_combined(df: pd.DataFrame, save_path: str = None):
    """
    将力、力矩、倾转角、姿态角画到一张图里 (4 个子图)
    """
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle('Force, Torque, Tilt Angles & Attitude (Combined)', fontsize=16, fontweight='bold')

    t = df['relative_time']
    t_max = t.max()

    # -------------------------- Subplot 1: 力 (Body Frame) --------------------------
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(t, df['f_body_x'], 'b-', linewidth=2, label='Fx (N)')
    ax1.plot(t, df['f_body_y'], 'g-', linewidth=2, label='Fy (N)')
    ax1.plot(t, df['f_body_z'], 'r-', linewidth=2, label='Fz (N)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Force (N)')
    ax1.set_title('3-Axis Force (Body Frame)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_xlim(0, t_max)

    # -------------------------- Subplot 2: 力矩 --------------------------
    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(t, df['tau_x'], 'b-', linewidth=2, label='τx (Nm)')
    ax2.plot(t, df['tau_y'], 'g-', linewidth=2, label='τy (Nm)')
    ax2.plot(t, df['tau_z'], 'r-', linewidth=2, label='τz (Nm)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Torque (Nm)')
    ax2.set_title('3-Axis Torque', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    ax2.set_xlim(0, t_max)

    # -------------------------- Subplot 3: 倾转角 (实际 vs 期望) --------------------------
    ax3 = plt.subplot(4, 1, 3)
    # 实际倾转角（实线）
    ax3.plot(t, df['alpha1_deg'], 'b-', linewidth=2, label='α1 (°)')
    ax3.plot(t, df['alpha2_deg'], 'g-', linewidth=2, label='α2 (°)')
    ax3.plot(t, df['theta1_deg'], 'r-', linewidth=2, label='θ1 (°)')
    ax3.plot(t, df['theta2_deg'], 'm-', linewidth=2, label='θ2 (°)')

    # 期望倾转角（虚线）- 如果数据存在
    if 'alpha1_cmd_deg' in df.columns:
        ax3.plot(t, df['alpha1_cmd_deg'], 'b--', linewidth=1.5, label='α1 cmd (°)')
    if 'alpha2_cmd_deg' in df.columns:
        ax3.plot(t, df['alpha2_cmd_deg'], 'g--', linewidth=1.5, label='α2 cmd (°)')
    if 'theta1_cmd_deg' in df.columns:
        ax3.plot(t, df['theta1_cmd_deg'], 'r--', linewidth=1.5, label='θ1 cmd (°)')
    if 'theta2_cmd_deg' in df.columns:
        ax3.plot(t, df['theta2_cmd_deg'], 'm--', linewidth=1.5, label='θ2 cmd (°)')

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Tilt Angle (°)')
    ax3.set_title('Tilt Angles (Actual vs Desired)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')
    ax3.set_xlim(0, t_max)

    # -------------------------- Subplot 4: 姿态角 (Actual vs Desired) --------------------------
    ax4 = plt.subplot(4, 1, 4)
    ax4.plot(t, df['roll_deg'], 'b-', linewidth=2, label='Roll (°)')
    ax4.plot(t, df['target_roll_deg'], 'b--', linewidth=1.5, label='Target Roll (°)')
    ax4.plot(t, df['pitch_deg'], 'g-', linewidth=2, label='Pitch (°)')
    ax4.plot(t, df['target_pitch_deg'], 'g--', linewidth=1.5, label='Target Pitch (°)')
    ax4.plot(t, df['yaw_deg'], 'r-', linewidth=2, label='Yaw (°)')
    ax4.plot(t, df['target_yaw_deg'], 'r--', linewidth=1.5, label='Target Yaw (°)')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Attitude (°)')
    ax4.set_title('Attitude Angles (Roll, Pitch, Yaw)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best')
    ax4.set_xlim(0, t_max)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to: {save_path}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'logs/force_torque_tilt_attitude_combined_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to: {save_path}")

    plt.close()


def plot_drone_data_and_save_all(df: pd.DataFrame):
    """
    Plot all drone data and save three PNG files according to requirements
    """
    # Generate three main plots
    plot_position_attitude_rotation(df)
    plot_force_torque(df)
    plot_tilt_thrust_u_values(df)
    # 新增：力/力矩/倾转角/姿态角 综合图
    plot_force_torque_tilt_attitude_combined(df)

if __name__ == "__main__":
    main()