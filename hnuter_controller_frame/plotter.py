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

def main():
    """
    Main function - Tilt-Rotor UAV Log Analysis
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tilt-Rotor UAV Log Data Analysis Tool')
    parser.add_argument('--file', '-f', type=str, help='Path to log file (optional, uses latest log by default)')
    parser.add_argument('--save', '-s', type=str, help='Path to save plot (optional)')
    
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
        
        # Generate and save visualization
        plot_drone_data_and_save(df, args.save)
        
        print("\nData analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()