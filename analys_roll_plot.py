#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无人机Roll轴姿态控制分析脚本
功能：分析roll轴的姿态角、角速度和力矩之间的关系
作者：基于搜索结果的无人机控制分析工具
日期：2025年11月12日
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy import signal
import argparse

class RollAxisAnalyzer:
    """Roll轴姿态控制分析器"""
    
    def __init__(self, log_file_path):
        """
        初始化分析器
        
        参数:
        log_file_path (str): 日志文件路径
        """
        self.log_file_path = log_file_path
        self.df = None
        self.analysis_results = {}
        
        # 设置图形样式
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def load_data(self):
        """加载日志数据"""
        try:
            if not os.path.exists(self.log_file_path):
                print(f"错误: 文件 {self.log_file_path} 不存在")
                return False
            
            self.df = pd.read_csv(self.log_file_path)
            print(f"成功加载日志文件: {self.log_file_path}")
            print(f"数据形状: {self.df.shape}")
            print(f"数据时间范围: {self.df['timestamp'].min():.2f} - {self.df['timestamp'].max():.2f} 秒")
            
            # 检查必要的列是否存在
            required_columns = ['timestamp', 'roll', 'target_roll', 'angular_vel_x', 
                              'target_angular_vel_x', 'tau_x']
            
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                print(f"警告: 缺少以下列: {missing_columns}")
                print("可用的列:", list(self.df.columns))
                
                # 尝试使用替代列名
                column_mapping = {
                    'roll': ['roll', 'euler_x', 'attitude_roll'],
                    'target_roll': ['target_roll', 'target_attitude_roll', 'target_euler_x'],
                    'angular_vel_x': ['angular_vel_x', 'body_gyro_x', 'gyro_x'],
                    'target_angular_vel_x': ['target_angular_vel_x', 'target_gyro_x'],
                    'tau_x': ['tau_x', 'control_torque_x', 'torque_x']
                }
                
                for required, alternatives in column_mapping.items():
                    if required not in self.df.columns:
                        for alt in alternatives:
                            if alt in self.df.columns:
                                self.df[required] = self.df[alt]
                                print(f"使用替代列: {alt} -> {required}")
                                break
                
            return True
            
        except Exception as e:
            print(f"读取文件时出错: {e}")
            return False
    
    def calculate_performance_metrics(self):
        """计算控制性能指标"""
        if self.df is None:
            return
        
        # 转换为度以便于理解
        roll_deg = np.degrees(self.df['roll'])
        target_roll_deg = np.degrees(self.df['target_roll'])
        roll_error_deg = roll_deg - target_roll_deg
        
        angular_vel = self.df['angular_vel_x']
        target_angular_vel = self.df['target_angular_vel_x']
        angular_vel_error = angular_vel - target_angular_vel
        
        tau_x = self.df['tau_x']
        
        # 计算性能指标
        self.analysis_results = {
            # 姿态角性能指标
            'max_roll_error': np.max(np.abs(roll_error_deg)),
            'rms_roll_error': np.sqrt(np.mean(roll_error_deg**2)),
            'mean_roll_error': np.mean(np.abs(roll_error_deg)),
            'settling_time': self._calculate_settling_time(roll_error_deg, self.df['timestamp']),
            
            # 角速度性能指标
            'max_angular_vel_error': np.max(np.abs(angular_vel_error)),
            'rms_angular_vel_error': np.sqrt(np.mean(angular_vel_error**2)),
            
            # 控制力矩指标
            'max_control_torque': np.max(np.abs(tau_x)),
            'rms_control_torque': np.sqrt(np.mean(tau_x**2)),
            'mean_control_torque': np.mean(np.abs(tau_x)),
            
            # 稳定性指标
            'overshoot': self._calculate_overshoot(roll_deg, target_roll_deg),
            'rise_time': self._calculate_rise_time(roll_deg, target_roll_deg, self.df['timestamp']),
            
            # 频域分析
            'bandwidth': self._calculate_bandwidth(roll_error_deg, 1.0)  # 假设采样率1Hz
        }
    
    def _calculate_settling_time(self, error, time, threshold=0.05):
        """计算稳定时间（误差小于5%的时间）"""
        abs_error = np.abs(error)
        threshold_value = threshold * np.max(abs_error)  # 5%最大误差作为稳定阈值
        
        # 找到首次进入稳定区域的时间
        settled_indices = np.where(abs_error < threshold_value)[0]
        if len(settled_indices) > 0:
            return time[settled_indices[0]]
        return time.iloc[-1]  # 返回最后时间点
    
    def _calculate_overshoot(self, actual, target):
        """计算超调量"""
        if len(actual) == 0:
            return 0
        
        max_overshoot = np.max(actual - target)
        if max_overshoot > 0:
            return (max_overshoot / np.max(np.abs(target))) * 100 if np.max(np.abs(target)) > 0 else 0
        return 0
    
    def _calculate_rise_time(self, actual, target, time, low_percent=10, high_percent=90):
        """计算上升时间（10%-90%）"""
        if len(actual) == 0:
            return 0
        
        target_range = np.max(target) - np.min(target)
        if target_range == 0:
            return 0
        
        low_threshold = np.min(target) + low_percent/100 * target_range
        high_threshold = np.min(target) + high_percent/100 * target_range
        
        low_idx = np.where(actual >= low_threshold)[0]
        high_idx = np.where(actual >= high_threshold)[0]
        
        if len(low_idx) > 0 and len(high_idx) > 0:
            return time[high_idx[0]] - time[low_idx[0]]
        return 0
    
    def _calculate_bandwidth(self, signal_data, sampling_rate):
        """计算控制带宽"""
        if len(signal_data) < 2:
            return 0
        
        # 计算功率谱密度
        freqs, psd = signal.welch(signal_data, fs=sampling_rate, nperseg=min(256, len(signal_data)))
        
        if len(psd) == 0:
            return 0
        
        # 找到-3dB带宽
        max_psd = np.max(psd)
        if max_psd > 0:
            half_power = max_psd / np.sqrt(2)  # -3dB点
            bandwidth_idx = np.where(psd >= half_power)[0]
            if len(bandwidth_idx) > 0:
                return freqs[bandwidth_idx[-1]]
        return 0
    
    def plot_roll_control_analysis(self, save_plot=True):
        """绘制Roll轴控制分析图"""
        if self.df is None:
            print("错误: 没有可用的数据")
            return
        
        # 创建图形
        fig = plt.figure(figsize=(15, 12))
        
        # 1. 姿态角图
        ax1 = plt.subplot(3, 1, 1)
        time = self.df['timestamp']
        
        # 转换为度
        roll_deg = np.degrees(self.df['roll'])
        target_roll_deg = np.degrees(self.df['target_roll'])
        roll_error_deg = roll_deg - target_roll_deg
        
        plt.plot(time, roll_deg, label='实际Roll角', color='#1f77b4', linewidth=2, alpha=0.8)
        plt.plot(time, target_roll_deg, label='目标Roll角', color='#d62728', linewidth=2, linestyle='--', alpha=0.8)
        plt.fill_between(time, roll_deg, target_roll_deg, where=(roll_error_deg >= 0), 
                         color='red', alpha=0.2, label='正误差')
        plt.fill_between(time, roll_deg, target_roll_deg, where=(roll_error_deg < 0), 
                         color='blue', alpha=0.2, label='负误差')
        
        plt.ylabel('姿态角 (度)', fontsize=12, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.title('Roll轴姿态控制分析', fontsize=16, fontweight='bold', pad=20)
        
        # 2. 角速度图
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        angular_vel = self.df['angular_vel_x']
        target_angular_vel = self.df['target_angular_vel_x']
        angular_vel_error = angular_vel - target_angular_vel
        
        plt.plot(time, angular_vel, label='实际Roll角速度', color='#2ca02c', linewidth=2, alpha=0.8)
        plt.plot(time, target_angular_vel, label='目标Roll角速度', color='#ff7f0e', linewidth=2, 
                linestyle='--', alpha=0.8)
        plt.fill_between(time, angular_vel, target_angular_vel, where=(angular_vel_error >= 0), 
                        color='orange', alpha=0.2, label='正误差')
        plt.fill_between(time, angular_vel, target_angular_vel, where=(angular_vel_error < 0), 
                        color='green', alpha=0.2, label='负误差')
        
        plt.ylabel('角速度 (rad/s)', fontsize=12, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 3. 控制力矩图
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        tau_x = self.df['tau_x']
        
        plt.plot(time, tau_x, label='Roll控制力矩', color='#9467bd', linewidth=2, alpha=0.8)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 标记最大力矩点
        max_tau_idx = np.argmax(np.abs(tau_x))
        plt.plot(time.iloc[max_tau_idx], tau_x.iloc[max_tau_idx], 'ro', markersize=8, 
                label=f'最大力矩: {tau_x.iloc[max_tau_idx]:.3f} Nm')
        
        plt.xlabel('时间 (秒)', fontsize=12, fontweight='bold')
        plt.ylabel('控制力矩 (Nm)', fontsize=12, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        if save_plot:
            plot_dir = os.path.dirname(self.log_file_path)
            plot_filename = os.path.splitext(os.path.basename(self.log_file_path))[0] + '_roll_analysis.png'
            plot_file_path = os.path.join(plot_dir, plot_filename)
            
            plt.savefig(plot_file_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"分析图已保存为: {plot_file_path}")
        
        plt.show()
        
        return fig
    
    def plot_correlation_analysis(self):
        """绘制相关性分析图"""
        if self.df is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 姿态误差 vs 控制力矩
        roll_error = np.degrees(self.df['roll'] - self.df['target_roll'])
        tau_x = self.df['tau_x']
        
        axes[0, 0].scatter(roll_error, tau_x, alpha=0.6, color='blue', s=20)
        axes[0, 0].set_xlabel('姿态误差 (度)')
        axes[0, 0].set_ylabel('控制力矩 (Nm)')
        axes[0, 0].set_title('姿态误差 vs 控制力矩')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 角速度误差 vs 控制力矩
        angular_vel_error = self.df['angular_vel_x'] - self.df['target_angular_vel_x']
        
        axes[0, 1].scatter(angular_vel_error, tau_x, alpha=0.6, color='green', s=20)
        axes[0, 1].set_xlabel('角速度误差 (rad/s)')
        axes[0, 1].set_ylabel('控制力矩 (Nm)')
        axes[0, 1].set_title('角速度误差 vs 控制力矩')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 控制力矩直方图
        axes[1, 0].hist(tau_x, bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_xlabel('控制力矩 (Nm)')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].set_title('控制力矩分布')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 姿态误差直方图
        axes[1, 1].hist(roll_error, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[1, 1].set_xlabel('姿态误差 (度)')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].set_title('姿态误差分布')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def print_performance_report(self):
        """打印性能报告"""
        if not self.analysis_results:
            self.calculate_performance_metrics()
        
        print("\n" + "="*60)
        print("            ROLL轴控制性能分析报告")
        print("="*60)
        
        print("\n 姿态控制性能:")
        print(f"  • 最大姿态误差: {self.analysis_results['max_roll_error']:.4f} 度")
        print(f"  • RMS姿态误差: {self.analysis_results['rms_roll_error']:.4f} 度")
        print(f"  • 平均姿态误差: {self.analysis_results['mean_roll_error']:.4f} 度")
        print(f"  • 超调量: {self.analysis_results['overshoot']:.2f} %")
        print(f"  • 稳定时间: {self.analysis_results['settling_time']:.2f} 秒")
        print(f"  • 上升时间: {self.analysis_results['rise_time']:.2f} 秒")
        
        print("\n 角速度控制性能:")
        print(f"  • 最大角速度误差: {self.analysis_results['max_angular_vel_error']:.4f} rad/s")
        print(f"  • RMS角速度误差: {self.analysis_results['rms_angular_vel_error']:.4f} rad/s")
        
        print("\n 控制力矩统计:")
        print(f"  • 最大控制力矩: {self.analysis_results['max_control_torque']:.4f} Nm")
        print(f"  • RMS控制力矩: {self.analysis_results['rms_control_torque']:.4f} Nm")
        print(f"  • 平均控制力矩: {self.analysis_results['mean_control_torque']:.4f} Nm")
        
        print("\n 系统特性:")
        print(f"  • 估计控制带宽: {self.analysis_results['bandwidth']:.2f} Hz")
        
        print("\n" + "="*60)
    
    def save_analysis_report(self):
        """保存分析报告到文件"""
        if not self.analysis_results:
            self.calculate_performance_metrics()
        
        report_dir = os.path.dirname(self.log_file_path)
        report_filename = os.path.splitext(os.path.basename(self.log_file_path))[0] + '_analysis_report.txt'
        report_file_path = os.path.join(report_dir, report_filename)
        
        with open(report_file_path, 'w', encoding='utf-8') as f:
            f.write("无人机Roll轴控制性能分析报告\n")
            f.write("="*50 + "\n\n")
            f.write(f"分析文件: {os.path.basename(self.log_file_path)}\n")
            f.write(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据点数: {len(self.df)}\n")
            f.write(f"时间范围: {self.df['timestamp'].min():.2f} - {self.df['timestamp'].max():.2f} 秒\n\n")
            
            f.write("性能指标:\n")
            f.write("-"*30 + "\n")
            for key, value in self.analysis_results.items():
                f.write(f"{key}: {value}\n")
        
        print(f"分析报告已保存为: {report_file_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='无人机Roll轴姿态控制分析工具')
    parser.add_argument('log_file', help='日志文件路径')
    parser.add_argument('--no-plot', action='store_true', help='不显示图形')
    parser.add_argument('--correlation', action='store_true', help='显示相关性分析')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = RollAxisAnalyzer(args.log_file)
    
    # 加载数据
    if not analyzer.load_data():
        sys.exit(1)
    
    # 计算性能指标
    analyzer.calculate_performance_metrics()
    
    # 打印性能报告
    analyzer.print_performance_report()
    
    # 绘制主要分析图
    if not args.no_plot:
        analyzer.plot_roll_control_analysis()
        
        if args.crelation:
            analyzer.plot_correlation_analysis()
    
    # 保存报告
    analyzer.save_analysis_report()
    
    print("\n分析完成！")

if __name__ == "__main__":
    # 如果没有提供命令行参数，使用示例或交互式模式
    if len(sys.argv) == 1:
        print("无人机Roll轴姿态控制分析工具")
        print("使用方法: python analyze_roll_control.py <日志文件路径> [选项]")
        print("\n选项:")
        print("  --no-plot     不显示图形")
        print("  --correlation 显示相关性分析")
        print("\n示例:")
        print("  python analyze_roll_control.py logs/drone_log_20250101_120000.csv")
        print("  python analyze_roll_control.py logs/drone_log_20250101_120000.csv --correlation")
        
        # 检查是否有默认的日志文件
        log_files = []
        if os.path.exists('logs'):
            log_files = [f for f in os.listdir('logs') if f.endswith('.csv')]
        
        if log_files:
            print(f"\n在logs目录中找到以下日志文件:")
            for i, f in enumerate(log_files[:5]):  # 显示前5个文件
                print(f"  {i+1}. {f}")
            print("\n请选择文件或直接提供文件路径作为参数")
        else:
            print("\n请在logs目录中放置日志文件，或直接提供文件路径")
    else:
        main()