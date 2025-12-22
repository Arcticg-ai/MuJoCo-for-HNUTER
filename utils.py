import numpy as np
import math
from typing import Tuple, List


def quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """四元数转旋转矩阵"""
    w, x, y, z = quat
    
    R11 = 1 - 2 * (y * y + z * z)
    R12 = 2 * (x * y - w * z)
    R13 = 2 * (x * z + w * y)
    
    R21 = 2 * (x * y + w * z)
    R22 = 1 - 2 * (x * x + z * z)
    R23 = 2 * (y * z - w * x)
    
    R31 = 2 * (x * z - w * y)
    R32 = 2 * (y * z + w * x)
    R33 = 1 - 2 * (x * x + y * y)
    
    return np.array([
        [R11, R12, R13],
        [R21, R22, R23],
        [R31, R32, R33]
    ])


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """四元数转欧拉角 (roll, pitch, yaw)"""
    w, x, y, z = quat
    
    # Roll (x轴旋转)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y轴旋转)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z轴旋转)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """欧拉角转四元数 [w, x, y, z]"""
    roll, pitch, yaw = euler
    
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])


def euler_to_rotation_matrix(euler: np.ndarray) -> np.ndarray:
    """将欧拉角转换为旋转矩阵（RPY顺序）"""
    roll, pitch, yaw = euler
    
    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])
    
    R_y = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    
    R_z = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    return R_z @ R_y @ R_x


def vee_map(S: np.ndarray) -> np.ndarray:
    """反对称矩阵的vee映射"""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def hat_map(v: np.ndarray) -> np.ndarray:
    """向量的hat映射（叉乘矩阵）"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def rotation_x(angle: float) -> np.ndarray:
    """绕X轴旋转矩阵"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rotation_z(angle: float) -> np.ndarray:
    """绕Z轴旋转矩阵"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def handle_angle_continuity(current: float, last: float) -> float:
    """处理角度连续性，避免跳变"""
    diff = current - last
    if diff > np.pi:
        return current - 2 * np.pi
    elif diff < -np.pi:
        return current + 2 * np.pi
    return current


def get_axis_type(axis_idx: int, pitch: float) -> str:
    """确定当前轴的响应类型"""
    pitch_deg = abs(np.degrees(pitch))
    
    if axis_idx == 0:  # 横滚轴
        if pitch_deg < 45:
            return 'fast'  # 水平时横滚是快轴
        else:
            return 'slow'  # 直立时横滚变慢轴
    elif axis_idx == 2:  # 偏航轴
        if pitch_deg < 45:
            return 'slow'  # 水平时偏航是慢轴
        else:
            return 'fast'  # 直立时偏航变快轴
    else:  # 俯仰轴
        return 'medium'  # 俯仰轴始终是中速


def get_all_axis_types(pitch: float) -> List[str]:
    """获取所有轴的响应类型"""
    return [
        get_axis_type(0, pitch),
        get_axis_type(1, pitch),
        get_axis_type(2, pitch)
    ]
