import numpy as np
import mujoco as mj
from typing import Dict, Any, Optional
from utils import quat_to_rotation_matrix, quat_to_euler


class SimulationFramework:
    def __init__(self, model_path: str = "scene.xml"):
        # 加载MuJoCo模型
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        
        # 打印模型诊断信息
        self._print_model_diagnostics()
        
        # 物理参数
        self.dt = self.model.opt.timestep
        
        # 执行器和传感器ID映射
        self.actuator_ids: Dict[str, int] = {}
        self.sensor_ids: Dict[str, int] = {}
        
        # 获取执行器和传感器ID
        self._get_actuator_ids()
        self._get_sensor_ids()
        
        # ========== 新增：俯仰角阈值参数 ==========  
        self.pitch_threshold_deg = 70.0  # 俯仰角阈值（度）
        self.pitch_threshold_rad = np.radians(self.pitch_threshold_deg)  # 转换为弧度
        self.is_pitch_exceed = False  # 标记是否超过阈值
        self._pitch_warned = False  # 避免重复打印警告
        
        print("仿真框架初始化完成")
    
    def _print_model_diagnostics(self):
        """打印模型诊断信息"""
        print("\n=== 模型诊断信息 ===")
        print(f"广义坐标数量 (nq): {self.model.nq}")
        print(f"速度自由度 (nv): {self.model.nv}")
        print(f"执行器数量 (nu): {self.model.nu}")
        print(f"身体数量: {self.model.nbody}")
        print(f"关节数量: {self.model.njnt}")
        print(f"几何体数量: {self.model.ngeom}")
        
        # 检查身体信息
        print("\n=== 身体列表 ===")
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            print(f"身体 {i}: {name}")
        
        # 检查关节信息
        print("\n=== 关节列表 ===")
        for i in range(self.model.njnt):
            jnt_type = self.model.jnt_type[i]
            jnt_name = self.model.jnt(i).name
            print(f"关节 {i}: {jnt_name}, 类型: {jnt_type}")
        
        # 检查执行器信息
        print("\n=== 执行器列表 ===")
        for i in range(self.model.nu):
            act_name = self.model.name_actuatoradr[i]
            print(f"执行器 {i}: {act_name}")
       
    def _get_actuator_ids(self):
        """获取执行器ID"""
        try:
            # 机臂偏航执行器
            self.actuator_ids['tilt_pitch_left'] = mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_pitch_left')
            self.actuator_ids['tilt_pitch_right'] = mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_pitch_right')
            
            # 螺旋桨倾转执行器
            self.actuator_ids['tilt_roll_left'] = mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_roll_left')
            self.actuator_ids['tilt_roll_right'] = mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_roll_right')
            
            # 推力执行器
            thrust_actuators = [
                'motor_r_upper', 'motor_r_lower', 
                'motor_l_upper', 'motor_l_lower', 
                'motor_rear_upper'
            ]
            for name in thrust_actuators:
                self.actuator_ids[name] = mj.mj_name2id(
                    self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
            
            print("执行器ID映射:", self.actuator_ids)
            
        except Exception as e:
            print(f"获取执行器ID失败: {e}")
            # 使用备用方案：直接按顺序获取
            self.actuator_ids = {}
            for i in range(self.model.nu):
                act_name = self.model.name_actuatoradr[i]
                if act_name:
                    self.actuator_ids[act_name] = i
            print("顺序执行器ID映射:", self.actuator_ids)
    
    def _get_sensor_ids(self):
        """获取传感器ID"""
        try:
            # 位置和姿态传感器
            self.sensor_ids['drone_pos'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, 'drone_pos')
            self.sensor_ids['drone_quat'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, 'drone_quat')
            
            # 倾转角度传感器
            tilt_sensors = [
                'arm_pitch_left_pos', 'arm_pitch_right_pos',
                'prop_tilt_left_pos', 'prop_tilt_right_pos'
            ]
            for name in tilt_sensors:
                self.sensor_ids[name] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, name)
            
            print("传感器ID映射:", self.sensor_ids)
            
        except Exception as e:
            print(f"获取传感器ID失败: {e}")
            # 创建默认映射
            self.sensor_ids = {}
            for i in range(self.model.nsensor):
                sensor_name = self.model.name_sensoradr[i]
                if sensor_name:
                    self.sensor_ids[sensor_name] = i
            print("顺序传感器ID映射:", self.sensor_ids)
    
    def get_state(self) -> Dict[str, Any]:
        """获取无人机当前状态（新增俯仰角超限判断和实际倾转角度）"""
        state = {
            'position': np.zeros(3),
            'quaternion': np.array([1.0, 0.0, 0.0, 0.0]),
            'rotation_matrix': np.eye(3),
            'velocity': np.zeros(3),
            'angular_velocity': np.zeros(3),
            'acceleration': np.zeros(3),
            'euler': np.zeros(3),
            'is_pitch_exceed': False
        }
        
        try:
            body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'drone')
            if body_id != -1:
                state['position'] = self.data.xpos[body_id].copy()
                state['quaternion'] = self.data.xquat[body_id].copy()
                state['velocity'] = self.data.cvel[body_id][3:6].copy()
                state['angular_velocity'] = self.data.cvel[body_id][0:3].copy()
            
            state['rotation_matrix'] = quat_to_rotation_matrix(state['quaternion'])
            state['euler'] = quat_to_euler(state['quaternion'])
            self.current_pitch = state['euler'][1]  # 更新当前俯仰角
            
            # ========== 核心修改：判断俯仰角是否超限 ==========  
            self.is_pitch_exceed = abs(state['euler'][1]) > self.pitch_threshold_rad
            state['is_pitch_exceed'] = self.is_pitch_exceed
            
            # 打印超限警告（仅首次超限/恢复时）
            if self.is_pitch_exceed and not self._pitch_warned:
                pitch_deg = np.degrees(state['euler'][1])
                print(f"\n⚠️ 警告：俯仰角 {pitch_deg:.1f}° 超过 {self.pitch_threshold_deg}°，启用几何解耦控制！")
                self._pitch_warned = True
            elif not self.is_pitch_exceed and self._pitch_warned:
                pitch_deg = np.degrees(state['euler'][1])
                print(f"\n✅ 恢复：俯仰角 {pitch_deg:.1f}° 低于 {self.pitch_threshold_deg}°，恢复正常控制！")
                self._pitch_warned = False
            
            # ========== 获取实际倾转角度 ==========  
            self.alpha1_actual = 0.0
            self.alpha2_actual = 0.0
            self.theta1_actual = 0.0
            self.theta2_actual = 0.0
            
            try:
                if 'arm_pitch_left_pos' in self.sensor_ids:
                    self.alpha1_actual = self.data.sensordata[self.sensor_ids['arm_pitch_left_pos']]
                if 'arm_pitch_right_pos' in self.sensor_ids:
                    self.alpha2_actual = self.data.sensordata[self.sensor_ids['arm_pitch_right_pos']]
                if 'prop_tilt_left_pos' in self.sensor_ids:
                    self.theta1_actual = self.data.sensordata[self.sensor_ids['prop_tilt_left_pos']]
                if 'prop_tilt_right_pos' in self.sensor_ids:
                    self.theta2_actual = self.data.sensordata[self.sensor_ids['prop_tilt_right_pos']]
            except:
                pass
            
            if np.any(np.isnan(state['position'])):
                print("警告: 位置数据包含NaN，使用零值")
                state['position'] = np.zeros(3)
                
            return state
        except Exception as e:
            print(f"状态获取错误: {e}")
            return state
    
    def step(self):
        """执行一次仿真步进"""
        mj.mj_step(self.model, self.data)
    
    def reset(self):
        """重置仿真"""
        mj.mj_resetData(self.model, self.data)
    
    def get_actual_tilt_angles(self) -> Dict[str, float]:
        """获取实际倾转角度"""
        return {
            'alpha1_actual': self.alpha1_actual,
            'alpha2_actual': self.alpha2_actual,
            'theta1_actual': self.theta1_actual,
            'theta2_actual': self.theta2_actual
        }
    
    def set_actuators(self, ctrl_values: Dict[str, float]):
        """设置执行器控制值"""
        for actuator_name, value in ctrl_values.items():
            if actuator_name in self.actuator_ids:
                self.data.ctrl[self.actuator_ids[actuator_name]] = value
    
    def get_body_position(self, body_name: str) -> np.ndarray:
        """获取指定身体的位置"""
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id != -1:
            return self.data.xpos[body_id].copy()
        return np.zeros(3)
    
    def get_body_quaternion(self, body_name: str) -> np.ndarray:
        """获取指定身体的四元数"""
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id != -1:
            return self.data.xquat[body_id].copy()
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    def get_actuator_torques(self) -> Dict[str, float]:
        """获取执行器力矩（从传感器反馈）"""
        torques = {}
        try:
            # 从传感器获取执行器力矩
            torque_sensors = [
                'motor_r_upper_torque', 'motor_r_lower_torque',
                'motor_l_upper_torque', 'motor_l_lower_torque',
                'motor_rear_upper_torque'
            ]
            
            for sensor_name in torque_sensors:
                if sensor_name in self.sensor_ids:
                    torques[sensor_name] = self.data.sensordata[self.sensor_ids[sensor_name]]
        except Exception as e:
            print(f"获取执行器力矩失败: {e}")
        
        return torques
