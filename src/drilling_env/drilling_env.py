"""
Drilling Environment Module
OpenAI Gym environment for drilling operations simulation.

This module provides a reinforcement learning environment for optimizing
drilling operations through control of WOB (Weight on Bit), RPM, and flow rate.
"""
import gym
import numpy as np
from gym import spaces
from typing import Tuple, Dict, Optional
from drilling_env.drilling_physics import DrillingPhysics, FormationType, AbnormalCondition

# Constants for DrillingEnv
MAX_WOB = 50000  # Maximum Weight on Bit (Newtons)
MAX_RPM = 200    # Maximum RPM
MAX_FLOW_RATE = 0.1  # Maximum flow rate (m³/s)
MAX_DEPTH = 10000  # Maximum depth (meters)
MAX_TORQUE = 10000  # Maximum torque (Nm)
MAX_PRESSURE = 50000000  # Maximum pressure (Pa)
DEFAULT_TIMESTEP = 60  # Default simulation timestep (seconds)
FORMATION_CHANGE_PROBABILITY = 0.02  # 2% probability of formation change
MAX_ANGLE = 30  # Maximum drilling angle (degrees)
EFFICIENCY_SCALE = 100  # Scale factor for efficiency calculation
VIBRATION_WEAR_FACTOR = 10  # Factor for vibration impact on efficiency
MAX_EFFICIENCY = 100  # Maximum efficiency percentage

# Reward calculation constants
ROP_REWARD_WEIGHT = 0.5
BIT_WEAR_PENALTY_WEIGHT = 20.0
VIBRATION_AXIAL_WEIGHT = 3.0
VIBRATION_LATERAL_WEIGHT = 4.0
VIBRATION_TORSIONAL_WEIGHT = 5.0
POWER_PENALTY_WEIGHT = 0.02
DEPTH_REWARD_WEIGHT = 5.0
POWER_DIVISOR = 1000
PRESSURE_POWER_DIVISOR = 1000000

# Termination conditions
MAX_BIT_WEAR = 0.9  # 90% bit wear threshold
MAX_TOTAL_VIBRATION = 2.5  # Maximum total vibration threshold
PRESSURE_DISPLAY_DIVISOR = 1e6  # For MPa conversion in display

class DrillingEnv(gym.Env):
    """
    Drilling simulation environment using OpenAI Gym interface.
    
    This environment simulates a drilling operation where an agent controls
    drilling parameters (WOB, RPM, flow rate) to optimize drilling efficiency
    while managing bit wear, vibrations, and other operational constraints.
    
    Attributes:
        physics: DrillingPhysics instance for physics simulation
        action_space: Gym Box space for actions [WOB, RPM, flow_rate]
        observation_space: Gym Box space for state observations
        max_episode_steps: Maximum number of steps per episode
        current_step: Current step in the episode
        current_state: Current drilling state dictionary
        current_formation: Current formation type being drilled
        current_angle: Current drilling angle in degrees
    """
    
    def __init__(self, max_episode_steps: int = 1000):
        """
        Initialize the drilling environment.
        
        Args:
            max_episode_steps: Maximum number of steps per episode (default: 1000)
        """
        super(DrillingEnv, self).__init__()
        
        # ایجاد نمونه از کلاس فیزیک حفاری
        self.physics = DrillingPhysics()
        
        # تعریف فضای عمل (Action Space)
        # 1. وزن روی مته (WOB): 0-MAX_WOB نیوتن
        # 2. سرعت چرخش (RPM): 0-MAX_RPM دور بر دقیقه
        # 3. نرخ جریان گل: 0-MAX_FLOW_RATE متر مکعب بر ثانیه
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([MAX_WOB, MAX_RPM, MAX_FLOW_RATE]),
            dtype=np.float32
        )
        
        # تعریف فضای حالت (State Space)
        # عمق، فرسایش مته، نرخ نفوذ، گشتاور، فشار، ارتعاشات (محوری، جانبی، پیچشی)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([MAX_DEPTH, 1, 100, MAX_TORQUE, MAX_PRESSURE, 1, 1, 1]),
            dtype=np.float32
        )
        
        # تنظیمات محیط
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.current_state = None
        self.current_formation = FormationType.SOFT_SAND
        self.current_angle = 0  # زاویه حفاری (درجه)
        
    def reset(self, random_init: bool = False) -> np.ndarray:
        """
        Reset the environment and return initial state.
        
        Resets all drilling parameters to initial values and optionally
        randomizes the initial formation type and drilling angle.
        
        Args:
            random_init: If True, randomizes initial formation and angle
            
        Returns:
            Initial observation array
        """
        self.current_step = 0
        
        # تنظیم حالت اولیه
        self.current_state = {
            'depth': 0.0,
            'bit_wear': 0.0,
            'rop': 0.0,
            'torque': 0.0,
            'pressure': 0.0,
            'vibration_axial': 0.0,
            'vibration_lateral': 0.0,
            'vibration_torsional': 0.0
        }
        
        # تصادفی‌سازی نوع سازند اولیه (اختیاری)
        if random_init:
            formations = list(FormationType)
            self.current_formation = np.random.choice(formations)
            self.current_angle = np.random.uniform(0, MAX_ANGLE)  # زاویه تصادفی بین 0 تا MAX_ANGLE درجه
        
        # تبدیل حالت به آرایه برای Gym
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment using the given action.
        
        Applies the action (WOB, RPM, flow_rate), simulates physics,
        calculates reward, and checks for episode termination.
        
        Args:
            action: Array of [WOB, RPM, flow_rate] values
            
        Returns:
            Tuple of (observation, reward, done, info):
            - observation: New state observation array
            - reward: Calculated reward for this step
            - done: Boolean indicating if episode is finished
            - info: Dictionary with additional information
        """
        self.current_step += 1
        
        # اعتبارسنجی و محدودسازی اعمال
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # استخراج اعمال
        wob, rpm, flow_rate = action
        
        # شبیه‌سازی یک گام با استفاده از فیزیک حفاری
        new_state, info = self.physics.simulate_step(
            self.current_state,
            {'wob': wob, 'rpm': rpm, 'flow_rate': flow_rate},
            timestep=DEFAULT_TIMESTEP  # شبیه‌سازی یک دقیقه
        )
        
        # به‌روزرسانی حالت فعلی
        self.current_state = new_state
        
        # تغییر تصادفی نوع سازند با احتمال کم
        if np.random.random() < FORMATION_CHANGE_PROBABILITY:  # 2% احتمال تغییر سازند
            formations = list(FormationType)
            self.current_formation = np.random.choice(formations)
            info['formation_changed'] = True
            info['new_formation'] = self.current_formation.value
        
        # محاسبه پاداش
        reward = self._calculate_reward(new_state, action)
        
        # بررسی پایان اپیزود
        done = self._check_termination()
        
        return self._get_observation(), reward, done, info
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the current environment state.
        
        Args:
            mode: Rendering mode ('human' for console output, 'rgb_array' for image)
            
        Returns:
            RGB array if mode is 'rgb_array', None otherwise
        """
        if mode == 'human':
            print("=== Drilling Environment State ===")
            print(f"Depth: {self.current_state['depth']:.2f} m")
            print(f"Bit Wear: {self.current_state['bit_wear']*100:.1f}%")
            print(f"ROP: {self.current_state['rop']:.2f} m/hr")
            print(f"Torque: {self.current_state['torque']:.2f} Nm")
            print(f"Pressure: {self.current_state['pressure']/PRESSURE_DISPLAY_DIVISOR:.2f} MPa")
            print(f"Vibrations (Axial/Lateral/Torsional): {self.current_state['vibration_axial']:.2f}/{self.current_state['vibration_lateral']:.2f}/{self.current_state['vibration_torsional']:.2f}")
            print(f"Formation: {self.current_formation.value}")
            print(f"Angle: {self.current_angle} degrees")
            print(f"Step: {self.current_step}/{self.max_episode_steps}")
            print(f"Estimated Efficiency: {self._calculate_efficiency():.2f}%")
            print("===================================")
        elif mode == 'rgb_array':
            # برای پیاده‌سازی آینده: بازگرداندن تصویر گرافیکی از وضعیت حفاری
            # این قابلیت می‌تواند برای ضبط ویدیو از فرآیند حفاری استفاده شود
            return np.zeros((400, 600, 3))
    
    def _calculate_efficiency(self) -> float:
        """
        Calculate drilling operation efficiency.
        
        Efficiency is based on rate of penetration relative to energy consumption
        and bit wear. Higher ROP with lower wear and vibrations results in higher efficiency.
        
        Returns:
            Efficiency percentage (0-100)
        """
        # نسبت نرخ نفوذ به مصرف انرژی و فرسایش مته
        if self.current_state['rop'] > 0:
            efficiency = (self.current_state['rop'] * EFFICIENCY_SCALE) / \
                        ((1 + self.current_state['bit_wear'] * VIBRATION_WEAR_FACTOR) * \
                         (1 + sum([self.current_state[f'vibration_{v}'] for v in ['axial', 'lateral', 'torsional']])))
            return min(efficiency, MAX_EFFICIENCY)  # حداکثر MAX_EFFICIENCY%
        return 0.0
    
    def _get_observation(self) -> np.ndarray:
        """
        Convert current state to observation array for Gym.
        
        Returns:
            NumPy array containing [depth, bit_wear, rop, torque, pressure,
            vibration_axial, vibration_lateral, vibration_torsional]
        """
        return np.array([
            self.current_state['depth'],
            self.current_state['bit_wear'],
            self.current_state['rop'],
            self.current_state['torque'],
            self.current_state['pressure'],
            self.current_state['vibration_axial'],
            self.current_state['vibration_lateral'],
            self.current_state['vibration_torsional']
        ], dtype=np.float32)
    
    def _calculate_reward(self, state: Dict, action: np.ndarray) -> float:
        """
        Calculate reward based on current state and action.
        
        Reward components:
        - Positive reward for rate of penetration (ROP)
        - Positive reward for depth progress
        - Negative penalty for bit wear
        - Negative penalty for excessive vibrations
        - Negative penalty for power consumption
        
        Args:
            state: Current drilling state dictionary
            action: Action array [WOB, RPM, flow_rate]
            
        Returns:
            Calculated reward value
        """
        # پاداش مثبت برای نرخ نفوذ (مهم‌ترین معیار)
        rop_reward = state['rop'] * ROP_REWARD_WEIGHT
        
        # جریمه برای فرسایش مته (با وزن بیشتر)
        bit_wear_penalty = state['bit_wear'] * BIT_WEAR_PENALTY_WEIGHT
        
        # جریمه برای ارتعاشات بیش از حد (با وزن‌های متفاوت برای انواع ارتعاش)
        vibration_penalty = (
            state['vibration_axial'] * VIBRATION_AXIAL_WEIGHT + 
            state['vibration_lateral'] * VIBRATION_LATERAL_WEIGHT + 
            state['vibration_torsional'] * VIBRATION_TORSIONAL_WEIGHT
        )
        
        # جریمه برای مصرف انرژی
        wob, rpm, flow_rate = action
        power_consumption = (wob * rpm / POWER_DIVISOR) + (flow_rate * state['pressure'] / PRESSURE_POWER_DIVISOR)
        power_penalty = power_consumption * POWER_PENALTY_WEIGHT
        
        # پاداش برای پیشرفت عمقی نسبت به گام قبل
        depth_progress = state['depth'] - self.previous_depth if hasattr(self, 'previous_depth') else state['depth']
        self.previous_depth = state['depth']
        depth_reward = depth_progress * DEPTH_REWARD_WEIGHT
        
        # محاسبه پاداش کل
        reward = rop_reward + depth_reward - bit_wear_penalty - vibration_penalty - power_penalty
        
        return reward
    
    def _check_termination(self) -> bool:
        """
        Check if episode should terminate.
        
        Termination conditions:
        - Maximum episode steps reached
        - Bit wear exceeds threshold (90%)
        - Total vibration exceeds threshold
        
        Returns:
            True if episode should terminate, False otherwise
        """
        # پایان بر اساس تعداد گام‌ها
        if self.current_step >= self.max_episode_steps:
            return True
        
        # پایان بر اساس فرسایش بیش از حد مته
        if self.current_state['bit_wear'] >= MAX_BIT_WEAR:  # MAX_BIT_WEAR فرسایش
            return True
        
        # پایان بر اساس ارتعاشات بیش از حد
        total_vibration = (
            self.current_state['vibration_axial'] + 
            self.current_state['vibration_lateral'] + 
            self.current_state['vibration_torsional']
        )
        if total_vibration >= MAX_TOTAL_VIBRATION:  # آستانه ارتعاش
            return True
        
        return False