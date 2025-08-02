import gym
import numpy as np
from gym import spaces
from .drilling_physics import DrillingPhysics, FormationType, AbnormalCondition

class DrillingEnv(gym.Env):
    """محیط شبیه‌سازی حفاری با استفاده از رابط OpenAI Gym"""
    
    def __init__(self, max_episode_steps=1000):
        super(DrillingEnv, self).__init__()
        
        # ایجاد نمونه از کلاس فیزیک حفاری
        self.physics = DrillingPhysics()
        
        # تعریف فضای عمل (Action Space)
        # 1. وزن روی مته (WOB): 0-50,000 نیوتن
        # 2. سرعت چرخش (RPM): 0-200 دور بر دقیقه
        # 3. نرخ جریان گل: 0-0.1 متر مکعب بر ثانیه
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([50000, 200, 0.1]),
            dtype=np.float32
        )
        
        # تعریف فضای حالت (State Space)
        # عمق، فرسایش مته، نرخ نفوذ، گشتاور، فشار، ارتعاشات (محوری، جانبی، پیچشی)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([10000, 1, 100, 10000, 50000000, 1, 1, 1]),
            dtype=np.float32
        )
        
        # تنظیمات محیط
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.current_state = None
        self.current_formation = FormationType.SOFT_SAND
        self.current_angle = 0  # زاویه حفاری (درجه)
        
    def reset(self, random_init=False):
        """بازنشانی محیط و بازگرداندن حالت اولیه"""
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
            self.current_angle = np.random.uniform(0, 30)  # زاویه تصادفی بین 0 تا 30 درجه
        
        # تبدیل حالت به آرایه برای Gym
        return self._get_observation()
    
    def step(self, action):
        """اجرای یک گام در محیط با استفاده از عمل داده شده"""
        self.current_step += 1
        
        # اعتبارسنجی و محدودسازی اعمال
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # استخراج اعمال
        wob, rpm, flow_rate = action
        
        # شبیه‌سازی یک گام با استفاده از فیزیک حفاری
        new_state, info = self.physics.simulate_step(
            self.current_state,
            {'wob': wob, 'rpm': rpm, 'flow_rate': flow_rate},
            timestep=60  # شبیه‌سازی یک دقیقه
        )
        
        # به‌روزرسانی حالت فعلی
        self.current_state = new_state
        
        # تغییر تصادفی نوع سازند با احتمال کم
        if np.random.random() < 0.02:  # 2% احتمال تغییر سازند
            formations = list(FormationType)
            self.current_formation = np.random.choice(formations)
            info['formation_changed'] = True
            info['new_formation'] = self.current_formation.value
        
        # محاسبه پاداش
        reward = self._calculate_reward(new_state, action)
        
        # بررسی پایان اپیزود
        done = self._check_termination()
        
        return self._get_observation(), reward, done, info
    
    def render(self, mode='human'):
        """نمایش وضعیت فعلی محیط"""
        if mode == 'human':
            print("=== Drilling Environment State ===")
            print(f"Depth: {self.current_state['depth']:.2f} m")
            print(f"Bit Wear: {self.current_state['bit_wear']*100:.1f}%")
            print(f"ROP: {self.current_state['rop']:.2f} m/hr")
            print(f"Torque: {self.current_state['torque']:.2f} Nm")
            print(f"Pressure: {self.current_state['pressure']/1e6:.2f} MPa")
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
    
    def _calculate_efficiency(self):
        """محاسبه کارایی عملیات حفاری"""
        # نسبت نرخ نفوذ به مصرف انرژی و فرسایش مته
        if self.current_state['rop'] > 0:
            efficiency = (self.current_state['rop'] * 100) / \
                        ((1 + self.current_state['bit_wear'] * 10) * \
                         (1 + sum([self.current_state[f'vibration_{v}'] for v in ['axial', 'lateral', 'torsional']])))
            return min(efficiency, 100)  # حداکثر 100%
        return 0.0
    
    def _get_observation(self):
        """تبدیل حالت فعلی به آرایه مشاهده برای Gym"""
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
    
    def _calculate_reward(self, state, action):
        """محاسبه پاداش بر اساس حالت و عمل"""
        # پاداش مثبت برای نرخ نفوذ (مهم‌ترین معیار)
        rop_reward = state['rop'] * 0.5
        
        # جریمه برای فرسایش مته (با وزن بیشتر)
        bit_wear_penalty = state['bit_wear'] * 20.0
        
        # جریمه برای ارتعاشات بیش از حد (با وزن‌های متفاوت برای انواع ارتعاش)
        vibration_penalty = (
            state['vibration_axial'] * 3.0 + 
            state['vibration_lateral'] * 4.0 + 
            state['vibration_torsional'] * 5.0
        )
        
        # جریمه برای مصرف انرژی
        wob, rpm, flow_rate = action
        power_consumption = (wob * rpm / 1000) + (flow_rate * state['pressure'] / 1000000)
        power_penalty = power_consumption * 0.02
        
        # پاداش برای پیشرفت عمقی نسبت به گام قبل
        depth_progress = state['depth'] - self.previous_depth if hasattr(self, 'previous_depth') else state['depth']
        self.previous_depth = state['depth']
        depth_reward = depth_progress * 5.0
        
        # محاسبه پاداش کل
        reward = rop_reward + depth_reward - bit_wear_penalty - vibration_penalty - power_penalty
        
        return reward
    
    def _check_termination(self):
        """بررسی شرایط پایان اپیزود"""
        # پایان بر اساس تعداد گام‌ها
        if self.current_step >= self.max_episode_steps:
            return True
        
        # پایان بر اساس فرسایش بیش از حد مته
        if self.current_state['bit_wear'] >= 0.9:  # 90% فرسایش
            return True
        
        # پایان بر اساس ارتعاشات بیش از حد
        total_vibration = (
            self.current_state['vibration_axial'] + 
            self.current_state['vibration_lateral'] + 
            self.current_state['vibration_torsional']
        )
        if total_vibration >= 2.5:  # آستانه ارتعاش
            return True
        
        return False