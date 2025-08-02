"""
Core Physics and Dynamics of Drilling Process

این ماژول معادلات و روابط فیزیکی اصلی فرآیند حفاری را پیاده‌سازی می‌کند.
با در نظر گرفتن انواع سازند، شرایط غیرعادی و روابط متقابل پارامترها.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from enum import Enum

class FormationType(Enum):
    """انواع مختلف سازند"""
    SOFT_SAND = 'soft_sand'           # ماسه نرم
    HARD_SAND = 'hard_sand'           # ماسه سخت
    SOFT_SHALE = 'soft_shale'         # شیل نرم
    HARD_SHALE = 'hard_shale'         # شیل سخت
    LIMESTONE = 'limestone'            # سنگ آهک
    DOLOMITE = 'dolomite'             # دولومیت

class AbnormalCondition(Enum):
    """شرایط غیرعادی حفاری"""
    NORMAL = 'normal'                  # شرایط عادی
    BIT_BALLING = 'bit_balling'        # گلوله شدن مته
    STICK_SLIP = 'stick_slip'          # گیر و رها شدن مته
    VIBRATION = 'vibration'            # ارتعاشات شدید
    FORMATION_CHANGE = 'formation_change'  # تغییر ناگهانی سازند

class DrillCollar:
    """کلاس مشخصات گیره حفاری"""
    def __init__(self, length: float, outer_diameter: float, inner_diameter: float):
        self.length = length                  # طول (متر)
        self.outer_diameter = outer_diameter  # قطر خارجی (متر)
        self.inner_diameter = inner_diameter  # قطر داخلی (متر)
        self.density = 7850                   # چگالی فولاد (کیلوگرم بر متر مکعب)
        
    def calculate_weight(self) -> float:
        """محاسبه وزن گیره حفاری"""
        # محاسبه حجم فولاد
        outer_area = np.pi * (self.outer_diameter/2)**2
        inner_area = np.pi * (self.inner_diameter/2)**2
        volume = (outer_area - inner_area) * self.length
        
        # محاسبه وزن (نیوتن)
        gravity = 9.81  # شتاب گرانش (متر بر مجذور ثانیه)
        return volume * self.density * gravity
    
    def calculate_moment_of_inertia(self) -> float:
        """محاسبه ممان اینرسی گیره حفاری"""
        return (np.pi/32) * (self.outer_diameter**4 - self.inner_diameter**4)

class DrillingPhysics:
    def __init__(self):
        # پارامترهای ثابت فیزیکی
        self.bit_diameter = 0.2159     # قطر مته (متر) - معادل 8.5 اینچ
        self.max_wob = 50000           # حداکثر وزن روی مته (نیوتن)
        self.max_rpm = 200             # حداکثر سرعت چرخش
        
        # تعریف گیره حفاری
        self.drill_collar = DrillCollar(
            length=30.0,              # طول 30 متر
            outer_diameter=0.1778,    # 7 اینچ
            inner_diameter=0.0762      # 3 اینچ
        )
        
        # پارامترهای گل حفاری
        self.mud_density = 1200        # چگالی گل (کیلوگرم بر متر مکعب)
        self.mud_viscosity = 0.03      # ویسکوزیته گل (پاسکال ثانیه)
        self.min_mud_temp = 20         # حداقل دمای گل (درجه سانتیگراد)
        self.max_mud_temp = 80         # حداکثر دمای گل (درجه سانتیگراد)
        
        # ضرایب معادلات
        self.rop_coefficient = 0.5     # ضریب نرخ نفوذ پایه
        self.wear_coefficient = 0.005  # ضریب فرسایش پایه
        self.friction_base = 0.25      # ضریب اصطکاک پایه
        self.pressure_coefficient = 50  # ضریب افت فشار
        
        # تعریف خصوصیات انواع سازند
        self.formation_properties = {
            FormationType.SOFT_SAND: {
                'strength': 0.6,
                'abrasiveness': 0.4,
                'porosity': 0.3,
                'permeability': 0.8
            },
            FormationType.HARD_SAND: {
                'strength': 1.2,
                'abrasiveness': 0.8,
                'porosity': 0.15,
                'permeability': 0.4
            },
            FormationType.SOFT_SHALE: {
                'strength': 0.4,
                'abrasiveness': 0.2,
                'porosity': 0.2,
                'permeability': 0.1
            },
            FormationType.HARD_SHALE: {
                'strength': 0.9,
                'abrasiveness': 0.5,
                'porosity': 0.1,
                'permeability': 0.05
            },
            FormationType.LIMESTONE: {
                'strength': 1.5,
                'abrasiveness': 0.6,
                'porosity': 0.1,
                'permeability': 0.3
            },
            FormationType.DOLOMITE: {
                'strength': 1.8,
                'abrasiveness': 0.9,
                'porosity': 0.05,
                'permeability': 0.2
            }
        }
        
        # وضعیت اولیه
        self.current_formation = FormationType.SOFT_SAND
        self.current_condition = AbnormalCondition.NORMAL
        self.current_temp = 30
    
    def update_temperature(self, depth: float) -> float:
        """محاسبه دما براساس عمق"""
        # گرادیان دمایی: 2.5 درجه per 100 متر
        temp_gradient = 0.025  # درجه سانتیگراد بر متر
        self.current_temp = self.min_mud_temp + (depth * temp_gradient)
        return min(self.current_temp, self.max_mud_temp)
    
    def check_abnormal_conditions(self, 
                                wob: float, 
                                rpm: float, 
                                flow_rate: float,
                                bit_wear: float,
                                vibrations: Dict[str, float]) -> AbnormalCondition:
        """بررسی و تشخیص شرایط غیرعادی"""
        max_vibration = max(vibrations.values())
        
        if max_vibration > 0.8:
            return AbnormalCondition.VIBRATION
        elif bit_wear > 0.7 and wob > 0.8 * self.max_wob:
            return AbnormalCondition.BIT_BALLING
        elif rpm < 0.2 * self.max_rpm and wob > 0.7 * self.max_wob:
            return AbnormalCondition.STICK_SLIP
        else:
            return AbnormalCondition.NORMAL
    
    def calculate_effective_wob(self, wob: float, angle: float = 0) -> float:
        """محاسبه وزن مؤثر روی مته با در نظر گرفتن وزن گیره حفاری و زاویه انحراف"""
        # محاسبه وزن گیره حفاری
        collar_weight = self.drill_collar.calculate_weight()
        
        # محاسبه مؤلفه عمودی با در نظر گرفتن زاویه انحراف
        angle_rad = np.radians(angle)
        effective_collar_weight = collar_weight * np.cos(angle_rad)
        effective_wob = wob * np.cos(angle_rad)
        
        # محاسبه وزن کل روی مته
        total_wob = effective_wob + effective_collar_weight
        
        # محدود کردن به حداکثر مجاز
        return min(total_wob, self.max_wob)
    
    def calculate_torque(self, wob: float, bit_wear: float, rpm: float) -> float:
        """محاسبه گشتاور کل با در نظر گرفتن گیره حفاری و اثر ژیروسکوپی"""
        # گشتاور ناشی از مته
        bit_torque = self.friction_base * (1 + bit_wear) * wob * self.bit_diameter / 2
        
        # گشتاور ناشی از گیره حفاری (اثر ژیروسکوپی)
        collar_inertia = self.drill_collar.calculate_moment_of_inertia()
        angular_velocity = rpm * 2 * np.pi / 60  # تبدیل به رادیان بر ثانیه
        
        # افزایش تأثیر سرعت چرخش بر گشتاور
        rpm_factor = (rpm / self.max_rpm)**1.5  # توان 1.5 برای رابطه غیرخطی
        gyroscopic_torque = collar_inertia * angular_velocity * rpm_factor
        
        # اضافه کردن گشتاور اصطکاکی گیره حفاری
        collar_weight = self.drill_collar.calculate_weight()
        collar_friction_torque = 0.1 * collar_weight * self.drill_collar.outer_diameter/2 * rpm_factor
        
        return bit_torque + gyroscopic_torque + collar_friction_torque
    
    def calculate_rop(self, wob: float, rpm: float, angle: float = 0) -> float:
        """محاسبه نرخ نفوذ با در نظر گرفتن وزن مؤثر و زاویه انحراف"""
        # محاسبه وزن مؤثر روی مته
        effective_wob = self.calculate_effective_wob(wob, angle)
        
        # پارامترهای سازند
        formation_props = self.formation_properties[self.current_formation]
        formation_strength = formation_props['strength']
        
        # نرمال‌سازی پارامترها
        normalized_wob = effective_wob / self.max_wob
        normalized_rpm = rpm / self.max_rpm
        
        # محاسبه ROP پایه
        base_rop = self.rop_coefficient * (
            (normalized_wob**0.8 * normalized_rpm**0.6) / (formation_strength**2)
        ) * 3600
        
        # اعمال تأثیر دما
        temp_factor = 1.0 + 0.001 * (self.current_temp - self.min_mud_temp)
        
        # اعمال تأثیر زاویه انحراف - تأثیر قوی‌تر
        angle_rad = np.radians(angle)
        angle_factor = np.cos(angle_rad)**2  # توان 2 برای تأثیر بیشتر زاویه
        
        # اعمال تأثیر شرایط غیرعادی
        condition_factors = {
            AbnormalCondition.NORMAL: 1.0,
            AbnormalCondition.BIT_BALLING: 0.3,
            AbnormalCondition.STICK_SLIP: 0.5,
            AbnormalCondition.VIBRATION: 0.7,
            AbnormalCondition.FORMATION_CHANGE: 0.8
        }
        condition_factor = condition_factors[self.current_condition]
        
        # محاسبه ROP نهایی
        rop = base_rop * temp_factor * condition_factor * angle_factor
        
        # محدودیت‌های متفاوت برای سازندهای مختلف
        max_rop_factors = {
            FormationType.SOFT_SAND: 30.0,
            FormationType.HARD_SAND: 20.0,
            FormationType.SOFT_SHALE: 25.0,
            FormationType.HARD_SHALE: 15.0,
            FormationType.LIMESTONE: 12.0,
            FormationType.DOLOMITE: 8.0
        }
        
        max_rop = max_rop_factors[self.current_formation] * angle_factor  # اعمال تأثیر زاویه بر حداکثر ROP
        return min(max(rop, 1.0), max_rop)
    
    def calculate_bit_wear(self, 
                          time_hours: float,
                          wob: float,
                          rpm: float,
                          current_wear: float) -> float:
        """محاسبه فرسایش مته با در نظر گرفتن نوع سازند و دما"""
        formation_props = self.formation_properties[self.current_formation]
        abrasiveness = formation_props['abrasiveness']
        
        # نرمال‌سازی پارامترها
        normalized_wob = wob / self.max_wob
        normalized_rpm = rpm / self.max_rpm
        
        # محاسبه نرخ فرسایش پایه
        base_wear_rate = self.wear_coefficient * (
            normalized_wob**1.2 * normalized_rpm**0.8 * abrasiveness
        )
        
        # افزایش نرخ فرسایش با افزایش دما
        temp_factor = 1.0 + 0.002 * (self.current_temp - self.min_mud_temp)
        
        # محاسبه فرسایش جدید
        wear_increment = base_wear_rate * time_hours * temp_factor
        total_wear = current_wear + wear_increment
        
        return min(1.0, total_wear)
    
    def calculate_pressure_loss(self, 
                              flow_rate: float,
                              depth: float,
                              formation_type: Optional[FormationType] = None) -> float:
        """محاسبه افت فشار با در نظر گرفتن نوع سازند و دما"""
        if formation_type is None:
            formation_type = self.current_formation
            
        formation_props = self.formation_properties[formation_type]
        permeability = formation_props['permeability']
        
        # تأثیر دما بر ویسکوزیته
        temp_factor = 1.0 - 0.002 * (self.current_temp - self.min_mud_temp)
        effective_viscosity = self.mud_viscosity * temp_factor
        
        pipe_radius = 0.1  # متر
        base_pressure = self.pressure_coefficient * (
            8 * effective_viscosity * depth * flow_rate
        ) / (np.pi * pipe_radius**4)
        
        # اعمال تأثیر تراوایی سازند
        formation_factor = 1.0 + (1.0 - permeability)
        
        return base_pressure * formation_factor
    
    def calculate_vibrations(self, wob: float, rpm: float, bit_wear: float) -> Dict[str, float]:
        """محاسبه ارتعاشات سیستم حفاری"""
        normalized_wob = wob / self.max_wob
        normalized_rpm = rpm / self.max_rpm
        
        # محاسبه ارتعاشات پایه با روابط غیرخطی
        axial_base = normalized_wob**1.2
        lateral_base = normalized_rpm**1.1
        torsional_base = (normalized_wob * normalized_rpm)**0.9
        
        # افزایش ارتعاشات با فرسایش مته
        wear_factor = 1 + bit_wear
        
        return {
            'axial': min(1.0, axial_base * wear_factor),
            'lateral': min(1.0, lateral_base * wear_factor),
            'torsional': min(1.0, torsional_base * wear_factor)
        }
    
    def simulate_step(self,
                     current_state: Dict[str, float],
                     action: Dict[str, float],
                     timestep: float,
                     angle: float = 0) -> Tuple[Dict[str, float], Dict[str, float]]:
        """شبیه‌سازی یک گام زمانی با در نظر گرفتن تمام شرایط"""
        # بروزرسانی دما
        self.update_temperature(current_state['depth'])
        
        # محدود کردن اعمال کنترلی
        wob = min(action['wob'], self.max_wob)
        rpm = min(action['rpm'], self.max_rpm)
        flow_rate = min(action['flow_rate'], 0.1)
        
        # محاسبه پارامترهای اصلی با در نظر گرفتن زاویه انحراف
        rop = self.calculate_rop(wob, rpm, angle)
        torque = self.calculate_torque(wob, current_state['bit_wear'], rpm)
        new_bit_wear = self.calculate_bit_wear(
            timestep/3600,
            wob,
            rpm,
            current_state['bit_wear']
        )
        pressure = self.calculate_pressure_loss(flow_rate, current_state['depth'])
        vibrations = self.calculate_vibrations(wob, rpm, new_bit_wear)
        
        # بررسی شرایط غیرعادی
        self.current_condition = self.check_abnormal_conditions(
            wob, rpm, flow_rate, new_bit_wear, vibrations
        )
        
        # محاسبه تغییر عمق
        depth_increment = rop * (timestep/3600)
        new_depth = current_state['depth'] + depth_increment
        
        # محاسبه وزن مؤثر روی مته
        effective_wob = self.calculate_effective_wob(wob, angle)
        
        # ساخت وضعیت جدید
        new_state = {
            'depth': new_depth,
            'bit_wear': new_bit_wear,
            'rop': rop,
            'torque': torque,
            'pressure': pressure,
            'vibration_axial': vibrations['axial'],
            'vibration_lateral': vibrations['lateral'],
            'vibration_torsional': vibrations['torsional'],
            'temperature': self.current_temp,
            'condition': self.current_condition.value,
            'effective_wob': effective_wob
        }
        
        return new_state, new_state 