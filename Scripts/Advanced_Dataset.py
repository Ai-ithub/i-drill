"""
Advanced SRS-Compliant Synthetic Drilling Data Generator
========================================================

این اسکریپت برای تولید داده‌های مصنوعی حفاری مطابق با مشخصات SRS پیشرفته طراحی شده است.
شامل شبیه‌سازی فیزیک واقعی، پروفایل چاه، خواص سازند و رویدادهای حفاری.

نیازمندی‌های SRS پیاده‌سازی شده:
- FR-01: Comprehensive Well Profile with Geological Layers
- FR-02: Physics-based Drilling Simulation
- FR-03: Formation Property Generation with Stratigraphic Changes
- FR-04: Equipment Failure Simulation with Progressive Degradation
- FR-05: Drilling Event Simulation (stick-slip, whirl, lost circulation)
- FR-06: Real-time Data Streaming (1-second resolution)
- FR-07: Multiple Data Export Formats
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import random
from pathlib import Path

# اضافه کردن مسیر ماژول‌های فیزیک حفاری
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'drilling_env'))
from drilling_physics import DrillingPhysics, FormationType, AbnormalCondition

class GeologicalProfile:
    """کلاس تولید پروفایل زمین‌شناسی جامع چاه"""
    
    def __init__(self, total_depth: float = 3000.0):
        self.total_depth = total_depth
        self.layers = self._generate_geological_layers()
        
    def _generate_geological_layers(self) -> List[Dict]:
        """تولید لایه‌های زمین‌شناسی واقعی"""
        layers = [
            {
                'name': 'Surface_Sand',
                'formation_type': FormationType.SOFT_SAND,
                'top_depth': 0.0,
                'bottom_depth': 200.0,
                'lithology': 'Unconsolidated Sand',
                'porosity_range': (25, 35),
                'density_range': (1.8, 2.1),
                'gamma_ray_range': (30, 60),
                'resistivity_range': (5, 20)
            },
            {
                'name': 'Shale_Cap',
                'formation_type': FormationType.SOFT_SHALE,
                'top_depth': 200.0,
                'bottom_depth': 800.0,
                'lithology': 'Marine Shale',
                'porosity_range': (8, 15),
                'density_range': (2.2, 2.4),
                'gamma_ray_range': (80, 120),
                'resistivity_range': (1, 5)
            },
            {
                'name': 'Sandstone_Reservoir',
                'formation_type': FormationType.HARD_SAND,
                'top_depth': 800.0,
                'bottom_depth': 1500.0,
                'lithology': 'Quartz Sandstone',
                'porosity_range': (15, 25),
                'density_range': (2.3, 2.5),
                'gamma_ray_range': (40, 80),
                'resistivity_range': (10, 100)
            },
            {
                'name': 'Limestone_Formation',
                'formation_type': FormationType.LIMESTONE,
                'top_depth': 1500.0,
                'bottom_depth': 2200.0,
                'lithology': 'Fossiliferous Limestone',
                'porosity_range': (5, 12),
                'density_range': (2.5, 2.7),
                'gamma_ray_range': (20, 50),
                'resistivity_range': (50, 200)
            },
            {
                'name': 'Deep_Shale',
                'formation_type': FormationType.HARD_SHALE,
                'top_depth': 2200.0,
                'bottom_depth': 2800.0,
                'lithology': 'Overpressured Shale',
                'porosity_range': (3, 8),
                'density_range': (2.4, 2.6),
                'gamma_ray_range': (100, 150),
                'resistivity_range': (0.5, 3)
            },
            {
                'name': 'Basement_Dolomite',
                'formation_type': FormationType.DOLOMITE,
                'top_depth': 2800.0,
                'bottom_depth': 3000.0,
                'lithology': 'Crystalline Dolomite',
                'porosity_range': (2, 5),
                'density_range': (2.7, 2.9),
                'gamma_ray_range': (15, 40),
                'resistivity_range': (100, 500)
            }
        ]
        return layers
    
    def get_formation_at_depth(self, depth: float) -> Dict:
        """دریافت اطلاعات سازند در عمق مشخص"""
        for layer in self.layers:
            if layer['top_depth'] <= depth < layer['bottom_depth']:
                return layer
        # اگر عمق از کل چاه بیشتر باشد، آخرین لایه را برگردان
        return self.layers[-1]
    
    def get_formation_properties(self, depth: float) -> Dict:
        """محاسبه خواص سازند در عمق مشخص با تغییرات تدریجی"""
        layer = self.get_formation_at_depth(depth)
        
        # محاسبه موقعیت نسبی در لایه
        layer_thickness = layer['bottom_depth'] - layer['top_depth']
        relative_position = (depth - layer['top_depth']) / layer_thickness
        
        # تولید خواص با تغییرات تدریجی در لایه
        properties = {}
        for prop in ['porosity', 'density', 'gamma_ray', 'resistivity']:
            prop_range = layer[f'{prop}_range']
            # تغییرات تدریجی + نویز تصادفی
            base_value = prop_range[0] + (prop_range[1] - prop_range[0]) * relative_position
            noise = np.random.normal(0, (prop_range[1] - prop_range[0]) * 0.05)
            properties[prop] = max(prop_range[0], min(prop_range[1], base_value + noise))
        
        properties['formation_type'] = layer['formation_type']
        properties['lithology'] = layer['lithology']
        properties['layer_name'] = layer['name']
        
        return properties

class DrillingEventSimulator:
    """شبیه‌ساز رویدادهای حفاری پیشرفته"""
    
    def __init__(self):
        self.event_probabilities = {
            'stick_slip': 0.02,      # 2% احتمال در هر گام
            'whirl': 0.015,          # 1.5% احتمال
            'lost_circulation': 0.01, # 1% احتمال
            'gas_influx': 0.005,     # 0.5% احتمال
            'wellbore_instability': 0.008  # 0.8% احتمال
        }
        
        self.active_events = []
        self.event_durations = {
            'stick_slip': (30, 300),      # 30 ثانیه تا 5 دقیقه
            'whirl': (60, 600),           # 1 تا 10 دقیقه
            'lost_circulation': (300, 1800), # 5 تا 30 دقیقه
            'gas_influx': (120, 900),     # 2 تا 15 دقیقه
            'wellbore_instability': (600, 3600)  # 10 تا 60 دقیقه
        }
    
    def update_events(self, current_time: datetime, formation_type: FormationType) -> List[str]:
        """بروزرسانی رویدادهای فعال"""
        # حذف رویدادهای منقضی شده
        self.active_events = [
            event for event in self.active_events 
            if event['end_time'] > current_time
        ]
        
        # بررسی رویدادهای جدید
        for event_type, base_prob in self.event_probabilities.items():
            # تنظیم احتمال براساس نوع سازند
            formation_factor = self._get_formation_event_factor(formation_type, event_type)
            adjusted_prob = base_prob * formation_factor
            
            if np.random.random() < adjusted_prob:
                duration_range = self.event_durations[event_type]
                duration = np.random.randint(duration_range[0], duration_range[1])
                
                new_event = {
                    'type': event_type,
                    'start_time': current_time,
                    'end_time': current_time + timedelta(seconds=duration),
                    'severity': np.random.uniform(0.3, 1.0)
                }
                self.active_events.append(new_event)
        
        return [event['type'] for event in self.active_events]
    
    def _get_formation_event_factor(self, formation_type: FormationType, event_type: str) -> float:
        """محاسبه ضریب احتمال رویداد براساس نوع سازند"""
        factors = {
            FormationType.SOFT_SAND: {
                'stick_slip': 0.5, 'whirl': 0.7, 'lost_circulation': 1.5,
                'gas_influx': 1.2, 'wellbore_instability': 0.8
            },
            FormationType.HARD_SAND: {
                'stick_slip': 1.2, 'whirl': 1.0, 'lost_circulation': 0.8,
                'gas_influx': 0.9, 'wellbore_instability': 0.6
            },
            FormationType.SOFT_SHALE: {
                'stick_slip': 0.8, 'whirl': 0.6, 'lost_circulation': 0.5,
                'gas_influx': 0.3, 'wellbore_instability': 2.0
            },
            FormationType.HARD_SHALE: {
                'stick_slip': 1.5, 'whirl': 1.3, 'lost_circulation': 0.4,
                'gas_influx': 0.2, 'wellbore_instability': 1.8
            },
            FormationType.LIMESTONE: {
                'stick_slip': 1.8, 'whirl': 1.5, 'lost_circulation': 2.0,
                'gas_influx': 1.5, 'wellbore_instability': 0.7
            },
            FormationType.DOLOMITE: {
                'stick_slip': 2.0, 'whirl': 1.8, 'lost_circulation': 1.8,
                'gas_influx': 1.3, 'wellbore_instability': 0.5
            }
        }
        
        return factors.get(formation_type, {}).get(event_type, 1.0)
    
    def get_event_effects(self) -> Dict[str, float]:
        """محاسبه تأثیرات رویدادهای فعال بر پارامترهای حفاری"""
        effects = {
            'rop_factor': 1.0,
            'torque_factor': 1.0,
            'vibration_factor': 1.0,
            'pressure_factor': 1.0,
            'mud_loss_rate': 0.0
        }
        
        for event in self.active_events:
            event_type = event['type']
            severity = event['severity']
            
            if event_type == 'stick_slip':
                effects['rop_factor'] *= (0.3 + 0.4 * (1 - severity))
                effects['torque_factor'] *= (1.5 + 0.8 * severity)
                effects['vibration_factor'] *= (2.0 + severity)
                
            elif event_type == 'whirl':
                effects['rop_factor'] *= (0.6 + 0.3 * (1 - severity))
                effects['vibration_factor'] *= (1.8 + 0.7 * severity)
                
            elif event_type == 'lost_circulation':
                effects['pressure_factor'] *= (0.4 + 0.4 * (1 - severity))
                effects['mud_loss_rate'] += severity * 50  # گالن در دقیقه
                
            elif event_type == 'gas_influx':
                effects['pressure_factor'] *= (1.3 + 0.5 * severity)
                effects['mud_loss_rate'] -= severity * 20  # کاهش جریان
                
            elif event_type == 'wellbore_instability':
                effects['rop_factor'] *= (0.5 + 0.3 * (1 - severity))
                effects['torque_factor'] *= (1.2 + 0.3 * severity)
        
        return effects

class EquipmentDegradationSimulator:
    """شبیه‌ساز تخریب تدریجی تجهیزات"""
    
    def __init__(self):
        self.equipment_states = {
            'bit': {'wear': 0.0, 'efficiency': 1.0},
            'mud_pump': {'wear': 0.0, 'efficiency': 1.0},
            'rotary_table': {'wear': 0.0, 'efficiency': 1.0},
            'drawworks': {'wear': 0.0, 'efficiency': 1.0}
        }
        
        self.degradation_rates = {
            'bit': 0.001,           # نرخ فرسایش مته
            'mud_pump': 0.0005,     # نرخ فرسایش پمپ گل
            'rotary_table': 0.0003, # نرخ فرسایش میز چرخان
            'drawworks': 0.0002     # نرخ فرسایش جرثقیل
        }
    
    def update_degradation(self, operating_conditions: Dict, timestep_hours: float):
        """بروزرسانی وضعیت تخریب تجهیزات"""
        # فرسایش مته
        wob_factor = operating_conditions.get('wob', 0) / 50000
        rpm_factor = operating_conditions.get('rpm', 0) / 200
        formation_hardness = operating_conditions.get('formation_hardness', 1.0)
        
        bit_wear_rate = self.degradation_rates['bit'] * (
            1 + wob_factor * 2 + rpm_factor * 1.5 + formation_hardness
        )
        self.equipment_states['bit']['wear'] += bit_wear_rate * timestep_hours
        
        # فرسایش پمپ گل
        flow_factor = operating_conditions.get('flow_rate', 0) / 0.1
        pressure_factor = operating_conditions.get('pressure', 0) / 5000
        
        pump_wear_rate = self.degradation_rates['mud_pump'] * (
            1 + flow_factor * 1.5 + pressure_factor * 1.2
        )
        self.equipment_states['mud_pump']['wear'] += pump_wear_rate * timestep_hours
        
        # بروزرسانی کارایی براساس فرسایش
        for equipment in self.equipment_states:
            wear = self.equipment_states[equipment]['wear']
            # کاهش کارایی با افزایش فرسایش
            self.equipment_states[equipment]['efficiency'] = max(0.1, 1.0 - wear)
    
    def get_failure_probability(self, equipment: str) -> float:
        """محاسبه احتمال خرابی تجهیز"""
        wear = self.equipment_states[equipment]['wear']
        # احتمال خرابی افزایش می‌یابد با فرسایش
        return min(0.1, wear * 0.1)  # حداکثر 10% احتمال خرابی
    
    def simulate_failure(self) -> List[str]:
        """شبیه‌سازی خرابی تجهیزات"""
        failed_equipment = []
        for equipment in self.equipment_states:
            failure_prob = self.get_failure_probability(equipment)
            if np.random.random() < failure_prob:
                failed_equipment.append(equipment)
                # بازنشانی فرسایش بعد از تعمیر
                self.equipment_states[equipment]['wear'] *= 0.1
                self.equipment_states[equipment]['efficiency'] = 0.9
        
        return failed_equipment

class AdvancedSRSDataGenerator:
    """تولیدکننده داده‌های پیشرفته مطابق SRS"""
    
    def __init__(self):
        # پارامترهای SRS
        self.duration_days = 180  # 6 ماه
        self.freq_seconds = 1     # فرکانس 1 ثانیه
        self.start_time = datetime(2024, 1, 1, 0, 0, 0)
        
        # محاسبه تعداد رکوردها
        self.total_seconds = self.duration_days * 24 * 3600
        self.total_records = self.total_seconds
        
        # ایجاد نمونه‌های شبیه‌ساز
        self.physics = DrillingPhysics()
        self.geology = GeologicalProfile()
        self.events = DrillingEventSimulator()
        self.equipment = EquipmentDegradationSimulator()
        
        # پارامترهای کنترلی
        self.target_wob = 25000    # هدف WOB
        self.target_rpm = 120      # هدف RPM
        self.target_flow = 0.06    # هدف نرخ جریان
        
        print(f"🎯 Advanced SRS Dataset Generator Initialized")
        print(f"📊 Configuration:")
        print(f"   - Duration: {self.duration_days} days (6 months)")
        print(f"   - Frequency: {self.freq_seconds} second")
        print(f"   - Total Records: {self.total_records:,}")
        print(f"   - Well Profile: {len(self.geology.layers)} geological layers")
        print(f"   - Physics Engine: Advanced drilling simulation")
        print(f"   - Event Simulation: 5 types of drilling events")
        print(f"   - Equipment Degradation: 4 equipment types")
    
    def generate_advanced_dataset(self) -> pd.DataFrame:
        """تولید دیتاست پیشرفته با تمام قابلیت‌های SRS"""
        print(f"\n🚀 Starting Advanced Dataset Generation...")
        
        data_records = []
        current_time = self.start_time
        current_depth = 0.0
        
        # وضعیت اولیه سیستم
        system_state = {
            'depth': 0.0,
            'bit_wear': 0.0,
            'rop': 0.0,
            'torque': 0.0,
            'pressure': 0.0,
            'vibration_axial': 0.0,
            'vibration_lateral': 0.0,
            'vibration_torsional': 0.0,
            'temperature': 25.0
        }
        
        # حلقه اصلی تولید داده
        for i in range(self.total_records):
            if i % 100000 == 0:
                progress = (i / self.total_records) * 100
                print(f"   Progress: {progress:.1f}% - Depth: {current_depth:.1f}m - Time: {current_time}")
            
            # دریافت خواص زمین‌شناسی
            formation_props = self.geology.get_formation_properties(current_depth)
            current_formation = formation_props['formation_type']
            
            # بروزرسانی نوع سازند در فیزیک
            self.physics.current_formation = current_formation
            
            # بروزرسانی رویدادهای حفاری
            active_events = self.events.update_events(current_time, current_formation)
            event_effects = self.events.get_event_effects()
            
            # بروزرسانی تخریب تجهیزات
            operating_conditions = {
                'wob': self.target_wob,
                'rpm': self.target_rpm,
                'flow_rate': self.target_flow,
                'formation_hardness': self.physics.formation_properties[current_formation]['strength']
            }
            self.equipment.update_degradation(operating_conditions, 1/3600)  # 1 ثانیه
            
            # تنظیم پارامترهای کنترلی با نویز
            wob = self.target_wob + np.random.normal(0, 2000)
            rpm = self.target_rpm + np.random.normal(0, 10)
            flow_rate = self.target_flow + np.random.normal(0, 0.005)
            
            # محدود کردن به محدوده‌های مجاز
            wob = np.clip(wob, 5000, 50000)
            rpm = np.clip(rpm, 50, 200)
            flow_rate = np.clip(flow_rate, 0.02, 0.1)
            
            # شبیه‌سازی فیزیک با اعمال تأثیرات رویدادها
            action = {'wob': wob, 'rpm': rpm, 'flow_rate': flow_rate}
            new_state, _ = self.physics.simulate_step(system_state, action, 1.0)
            
            # اعمال تأثیرات رویدادها
            new_state['rop'] *= event_effects['rop_factor']
            new_state['torque'] *= event_effects['torque_factor']
            new_state['pressure'] *= event_effects['pressure_factor']
            new_state['vibration_axial'] *= event_effects['vibration_factor']
            new_state['vibration_lateral'] *= event_effects['vibration_factor']
            new_state['vibration_torsional'] *= event_effects['vibration_factor']
            
            # اعمال تأثیرات تخریب تجهیزات
            bit_efficiency = self.equipment.equipment_states['bit']['efficiency']
            pump_efficiency = self.equipment.equipment_states['mud_pump']['efficiency']
            
            new_state['rop'] *= bit_efficiency
            new_state['pressure'] /= pump_efficiency
            
            # بروزرسانی عمق
            depth_increment = new_state['rop'] / 3600  # تبدیل از فوت/ساعت به فوت/ثانیه
            current_depth += depth_increment
            new_state['depth'] = current_depth
            
            # شبیه‌سازی خرابی تجهیزات
            failed_equipment = self.equipment.simulate_failure()
            
            # ساخت رکورد داده
            record = {
                'Timestamp': current_time,
                'Well_ID': 'WELL_01',
                'Depth': current_depth,
                
                # Core Drilling Parameters
                'WOB': wob,
                'RPM': rpm,
                'Torque': new_state['torque'],
                'ROP': new_state['rop'],
                
                # Mud System Parameters
                'Mud_Flow_Rate': flow_rate * 15850,  # تبدیل به gpm
                'Standpipe_Pressure': new_state['pressure'],
                'Mud_Temperature': new_state['temperature'],
                
                # Formation Evaluation Parameters
                'Gamma_Ray': formation_props['gamma_ray'],
                'Resistivity': formation_props['resistivity'],
                'Density': formation_props['density'],
                'Porosity': formation_props['porosity'],
                
                # Mechanical Parameters
                'Hook_Load': wob + np.random.normal(200000, 20000),
                'Vibration': max(new_state['vibration_axial'], 
                               new_state['vibration_lateral'], 
                               new_state['vibration_torsional']) * 10,
                
                # Geological Information
                'Formation_Type': current_formation.value,
                'Lithology': formation_props['lithology'],
                'Layer_Name': formation_props['layer_name'],
                
                # Drilling Events
                'Active_Events': ','.join(active_events) if active_events else 'None',
                'Event_Count': len(active_events),
                
                # Equipment Status
                'Bit_Wear': self.equipment.equipment_states['bit']['wear'],
                'Pump_Efficiency': pump_efficiency,
                'Failed_Equipment': ','.join(failed_equipment) if failed_equipment else 'None',
                
                # Data Quality Flags
                'Maintenance_Flag': 1 if failed_equipment else 0,
                'Abnormal_Condition': self.physics.current_condition.value
            }
            
            data_records.append(record)
            system_state = new_state
            current_time += timedelta(seconds=1)
        
        print(f"\n✅ Dataset Generation Complete!")
        print(f"📈 Final Statistics:")
        print(f"   - Total Records: {len(data_records):,}")
        print(f"   - Final Depth: {current_depth:.1f} meters")
        print(f"   - Duration: {(current_time - self.start_time).days} days")
        
        return pd.DataFrame(data_records)
    
    def export_multiple_formats(self, df: pd.DataFrame, output_dir: str = "output_advanced_srs"):
        """صادرات داده‌ها در فرمت‌های مختلف"""
        os.makedirs(output_dir, exist_ok=True)
        
        base_filename = "WELL_01_Advanced_6months_1sec"
        
        print(f"\n💾 Exporting to Multiple Formats...")
        
        # 1. Parquet (بهینه برای تحلیل)
        parquet_path = os.path.join(output_dir, f"{base_filename}.parquet")
        df.to_parquet(parquet_path, index=False)
        print(f"   ✅ Parquet: {parquet_path}")
        
        # 2. CSV (سازگاری عمومی)
        csv_path = os.path.join(output_dir, f"{base_filename}.csv")
        df.to_csv(csv_path, index=False)
        print(f"   ✅ CSV: {csv_path}")
        
        # 3. JSON (برای API و وب)
        json_path = os.path.join(output_dir, f"{base_filename}.json")
        # تبدیل timestamp به string برای JSON
        df_json = df.copy()
        df_json['Timestamp'] = df_json['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_json.to_json(json_path, orient='records', indent=2)
        print(f"   ✅ JSON: {json_path}")
        
        # 4. Metadata file
        metadata = {
            'dataset_info': {
                'name': 'Advanced SRS-Compliant Drilling Dataset',
                'version': '2.0',
                'generated_at': datetime.now().isoformat(),
                'total_records': len(df),
                'duration_days': self.duration_days,
                'frequency_seconds': self.freq_seconds,
                'well_id': 'WELL_01'
            },
            'geological_profile': {
                'total_depth': self.geology.total_depth,
                'layers': [
                    {
                        'name': layer['name'],
                        'lithology': layer['lithology'],
                        'top_depth': layer['top_depth'],
                        'bottom_depth': layer['bottom_depth']
                    }
                    for layer in self.geology.layers
                ]
            },
            'parameters': {
                'drilling_parameters': [
                    'WOB', 'RPM', 'Torque', 'ROP', 'Mud_Flow_Rate', 
                    'Standpipe_Pressure', 'Mud_Temperature'
                ],
                'formation_parameters': [
                    'Gamma_Ray', 'Resistivity', 'Density', 'Porosity'
                ],
                'mechanical_parameters': [
                    'Hook_Load', 'Vibration'
                ],
                'advanced_features': [
                    'Formation_Type', 'Lithology', 'Active_Events', 
                    'Bit_Wear', 'Equipment_Status'
                ]
            }
        }
        
        metadata_path = os.path.join(output_dir, f"{base_filename}_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Metadata: {metadata_path}")
        
        # 5. Summary Statistics
        summary_path = os.path.join(output_dir, f"{base_filename}_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Advanced SRS-Compliant Drilling Dataset Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset Size: {len(df):,} records\n")
            f.write(f"Time Range: {df['Timestamp'].min()} to {df['Timestamp'].max()}\n")
            f.write(f"Depth Range: {df['Depth'].min():.1f} - {df['Depth'].max():.1f} meters\n\n")
            
            f.write("Parameter Ranges:\n")
            f.write("-" * 20 + "\n")
            for col in ['WOB', 'RPM', 'Torque', 'ROP', 'Gamma_Ray', 'Resistivity', 'Density', 'Porosity']:
                if col in df.columns:
                    f.write(f"{col}: {df[col].min():.2f} - {df[col].max():.2f}\n")
            
            f.write(f"\nFormation Types:\n")
            f.write("-" * 20 + "\n")
            formation_counts = df['Formation_Type'].value_counts()
            for formation, count in formation_counts.items():
                percentage = (count / len(df)) * 100
                f.write(f"{formation}: {count:,} records ({percentage:.1f}%)\n")
        
        print(f"   ✅ Summary: {summary_path}")
        
        return output_dir

def main():
    """تابع اصلی برای اجرای تولید داده‌های پیشرفته"""
    print("🔥 Advanced SRS-Compliant Drilling Data Generator")
    print("=" * 60)
    
    # ایجاد تولیدکننده داده
    generator = AdvancedSRSDataGenerator()
    
    # تولید دیتاست
    dataset = generator.generate_advanced_dataset()
    
    # صادرات در فرمت‌های مختلف
    output_directory = generator.export_multiple_formats(dataset)
    
    print(f"\n🎉 Advanced Dataset Generation Complete!")
    print(f"📁 Output Directory: {output_directory}")
    print(f"📊 Total Records Generated: {len(dataset):,}")
    print(f"🔬 Features Implemented:")
    print(f"   ✅ FR-01: Comprehensive Well Profile with 6 Geological Layers")
    print(f"   ✅ FR-02: Physics-based Drilling Simulation")
    print(f"   ✅ FR-03: Formation Property Generation with Stratigraphic Changes")
    print(f"   ✅ FR-04: Equipment Failure Simulation with Progressive Degradation")
    print(f"   ✅ FR-05: Drilling Event Simulation (5 event types)")
    print(f"   ✅ FR-06: Real-time Data Streaming (1-second resolution)")
    print(f"   ✅ FR-07: Multiple Export Formats (Parquet, CSV, JSON)")

if __name__ == "__main__":
    main()