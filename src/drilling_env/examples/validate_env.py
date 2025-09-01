"""
Validation script for the drilling environment.

This script loads historical drilling data and compares it with the
environment's simulated behavior to validate the physics model.
"""

import sys
import pandas as pd
import numpy as np
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from drilling_env.drilling_env import DrillingEnv
import matplotlib.pyplot as plt

# مسیر فایل داده واقعی
DATA_PATH = os.path.join(os.path.dirname(__file__), '../../dataset/sensor_extended_processed.csv')

# خواندن داده واقعی
df = pd.read_csv(DATA_PATH)

# استخراج ستون‌های مورد نیاز برای اکشن (فرض: WOB، RPM، Flow Rate)
# توجه: نام ستون‌ها باید با داده واقعی تطبیق داده شود
WOB_COL = 'Weight on Bit Sensor'
RPM_COL = 'RPM Sensor'
FLOW_COL = 'Flow Sensor'

# استخراج اکشن‌ها از داده واقعی
actions = df[[WOB_COL, RPM_COL, FLOW_COL]].values

# استخراج مقادیر واقعی برای مقایسه (مثلاً ROP، Torque، Pressure)
ROP_COL = 'Rate of Penetration Sensor'
TORQUE_COL = 'Tension Sensor'  # فرضی، بسته به داده واقعی
PRESSURE_COL = 'Pressure Sensor'

real_rop = df[ROP_COL].values
real_torque = df[TORQUE_COL].values
real_pressure = df[PRESSURE_COL].values

# ایجاد محیط شبیه‌ساز
env = DrillingEnv()
obs = env.reset()

sim_rop = []
sim_torque = []
sim_pressure = []

print(f'Total actions in dataset: {len(actions)}')

# محدوده‌های محافظه‌کارانه‌تر برای اکشن‌ها
WOB_MIN, WOB_MAX = 0, 20000
RPM_MIN, RPM_MAX = 0, 80
FLOW_MIN, FLOW_MAX = 0, 0.05

for i, action in enumerate(actions):
    # محدود کردن اکشن‌ها به بازه محافظه‌کارانه
    action = np.clip(action, [WOB_MIN, RPM_MIN, FLOW_MIN], [WOB_MAX, RPM_MAX, FLOW_MAX])
    obs, reward, done, info = env.step(action)
    # ذخیره خروجی شبیه‌ساز
    sim_rop.append(obs[2])  # فرض: ROP در اندیس 2
    sim_torque.append(obs[3])  # فرض: Torque در اندیس 3
    sim_pressure.append(obs[4])  # فرض: Pressure در اندیس 4
    if done:
        print(f'Simulation stopped at step {i+1} (done=True).')
        print(f'Current state: {env.current_state}')
        print(f'Info: {info}')
        break
else:
    print('Simulation completed all steps without early termination.')

# تبدیل Pressure شبیه‌ساز به kPa برای مقایسه با داده واقعی
sim_pressure_kpa = [p / 1000 for p in sim_pressure]

# رسم نمودار مقایسه‌ای برای 100 گام اول
steps = min(100, len(sim_rop), len(real_rop))
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.plot(range(steps), real_rop[:steps], label='Real ROP')
plt.plot(range(steps), sim_rop[:steps], label='Sim ROP')
plt.title('ROP (100 steps)')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(range(steps), real_torque[:steps], label='Real Torque')
plt.plot(range(steps), sim_torque[:steps], label='Sim Torque')
plt.title('Torque (100 steps)')
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(range(steps), real_pressure[:steps], label='Real Pressure (kPa)')
plt.plot(range(steps), sim_pressure_kpa[:steps], label='Sim Pressure (kPa)')
plt.title('Pressure (100 steps, kPa)')
plt.legend()
plt.tight_layout()
plt.show()

# محاسبه خطاها
rop_mae = np.mean(np.abs(np.array(sim_rop) - real_rop[:len(sim_rop)]))
torque_mae = np.mean(np.abs(np.array(sim_torque) - real_torque[:len(sim_torque)]))
pressure_mae_kpa = np.mean(np.abs(np.array(sim_pressure_kpa) - real_pressure[:len(sim_pressure_kpa)]))

print('--- Validation Results ---')
print(f'ROP MAE: {rop_mae:.2f}')
print(f'Torque MAE: {torque_mae:.2f}')
print(f'Pressure MAE (kPa): {pressure_mae_kpa:.2f}')
print(f'Total steps compared: {len(sim_rop)}') 