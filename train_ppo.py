import gym
from stable_baselines3 import PPO
import sys
import os

# اضافه کردن مسیر پروژه به sys.path
current_dir = os.getcwd()
sys.path.append(current_dir)

from drilling_env.drilling_env import DrillingEnv

def main():
    print("=== شروع آموزش PPO روی محیط حفاری ===")
    
    # ساخت محیط
    print("ساخت محیط DrillingEnv...")
    env = DrillingEnv()
    
    # ساخت مدل PPO
    print("ساخت مدل PPO...")
    model = PPO("MlpPolicy", env, verbose=1)
    
    # آموزش مدل
    print("شروع آموزش مدل...")
    model.learn(total_timesteps=100_000)
    
    # ذخیره مدل آموزش‌دیده
    print("ذخیره مدل...")
    model.save("ppo_drilling_env")
    
    # تست مدل آموزش‌دیده
    print("تست مدل آموزش‌دیده...")
    obs = env.reset()
    total_reward = 0
    steps = 0
    
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if steps % 100 == 0:
            print(f"گام {steps}: پاداش کل = {total_reward:.2f}")
        
        if done:
            print(f"اپیزود تمام شد. پاداش کل = {total_reward:.2f}")
            obs = env.reset()
            total_reward = 0
    
    print("=== آموزش و تست کامل شد ===")

if __name__ == "__main__":
    main() 