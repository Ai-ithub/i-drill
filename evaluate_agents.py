import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO, SAC
import sys
import os
from collections import defaultdict
import pandas as pd

# اضافه کردن مسیر پروژه به sys.path
current_dir = os.getcwd()
sys.path.append(current_dir)

from drilling_env.drilling_env import DrillingEnv

class AgentEvaluator:
    def __init__(self, env):
        self.env = env
        self.results = {}
        
    def evaluate_agent(self, model, agent_name, num_episodes=10):
        """ارزیابی کامل یک عامل"""
        print(f"\n=== ارزیابی {agent_name} ===")
        
        episode_rewards = []
        episode_lengths = []
        episode_depths = []
        episode_bit_wears = []
        episode_rops = []
        episode_torques = []
        episode_pressures = []
        episode_vibrations = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            total_reward = 0
            steps = 0
            episode_data = {
                'rewards': [],
                'depths': [],
                'bit_wears': [],
                'rops': [],
                'torques': [],
                'pressures': [],
                'vibrations': []
            }
            
            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                
                total_reward += reward
                steps += 1
                
                # ذخیره داده‌های اپیزود
                episode_data['rewards'].append(reward)
                episode_data['depths'].append(obs[0])  # depth
                episode_data['bit_wears'].append(obs[1])  # bit_wear
                episode_data['rops'].append(obs[2])  # rop
                episode_data['torques'].append(obs[3])  # torque
                episode_data['pressures'].append(obs[4])  # pressure
                episode_data['vibrations'].append(
                    (obs[5] + obs[6] + obs[7]) / 3  # vibration_axial, lateral, torsional
                )
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            episode_depths.append(max(episode_data['depths']))
            episode_bit_wears.append(max(episode_data['bit_wears']))
            episode_rops.append(np.mean(episode_data['rops']))
            episode_torques.append(np.mean(episode_data['torques']))
            episode_pressures.append(np.mean(episode_data['pressures']))
            episode_vibrations.append(np.mean(episode_data['vibrations']))
            
            print(f"اپیزود {episode + 1}: پاداش = {total_reward:.2f}, طول = {steps}")
        
        # محاسبه آمار
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'mean_depth': np.mean(episode_depths),
            'mean_bit_wear': np.mean(episode_bit_wears),
            'mean_rop': np.mean(episode_rops),
            'mean_torque': np.mean(episode_torques),
            'mean_pressure': np.mean(episode_pressures),
            'mean_vibration': np.mean(episode_vibrations),
            'episode_data': episode_data
        }
        
        self.results[agent_name] = results
        return results
    
    def compare_agents(self):
        """مقایسه عملکرد دو عامل"""
        print("\n" + "="*60)
        print("📊 مقایسه عملکرد عوامل")
        print("="*60)
        
        comparison_data = []
        for agent_name, results in self.results.items():
            comparison_data.append({
                'عامل': agent_name,
                'میانگین پاداش': f"{results['mean_reward']:.2f} ± {results['std_reward']:.2f}",
                'میانگین طول اپیزود': f"{results['mean_length']:.1f} ± {results['std_length']:.1f}",
                'میانگین عمق': f"{results['mean_depth']:.2f}",
                'میانگین سایش مته': f"{results['mean_bit_wear']:.6f}",
                'میانگین ROP': f"{results['mean_rop']:.2f}",
                'میانگین گشتاور': f"{results['mean_torque']:.2f}",
                'میانگین فشار': f"{results['mean_pressure']:.2f}",
                'میانگین ارتعاش': f"{results['mean_vibration']:.4f}"
            })
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        return df
    
    def plot_performance_comparison(self):
        """رسم نمودار مقایسه عملکرد"""
        if len(self.results) < 2:
            print("نیاز به حداقل دو عامل برای مقایسه")
            return
        
        # تنظیم فونت فارسی
        plt.rcParams['font.family'] = 'DejaVu Sans'
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('مقایسه عملکرد عوامل PPO و SAC', fontsize=16, fontweight='bold')
        
        # 1. پاداش
        agents = list(self.results.keys())
        rewards = [self.results[agent]['mean_reward'] for agent in agents]
        reward_stds = [self.results[agent]['std_reward'] for agent in agents]
        
        axes[0, 0].bar(agents, rewards, yerr=reward_stds, capsize=5, alpha=0.7)
        axes[0, 0].set_title('میانگین پاداش')
        axes[0, 0].set_ylabel('پاداش')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. طول اپیزود
        lengths = [self.results[agent]['mean_length'] for agent in agents]
        length_stds = [self.results[agent]['std_length'] for agent in agents]
        
        axes[0, 1].bar(agents, lengths, yerr=length_stds, capsize=5, alpha=0.7, color='orange')
        axes[0, 1].set_title('میانگین طول اپیزود')
        axes[0, 1].set_ylabel('تعداد گام‌ها')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. عمق
        depths = [self.results[agent]['mean_depth'] for agent in agents]
        axes[0, 2].bar(agents, depths, alpha=0.7, color='green')
        axes[0, 2].set_title('میانگین عمق حفاری')
        axes[0, 2].set_ylabel('عمق (متر)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. سایش مته
        bit_wears = [self.results[agent]['mean_bit_wear'] for agent in agents]
        axes[1, 0].bar(agents, bit_wears, alpha=0.7, color='red')
        axes[1, 0].set_title('میانگین سایش مته')
        axes[1, 0].set_ylabel('سایش مته')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. ROP
        rops = [self.results[agent]['mean_rop'] for agent in agents]
        axes[1, 1].bar(agents, rops, alpha=0.7, color='purple')
        axes[1, 1].set_title('میانگین نرخ حفاری (ROP)')
        axes[1, 1].set_ylabel('ROP (متر/ساعت)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. ارتعاش
        vibrations = [self.results[agent]['mean_vibration'] for agent in agents]
        axes[1, 2].bar(agents, vibrations, alpha=0.7, color='brown')
        axes[1, 2].set_title('میانگین ارتعاش')
        axes[1, 2].set_ylabel('سطح ارتعاش')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('agent_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n📈 نمودار مقایسه عملکرد در فایل 'agent_performance_comparison.png' ذخیره شد.")
    
    def generate_report(self):
        """تولید گزارش کامل ارزیابی"""
        print("\n" + "="*60)
        print("📋 گزارش کامل ارزیابی عوامل")
        print("="*60)
        
        for agent_name, results in self.results.items():
            print(f"\n🔍 {agent_name}:")
            print(f"   📊 پاداش: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"   ⏱️  طول اپیزود: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
            print(f"   🕳️  عمق: {results['mean_depth']:.2f} متر")
            print(f"   🔧 سایش مته: {results['mean_bit_wear']:.6f}")
            print(f"   ⚡ ROP: {results['mean_rop']:.2f} متر/ساعت")
            print(f"   🔄 گشتاور: {results['mean_torque']:.2f} N.m")
            print(f"   💨 فشار: {results['mean_pressure']:.2f} Pa")
            print(f"   📳 ارتعاش: {results['mean_vibration']:.4f}")
        
        # تعیین برنده
        if len(self.results) >= 2:
            best_agent = max(self.results.keys(), key=lambda x: self.results[x]['mean_reward'])
            print(f"\n🏆 برنده: {best_agent}")
            print(f"   با پاداش: {self.results[best_agent]['mean_reward']:.2f}")

def main():
    print("=== ارزیابی عملکرد عوامل PPO و SAC ===")
    
    # ساخت محیط
    print("ساخت محیط DrillingEnv...")
    env = DrillingEnv()
    
    # ساخت ارزیاب
    evaluator = AgentEvaluator(env)
    
    try:
        # بارگذاری مدل PPO
        print("بارگذاری مدل PPO...")
        ppo_model = PPO.load("ppo_drilling_env")
        evaluator.evaluate_agent(ppo_model, "PPO", num_episodes=5)
        
        # بارگذاری مدل SAC
        print("بارگذاری مدل SAC...")
        sac_model = SAC.load("sac_drilling_env")
        evaluator.evaluate_agent(sac_model, "SAC", num_episodes=5)
        
        # مقایسه عوامل
        comparison_df = evaluator.compare_agents()
        
        # رسم نمودار
        evaluator.plot_performance_comparison()
        
        # تولید گزارش
        evaluator.generate_report()
        
        # ذخیره نتایج
        comparison_df.to_csv('agent_evaluation_results.csv', index=False, encoding='utf-8-sig')
        print("\n💾 نتایج در فایل 'agent_evaluation_results.csv' ذخیره شد.")
        
    except FileNotFoundError as e:
        print(f"❌ خطا: فایل مدل یافت نشد - {e}")
        print("لطفاً ابتدا مدل‌ها را آموزش دهید.")
    except Exception as e:
        print(f"❌ خطا در ارزیابی: {e}")

if __name__ == "__main__":
    main() 