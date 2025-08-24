import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO, SAC
import sys
import os
from itertools import product
import json

# اضافه کردن مسیر پروژه به sys.path
current_dir = os.getcwd()
sys.path.append(current_dir)

from drilling_env.drilling_env import DrillingEnv

class HyperparameterTuner:
    def __init__(self, env):
        self.env = env
        self.results = []
        
    def evaluate_hyperparameters(self, model_class, hyperparams, agent_name, num_episodes=3):
        """ارزیابی یک مجموعه هایپرپارامتر"""
        print(f"\n--- تست {agent_name} با هایپرپارامترهای جدید ---")
        
        # ساخت مدل با هایپرپارامترهای جدید
        model = model_class("MlpPolicy", self.env, verbose=0, **hyperparams)
        
        # آموزش کوتاه مدت
        model.learn(total_timesteps=50_000)
        
        # ارزیابی
        episode_rewards = []
        for episode in range(num_episodes):
            obs = self.env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        result = {
            'agent': agent_name,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'hyperparams': hyperparams.copy()
        }
        
        self.results.append(result)
        
        print(f"پاداش: {mean_reward:.2f} ± {std_reward:.2f}")
        return result
    
    def tune_ppo(self):
        """بهینه‌سازی هایپرپارامترهای PPO"""
        print("\n=== بهینه‌سازی هایپرپارامترهای PPO ===")
        
        # تعریف هایپرپارامترهای مختلف
        learning_rates = [0.0001, 0.0003, 0.001]
        gammas = [0.9, 0.95, 0.99]
        clip_ranges = [0.1, 0.2, 0.3]
        
        for lr, gamma, clip_range in product(learning_rates, gammas, clip_ranges):
            hyperparams = {
                'learning_rate': lr,
                'gamma': gamma,
                'clip_range': clip_range,
                'ent_coef': 0.01
            }
            
            self.evaluate_hyperparameters(PPO, hyperparams, "PPO")
    
    def tune_sac(self):
        """بهینه‌سازی هایپرپارامترهای SAC"""
        print("\n=== بهینه‌سازی هایپرپارامترهای SAC ===")
        
        # تعریف هایپرپارامترهای مختلف
        learning_rates = [0.0001, 0.0003, 0.001]
        gammas = [0.9, 0.95, 0.99]
        tau_values = [0.005, 0.01, 0.02]
        
        for lr, gamma, tau in product(learning_rates, gammas, tau_values):
            hyperparams = {
                'learning_rate': lr,
                'gamma': gamma,
                'tau': tau,
                'ent_coef': 'auto'
            }
            
            self.evaluate_hyperparameters(SAC, hyperparams, "SAC")
    
    def find_best_hyperparameters(self):
        """یافتن بهترین هایپرپارامترها"""
        print("\n" + "="*60)
        print("🏆 بهترین هایپرپارامترها")
        print("="*60)
        
        # بهترین PPO
        ppo_results = [r for r in self.results if r['agent'] == 'PPO']
        if ppo_results:
            best_ppo = max(ppo_results, key=lambda x: x['mean_reward'])
            print(f"\n🥇 بهترین PPO:")
            print(f"   پاداش: {best_ppo['mean_reward']:.2f} ± {best_ppo['std_reward']:.2f}")
            print(f"   هایپرپارامترها: {best_ppo['hyperparams']}")
        
        # بهترین SAC
        sac_results = [r for r in self.results if r['agent'] == 'SAC']
        if sac_results:
            best_sac = max(sac_results, key=lambda x: x['mean_reward'])
            print(f"\n🥈 بهترین SAC:")
            print(f"   پاداش: {best_sac['mean_reward']:.2f} ± {best_sac['std_reward']:.2f}")
            print(f"   هایپرپارامترها: {best_sac['hyperparams']}")
        
        # بهترین کلی
        best_overall = max(self.results, key=lambda x: x['mean_reward'])
        print(f"\n🏆 بهترین کلی: {best_overall['agent']}")
        print(f"   پاداش: {best_overall['mean_reward']:.2f} ± {best_overall['std_reward']:.2f}")
        
        return best_ppo, best_sac, best_overall
    
    def plot_results(self):
        """رسم نمودار نتایج"""
        if not self.results:
            print("هیچ نتیجه‌ای برای رسم وجود ندارد")
            return
        
        # جداسازی نتایج PPO و SAC
        ppo_results = [r for r in self.results if r['agent'] == 'PPO']
        sac_results = [r for r in self.results if r['agent'] == 'SAC']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # نمودار PPO
        if ppo_results:
            ppo_rewards = [r['mean_reward'] for r in ppo_results]
            ppo_stds = [r['std_reward'] for r in ppo_results]
            ppo_labels = [f"LR={r['hyperparams']['learning_rate']}, γ={r['hyperparams']['gamma']}, CR={r['hyperparams']['clip_range']}" 
                         for r in ppo_results]
            
            axes[0].bar(range(len(ppo_rewards)), ppo_rewards, yerr=ppo_stds, capsize=5, alpha=0.7)
            axes[0].set_title('نتایج PPO')
            axes[0].set_ylabel('پاداش')
            axes[0].set_xticks(range(len(ppo_rewards)))
            axes[0].set_xticklabels(ppo_labels, rotation=45, ha='right')
            axes[0].grid(True, alpha=0.3)
        
        # نمودار SAC
        if sac_results:
            sac_rewards = [r['mean_reward'] for r in sac_results]
            sac_stds = [r['std_reward'] for r in sac_results]
            sac_labels = [f"LR={r['hyperparams']['learning_rate']}, γ={r['hyperparams']['gamma']}, τ={r['hyperparams']['tau']}" 
                         for r in sac_results]
            
            axes[1].bar(range(len(sac_rewards)), sac_rewards, yerr=sac_stds, capsize=5, alpha=0.7, color='orange')
            axes[1].set_title('نتایج SAC')
            axes[1].set_ylabel('پاداش')
            axes[1].set_xticks(range(len(sac_rewards)))
            axes[1].set_xticklabels(sac_labels, rotation=45, ha='right')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('hyperparameter_tuning_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n📈 نمودار نتایج در فایل 'hyperparameter_tuning_results.png' ذخیره شد.")
    
    def save_results(self):
        """ذخیره نتایج"""
        # تبدیل به DataFrame
        df_data = []
        for result in self.results:
            row = {
                'agent': result['agent'],
                'mean_reward': result['mean_reward'],
                'std_reward': result['std_reward']
            }
            # اضافه کردن هایپرپارامترها
            for key, value in result['hyperparams'].items():
                row[f'hyperparam_{key}'] = value
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv('hyperparameter_tuning_results.csv', index=False, encoding='utf-8-sig')
        
        # ذخیره بهترین هایپرپارامترها
        best_ppo, best_sac, best_overall = self.find_best_hyperparameters()
        
        best_configs = {
            'best_ppo': best_ppo['hyperparams'] if best_ppo else None,
            'best_sac': best_sac['hyperparams'] if best_sac else None,
            'best_overall': {
                'agent': best_overall['agent'],
                'hyperparams': best_overall['hyperparams'],
                'mean_reward': best_overall['mean_reward']
            }
        }
        
        with open('best_hyperparameters.json', 'w', encoding='utf-8') as f:
            json.dump(best_configs, f, indent=2, ensure_ascii=False)
        
        print("\n💾 نتایج در فایل‌های زیر ذخیره شد:")
        print("   - hyperparameter_tuning_results.csv")
        print("   - best_hyperparameters.json")

def main():
    print("=== بهینه‌سازی هایپرپارامترهای عوامل ===")
    
    # ساخت محیط
    print("ساخت محیط DrillingEnv...")
    env = DrillingEnv()
    
    # ساخت tuner
    tuner = HyperparameterTuner(env)
    
    # بهینه‌سازی PPO
    tuner.tune_ppo()
    
    # بهینه‌سازی SAC
    tuner.tune_sac()
    
    # یافتن بهترین ها
    tuner.find_best_hyperparameters()
    
    # رسم نمودار
    tuner.plot_results()
    
    # ذخیره نتایج
    tuner.save_results()
    
    print("\n✅ بهینه‌سازی هایپرپارامترها کامل شد!")

if __name__ == "__main__":
    main() 