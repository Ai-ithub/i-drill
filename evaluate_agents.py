import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO, SAC
import sys
import os
from collections import defaultdict
import pandas as pd

# Add project path to sys.path
current_dir = os.getcwd()
sys.path.append(current_dir)

from drilling_env.drilling_env import DrillingEnv

class AgentEvaluator:
    def __init__(self, env):
        self.env = env
        self.results = {}
        
    def evaluate_agent(self, model, agent_name, num_episodes=10):
        """Complete evaluation of an agent"""
        print(f"\n=== Evaluating {agent_name} ===")
        
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
                
                # Store episode data
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
            
            print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Length = {steps}")
        
        # Calculate statistics
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
        """Compare performance of two agents"""
        print("\n" + "="*60)
        print("📊 Agent Performance Comparison")
        print("="*60)
        
        comparison_data = []
        for agent_name, results in self.results.items():
            comparison_data.append({
                'Agent': agent_name,
                'Mean Reward': f"{results['mean_reward']:.2f} ± {results['std_reward']:.2f}",
                'Mean Episode Length': f"{results['mean_length']:.1f} ± {results['std_length']:.1f}",
                'Mean Depth': f"{results['mean_depth']:.2f}",
                'Mean Bit Wear': f"{results['mean_bit_wear']:.6f}",
                'Mean ROP': f"{results['mean_rop']:.2f}",
                'Mean Torque': f"{results['mean_torque']:.2f}",
                'Mean Pressure': f"{results['mean_pressure']:.2f}",
                'Mean Vibration': f"{results['mean_vibration']:.4f}"
            })
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        return df
    
    def plot_performance_comparison(self):
        """Plot performance comparison graph"""
        if len(self.results) < 2:
            print("Need at least two agents for comparison")
            return
        
        # Set font
        plt.rcParams['font.family'] = 'DejaVu Sans'
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PPO and SAC Agents Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Reward
        agents = list(self.results.keys())
        rewards = [self.results[agent]['mean_reward'] for agent in agents]
        reward_stds = [self.results[agent]['std_reward'] for agent in agents]
        
        axes[0, 0].bar(agents, rewards, yerr=reward_stds, capsize=5, alpha=0.7)
        axes[0, 0].set_title('Mean Reward')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Episode Length
        lengths = [self.results[agent]['mean_length'] for agent in agents]
        length_stds = [self.results[agent]['std_length'] for agent in agents]
        
        axes[0, 1].bar(agents, lengths, yerr=length_stds, capsize=5, alpha=0.7, color='orange')
        axes[0, 1].set_title('Mean Episode Length')
        axes[0, 1].set_ylabel('Number of Steps')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Depth
        depths = [self.results[agent]['mean_depth'] for agent in agents]
        axes[0, 2].bar(agents, depths, alpha=0.7, color='green')
        axes[0, 2].set_title('Mean Drilling Depth')
        axes[0, 2].set_ylabel('Depth (meters)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Bit Wear
        bit_wears = [self.results[agent]['mean_bit_wear'] for agent in agents]
        axes[1, 0].bar(agents, bit_wears, alpha=0.7, color='red')
        axes[1, 0].set_title('Mean Bit Wear')
        axes[1, 0].set_ylabel('Bit Wear')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. ROP
        rops = [self.results[agent]['mean_rop'] for agent in agents]
        axes[1, 1].bar(agents, rops, alpha=0.7, color='purple')
        axes[1, 1].set_title('Mean Rate of Penetration (ROP)')
        axes[1, 1].set_ylabel('ROP (meters/hour)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Vibration
        vibrations = [self.results[agent]['mean_vibration'] for agent in agents]
        axes[1, 2].bar(agents, vibrations, alpha=0.7, color='brown')
        axes[1, 2].set_title('Mean Vibration')
        axes[1, 2].set_ylabel('Vibration Level')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('agent_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n📈 Performance comparison chart saved to 'agent_performance_comparison.png'.")
    
    def generate_report(self):
        """Generate complete evaluation report"""
        print("\n" + "="*60)
        print("📋 Complete Agent Evaluation Report")
        print("="*60)
        
        for agent_name, results in self.results.items():
            print(f"\n🔍 {agent_name}:")
            print(f"   📊 Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"   ⏱️  Episode Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
            print(f"   🕳️  Depth: {results['mean_depth']:.2f} meters")
            print(f"   🔧 Bit Wear: {results['mean_bit_wear']:.6f}")
            print(f"   ⚡ ROP: {results['mean_rop']:.2f} meters/hour")
            print(f"   🔄 Torque: {results['mean_torque']:.2f} N.m")
            print(f"   💨 Pressure: {results['mean_pressure']:.2f} Pa")
            print(f"   📳 Vibration: {results['mean_vibration']:.4f}")
        
        # Determine winner
        if len(self.results) >= 2:
            best_agent = max(self.results.keys(), key=lambda x: self.results[x]['mean_reward'])
            print(f"\n🏆 Winner: {best_agent}")
            print(f"   With reward: {self.results[best_agent]['mean_reward']:.2f}")

def main():
    print("=== PPO and SAC Agents Performance Evaluation ===")
    
    # Create environment
    print("Creating DrillingEnv...")
    env = DrillingEnv()
    
    # Create evaluator
    evaluator = AgentEvaluator(env)
    
    try:
        # Load PPO model
        print("Loading PPO model...")
        ppo_model = PPO.load("ppo_drilling_env")
        evaluator.evaluate_agent(ppo_model, "PPO", num_episodes=5)
        
        # Load SAC model
        print("Loading SAC model...")
        sac_model = SAC.load("sac_drilling_env")
        evaluator.evaluate_agent(sac_model, "SAC", num_episodes=5)
        
        # Compare agents
        comparison_df = evaluator.compare_agents()
        
        # Plot graph
        evaluator.plot_performance_comparison()
        
        # Generate report
        evaluator.generate_report()
        
        # Save results
        comparison_df.to_csv('agent_evaluation_results.csv', index=False, encoding='utf-8-sig')
        print("\n💾 Results saved to 'agent_evaluation_results.csv'.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Model file not found - {e}")
        print("Please train the models first.")
    except Exception as e:
        print(f"❌ Error in evaluation: {e}")

if __name__ == "__main__":
    main() 