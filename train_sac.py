import gym
from stable_baselines3 import SAC
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# اضافه کردن مسیر پروژه به sys.path
current_dir = os.getcwd()
sys.path.append(current_dir)

try:
    import mlflow
    from src.backend.services.mlflow_service import mlflow_service
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow not available. Training will continue without logging.")

from drilling_env.drilling_env import DrillingEnv


def evaluate_model(model, env, num_episodes=10):
    """Evaluate model and return metrics"""
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if steps > 10000:  # Safety limit
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if (episode + 1) % 5 == 0:
            print(f"اپیزود {episode + 1}/{num_episodes}: پاداش = {total_reward:.2f}")
    
    return {
        "mean_reward": sum(episode_rewards) / len(episode_rewards),
        "std_reward": (sum([(r - sum(episode_rewards)/len(episode_rewards))**2 for r in episode_rewards]) / len(episode_rewards))**0.5,
        "min_reward": min(episode_rewards),
        "max_reward": max(episode_rewards),
        "mean_episode_length": sum(episode_lengths) / len(episode_lengths),
    }


def main():
    print("=== شروع آموزش SAC روی محیط حفاری ===")
    
    # Setup MLflow
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "i-drill-training")
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment(experiment_name)
            print(f"✅ MLflow experiment: {experiment_name}")
        except Exception as e:
            print(f"⚠️ MLflow setup failed: {e}")
    
    # ساخت محیط
    print("ساخت محیط DrillingEnv...")
    env = DrillingEnv()
    
    # Training parameters
    total_timesteps = int(os.getenv("TOTAL_TIMESTEPS", 100_000))
    learning_rate = float(os.getenv("LEARNING_RATE", 0.0003))
    gamma = float(os.getenv("GAMMA", 0.99))
    tau = float(os.getenv("TAU", 0.005))
    
    # Start MLflow run
    run_name = f"sac-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    if MLFLOW_AVAILABLE:
        mlflow.start_run(run_name=run_name)
        mlflow.log_params({
            "algorithm": "SAC",
            "total_timesteps": total_timesteps,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "tau": tau,
            "policy": "MlpPolicy",
        })
        mlflow.set_tags({
            "model_type": "sac",
            "env": "DrillingEnv",
            "training_type": "automated",
        })
    
    try:
        # ساخت مدل SAC
        print("ساخت مدل SAC...")
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            verbose=1
        )
        
        # آموزش مدل
        print(f"شروع آموزش مدل ({total_timesteps} timesteps)...")
        model.learn(total_timesteps=total_timesteps)
        
        # ذخیره مدل
        model_dir = Path("models/sac")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "sac_drilling_env"
        print(f"ذخیره مدل در {model_path}...")
        model.save(str(model_path))
        
        # ارزیابی مدل
        print("ارزیابی مدل آموزش‌دیده...")
        eval_metrics = evaluate_model(model, env, num_episodes=10)
        
        print(f"\n📊 نتایج ارزیابی:")
        print(f"  میانگین پاداش: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
        print(f"  پاداش حداقل: {eval_metrics['min_reward']:.2f}")
        print(f"  پاداش حداکثر: {eval_metrics['max_reward']:.2f}")
        print(f"  میانگین طول اپیزود: {eval_metrics['mean_episode_length']:.2f}")
        
        # Log metrics to MLflow
        if MLFLOW_AVAILABLE:
            mlflow.log_metrics(eval_metrics)
            
            # Log model
            try:
                mlflow.log_artifact(str(model_path) + ".zip", artifact_path="model")
                
                # Register model
                mlflow.register_model(
                    f"runs:/{mlflow.active_run().info.run_id}/model",
                    "sac_drilling_env"
                )
                print("✅ مدل در MLflow ثبت شد")
            except Exception as e:
                print(f"⚠️ Failed to register model in MLflow: {e}")
        
        # Save metrics to file
        metrics_file = model_dir / "training_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(eval_metrics, f, indent=2)
        
        print("\n=== آموزش و ارزیابی کامل شد ===")
        
    finally:
        if MLFLOW_AVAILABLE:
            mlflow.end_run()


if __name__ == "__main__":
    main() 