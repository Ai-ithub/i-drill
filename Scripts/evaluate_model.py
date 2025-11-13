#!/usr/bin/env python3
"""
Script to evaluate trained models
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import mlflow
    import numpy as np
    from src.backend.services.mlflow_service import mlflow_service
except ImportError as e:
    print(f"⚠️ Required packages not available: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_rl_model(model_path: str, env, num_episodes: int = 10) -> Dict[str, float]:
    """Evaluate a reinforcement learning model"""
    try:
        from stable_baselines3 import PPO, SAC
        import gym
    except ImportError:
        logger.error("stable_baselines3 not available")
        return {}
    
    # Determine model type from path
    if "ppo" in str(model_path).lower():
        ModelClass = PPO
    elif "sac" in str(model_path).lower():
        ModelClass = SAC
    else:
        logger.error("Unknown RL model type")
        return {}
    
    # Load model
    try:
        model = ModelClass.load(str(model_path))
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {}
    
    # Evaluate
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if steps > 10000:  # Safety limit
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    
    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "evaluation_episodes": num_episodes,
    }


def evaluate_supervised_model(model_path: str, test_data, model_type: str) -> Dict[str, float]:
    """Evaluate a supervised learning model"""
    try:
        import torch
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    except ImportError:
        logger.error("Required packages not available")
        return {}
    
    # This is a placeholder - implement based on your model structure
    logger.warning("Supervised model evaluation not fully implemented")
    
    return {
        "mse": 0.0,
        "mae": 0.0,
        "r2": 0.0,
        "rmse": 0.0,
    }


def save_evaluation_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to file"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Evaluation results saved to {output_file}")


def log_evaluation_to_mlflow(
    model_name: str,
    metrics: Dict[str, float],
    experiment_name: str
):
    """Log evaluation metrics to MLflow"""
    if not mlflow_service or mlflow_service.client is None:
        logger.warning("MLflow service not available, skipping logging")
        return
    
    try:
        mlflow.set_experiment(experiment_name)
        
        # Find the latest run for this model
        client = mlflow_service.client
        runs = client.search_runs(
            experiment_ids=[mlflow.get_experiment_by_name(experiment_name).experiment_id],
            filter_string=f"tags.model_name='{model_name}'",
            max_results=1
        )
        
        if runs:
            run_id = runs[0].info.run_id
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(metrics, step=0)
                mlflow.set_tags({"evaluation_completed": "true"})
                logger.info(f"✅ Logged evaluation metrics to MLflow run: {run_id}")
        else:
            logger.warning(f"No MLflow run found for model: {model_name}")
    
    except Exception as e:
        logger.error(f"Failed to log to MLflow: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--model_dir", required=True, help="Directory containing models")
    parser.add_argument("--model_type", help="Model type (auto-detected if not provided)")
    parser.add_argument("--experiment", default="i-drill-training", help="MLflow experiment name")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--output_dir", default="./evaluation_results", help="Output directory")
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        logger.error(f"Model directory does not exist: {model_dir}")
        sys.exit(1)
    
    # Find models
    model_files = list(model_dir.glob("**/*.zip")) + list(model_dir.glob("**/*.pkl"))
    
    if not model_files:
        logger.error("No model files found")
        sys.exit(1)
    
    all_results = {}
    
    for model_file in model_files:
        logger.info(f"Evaluating model: {model_file}")
        
        # Detect model type
        model_type = args.model_type
        if not model_type:
            if "ppo" in str(model_file).lower():
                model_type = "ppo"
            elif "sac" in str(model_file).lower():
                model_type = "sac"
            elif "lstm" in str(model_file).lower():
                model_type = "lstm"
            else:
                model_type = "unknown"
        
        # Evaluate based on type
        if model_type in ["ppo", "sac"]:
            try:
                from src.drilling_env.drilling_env import DrillingEnv
                env = DrillingEnv()
                metrics = evaluate_rl_model(str(model_file), env, args.num_episodes)
            except Exception as e:
                logger.error(f"Failed to evaluate RL model: {e}")
                continue
        else:
            metrics = evaluate_supervised_model(str(model_file), None, model_type)
        
        model_name = model_file.stem
        all_results[model_name] = {
            "model_path": str(model_file),
            "model_type": model_type,
            "metrics": metrics,
        }
        
        # Log to MLflow
        log_evaluation_to_mlflow(model_name, metrics, args.experiment)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_evaluation_results(all_results, str(output_dir / "evaluation_results.json"))
    
    logger.info("✅ Evaluation completed")


if __name__ == "__main__":
    main()

