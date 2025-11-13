"""
Automated ML Model Retraining Service
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

try:
    from services.mlflow_service import mlflow_service
    from services.training_pipeline_service import TrainingPipelineService
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow_service = None
    TrainingPipelineService = None


class MLRetrainingService:
    """Service for automated ML model retraining"""
    
    def __init__(self):
        self.scheduler = None
        self.training_service = None
        self.enabled = os.getenv("ENABLE_AUTO_RETRAINING", "false").lower() == "true"
        
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available. Auto-retraining disabled.")
            self.enabled = False
            return
        
        try:
            self.training_service = TrainingPipelineService()
            if self.enabled:
                self._setup_scheduler()
        except Exception as e:
            logger.error(f"Failed to initialize retraining service: {e}")
            self.enabled = False
    
    def _setup_scheduler(self) -> None:
        """
        Setup scheduled retraining jobs.
        
        Configures the background scheduler with cron-based retraining schedule.
        Default schedule is daily at 2 AM, configurable via RETRAINING_SCHEDULE env var.
        """
        self.scheduler = BackgroundScheduler()
        
        # Daily retraining at 2 AM
        retraining_schedule = os.getenv("RETRAINING_SCHEDULE", "0 2 * * *")  # Cron format
        
        self.scheduler.add_job(
            self.retrain_models,
            trigger=CronTrigger.from_crontab(retraining_schedule),
            id='daily_retraining',
            name='Daily Model Retraining',
            replace_existing=True
        )
        
        logger.info(f"✅ Auto-retraining scheduled: {retraining_schedule}")
    
    def start(self) -> None:
        """
        Start the retraining scheduler.
        
        Begins executing scheduled retraining jobs. Does nothing if auto-retraining
        is disabled or scheduler is not initialized.
        """
        if not self.enabled or not self.scheduler:
            logger.warning("Auto-retraining is disabled")
            return
        
        try:
            self.scheduler.start()
            logger.info("✅ ML retraining scheduler started")
        except Exception as e:
            logger.error(f"Failed to start retraining scheduler: {e}")
    
    def stop(self) -> None:
        """
        Stop the retraining scheduler.
        
        Shuts down the scheduler gracefully, preventing any new retraining jobs
        from starting. Existing jobs will complete.
        """
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("ML retraining scheduler stopped")
    
    def retrain_models(self) -> None:
        """
        Retrain all configured ML models.
        
        Triggers retraining for all models in the retraining list:
        - rul_lstm (LSTM model)
        - rul_transformer (Transformer model)
        - rul_cnn_lstm (CNN-LSTM model)
        
        Logs results for each model retraining attempt.
        """
        if not self.training_service:
            logger.error("Training service not available")
            return
        
        logger.info("🔄 Starting automated model retraining...")
        
        models_to_retrain = [
            {"model_name": "rul_lstm", "model_type": "lstm"},
            {"model_name": "rul_transformer", "model_type": "transformer"},
            {"model_name": "rul_cnn_lstm", "model_type": "cnn_lstm"},
        ]
        
        results = []
        for model_config in models_to_retrain:
            try:
                result = self.training_service.trigger_training(
                    model_name=model_config["model_name"],
                    experiment_name="auto-retraining",
                    params={
                        "model_type": model_config["model_type"],
                        "retraining_date": datetime.now().isoformat()
                    }
                )
                results.append({
                    "model": model_config["model_name"],
                    "success": result.get("success", False),
                    "run_id": result.get("run_id")
                })
            except Exception as e:
                logger.error(f"Failed to retrain {model_config['model_name']}: {e}")
                results.append({
                    "model": model_config["model_name"],
                    "success": False,
                    "error": str(e)
                })
        
        logger.info(f"✅ Retraining completed. Results: {results}")
        return results
    
    def retrain_model_on_demand(self, model_name: str) -> Dict[str, Any]:
        """
        Retrain a specific model on demand.
        
        Triggers immediate retraining of a single model by name.
        
        Args:
            model_name: Name of the model to retrain
            
        Returns:
            Dictionary containing:
            - success: Boolean indicating if retraining succeeded
            - run_id: MLflow run ID if successful
            - message: Status message
        """
        """Manually trigger retraining for a specific model"""
        if not self.training_service:
            return {"success": False, "message": "Training service not available"}
        
        try:
            result = self.training_service.trigger_training(
                model_name=model_name,
                experiment_name="manual-retraining",
                params={"triggered_by": "manual", "date": datetime.now().isoformat()}
            )
            return result
        except Exception as e:
            logger.error(f"Manual retraining failed for {model_name}: {e}")
            return {"success": False, "message": str(e)}


# Global retraining service instance
ml_retraining_service = MLRetrainingService()

