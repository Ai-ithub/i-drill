"""
Automated Backup Service for i-Drill
"""
import os
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import tarfile
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)


class BackupService:
    """
    Service for automated system backups.
    
    Provides scheduled backup functionality for database, ML models, configuration,
    and logs. Supports automatic cleanup of old backups based on retention policy.
    
    Attributes:
        scheduler: Background scheduler for scheduled backups
        backup_dir: Directory path where backups are stored
        enabled: Boolean indicating if auto-backup is enabled
        retention_days: Number of days to retain backups
    """
    
    def __init__(self):
        """
        Initialize BackupService.
        
        Sets up backup directory and scheduler if auto-backup is enabled.
        Configuration is read from environment variables.
        """
        self.scheduler = None
        self.backup_dir = Path(os.getenv("BACKUP_DIR", "./backups"))
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = os.getenv("ENABLE_AUTO_BACKUP", "true").lower() == "true"
        self.retention_days = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))
        
        if self.enabled:
            self._setup_scheduler()
    
    def _setup_scheduler(self) -> None:
        """
        Setup scheduled backup jobs.
        
        Configures the background scheduler with cron-based backup schedule.
        Default schedule is daily at 3 AM, configurable via BACKUP_SCHEDULE env var.
        """
        self.scheduler = BackgroundScheduler()
        
        # Daily backup at 3 AM
        backup_schedule = os.getenv("BACKUP_SCHEDULE", "0 3 * * *")
        
        self.scheduler.add_job(
            self.create_backup,
            trigger=CronTrigger.from_crontab(backup_schedule),
            id='daily_backup',
            name='Daily Backup',
            replace_existing=True
        )
        
        logger.info(f"✅ Auto-backup scheduled: {backup_schedule}")
    
    def start(self) -> None:
        """
        Start the backup scheduler.
        
        Begins executing scheduled backup jobs. Does nothing if auto-backup
        is disabled or scheduler is not initialized.
        """
        if not self.enabled or not self.scheduler:
            logger.warning("Auto-backup is disabled")
            return
        
        try:
            self.scheduler.start()
            logger.info("✅ Backup scheduler started")
        except Exception as e:
            logger.error(f"Failed to start backup scheduler: {e}")
    
    def stop(self) -> None:
        """
        Stop the backup scheduler.
        
        Shuts down the scheduler gracefully, preventing any new backup jobs
        from starting. Existing jobs will complete.
        """
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Backup scheduler stopped")
    
    def create_backup(
        self,
        include_database: bool = True,
        include_models: bool = True,
        include_config: bool = True,
        include_logs: bool = False
    ) -> Dict[str, Any]:
        """
        Create a backup of the system.
        
        Creates a timestamped backup directory and backs up specified components.
        Automatically cleans up old backups after creation.
        
        Args:
            include_database: Whether to backup database (default: True)
            include_models: Whether to backup ML models (default: True)
            include_config: Whether to backup configuration files (default: True)
            include_logs: Whether to backup log files (default: False)
            
        Returns:
            Dictionary containing backup information:
            - timestamp: ISO timestamp of backup creation
            - backup_name: Name of the backup directory
            - files: List of backed up components
            - size: Total size of backup in bytes
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"i-drill-backup-{timestamp}"
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔄 Creating backup: {backup_name}")
        
        backup_info = {
            "timestamp": datetime.now().isoformat(),
            "backup_name": backup_name,
            "files": [],
            "size": 0
        }
        
        try:
            # Backup database
            if include_database:
                self._backup_database(backup_path)
                backup_info["files"].append("database")
            
            # Backup ML models
            if include_models:
                self._backup_models(backup_path)
                backup_info["files"].append("models")
            
            # Backup configuration
            if include_config:
                self._backup_config(backup_path)
                backup_info["files"].append("config")
            
            # Backup logs (optional)
            if include_logs:
                self._backup_logs(backup_path)
                backup_info["files"].append("logs")
            
            # Create backup manifest
            manifest_path = backup_path / "manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(backup_info, f, indent=2)
            
            # Create tar archive
            archive_path = self.backup_dir / f"{backup_name}.tar.gz"
            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(backup_path, arcname=backup_name)
            
            # Remove uncompressed backup
            shutil.rmtree(backup_path)
            
            # Calculate size
            backup_info["size"] = archive_path.stat().st_size
            backup_info["archive_path"] = str(archive_path)
            
            logger.info(f"✅ Backup created: {archive_path} ({backup_info['size'] / 1024 / 1024:.2f} MB)")
            
            # Cleanup old backups
            self._cleanup_old_backups()
            
            return backup_info
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            # Cleanup on failure
            if backup_path.exists():
                shutil.rmtree(backup_path)
            raise
    
    def _backup_database(self, backup_path: Path) -> None:
        """
        Backup database to backup directory.
        
        This is an internal helper method that exports the database schema
        and data. Currently supports PostgreSQL via pg_dump.
        
        Args:
            backup_path: Path to the backup directory
        """
        try:
            from database import db_manager
            
            db_backup_dir = backup_path / "database"
            db_backup_dir.mkdir(exist_ok=True)
            
            # Export database schema and data
            # This is a simplified version - adjust based on your database
            database_url = os.getenv("DATABASE_URL")
            if database_url and "postgresql" in database_url:
                # Use pg_dump for PostgreSQL
                import subprocess
                dump_file = db_backup_dir / "database.sql"
                subprocess.run(
                    ["pg_dump", database_url],
                    stdout=open(dump_file, 'w'),
                    check=True
                )
            else:
                logger.warning("Database backup not implemented for this database type")
                
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
    
    def _backup_models(self, backup_path: Path) -> None:
        """
        Backup ML models to backup directory.
        
        This is an internal helper method that copies the models directory
        to the backup location.
        
        Args:
            backup_path: Path to the backup directory
        """
        try:
            models_dir = Path("models")
            if models_dir.exists():
                backup_models_dir = backup_path / "models"
                shutil.copytree(models_dir, backup_models_dir, dirs_exist_ok=True)
        except Exception as e:
            logger.error(f"Models backup failed: {e}")
    
    def _backup_config(self, backup_path: Path) -> None:
        """
        Backup configuration files to backup directory.
        
        This is an internal helper method that backs up .env files,
        config directory, and alembic.ini.
        
        Args:
            backup_path: Path to the backup directory
        """
        try:
            config_backup_dir = backup_path / "config"
            config_backup_dir.mkdir(exist_ok=True)
            
            # Backup .env file (if exists)
            env_file = Path(".env")
            if env_file.exists():
                shutil.copy(env_file, config_backup_dir / ".env")
            
            # Backup config files
            config_dir = Path("config")
            if config_dir.exists():
                shutil.copytree(config_dir, config_backup_dir / "config", dirs_exist_ok=True)
            
            # Backup alembic.ini
            alembic_ini = Path("src/backend/alembic.ini")
            if alembic_ini.exists():
                shutil.copy(alembic_ini, config_backup_dir / "alembic.ini")
                
        except Exception as e:
            logger.error(f"Config backup failed: {e}")
    
    def _backup_logs(self, backup_path: Path) -> None:
        """
        Backup log files to backup directory.
        
        This is an internal helper method that copies log files
        to the backup location.
        
        Args:
            backup_path: Path to the backup directory
        """
        try:
            logs_backup_dir = backup_path / "logs"
            logs_backup_dir.mkdir(exist_ok=True)
            
            # Find and copy log files
            for log_file in Path(".").glob("*.log"):
                shutil.copy(log_file, logs_backup_dir / log_file.name)
                
        except Exception as e:
            logger.error(f"Logs backup failed: {e}")
    
    def _cleanup_old_backups(self) -> None:
        """
        Clean up old backups based on retention policy.
        
        This is an internal helper method that removes backup archives
        older than the retention_days threshold.
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            for backup_file in self.backup_dir.glob("i-drill-backup-*.tar.gz"):
                file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                if file_time < cutoff_date:
                    backup_file.unlink()
                    logger.info(f"Deleted old backup: {backup_file.name}")
                    
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all available backup archives.
        
        Returns:
            List of dictionaries containing backup information:
            - name: Backup file name
            - path: Full path to backup file
            - size: Size in bytes
            - created: ISO timestamp of creation
            Sorted by creation date (newest first)
        """
        backups = []
        
        for backup_file in self.backup_dir.glob("i-drill-backup-*.tar.gz"):
            stat = backup_file.stat()
            backups.append({
                "name": backup_file.name,
                "path": str(backup_file),
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        
        return sorted(backups, key=lambda x: x["created"], reverse=True)
    
    def restore_backup(self, backup_path: str) -> Dict[str, Any]:
        """
        Restore system from a backup archive.
        
        Extracts the backup archive and provides manifest information.
        Actual restoration of files may require manual intervention.
        
        Args:
            backup_path: Path to the backup archive file
            
        Returns:
            Dictionary containing:
            - success: Boolean indicating if extraction succeeded
            - manifest: Backup manifest information
            - extract_dir: Directory where backup was extracted
            
        Raises:
            FileNotFoundError: If backup file does not exist
        """
        backup_file = Path(backup_path)
        
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        logger.info(f"🔄 Restoring from backup: {backup_file.name}")
        
        # Extract backup
        extract_dir = self.backup_dir / "restore_temp"
        extract_dir.mkdir(exist_ok=True)
        
        try:
            with tarfile.open(backup_file, 'r:gz') as tar:
                tar.extractall(extract_dir)
            
            # Read manifest
            manifest_path = extract_dir / backup_file.stem / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
            else:
                manifest = {}
            
            # Restore files based on manifest
            # This is a simplified version - implement based on your needs
            logger.info("✅ Backup extracted. Manual restore may be required.")
            
            return {
                "success": True,
                "manifest": manifest,
                "extract_dir": str(extract_dir)
            }
            
        except Exception as e:
            logger.error(f"❌ Restore failed: {e}")
            raise


# Global backup service instance
backup_service = BackupService()

