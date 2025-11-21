"""
Data Retention Service
Manages data retention policies: keeping real-time data (30 days), archiving old data
"""
import logging
import threading
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import os

try:
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

from config_loader import config_loader
from database_manager import db_manager

logger = logging.getLogger(__name__)


class DataRetentionService:
    """
    Service for managing data retention policies.
    
    Features:
    - Keep real-time data (30 days by default)
    - Archive old data
    - Automatic cleanup
    - Configurable retention periods
    """
    
    def __init__(self):
        """Initialize DataRetentionService."""
        self.available = SQLALCHEMY_AVAILABLE and db_manager.is_available()
        if not self.available:
            logger.warning("Data retention service not available")
            self.running = False
            self.cleanup_thread = None
            return
        
        try:
            db_config = config_loader.get_database_config()
            self.database_url = self._build_database_url(db_config)
            self.engine = None
            self._initialize_engine()
            
            # Retention configuration
            self.realtime_retention_days = int(os.getenv('REALTIME_RETENTION_DAYS', '30'))
            self.archive_retention_days = int(os.getenv('ARCHIVE_RETENTION_DAYS', '365'))
            self.cleanup_interval_hours = int(os.getenv('RETENTION_CLEANUP_INTERVAL_HOURS', '24'))
            
            self.running = False
            self.cleanup_thread: Optional[threading.Thread] = None
            
            logger.info(
                f"Data retention service initialized: "
                f"realtime={self.realtime_retention_days} days, "
                f"archive={self.archive_retention_days} days"
            )
        except Exception as e:
            logger.error(f"Error initializing data retention service: {e}")
            self.available = False
    
    def _build_database_url(self, db_config: Dict[str, Any]) -> str:
        """Build database URL from config."""
        host = db_config.get('host', 'localhost')
        port = db_config.get('port', 5432)
        database = db_config.get('database', 'drilling_db')
        username = db_config.get('username', 'drill_user')
        password = db_config.get('password', '')
        
        return f"postgresql://{username}:{password}@{host}:{port}/{database}"
    
    def _initialize_engine(self) -> None:
        """Initialize SQLAlchemy engine."""
        try:
            self.engine = create_engine(
                self.database_url,
                pool_pre_ping=True,
                echo=False
            )
        except Exception as e:
            logger.error(f"Error initializing engine: {e}")
            self.engine = None
    
    def start(self) -> None:
        """Start automatic cleanup thread."""
        if not self.available:
            return
        
        if self.running:
            logger.warning("Data retention service is already running")
            return
        
        self.running = True
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="DataRetention-Cleanup"
        )
        self.cleanup_thread.start()
        logger.info("Data retention service started")
    
    def stop(self) -> None:
        """Stop automatic cleanup."""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=10)
        logger.info("Data retention service stopped")
    
    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self.running:
            try:
                # Run cleanup
                self.cleanup_old_data()
                
                # Sleep until next cleanup
                sleep_seconds = self.cleanup_interval_hours * 3600
                for _ in range(sleep_seconds):
                    if not self.running:
                        break
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def cleanup_old_data(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Clean up old data based on retention policy.
        
        Args:
            dry_run: If True, only report what would be deleted without actually deleting
        
        Returns:
            Dictionary with cleanup statistics
        """
        if not self.engine:
            return {"success": False, "message": "Database engine not available"}
        
        try:
            cutoff_date = datetime.now() - timedelta(days=self.realtime_retention_days)
            
            with self.engine.connect() as conn:
                # Count records to be archived
                count_result = conn.execute(text("""
                    SELECT COUNT(*) FROM sensor_data 
                    WHERE timestamp < :cutoff_date;
                """), {"cutoff_date": cutoff_date})
                count_to_archive = count_result.scalar()
                
                if dry_run:
                    return {
                        "success": True,
                        "dry_run": True,
                        "records_to_archive": count_to_archive,
                        "cutoff_date": cutoff_date.isoformat(),
                        "message": f"Would archive {count_to_archive} records"
                    }
                
                # Archive old data
                if count_to_archive > 0:
                    archived = self._archive_data(conn, cutoff_date)
                else:
                    archived = 0
                
                # Delete very old archived data (beyond archive retention)
                archive_cutoff = datetime.now() - timedelta(days=self.archive_retention_days)
                deleted = self._delete_very_old_data(conn, archive_cutoff)
                
                conn.commit()
                
                logger.info(
                    f"Cleanup completed: archived={archived}, deleted={deleted}"
                )
                
                return {
                    "success": True,
                    "archived": archived,
                    "deleted": deleted,
                    "cutoff_date": cutoff_date.isoformat(),
                    "archive_cutoff_date": archive_cutoff.isoformat()
                }
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return {
                "success": False,
                "message": f"Cleanup failed: {e}",
                "error": str(e)
            }
    
    def _archive_data(self, conn, cutoff_date: datetime) -> int:
        """Archive data older than cutoff date."""
        try:
            # Create archive table if it doesn't exist
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sensor_data_archive (
                    LIKE sensor_data INCLUDING ALL
                );
            """))
            
            # Copy data to archive
            result = conn.execute(text("""
                INSERT INTO sensor_data_archive
                SELECT * FROM sensor_data
                WHERE timestamp < :cutoff_date
                ON CONFLICT DO NOTHING;
            """), {"cutoff_date": cutoff_date})
            
            archived_count = result.rowcount
            
            # Delete archived data from main table
            if archived_count > 0:
                conn.execute(text("""
                    DELETE FROM sensor_data
                    WHERE timestamp < :cutoff_date;
                """), {"cutoff_date": cutoff_date})
            
            logger.info(f"Archived {archived_count} records")
            return archived_count
        except Exception as e:
            logger.error(f"Error archiving data: {e}")
            return 0
    
    def _delete_very_old_data(self, conn, cutoff_date: datetime) -> int:
        """Delete very old archived data."""
        try:
            # Check if archive table exists
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'sensor_data_archive'
                );
            """))
            
            if not result.scalar():
                return 0
            
            # Delete very old archived data
            result = conn.execute(text("""
                DELETE FROM sensor_data_archive
                WHERE timestamp < :cutoff_date;
            """), {"cutoff_date": cutoff_date})
            
            deleted_count = result.rowcount
            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} very old archived records")
            
            return deleted_count
        except Exception as e:
            logger.error(f"Error deleting very old data: {e}")
            return 0
    
    def get_retention_stats(self) -> Dict[str, Any]:
        """Get current retention statistics."""
        if not self.engine:
            return {}
        
        try:
            with self.engine.connect() as conn:
                # Count records in main table
                main_count = conn.execute(text("SELECT COUNT(*) FROM sensor_data;")).scalar()
                
                # Count records in archive
                archive_result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'sensor_data_archive'
                    );
                """))
                archive_exists = archive_result.scalar()
                
                archive_count = 0
                if archive_exists:
                    archive_count = conn.execute(text("SELECT COUNT(*) FROM sensor_data_archive;")).scalar()
                
                # Get date ranges
                main_oldest = conn.execute(text("""
                    SELECT MIN(timestamp) FROM sensor_data;
                """)).scalar()
                
                main_newest = conn.execute(text("""
                    SELECT MAX(timestamp) FROM sensor_data;
                """)).scalar()
                
                return {
                    "main_table": {
                        "record_count": main_count,
                        "oldest_record": main_oldest.isoformat() if main_oldest else None,
                        "newest_record": main_newest.isoformat() if main_newest else None
                    },
                    "archive_table": {
                        "exists": archive_exists,
                        "record_count": archive_count
                    },
                    "retention_policy": {
                        "realtime_days": self.realtime_retention_days,
                        "archive_days": self.archive_retention_days
                    }
                }
        except Exception as e:
            logger.error(f"Error getting retention stats: {e}")
            return {}


# Global instance
data_retention_service = DataRetentionService()

