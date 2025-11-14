"""
Database migration script to add performance indexes

Run this script to add composite indexes for better query performance.

Usage:
    python migrations/add_performance_indexes.py
"""
import logging
from sqlalchemy import text
from database import db_manager

logger = logging.getLogger(__name__)


def add_performance_indexes():
    """
    Add composite indexes for common query patterns
    
    These indexes significantly improve performance for:
    - Queries filtering by rig_id and ordering by timestamp
    - Queries filtering by rig_id and severity
    - Time-range queries on sensor_data
    """
    indexes = [
        # Composite index for sensor_data: rig_id + timestamp (most common query pattern)
        {
            "name": "ix_sensor_data_rig_timestamp",
            "table": "sensor_data",
            "columns": ["rig_id", "timestamp"],
            "sql": """
                CREATE INDEX IF NOT EXISTS ix_sensor_data_rig_timestamp 
                ON sensor_data(rig_id, timestamp DESC);
            """
        },
        # Index for maintenance_alerts: rig_id + severity
        {
            "name": "ix_maintenance_alerts_rig_severity",
            "table": "maintenance_alerts",
            "columns": ["rig_id", "severity"],
            "sql": """
                CREATE INDEX IF NOT EXISTS ix_maintenance_alerts_rig_severity 
                ON maintenance_alerts(rig_id, severity);
            """
        },
        # Index for maintenance_alerts: status + created_at
        {
            "name": "ix_maintenance_alerts_status_created",
            "table": "maintenance_alerts",
            "columns": ["status", "created_at"],
            "sql": """
                CREATE INDEX IF NOT EXISTS ix_maintenance_alerts_status_created 
                ON maintenance_alerts(status, created_at DESC);
            """
        },
        # Index for rul_predictions: rig_id + timestamp
        {
            "name": "ix_rul_predictions_rig_timestamp",
            "table": "rul_predictions",
            "columns": ["rig_id", "timestamp"],
            "sql": """
                CREATE INDEX IF NOT EXISTS ix_rul_predictions_rig_timestamp 
                ON rul_predictions(rig_id, timestamp DESC);
            """
        },
        # Index for anomaly_detections: rig_id + timestamp
        {
            "name": "ix_anomaly_detections_rig_timestamp",
            "table": "anomaly_detections",
            "columns": ["rig_id", "timestamp"],
            "sql": """
                CREATE INDEX IF NOT EXISTS ix_anomaly_detections_rig_timestamp 
                ON anomaly_detections(rig_id, timestamp DESC);
            """
        },
    ]
    
    if not db_manager._initialized:
        logger.error("Database not initialized. Please initialize database first.")
        return False
    
    try:
        with db_manager.session_scope() as session:
            for index_info in indexes:
                try:
                    # Check if index already exists
                    check_sql = f"""
                        SELECT COUNT(*) 
                        FROM pg_indexes 
                        WHERE indexname = '{index_info["name"]}'
                    """
                    result = session.execute(text(check_sql))
                    exists = result.scalar() > 0
                    
                    if exists:
                        logger.info(f"Index {index_info['name']} already exists. Skipping.")
                        continue
                    
                    # Create index
                    session.execute(text(index_info["sql"]))
                    session.commit()
                    logger.info(
                        f"✅ Created index {index_info['name']} on "
                        f"{index_info['table']}({', '.join(index_info['columns'])})"
                    )
                    
                except Exception as e:
                    logger.error(f"Error creating index {index_info['name']}: {e}")
                    session.rollback()
                    continue
        
        logger.info("✅ Performance indexes migration completed")
        return True
        
    except Exception as e:
        logger.error(f"Error during index migration: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize database if needed
    from database import init_database
    init_database()
    
    # Run migration
    success = add_performance_indexes()
    if success:
        print("✅ Performance indexes added successfully")
    else:
        print("❌ Failed to add performance indexes")

