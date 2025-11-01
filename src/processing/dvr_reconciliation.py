from backend.database_manager import db_manager
import logging

logger = logging.getLogger(__name__)

def reconcile_data(data):
    """
    Reconciles sensor data by filling missing fields and storing to database
    """
    # Fill missing status field
    if "status" not in data:
        data["status"] = "OK"
    
    # Store to database
    try:
        success = db_manager.insert_sensor_data(data)
        if success:
            logger.info(f"Data stored successfully for RIG {data.get('Rig_ID')}")
        else:
            logger.warning(f"Failed to store data for RIG {data.get('Rig_ID')}")
    except Exception as e:
        logger.error(f"Database error during reconciliation: {e}")
    
    return data
