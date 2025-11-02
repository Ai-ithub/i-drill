"""
Producer API Routes - Integration with Producer/Consumer
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from api.models.schemas import SensorDataPoint
from services.data_bridge import data_bridge
from services.kafka_service import kafka_service
from services.websocket_manager import websocket_manager
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/producer", tags=["producer"])


@router.post("/sensor-data")
async def produce_sensor_data(data: SensorDataPoint):
    """
    Produce sensor data to Kafka stream
    
    This endpoint allows external systems to send sensor data to the Kafka stream,
    which will then be consumed by the Data Bridge and broadcast via WebSocket.
    """
    try:
        # Convert Pydantic model to dict
        data_dict = data.dict()
        
        # Ensure timestamp is present
        if 'timestamp' not in data_dict or not data_dict['timestamp']:
            data_dict['timestamp'] = datetime.now().isoformat()
        
        # Produce to Kafka via data bridge
        success = data_bridge.produce_sensor_data(data_dict)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to produce data to Kafka")
        
        logger.info(f"Produced sensor data for rig {data_dict.get('rig_id', 'unknown')}")
        
        return {
            "success": True,
            "message": "Sensor data produced successfully",
            "rig_id": data_dict.get('rig_id'),
            "timestamp": data_dict.get('timestamp')
        }
        
    except Exception as e:
        logger.error(f"Error producing sensor data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_producer_status():
    """Get status of the data bridge and producer"""
    return {
        "data_bridge_running": data_bridge.running,
        "kafka_producer_available": kafka_service.producer is not None,
        "connected_rigs": list(websocket_manager.get_connected_rigs()) if data_bridge.running else [],
        "total_connections": websocket_manager.get_connection_count() if data_bridge.running else 0
    }

