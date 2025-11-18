"""
Sensor Data API Routes

This module provides REST API endpoints and WebSocket support for sensor data operations.

Endpoints:
- GET /realtime: Retrieve real-time sensor data
- GET /historical: Retrieve historical sensor data with filtering
- GET /aggregated: Get aggregated sensor data
- GET /analytics/{rig_id}: Get analytics summary for a rig
- POST /: Create new sensor data record
- WebSocket /ws/{rig_id}: Real-time sensor data streaming

All endpoints support:
- Pagination (limit, offset)
- Filtering by rig_id
- Time range filtering (for historical data)
- Error handling and validation

Example:
    >>> # Get real-time data
    >>> GET /api/v1/sensor-data/realtime?rig_id=RIG_01&limit=10
    >>> 
    >>> # Get historical data
    >>> GET /api/v1/sensor-data/historical?start_time=2024-01-01&end_time=2024-01-02
    >>> 
    >>> # Create sensor data
    >>> POST /api/v1/sensor-data/
    >>> {
    ...     "rig_id": "RIG_01",
    ...     "timestamp": "2024-01-01T00:00:00",
    ...     "depth": 5000.0,
    ...     "wob": 15000.0
    ... }
"""
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, HTTPException
from typing import Optional, List
from datetime import datetime, timedelta
from api.models.schemas import (
    SensorDataResponse,
    HistoricalDataQuery,
    SensorDataPoint,
    WebSocketMessage
)
from services.data_service import DataService
from services.kafka_service import kafka_service
from services.websocket_manager import websocket_manager
from utils.validators import validate_rig_id
import logging
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sensor-data", tags=["sensor-data"])
data_service = DataService()


@router.get(
    "/realtime",
    response_model=SensorDataResponse,
    summary="Get real-time sensor data",
    description="""
    Retrieve the latest real-time sensor data from the database.
    
    This endpoint returns the most recent sensor readings, optionally filtered by rig ID.
    Data is sorted by timestamp in descending order (newest first).
    
    **Use Cases:**
    - Dashboard real-time monitoring
    - Latest sensor readings display
    - Quick status checks
    
    **Rate Limits:** 100 requests per minute
    
    **Example Request:**
    ```bash
    GET /api/v1/sensor-data/realtime?rig_id=RIG_01&limit=10
    ```
    
    **Example Response:**
    ```json
    {
      "success": true,
      "count": 10,
      "data": [
        {
          "rig_id": "RIG_01",
          "timestamp": "2024-01-01T12:00:00",
          "depth": 5000.0,
          "wob": 15000.0,
          "rpm": 100.0
        }
      ]
    }
    ```
    """,
    responses={
        200: {
            "description": "Successfully retrieved sensor data",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "count": 10,
                        "data": [
                            {
                                "rig_id": "RIG_01",
                                "timestamp": "2025-01-15T10:30:00Z",
                                "depth": 5000.0,
                                "wob": 1500.0,
                                "rpm": 80.0,
                                "torque": 400.0
                            }
                        ]
                    }
                }
            }
        },
        400: {"description": "Invalid request parameters"},
        500: {"description": "Internal server error"}
    }
)
async def get_realtime_data(
    rig_id: Optional[str] = Query(None, description="Filter by rig ID (e.g., 'RIG_01')", examples=["RIG_01"]),
    limit: int = Query(100, le=10000, ge=1, description="Number of records to return (max 10000)", examples=[100])
):
    """
    Get latest real-time sensor data
    
    Returns the most recent sensor readings from the database.
    """
    try:
        data = data_service.get_latest_sensor_data(rig_id=rig_id, limit=limit)
        
        # Convert to Pydantic models
        sensor_points = [SensorDataPoint(**record) for record in data]
        
        return SensorDataResponse(
            success=True,
            count=len(sensor_points),
            data=sensor_points
        )
    except Exception as e:
        logger.error(f"Error getting realtime data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/historical", response_model=SensorDataResponse)
async def get_historical_data(
    rig_id: Optional[str] = Query(None, description="Filter by rig ID"),
    start_time: datetime = Query(..., description="Start time for query"),
    end_time: datetime = Query(..., description="End time for query"),
    parameters: Optional[str] = Query(None, description="Comma-separated list of parameters to include"),
    limit: int = Query(1000, le=10000, ge=1, description="Number of records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    Get historical sensor data
    
    Query historical sensor data with flexible filtering options.
    """
    try:
        # Parse parameters
        param_list = None
        if parameters:
            param_list = [p.strip() for p in parameters.split(',')]
        
        # Validate time range
        if start_time >= end_time:
            raise HTTPException(status_code=400, detail="start_time must be before end_time")
        
        data = data_service.get_historical_data(
            rig_id=rig_id,
            start_time=start_time,
            end_time=end_time,
            parameters=param_list,
            limit=limit,
            offset=offset
        )
        
        # Convert to Pydantic models
        sensor_points = [SensorDataPoint(**record) for record in data]
        
        return SensorDataResponse(
            success=True,
            count=len(sensor_points),
            data=sensor_points
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/aggregated")
async def get_aggregated_data(
    rig_id: str = Query(..., description="Rig ID"),
    time_bucket_seconds: int = Query(60, ge=1, le=3600, description="Time bucket size in seconds"),
    start_time: Optional[datetime] = Query(None, description="Start time (defaults to 24 hours ago)"),
    end_time: Optional[datetime] = Query(None, description="End time (defaults to now)")
):
    """
    Get aggregated time series data
    
    Returns aggregated sensor data over time buckets for visualization.
    """
    try:
        data = data_service.get_time_series_aggregated(
            rig_id=rig_id,
            time_bucket_seconds=time_bucket_seconds,
            start_time=start_time,
            end_time=end_time
        )
        
        return {
            "success": True,
            "count": len(data),
            "data": data
        }
    except Exception as e:
        logger.error(f"Error getting aggregated data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/{rig_id}")
async def get_analytics_summary(rig_id: str):
    """
    Get analytics summary for a specific rig
    
    Returns drilling statistics and summary metrics.
    """
    try:
        summary = data_service.get_analytics_summary(rig_id)
        
        if summary is None:
            raise HTTPException(status_code=404, detail=f"No data found for rig {rig_id}")
        
        return {
            "success": True,
            "summary": summary
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analytics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/")
async def create_sensor_data(data: SensorDataPoint):
    """
    Create a new sensor data record
    
    Inserts a new sensor reading into the database.
    """
    try:
        success = data_service.insert_sensor_data(data.dict())
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to insert sensor data")
        
        return {
            "success": True,
            "message": "Sensor data created successfully",
            "data": data
        }
    except Exception as e:
        logger.error(f"Error creating sensor data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/{rig_id}")
async def websocket_sensor_data(websocket: WebSocket, rig_id: str):
    """
    WebSocket endpoint for real-time sensor data streaming
    
    Establishes a WebSocket connection and streams real-time data from Kafka.
    
    Authentication:
    - Token can be provided via query parameter: ?token=<access_token>
    - Token can be provided via cookie (httpOnly cookie is automatically sent)
    - Connection will be rejected if authentication fails
    
    Rate Limiting:
    - Maximum connections per user: 5 (configurable via WS_MAX_CONNECTIONS_PER_USER)
    - Maximum connections per IP: 10 (configurable via WS_MAX_CONNECTIONS_PER_IP)
    - Maximum messages per minute: 100 (configurable via WS_MAX_MESSAGES_PER_MINUTE)
    
    Args:
        websocket: WebSocket connection instance
        rig_id: Rig identifier for the data stream
    """
    from api.dependencies import authenticate_websocket
    from utils.websocket_rate_limiter import websocket_rate_limiter
    from utils.security_logging import log_security_event
    
    # Authenticate WebSocket connection
    user = await authenticate_websocket(websocket)
    if not user:
        # authenticate_websocket already closes the connection on failure
        return
    
    # Check rate limiting before accepting connection
    is_allowed, reason = websocket_rate_limiter.check_connection_allowed(
        websocket, user_id=user.id
    )
    if not is_allowed:
        await websocket.close(code=1008, reason=reason or "Rate limit exceeded")
        log_security_event(
            event_type="websocket_rate_limit",
            severity="warning",
            message=f"WebSocket connection rate limited for user {user.username}",
            user_id=user.id,
            details={"reason": reason, "rig_id": rig_id}
        )
        logger.warning(f"WebSocket connection rate limited: {reason} (user: {user.username})")
        return
    
    # Validate rig_id format to prevent injection attacks
    if not validate_rig_id(rig_id):
        await websocket.close(code=1008, reason="Invalid rig_id format")
        logger.warning(f"WebSocket connection rejected: invalid rig_id format: {rig_id} (user: {user.username})")
        return
    
    # Register connection with rate limiter
    websocket_rate_limiter.register_connection(websocket, user_id=user.id)
    
    # Connect WebSocket with user authentication
    await websocket_manager.connect(websocket, rig_id, user_id=user.id)
    logger.info(f"WebSocket connection established for rig {rig_id} (user: {user.username}, user_id: {user.id})")
    
    consumer_id = None
    
    try:
        # Create Kafka consumer for this connection
        consumer_id = f"ws_{rig_id}_{id(websocket)}"
        topic = kafka_service.kafka_config.get('topics', {}).get('sensor_stream', 'rig.sensor.stream')
        
        if not kafka_service.create_consumer(consumer_id, topic):
            await websocket.send_json({
                "message_type": "error",
                "data": {
                    "error": "Failed to connect to Kafka stream",
                    "timestamp": datetime.now().isoformat()
                }
            })
            await websocket.close()
            return
        
        # Send initial connection message
        await websocket.send_json({
            "message_type": "status_update",
            "data": {
                "status": "connected",
                "rig_id": rig_id,
                "timestamp": datetime.now().isoformat()
            },
            "timestamp": datetime.now().isoformat()
        })
        
        # Start ping task
        ping_task = asyncio.create_task(_ping_client(websocket))
        
        # Stream data
        while True:
            # Check for client messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                if data == "ping":
                    await websocket.send_json({
                        "message_type": "status_update",
                        "data": {"status": "alive"},
                        "timestamp": datetime.now().isoformat()
                    })
                elif data == "close":
                    break
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                logger.debug(f"Error receiving client message: {e}")
            
            # Consume from Kafka (non-blocking)
            kafka_data = kafka_service.consume_messages(consumer_id, timeout=0.1)
            
            if kafka_data:
                # Filter by rig_id if needed
                if kafka_data.get('rig_id') == rig_id or not rig_id:
                    message = WebSocketMessage(
                        message_type="sensor_data",
                        data=kafka_data,
                        timestamp=datetime.now()
                    )
                    await websocket.send_json(message.dict())
            
            # Small delay to prevent CPU spinning
            await asyncio.sleep(0.05)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for rig {rig_id}")
    except Exception as e:
        logger.error(f"WebSocket error for rig {rig_id}: {e}")
        try:
            await websocket.send_json({
                "message_type": "error",
                "data": {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                },
                "timestamp": datetime.now().isoformat()
            })
        except:
            pass
    finally:
        # Cancel ping task
        if 'ping_task' in locals():
            ping_task.cancel()
        
        # Unregister from rate limiter
        from utils.websocket_rate_limiter import websocket_rate_limiter
        websocket_rate_limiter.unregister_connection(websocket)
        
        # Cleanup Kafka consumer
        if consumer_id:
            kafka_service.close_consumer(consumer_id)
        
        # Remove from manager
        websocket_manager.disconnect(websocket)
        logger.info(f"WebSocket cleanup completed for rig {rig_id}")


async def _ping_client(websocket: WebSocket):
    """Send periodic ping to keep connection alive"""
    try:
        while True:
            await asyncio.sleep(30)  # Ping every 30 seconds
            await websocket.send_json({
                "message_type": "status_update",
                "data": {"status": "ping"},
                "timestamp": datetime.now().isoformat()
            })
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.debug(f"Ping task error: {e}")

