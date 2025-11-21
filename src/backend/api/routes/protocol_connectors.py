"""
Protocol Connectors API Routes
For managing protocol connections (Modbus, OPC UA, MQTT)
"""
from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any, List, Optional
import logging

from services.protocols import modbus_service, opcua_service, mqtt_service
from api.dependencies import get_current_user, require_role
from api.models.schemas import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/protocols", tags=["Protocol Connectors"])


@router.get("/modbus/connections", response_model=List[str])
async def list_modbus_connections(
    current_user: User = Depends(get_current_user)
):
    """List all Modbus connection IDs."""
    try:
        return modbus_service.list_connections()
    except Exception as e:
        logger.error(f"Error listing Modbus connections: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/modbus/connections/{connection_id}/status", response_model=Dict[str, Any])
async def get_modbus_status(
    connection_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get Modbus connection status."""
    try:
        status_info = modbus_service.get_connection_status(connection_id)
        if status_info:
            return status_info
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Modbus connection {connection_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Modbus status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/opcua/connections", response_model=List[str])
async def list_opcua_connections(
    current_user: User = Depends(get_current_user)
):
    """List all OPC UA connection IDs."""
    try:
        return opcua_service.list_connections()
    except Exception as e:
        logger.error(f"Error listing OPC UA connections: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/opcua/connections/{connection_id}/status", response_model=Dict[str, Any])
async def get_opcua_status(
    connection_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get OPC UA connection status."""
    try:
        status_info = opcua_service.get_connection_status(connection_id)
        if status_info:
            return status_info
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"OPC UA connection {connection_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting OPC UA status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/mqtt/connections", response_model=List[str])
async def list_mqtt_connections(
    current_user: User = Depends(get_current_user)
):
    """List all MQTT connection IDs."""
    try:
        return mqtt_service.list_connections()
    except Exception as e:
        logger.error(f"Error listing MQTT connections: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/mqtt/connections/{connection_id}/status", response_model=Dict[str, Any])
async def get_mqtt_status(
    connection_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get MQTT connection status."""
    try:
        status_info = mqtt_service.get_connection_status(connection_id)
        if status_info:
            return status_info
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MQTT connection {connection_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting MQTT status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
