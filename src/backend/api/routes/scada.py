"""
SCADA/PLC Connection API Routes
"""
from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import logging

from services.scada_connector_service import scada_connector_service
from api.dependencies import get_current_user, require_role
from api.models.schemas import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scada", tags=["SCADA/PLC"])


# Request/Response Models
class ModbusConnectionConfig(BaseModel):
    protocol: str = Field(..., description="Protocol type: TCP or RTU")
    host: Optional[str] = Field(None, description="IP address for TCP")
    port: int = Field(502, description="Port for TCP")
    slave_id: int = Field(1, description="Modbus slave/unit ID")
    serial_port: Optional[str] = Field(None, description="Serial port for RTU (e.g., COM3, /dev/ttyUSB0)")
    baudrate: int = Field(9600, description="Baud rate for RTU")
    timeout: float = Field(3.0, description="Connection timeout in seconds")


class OPCUAConnectionConfig(BaseModel):
    endpoint_url: str = Field(..., description="OPC UA server endpoint (e.g., opc.tcp://192.168.1.100:4840)")
    username: Optional[str] = Field(None, description="Username for authentication")
    password: Optional[str] = Field(None, description="Password for authentication")
    security_policy: str = Field("None", description="Security policy")
    security_mode: str = Field("None", description="Security mode")
    timeout: float = Field(10.0, description="Connection timeout in seconds")


class MQTTConnectionConfig(BaseModel):
    broker_host: str = Field(..., description="MQTT broker hostname or IP")
    broker_port: int = Field(1883, description="MQTT broker port")
    username: Optional[str] = Field(None, description="Username for authentication")
    password: Optional[str] = Field(None, description="Password for authentication")
    client_id: Optional[str] = Field(None, description="Client ID (auto-generated if not provided)")
    keepalive: int = Field(60, description="Keepalive interval in seconds")
    clean_session: bool = Field(True, description="Clean session flag")
    tls_enabled: bool = Field(False, description="Enable TLS encryption")
    ca_certs: Optional[str] = Field(None, description="Path to CA certificate file")
    qos: int = Field(1, description="Quality of Service level (0, 1, or 2)")


class RigConfigurationRequest(BaseModel):
    protocol: str = Field(..., description="Protocol: modbus, opcua, or mqtt")
    connection_config: Dict[str, Any] = Field(..., description="Protocol-specific connection configuration")
    parameter_mapping: Dict[str, Any] = Field(..., description="Mapping of parameter names to protocol addresses/tags")
    read_interval: float = Field(1.0, description="Interval between reads in seconds (for Modbus/OPC UA)")


@router.post("/rigs/{rig_id}/configure", response_model=Dict[str, Any])
async def configure_rig(
    rig_id: str,
    config: RigConfigurationRequest,
    current_user: User = Depends(require_role(["admin", "engineer"]))
):
    """
    Configure a rig connection to SCADA/PLC system.
    
    Supports Modbus RTU/TCP, OPC UA, and MQTT protocols.
    """
    try:
        success = scada_connector_service.configure_rig(
            rig_id=rig_id,
            protocol=config.protocol,
            connection_config=config.connection_config,
            parameter_mapping=config.parameter_mapping,
            read_interval=config.read_interval
        )
        
        if success:
            return {
                "success": True,
                "message": f"Rig {rig_id} configured successfully with {config.protocol}",
                "rig_id": rig_id,
                "protocol": config.protocol
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to configure rig {rig_id}"
            )
            
    except Exception as e:
        logger.error(f"Error configuring rig {rig_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/rigs", response_model=List[str])
async def list_rigs(
    current_user: User = Depends(get_current_user)
):
    """List all configured rig IDs."""
    try:
        return scada_connector_service.list_rigs()
    except Exception as e:
        logger.error(f"Error listing rigs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/rigs/{rig_id}/status", response_model=Dict[str, Any])
async def get_rig_status(
    rig_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get connection status for a rig."""
    try:
        status = scada_connector_service.get_rig_status(rig_id)
        if status:
            return status
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Rig {rig_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting rig status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/rigs/{rig_id}", response_model=Dict[str, Any])
async def remove_rig(
    rig_id: str,
    current_user: User = Depends(require_role(["admin", "engineer"]))
):
    """Remove a rig configuration and close connection."""
    try:
        success = scada_connector_service.remove_rig(rig_id)
        if success:
            return {
                "success": True,
                "message": f"Rig {rig_id} removed successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Rig {rig_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing rig: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/start", response_model=Dict[str, Any])
async def start_scada_connector(
    current_user: User = Depends(require_role(["admin", "engineer"]))
):
    """Start the SCADA connector service."""
    try:
        scada_connector_service.start()
        return {
            "success": True,
            "message": "SCADA connector started"
        }
    except Exception as e:
        logger.error(f"Error starting SCADA connector: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/stop", response_model=Dict[str, Any])
async def stop_scada_connector(
    current_user: User = Depends(require_role(["admin", "engineer"]))
):
    """Stop the SCADA connector service."""
    try:
        scada_connector_service.stop()
        return {
            "success": True,
            "message": "SCADA connector stopped"
        }
    except Exception as e:
        logger.error(f"Error stopping SCADA connector: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/protocols/modbus/connections", response_model=List[str])
async def list_modbus_connections(
    current_user: User = Depends(get_current_user)
):
    """List all Modbus connection IDs."""
    try:
        from services.protocols import modbus_service
        return modbus_service.list_connections()
    except Exception as e:
        logger.error(f"Error listing Modbus connections: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/protocols/opcua/connections", response_model=List[str])
async def list_opcua_connections(
    current_user: User = Depends(get_current_user)
):
    """List all OPC UA connection IDs."""
    try:
        from services.protocols import opcua_service
        return opcua_service.list_connections()
    except Exception as e:
        logger.error(f"Error listing OPC UA connections: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/protocols/mqtt/connections", response_model=List[str])
async def list_mqtt_connections(
    current_user: User = Depends(get_current_user)
):
    """List all MQTT connection IDs."""
    try:
        from services.protocols import mqtt_service
        return mqtt_service.list_connections()
    except Exception as e:
        logger.error(f"Error listing MQTT connections: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

