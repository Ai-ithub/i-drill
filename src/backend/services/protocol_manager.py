"""
Protocol Manager Service
Manages all protocol connectors (Modbus, OPC UA, MQTT) and coordinates data flow
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import threading

from services.protocol_connectors import ModbusConnector, ModbusProtocol, OPCUAConnector, MQTTConnector
from services.protocol_adapter import ProtocolType
from services.data_bridge import data_bridge

logger = logging.getLogger(__name__)


class ProtocolManager:
    """
    Manages protocol connectors for rig data acquisition.
    
    Features:
    - Start/stop protocol connectors
    - Monitor connector status
    - Handle data flow from connectors to data bridge
    """
    
    def __init__(self):
        """Initialize protocol manager"""
        self.connectors: Dict[str, Any] = {}
        self.connector_configs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
    
    def register_modbus_connector(
        self,
        connector_id: str,
        protocol: ModbusProtocol,
        rig_id: str,
        host: Optional[str] = None,
        port: int = 502,
        unit_id: int = 1,
        baudrate: int = 9600,
        port_name: Optional[str] = None,
        register_mapping: Optional[Dict[str, Dict[str, Any]]] = None,
        reading_interval: float = 1.0
    ) -> bool:
        """
        Register and start a Modbus connector.
        
        Args:
            connector_id: Unique identifier for this connector
            protocol: Modbus protocol type (TCP or RTU)
            rig_id: Rig identifier
            host: IP address for TCP (required for TCP)
            port: Port number for TCP
            unit_id: Modbus unit/slave ID
            baudrate: Baud rate for RTU
            port_name: Serial port name for RTU
            register_mapping: Custom register mapping
            reading_interval: Reading interval in seconds
            
        Returns:
            True if successfully registered and started
        """
        try:
            with self.lock:
                if connector_id in self.connectors:
                    logger.warning(f"Connector {connector_id} already exists")
                    return False
                
                # Create callback to forward data to data bridge
                def data_callback(data: Dict[str, Any]):
                    protocol_type = ProtocolType.MODBUS_TCP if protocol == ModbusProtocol.TCP else ProtocolType.MODBUS_RTU
                    data_bridge.enqueue_protocol_data(data, protocol_type, rig_id)
                
                # Create connector
                connector = ModbusConnector(
                    protocol=protocol,
                    host=host,
                    port=port,
                    unit_id=unit_id,
                    baudrate=baudrate,
                    port_name=port_name,
                    register_mapping=register_mapping,
                    callback=data_callback
                )
                
                # Connect and start
                if connector.connect():
                    connector.start_continuous_reading(rig_id, reading_interval)
                    
                    self.connectors[connector_id] = connector
                    self.connector_configs[connector_id] = {
                        "type": "modbus",
                        "protocol": protocol.value,
                        "rig_id": rig_id,
                        "status": "running"
                    }
                    
                    logger.info(f"Modbus connector {connector_id} registered and started")
                    return True
                else:
                    logger.error(f"Failed to connect Modbus connector {connector_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error registering Modbus connector: {e}")
            return False
    
    def register_opcua_connector(
        self,
        connector_id: str,
        rig_id: str,
        endpoint_url: str,
        tag_mapping: Optional[Dict[str, str]] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        reading_interval: float = 1.0,
        use_subscription: bool = True
    ) -> bool:
        """
        Register and start an OPC UA connector.
        
        Args:
            connector_id: Unique identifier for this connector
            rig_id: Rig identifier
            endpoint_url: OPC UA server endpoint URL
            tag_mapping: Custom tag mapping
            username: Optional username for authentication
            password: Optional password for authentication
            reading_interval: Reading interval in seconds (if not using subscription)
            use_subscription: Use subscription-based updates (recommended)
            
        Returns:
            True if successfully registered and started
        """
        try:
            with self.lock:
                if connector_id in self.connectors:
                    logger.warning(f"Connector {connector_id} already exists")
                    return False
                
                # Create callback to forward data to data bridge
                def data_callback(data: Dict[str, Any]):
                    data_bridge.enqueue_protocol_data(data, ProtocolType.OPCUA, rig_id)
                
                # Create connector
                connector = OPCUAConnector(
                    endpoint_url=endpoint_url,
                    tag_mapping=tag_mapping,
                    username=username,
                    password=password,
                    callback=data_callback
                )
                
                # Connect and start
                if connector.connect():
                    if use_subscription:
                        connector.start_subscription(rig_id)
                    else:
                        connector.start_continuous_reading(rig_id, reading_interval)
                    
                    self.connectors[connector_id] = connector
                    self.connector_configs[connector_id] = {
                        "type": "opcua",
                        "rig_id": rig_id,
                        "endpoint_url": endpoint_url,
                        "status": "running"
                    }
                    
                    logger.info(f"OPC UA connector {connector_id} registered and started")
                    return True
                else:
                    logger.error(f"Failed to connect OPC UA connector {connector_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error registering OPC UA connector: {e}")
            return False
    
    def register_mqtt_connector(
        self,
        connector_id: str,
        rig_id: str,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        topics: Optional[List[str]] = None,
        qos: int = 1,
        topic_mapping: Optional[Dict[str, str]] = None,
        username: Optional[str] = None,
        password: Optional[str] = None
    ) -> bool:
        """
        Register and start an MQTT connector.
        
        Args:
            connector_id: Unique identifier for this connector
            rig_id: Rig identifier
            broker_host: MQTT broker hostname or IP
            broker_port: MQTT broker port
            topics: List of MQTT topics to subscribe to
            qos: Quality of Service level
            topic_mapping: Custom topic mapping
            username: Optional username for authentication
            password: Optional password for authentication
            
        Returns:
            True if successfully registered and started
        """
        try:
            with self.lock:
                if connector_id in self.connectors:
                    logger.warning(f"Connector {connector_id} already exists")
                    return False
                
                # Create callback to forward data to data bridge
                def data_callback(data: Dict[str, Any]):
                    # Ensure rig_id is set
                    if "rig_id" not in data or data["rig_id"] == "unknown":
                        data["rig_id"] = rig_id
                    data_bridge.enqueue_protocol_data(data, ProtocolType.MQTT, rig_id)
                
                # Create connector
                connector = MQTTConnector(
                    broker_host=broker_host,
                    broker_port=broker_port,
                    username=username,
                    password=password,
                    topics=topics,
                    qos=qos,
                    topic_mapping=topic_mapping,
                    callback=data_callback
                )
                
                # Connect and start
                if connector.connect():
                    connector.start()
                    
                    self.connectors[connector_id] = connector
                    self.connector_configs[connector_id] = {
                        "type": "mqtt",
                        "rig_id": rig_id,
                        "broker_host": broker_host,
                        "broker_port": broker_port,
                        "status": "running"
                    }
                    
                    logger.info(f"MQTT connector {connector_id} registered and started")
                    return True
                else:
                    logger.error(f"Failed to connect MQTT connector {connector_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error registering MQTT connector: {e}")
            return False
    
    def stop_connector(self, connector_id: str) -> bool:
        """
        Stop and remove a connector.
        
        Args:
            connector_id: Connector identifier
            
        Returns:
            True if successfully stopped
        """
        try:
            with self.lock:
                if connector_id not in self.connectors:
                    logger.warning(f"Connector {connector_id} not found")
                    return False
                
                connector = self.connectors[connector_id]
                
                # Stop connector based on type
                if isinstance(connector, ModbusConnector):
                    connector.stop_continuous_reading()
                    connector.disconnect()
                elif isinstance(connector, OPCUAConnector):
                    connector.stop_continuous_reading()
                    connector.disconnect()
                elif isinstance(connector, MQTTConnector):
                    connector.stop()
                
                # Remove from tracking
                del self.connectors[connector_id]
                if connector_id in self.connector_configs:
                    self.connector_configs[connector_id]["status"] = "stopped"
                
                logger.info(f"Connector {connector_id} stopped")
                return True
                
        except Exception as e:
            logger.error(f"Error stopping connector {connector_id}: {e}")
            return False
    
    def get_connector_status(self, connector_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of connector(s).
        
        Args:
            connector_id: Optional specific connector ID, or None for all
            
        Returns:
            Dictionary with connector status information
        """
        with self.lock:
            if connector_id:
                if connector_id not in self.connector_configs:
                    return {"error": f"Connector {connector_id} not found"}
                
                config = self.connector_configs[connector_id].copy()
                connector = self.connectors.get(connector_id)
                
                if connector:
                    if isinstance(connector, ModbusConnector):
                        config["connected"] = connector.connected
                        config["running"] = connector.running
                    elif isinstance(connector, OPCUAConnector):
                        config["connected"] = connector.connected
                        config["running"] = connector.running
                    elif isinstance(connector, MQTTConnector):
                        config["connected"] = connector.connected
                        config["running"] = connector.running
                
                return config
            else:
                # Return all connectors
                status = {}
                for cid, config in self.connector_configs.items():
                    connector = self.connectors.get(cid)
                    status[cid] = config.copy()
                    if connector:
                        if isinstance(connector, (ModbusConnector, OPCUAConnector, MQTTConnector)):
                            status[cid]["connected"] = connector.connected
                            status[cid]["running"] = connector.running
                
                return status
    
    def list_connectors(self) -> List[str]:
        """
        List all registered connector IDs.
        
        Returns:
            List of connector IDs
        """
        with self.lock:
            return list(self.connectors.keys())


# Global protocol manager instance
protocol_manager = ProtocolManager()

