"""
SCADA Connector Service - Main service for connecting to rig control systems
Integrates Modbus, OPC UA, and MQTT protocols
"""
import logging
import threading
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from services.protocols import modbus_service, opcua_service, mqtt_service
from services.kafka_service import kafka_service

logger = logging.getLogger(__name__)


class SCADAConnectorService:
    """
    Main service for connecting to rig SCADA/PLC systems.
    
    This service provides a unified interface for:
    - Reading sensor data from multiple protocols (Modbus, OPC UA, MQTT)
    - Mapping rig parameters to protocol-specific addresses/tags
    - Publishing data to Kafka for processing
    - Managing connections and automatic reconnection
    """
    
    def __init__(self):
        """Initialize SCADAConnectorService."""
        self.running = False
        self.read_thread: Optional[threading.Thread] = None
        self.read_interval = 1.0  # Read every 1 second by default
        self.rig_configs: Dict[str, Dict[str, Any]] = {}  # Configuration per rig
        
        # Parameter mappings for standard drilling parameters
        self.standard_parameters = [
            "WOB", "RPM", "Torque", "ROP", "Mud_Flow_Rate",
            "Standpipe_Pressure", "Casing_Pressure", "Hook_Load",
            "Block_Position", "Pump_Status", "Power_Consumption",
            "Temperature_Bit", "Temperature_Motor", "Temperature_Surface"
        ]
        
        logger.info("SCADAConnectorService initialized")
    
    def configure_rig(
        self,
        rig_id: str,
        protocol: str,
        connection_config: Dict[str, Any],
        parameter_mapping: Dict[str, Any],
        read_interval: float = 1.0
    ) -> bool:
        """
        Configure a rig connection.
        
        Args:
            rig_id: Rig identifier
            protocol: "modbus", "opcua", or "mqtt"
            connection_config: Protocol-specific connection configuration
            parameter_mapping: Mapping of parameter names to protocol addresses/tags
            read_interval: Interval between reads in seconds
        
        Returns:
            True if configuration was successful, False otherwise
        """
        try:
            connection_id = f"{rig_id}_{protocol}"
            
            # Configure protocol connection
            if protocol.lower() == "modbus":
                success = modbus_service.add_connection(
                    connection_id=connection_id,
                    protocol=connection_config.get("protocol", "TCP"),
                    host=connection_config.get("host"),
                    port=connection_config.get("port", 502),
                    slave_id=connection_config.get("slave_id", 1),
                    serial_port=connection_config.get("serial_port"),
                    baudrate=connection_config.get("baudrate", 9600),
                    timeout=connection_config.get("timeout", 3.0),
                    parameter_mapping=parameter_mapping
                )
            elif protocol.lower() == "opcua":
                success = opcua_service.add_connection(
                    connection_id=connection_id,
                    endpoint_url=connection_config.get("endpoint_url"),
                    username=connection_config.get("username"),
                    password=connection_config.get("password"),
                    security_policy=connection_config.get("security_policy", "None"),
                    security_mode=connection_config.get("security_mode", "None"),
                    timeout=connection_config.get("timeout", 10.0),
                    parameter_mapping=parameter_mapping
                )
            elif protocol.lower() == "mqtt":
                # For MQTT, we subscribe to topics instead of polling
                success = mqtt_service.add_connection(
                    connection_id=connection_id,
                    broker_host=connection_config.get("broker_host"),
                    broker_port=connection_config.get("broker_port", 1883),
                    username=connection_config.get("username"),
                    password=connection_config.get("password"),
                    client_id=connection_config.get("client_id"),
                    keepalive=connection_config.get("keepalive", 60),
                    clean_session=connection_config.get("clean_session", True),
                    tls_enabled=connection_config.get("tls_enabled", False),
                    ca_certs=connection_config.get("ca_certs")
                )
                
                # Set up MQTT message handler
                if success:
                    def mqtt_handler(topic: str, data: Any, conn_id: str):
                        self._handle_mqtt_message(rig_id, topic, data)
                    
                    mqtt_service.set_message_handler(connection_id, mqtt_handler)
                    
                    # Subscribe to topics from parameter mapping
                    for param_name, topic in parameter_mapping.items():
                        mqtt_service.subscribe(
                            connection_id=connection_id,
                            topic=topic,
                            qos=connection_config.get("qos", 1)
                        )
            else:
                logger.error(f"Unsupported protocol: {protocol}")
                return False
            
            if success:
                self.rig_configs[rig_id] = {
                    "protocol": protocol.lower(),
                    "connection_id": connection_id,
                    "connection_config": connection_config,
                    "parameter_mapping": parameter_mapping,
                    "read_interval": read_interval
                }
                logger.info(f"Rig {rig_id} configured with {protocol}")
                return True
            else:
                logger.error(f"Failed to configure rig {rig_id} with {protocol}")
                return False
                
        except Exception as e:
            logger.error(f"Error configuring rig {rig_id}: {e}")
            return False
    
    def start(self) -> None:
        """Start reading data from all configured rigs."""
        if self.running:
            logger.warning("SCADA connector is already running")
            return
        
        self.running = True
        self.read_thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
            name="SCADAConnector-ReadLoop"
        )
        self.read_thread.start()
        logger.info("SCADA connector started")
    
    def stop(self) -> None:
        """Stop reading data from rigs."""
        self.running = False
        if self.read_thread:
            self.read_thread.join(timeout=5)
        logger.info("SCADA connector stopped")
    
    def _read_loop(self) -> None:
        """Main loop for reading data from Modbus and OPC UA connections."""
        while self.running:
            try:
                for rig_id, config in self.rig_configs.items():
                    protocol = config["protocol"]
                    connection_id = config["connection_id"]
                    
                    # MQTT is event-driven, skip polling
                    if protocol == "mqtt":
                        continue
                    
                    try:
                        # Read all parameters
                        if protocol == "modbus":
                            data = modbus_service.read_all_parameters(connection_id)
                        elif protocol == "opcua":
                            data = opcua_service.read_all_parameters(connection_id)
                        else:
                            continue
                        
                        if data:
                            # Add metadata
                            sensor_data = {
                                "rig_id": rig_id,
                                "timestamp": datetime.now().isoformat(),
                                **data
                            }
                            
                            # Publish to Kafka
                            self._publish_to_kafka(sensor_data)
                            
                    except Exception as e:
                        logger.error(f"Error reading data from rig {rig_id}: {e}")
                
                # Sleep for shortest read interval
                min_interval = min(
                    (config.get("read_interval", 1.0) for config in self.rig_configs.values()),
                    default=1.0
                )
                time.sleep(min_interval)
                
            except Exception as e:
                logger.error(f"Error in SCADA connector read loop: {e}")
                time.sleep(1)
    
    def _handle_mqtt_message(self, rig_id: str, topic: str, data: Any) -> None:
        """Handle incoming MQTT message."""
        try:
            # Find parameter name from topic
            config = self.rig_configs.get(rig_id)
            if not config:
                return
            
            parameter_mapping = config["parameter_mapping"]
            param_name = None
            for param, mapped_topic in parameter_mapping.items():
                if mapped_topic == topic or topic.endswith(mapped_topic.split("/")[-1]):
                    param_name = param
                    break
            
            if param_name:
                sensor_data = {
                    "rig_id": rig_id,
                    "timestamp": datetime.now().isoformat(),
                    param_name: data if isinstance(data, (int, float, bool, str)) else data
                }
                
                # Publish to Kafka
                self._publish_to_kafka(sensor_data)
            else:
                # If topic doesn't match, try to extract parameter from topic
                # or use entire data as sensor data
                sensor_data = {
                    "rig_id": rig_id,
                    "timestamp": datetime.now().isoformat(),
                    "topic": topic,
                    "data": data
                }
                self._publish_to_kafka(sensor_data)
                
        except Exception as e:
            logger.error(f"Error handling MQTT message for rig {rig_id}: {e}")
    
    def _publish_to_kafka(self, sensor_data: Dict[str, Any]) -> None:
        """Publish sensor data to Kafka."""
        try:
            topic = "rig.sensor.stream"
            kafka_service.produce_sensor_data(topic, sensor_data)
        except Exception as e:
            logger.error(f"Error publishing to Kafka: {e}")
    
    def get_rig_status(self, rig_id: str) -> Optional[Dict[str, Any]]:
        """Get connection status for a rig."""
        if rig_id not in self.rig_configs:
            return None
        
        config = self.rig_configs[rig_id]
        connection_id = config["connection_id"]
        protocol = config["protocol"]
        
        status = {
            "rig_id": rig_id,
            "protocol": protocol,
            "configured": True
        }
        
        if protocol == "modbus":
            conn_status = modbus_service.get_connection_status(connection_id)
            if conn_status:
                status.update(conn_status)
        elif protocol == "opcua":
            conn_status = opcua_service.get_connection_status(connection_id)
            if conn_status:
                status.update(conn_status)
        elif protocol == "mqtt":
            conn_status = mqtt_service.get_connection_status(connection_id)
            if conn_status:
                status.update(conn_status)
        
        return status
    
    def list_rigs(self) -> List[str]:
        """List all configured rig IDs."""
        return list(self.rig_configs.keys())
    
    def remove_rig(self, rig_id: str) -> bool:
        """Remove a rig configuration and close connection."""
        if rig_id not in self.rig_configs:
            return False
        
        try:
            config = self.rig_configs[rig_id]
            connection_id = config["connection_id"]
            protocol = config["protocol"]
            
            if protocol == "modbus":
                modbus_service.remove_connection(connection_id)
            elif protocol == "opcua":
                opcua_service.remove_connection(connection_id)
            elif protocol == "mqtt":
                mqtt_service.remove_connection(connection_id)
            
            del self.rig_configs[rig_id]
            logger.info(f"Rig {rig_id} removed")
            return True
            
        except Exception as e:
            logger.error(f"Error removing rig {rig_id}: {e}")
            return False


# Global instance
scada_connector_service = SCADAConnectorService()

