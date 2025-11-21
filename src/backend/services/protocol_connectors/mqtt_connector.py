"""
MQTT Connector for IoT Sensors
Subscribes to MQTT topics related to rig sensors with QoS support
"""
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import json

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    logging.warning("paho-mqtt not installed. MQTT connector will not work.")

logger = logging.getLogger(__name__)


class MQTTConnector:
    """
    MQTT connector for IoT sensors on the rig.
    
    Features:
    - Subscribe to MQTT topics
    - Support for different QoS levels
    - Automatic reconnection
    - Topic-based data mapping
    """
    
    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        client_id: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        topics: Optional[List[str]] = None,
        qos: int = 1,
        topic_mapping: Optional[Dict[str, str]] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Initialize MQTT connector.
        
        Args:
            broker_host: MQTT broker hostname or IP
            broker_port: MQTT broker port (default: 1883)
            client_id: MQTT client ID (auto-generated if None)
            username: Optional username for authentication
            password: Optional password for authentication
            topics: List of MQTT topics to subscribe to
            qos: Quality of Service level (0, 1, or 2)
            topic_mapping: Mapping of MQTT topics to rig parameters
            callback: Callback function to handle received messages
        """
        if not MQTT_AVAILABLE:
            raise ImportError("paho-mqtt is required. Install with: pip install paho-mqtt")
        
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id or f"idrill_mqtt_{datetime.now().timestamp()}"
        self.username = username
        self.password = password
        self.topics = topics or []
        self.qos = qos
        self.callback = callback
        
        # Default topic mapping
        # Format: {"topic/path": "parameter_name"}
        self.topic_mapping = topic_mapping or self._get_default_mapping()
        
        # MQTT client
        self.client: Optional[mqtt.Client] = None
        self.connected = False
        self.running = False
        
        # Connection retry settings
        self.max_retries = 5
        self.retry_delay = 2.0
        self.keepalive = 60
    
    def _get_default_mapping(self) -> Dict[str, str]:
        """Get default MQTT topic mapping for rig parameters"""
        return {
            "rig/+/wob": "wob",
            "rig/+/rpm": "rpm",
            "rig/+/torque": "torque",
            "rig/+/rop": "rop",
            "rig/+/mud_flow": "mud_flow",
            "rig/+/mud_pressure": "mud_pressure",
            "rig/+/hook_load": "hook_load",
            "rig/+/depth": "depth",
            "rig/+/pump_status": "pump_status",
            "rig/+/power_consumption": "power_consumption",
            "rig/+/bit_temperature": "bit_temperature",
            "rig/+/motor_temperature": "motor_temperature",
            "rig/+/casing_pressure": "casing_pressure",
            "rig/+/block_position": "block_position",
        }
    
    def _on_connect(self, client: mqtt.Client, userdata: Any, flags: Dict, rc: int) -> None:
        """Callback for when the client receives a CONNACK response from the server"""
        if rc == 0:
            self.connected = True
            logger.info(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
            
            # Subscribe to topics
            for topic in self.topics:
                try:
                    client.subscribe(topic, self.qos)
                    logger.info(f"Subscribed to topic: {topic} with QoS {self.qos}")
                except Exception as e:
                    logger.error(f"Error subscribing to topic {topic}: {e}")
            
            # Also subscribe to mapped topics
            for topic_pattern in self.topic_mapping.keys():
                if topic_pattern not in self.topics:
                    try:
                        client.subscribe(topic_pattern, self.qos)
                        logger.info(f"Subscribed to mapped topic: {topic_pattern} with QoS {self.qos}")
                    except Exception as e:
                        logger.error(f"Error subscribing to mapped topic {topic_pattern}: {e}")
        else:
            self.connected = False
            logger.error(f"Failed to connect to MQTT broker. Return code: {rc}")
    
    def _on_disconnect(self, client: mqtt.Client, userdata: Any, rc: int) -> None:
        """Callback for when the client disconnects from the server"""
        self.connected = False
        if rc != 0:
            logger.warning(f"Unexpected MQTT disconnection. Return code: {rc}")
        else:
            logger.info("Disconnected from MQTT broker")
    
    def _on_message(self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
        """Callback for when a PUBLISH message is received from the server"""
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            
            # Try to parse as JSON
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                # If not JSON, treat as simple value
                try:
                    data = {"value": float(payload)}
                except ValueError:
                    data = {"value": payload}
            
            # Extract rig_id from topic if possible (format: rig/{rig_id}/parameter)
            rig_id = "unknown"
            topic_parts = topic.split('/')
            if len(topic_parts) >= 2 and topic_parts[0] == "rig":
                rig_id = topic_parts[1]
            
            # Find parameter name from topic mapping
            param_name = None
            for topic_pattern, param in self.topic_mapping.items():
                if self._topic_matches(topic, topic_pattern):
                    param_name = param
                    break
            
            # If no mapping found, try to extract from topic
            if not param_name and len(topic_parts) >= 3:
                param_name = topic_parts[2]
            
            # Create data dictionary
            result_data = {
                "rig_id": rig_id,
                "timestamp": datetime.now().isoformat(),
                "protocol": "mqtt",
                "topic": topic,
            }
            
            # Add parameter value
            if param_name:
                if isinstance(data, dict):
                    result_data[param_name] = data.get("value", data)
                else:
                    result_data[param_name] = data
            else:
                # If no parameter name, add all data
                if isinstance(data, dict):
                    result_data.update(data)
                else:
                    result_data["value"] = data
            
            # Call callback if provided
            if self.callback:
                try:
                    self.callback(result_data)
                except Exception as e:
                    logger.error(f"Error in MQTT callback: {e}")
            
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    def _topic_matches(self, topic: str, pattern: str) -> bool:
        """
        Check if a topic matches a pattern (supports + and # wildcards).
        
        Args:
            topic: Actual topic string
            pattern: Pattern with wildcards
            
        Returns:
            True if topic matches pattern
        """
        # Simple wildcard matching
        pattern_parts = pattern.split('/')
        topic_parts = topic.split('/')
        
        if len(pattern_parts) != len(topic_parts):
            # Check for # wildcard (matches remaining)
            if '#' in pattern:
                pattern_idx = pattern_parts.index('#')
                return pattern_parts[:pattern_idx] == topic_parts[:pattern_idx]
            return False
        
        for p, t in zip(pattern_parts, topic_parts):
            if p == '+':
                continue
            elif p == '#':
                return True
            elif p != t:
                return False
        
        return True
    
    def connect(self) -> bool:
        """
        Connect to MQTT broker.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.connected:
            logger.warning("Already connected to MQTT broker")
            return True
        
        try:
            self.client = mqtt.Client(client_id=self.client_id)
            
            # Set callbacks
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            
            # Set authentication if provided
            if self.username and self.password:
                self.client.username_pw_set(self.username, self.password)
            
            # Connect
            self.client.connect(self.broker_host, self.broker_port, self.keepalive)
            self.client.loop_start()
            
            # Wait for connection
            import time
            timeout = 5
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if self.connected:
                logger.info(f"Successfully connected to MQTT broker")
                return True
            else:
                logger.error("Failed to connect to MQTT broker (timeout)")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to MQTT broker: {e}")
            self.connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from MQTT broker"""
        self.running = False
        
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except:
                pass
        
        self.connected = False
        logger.info("Disconnected from MQTT broker")
    
    def subscribe(self, topic: str, qos: Optional[int] = None) -> None:
        """
        Subscribe to an additional topic.
        
        Args:
            topic: MQTT topic to subscribe to
            qos: Quality of Service level (uses default if None)
        """
        if not self.connected or not self.client:
            logger.error("Not connected to MQTT broker")
            return
        
        qos_level = qos if qos is not None else self.qos
        
        try:
            self.client.subscribe(topic, qos_level)
            if topic not in self.topics:
                self.topics.append(topic)
            logger.info(f"Subscribed to topic: {topic} with QoS {qos_level}")
        except Exception as e:
            logger.error(f"Error subscribing to topic {topic}: {e}")
    
    def publish(self, topic: str, payload: Any, qos: Optional[int] = None) -> bool:
        """
        Publish a message to a topic.
        
        Args:
            topic: MQTT topic to publish to
            payload: Message payload (will be JSON-encoded if dict)
            qos: Quality of Service level (uses default if None)
            
        Returns:
            True if published successfully, False otherwise
        """
        if not self.connected or not self.client:
            logger.error("Not connected to MQTT broker")
            return False
        
        qos_level = qos if qos is not None else self.qos
        
        try:
            if isinstance(payload, dict):
                payload = json.dumps(payload)
            elif not isinstance(payload, (str, bytes)):
                payload = str(payload)
            
            result = self.client.publish(topic, payload, qos_level)
            return result.rc == mqtt.MQTT_ERR_SUCCESS
            
        except Exception as e:
            logger.error(f"Error publishing to topic {topic}: {e}")
            return False
    
    def start(self) -> None:
        """Start the MQTT connector (connect and begin listening)"""
        if self.running:
            logger.warning("MQTT connector already running")
            return
        
        if not self.connected:
            if not self.connect():
                logger.error("Failed to connect, cannot start MQTT connector")
                return
        
        self.running = True
        logger.info("MQTT connector started")
    
    def stop(self) -> None:
        """Stop the MQTT connector"""
        self.running = False
        self.disconnect()
        logger.info("MQTT connector stopped")

