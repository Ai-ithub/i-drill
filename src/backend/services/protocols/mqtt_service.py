"""
MQTT Service for connecting to IoT sensors and devices
Supports subscribing to topics and publishing commands
"""
import logging
import threading
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

# Try to import paho-mqtt
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    mqtt = None
    MQTT_AVAILABLE = False
    logger.warning("paho-mqtt not installed. MQTT support disabled. Install with: pip install paho-mqtt")


class MQTTService:
    """
    Service for MQTT communication with IoT sensors and devices.
    
    Supports:
    - Subscribing to topics for sensor data
    - Publishing commands to devices
    - QoS levels (0, 1, 2)
    - Automatic reconnection with exponential backoff
    - Multiple broker connections
    - Message callbacks
    """
    
    def __init__(self):
        """Initialize MQTTService."""
        self.available = MQTT_AVAILABLE
        if not MQTT_AVAILABLE:
            logger.warning("MQTT support is not available")
            self.clients: Dict[str, Any] = {}
            return
        
        self.clients: Dict[str, Any] = {}  # Store clients by connection_id
        self.lock = threading.Lock()
        self.configs: Dict[str, Dict[str, Any]] = {}  # Store configurations
        self.subscriptions: Dict[str, Dict[str, Callable]] = {}  # Store topic subscriptions and callbacks
        self.message_handlers: Dict[str, Callable] = {}  # Global message handlers
        
        logger.info("MQTTService initialized")
    
    def add_connection(
        self,
        connection_id: str,
        broker_host: str,
        broker_port: int = 1883,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_id: Optional[str] = None,
        keepalive: int = 60,
        clean_session: bool = True,
        tls_enabled: bool = False,
        ca_certs: Optional[str] = None
    ) -> bool:
        """
        Add an MQTT broker connection.
        
        Args:
            connection_id: Unique identifier for this connection
            broker_host: MQTT broker hostname or IP
            broker_port: MQTT broker port (default: 1883)
            username: Optional username for authentication
            password: Optional password for authentication
            client_id: Optional client ID (auto-generated if not provided)
            keepalive: Keepalive interval in seconds
            clean_session: Clean session flag
            tls_enabled: Enable TLS encryption
            ca_certs: Path to CA certificate file for TLS
        
        Returns:
            True if connection was added successfully, False otherwise
        """
        if not self.available:
            logger.error("MQTT is not available")
            return False
        
        try:
            client_id = client_id or f"i-drill-{connection_id}-{int(time.time())}"
            client = mqtt.Client(client_id=client_id, clean_session=clean_session)
            
            # Set credentials if provided
            if username and password:
                client.username_pw_set(username, password)
            
            # Set TLS if enabled
            if tls_enabled:
                if ca_certs:
                    client.tls_set(ca_certs=ca_certs)
                else:
                    client.tls_set()
            
            # Set callbacks
            client.on_connect = self._on_connect_factory(connection_id)
            client.on_disconnect = self._on_disconnect_factory(connection_id)
            client.on_message = self._on_message_factory(connection_id)
            client.on_subscribe = self._on_subscribe_factory(connection_id)
            client.on_publish = self._on_publish_factory(connection_id)
            
            # Connect
            client.connect(broker_host, broker_port, keepalive)
            client.loop_start()  # Start network loop in background thread
            
            with self.lock:
                self.clients[connection_id] = client
                self.configs[connection_id] = {
                    "broker_host": broker_host,
                    "broker_port": broker_port,
                    "username": username,
                    "client_id": client_id,
                    "keepalive": keepalive,
                    "clean_session": clean_session,
                    "tls_enabled": tls_enabled,
                    "connected": True,
                    "last_connection": datetime.now().isoformat()
                }
                self.subscriptions[connection_id] = {}
            
            # Wait a bit for connection to establish
            time.sleep(0.5)
            
            logger.info(f"MQTT connection {connection_id} established to {broker_host}:{broker_port}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding MQTT connection {connection_id}: {e}")
            return False
    
    def _on_connect_factory(self, connection_id: str):
        """Factory for on_connect callback."""
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                logger.info(f"MQTT connection {connection_id} connected successfully")
                with self.lock:
                    if connection_id in self.configs:
                        self.configs[connection_id]["connected"] = True
                        self.configs[connection_id]["last_connection"] = datetime.now().isoformat()
                
                # Re-subscribe to all topics
                if connection_id in self.subscriptions:
                    for topic, qos in self.subscriptions[connection_id].keys():
                        client.subscribe(topic, qos)
            else:
                logger.error(f"MQTT connection {connection_id} failed with code {rc}")
        return on_connect
    
    def _on_disconnect_factory(self, connection_id: str):
        """Factory for on_disconnect callback."""
        def on_disconnect(client, userdata, rc):
            logger.warning(f"MQTT connection {connection_id} disconnected (rc={rc})")
            with self.lock:
                if connection_id in self.configs:
                    self.configs[connection_id]["connected"] = False
        return on_disconnect
    
    def _on_message_factory(self, connection_id: str):
        """Factory for on_message callback."""
        def on_message(client, userdata, msg):
            topic = msg.topic
            try:
                payload = msg.payload.decode('utf-8')
                # Try to parse as JSON
                try:
                    data = json.loads(payload)
                except:
                    data = payload
                
                # Call topic-specific callback if exists
                if connection_id in self.subscriptions:
                    if topic in self.subscriptions[connection_id]:
                        callback = self.subscriptions[connection_id][topic]
                        if callback:
                            try:
                                callback(topic, data, connection_id)
                            except Exception as e:
                                logger.error(f"Error in MQTT callback for {topic}: {e}")
                
                # Call global message handler if exists
                if connection_id in self.message_handlers:
                    handler = self.message_handlers[connection_id]
                    if handler:
                        try:
                            handler(topic, data, connection_id)
                        except Exception as e:
                            logger.error(f"Error in global MQTT handler: {e}")
                            
            except Exception as e:
                logger.error(f"Error processing MQTT message from {topic}: {e}")
        return on_message
    
    def _on_subscribe_factory(self, connection_id: str):
        """Factory for on_subscribe callback."""
        def on_subscribe(client, userdata, mid, granted_qos):
            logger.debug(f"MQTT connection {connection_id} subscribed (mid={mid}, qos={granted_qos})")
        return on_subscribe
    
    def _on_publish_factory(self, connection_id: str):
        """Factory for on_publish callback."""
        def on_publish(client, userdata, mid):
            logger.debug(f"MQTT connection {connection_id} published (mid={mid})")
        return on_publish
    
    def remove_connection(self, connection_id: str) -> bool:
        """Remove and close an MQTT connection."""
        try:
            with self.lock:
                if connection_id in self.clients:
                    client = self.clients[connection_id]
                    client.loop_stop()
                    client.disconnect()
                    del self.clients[connection_id]
                    if connection_id in self.configs:
                        del self.configs[connection_id]
                    if connection_id in self.subscriptions:
                        del self.subscriptions[connection_id]
                    if connection_id in self.message_handlers:
                        del self.message_handlers[connection_id]
                    logger.info(f"MQTT connection {connection_id} removed")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error removing MQTT connection {connection_id}: {e}")
            return False
    
    def subscribe(
        self,
        connection_id: str,
        topic: str,
        qos: int = 1,
        callback: Optional[Callable] = None
    ) -> bool:
        """
        Subscribe to an MQTT topic.
        
        Args:
            connection_id: Connection identifier
            topic: MQTT topic (supports wildcards: +, #)
            qos: Quality of Service level (0, 1, or 2)
            callback: Optional callback function(topic, data, connection_id)
        
        Returns:
            True if subscription was successful, False otherwise
        """
        if not self.available:
            return False
        
        if connection_id not in self.clients:
            logger.error(f"MQTT connection {connection_id} not found")
            return False
        
        try:
            client = self.clients[connection_id]
            
            # Check if connected
            if not client.is_connected():
                logger.warning(f"MQTT connection {connection_id} not connected, attempting reconnect...")
                config = self.configs[connection_id]
                client.reconnect()
                time.sleep(0.5)
            
            result, mid = client.subscribe(topic, qos)
            
            if result == mqtt.MQTT_ERR_SUCCESS:
                with self.lock:
                    if connection_id not in self.subscriptions:
                        self.subscriptions[connection_id] = {}
                    self.subscriptions[connection_id][topic] = callback
                
                logger.info(f"MQTT connection {connection_id} subscribed to {topic} (qos={qos})")
                return True
            else:
                logger.error(f"Failed to subscribe to {topic}: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error subscribing to MQTT topic {topic}: {e}")
            return False
    
    def unsubscribe(self, connection_id: str, topic: str) -> bool:
        """Unsubscribe from an MQTT topic."""
        if connection_id not in self.clients:
            return False
        
        try:
            client = self.clients[connection_id]
            result, mid = client.unsubscribe(topic)
            
            if result == mqtt.MQTT_ERR_SUCCESS:
                with self.lock:
                    if connection_id in self.subscriptions and topic in self.subscriptions[connection_id]:
                        del self.subscriptions[connection_id][topic]
                logger.info(f"MQTT connection {connection_id} unsubscribed from {topic}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error unsubscribing from MQTT topic {topic}: {e}")
            return False
    
    def publish(
        self,
        connection_id: str,
        topic: str,
        payload: Any,
        qos: int = 1,
        retain: bool = False
    ) -> bool:
        """
        Publish a message to an MQTT topic.
        
        Args:
            connection_id: Connection identifier
            topic: MQTT topic
            payload: Message payload (dict will be JSON-encoded, str sent as-is)
            qos: Quality of Service level (0, 1, or 2)
            retain: Retain flag
        
        Returns:
            True if publish was successful, False otherwise
        """
        if not self.available:
            return False
        
        if connection_id not in self.clients:
            logger.error(f"MQTT connection {connection_id} not found")
            return False
        
        try:
            client = self.clients[connection_id]
            
            # Check if connected
            if not client.is_connected():
                logger.warning(f"MQTT connection {connection_id} not connected")
                return False
            
            # Encode payload
            if isinstance(payload, dict):
                payload_str = json.dumps(payload)
            else:
                payload_str = str(payload)
            
            result = client.publish(topic, payload_str, qos=qos, retain=retain)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.debug(f"MQTT connection {connection_id} published to {topic}")
                return True
            else:
                logger.error(f"Failed to publish to {topic}: {result.rc}")
                return False
                
        except Exception as e:
            logger.error(f"Error publishing to MQTT topic {topic}: {e}")
            return False
    
    def set_message_handler(self, connection_id: str, handler: Callable) -> None:
        """
        Set a global message handler for a connection.
        
        Args:
            connection_id: Connection identifier
            handler: Callback function(topic, data, connection_id)
        """
        self.message_handlers[connection_id] = handler
    
    def is_connected(self, connection_id: str) -> bool:
        """Check if a connection is active."""
        if connection_id not in self.clients:
            return False
        
        try:
            return self.clients[connection_id].is_connected()
        except:
            return False
    
    def get_connection_status(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get connection status and configuration."""
        if connection_id not in self.configs:
            return None
        
        config = self.configs[connection_id].copy()
        config["connected"] = self.is_connected(connection_id)
        if connection_id in self.subscriptions:
            config["subscriptions"] = list(self.subscriptions[connection_id].keys())
        return config
    
    def list_connections(self) -> List[str]:
        """List all connection IDs."""
        return list(self.clients.keys())


# Global instance
mqtt_service = MQTTService()

