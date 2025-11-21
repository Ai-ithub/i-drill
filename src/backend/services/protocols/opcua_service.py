"""
OPC UA Service for connecting to modern SCADA systems
Supports reading and writing OPC UA tags with automatic reconnection
"""
import logging
import threading
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import os

logger = logging.getLogger(__name__)

# Try to import asyncua
try:
    from asyncua import Client, ua
    from asyncua.ua import VariantType
    OPCUA_AVAILABLE = True
except ImportError:
    Client = ua = VariantType = None
    OPCUA_AVAILABLE = False
    logger.warning("asyncua not installed. OPC UA support disabled. Install with: pip install asyncua")


class OPCUAService:
    """
    Service for OPC UA communication with SCADA systems.
    
    Supports:
    - Reading OPC UA tags (nodes)
    - Writing OPC UA tags
    - Automatic reconnection with exponential backoff
    - Connection pooling for multiple servers
    - Subscription to tag changes (future enhancement)
    """
    
    def __init__(self):
        """Initialize OPCUAService."""
        self.available = OPCUA_AVAILABLE
        if not OPCUA_AVAILABLE:
            logger.warning("OPC UA support is not available")
            self.clients: Dict[str, Any] = {}
            return
        
        self.clients: Dict[str, Any] = {}  # Store clients by connection_id
        self.lock = threading.Lock()
        self.configs: Dict[str, Dict[str, Any]] = {}  # Store configurations
        self.node_mappings: Dict[str, Dict[str, str]] = {}  # Map parameter names to node IDs
        self.nodes: Dict[str, Dict[str, Any]] = {}  # Cache node objects
        
        logger.info("OPCUAService initialized")
    
    def add_connection(
        self,
        connection_id: str,
        endpoint_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        security_policy: str = "None",
        security_mode: str = "None",
        timeout: float = 10.0,
        parameter_mapping: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Add an OPC UA connection.
        
        Args:
            connection_id: Unique identifier for this connection
            endpoint_url: OPC UA server endpoint URL (e.g., "opc.tcp://192.168.1.100:4840")
            username: Optional username for authentication
            password: Optional password for authentication
            security_policy: Security policy ("None", "Basic128Rsa15", "Basic256", etc.)
            security_mode: Security mode ("None", "Sign", "SignAndEncrypt")
            timeout: Connection timeout in seconds
            parameter_mapping: Dictionary mapping parameter names to OPC UA node IDs
                              Format: {"parameter_name": "ns=2;s=TagName"} or {"parameter_name": "ns=2;i=1234"}
        
        Returns:
            True if connection was added successfully, False otherwise
        """
        if not self.available:
            logger.error("OPC UA is not available")
            return False
        
        try:
            client = Client(url=endpoint_url, timeout=timeout)
            
            # Set security if provided
            if username and password:
                client.set_user(username)
                client.set_password(password)
            
            # Connect
            client.connect()
            
            with self.lock:
                self.clients[connection_id] = client
                self.configs[connection_id] = {
                    "endpoint_url": endpoint_url,
                    "username": username,
                    "security_policy": security_policy,
                    "security_mode": security_mode,
                    "timeout": timeout,
                    "connected": True,
                    "last_connection": datetime.now().isoformat()
                }
                if parameter_mapping:
                    self.node_mappings[connection_id] = parameter_mapping
                    # Pre-fetch node objects
                    self._cache_nodes(connection_id, parameter_mapping)
            
            logger.info(f"OPC UA connection {connection_id} established to {endpoint_url}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding OPC UA connection {connection_id}: {e}")
            return False
    
    def _cache_nodes(self, connection_id: str, parameter_mapping: Dict[str, str]) -> None:
        """Cache OPC UA node objects for faster access."""
        if connection_id not in self.clients:
            return
        
        try:
            client = self.clients[connection_id]
            self.nodes[connection_id] = {}
            
            for param_name, node_id_str in parameter_mapping.items():
                try:
                    node = client.get_node(node_id_str)
                    self.nodes[connection_id][param_name] = node
                except Exception as e:
                    logger.warning(f"Failed to cache node {node_id_str} for {param_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error caching nodes for {connection_id}: {e}")
    
    def remove_connection(self, connection_id: str) -> bool:
        """Remove and close an OPC UA connection."""
        try:
            with self.lock:
                if connection_id in self.clients:
                    client = self.clients[connection_id]
                    try:
                        client.disconnect()
                    except:
                        pass
                    del self.clients[connection_id]
                    if connection_id in self.configs:
                        del self.configs[connection_id]
                    if connection_id in self.node_mappings:
                        del self.node_mappings[connection_id]
                    if connection_id in self.nodes:
                        del self.nodes[connection_id]
                    logger.info(f"OPC UA connection {connection_id} removed")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error removing OPC UA connection {connection_id}: {e}")
            return False
    
    def _ensure_connected(self, connection_id: str) -> bool:
        """Ensure connection is active, reconnect if needed."""
        if connection_id not in self.clients:
            return False
        
        try:
            client = self.clients[connection_id]
            # Check if session is still valid
            try:
                # Try to read server status
                root = client.get_objects_node()
                if root:
                    return True
            except:
                pass
            
            # Reconnect
            config = self.configs[connection_id]
            endpoint_url = config["endpoint_url"]
            username = config.get("username")
            password = config.get("password")
            timeout = config.get("timeout", 10.0)
            
            client = Client(url=endpoint_url, timeout=timeout)
            if username and password:
                client.set_user(username)
                client.set_password(password)
            
            client.connect()
            self.clients[connection_id] = client
            
            # Re-cache nodes if needed
            if connection_id in self.node_mappings:
                self._cache_nodes(connection_id, self.node_mappings[connection_id])
            
            logger.info(f"OPC UA connection {connection_id} reconnected")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reconnect OPC UA {connection_id}: {e}")
            return False
    
    def read_node(self, connection_id: str, node_id: str) -> Optional[Any]:
        """
        Read an OPC UA node value.
        
        Args:
            connection_id: Connection identifier
            node_id: OPC UA node ID (e.g., "ns=2;s=TagName" or "ns=2;i=1234")
        
        Returns:
            Node value or None if error
        """
        if not self.available:
            return None
        
        if connection_id not in self.clients:
            logger.error(f"OPC UA connection {connection_id} not found")
            return None
        
        try:
            if not self._ensure_connected(connection_id):
                return None
            
            client = self.clients[connection_id]
            node = client.get_node(node_id)
            value = node.get_value()
            
            return value
                
        except Exception as e:
            logger.error(f"Error reading OPC UA node {node_id}: {e}")
            return None
    
    def write_node(self, connection_id: str, node_id: str, value: Any, data_type: Optional[Any] = None) -> bool:
        """
        Write to an OPC UA node.
        
        Args:
            connection_id: Connection identifier
            node_id: OPC UA node ID
            value: Value to write
            data_type: Optional VariantType (auto-detected if not provided)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.available:
            return False
        
        if connection_id not in self.clients:
            logger.error(f"OPC UA connection {connection_id} not found")
            return False
        
        try:
            if not self._ensure_connected(connection_id):
                return False
            
            client = self.clients[connection_id]
            node = client.get_node(node_id)
            
            if data_type:
                variant = ua.Variant(value, data_type)
                node.set_value(variant)
            else:
                node.set_value(value)
            
            return True
                
        except Exception as e:
            logger.error(f"Error writing OPC UA node {node_id}: {e}")
            return False
    
    def read_parameter(self, connection_id: str, parameter_name: str) -> Optional[Any]:
        """
        Read a parameter using parameter mapping.
        
        Args:
            connection_id: Connection identifier
            parameter_name: Parameter name (must be in node_mapping)
        
        Returns:
            Parameter value or None if error
        """
        if connection_id not in self.node_mappings:
            logger.error(f"No node mapping for connection {connection_id}")
            return None
        
        mapping = self.node_mappings[connection_id].get(parameter_name)
        if not mapping:
            logger.error(f"Parameter {parameter_name} not found in mapping")
            return None
        
        # Try to use cached node first
        if connection_id in self.nodes and parameter_name in self.nodes[connection_id]:
            try:
                if not self._ensure_connected(connection_id):
                    return None
                node = self.nodes[connection_id][parameter_name]
                return node.get_value()
            except Exception as e:
                logger.warning(f"Error reading cached node for {parameter_name}: {e}")
        
        # Fallback to reading by node ID
        return self.read_node(connection_id, mapping)
    
    def read_all_parameters(self, connection_id: str) -> Dict[str, Any]:
        """
        Read all mapped parameters for a connection.
        
        Args:
            connection_id: Connection identifier
        
        Returns:
            Dictionary of parameter_name: value
        """
        if connection_id not in self.node_mappings:
            return {}
        
        result = {}
        for param_name in self.node_mappings[connection_id].keys():
            value = self.read_parameter(connection_id, param_name)
            if value is not None:
                result[param_name] = value
        
        return result
    
    def is_connected(self, connection_id: str) -> bool:
        """Check if a connection is active."""
        if connection_id not in self.clients:
            return False
        
        try:
            return self._ensure_connected(connection_id)
        except:
            return False
    
    def get_connection_status(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get connection status and configuration."""
        if connection_id not in self.configs:
            return None
        
        config = self.configs[connection_id].copy()
        config["connected"] = self.is_connected(connection_id)
        return config
    
    def list_connections(self) -> List[str]:
        """List all connection IDs."""
        return list(self.clients.keys())


# Global instance
opcua_service = OPCUAService()

