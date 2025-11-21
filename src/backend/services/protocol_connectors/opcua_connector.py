"""
OPC UA Connector for Modern SCADA Systems
Connects to OPC UA servers and reads real-time data from defined tags
"""
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import threading
import time
import asyncio

try:
    from asyncua import Client, ua
    from asyncua.common.subscription import Subscription
    OPCUA_AVAILABLE = True
except ImportError:
    OPCUA_AVAILABLE = False
    logging.warning("asyncua not installed. OPC UA connector will not work.")

logger = logging.getLogger(__name__)


class OPCUAConnector:
    """
    OPC UA connector for connecting to modern SCADA systems.
    
    Features:
    - Connection to OPC UA server
    - Real-time data reading from defined tags
    - Automatic reconnection
    - Subscription-based updates
    """
    
    def __init__(
        self,
        endpoint_url: str,
        tag_mapping: Optional[Dict[str, str]] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Initialize OPC UA connector.
        
        Args:
            endpoint_url: OPC UA server endpoint URL (e.g., 'opc.tcp://localhost:4840')
            tag_mapping: Mapping of rig parameters to OPC UA node IDs
            username: Optional username for authentication
            password: Optional password for authentication
            callback: Callback function to handle received data
        """
        if not OPCUA_AVAILABLE:
            raise ImportError("asyncua is required. Install with: pip install asyncua")
        
        self.endpoint_url = endpoint_url
        self.username = username
        self.password = password
        self.callback = callback
        
        self.client: Optional[Client] = None
        self.connected = False
        self.running = False
        self.subscription: Optional[Subscription] = None
        self.subscription_handles: Dict[str, int] = {}
        
        # Default tag mapping for common rig parameters
        # Format: {parameter_name: "ns=2;s=TagName"}
        self.tag_mapping = tag_mapping or self._get_default_mapping()
        
        # Connection retry settings
        self.max_retries = 5
        self.retry_delay = 2.0
        self.reconnect_delay = 5.0
        
        # Event loop for async operations
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.read_thread: Optional[threading.Thread] = None
    
    def _get_default_mapping(self) -> Dict[str, str]:
        """Get default OPC UA tag mapping for rig parameters"""
        return {
            "wob": "ns=2;s=Rig.WOB",
            "rpm": "ns=2;s=Rig.RPM",
            "torque": "ns=2;s=Rig.Torque",
            "rop": "ns=2;s=Rig.ROP",
            "mud_flow": "ns=2;s=Rig.MudFlow",
            "mud_pressure": "ns=2;s=Rig.MudPressure",
            "hook_load": "ns=2;s=Rig.HookLoad",
            "depth": "ns=2;s=Rig.Depth",
            "pump_status": "ns=2;s=Rig.PumpStatus",
            "power_consumption": "ns=2;s=Rig.PowerConsumption",
            "bit_temperature": "ns=2;s=Rig.BitTemperature",
            "motor_temperature": "ns=2;s=Rig.MotorTemperature",
            "casing_pressure": "ns=2;s=Rig.CasingPressure",
            "block_position": "ns=2;s=Rig.BlockPosition",
        }
    
    async def _async_connect(self) -> bool:
        """Async connection to OPC UA server"""
        try:
            self.client = Client(url=self.endpoint_url)
            
            if self.username and self.password:
                await self.client.set_user(self.username)
                await self.client.set_password(self.password)
            
            await self.client.connect()
            self.connected = True
            logger.info(f"Connected to OPC UA server: {self.endpoint_url}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to OPC UA server: {e}")
            self.connected = False
            return False
    
    def connect(self) -> bool:
        """
        Connect to OPC UA server.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.connected:
            logger.warning("Already connected to OPC UA server")
            return True
        
        try:
            if self.loop is None:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
            
            return self.loop.run_until_complete(self._async_connect())
            
        except Exception as e:
            logger.error(f"Error in connect: {e}")
            return False
    
    async def _async_disconnect(self) -> None:
        """Async disconnection from OPC UA server"""
        try:
            if self.subscription:
                await self.subscription.delete()
                self.subscription = None
            
            if self.client:
                await self.client.disconnect()
            
            self.connected = False
            logger.info("Disconnected from OPC UA server")
            
        except Exception as e:
            logger.error(f"Error disconnecting from OPC UA server: {e}")
    
    def disconnect(self) -> None:
        """Disconnect from OPC UA server"""
        self.running = False
        
        if self.loop:
            try:
                self.loop.run_until_complete(self._async_disconnect())
            except:
                pass
        
        if self.loop:
            try:
                self.loop.close()
            except:
                pass
        self.loop = None
    
    async def _async_read_tag(self, node_id: str) -> Optional[Any]:
        """
        Read a single tag value.
        
        Args:
            node_id: OPC UA node ID string
            
        Returns:
            Tag value or None on error
        """
        if not self.connected or not self.client:
            return None
        
        try:
            node = self.client.get_node(node_id)
            value = await node.read_value()
            return value
            
        except Exception as e:
            logger.error(f"Error reading OPC UA tag {node_id}: {e}")
            return None
    
    def read_tag(self, node_id: str) -> Optional[Any]:
        """
        Synchronous wrapper for reading a tag.
        
        Args:
            node_id: OPC UA node ID string
            
        Returns:
            Tag value or None on error
        """
        if not self.loop:
            return None
        
        try:
            return self.loop.run_until_complete(self._async_read_tag(node_id))
        except Exception as e:
            logger.error(f"Error in read_tag: {e}")
            return None
    
    def read_all_parameters(self, rig_id: str) -> Dict[str, Any]:
        """
        Read all mapped parameters from OPC UA server.
        
        Args:
            rig_id: Rig identifier
            
        Returns:
            Dictionary containing all read parameters
        """
        data = {
            "rig_id": rig_id,
            "timestamp": datetime.now().isoformat(),
            "protocol": "opcua",
        }
        
        if not self.loop:
            return data
        
        try:
            for param_name, node_id in self.tag_mapping.items():
                try:
                    value = self.read_tag(node_id)
                    if value is not None:
                        data[param_name] = float(value) if isinstance(value, (int, float)) else value
                    else:
                        logger.warning(f"Failed to read {param_name} from {node_id}")
                        
                except Exception as e:
                    logger.error(f"Error reading parameter {param_name}: {e}")
            
        except Exception as e:
            logger.error(f"Error reading all parameters: {e}")
        
        return data
    
    async def _async_setup_subscription(self, rig_id: str, callback: Callable) -> None:
        """Setup subscription for real-time updates"""
        try:
            # Create subscription
            self.subscription = await self.client.create_subscription(
                period=100,  # 100ms update period
                callback=callback
            )
            
            # Subscribe to all tags
            for param_name, node_id in self.tag_mapping.items():
                try:
                    node = self.client.get_node(node_id)
                    handle = await self.subscription.subscribe_data_change(node)
                    self.subscription_handles[param_name] = handle
                    logger.debug(f"Subscribed to {param_name} ({node_id})")
                except Exception as e:
                    logger.error(f"Error subscribing to {param_name}: {e}")
            
            logger.info(f"OPC UA subscription setup complete for rig {rig_id}")
            
        except Exception as e:
            logger.error(f"Error setting up subscription: {e}")
    
    def _data_change_callback(self, node: Any, val: Any, data: Any) -> None:
        """Callback for OPC UA data change notifications"""
        try:
            # Find which parameter this node corresponds to
            param_name = None
            for pname, node_id in self.tag_mapping.items():
                if str(node) == node_id:
                    param_name = pname
                    break
            
            if param_name and self.callback:
                # Create data dict
                data_dict = {
                    "rig_id": "unknown",  # Should be set from context
                    "timestamp": datetime.now().isoformat(),
                    "protocol": "opcua",
                    param_name: float(val) if isinstance(val, (int, float)) else val
                }
                
                try:
                    self.callback(data_dict)
                except Exception as e:
                    logger.error(f"Error in data change callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error in data change callback: {e}")
    
    def start_subscription(self, rig_id: str) -> None:
        """
        Start subscription-based real-time updates.
        
        Args:
            rig_id: Rig identifier
        """
        if self.running:
            logger.warning("Subscription already running")
            return
        
        if not self.connected:
            if not self.connect():
                logger.error("Failed to connect, cannot start subscription")
                return
        
        self.running = True
        
        if self.loop:
            try:
                # Wrap callback to include rig_id
                def wrapped_callback(node, val, data):
                    self._data_change_callback(node, val, data)
                    # Update rig_id in callback context if needed
                
                self.loop.run_until_complete(
                    self._async_setup_subscription(rig_id, wrapped_callback)
                )
                logger.info(f"Started OPC UA subscription for rig {rig_id}")
            except Exception as e:
                logger.error(f"Error starting subscription: {e}")
                self.running = False
    
    def start_continuous_reading(self, rig_id: str, interval: float = 1.0) -> None:
        """
        Start continuous reading from OPC UA server.
        
        Args:
            rig_id: Rig identifier
            interval: Reading interval in seconds
        """
        if self.running:
            logger.warning("Continuous reading already running")
            return
        
        if not self.connected:
            if not self.connect():
                logger.error("Failed to connect, cannot start continuous reading")
                return
        
        self.running = True
        self.read_thread = threading.Thread(
            target=self._continuous_read_loop,
            args=(rig_id, interval),
            daemon=True,
            name=f"OPCUAReader-{rig_id}"
        )
        self.read_thread.start()
        logger.info(f"Started continuous OPC UA reading for rig {rig_id}")
    
    def stop_continuous_reading(self) -> None:
        """Stop continuous reading"""
        self.running = False
        if self.read_thread:
            self.read_thread.join(timeout=5)
        logger.info("Stopped continuous OPC UA reading")
    
    def _continuous_read_loop(self, rig_id: str, interval: float) -> None:
        """Internal loop for continuous reading"""
        retry_count = 0
        
        while self.running:
            try:
                if not self.connected:
                    logger.warning("Connection lost, attempting reconnect...")
                    if self.connect():
                        retry_count = 0
                    else:
                        retry_count += 1
                        if retry_count >= self.max_retries:
                            logger.error("Max retries reached, stopping continuous reading")
                            break
                        time.sleep(self.retry_delay)
                        continue
                
                # Read all parameters
                data = self.read_all_parameters(rig_id)
                
                # Call callback if provided
                if self.callback:
                    try:
                        self.callback(data)
                    except Exception as e:
                        logger.error(f"Error in callback: {e}")
                
                retry_count = 0
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in continuous read loop: {e}")
                retry_count += 1
                if retry_count >= self.max_retries:
                    logger.error("Max retries reached, stopping continuous reading")
                    break
                time.sleep(self.retry_delay)
        
        self.running = False

