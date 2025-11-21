"""
Modbus RTU/TCP Connector for Rig PLC Connection
Supports both Modbus TCP and Modbus RTU protocols
"""
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import threading
import time
from enum import Enum

try:
    from pymodbus.client import ModbusTcpClient, ModbusSerialClient
    from pymodbus.exceptions import ModbusException
    from pymodbus.constants import Endian
    from pymodbus.payload import BinaryPayloadDecoder
    MODBUS_AVAILABLE = True
except ImportError:
    MODBUS_AVAILABLE = False
    logging.warning("pymodbus not installed. Modbus connector will not work.")

logger = logging.getLogger(__name__)


class ModbusProtocol(str, Enum):
    """Modbus protocol types"""
    TCP = "tcp"
    RTU = "rtu"


class ModbusConnector:
    """
    Modbus RTU/TCP connector for connecting to rig PLC systems.
    
    Supports:
    - Modbus TCP for main sensors
    - Modbus RTU for older sensors
    - Automatic reconnection
    - Data mapping from Modbus registers to system data model
    """
    
    def __init__(
        self,
        protocol: ModbusProtocol = ModbusProtocol.TCP,
        host: Optional[str] = None,
        port: int = 502,
        unit_id: int = 1,
        baudrate: int = 9600,
        port_name: Optional[str] = None,
        register_mapping: Optional[Dict[str, Dict[str, Any]]] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Initialize Modbus connector.
        
        Args:
            protocol: Modbus protocol type (TCP or RTU)
            host: IP address for TCP (required for TCP)
            port: Port number for TCP (default: 502)
            unit_id: Modbus unit/slave ID
            baudrate: Baud rate for RTU (default: 9600)
            port_name: Serial port name for RTU (e.g., 'COM1', '/dev/ttyUSB0')
            register_mapping: Mapping of rig parameters to Modbus registers
            callback: Callback function to handle received data
        """
        if not MODBUS_AVAILABLE:
            raise ImportError("pymodbus is required. Install with: pip install pymodbus")
        
        self.protocol = protocol
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.baudrate = baudrate
        self.port_name = port_name
        self.callback = callback
        
        self.client = None
        self.connected = False
        self.running = False
        self.read_thread: Optional[threading.Thread] = None
        
        # Default register mapping for common rig parameters
        # Format: {parameter_name: {address: int, register_type: str, data_type: str, scale: float}}
        self.register_mapping = register_mapping or self._get_default_mapping()
        
        # Connection retry settings
        self.max_retries = 5
        self.retry_delay = 2.0
        
    def _get_default_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Get default Modbus register mapping for rig parameters"""
        return {
            "wob": {"address": 0, "register_type": "holding", "data_type": "float32", "scale": 1.0},
            "rpm": {"address": 2, "register_type": "holding", "data_type": "uint16", "scale": 1.0},
            "torque": {"address": 3, "register_type": "holding", "data_type": "float32", "scale": 1.0},
            "rop": {"address": 5, "register_type": "holding", "data_type": "float32", "scale": 1.0},
            "mud_flow": {"address": 7, "register_type": "holding", "data_type": "float32", "scale": 1.0},
            "mud_pressure": {"address": 9, "register_type": "holding", "data_type": "float32", "scale": 1.0},
            "hook_load": {"address": 11, "register_type": "holding", "data_type": "float32", "scale": 1.0},
            "depth": {"address": 13, "register_type": "holding", "data_type": "float32", "scale": 1.0},
            "pump_status": {"address": 15, "register_type": "coil", "data_type": "bool", "scale": 1.0},
            "power_consumption": {"address": 16, "register_type": "holding", "data_type": "float32", "scale": 1.0},
        }
    
    def connect(self) -> bool:
        """
        Connect to Modbus device.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.connected:
            logger.warning("Already connected to Modbus device")
            return True
        
        try:
            if self.protocol == ModbusProtocol.TCP:
                if not self.host:
                    raise ValueError("Host is required for Modbus TCP")
                self.client = ModbusTcpClient(host=self.host, port=self.port)
            else:  # RTU
                if not self.port_name:
                    raise ValueError("Port name is required for Modbus RTU")
                self.client = ModbusSerialClient(
                    port=self.port_name,
                    baudrate=self.baudrate,
                    method='rtu'
                )
            
            if self.client.connect():
                self.connected = True
                logger.info(f"Connected to Modbus {self.protocol.value} device")
                return True
            else:
                logger.error(f"Failed to connect to Modbus {self.protocol.value} device")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to Modbus device: {e}")
            self.connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Modbus device"""
        self.running = False
        if self.client:
            try:
                self.client.close()
            except:
                pass
        self.connected = False
        logger.info("Disconnected from Modbus device")
    
    def read_register(
        self,
        address: int,
        register_type: str = "holding",
        count: int = 1,
        data_type: str = "uint16"
    ) -> Optional[Any]:
        """
        Read a single register or register block.
        
        Args:
            address: Register address
            register_type: Type of register (holding, input, coil, discrete)
            count: Number of registers to read
            data_type: Data type (uint16, int16, float32, bool)
            
        Returns:
            Decoded register value or None on error
        """
        if not self.connected or not self.client:
            logger.error("Not connected to Modbus device")
            return None
        
        try:
            if register_type == "holding":
                result = self.client.read_holding_registers(address, count, unit=self.unit_id)
            elif register_type == "input":
                result = self.client.read_input_registers(address, count, unit=self.unit_id)
            elif register_type == "coil":
                result = self.client.read_coils(address, count, unit=self.unit_id)
            elif register_type == "discrete":
                result = self.client.read_discrete_inputs(address, count, unit=self.unit_id)
            else:
                logger.error(f"Unknown register type: {register_type}")
                return None
            
            if result.isError():
                logger.error(f"Modbus read error: {result}")
                return None
            
            # Decode based on data type
            if data_type == "bool":
                return bool(result.bits[0] if hasattr(result, 'bits') else result.registers[0])
            elif data_type == "uint16":
                return result.registers[0] if result.registers else None
            elif data_type == "int16":
                decoder = BinaryPayloadDecoder.fromRegisters(
                    result.registers,
                    byteorder=Endian.BIG,
                    wordorder=Endian.BIG
                )
                return decoder.decode_16bit_int()
            elif data_type == "float32":
                if len(result.registers) < 2:
                    return None
                decoder = BinaryPayloadDecoder.fromRegisters(
                    result.registers,
                    byteorder=Endian.BIG,
                    wordorder=Endian.BIG
                )
                return decoder.decode_32bit_float()
            else:
                logger.warning(f"Unsupported data type: {data_type}")
                return result.registers[0] if result.registers else None
                
        except ModbusException as e:
            logger.error(f"Modbus exception: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading Modbus register: {e}")
            return None
    
    def read_all_parameters(self, rig_id: str) -> Dict[str, Any]:
        """
        Read all mapped parameters from Modbus device.
        
        Args:
            rig_id: Rig identifier
            
        Returns:
            Dictionary containing all read parameters
        """
        data = {
            "rig_id": rig_id,
            "timestamp": datetime.now().isoformat(),
            "protocol": f"modbus_{self.protocol.value}",
        }
        
        for param_name, mapping in self.register_mapping.items():
            try:
                address = mapping["address"]
                register_type = mapping.get("register_type", "holding")
                data_type = mapping.get("data_type", "uint16")
                scale = mapping.get("scale", 1.0)
                
                # Determine count based on data type
                count = 2 if data_type == "float32" else 1
                
                value = self.read_register(address, register_type, count, data_type)
                if value is not None:
                    data[param_name] = float(value) * scale
                else:
                    logger.warning(f"Failed to read {param_name} from address {address}")
                    
            except Exception as e:
                logger.error(f"Error reading parameter {param_name}: {e}")
        
        return data
    
    def start_continuous_reading(self, rig_id: str, interval: float = 1.0) -> None:
        """
        Start continuous reading from Modbus device.
        
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
            name=f"ModbusReader-{rig_id}"
        )
        self.read_thread.start()
        logger.info(f"Started continuous Modbus reading for rig {rig_id}")
    
    def stop_continuous_reading(self) -> None:
        """Stop continuous reading"""
        self.running = False
        if self.read_thread:
            self.read_thread.join(timeout=5)
        logger.info("Stopped continuous Modbus reading")
    
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

