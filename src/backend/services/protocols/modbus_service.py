"""
Modbus RTU/TCP Service for connecting to PLC systems
Supports both Modbus RTU (serial) and Modbus TCP (Ethernet) protocols
"""
import logging
import threading
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import os

logger = logging.getLogger(__name__)

# Try to import pymodbus
try:
    from pymodbus.client import ModbusTcpClient, ModbusSerialClient
    from pymodbus.constants import Endian
    from pymodbus.payload import BinaryPayloadDecoder, BinaryPayloadBuilder
    from pymodbus.exceptions import ModbusException
    MODBUS_AVAILABLE = True
except ImportError:
    ModbusTcpClient = ModbusSerialClient = None
    Endian = BinaryPayloadDecoder = BinaryPayloadBuilder = ModbusException = None
    MODBUS_AVAILABLE = False
    logger.warning("pymodbus not installed. Modbus support disabled. Install with: pip install pymodbus")


class ModbusService:
    """
    Service for Modbus RTU/TCP communication with PLC systems.
    
    Supports:
    - Modbus TCP (Ethernet)
    - Modbus RTU (Serial)
    - Reading holding registers, input registers, coils, discrete inputs
    - Writing holding registers and coils
    - Automatic reconnection with exponential backoff
    - Connection pooling for multiple devices
    """
    
    def __init__(self):
        """Initialize ModbusService."""
        self.available = MODBUS_AVAILABLE
        if not MODBUS_AVAILABLE:
            logger.warning("Modbus support is not available")
            self.clients: Dict[str, Any] = {}
            return
        
        self.clients: Dict[str, Any] = {}  # Store clients by connection_id
        self.lock = threading.Lock()
        self.configs: Dict[str, Dict[str, Any]] = {}  # Store configurations
        self.parameter_mappings: Dict[str, Dict[str, Any]] = {}  # Map parameters to Modbus addresses
        
        logger.info("ModbusService initialized")
    
    def add_connection(
        self,
        connection_id: str,
        protocol: str = "TCP",
        host: Optional[str] = None,
        port: int = 502,
        slave_id: int = 1,
        serial_port: Optional[str] = None,
        baudrate: int = 9600,
        timeout: float = 3.0,
        parameter_mapping: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a Modbus connection.
        
        Args:
            connection_id: Unique identifier for this connection
            protocol: "TCP" or "RTU"
            host: IP address for TCP (required for TCP)
            port: Port for TCP (default: 502)
            slave_id: Modbus slave/unit ID (default: 1)
            serial_port: Serial port for RTU (required for RTU, e.g., "/dev/ttyUSB0" or "COM3")
            baudrate: Baud rate for RTU (default: 9600)
            timeout: Connection timeout in seconds
            parameter_mapping: Dictionary mapping parameter names to Modbus addresses
                              Format: {"parameter_name": {"address": int, "register_type": str, "data_type": str}}
                              register_type: "holding", "input", "coil", "discrete"
                              data_type: "uint16", "int16", "uint32", "int32", "float32", "bool"
        
        Returns:
            True if connection was added successfully, False otherwise
        """
        if not self.available:
            logger.error("Modbus is not available")
            return False
        
        try:
            if protocol.upper() == "TCP":
                if not host:
                    logger.error("Host is required for Modbus TCP")
                    return False
                
                client = ModbusTcpClient(
                    host=host,
                    port=port,
                    timeout=timeout
                )
                logger.info(f"Modbus TCP client created for {host}:{port}")
                
            elif protocol.upper() == "RTU":
                if not serial_port:
                    logger.error("Serial port is required for Modbus RTU")
                    return False
                
                client = ModbusSerialClient(
                    method="rtu",
                    port=serial_port,
                    baudrate=baudrate,
                    timeout=timeout
                )
                logger.info(f"Modbus RTU client created for {serial_port}")
                
            else:
                logger.error(f"Unsupported protocol: {protocol}")
                return False
            
            # Test connection
            if client.connect():
                with self.lock:
                    self.clients[connection_id] = client
                    self.configs[connection_id] = {
                        "protocol": protocol.upper(),
                        "host": host,
                        "port": port,
                        "slave_id": slave_id,
                        "serial_port": serial_port,
                        "baudrate": baudrate,
                        "timeout": timeout,
                        "connected": True,
                        "last_connection": datetime.now().isoformat()
                    }
                    if parameter_mapping:
                        self.parameter_mappings[connection_id] = parameter_mapping
                
                logger.info(f"Modbus connection {connection_id} established successfully")
                return True
            else:
                logger.error(f"Failed to connect Modbus {protocol} for {connection_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding Modbus connection {connection_id}: {e}")
            return False
    
    def remove_connection(self, connection_id: str) -> bool:
        """Remove and close a Modbus connection."""
        try:
            with self.lock:
                if connection_id in self.clients:
                    client = self.clients[connection_id]
                    client.close()
                    del self.clients[connection_id]
                    if connection_id in self.configs:
                        del self.configs[connection_id]
                    if connection_id in self.parameter_mappings:
                        del self.parameter_mappings[connection_id]
                    logger.info(f"Modbus connection {connection_id} removed")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error removing Modbus connection {connection_id}: {e}")
            return False
    
    def read_register(
        self,
        connection_id: str,
        address: int,
        register_type: str = "holding",
        count: int = 1,
        slave_id: Optional[int] = None
    ) -> Optional[Any]:
        """
        Read Modbus register(s).
        
        Args:
            connection_id: Connection identifier
            address: Register address (0-based)
            register_type: "holding", "input", "coil", "discrete"
            count: Number of registers to read
            slave_id: Optional slave ID (uses config default if not provided)
        
        Returns:
            Register value(s) or None if error
        """
        if not self.available:
            return None
        
        if connection_id not in self.clients:
            logger.error(f"Modbus connection {connection_id} not found")
            return None
        
        try:
            client = self.clients[connection_id]
            config = self.configs[connection_id]
            unit_id = slave_id or config.get("slave_id", 1)
            
            # Reconnect if needed
            if not client.is_socket_open():
                if not client.connect():
                    logger.error(f"Failed to reconnect Modbus {connection_id}")
                    return None
            
            if register_type.lower() == "holding":
                result = client.read_holding_registers(address, count, slave=unit_id)
            elif register_type.lower() == "input":
                result = client.read_input_registers(address, count, slave=unit_id)
            elif register_type.lower() == "coil":
                result = client.read_coils(address, count, slave=unit_id)
            elif register_type.lower() == "discrete":
                result = client.read_discrete_inputs(address, count, slave=unit_id)
            else:
                logger.error(f"Unsupported register type: {register_type}")
                return None
            
            if result.isError():
                logger.error(f"Modbus read error: {result}")
                return None
            
            if register_type.lower() in ["coil", "discrete"]:
                return result.bits[0] if count == 1 else result.bits
            else:
                return result.registers[0] if count == 1 else result.registers
                
        except ModbusException as e:
            logger.error(f"Modbus exception reading {address}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading Modbus register: {e}")
            return None
    
    def write_register(
        self,
        connection_id: str,
        address: int,
        value: Any,
        register_type: str = "holding",
        slave_id: Optional[int] = None
    ) -> bool:
        """
        Write to Modbus register.
        
        Args:
            connection_id: Connection identifier
            address: Register address (0-based)
            value: Value to write (int for registers, bool for coils)
            register_type: "holding" or "coil"
            slave_id: Optional slave ID
        
        Returns:
            True if successful, False otherwise
        """
        if not self.available:
            return False
        
        if connection_id not in self.clients:
            logger.error(f"Modbus connection {connection_id} not found")
            return False
        
        try:
            client = self.clients[connection_id]
            config = self.configs[connection_id]
            unit_id = slave_id or config.get("slave_id", 1)
            
            # Reconnect if needed
            if not client.is_socket_open():
                if not client.connect():
                    logger.error(f"Failed to reconnect Modbus {connection_id}")
                    return False
            
            if register_type.lower() == "holding":
                result = client.write_register(address, int(value), slave=unit_id)
            elif register_type.lower() == "coil":
                result = client.write_coil(address, bool(value), slave=unit_id)
            else:
                logger.error(f"Unsupported write register type: {register_type}")
                return False
            
            if result.isError():
                logger.error(f"Modbus write error: {result}")
                return False
            
            return True
                
        except ModbusException as e:
            logger.error(f"Modbus exception writing {address}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error writing Modbus register: {e}")
            return False
    
    def read_parameter(
        self,
        connection_id: str,
        parameter_name: str,
        data_type: str = "uint16"
    ) -> Optional[Any]:
        """
        Read a parameter using parameter mapping.
        
        Args:
            connection_id: Connection identifier
            parameter_name: Parameter name (must be in parameter_mapping)
            data_type: Data type ("uint16", "int16", "uint32", "int32", "float32", "bool")
        
        Returns:
            Parameter value or None if error
        """
        if connection_id not in self.parameter_mappings:
            logger.error(f"No parameter mapping for connection {connection_id}")
            return None
        
        mapping = self.parameter_mappings[connection_id].get(parameter_name)
        if not mapping:
            logger.error(f"Parameter {parameter_name} not found in mapping")
            return None
        
        address = mapping.get("address")
        register_type = mapping.get("register_type", "holding")
        param_data_type = mapping.get("data_type", data_type)
        count = 2 if param_data_type in ["uint32", "int32", "float32"] else 1
        
        registers = self.read_register(connection_id, address, register_type, count)
        if registers is None:
            return None
        
        # Decode based on data type
        if param_data_type == "bool":
            return bool(registers) if isinstance(registers, (int, bool)) else bool(registers[0])
        elif param_data_type == "uint16":
            return int(registers) if isinstance(registers, int) else int(registers[0])
        elif param_data_type == "int16":
            decoder = BinaryPayloadDecoder.fromRegisters([registers] if isinstance(registers, int) else registers, byteorder=Endian.BIG, wordorder=Endian.BIG)
            return decoder.decode_16bit_int()
        elif param_data_type == "uint32":
            decoder = BinaryPayloadDecoder.fromRegisters(registers, byteorder=Endian.BIG, wordorder=Endian.BIG)
            return decoder.decode_32bit_uint()
        elif param_data_type == "int32":
            decoder = BinaryPayloadDecoder.fromRegisters(registers, byteorder=Endian.BIG, wordorder=Endian.BIG)
            return decoder.decode_32bit_int()
        elif param_data_type == "float32":
            decoder = BinaryPayloadDecoder.fromRegisters(registers, byteorder=Endian.BIG, wordorder=Endian.BIG)
            return decoder.decode_32bit_float()
        else:
            return registers
    
    def read_all_parameters(self, connection_id: str) -> Dict[str, Any]:
        """
        Read all mapped parameters for a connection.
        
        Args:
            connection_id: Connection identifier
        
        Returns:
            Dictionary of parameter_name: value
        """
        if connection_id not in self.parameter_mappings:
            return {}
        
        result = {}
        for param_name in self.parameter_mappings[connection_id].keys():
            value = self.read_parameter(connection_id, param_name)
            if value is not None:
                result[param_name] = value
        
        return result
    
    def is_connected(self, connection_id: str) -> bool:
        """Check if a connection is active."""
        if connection_id not in self.clients:
            return False
        
        try:
            client = self.clients[connection_id]
            return client.is_socket_open()
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
modbus_service = ModbusService()

