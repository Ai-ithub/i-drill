"""
Protocol Adapter for Data Transformation
Converts rig data from various protocols to standard system format
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ProtocolType(str, Enum):
    """Supported protocol types"""
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    OPCUA = "opcua"
    MQTT = "mqtt"
    KAFKA = "kafka"  # Internal protocol


class ProtocolAdapter:
    """
    Protocol adapter for transforming rig data from various protocols
    to the standard system data format.
    
    Features:
    - Data format conversion
    - Initial data validation
    - Unit conversion
    - Parameter mapping
    """
    
    def __init__(self, rig_id: str):
        """
        Initialize protocol adapter.
        
        Args:
            rig_id: Rig identifier
        """
        self.rig_id = rig_id
        
        # Parameter mapping from various names to standard names
        self.parameter_mapping = self._get_parameter_mapping()
        
        # Unit conversion factors (if needed)
        self.unit_conversions = self._get_unit_conversions()
    
    def _get_parameter_mapping(self) -> Dict[str, str]:
        """Get parameter name mapping from various formats to standard"""
        return {
            # Standard names (no change)
            "wob": "wob",
            "WOB": "wob",
            "weight_on_bit": "wob",
            "WeightOnBit": "wob",
            
            "rpm": "rpm",
            "RPM": "rpm",
            "rotary_speed": "rpm",
            "RotarySpeed": "rpm",
            
            "torque": "torque",
            "Torque": "torque",
            
            "rop": "rop",
            "ROP": "rop",
            "rate_of_penetration": "rop",
            "RateOfPenetration": "rop",
            
            "mud_flow": "mud_flow",
            "mud_flow_rate": "mud_flow",
            "MudFlow": "mud_flow",
            "MudFlowRate": "mud_flow",
            
            "mud_pressure": "mud_pressure",
            "standpipe_pressure": "mud_pressure",
            "MudPressure": "mud_pressure",
            "StandpipePressure": "mud_pressure",
            
            "hook_load": "hook_load",
            "HookLoad": "hook_load",
            
            "depth": "depth",
            "Depth": "depth",
            
            "pump_status": "pump_status",
            "PumpStatus": "pump_status",
            
            "power_consumption": "power_consumption",
            "PowerConsumption": "power_consumption",
            
            "bit_temperature": "bit_temperature",
            "BitTemperature": "bit_temperature",
            
            "motor_temperature": "motor_temperature",
            "MotorTemperature": "motor_temperature",
            
            "casing_pressure": "casing_pressure",
            "CasingPressure": "casing_pressure",
            
            "block_position": "block_position",
            "BlockPosition": "block_position",
            
            "mud_temperature": "mud_temperature",
            "MudTemperature": "mud_temperature",
            
            "gamma_ray": "gamma_ray",
            "GammaRay": "gamma_ray",
            
            "resistivity": "resistivity",
            "Resistivity": "resistivity",
            
            "density": "density",
            "Density": "density",
            
            "porosity": "porosity",
            "Porosity": "porosity",
            
            "vibration": "vibration",
            "Vibration": "vibration",
        }
    
    def _get_unit_conversions(self) -> Dict[str, Dict[str, float]]:
        """Get unit conversion factors"""
        return {
            # Example conversions (adjust based on actual requirements)
            "wob": {
                "kg_to_lbs": 2.20462,
                "tons_to_lbs": 2000.0,
            },
            "depth": {
                "meters_to_feet": 3.28084,
            },
            "mud_flow": {
                "lpm_to_gpm": 0.264172,
            },
            "mud_pressure": {
                "bar_to_psi": 14.5038,
                "kpa_to_psi": 0.145038,
            },
            "torque": {
                "nm_to_ftlbs": 0.737562,
            },
        }
    
    def transform(self, raw_data: Dict[str, Any], protocol: ProtocolType) -> Dict[str, Any]:
        """
        Transform raw data from protocol to standard format.
        
        Args:
            raw_data: Raw data dictionary from protocol
            protocol: Protocol type
            
        Returns:
            Transformed data in standard format
        """
        try:
            # Start with base structure
            transformed = {
                "rig_id": raw_data.get("rig_id") or self.rig_id,
                "timestamp": self._parse_timestamp(raw_data.get("timestamp")),
                "protocol": protocol.value,
            }
            
            # Map and transform parameters
            for key, value in raw_data.items():
                # Skip metadata fields
                if key in ["rig_id", "timestamp", "protocol", "topic"]:
                    continue
                
                # Map parameter name to standard
                standard_name = self.parameter_mapping.get(key, key.lower())
                
                # Convert value if needed
                converted_value = self._convert_value(standard_name, value, raw_data)
                
                # Validate value
                if self._validate_value(standard_name, converted_value):
                    transformed[standard_name] = converted_value
                else:
                    logger.warning(f"Invalid value for {standard_name}: {converted_value}")
            
            # Ensure required fields have defaults
            transformed = self._ensure_required_fields(transformed)
            
            return transformed
            
        except Exception as e:
            logger.error(f"Error transforming data: {e}")
            # Return minimal valid structure
            return {
                "rig_id": raw_data.get("rig_id") or self.rig_id,
                "timestamp": datetime.now().isoformat(),
                "protocol": protocol.value,
                "status": "error",
            }
    
    def _parse_timestamp(self, timestamp: Any) -> str:
        """Parse timestamp from various formats"""
        if timestamp is None:
            return datetime.now().isoformat()
        
        if isinstance(timestamp, datetime):
            return timestamp.isoformat()
        
        if isinstance(timestamp, str):
            try:
                # Try ISO format
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return dt.isoformat()
            except:
                try:
                    # Try common formats
                    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
                        try:
                            dt = datetime.strptime(timestamp, fmt)
                            return dt.isoformat()
                        except:
                            continue
                except:
                    pass
        
        return datetime.now().isoformat()
    
    def _convert_value(self, param_name: str, value: Any, context: Dict[str, Any]) -> Any:
        """Convert value with unit conversion if needed"""
        if not isinstance(value, (int, float)):
            return value
        
        # Apply unit conversions if needed
        # (This is a simplified version - can be extended based on requirements)
        conversions = self.unit_conversions.get(param_name, {})
        
        # Check for unit hints in context
        unit = context.get(f"{param_name}_unit") or context.get("unit")
        
        if unit and unit in conversions:
            return float(value) * conversions[unit]
        
        return float(value)
    
    def _validate_value(self, param_name: str, value: Any) -> bool:
        """Validate parameter value"""
        if value is None:
            return False
        
        # Basic range checks
        validation_rules = {
            "wob": {"min": 0, "max": 1000000},  # lbs
            "rpm": {"min": 0, "max": 300},
            "torque": {"min": 0, "max": 1000000},  # ft-lbs
            "rop": {"min": 0, "max": 1000},  # ft/hr
            "mud_flow": {"min": 0, "max": 10000},  # gpm
            "mud_pressure": {"min": 0, "max": 10000},  # psi
            "depth": {"min": 0, "max": 50000},  # feet
            "hook_load": {"min": 0, "max": 2000000},  # lbs
            "porosity": {"min": 0, "max": 100},  # percentage
        }
        
        if param_name in validation_rules:
            rule = validation_rules[param_name]
            if isinstance(value, (int, float)):
                if value < rule["min"] or value > rule["max"]:
                    return False
        
        return True
    
    def _ensure_required_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure required fields are present with defaults"""
        # Required fields with defaults
        required_fields = {
            "depth": 0.0,
            "wob": 0.0,
            "rpm": 0.0,
            "torque": 0.0,
            "rop": 0.0,
            "mud_flow": 0.0,
            "mud_pressure": 0.0,
            "status": "normal",
        }
        
        for field, default in required_fields.items():
            if field not in data:
                data[field] = default
        
        return data

