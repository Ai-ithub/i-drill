"""
Control Service
Handles integration with drilling control system for applying parameter changes
"""
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

# Control system configuration from environment variables
CONTROL_SYSTEM_TYPE = os.getenv("CONTROL_SYSTEM_TYPE", "REST").upper()
CONTROL_SYSTEM_URL = os.getenv("CONTROL_SYSTEM_URL", "http://localhost:8080/api/v1")
CONTROL_SYSTEM_TOKEN = os.getenv("CONTROL_SYSTEM_TOKEN", "")
CONTROL_SYSTEM_TIMEOUT = int(os.getenv("CONTROL_SYSTEM_TIMEOUT", "10"))
CONTROL_SYSTEM_ENABLED = os.getenv("CONTROL_SYSTEM_ENABLED", "false").lower() == "true"


class ControlService:
    """
    Service for integrating with drilling control system.
    
    This service provides a unified interface for applying changes to drilling parameters.
    In production, this would integrate with actual PLC/SCADA systems or drilling control APIs.
    
    For now, it provides a mock implementation that can be replaced with actual control
    system integration in the future.
    """
    
    def __init__(self):
        """Initialize ControlService."""
        self.available = CONTROL_SYSTEM_ENABLED or True  # Available even if disabled (for mock mode)
        self.system_type = CONTROL_SYSTEM_TYPE
        self.system_url = CONTROL_SYSTEM_URL
        self.system_token = CONTROL_SYSTEM_TOKEN
        self.timeout = CONTROL_SYSTEM_TIMEOUT
        self.enabled = CONTROL_SYSTEM_ENABLED
        
        # Try to import httpx for REST API calls (optional dependency)
        try:
            import httpx
            self.httpx_available = True
        except ImportError:
            self.httpx_available = False
            logger.warning(
                "httpx not available. REST API integration will not work. "
                "Install with: pip install httpx"
            )
        
        logger.info(
            f"ControlService initialized: type={self.system_type}, "
            f"enabled={self.enabled}, url={self.system_url if self.enabled else 'mock'}"
        )
    
    def apply_parameter_change(
        self,
        rig_id: str,
        component: str,
        parameter: str,
        new_value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply a parameter change to the drilling control system.
        
        This method integrates with the actual drilling control system (PLC, SCADA, etc.)
        to apply parameter changes. In production, this would make API calls or send
        commands to the control system.
        
        Args:
            rig_id: Rig identifier
            component: Component name (e.g., "drilling", "mud_system")
            parameter: Parameter name (e.g., "rpm", "wob", "mud_flow")
            new_value: New value for the parameter
            metadata: Optional metadata (user, timestamp, etc.)
            
        Returns:
            Dictionary containing:
            - success: Boolean indicating if change was applied successfully
            - message: Status message
            - applied_at: Timestamp when change was applied
            - error: Error message if failed
            
        Example:
            ```python
            result = control_service.apply_parameter_change(
                rig_id="RIG_01",
                component="drilling",
                parameter="rpm",
                new_value=120.0,
                metadata={"user": "engineer1", "reason": "optimization"}
            )
            ```
        """
        try:
            logger.info(
                f"Applying parameter change: {component}.{parameter} = {new_value} "
                f"for rig {rig_id}"
            )
            
            # Validate parameter value (safety checks) - always do this first
            validation_result = self._validate_parameter_value(
                rig_id=rig_id,
                component=component,
                parameter=parameter,
                value=new_value
            )
            
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "message": f"Parameter value validation failed: {validation_result['reason']}",
                    "applied_at": None,
                    "error": validation_result["reason"]
                }
            
            # Apply change based on control system type
            if self.enabled and self.system_type == "REST":
                return self._apply_change_rest_api(
                    rig_id=rig_id,
                    component=component,
                    parameter=parameter,
                    new_value=new_value,
                    metadata=metadata
                )
            elif self.enabled and self.system_type == "MQTT":
                return self._apply_change_mqtt(
                    rig_id=rig_id,
                    component=component,
                    parameter=parameter,
                    new_value=new_value,
                    metadata=metadata
                )
            elif self.enabled and self.system_type == "MODBUS":
                return self._apply_change_modbus(
                    rig_id=rig_id,
                    component=component,
                    parameter=parameter,
                    new_value=new_value,
                    metadata=metadata
                )
            else:
                # Mock implementation for development/testing
                return self._apply_change_mock(
                    rig_id=rig_id,
                    component=component,
                    parameter=parameter,
                    new_value=new_value,
                    metadata=metadata
                )
            
        except Exception as e:
            logger.error(
                f"Error applying parameter change for rig {rig_id}, "
                f"component {component}, parameter {parameter}: {e}"
            )
            return {
                "success": False,
                "message": f"Failed to apply parameter change: {str(e)}",
                "applied_at": None,
                "error": str(e)
            }
    
    def _validate_parameter_value(
        self,
        rig_id: str,
        component: str,
        parameter: str,
        value: Any
    ) -> Dict[str, Any]:
        """
        Validate parameter value before applying.
        
        Performs safety checks to ensure parameter values are within acceptable ranges.
        In production, this could include checking against equipment specifications,
        safety limits, and operational constraints.
        
        Args:
            rig_id: Rig identifier
            component: Component name
            parameter: Parameter name
            value: Value to validate
            
        Returns:
            Dictionary with:
            - valid: Boolean indicating if value is valid
            - reason: Reason if invalid
        """
        try:
            # Convert value to float for numeric parameters
            try:
                numeric_value = float(value)
            except (ValueError, TypeError):
                # Non-numeric values might be valid for some parameters
                return {"valid": True, "reason": None}
            
            # Define safety limits for common parameters
            # In production, these should come from equipment specifications or database
            safety_limits = {
                "rpm": {"min": 0, "max": 500, "unit": "RPM"},
                "wob": {"min": 0, "max": 500000, "unit": "lbs"},
                "torque": {"min": 0, "max": 100000, "unit": "ft-lbs"},
                "rop": {"min": 0, "max": 200, "unit": "ft/hr"},
                "mud_flow": {"min": 0, "max": 2000, "unit": "gpm"},
                "mud_pressure": {"min": 0, "max": 10000, "unit": "psi"},
                "depth": {"min": 0, "max": 50000, "unit": "ft"},
            }
            
            param_lower = parameter.lower()
            if param_lower in safety_limits:
                limits = safety_limits[param_lower]
                if numeric_value < limits["min"]:
                    return {
                        "valid": False,
                        "reason": f"{parameter} value {numeric_value} is below minimum {limits['min']} {limits['unit']}"
                    }
                if numeric_value > limits["max"]:
                    return {
                        "valid": False,
                        "reason": f"{parameter} value {numeric_value} exceeds maximum {limits['max']} {limits['unit']}"
                    }
            
            return {"valid": True, "reason": None}
            
        except Exception as e:
            logger.warning(f"Error validating parameter value: {e}")
            # Fail open - allow the change if validation fails
            return {"valid": True, "reason": None}
    
    def get_parameter_value(
        self,
        rig_id: str,
        component: str,
        parameter: str
    ) -> Optional[Any]:
        """
        Get current parameter value from control system.
        
        In production, this would query the actual control system for the current value.
        
        Args:
            rig_id: Rig identifier
            component: Component name
            parameter: Parameter name
            
        Returns:
            Current parameter value, or None if unavailable
        """
        try:
            # Query control system for current value if enabled
            if self.enabled and self.system_type == "REST":
                return self._get_parameter_value_rest_api(
                    rig_id=rig_id,
                    component=component,
                    parameter=parameter
                )
            elif self.enabled and self.system_type == "MQTT":
                return self._get_parameter_value_mqtt(
                    rig_id=rig_id,
                    component=component,
                    parameter=parameter
                )
            elif self.enabled and self.system_type == "MODBUS":
                return self._get_parameter_value_modbus(
                    rig_id=rig_id,
                    component=component,
                    parameter=parameter
                )
            else:
                # Mock mode: return None (will fall back to sensor data)
                logger.debug(
                    f"Control system disabled or mock mode. "
                    f"Returning None for parameter {parameter} (will use sensor data)"
                )
                return None
            
        except Exception as e:
            logger.warning(f"Error getting parameter value from control system: {e}")
            return None
    
    def is_available(self) -> bool:
        """
        Check if control system is available.
        
        Returns:
            True if control system is available and ready, False otherwise
        """
        return self.available
    
    def _apply_change_rest_api(
        self,
        rig_id: str,
        component: str,
        parameter: str,
        new_value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Apply parameter change via REST API."""
        if not self.httpx_available:
            logger.error("httpx not available. Cannot use REST API integration.")
            return {
                "success": False,
                "message": "REST API integration requires httpx library",
                "applied_at": None,
                "error": "httpx not installed"
            }
        
        try:
            import httpx
            import json
            
            # Prepare request
            url = f"{self.system_url}/rigs/{rig_id}/parameters"
            payload = {
                "component": component,
                "parameter": parameter,
                "value": new_value,
                "timestamp": datetime.now().isoformat()
            }
            if metadata:
                payload["metadata"] = metadata
            
            headers = {"Content-Type": "application/json"}
            if self.system_token:
                headers["Authorization"] = f"Bearer {self.system_token}"
            
            # Make API call
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload, headers=headers)
                
                if response.status_code == 200 or response.status_code == 201:
                    result_data = response.json() if response.content else {}
                    applied_at = datetime.now()
                    
                    logger.info(
                        f"Parameter change applied via REST API: "
                        f"rig={rig_id}, component={component}, parameter={parameter}, "
                        f"new_value={new_value}, applied_at={applied_at.isoformat()}"
                    )
                    
                    return {
                        "success": True,
                        "message": f"Parameter {parameter} changed to {new_value} successfully",
                        "applied_at": applied_at.isoformat(),
                        "error": None,
                        "metadata": metadata,
                        "control_system_response": result_data
                    }
                else:
                    error_msg = f"Control system API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return {
                        "success": False,
                        "message": error_msg,
                        "applied_at": None,
                        "error": error_msg
                    }
                    
        except httpx.TimeoutException:
            error_msg = f"Control system API timeout after {self.timeout}s"
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "applied_at": None,
                "error": "timeout"
            }
        except Exception as e:
            error_msg = f"Error calling control system API: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "applied_at": None,
                "error": str(e)
            }
    
    def _apply_change_mqtt(
        self,
        rig_id: str,
        component: str,
        parameter: str,
        new_value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Apply parameter change via MQTT (placeholder for future implementation)."""
        logger.warning("MQTT integration not yet implemented")
        return {
            "success": False,
            "message": "MQTT integration not yet implemented",
            "applied_at": None,
            "error": "Not implemented"
        }
    
    def _apply_change_modbus(
        self,
        rig_id: str,
        component: str,
        parameter: str,
        new_value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Apply parameter change via Modbus (placeholder for future implementation)."""
        logger.warning("Modbus integration not yet implemented")
        return {
            "success": False,
            "message": "Modbus integration not yet implemented",
            "applied_at": None,
            "error": "Not implemented"
        }
    
    def _apply_change_mock(
        self,
        rig_id: str,
        component: str,
        parameter: str,
        new_value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Mock implementation for development/testing."""
        applied_at = datetime.now()
        
        logger.info(
            f"Parameter change applied (mock mode): "
            f"rig={rig_id}, component={component}, parameter={parameter}, "
            f"new_value={new_value}, applied_at={applied_at.isoformat()}"
        )
        
        return {
            "success": True,
            "message": f"Parameter {parameter} changed to {new_value} successfully (mock mode)",
            "applied_at": applied_at.isoformat(),
            "error": None,
            "metadata": metadata
        }
    
    def _get_parameter_value_rest_api(
        self,
        rig_id: str,
        component: str,
        parameter: str
    ) -> Optional[Any]:
        """Get current parameter value from REST API."""
        if not self.httpx_available:
            return None
        
        try:
            import httpx
            
            url = f"{self.system_url}/rigs/{rig_id}/parameters/{component}/{parameter}"
            headers = {}
            if self.system_token:
                headers["Authorization"] = f"Bearer {self.system_token}"
            
            with httpx.Client(timeout=5.0) as client:
                response = client.get(url, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("value")
                else:
                    logger.debug(
                        f"Could not get parameter value from control system: "
                        f"{response.status_code} - {response.text}"
                    )
                    return None
                    
        except Exception as e:
            logger.debug(f"Error getting parameter value from REST API: {e}")
            return None
    
    def _get_parameter_value_mqtt(
        self,
        rig_id: str,
        component: str,
        parameter: str
    ) -> Optional[Any]:
        """Get current parameter value from MQTT (placeholder for future implementation)."""
        logger.debug("MQTT get parameter value not yet implemented")
        return None
    
    def _get_parameter_value_modbus(
        self,
        rig_id: str,
        component: str,
        parameter: str
    ) -> Optional[Any]:
        """Get current parameter value from Modbus (placeholder for future implementation)."""
        logger.debug("Modbus get parameter value not yet implemented")
        return None


# Global singleton instance
control_service = ControlService()

