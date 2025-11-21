"""
Real-time Data Validation & Reconciliation Service
Enhanced DVR service for live data with real-time validation, reconciliation, and alerts
"""
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import deque
import hashlib
import json

from services.dvr_service import dvr_service
from services.data_service import DataService
from services.websocket_manager import websocket_manager
from services.safety_service import safety_service

logger = logging.getLogger(__name__)


class RealTimeDVRService:
    """
    Real-time Data Validation & Reconciliation Service.
    
    Features:
    - Real-time range checking for all parameters
    - Rate of change detection (spike detection)
    - Missing data detection
    - Duplicate data detection
    - LWD/surface data reconciliation
    - Cross-validation between sensors
    - Outlier detection and correction
    - Real-time alerts for driller
    """
    
    def __init__(self):
        """Initialize RealTimeDVRService."""
        self.data_service = DataService()
        
        # Parameter ranges (min, max) for validation
        self.parameter_ranges = {
            "WOB": (0, 500),  # Weight on Bit (tons)
            "RPM": (0, 300),  # Rotary Speed (RPM)
            "Torque": (0, 50000),  # Torque (N.m)
            "ROP": (0, 100),  # Rate of Penetration (m/h)
            "Mud_Flow_Rate": (0, 5000),  # Mud Flow (L/min)
            "Standpipe_Pressure": (0, 500),  # Pressure (bar)
            "Casing_Pressure": (0, 500),  # Pressure (bar)
            "Hook_Load": (0, 1000),  # Hook Load (tons)
            "Block_Position": (0, 50),  # Block Position (m)
            "Power_Consumption": (0, 2000),  # Power (kW)
            "Temperature_Bit": (0, 200),  # Temperature (°C)
            "Temperature_Motor": (0, 150),  # Temperature (°C)
            "Temperature_Surface": (-20, 60),  # Temperature (°C)
            "Gamma_Ray": (0, 200),  # Gamma Ray (API units)
            "Resistivity": (0, 1000),  # Resistivity (ohm-m)
            "Density": (1.5, 3.0),  # Density (g/cc)
            "Porosity": (0, 50),  # Porosity (%)
        }
        
        # Rate of change thresholds (percentage change per second)
        self.rate_of_change_thresholds = {
            "WOB": 10.0,  # 10% per second
            "RPM": 15.0,
            "Torque": 20.0,
            "ROP": 25.0,
            "Mud_Flow_Rate": 15.0,
            "Standpipe_Pressure": 20.0,
            "Casing_Pressure": 20.0,
            "Hook_Load": 10.0,
            "Block_Position": 5.0,
            "Power_Consumption": 15.0,
        }
        
        # History for rate of change detection (per rig_id)
        self.parameter_history: Dict[str, deque] = {}  # rig_id -> deque of recent values
        self.history_size = 10  # Keep last 10 values
        
        # Duplicate detection (hash of data)
        self.recent_hashes: Dict[str, deque] = {}  # rig_id -> deque of recent hashes
        self.duplicate_window_seconds = 5  # Consider duplicate if within 5 seconds
        
        # Missing data tracking
        self.expected_parameters = [
            "WOB", "RPM", "Torque", "ROP", "Mud_Flow_Rate",
            "Standpipe_Pressure", "Casing_Pressure", "Hook_Load"
        ]
        
        # LWD data cache for reconciliation
        self.lwd_data_cache: Dict[str, List[Dict[str, Any]]] = {}  # rig_id -> recent LWD data
        
        logger.info("Real-time DVR service initialized")
    
    def validate_realtime(
        self,
        data: Dict[str, Any],
        previous_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform real-time validation on sensor data.
        
        Args:
            data: Current sensor data record
            previous_data: Previous data record for comparison
        
        Returns:
            Dictionary with validation results:
            - is_valid: Boolean
            - validation_errors: List of error messages
            - warnings: List of warning messages
            - validated_data: Data with corrections applied
        """
        rig_id = data.get("rig_id") or data.get("Rig_ID", "unknown")
        validation_errors = []
        warnings = []
        validated_data = data.copy()
        
        # 1. Range checking
        range_errors = self._check_ranges(data)
        validation_errors.extend(range_errors)
        
        # 2. Missing data detection
        missing_params = self._detect_missing_data(data)
        if missing_params:
            warnings.append(f"Missing parameters: {', '.join(missing_params)}")
        
        # 3. Duplicate detection
        is_duplicate = self._detect_duplicate(data, rig_id)
        if is_duplicate:
            validation_errors.append("Duplicate data detected")
        
        # 4. Rate of change detection (spike detection)
        if previous_data:
            spike_detections = self._detect_spikes(data, previous_data, rig_id)
            warnings.extend(spike_detections)
        else:
            # Use history if available
            spike_detections = self._detect_spikes_from_history(data, rig_id)
            warnings.extend(spike_detections)
        
        # Update history
        self._update_history(data, rig_id)
        
        return {
            "is_valid": len(validation_errors) == 0,
            "validation_errors": validation_errors,
            "warnings": warnings,
            "validated_data": validated_data,
            "rig_id": rig_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def _check_ranges(self, data: Dict[str, Any]) -> List[str]:
        """Check if all parameters are within valid ranges."""
        errors = []
        
        for param_name, (min_val, max_val) in self.parameter_ranges.items():
            # Try different naming conventions
            value = (
                data.get(param_name) or
                data.get(param_name.lower()) or
                data.get(param_name.upper()) or
                data.get(param_name.replace("_", ""))
            )
            
            if value is not None:
                try:
                    value_float = float(value)
                    if value_float < min_val or value_float > max_val:
                        errors.append(
                            f"{param_name} value {value_float} is out of range "
                            f"({min_val} - {max_val})"
                        )
                except (ValueError, TypeError):
                    errors.append(f"{param_name} has invalid value type: {value}")
        
        return errors
    
    def _detect_missing_data(self, data: Dict[str, Any]) -> List[str]:
        """Detect missing expected parameters."""
        missing = []
        
        for param in self.expected_parameters:
            if param not in data and param.lower() not in data and param.upper() not in data:
                missing.append(param)
        
        return missing
    
    def _detect_duplicate(self, data: Dict[str, Any], rig_id: str) -> bool:
        """Detect duplicate data within time window."""
        # Create hash of data (excluding timestamp)
        data_copy = data.copy()
        data_copy.pop("timestamp", None)
        data_copy.pop("id", None)
        data_hash = hashlib.md5(json.dumps(data_copy, sort_keys=True).encode()).hexdigest()
        
        # Check recent hashes
        if rig_id not in self.recent_hashes:
            self.recent_hashes[rig_id] = deque(maxlen=100)
        
        # Check if hash exists in recent history
        if data_hash in self.recent_hashes[rig_id]:
            return True
        
        # Add to history
        self.recent_hashes[rig_id].append(data_hash)
        return False
    
    def _detect_spikes(
        self,
        current_data: Dict[str, Any],
        previous_data: Dict[str, Any],
        rig_id: str
    ) -> List[str]:
        """Detect spikes (rapid rate of change) in parameters."""
        spikes = []
        
        for param_name, threshold in self.rate_of_change_thresholds.items():
            current_val = (
                current_data.get(param_name) or
                current_data.get(param_name.lower()) or
                current_data.get(param_name.upper())
            )
            previous_val = (
                previous_data.get(param_name) or
                previous_data.get(param_name.lower()) or
                previous_data.get(param_name.upper())
            )
            
            if current_val is not None and previous_val is not None:
                try:
                    current_float = float(current_val)
                    previous_float = float(previous_val)
                    
                    if previous_float != 0:
                        rate_of_change = abs((current_float - previous_float) / previous_float) * 100
                        
                        # Assume 1 second between readings
                        rate_per_second = rate_of_change
                        
                        if rate_per_second > threshold:
                            spikes.append(
                                f"{param_name} spike detected: "
                                f"{rate_per_second:.2f}% change (threshold: {threshold}%)"
                            )
                except (ValueError, TypeError):
                    pass
        
        return spikes
    
    def _detect_spikes_from_history(self, data: Dict[str, Any], rig_id: str) -> List[str]:
        """Detect spikes using history if previous data not available."""
        spikes = []
        
        if rig_id not in self.parameter_history:
            return spikes
        
        history = self.parameter_history[rig_id]
        if len(history) < 2:
            return spikes
        
        # Get most recent value from history
        previous_data = history[-1] if history else {}
        
        return self._detect_spikes(data, previous_data, rig_id)
    
    def _update_history(self, data: Dict[str, Any], rig_id: str) -> None:
        """Update parameter history for rate of change detection."""
        if rig_id not in self.parameter_history:
            self.parameter_history[rig_id] = deque(maxlen=self.history_size)
        
        self.parameter_history[rig_id].append(data.copy())
    
    def reconcile_realtime(
        self,
        surface_data: Dict[str, Any],
        lwd_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Reconcile surface and LWD data in real-time.
        
        Args:
            surface_data: Surface sensor data
            lwd_data: Optional LWD data for reconciliation
        
        Returns:
            Reconciled data dictionary
        """
        rig_id = surface_data.get("rig_id") or surface_data.get("Rig_ID", "unknown")
        reconciled = surface_data.copy()
        
        # 1. Cross-validation between sensors
        cross_validation_results = self._cross_validate_sensors(surface_data)
        if cross_validation_results.get("issues"):
            logger.warning(f"Cross-validation issues for rig {rig_id}: {cross_validation_results['issues']}")
            # Apply corrections
            for correction in cross_validation_results.get("corrections", []):
                param = correction.get("parameter")
                corrected_value = correction.get("corrected_value")
                if param and corrected_value is not None:
                    reconciled[param] = corrected_value
        
        # 2. LWD/Surface reconciliation
        if lwd_data:
            lwd_reconciliation = self._reconcile_lwd_surface(surface_data, lwd_data)
            reconciled.update(lwd_reconciliation.get("reconciled_data", {}))
        
        # 3. Outlier detection and correction
        outlier_results = self._detect_and_correct_outliers(reconciled, rig_id)
        if outlier_results.get("outliers_detected"):
            for correction in outlier_results.get("corrections", []):
                param = correction.get("parameter")
                corrected_value = correction.get("corrected_value")
                if param and corrected_value is not None:
                    reconciled[param] = corrected_value
        
        return reconciled
    
    def _cross_validate_sensors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate between related sensors."""
        issues = []
        corrections = []
        
        # Example: Validate WOB vs Hook Load relationship
        wob = data.get("WOB") or data.get("wob")
        hook_load = data.get("Hook_Load") or data.get("hook_load")
        
        if wob is not None and hook_load is not None:
            try:
                wob_float = float(wob)
                hook_load_float = float(hook_load)
                
                # WOB should be less than Hook Load
                if wob_float > hook_load_float * 1.1:  # Allow 10% tolerance
                    issues.append("WOB exceeds Hook Load (physically impossible)")
                    # Suggest correction: use Hook Load as upper bound
                    corrections.append({
                        "parameter": "WOB",
                        "original_value": wob_float,
                        "corrected_value": hook_load_float * 0.95,
                        "reason": "WOB cannot exceed Hook Load"
                    })
            except (ValueError, TypeError):
                pass
        
        # Example: Validate ROP vs RPM relationship
        rop = data.get("ROP") or data.get("rop")
        rpm = data.get("RPM") or data.get("rpm")
        
        if rop is not None and rpm is not None:
            try:
                rop_float = float(rop)
                rpm_float = float(rpm)
                
                # ROP should correlate with RPM (rough check)
                if rpm_float > 50 and rop_float < 0.1:
                    issues.append("Low ROP despite high RPM (possible stuck pipe)")
            except (ValueError, TypeError):
                pass
        
        return {
            "issues": issues,
            "corrections": corrections
        }
    
    def _reconcile_lwd_surface(
        self,
        surface_data: Dict[str, Any],
        lwd_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Reconcile LWD data with surface data."""
        reconciled = surface_data.copy()
        
        # Match by depth
        surface_depth = surface_data.get("depth") or surface_data.get("Depth")
        lwd_depth = lwd_data.get("depth") or lwd_data.get("Depth")
        
        if surface_depth is not None and lwd_depth is not None:
            try:
                depth_diff = abs(float(surface_depth) - float(lwd_depth))
                
                # If depths match (within 1 meter), merge LWD data
                if depth_diff < 1.0:
                    # Add LWD parameters to reconciled data
                    lwd_params = ["Gamma_Ray", "Resistivity", "Density", "Porosity"]
                    for param in lwd_params:
                        lwd_value = lwd_data.get(param) or lwd_data.get(param.lower())
                        if lwd_value is not None:
                            reconciled[param] = lwd_value
                else:
                    logger.warning(
                        f"Depth mismatch: surface={surface_depth}, LWD={lwd_depth}, "
                        f"diff={depth_diff}m"
                    )
            except (ValueError, TypeError):
                pass
        
        return {
            "reconciled_data": reconciled,
            "depth_match": depth_diff < 1.0 if surface_depth and lwd_depth else False
        }
    
    def _detect_and_correct_outliers(
        self,
        data: Dict[str, Any],
        rig_id: str
    ) -> Dict[str, Any]:
        """Detect and correct outliers using statistical methods."""
        outliers_detected = []
        corrections = []
        
        # Get recent history for statistical analysis
        if rig_id in self.parameter_history and len(self.parameter_history[rig_id]) >= 5:
            history = list(self.parameter_history[rig_id])
            
            for param_name in self.expected_parameters:
                param_value = (
                    data.get(param_name) or
                    data.get(param_name.lower()) or
                    data.get(param_name.upper())
                )
                
                if param_value is None:
                    continue
                
                try:
                    value_float = float(param_value)
                    
                    # Get historical values for this parameter
                    historical_values = []
                    for record in history:
                        hist_value = (
                            record.get(param_name) or
                            record.get(param_name.lower()) or
                            record.get(param_name.upper())
                        )
                        if hist_value is not None:
                            try:
                                historical_values.append(float(hist_value))
                            except (ValueError, TypeError):
                                pass
                    
                    if len(historical_values) >= 5:
                        import statistics
                        mean = statistics.mean(historical_values)
                        stdev = statistics.stdev(historical_values) if len(historical_values) > 1 else 0
                        
                        # Detect outlier (more than 3 standard deviations from mean)
                        if stdev > 0 and abs(value_float - mean) > 3 * stdev:
                            outliers_detected.append(param_name)
                            # Correct to mean value
                            corrections.append({
                                "parameter": param_name,
                                "original_value": value_float,
                                "corrected_value": mean,
                                "reason": f"Outlier detected: {value_float:.2f} vs mean {mean:.2f} (±{3*stdev:.2f})"
                            })
                except (ValueError, TypeError):
                    pass
        
        return {
            "outliers_detected": outliers_detected,
            "corrections": corrections
        }
    
    async def process_and_alert(
        self,
        data: Dict[str, Any],
        lwd_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process data through real-time DVR and send alerts if needed.
        
        Args:
            data: Sensor data to process
            lwd_data: Optional LWD data for reconciliation
        
        Returns:
            Processing result with alerts
        """
        rig_id = data.get("rig_id") or data.get("Rig_ID", "unknown")
        
        # Get previous data for comparison
        previous_data = None
        if rig_id in self.parameter_history and len(self.parameter_history[rig_id]) > 0:
            previous_data = self.parameter_history[rig_id][-1]
        
        # 1. Real-time validation
        validation_result = self.validate_realtime(data, previous_data)
        
        # 2. Reconciliation
        reconciled_data = self.reconcile_realtime(
            validation_result["validated_data"],
            lwd_data
        )
        
        # 3. Send alerts if needed
        alerts_sent = []
        
        if validation_result["validation_errors"]:
            # Critical errors - send alert immediately
            alert_message = {
                "type": "validation_error",
                "severity": "critical",
                "rig_id": rig_id,
                "errors": validation_result["validation_errors"],
                "timestamp": datetime.now().isoformat()
            }
            await websocket_manager.send_to_rig(rig_id, {
                "message_type": "dvr_alert",
                "data": alert_message
            })
            alerts_sent.append(alert_message)
            logger.warning(f"Validation errors for rig {rig_id}: {validation_result['validation_errors']}")
        
        if validation_result["warnings"]:
            # Warnings - send alert
            alert_message = {
                "type": "validation_warning",
                "severity": "warning",
                "rig_id": rig_id,
                "warnings": validation_result["warnings"],
                "timestamp": datetime.now().isoformat()
            }
            await websocket_manager.send_to_rig(rig_id, {
                "message_type": "dvr_alert",
                "data": alert_message
            })
            alerts_sent.append(alert_message)
        
        # 4. Log discrepancies
        if validation_result["validation_errors"] or validation_result["warnings"]:
            self._log_discrepancy(rig_id, data, validation_result, reconciled_data)
        
        return {
            "success": validation_result["is_valid"],
            "validated_data": reconciled_data,
            "validation_result": validation_result,
            "alerts_sent": alerts_sent,
            "rig_id": rig_id
        }
    
    def _log_discrepancy(
        self,
        rig_id: str,
        original_data: Dict[str, Any],
        validation_result: Dict[str, Any],
        reconciled_data: Dict[str, Any]
    ) -> None:
        """Log all discrepancies for audit."""
        discrepancy_log = {
            "rig_id": rig_id,
            "timestamp": datetime.now().isoformat(),
            "original_data": original_data,
            "validation_errors": validation_result.get("validation_errors", []),
            "warnings": validation_result.get("warnings", []),
            "reconciled_data": reconciled_data,
            "corrections_applied": len(validation_result.get("warnings", [])) > 0
        }
        
        logger.warning(f"DVR Discrepancy for rig {rig_id}: {json.dumps(discrepancy_log, indent=2)}")
        
        # Also persist to DVR history
        try:
            dvr_service.process_record(original_data, source="realtime_dvr")
        except Exception as e:
            logger.error(f"Error logging to DVR history: {e}")


# Global instance
realtime_dvr_service = RealTimeDVRService()

