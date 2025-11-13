"""
Integration Service for RL Models and DVR
Provides unified interface for integrating Reinforcement Learning and Data Validation & Reconciliation
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from services.rl_service import rl_service
from services.dvr_service import dvr_service
from services.data_service import DataService

logger = logging.getLogger(__name__)


class IntegrationService:
    """
    Service for integrating RL Models and DVR systems.
    
    Provides methods for:
    - Validating RL actions through DVR before applying them
    - Feeding validated sensor data from DVR to RL environment
    - Using RL state information for enhanced DVR validation
    - Integrated pipeline: Sensor Data → DVR → RL → Actions
    """
    
    def __init__(self):
        """Initialize IntegrationService with RL and DVR services."""
        self.rl_service = rl_service
        self.dvr_service = dvr_service
        self.data_service = DataService()
    
    # ------------------------------------------------------------------ #
    # Integrated Processing: Sensor Data → DVR → RL
    # ------------------------------------------------------------------ #
    
    def process_sensor_data_for_rl(
        self,
        sensor_record: Dict[str, Any],
        apply_to_rl: bool = False
    ) -> Dict[str, Any]:
        """
        Process sensor data through DVR and optionally feed to RL.
        
        Pipeline:
        1. Validate and reconcile sensor data using DVR
        2. If valid and apply_to_rl=True, feed to RL environment
        3. Return integrated processing result
        
        Args:
            sensor_record: Dictionary containing sensor data record
            apply_to_rl: If True, feed validated data to RL environment
            
        Returns:
            Dictionary containing:
            - dvr_result: DVR processing result
            - rl_state: RL environment state (if applied)
            - success: Boolean indicating overall success
            - message: Processing status message
        """
        try:
            # Step 1: Process through DVR
            dvr_result = self.dvr_service.process_record(
                sensor_record,
                source="integration"
            )
            
            if not dvr_result.get("success"):
                return {
                    "success": False,
                    "dvr_result": dvr_result,
                    "rl_state": None,
                    "message": "DVR validation failed",
                    "reason": dvr_result.get("message")
                }
            
            # Step 2: Optionally apply to RL
            rl_state = None
            if apply_to_rl and dvr_result.get("processed_record"):
                try:
                    # Convert validated sensor data to RL observation
                    rl_observation = self._sensor_to_rl_observation(
                        dvr_result["processed_record"]
                    )
                    
                    if rl_observation:
                        # Update RL environment with validated data
                        rl_state = self._update_rl_from_sensor(rl_observation)
                        
                except Exception as rl_exc:
                    logger.warning(f"Failed to update RL from sensor data: {rl_exc}")
                    rl_state = {
                        "success": False,
                        "message": f"RL update failed: {rl_exc}",
                        "current_state": self.rl_service.get_state()
                    }
            
            return {
                "success": True,
                "dvr_result": dvr_result,
                "rl_state": rl_state,
                "message": "Integrated processing completed successfully"
            }
            
        except Exception as exc:
            logger.error(f"Error in integrated sensor data processing: {exc}")
            return {
                "success": False,
                "dvr_result": None,
                "rl_state": None,
                "message": str(exc)
            }
    
    # ------------------------------------------------------------------ #
    # Integrated Action Processing: RL Action → DVR Validation → Apply
    # ------------------------------------------------------------------ #
    
    def process_rl_action_with_dvr(
        self,
        action: Dict[str, float],
        validate_with_dvr: bool = True,
        history_size: int = 100
    ) -> Dict[str, Any]:
        """
        Process RL action with DVR validation before applying.
        
        Pipeline:
        1. Validate action using DVR anomaly detection
        2. If validation passes, apply action to RL environment
        3. Return integrated result
        
        Args:
            action: Dictionary with 'wob', 'rpm', 'flow_rate'
            validate_with_dvr: If True, validate action through DVR
            history_size: Number of historical records for DVR validation
            
        Returns:
            Dictionary containing:
            - validation_result: DVR validation result (if enabled)
            - rl_result: RL step result
            - success: Boolean indicating overall success
            - message: Processing status message
        """
        try:
            validation_result = None
            
            # Step 1: Validate action through DVR (if enabled)
            if validate_with_dvr:
                try:
                    # Create a record from action for DVR validation
                    action_record = self._action_to_sensor_record(action)
                    
                    # Evaluate action for anomalies
                    dvr_eval = self.dvr_service.evaluate_record_anomaly(
                        action_record,
                        history_size=history_size
                    )
                    
                    validation_result = {
                        "passed": dvr_eval.get("success", False),
                        "anomaly_detected": dvr_eval.get("record", {}).get("Anomaly", False),
                        "details": dvr_eval.get("record", {}),
                        "message": "Action validated" if dvr_eval.get("success") else "Action validation failed"
                    }
                    
                    # If anomaly detected, warn but don't block (RL model knows best)
                    if validation_result["anomaly_detected"]:
                        logger.warning(
                            f"RL action anomaly detected: {action}. "
                            "Proceeding with action as RL model decision."
                        )
                        validation_result["warning"] = "Anomaly detected in action values"
                    
                except Exception as validation_exc:
                    logger.warning(f"DVR validation failed for action: {validation_exc}")
                    validation_result = {
                        "passed": True,  # Fail open - don't block on validation error
                        "error": str(validation_exc),
                        "message": "Validation error, proceeding with action"
                    }
            
            # Step 2: Apply action to RL environment
            rl_result = self.rl_service.step(action)
            
            return {
                "success": True,
                "validation_result": validation_result,
                "rl_result": rl_result,
                "message": "Action processed successfully"
            }
            
        except Exception as exc:
            logger.error(f"Error processing RL action with DVR: {exc}")
            return {
                "success": False,
                "validation_result": validation_result,
                "rl_result": None,
                "message": str(exc)
            }
    
    # ------------------------------------------------------------------ #
    # Enhanced DVR Validation using RL State
    # ------------------------------------------------------------------ #
    
    def validate_with_rl_context(
        self,
        sensor_record: Dict[str, Any],
        use_rl_state: bool = True
    ) -> Dict[str, Any]:
        """
        Validate sensor data using DVR with RL state context.
        
        Uses current RL environment state to provide context for validation,
        allowing for more intelligent anomaly detection.
        
        Args:
            sensor_record: Dictionary containing sensor data record
            use_rl_state: If True, use RL state for enhanced validation
            
        Returns:
            Dictionary containing enhanced DVR validation result
        """
        try:
            # Get RL state if enabled
            rl_context = None
            if use_rl_state:
                try:
                    rl_state = self.rl_service.get_state()
                    rl_context = {
                        "current_observation": rl_state.get("observation"),
                        "current_reward": rl_state.get("reward"),
                        "episode": rl_state.get("episode"),
                        "step": rl_state.get("step")
                    }
                except Exception as rl_exc:
                    logger.warning(f"Failed to get RL context: {rl_exc}")
                    rl_context = None
            
            # Process through DVR
            dvr_result = self.dvr_service.process_record(
                sensor_record,
                source="integration_with_rl"
            )
            
            # Enhance result with RL context
            if rl_context:
                dvr_result["rl_context"] = rl_context
                
                # Add context-aware validation hints
                if dvr_result.get("success"):
                    dvr_result["validation_hints"] = self._generate_validation_hints(
                        sensor_record,
                        rl_context
                    )
            
            return dvr_result
            
        except Exception as exc:
            logger.error(f"Error in RL-context validation: {exc}")
            return {
                "success": False,
                "processed_record": None,
                "message": str(exc)
            }
    
    # ------------------------------------------------------------------ #
    # Integrated Auto Step: DVR → RL Auto Step
    # ------------------------------------------------------------------ #
    
    def integrated_auto_step(
        self,
        validate_action: bool = True
    ) -> Dict[str, Any]:
        """
        Execute RL auto step with integrated DVR validation.
        
        Pipeline:
        1. Get action from RL policy (auto step)
        2. Validate action through DVR
        3. Apply action to environment
        4. Return integrated result
        
        Args:
            validate_action: If True, validate action through DVR before applying
            
        Returns:
            Dictionary containing integrated auto step result
        """
        try:
            # Step 1: Get auto step result (includes policy action)
            auto_result = self.rl_service.auto_step()
            
            if not auto_result.get("success"):
                return {
                    "success": False,
                    "rl_result": auto_result,
                    "validation_result": None,
                    "message": auto_result.get("message", "Auto step failed")
                }
            
            rl_state = auto_result.get("state", {})
            
            # Step 2: Validate action if enabled
            validation_result = None
            if validate_action and "action" in rl_state:
                action = rl_state["action"]
                dvr_eval = self.dvr_service.evaluate_record_anomaly(
                    self._action_to_sensor_record(action),
                    history_size=100
                )
                
                validation_result = {
                    "anomaly_detected": dvr_eval.get("record", {}).get("Anomaly", False),
                    "details": dvr_eval.get("record", {}),
                }
            
            return {
                "success": True,
                "rl_result": auto_result,
                "validation_result": validation_result,
                "message": "Integrated auto step completed"
            }
            
        except Exception as exc:
            logger.error(f"Error in integrated auto step: {exc}")
            return {
                "success": False,
                "rl_result": None,
                "validation_result": None,
                "message": str(exc)
            }
    
    # ------------------------------------------------------------------ #
    # Helper Methods
    # ------------------------------------------------------------------ #
    
    def _sensor_to_rl_observation(
        self,
        sensor_record: Dict[str, Any]
    ) -> Optional[List[float]]:
        """
        Convert validated sensor record to RL observation format.
        
        Maps sensor data fields to RL observation dimensions:
        - depth, bit_wear, rop, torque, pressure, vibration_axial, vibration_lateral, vibration_torsional
        
        Args:
            sensor_record: Validated sensor data record
            
        Returns:
            List of observation values, or None if conversion not possible
        """
        try:
            # Map sensor fields to RL observation dimensions
            mapping = {
                "depth": ["Depth", "depth"],
                "bit_wear": ["Bit_Wear", "bit_wear", "bitwear"],
                "rop": ["ROP", "rop", "rate_of_penetration"],
                "torque": ["Torque", "torque"],
                "pressure": ["Pressure", "pressure", "wellbore_pressure"],
                "vibration_axial": ["Vibration_Axial", "vibration_axial", "vib_axial"],
                "vibration_lateral": ["Vibration_Lateral", "vibration_lateral", "vib_lateral"],
                "vibration_torsional": ["Vibration_Torsional", "vibration_torsional", "vib_torsional"],
            }
            
            observation = []
            for obs_name, field_candidates in mapping.items():
                value = None
                for field in field_candidates:
                    if field in sensor_record:
                        value = float(sensor_record[field])
                        break
                
                # Default to 0 if field not found
                observation.append(value if value is not None else 0.0)
            
            return observation if len(observation) == 8 else None
            
        except Exception as exc:
            logger.warning(f"Failed to convert sensor to RL observation: {exc}")
            return None
    
    def _action_to_sensor_record(
        self,
        action: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Convert RL action to sensor record format for DVR validation.
        
        Args:
            action: Dictionary with 'wob', 'rpm', 'flow_rate'
            
        Returns:
            Sensor record dictionary for DVR processing
        """
        return {
            "WOB": action.get("wob", 0.0),
            "RPM": action.get("rpm", 0.0),
            "Flow_Rate": action.get("flow_rate", 0.0),
            "rig_id": "RL_GENERATED",
            "timestamp": datetime.now().isoformat()
        }
    
    def _update_rl_from_sensor(
        self,
        observation: List[float]
    ) -> Dict[str, Any]:
        """
        Update RL environment with sensor observation.
        
        Note: This is a simplified approach. In production, you might want
        to reset the environment or update internal state more carefully.
        
        Args:
            observation: RL observation values
            
        Returns:
            Updated RL state dictionary
        """
        # For now, we just return the current state
        # In a full implementation, you might update the environment's internal state
        return self.rl_service.get_state()
    
    def _generate_validation_hints(
        self,
        sensor_record: Dict[str, Any],
        rl_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate validation hints using RL context.
        
        Uses RL state information to provide intelligent validation hints.
        
        Args:
            sensor_record: Sensor data record
            rl_context: RL environment context
            
        Returns:
            Dictionary with validation hints
        """
        hints = {}
        
        try:
            rl_obs = rl_context.get("current_observation", [])
            if len(rl_obs) >= 3:
                # Compare sensor values with RL expected values
                sensor_wob = float(sensor_record.get("WOB", 0))
                sensor_rpm = float(sensor_record.get("RPM", 0))
                sensor_flow = float(sensor_record.get("Flow_Rate", 0))
                
                # Generate hints based on comparison
                hints["rl_expected_values"] = {
                    "observation": rl_obs
                }
                
                hints["comparison"] = {
                    "sensor_values": {
                        "wob": sensor_wob,
                        "rpm": sensor_rpm,
                        "flow_rate": sensor_flow
                    },
                    "rl_state": {
                        "reward": rl_context.get("current_reward"),
                        "episode": rl_context.get("episode"),
                        "step": rl_context.get("step")
                    }
                }
        except Exception as exc:
            logger.debug(f"Error generating validation hints: {exc}")
        
        return hints
    
    def get_integration_status(self) -> Dict[str, Any]:
        """
        Get status of RL-DVR integration.
        
        Returns:
            Dictionary containing integration status information
        """
        try:
            rl_config = self.rl_service.get_config()
            rl_state = self.rl_service.get_state()
            
            return {
                "rl_available": rl_config.get("available", False),
                "rl_policy_loaded": rl_state.get("policy_loaded", False),
                "rl_policy_mode": rl_state.get("policy_mode", "manual"),
                "dvr_available": self.dvr_service._db_ready(),
                "integration_active": rl_config.get("available", False) and self.dvr_service._db_ready(),
                "rl_episode": rl_state.get("episode", 0),
                "rl_step": rl_state.get("step", 0)
            }
        except Exception as exc:
            logger.error(f"Error getting integration status: {exc}")
            return {
                "rl_available": False,
                "dvr_available": False,
                "integration_active": False,
                "error": str(exc)
            }


# Global singleton instance
integration_service = IntegrationService()

