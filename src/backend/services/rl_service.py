from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from services.mlflow_service import mlflow_service  # type: ignore

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

try:
    from drilling_env.drilling_env import DrillingEnv, MAX_WOB, MAX_RPM, MAX_FLOW_RATE

    RL_AVAILABLE = True
except ImportError:  # pragma: no cover - environment optional
    DrillingEnv = None  # type: ignore
    MAX_WOB = MAX_RPM = MAX_FLOW_RATE = 0  # type: ignore
    RL_AVAILABLE = False


DEFAULT_OBS_SIZE = 8
AUTO_STEP_MIN_INTERVAL = 0.5  # seconds


class _PolicyWrapper:
    """
    Wraps loaded policy models to provide a unified action interface.
    
    Handles different policy model types (Stable-Baselines3, PyTorch models)
    and provides a consistent interface for action prediction with bounds clipping.
    
    Attributes:
        model: Loaded policy model (Stable-Baselines3 or PyTorch)
        action_size: Number of action dimensions
        action_bounds: Dictionary mapping action names to min/max bounds
    """

    def __init__(self, model, action_size: int, action_bounds: Dict[str, Dict[str, float]]):
        """
        Initialize policy wrapper.
        
        Args:
            model: Policy model to wrap (Stable-Baselines3 or PyTorch model)
            action_size: Number of action dimensions
            action_bounds: Dictionary with action bounds, e.g. {"wob": {"min": 0, "max": 50000}}
        """
        self.model = model
        self.action_size = action_size
        self.action_bounds = action_bounds

    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Get action from policy model given observation.
        
        Handles both Stable-Baselines3 models (with predict method) and
        raw PyTorch models. Clips actions to specified bounds.
        
        Args:
            observation: Current environment observation array
            
        Returns:
            Action array clipped to bounds
            
        Raises:
            RuntimeError: If PyTorch is not available
            ValueError: If action size doesn't match expected size
        """
        if not TORCH_AVAILABLE or torch is None:
            raise RuntimeError("PyTorch not available for policy inference")

        if observation.ndim == 1:
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        else:
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32)

        with torch.no_grad():
            if hasattr(self.model, "predict"):
                action, _ = self.model.predict(observation, deterministic=True)  # type: ignore[attr-defined]
                action_array = np.array(action, dtype=np.float32).reshape(-1)
            else:
                output = self.model(observation_tensor)  # type: ignore[call-arg]
                if isinstance(output, tuple):
                    output = output[0]
                if hasattr(output, "detach"):
                    output = output.detach()
                action_array = output.squeeze().cpu().numpy().astype(np.float32)

        if action_array.size != self.action_size:
            raise ValueError(f"Policy returned action of size {action_array.size}, expected {self.action_size}")

        clipped = []
        for idx, key in enumerate(["wob", "rpm", "flow_rate"][: self.action_size]):
            bounds = self.action_bounds.get(key, {"min": float("-inf"), "max": float("inf")})
            clipped.append(float(np.clip(action_array[idx], bounds["min"], bounds["max"])))
        return np.array(clipped, dtype=np.float32)


class RLService:
    """
    Service wrapper around the drilling reinforcement learning environment.
    
    Provides a service interface for interacting with the DrillingEnv RL environment,
    including manual control, automatic policy-based control, episode management,
    and history tracking.
    
    Attributes:
        _lock: Thread lock for thread-safe operations
        _history: List of episode history dictionaries
        _episode_index: Current episode index
        _step_index: Current step index within episode
        _last_reward: Last reward received
        _last_done: Last done flag
        _last_info: Last info dictionary from environment
        _policy_mode: Current policy mode ("manual" or "auto")
        _policy_wrapper: Wrapped policy model for automatic control
        _policy_metadata: Metadata about loaded policy
        _auto_interval_seconds: Interval between auto steps in seconds
        _last_auto_step: Timestamp of last auto step
        _env: DrillingEnv instance (if available)
        _current_observation: Current environment observation
    """

    def __init__(self):
        """
        Initialize RLService.
        
        Sets up the service with default state and initializes the drilling
        environment if available. Starts in manual control mode.
        """
        self._lock = threading.Lock()
        self._history: List[Dict[str, Any]] = []
        self._episode_index = 0
        self._step_index = 0
        self._last_reward = 0.0
        self._last_done = False
        self._last_info: Dict[str, Any] = {}
        self._policy_mode: Literal["manual", "auto"] = "manual"
        self._policy_wrapper: Optional[_PolicyWrapper] = None
        self._policy_metadata: Dict[str, Any] = {
            "loaded": False,
            "source": None,
            "identifier": None,
            "stage": None,
            "loaded_at": None,
            "message": None,
        }
        self._auto_interval_seconds: float = 1.0
        self._last_auto_step: Optional[float] = None

        if RL_AVAILABLE and DrillingEnv is not None:
            self._env = DrillingEnv()
            self._current_observation = self._env.reset()
        else:
            self._env = None
            self._current_observation = np.zeros(DEFAULT_OBS_SIZE, dtype=np.float32)
            self._last_info = {"warning": "Reinforcement learning environment unavailable"}

    # ------------------------------------------------------------------ #
    # Environment control
    # ------------------------------------------------------------------ #
    def reset(self, random_init: bool = False) -> Dict[str, Any]:
        """
        Reset the RL environment to initial state.
        
        Resets the drilling environment, clears episode history, increments
        episode index, and resets step counter. Optionally randomizes initial
        formation and angle.
        
        Args:
            random_init: If True, randomizes initial formation type and drilling angle
            
        Returns:
            Dictionary containing current environment state after reset
        """
        with self._lock:
            if not RL_AVAILABLE or self._env is None:
                return self._build_state()

            self._current_observation = self._env.reset(random_init=random_init)
            self._last_reward = 0.0
            self._last_done = False
            self._last_info = {"message": "environment reset"}
            self._step_index = 0
            self._episode_index += 1
            self._history.clear()
            self._last_auto_step = None
            return self._build_state()

    def step(self, action: Dict[str, float]) -> Dict[str, Any]:
        """
        Execute one step in the RL environment with given action.
        
        Applies the action to the environment, updates state, and records
        the step in history. Returns updated state with reward and done flag.
        
        Args:
            action: Dictionary with 'wob', 'rpm', and 'flow_rate' values
            
        Returns:
            Dictionary containing:
            - observation: Current environment observation
            - reward: Reward received for this step
            - done: Boolean indicating if episode is finished
            - info: Additional information dictionary
            - action: Action that was applied
            - step_index: Current step number
            - episode_index: Current episode number
        """
        with self._lock:
            if not RL_AVAILABLE or self._env is None:
                state = self._build_state()
                state["warning"] = "RL environment not available"
                return state

            if self._last_done:
                result = self._build_state()
                result["warning"] = "Episode finished. Reset environment to continue."
                return result

            action_array = np.array(
                [
                    float(action.get("wob", 0.0)),
                    float(action.get("rpm", 0.0)),
                    float(action.get("flow_rate", 0.0)),
                ],
                dtype=np.float32,
            )

            observation, reward, done, info = self._env.step(action_array)

            self._current_observation = observation
            self._last_reward = float(reward)
            self._last_done = bool(done)
            self._last_info = info or {}
            self._step_index += 1

            snapshot = self._build_state()
            snapshot["action"] = {
                "wob": float(action_array[0]),
                "rpm": float(action_array[1]),
                "flow_rate": float(action_array[2]),
            }
            self._history.append(snapshot)
            return snapshot

    def auto_step(self) -> Dict[str, Any]:
        """
        Automatically execute a step using the loaded policy.
        
        Uses the loaded policy model to select an action and execute it.
        Respects auto_interval_seconds to prevent too frequent steps.
        Only works when policy mode is set to "auto" and a policy is loaded.
        
        Returns:
            Dictionary containing step result with success flag and state,
            or error message if conditions are not met
        """
        with self._lock:
            if self._policy_mode != "auto":
                return {
                    "success": False,
                    "message": "Policy mode is not set to auto",
                    "state": self._build_state(),
                }

            if self._policy_wrapper is None or not self._policy_metadata.get("loaded"):
                return {
                    "success": False,
                    "message": "No policy loaded for auto mode",
                    "state": self._build_state(),
                }

            if not RL_AVAILABLE or self._env is None:
                return {
                    "success": False,
                    "message": "RL environment not available",
                    "state": self._build_state(),
                }

            if self._last_done:
                return {
                    "success": False,
                    "message": "Episode finished. Reset environment before auto-stepping.",
                    "state": self._build_state(),
                }

            now = time.time()
            if self._last_auto_step and (now - self._last_auto_step) < max(AUTO_STEP_MIN_INTERVAL, self._auto_interval_seconds):
                return {
                    "success": False,
                    "message": "Auto step throttled; wait before requesting another step",
                    "state": self._build_state(),
                }

            try:
                action_array = self._policy_wrapper.act(np.asarray(self._current_observation, dtype=np.float32))
            except Exception as exc:  # pragma: no cover - defensive logging
                error_state = self._build_state()
                error_state["warning"] = f"Policy action failed: {exc}"
                return {
                    "success": False,
                    "message": str(exc),
                    "state": error_state,
                }

            result_state = self.step(
                {
                    "wob": float(action_array[0]),
                    "rpm": float(action_array[1]),
                    "flow_rate": float(action_array[2]),
                }
            )
            self._last_auto_step = now
            return {
                "success": True,
                "message": "Auto step executed",
                "state": result_state,
            }

    def get_state(self) -> Dict[str, Any]:
        """
        Get current environment state.
        
        Returns:
            Dictionary containing current observation, reward, done flag,
            info, step index, and episode index
        """
        with self._lock:
            return self._build_state()

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get episode history.
        
        Args:
            limit: Maximum number of history entries to return (default: 100)
            
        Returns:
            List of state dictionaries from current episode history
        """
        with self._lock:
            if limit <= 0:
                return []
            return self._history[-limit:]

    def get_config(self) -> Dict[str, Any]:
        """
        Get RL environment configuration.
        
        Returns:
            Dictionary containing:
            - available: Boolean indicating if RL environment is available
            - action_space: Action space bounds for wob, rpm, flow_rate
            - observation_labels: List of observation dimension names
            - max_episode_steps: Maximum steps per episode
            - policy_mode: Current policy mode ("manual" or "auto")
        """
        base_config = {
            "available": bool(RL_AVAILABLE and self._env is not None),
            "action_space": {
                "wob": {"min": 0.0, "max": float(MAX_WOB) if RL_AVAILABLE else 0.0},
                "rpm": {"min": 0.0, "max": float(MAX_RPM) if RL_AVAILABLE else 0.0},
                "flow_rate": {"min": 0.0, "max": float(MAX_FLOW_RATE) if RL_AVAILABLE else 0.0},
            },
            "observation_labels": [
                "depth",
                "bit_wear",
                "rop",
                "torque",
                "pressure",
                "vibration_axial",
                "vibration_lateral",
                "vibration_torsional",
            ],
            "max_episode_steps": getattr(self._env, "max_episode_steps", 0) if RL_AVAILABLE else 0,
            "policy_mode": self._policy_mode,
        }

        if not base_config["available"]:
            base_config["message"] = "Reinforcement learning environment unavailable"
        return base_config

    # ------------------------------------------------------------------ #
    # Policy management
    # ------------------------------------------------------------------ #
    def load_policy_from_mlflow(self, model_name: str, stage: str = "Production") -> Dict[str, Any]:
        """
        Load policy model from MLflow model registry.
        
        Args:
            model_name: Name of the model in MLflow registry
            stage: Model stage to load (default: "Production")
            
        Returns:
            Dictionary with success flag and status message
        """
        with self._lock:
            if mlflow_service is None:
                return {
                    "success": False,
                    "message": "MLflow service not available",
                }

            model = mlflow_service.load_pytorch_model(model_name=model_name, stage=stage)
            if model is None:
                return {
                    "success": False,
                    "message": f"Failed to load model '{model_name}' from MLflow stage '{stage}'",
                }

            return self._attach_policy(model, source="mlflow", identifier=model_name, stage=stage)

    def load_policy_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load policy model from local file.
        
        Supports both .pt/.pth (TorchScript) and regular PyTorch model files.
        
        Args:
            file_path: Path to the policy model file
            
        Returns:
            Dictionary with success flag and status message
        """
        with self._lock:
            if not TORCH_AVAILABLE or torch is None:
                return {
                    "success": False,
                    "message": "PyTorch not available; cannot load file policy",
                }

            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "message": f"Policy file not found: {file_path}",
                }

            try:
                if file_path.endswith(".pt") or file_path.endswith(".pth"):
                    model = torch.jit.load(file_path, map_location="cpu")
                else:
                    model = torch.load(file_path, map_location="cpu")
            except Exception as exc:
                return {
                    "success": False,
                    "message": f"Failed to load policy file: {exc}",
                }

            return self._attach_policy(model, source="file", identifier=file_path, stage=None)

    def _attach_policy(self, model, source: str, identifier: Optional[str], stage: Optional[str]) -> Dict[str, Any]:
        """
        Attach a policy model to the service.
        
        Wraps the model in a PolicyWrapper and updates policy metadata.
        This is an internal method called by load_policy_from_mlflow and load_policy_from_file.
        
        Args:
            model: Policy model to attach (PyTorch or Stable-Baselines3)
            source: Source identifier ("mlflow" or "file")
            identifier: Model identifier (name or file path)
            stage: Model stage (for MLflow models)
            
        Returns:
            Dictionary with success flag and policy status
        """
        if not RL_AVAILABLE or self._env is None:
            return {
                "success": False,
                "message": "RL environment not available; cannot attach policy",
            }

        action_bounds = {
            "wob": {"min": 0.0, "max": float(MAX_WOB) if RL_AVAILABLE else 0.0},
            "rpm": {"min": 0.0, "max": float(MAX_RPM) if RL_AVAILABLE else 0.0},
            "flow_rate": {"min": 0.0, "max": float(MAX_FLOW_RATE) if RL_AVAILABLE else 0.0},
        }

        try:
            wrapper = _PolicyWrapper(model=model, action_size=len(action_bounds), action_bounds=action_bounds)
        except Exception as exc:
            return {
                "success": False,
                "message": f"Unable to wrap policy: {exc}",
            }

        self._policy_wrapper = wrapper
        self._policy_metadata = {
            "loaded": True,
            "source": source,
            "identifier": identifier,
            "stage": stage,
            "loaded_at": time.time(),
            "message": None,
        }
        return {
            "success": True,
            "message": f"Policy loaded from {source}",
            "status": self.get_policy_status(),
        }

    def set_policy_mode(self, mode: Literal["manual", "auto"], auto_interval_seconds: Optional[float] = None) -> Dict[str, Any]:
        """
        Set policy control mode.
        
        Switches between manual control (actions provided by user) and
        automatic control (actions selected by loaded policy).
        
        Args:
            mode: Control mode ("manual" or "auto")
            auto_interval_seconds: Minimum interval between auto steps in seconds
                                  (only used when mode is "auto")
            
        Returns:
            Dictionary with success flag and updated policy status
        """
        with self._lock:
            if mode == "auto":
                if self._policy_wrapper is None or not self._policy_metadata.get("loaded"):
                    return {
                        "success": False,
                        "message": "Cannot switch to auto mode without loading a policy",
                        "status": self.get_policy_status(),
                    }

            if auto_interval_seconds is not None:
                self._auto_interval_seconds = max(AUTO_STEP_MIN_INTERVAL, float(auto_interval_seconds))

            self._policy_mode = mode
            self._last_auto_step = None
            status = self.get_policy_status()
            status["mode"] = mode
            return {
                "success": True,
                "message": f"Policy mode set to {mode}",
                "status": status,
            }

    def get_policy_status(self) -> Dict[str, Any]:
        """
        Get current policy status and metadata.
        
        Returns:
            Dictionary containing:
            - loaded: Boolean indicating if policy is loaded
            - source: Policy source ("mlflow" or "file")
            - identifier: Model identifier
            - stage: Model stage (for MLflow)
            - loaded_at: ISO timestamp when policy was loaded
            - mode: Current policy mode
            - auto_interval_seconds: Auto step interval
            - policy_loaded: Boolean indicating if policy wrapper exists
        """
        metadata = dict(self._policy_metadata)
        metadata.update(
            {
                "mode": self._policy_mode,
                "auto_interval_seconds": self._auto_interval_seconds,
                "policy_loaded": bool(self._policy_wrapper is not None and metadata.get("loaded")),
            }
        )
        loaded_at = metadata.get("loaded_at")
        if loaded_at:
            try:
                metadata["loaded_at"] = datetime.fromtimestamp(float(loaded_at)).isoformat()
            except Exception:
                metadata["loaded_at"] = None
        return metadata

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _build_state(self) -> Dict[str, Any]:
        """
        Build current state dictionary from internal state.
        
        This is an internal helper method that constructs a state dictionary
        from the service's internal state variables.
        
        Returns:
            Dictionary containing observation, reward, done, info, step_index,
            and episode_index
        """
        observation = (
            self._current_observation.tolist()
            if isinstance(self._current_observation, np.ndarray)
            else list(self._current_observation)
        )
        state = {
            "observation": [float(x) for x in observation],
            "reward": float(self._last_reward),
            "done": bool(self._last_done),
            "info": self._last_info,
            "step": int(self._step_index),
            "episode": int(self._episode_index),
            "policy_mode": self._policy_mode,
            "policy_loaded": bool(self._policy_wrapper is not None and self._policy_metadata.get("loaded")),
        }
        return state


# Global singleton instance
rl_service = RLService()
