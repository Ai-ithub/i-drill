from typing import Dict, Any, List
import threading

import numpy as np

try:
    from drilling_env.drilling_env import DrillingEnv, MAX_WOB, MAX_RPM, MAX_FLOW_RATE
    RL_AVAILABLE = True
except ImportError:
    DrillingEnv = None  # type: ignore
    MAX_WOB = MAX_RPM = MAX_FLOW_RATE = 0  # type: ignore
    RL_AVAILABLE = False


class RLService:
    """Service wrapper around the drilling reinforcement learning environment."""

    def __init__(self):
        self._lock = threading.Lock()
        self._history: List[Dict[str, Any]] = []
        self._episode_index = 0
        self._step_index = 0
        self._last_reward = 0.0
        self._last_done = False
        self._last_info: Dict[str, Any] = {}

        if RL_AVAILABLE and DrillingEnv is not None:
            self._env = DrillingEnv()
            self._current_observation = self._env.reset()
        else:
            self._env = None
            self._current_observation = np.zeros(8, dtype=np.float32)
            self._last_info = {"warning": "Reinforcement learning environment unavailable"}

    def reset(self, random_init: bool = False) -> Dict[str, Any]:
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
            return self._build_state()

    def step(self, action: Dict[str, float]) -> Dict[str, Any]:
        with self._lock:
            if not RL_AVAILABLE or self._env is None:
                state = self._build_state()
                state["warning"] = "RL environment not available"
                return state

            if self._last_done:
                result = self._build_state()
                result["warning"] = "Episode finished. Reset environment to continue."
                return result

            action_array = np.array([
                float(action.get("wob", 0.0)),
                float(action.get("rpm", 0.0)),
                float(action.get("flow_rate", 0.0)),
            ], dtype=np.float32)

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

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return self._build_state()

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            if limit <= 0:
                return []
            return self._history[-limit:]

    def get_config(self) -> Dict[str, Any]:
        if not RL_AVAILABLE or self._env is None:
            return {
                "available": False,
                "message": "Reinforcement learning environment unavailable",
                "action_space": {},
                "observation_labels": [],
                "max_episode_steps": 0,
            }

        return {
            "available": True,
            "action_space": {
                "wob": {"min": 0.0, "max": float(MAX_WOB)},
                "rpm": {"min": 0.0, "max": float(MAX_RPM)},
                "flow_rate": {"min": 0.0, "max": float(MAX_FLOW_RATE)},
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
            "max_episode_steps": getattr(self._env, "max_episode_steps", 0),
        }

    def _build_state(self) -> Dict[str, Any]:
        observation = self._current_observation.tolist() if isinstance(self._current_observation, np.ndarray) else list(self._current_observation)
        return {
            "observation": [float(x) for x in observation],
            "reward": float(self._last_reward),
            "done": bool(self._last_done),
            "info": self._last_info,
            "step": int(self._step_index),
            "episode": int(self._episode_index),
        }


# Global singleton instance
rl_service = RLService()
