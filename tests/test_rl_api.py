from pathlib import Path
import sys

from fastapi.testclient import TestClient

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src" / "backend"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from app import app  # noqa: E402


class _StubRLService:
    def __init__(self):
        self.state = {
            "observation": [1.0, 2.0],
            "reward": 0.5,
            "done": False,
            "info": {"note": "ok"},
            "step": 3,
            "episode": 1,
        }
        self.policy_mode = "manual"
        self.policy_loaded = False
        self.auto_interval_seconds = 1.0

    def get_config(self):
        return {"action_space": {"wob": {"min": 0, "max": 10}}, "policy_mode": self.policy_mode, "available": True}

    def get_state(self):
        return self.state

    def reset(self, random_init=False):
        self.state["step"] = 0
        return self.state

    def step(self, action):
        self.state["step"] += 1
        self.state["reward"] = 1.0
        self.state["action"] = action
        return self.state

    def get_history(self, limit):
        return [self.state]

    def auto_step(self):
        if self.policy_mode != "auto" or not self.policy_loaded:
            return {"success": False, "message": "not auto", "state": self.state}
        self.state["step"] += 1
        self.state["reward"] = 2.0
        return {"success": True, "state": self.state, "message": "Auto step"}

    def load_policy_from_mlflow(self, model_name, stage):
        self.policy_loaded = True
        return {"success": True, "message": "loaded"}

    def load_policy_from_file(self, file_path):
        self.policy_loaded = True
        return {"success": True, "message": "loaded"}

    def set_policy_mode(self, mode, auto_interval_seconds=None):
        if mode == "auto" and not self.policy_loaded:
            return {"success": False, "message": "Cannot switch", "status": self.get_policy_status()}
        if auto_interval_seconds is not None:
            self.auto_interval_seconds = auto_interval_seconds
        self.policy_mode = mode
        status = self.get_policy_status()
        status["mode"] = mode
        return {"success": True, "message": "mode set", "status": status}

    def get_policy_status(self):
        return {
            "mode": self.policy_mode,
            "policy_loaded": self.policy_loaded,
            "source": "mlflow" if self.policy_loaded else None,
            "identifier": "test-model" if self.policy_loaded else None,
            "stage": "Production" if self.policy_loaded else None,
            "loaded_at": None,
            "auto_interval_seconds": self.auto_interval_seconds,
            "message": None,
        }


client = TestClient(app)


def test_rl_state(monkeypatch):
    from api.routes import rl as rl_routes

    stub = _StubRLService()
    monkeypatch.setattr(rl_routes, "rl_service", stub, raising=False)

    response = client.get("/api/v1/rl/state")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["state"]["observation"] == [1.0, 2.0]


def test_rl_step(monkeypatch):
    from api.routes import rl as rl_routes

    stub = _StubRLService()
    monkeypatch.setattr(rl_routes, "rl_service", stub, raising=False)

    payload = {"wob": 5.0, "rpm": 2.0, "flow_rate": 1.0}
    response = client.post("/api/v1/rl/step", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["state"]["reward"] == 1.0
    assert body["state"]["action"]["wob"] == 5.0


def test_rl_history(monkeypatch):
    from api.routes import rl as rl_routes

    stub = _StubRLService()
    monkeypatch.setattr(rl_routes, "rl_service", stub, raising=False)

    response = client.get("/api/v1/rl/history?limit=5")
    assert response.status_code == 200
    body = response.json()
    assert len(body["history"]) == 1


def test_rl_policy_load_mlflow(monkeypatch):
    from api.routes import rl as rl_routes

    stub = _StubRLService()
    monkeypatch.setattr(rl_routes, "rl_service", stub, raising=False)

    response = client.post(
        "/api/v1/rl/policy/load",
        json={"source": "mlflow", "model_name": "ppo-drill", "stage": "Production"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["status"]["policy_loaded"] is True


def test_rl_policy_mode_switch(monkeypatch):
    from api.routes import rl as rl_routes

    stub = _StubRLService()
    stub.policy_loaded = True
    monkeypatch.setattr(rl_routes, "rl_service", stub, raising=False)

    response = client.post(
        "/api/v1/rl/policy/mode",
        json={"mode": "auto", "auto_interval_seconds": 0.75},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"]["mode"] == "auto"
    assert body["status"]["auto_interval_seconds"] == 0.75


def test_rl_auto_step(monkeypatch):
    from api.routes import rl as rl_routes

    stub = _StubRLService()
    stub.policy_loaded = True
    stub.policy_mode = "auto"
    monkeypatch.setattr(rl_routes, "rl_service", stub, raising=False)

    response = client.post("/api/v1/rl/auto-step")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["state"]["reward"] == 2.0


def test_rl_policy_status(monkeypatch):
    from api.routes import rl as rl_routes

    stub = _StubRLService()
    stub.policy_loaded = True
    monkeypatch.setattr(rl_routes, "rl_service", stub, raising=False)

    response = client.get("/api/v1/rl/policy/status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"]["policy_loaded"] is True
