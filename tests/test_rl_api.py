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

    def get_config(self):
        return {"action_space": {"wob": {"min": 0, "max": 10}}}

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
