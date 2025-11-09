from pathlib import Path
import sys

from fastapi.testclient import TestClient

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src" / "backend"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from app import app  # noqa: E402


class _StubDVRService:
    def process_record(self, record):
        return {"success": True, "processed_record": {**record, "Anomaly": False}, "message": None}

    def get_recent_stats(self, limit):
        return {"success": True, "summary": {"count": 1}, "message": None}

    def get_anomaly_snapshot(self, history_size):
        return {
            "success": True,
            "numeric_columns": ["wob"],
            "history_sizes": {"wob": 10},
            "message": None,
        }

    def evaluate_record_anomaly(self, record, history_size):
        return {"success": True, "record": {**record, "Anomaly": False}}


client = TestClient(app)


def test_dvr_process(monkeypatch):
    from api.routes import dvr as dvr_routes

    stub = _StubDVRService()
    monkeypatch.setattr(dvr_routes, "dvr_service", stub, raising=False)

    payload = {"record": {"value": 42}}
    response = client.post("/api/v1/dvr/process", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["processed_record"]["value"] == 42


def test_dvr_stats(monkeypatch):
    from api.routes import dvr as dvr_routes

    stub = _StubDVRService()
    monkeypatch.setattr(dvr_routes, "dvr_service", stub, raising=False)

    response = client.get("/api/v1/dvr/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["count"] == 1


def test_dvr_anomalies(monkeypatch):
    from api.routes import dvr as dvr_routes

    stub = _StubDVRService()
    monkeypatch.setattr(dvr_routes, "dvr_service", stub, raising=False)

    response = client.get("/api/v1/dvr/anomalies")
    assert response.status_code == 200
    body = response.json()
    assert "wob" in body["numeric_columns"]
