from pathlib import Path
import sys

from fastapi.testclient import TestClient

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src" / "backend"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from app import app  # noqa: E402


class _StubDVRService:
    def __init__(self):
        self.history_entry = {
            "id": 1,
            "rig_id": "RIG-01",
            "raw_record": {"value": 42},
            "reconciled_record": {"value": 42, "status": "OK"},
            "is_valid": True,
            "reason": "Data is valid.",
            "anomaly_flag": False,
            "anomaly_details": {"Anomaly": False},
            "status": "processed",
            "notes": None,
            "source": "api",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }

    def process_record(self, record):
        return {
            "success": True,
            "processed_record": {**record, "Anomaly": False},
            "message": None,
            "history_id": 123,
            "anomaly_flag": False,
        }

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

    def get_history(self, limit=50, rig_id=None, status=None):
        return [self.history_entry]

    def update_history_entry(self, entry_id, updates):
        if entry_id != 1:
            return None
        updated = {**self.history_entry, **updates}
        self.history_entry = updated
        return updated

    def delete_history_entry(self, entry_id):
        return entry_id == 1


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
    assert body["history_id"] == 123


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


def test_dvr_history_list(monkeypatch):
    from api.routes import dvr as dvr_routes

    stub = _StubDVRService()
    monkeypatch.setattr(dvr_routes, "dvr_service", stub, raising=False)

    response = client.get("/api/v1/dvr/history")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["history"][0]["id"] == 1


def test_dvr_history_update(monkeypatch):
    from api.routes import dvr as dvr_routes

    stub = _StubDVRService()
    monkeypatch.setattr(dvr_routes, "dvr_service", stub, raising=False)

    response = client.patch("/api/v1/dvr/history/1", json={"status": "acknowledged", "notes": "Reviewed"})
    assert response.status_code == 200
    body = response.json()
    assert body["entry"]["status"] == "acknowledged"
    assert body["entry"]["notes"] == "Reviewed"


def test_dvr_history_delete(monkeypatch):
    from api.routes import dvr as dvr_routes

    stub = _StubDVRService()
    monkeypatch.setattr(dvr_routes, "dvr_service", stub, raising=False)

    response = client.delete("/api/v1/dvr/history/1")
    assert response.status_code == 200
    assert response.json()["success"] is True
