from datetime import datetime, timedelta
from pathlib import Path
import sys

from fastapi.testclient import TestClient

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src" / "backend"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from app import app  # noqa: E402


class _MockMaintenanceService:
    def __init__(self):
        self.created_alert_payload = None
        self.created_schedule_payload = None
        self.updated_payload = None
        self.deleted_ids = []

    # Alerts
    def get_maintenance_alerts(self, **kwargs):
        return [
            {
                "id": 1,
                "rig_id": kwargs.get("rig_id") or "RIG_01",
                "component": "pump",
                "alert_type": "overheat",
                "severity": "critical",
                "message": "Temperature exceeded threshold",
                "predicted_failure_time": None,
                "created_at": datetime.now(),
                "acknowledged": False,
                "acknowledged_by": None,
                "acknowledged_at": None,
                "resolved": False,
                "resolved_at": None,
            }
        ]

    def get_maintenance_alert_by_id(self, alert_id: int):
        if alert_id == 1:
            return {
                "id": 1,
                "rig_id": "RIG_01",
                "component": "pump",
                "alert_type": "overheat",
                "severity": "critical",
                "message": "Temperature exceeded threshold",
                "predicted_failure_time": None,
                "created_at": datetime.now(),
                "acknowledged": False,
                "acknowledged_by": None,
                "acknowledged_at": None,
                "resolved": False,
                "resolved_at": None,
            }
        if alert_id == 2 and self.created_alert_payload:
            return {
                "id": 2,
                "rig_id": self.created_alert_payload["rig_id"],
                "component": self.created_alert_payload["component"],
                "alert_type": self.created_alert_payload["alert_type"],
                "severity": self.created_alert_payload["severity"],
                "message": self.created_alert_payload["message"],
                "predicted_failure_time": None,
                "created_at": datetime.now(),
                "acknowledged": False,
                "acknowledged_by": None,
                "acknowledged_at": None,
                "resolved": False,
                "resolved_at": None,
            }
        return None

    def create_maintenance_alert(self, payload):
        self.created_alert_payload = payload
        return 2

    # Schedules
    def get_maintenance_schedules(self, **kwargs):
        return [
            {
                "id": 10,
                "rig_id": kwargs.get("rig_id") or "RIG_01",
                "component": "pump",
                "maintenance_type": "inspection",
                "scheduled_date": datetime.now() + timedelta(days=1),
                "estimated_duration_hours": 2.0,
                "priority": "high",
                "status": "scheduled",
                "assigned_to": "engineer_a",
                "notes": None,
                "created_at": datetime.now() - timedelta(days=1),
                "updated_at": datetime.now() - timedelta(days=1),
            }
        ]

    def create_maintenance_schedule(self, payload):
        self.created_schedule_payload = payload
        return {
            "id": 11,
            **payload,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

    def update_maintenance_schedule(self, schedule_id, payload):
        if schedule_id == 11:
            self.updated_payload = payload
            return {
                "id": schedule_id,
                "rig_id": "RIG_01",
                "component": "pump",
                "maintenance_type": "inspection",
                "scheduled_date": datetime.now() + timedelta(days=1),
                "estimated_duration_hours": payload.get("estimated_duration_hours", 2.0),
                "priority": payload.get("priority", "high"),
                "status": payload.get("status", "scheduled"),
                "assigned_to": payload.get("assigned_to", "engineer_a"),
                "notes": payload.get("notes"),
                "created_at": datetime.now() - timedelta(days=1),
                "updated_at": datetime.now(),
            }
        return None

    def delete_maintenance_schedule(self, schedule_id):
        self.deleted_ids.append(schedule_id)
        return schedule_id == 11


client = TestClient(app)


def test_get_alerts_returns_alert_list(monkeypatch):
    from api.routes import maintenance as maintenance_router

    mock_service = _MockMaintenanceService()
    monkeypatch.setattr(maintenance_router, "data_service", mock_service, raising=False)

    response = client.get("/api/v1/maintenance/alerts?rig_id=RIG_77")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert body[0]["rig_id"] == "RIG_77"
    assert body[0]["severity"] == "critical"


def test_create_schedule_persists_data(monkeypatch):
    from api.routes import maintenance as maintenance_router

    mock_service = _MockMaintenanceService()
    monkeypatch.setattr(maintenance_router, "data_service", mock_service, raising=False)

    payload = {
        "rig_id": "RIG_02",
        "component": "motor",
        "maintenance_type": "repair",
        "scheduled_date": (datetime.now() + timedelta(days=2)).isoformat(),
        "estimated_duration_hours": 4.0,
        "priority": "medium",
        "status": "scheduled",
        "assigned_to": "tech_b",
        "notes": "Check vibration levels",
    }

    response = client.post("/api/v1/maintenance/schedule", json=payload)

    assert response.status_code == 201
    body = response.json()
    assert body["rig_id"] == "RIG_02"
    assert mock_service.created_schedule_payload["component"] == "motor"


def test_delete_schedule_handles_missing_record(monkeypatch):
    from api.routes import maintenance as maintenance_router

    mock_service = _MockMaintenanceService()
    monkeypatch.setattr(maintenance_router, "data_service", mock_service, raising=False)

    response = client.delete("/api/v1/maintenance/schedule/99")

    assert response.status_code == 404
    assert mock_service.deleted_ids == [99]


def test_create_alert_uses_service(monkeypatch):
    from api.routes import maintenance as maintenance_router

    mock_service = _MockMaintenanceService()
    monkeypatch.setattr(maintenance_router, "data_service", mock_service, raising=False)

    payload = {
        "rig_id": "RIG_05",
        "component": "pump",
        "alert_type": "pressure_spike",
        "severity": "critical",
        "message": "Pressure rose rapidly",
    }

    response = client.post("/api/v1/maintenance/alerts", json=payload)

    assert response.status_code == 201
    body = response.json()
    assert body["rig_id"] == "RIG_05"
    assert mock_service.created_alert_payload["rig_id"] == "RIG_05"

