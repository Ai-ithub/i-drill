from datetime import datetime, timedelta
from pathlib import Path
import sys

from fastapi.testclient import TestClient

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src" / "backend"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from app import app  # noqa: E402


class _StubDataService:
    def get_historical_data(self, **kwargs):
        timestamp = datetime.now().isoformat()
        return [
            {
                "id": 1,
                "rig_id": kwargs.get("rig_id") or "RIG-01",
                "timestamp": timestamp,
                "depth": 1200.5,
                "wob": 15000.0,
                "rpm": 90.0,
                "torque": 8000.0,
                "rop": 45.0,
                "mud_flow": 700.0,
                "mud_pressure": 2900.0,
                "mud_temperature": 65.0,
                "gamma_ray": 75.0,
                "resistivity": 2.4,
                "density": 1.2,
                "porosity": 12.0,
                "hook_load": 20000.0,
                "vibration": 1.2,
                "status": "normal",
            }
        ]

    def get_time_series_aggregated(self, **kwargs):
        return [
            {
                "timestamp": datetime.now().isoformat(),
                "avg_wob": 14000.0,
                "avg_rpm": 88.0,
                "avg_torque": 7600.0,
                "avg_rop": 40.0,
                "avg_mud_flow": 680.0,
                "avg_mud_pressure": 2800.0,
                "max_depth": 1250.0,
            }
        ]


client = TestClient(app)


def test_historical_sensor_data(monkeypatch):
    from api.routes import sensor_data as sensor_routes

    stub_service = _StubDataService()
    monkeypatch.setattr(sensor_routes, "data_service", stub_service, raising=False)

    params = {
        "start_time": (datetime.now() - timedelta(hours=2)).isoformat(),
        "end_time": datetime.now().isoformat(),
        "rig_id": "RIG-77",
        "limit": 10,
    }

    response = client.get("/api/v1/sensor-data/historical", params=params)

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["count"] == 1
    assert body["data"][0]["rig_id"] == "RIG-77"


def test_historical_sensor_data_invalid_time(monkeypatch):
    from api.routes import sensor_data as sensor_routes

    stub_service = _StubDataService()
    monkeypatch.setattr(sensor_routes, "data_service", stub_service, raising=False)

    now = datetime.now().isoformat()
    response = client.get(
        "/api/v1/sensor-data/historical",
        params={
            "start_time": now,
            "end_time": now,
        },
    )

    assert response.status_code == 400


def test_aggregated_sensor_data(monkeypatch):
    from api.routes import sensor_data as sensor_routes

    stub_service = _StubDataService()
    monkeypatch.setattr(sensor_routes, "data_service", stub_service, raising=False)

    response = client.get(
        "/api/v1/sensor-data/aggregated",
        params={"rig_id": "RIG-01", "time_bucket_seconds": 120},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["count"] == 1
    assert body["data"][0]["avg_wob"] == 14000.0
