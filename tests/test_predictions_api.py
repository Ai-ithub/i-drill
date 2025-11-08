from datetime import datetime
from pathlib import Path
import sys

from fastapi.testclient import TestClient

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src" / "backend"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from app import app  # noqa: E402
from services.prediction_service import PredictionService, FEATURE_ORDER  # noqa: E402


class _SuccessfulMockPredictionService:
    def predict_rul(self, rig_id, sensor_data, model_type, lookback_window):
        return {
            "success": True,
            "predictions": [
                {
                    "rig_id": rig_id,
                    "component": "general",
                    "predicted_rul": 120.0,
                    "confidence": 0.92,
                    "timestamp": datetime.now(),
                    "model_used": model_type,
                    "recommendation": "Monitor closely",
                }
            ],
            "message": None,
        }


class _FailingMockPredictionService:
    def predict_rul(self, rig_id, sensor_data, model_type, lookback_window):
        return {
            "success": False,
            "predictions": [],
            "message": "Model unavailable",
        }


client = TestClient(app)


def test_predict_rul_returns_rul_prediction_response(monkeypatch):
    from api.routes import predictions as predictions_router

    monkeypatch.setattr(
        predictions_router,
        "prediction_service",
        _SuccessfulMockPredictionService(),
        raising=False,
    )

    payload = {
        "rig_id": "RIG_01",
        "sensor_data": [{"depth": 1.0, "wob": 1.0, "rpm": 1.0, "torque": 1.0, "rop": 1.0, "mud_flow": 1.0, "mud_pressure": 1.0}],
        "model_type": "lstm",
        "lookback_window": 10,
    }

    response = client.post("/api/v1/predictions/rul", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert isinstance(body["predictions"], list)
    assert len(body["predictions"]) == 1
    prediction = body["predictions"][0]
    assert prediction["rig_id"] == "RIG_01"
    assert prediction["predicted_rul"] == 120.0
    assert prediction["component"] == "general"


def test_predict_rul_handles_service_failure(monkeypatch):
    from api.routes import predictions as predictions_router

    monkeypatch.setattr(
        predictions_router,
        "prediction_service",
        _FailingMockPredictionService(),
        raising=False,
    )

    payload = {
        "rig_id": "RIG_02",
        "sensor_data": [{"depth": 1.0, "wob": 1.0, "rpm": 1.0, "torque": 1.0, "rop": 1.0, "mud_flow": 1.0, "mud_pressure": 1.0}],
        "model_type": "lstm",
        "lookback_window": 10,
    }

    response = client.post("/api/v1/predictions/rul", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["predictions"] == []
    assert body["message"] == "Model unavailable"


def test_prepare_input_data_matches_sensor_schema():
    service = PredictionService()
    sensor_point = {
        "depth": 1000.0,
        "wob": 15000.0,
        "rpm": 90.0,
        "torque": 8000.0,
        "rop": 45.0,
        "mud_flow": 700.0,
        "mud_pressure": 2900.0,
        "mud_temperature": 65.0,
        "gamma_ray": 75.0,
        "resistivity": 2.4,
        "density": 1.1,
        "porosity": 12.0,
        "hook_load": 20000.0,
        "vibration": 1.2,
        "power_consumption": 250.0,
    }
    sensor_sequence = [sensor_point for _ in range(60)]

    prepared = service._prepare_input_data(sensor_sequence, lookback_window=50)

    assert prepared is not None
    assert prepared.shape == (50, len(FEATURE_ORDER))
    assert prepared[0, 0] == sensor_point["depth"]
    assert prepared[0, 1] == sensor_point["wob"]

