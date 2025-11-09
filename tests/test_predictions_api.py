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


class _StubTrainingPipelineService:
    def __init__(self, success: bool = True):
        self.success = success
        self.last_train_args = None
        self.last_promote_args = None

    def start_training_job(self, model_name, parameters=None, experiment_name=None):
        self.last_train_args = {
            "model_name": model_name,
            "parameters": parameters,
            "experiment_name": experiment_name,
        }
        if not self.success:
            return {"success": False, "message": "MLflow is not configured"}
        return {
            "success": True,
            "run_id": "run-123",
            "experiment_name": experiment_name or "default",
            "metrics": {"placeholder_accuracy": 0.9},
        }

    def promote_model(self, model_name, version, stage):
        self.last_promote_args = {
            "model_name": model_name,
            "version": version,
            "stage": stage,
        }
        if not self.success:
            return {"success": False, "message": "Promotion failed"}
        return {"success": True}

    def list_registered_models(self):
        return [
            {
                "name": "demo-model",
                "creation_timestamp": 123,
                "last_updated_timestamp": 456,
                "latest_versions": [{"version": "1", "stage": "Production"}],
            }
        ]

    def list_model_versions(self, model_name):
        return [{"version": "1", "stage": "Production", "run_id": "run-123"}]


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


def test_start_training_job_endpoint(monkeypatch):
    from api.routes import predictions as predictions_router

    stub_service = _StubTrainingPipelineService(success=True)
    monkeypatch.setattr(predictions_router, "training_pipeline_service", stub_service, raising=False)

    payload = {
        "model_name": "demo-model",
        "experiment_name": "demo-exp",
        "parameters": {"learning_rate": 0.001},
    }

    response = client.post("/api/v1/predictions/pipeline/train", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["run_id"] == "run-123"
    assert stub_service.last_train_args["model_name"] == "demo-model"


def test_start_training_job_handles_configuration_error(monkeypatch):
    from api.routes import predictions as predictions_router

    stub_service = _StubTrainingPipelineService(success=False)
    monkeypatch.setattr(predictions_router, "training_pipeline_service", stub_service, raising=False)

    payload = {"model_name": "demo-model"}
    response = client.post("/api/v1/predictions/pipeline/train", json=payload)

    assert response.status_code == 503


def test_promote_model_endpoint(monkeypatch):
    from api.routes import predictions as predictions_router

    stub_service = _StubTrainingPipelineService(success=True)
    monkeypatch.setattr(predictions_router, "training_pipeline_service", stub_service, raising=False)

    payload = {"model_name": "demo-model", "version": "1", "stage": "Production"}
    response = client.post("/api/v1/predictions/pipeline/promote", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert stub_service.last_promote_args["stage"] == "Production"


def test_list_registered_models(monkeypatch):
    from api.routes import predictions as predictions_router

    stub_service = _StubTrainingPipelineService(success=True)
    monkeypatch.setattr(predictions_router, "training_pipeline_service", stub_service, raising=False)

    response = client.get("/api/v1/predictions/pipeline/models")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert len(body["models"]) == 1
    assert body["models"][0]["name"] == "demo-model"


def test_list_model_versions(monkeypatch):
    from api.routes import predictions as predictions_router

    stub_service = _StubTrainingPipelineService(success=True)
    monkeypatch.setattr(predictions_router, "training_pipeline_service", stub_service, raising=False)

    response = client.get("/api/v1/predictions/pipeline/models/demo-model/versions")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["versions"][0]["version"] == "1"

