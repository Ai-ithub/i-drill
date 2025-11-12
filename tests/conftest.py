"""
Pytest configuration and fixtures
"""
import pytest
import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src" / "backend"
sys.path.insert(0, str(src_path))

from fastapi.testclient import TestClient
from app import app as fastapi_app


@pytest.fixture(scope="session")
def app():
    """Create FastAPI app instance for testing"""
    # Set test environment variables
    os.environ.setdefault("APP_ENV", "test")
    os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
    os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing-only")
    os.environ.setdefault("REDIS_HOST", "localhost")
    os.environ.setdefault("REDIS_PORT", "6379")
    os.environ.setdefault("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    
    return fastapi_app


@pytest.fixture(scope="function")
def client(app):
    """Create test client"""
    return TestClient(app)


@pytest.fixture(scope="function")
def auth_headers(client):
    """Get authentication headers for testing"""
    # Create a test user and get token
    # This is a simplified version - adjust based on your auth setup
    login_data = {
        "username": "admin",
        "password": "admin"  # Default admin password
    }
    
    try:
        response = client.post("/api/v1/auth/login", data=login_data)
        if response.status_code == 200:
            token = response.json().get("access_token")
            return {"Authorization": f"Bearer {token}"}
    except Exception:
        pass
    
    return {}


@pytest.fixture(scope="function")
def sample_sensor_data():
    """Sample sensor data for testing"""
    return {
        "rig_id": "RIG_01",
        "timestamp": "2025-01-15T10:30:00Z",
        "depth": 5000.0,
        "wob": 1500.0,
        "rpm": 80.0,
        "torque": 400.0,
        "rop": 12.0,
        "mud_flow": 1200.0,
        "mud_pressure": 3000.0,
        "mud_temperature": 60.0,
        "gamma_ray": 85.0,
        "resistivity": 20.0,
        "status": "normal"
    }


@pytest.fixture(scope="function")
def sample_rul_request():
    """Sample RUL prediction request"""
    return {
        "rig_id": "RIG_01",
        "model_type": "lstm",
        "sensor_data": {
            "depth": 5000.0,
            "wob": 1500.0,
            "rpm": 80.0,
            "torque": 400.0
        }
    }


@pytest.fixture(scope="function")
def sample_maintenance_alert():
    """Sample maintenance alert"""
    return {
        "rig_id": "RIG_01",
        "severity": "high",
        "message": "High vibration detected",
        "equipment": "Motor",
        "recommended_action": "Inspect motor bearings"
    }

