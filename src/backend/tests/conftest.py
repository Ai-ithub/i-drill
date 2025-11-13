"""
Pytest configuration and fixtures
"""
import pytest
import sys
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, MagicMock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from app import app
from database import DatabaseManager
from api.models.database_models import Base

# Test database URL (use in-memory SQLite for testing)
TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine"""
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def test_db_session(test_engine) -> Generator[Session, None, None]:
    """Create a test database session"""
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(scope="function")
def test_client() -> TestClient:
    """Create a test client for FastAPI app"""
    return TestClient(app)


@pytest.fixture
def mock_db_manager():
    """Mock database manager"""
    manager = Mock(spec=DatabaseManager)
    manager._initialized = True
    manager.session_scope = MagicMock()
    return manager


@pytest.fixture
def mock_kafka_service():
    """Mock Kafka service"""
    service = Mock()
    service.available = True
    service.produce_sensor_data = Mock(return_value=True)
    service.create_consumer = Mock(return_value=True)
    service.kafka_config = {
        'bootstrap_servers': 'localhost:9092',
        'topics': {
            'sensor_stream': 'rig.sensor.stream'
        }
    }
    return service


@pytest.fixture
def sample_sensor_data():
    """Sample sensor data for testing"""
    from datetime import datetime
    return {
        "rig_id": "RIG_01",
        "timestamp": datetime.now().isoformat(),
        "depth": 5000.0,
        "wob": 15000.0,
        "rpm": 100.0,
        "torque": 10000.0,
        "rop": 50.0,
        "mud_flow": 800.0,
        "mud_pressure": 3000.0,
        "mud_temperature": 60.0,
        "bit_temperature": 90.0,
        "motor_temperature": 75.0,
        "power_consumption": 200.0,
        "vibration_level": 0.8
    }


@pytest.fixture
def sample_user_data():
    """Sample user data for testing"""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "TestPassword123!",
        "role": "engineer",
        "is_active": True
    }


@pytest.fixture
def auth_headers():
    """Mock authentication headers"""
    return {
        "Authorization": "Bearer mock_access_token"
    }


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks before each test"""
    yield
    # Cleanup after test

