# i-Drill Backend Test Suite

Comprehensive test suite for the i-Drill backend API using pytest.

## 📋 Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [Writing Tests](#writing-tests)
- [CI/CD Integration](#cicd-integration)

## 🎯 Overview

This test suite provides comprehensive coverage for:
- **Unit Tests**: Individual components and utilities
- **Integration Tests**: API endpoints and database operations
- **Authentication Tests**: Security and authorization
- **Validation Tests**: Input validation and sanitization

## 📁 Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest fixtures and configuration
├── test_validators.py       # Input validation tests
├── test_services.py         # Service layer tests
├── test_api_routes.py       # API endpoint tests
├── test_auth.py             # Authentication tests
└── test_database.py         # Database operation tests
```

## 🚀 Running Tests

### Install Dependencies

```bash
pip install -r requirements_test.txt
```

### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=. --cov-report=html
```

### Run Specific Test Categories

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only authentication tests
pytest -m auth

# Run only database tests
pytest -m database
```

### Run Specific Test Files

```bash
# Run validator tests
pytest tests/test_validators.py

# Run API route tests
pytest tests/test_api_routes.py

# Run authentication tests
pytest tests/test_auth.py
```

### Run with Coverage

```bash
# Generate HTML coverage report
pytest --cov=. --cov-report=html

# View coverage in terminal
pytest --cov=. --cov-report=term-missing

# Generate XML coverage report (for CI/CD)
pytest --cov=. --cov-report=xml
```

## 📊 Test Coverage

Current coverage targets:
- **Minimum Coverage**: 60% (configured in `pytest.ini`)
- **Target Coverage**: 80%+
- **Critical Components**: 90%+

### View Coverage Report

After running tests with coverage:

```bash
# HTML report will be generated in htmlcov/index.html
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html  # Windows
```

## ✍️ Writing Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Example Test Structure

```python
"""
Tests for [Component Name]
"""
import pytest
from unittest.mock import Mock, MagicMock

class TestComponentName:
    """Tests for ComponentName"""
    
    @pytest.fixture
    def component(self):
        """Create component instance"""
        return ComponentName()
    
    def test_feature_name(self, component):
        """Test description"""
        # Arrange
        input_data = "test"
        
        # Act
        result = component.method(input_data)
        
        # Assert
        assert result == expected_value
```

### Using Fixtures

Fixtures are defined in `conftest.py`:

```python
# Available fixtures:
- test_client: FastAPI TestClient
- test_db_session: Database session
- mock_db_manager: Mocked database manager
- mock_kafka_service: Mocked Kafka service
- sample_sensor_data: Sample sensor data dict
- sample_user_data: Sample user data dict
- auth_headers: Mock authentication headers
```

### Example: Testing API Endpoint

```python
def test_get_sensor_data(test_client):
    """Test getting sensor data"""
    response = test_client.get("/api/v1/sensor-data/realtime")
    assert response.status_code == 200
    assert "data" in response.json()
```

### Example: Testing with Mocks

```python
@patch('services.kafka_service.KafkaService')
def test_service_with_mock(mock_kafka, test_client):
    """Test service with mocked dependencies"""
    mock_kafka.available = True
    # ... test implementation
```

## 🔐 Authentication Testing

Authentication tests verify:
- User registration
- Login/logout
- Token creation and validation
- Password reset
- Account lockout
- Role-based access control

### Example: Testing Authentication

```python
def test_login_success(test_client, sample_user_data):
    """Test successful login"""
    # Register user
    test_client.post("/api/v1/auth/register", json=sample_user_data)
    
    # Login
    response = test_client.post(
        "/api/v1/auth/login",
        data={
            "username": sample_user_data["username"],
            "password": sample_user_data["password"]
        }
    )
    
    assert response.status_code == 200
    assert "access_token" in response.json()
```

## 🗄️ Database Testing

Database tests use an in-memory SQLite database for fast, isolated tests.

### Example: Testing Database Operations

```python
def test_create_sensor_data(test_db_session):
    """Test creating sensor data"""
    sensor_data = SensorData(
        rig_id="RIG_01",
        timestamp=datetime.now(),
        depth=5000.0
    )
    test_db_session.add(sensor_data)
    test_db_session.commit()
    
    assert sensor_data.id is not None
```

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements_test.txt
      - run: pytest --cov=. --cov-report=xml
      - uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

pytest --cov=. --cov-report=term-missing --cov-fail-under=60
```

## 📝 Test Markers

Available pytest markers:

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Slow-running tests
- `@pytest.mark.auth`: Authentication tests
- `@pytest.mark.database`: Database tests
- `@pytest.mark.api`: API endpoint tests
- `@pytest.mark.service`: Service layer tests
- `@pytest.mark.utils`: Utility function tests

### Using Markers

```python
@pytest.mark.unit
def test_unit_function():
    """Unit test"""
    pass

@pytest.mark.integration
def test_integration_endpoint():
    """Integration test"""
    pass
```

## 🐛 Debugging Tests

### Run with Debug Output

```bash
# Run with print statements
pytest -s

# Run with detailed traceback
pytest --tb=long

# Run single test
pytest tests/test_validators.py::TestRigIdValidation::test_valid_rig_id -v
```

### Using PDB Debugger

```python
def test_with_debugger():
    """Test with debugger"""
    import pdb; pdb.set_trace()
    # Test code
```

## 📈 Coverage Goals

| Component | Current | Target |
|-----------|---------|--------|
| Validators | 90%+ | 95% |
| Services | 70%+ | 85% |
| API Routes | 60%+ | 80% |
| Authentication | 80%+ | 90% |
| Database | 70%+ | 85% |
| **Overall** | **60%+** | **80%+** |

## 🔧 Configuration

Test configuration is in `pytest.ini`:

- Test paths: `tests/`
- Coverage threshold: 60%
- Coverage exclusions: migrations, scripts, test files
- Markers: unit, integration, auth, database, etc.

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)

## 🤝 Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure tests pass
3. Maintain or improve coverage
4. Update this README if needed

## ⚠️ Notes

- Tests use in-memory SQLite for speed
- External services (Kafka, Redis) are mocked
- Some tests may require environment variables
- Database migrations should be tested separately

