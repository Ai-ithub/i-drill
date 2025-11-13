# Test Suite Documentation

## Overview

This document describes the comprehensive test suite for the i-Drill backend API. The test suite uses `pytest` as the testing framework and includes unit tests, integration tests, and end-to-end tests.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest fixtures and configuration
├── test_validators.py       # Unit tests for validation utilities
├── test_services.py         # Unit tests for service classes
├── test_api_routes.py       # Integration tests for API endpoints
├── test_auth.py             # Authentication and authorization tests
└── test_database.py         # Database operation tests
```

## Running Tests

### Install Test Dependencies

```bash
pip install -r requirements_test.txt
```

### Run All Tests

```bash
pytest
```

### Run Tests with Coverage

```bash
pytest --cov=. --cov-report=html
```

This will generate:
- Terminal coverage report
- HTML report in `htmlcov/index.html`
- XML report for CI/CD integration

### Run Specific Test Files

```bash
# Run only validator tests
pytest tests/test_validators.py

# Run only API route tests
pytest tests/test_api_routes.py

# Run only authentication tests
pytest tests/test_auth.py
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only authentication tests
pytest -m auth
```

### Run Tests in Parallel

```bash
pytest -n auto
```

## Test Categories

### Unit Tests

Unit tests focus on testing individual functions and methods in isolation:

- **Validators** (`test_validators.py`): Test input validation, sanitization, and format checking
- **Services** (`test_services.py`): Test business logic in service classes
- **Utilities**: Test helper functions and utilities

### Integration Tests

Integration tests verify that different components work together:

- **API Routes** (`test_api_routes.py`): Test HTTP endpoints, request/response handling
- **Database** (`test_database.py`): Test database operations, models, and queries
- **Authentication** (`test_auth.py`): Test authentication flow and authorization

## Test Fixtures

### Common Fixtures (in `conftest.py`)

- `test_engine`: SQLAlchemy engine for test database
- `test_db_session`: Database session for each test
- `test_client`: FastAPI TestClient instance
- `mock_db_manager`: Mocked database manager
- `mock_kafka_service`: Mocked Kafka service
- `sample_sensor_data`: Sample sensor data dictionary
- `sample_user_data`: Sample user data dictionary
- `auth_headers`: Mock authentication headers

## Writing New Tests

### Example: Unit Test

```python
def test_validate_rig_id():
    """Test rig ID validation"""
    assert validate_rig_id("RIG_01") is True
    assert validate_rig_id("") is False
```

### Example: Integration Test

```python
def test_get_realtime_data(test_client):
    """Test getting realtime sensor data"""
    response = test_client.get("/api/v1/sensor-data/realtime")
    assert response.status_code == 200
```

### Example: Service Test with Mocking

```python
@patch('services.kafka_service.KAFKA_AVAILABLE', False)
def test_kafka_unavailable(kafka_service):
    """Test behavior when Kafka is unavailable"""
    assert kafka_service.available is False
```

## Test Coverage Goals

- **Overall Coverage**: Minimum 60% (enforced by `pytest.ini`)
- **Critical Paths**: 80%+ coverage
- **Services**: 70%+ coverage
- **API Routes**: 75%+ coverage

## Continuous Integration

Tests are designed to run in CI/CD pipelines:

1. **Fast Execution**: Most tests complete in < 1 second
2. **Isolated**: Tests don't depend on external services
3. **Deterministic**: Tests produce consistent results
4. **Parallelizable**: Tests can run in parallel

## Mocking Strategy

### When to Mock

- **External Services**: Kafka, MLflow, external APIs
- **Database**: Use in-memory SQLite for unit tests
- **File System**: Mock file operations
- **Time**: Mock datetime for time-dependent tests

### Mocking Best Practices

1. Mock at the boundary (external services, not internal functions)
2. Use `unittest.mock` for Python mocks
3. Use `pytest-mock` for pytest-specific mocking
4. Keep mocks simple and focused

## Test Data

### Test Database

- Uses in-memory SQLite (`sqlite:///:memory:`)
- Created fresh for each test session
- No data persistence between tests

### Sample Data

Fixtures provide sample data:
- `sample_sensor_data`: Valid sensor reading
- `sample_user_data`: Valid user registration data

## Troubleshooting

### Tests Failing Due to Imports

If tests fail with import errors:
```bash
# Ensure backend is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src/backend"
```

### Database Connection Errors

Tests use in-memory SQLite, so no database setup is required. If you see connection errors:
1. Check that `conftest.py` fixtures are working
2. Verify SQLAlchemy models are imported correctly

### Coverage Not Reporting

If coverage reports are empty:
1. Ensure `pytest-cov` is installed
2. Check `pytest.ini` coverage configuration
3. Verify source files are not excluded

## Best Practices

1. **Test Naming**: Use descriptive names: `test_<functionality>_<scenario>`
2. **Arrange-Act-Assert**: Structure tests clearly
3. **One Assertion Per Test**: Focus each test on one behavior
4. **Test Edge Cases**: Include boundary conditions and error cases
5. **Keep Tests Fast**: Avoid slow operations, use mocks
6. **Test Independence**: Tests should not depend on each other
7. **Clear Assertions**: Use descriptive assertion messages

## Future Improvements

- [ ] Add performance/load tests
- [ ] Add end-to-end tests with real services
- [ ] Add contract tests for API
- [ ] Add mutation testing
- [ ] Increase coverage to 80%+

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Coverage](https://pytest-cov.readthedocs.io/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)

