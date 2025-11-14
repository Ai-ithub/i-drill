# Development Guide

Guide for developers working on the i-Drill backend.

## Development Setup

### Prerequisites

- Python 3.12+
- PostgreSQL 12+
- Git
- Virtual environment tool (venv, virtualenv, or conda)

### Initial Setup

```bash
# Clone repository
git clone <repository-url>
cd i-drill/src/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_backend.txt
pip install -r requirements_test.txt

# Copy configuration
cp config.env.example .env

# Edit .env with your settings
# Generate secret key
python scripts/generate_secret_key.py

# Initialize database
python setup_backend.py

# Run tests
pytest
```

## Code Structure

```
src/backend/
├── api/                    # API layer
│   ├── routes/            # Route handlers
│   ├── models/            # Data models and schemas
│   ├── dependencies.py    # Dependency injection
│   └── exceptions.py      # Custom exceptions
├── services/              # Business logic layer
│   ├── auth_service.py
│   ├── data_service.py
│   └── ...
├── utils/                 # Utility functions
│   ├── validators.py
│   └── security.py
├── database.py            # Database manager
├── app.py                 # FastAPI application
└── tests/                 # Test suite
```

## Coding Standards

### Python Style

- Follow PEP 8 style guide
- Use type hints for all functions
- Maximum line length: 100 characters
- Use black for code formatting (optional)

### Type Hints

Always use type hints:

```python
def get_sensor_data(rig_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Get sensor data"""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def process_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process sensor data.
    
    Args:
        data: Raw sensor data dictionary
        
    Returns:
        Processed data dictionary
        
    Raises:
        ValueError: If data is invalid
        
    Example:
        >>> data = {"rig_id": "RIG_01", "depth": 5000.0}
        >>> processed = process_data(data)
        >>> processed["rig_id"]
        'RIG_01'
    """
    pass
```

### Error Handling

Always handle errors appropriately:

```python
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Error in operation: {e}")
    raise HTTPException(status_code=500, detail=str(e))
except Exception as e:
    logger.exception("Unexpected error")
    raise HTTPException(status_code=500, detail="Internal server error")
```

### Logging

Use structured logging:

```python
import logging

logger = logging.getLogger(__name__)

logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message", exc_info=True)
```

## Adding New Features

### 1. Create Feature Branch

```bash
git checkout -b feature/new-feature
```

### 2. Implement Feature

- Add route handlers in `api/routes/`
- Add business logic in `services/`
- Add data models in `api/models/`
- Add validation in `utils/validators.py`

### 3. Add Tests

- Unit tests in `tests/test_services.py`
- Integration tests in `tests/test_api_routes.py`
- Aim for 80%+ coverage for new code

### 4. Update Documentation

- Update API documentation
- Add docstrings
- Update README if needed

### 5. Submit Pull Request

- Ensure all tests pass
- Ensure code coverage is maintained
- Get code review
- Merge after approval

## Testing

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_validators.py

# With coverage
pytest --cov=. --cov-report=html

# With markers
pytest -m unit
pytest -m integration
```

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch

def test_function_name():
    """Test description"""
    # Arrange
    input_data = {"key": "value"}
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result is not None
    assert result["key"] == "value"
```

### Test Organization

- **Unit Tests**: Test individual functions/classes
- **Integration Tests**: Test API endpoints
- **E2E Tests**: Test complete workflows

## Database Migrations

### Creating Migrations

```bash
# Using Alembic
alembic revision --autogenerate -m "description"
alembic upgrade head
```

### Manual SQL Migrations

Create SQL files in `migrations/`:

```sql
-- migrations/add_new_table.sql
CREATE TABLE new_table (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);
```

## API Development

### Adding New Endpoint

1. **Define Schema** (in `api/models/schemas.py`):

```python
class NewRequest(BaseModel):
    field: str = Field(..., description="Field description")
    
class NewResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
```

2. **Add Route** (in `api/routes/`):

```python
@router.post("/new-endpoint", response_model=NewResponse)
async def new_endpoint(request: NewRequest):
    """Endpoint description"""
    # Implementation
    pass
```

3. **Register Route** (in `app.py`):

```python
app.include_router(new_router, prefix="/api/v1", tags=["New"])
```

### Request Validation

Use Pydantic for validation:

```python
class RequestSchema(BaseModel):
    rig_id: str = Field(..., min_length=1, max_length=50)
    value: float = Field(..., ge=0, le=1000)
    
    @validator('rig_id')
    def validate_rig_id(cls, v):
        if not validate_rig_id(v):
            raise ValueError('Invalid rig_id format')
        return v
```

## Debugging

### Local Development

```bash
# Run with debug logging
LOG_LEVEL=DEBUG python start_server.py

# Run with reload
uvicorn app:app --reload --log-level debug
```

### Debugging Tips

1. Use logging extensively
2. Use breakpoints in IDE
3. Check database directly
4. Use API documentation for testing
5. Check logs for errors

## Performance Optimization

### Database Queries

- Use indexes on frequently queried columns
- Avoid N+1 queries
- Use pagination for large datasets
- Use connection pooling

### Caching

- Cache expensive computations
- Cache frequently accessed data
- Use Redis for distributed caching

### Async Operations

- Use async/await for I/O operations
- Use background tasks for heavy operations
- Avoid blocking operations

## Security

### Input Validation

Always validate and sanitize input:

```python
from utils.validators import validate_rig_id, sanitize_string

rig_id = sanitize_string(request.rig_id)
if not validate_rig_id(rig_id):
    raise HTTPException(status_code=400, detail="Invalid rig_id")
```

### Authentication

- Always check authentication for protected endpoints
- Use role-based access control
- Validate tokens properly

### Secrets

- Never commit secrets to repository
- Use environment variables
- Rotate secrets regularly

## Git Workflow

### Commit Messages

Use conventional commits:

```
feat: Add new sensor data endpoint
fix: Fix authentication token refresh
docs: Update API documentation
test: Add tests for validation utilities
refactor: Refactor data service
```

### Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation
- `test/` - Tests
- `refactor/` - Refactoring

## Code Review Checklist

- [ ] Code follows style guide
- [ ] Type hints are present
- [ ] Docstrings are complete
- [ ] Tests are added/updated
- [ ] All tests pass
- [ ] Code coverage maintained
- [ ] No security issues
- [ ] Error handling is appropriate
- [ ] Logging is adequate
- [ ] Documentation is updated

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Check DATABASE_URL in .env
   - Verify PostgreSQL is running
   - Check network connectivity

2. **Import Errors**
   - Verify virtual environment is activated
   - Check PYTHONPATH
   - Reinstall dependencies

3. **Authentication Errors**
   - Check SECRET_KEY is set
   - Verify token format
   - Check token expiration

4. **Kafka Connection Errors**
   - Verify Kafka is running
   - Check KAFKA_BOOTSTRAP_SERVERS
   - Check network connectivity

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)

