# Developer Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Project Structure](#project-structure)
3. [Development Workflow](#development-workflow)
4. [Code Standards](#code-standards)
5. [Testing](#testing)
6. [API Development](#api-development)
7. [Database Management](#database-management)
8. [Deployment](#deployment)

## Getting Started

### Prerequisites

- Python 3.12+
- PostgreSQL 12+
- Kafka (optional, for real-time streaming)
- Redis (optional, for caching)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd i-drill/src/backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements_backend.txt
   pip install -r requirements_test.txt
   ```

4. **Configure environment**
   ```bash
   cp config.env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize database**
   ```bash
   python setup_backend.py
   ```

6. **Run the server**
   ```bash
   python start_server.py
   ```

## Project Structure

```
src/backend/
├── api/                    # API layer
│   ├── models/            # Data models and schemas
│   │   ├── database_models.py  # SQLAlchemy models
│   │   └── schemas.py     # Pydantic schemas
│   ├── routes/            # API route handlers
│   │   ├── auth.py        # Authentication routes
│   │   ├── sensor_data.py # Sensor data routes
│   │   ├── predictions.py # Prediction routes
│   │   └── ...
│   ├── dependencies.py     # FastAPI dependencies
│   └── error_handlers.py  # Error handling
├── services/               # Business logic layer
│   ├── auth_service.py    # Authentication service
│   ├── data_service.py    # Data operations service
│   ├── kafka_service.py   # Kafka streaming service
│   └── ...
├── utils/                  # Utility functions
│   ├── validators.py      # Input validation
│   ├── security.py        # Security utilities
│   └── ...
├── database.py             # Database manager
├── app.py                  # FastAPI application
├── tests/                  # Test suite
│   ├── conftest.py        # Pytest fixtures
│   ├── test_api_routes.py # API route tests
│   └── ...
└── migrations/             # Database migrations
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code following the code standards
- Add tests for new functionality
- Update documentation

### 3. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_api_routes.py
```

### 4. Check Code Quality

```bash
# Linting (if configured)
flake8 .

# Type checking (if configured)
mypy .
```

### 5. Commit Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

## Code Standards

### Python Style Guide

Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with the following additions:

- **Line length**: 100 characters (soft limit)
- **Type hints**: Use type hints for all function signatures
- **Docstrings**: Use Google-style docstrings

### Type Hints

Always include type hints:

```python
from typing import Optional, List, Dict, Any

def get_sensor_data(
    rig_id: str,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Get sensor data for a rig."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_rop(
    depth: float,
    time_elapsed: float
) -> float:
    """
    Calculate rate of penetration (ROP).
    
    Args:
        depth: Current drilling depth in meters
        time_elapsed: Time elapsed in hours
        
    Returns:
        Rate of penetration in meters per hour
        
    Raises:
        ValueError: If depth or time_elapsed is negative
    """
    if depth < 0 or time_elapsed < 0:
        raise ValueError("Depth and time must be non-negative")
    return depth / time_elapsed if time_elapsed > 0 else 0.0
```

### Error Handling

Always handle errors appropriately:

```python
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
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

logger.info("Operation started", extra={"rig_id": rig_id})
logger.error("Operation failed", exc_info=True)
```

## Testing

### Writing Tests

1. **Unit Tests**: Test individual functions/methods
2. **Integration Tests**: Test component interactions
3. **API Tests**: Test HTTP endpoints

### Test Structure

```python
import pytest
from unittest.mock import Mock, patch

class TestFeature:
    """Tests for Feature class."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        feature = Feature()
        
        # Act
        result = feature.do_something()
        
        # Assert
        assert result is not None
    
    @pytest.fixture
    def sample_data(self):
        """Sample data fixture."""
        return {"key": "value"}
    
    def test_with_fixture(self, sample_data):
        """Test using fixture."""
        assert sample_data["key"] == "value"
```

### Test Coverage

Maintain minimum 60% test coverage. Aim for:
- Critical paths: 80%+
- Services: 70%+
- API routes: 75%+

## API Development

### Creating a New Endpoint

1. **Define the schema** (`api/models/schemas.py`):
   ```python
   class NewRequest(BaseModel):
       field1: str
       field2: int
       
       class Config:
           schema_extra = {
               "example": {
                   "field1": "value1",
                   "field2": 42
               }
           }
   ```

2. **Create the route** (`api/routes/new_route.py`):
   ```python
   from fastapi import APIRouter
   from api.models.schemas import NewRequest
   
   router = APIRouter(prefix="/new", tags=["new"])
   
   @router.post("/", response_model=dict)
   async def create_new(request: NewRequest):
       """Create a new resource."""
       # Implementation
       return {"success": True}
   ```

3. **Register the router** (`app.py`):
   ```python
   from api.routes import new_route
   
   app.include_router(
       new_route.router,
       prefix="/api/v1",
       tags=["New"]
   )
   ```

### Authentication

Use dependency injection for authentication:

```python
from api.dependencies import get_current_active_user
from api.models.database_models import UserDB

@router.get("/protected")
async def protected_endpoint(
    current_user: UserDB = Depends(get_current_active_user)
):
    """Protected endpoint requiring authentication."""
    return {"user": current_user.username}
```

### Role-Based Access

```python
from api.dependencies import get_current_engineer_user

@router.post("/admin-only")
async def admin_endpoint(
    current_user: UserDB = Depends(get_current_engineer_user)
):
    """Endpoint requiring engineer role."""
    ...
```

## Database Management

### Creating Migrations

1. Create migration SQL file in `migrations/`:
   ```sql
   -- migrations/add_new_table.sql
   CREATE TABLE new_table (
       id SERIAL PRIMARY KEY,
       name VARCHAR(255) NOT NULL,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

2. Apply migration:
   ```bash
   psql -d drilling_db -f migrations/add_new_table.sql
   ```

### Database Models

Define models in `api/models/database_models.py`:

```python
from sqlalchemy import Column, Integer, String, DateTime
from database import Base

class NewModel(Base):
    __tablename__ = "new_table"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
```

## Deployment

### Environment Variables

Set production environment variables:

```bash
APP_ENV=production
DATABASE_URL=postgresql://user:pass@host:port/db
SECRET_KEY=<secure-random-key>
CORS_ORIGINS=https://yourdomain.com
```

### Running in Production

1. **Use production WSGI server**:
   ```bash
   gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

2. **Set up reverse proxy** (nginx):
   ```nginx
   location /api {
       proxy_pass http://localhost:8001;
   }
   ```

3. **Enable HTTPS**: Use Let's Encrypt or similar

4. **Monitor**: Set up logging and monitoring

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
