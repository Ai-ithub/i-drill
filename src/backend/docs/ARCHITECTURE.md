# i-Drill Backend Architecture

## Overview

This document describes the architecture, design patterns, and component structure of the i-Drill backend API.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                            │
│  (React Frontend, Mobile Apps, External Integrations)          │
└────────────────────────────┬──────────────────────────────────┘
                              │
                              │ HTTP/REST + WebSocket
                              │
┌─────────────────────────────▼──────────────────────────────────┐
│                      API Gateway Layer                         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              FastAPI Application (app.py)               │  │
│  │  - Request Routing                                      │  │
│  │  - Authentication/Authorization                        │  │
│  │  - Request Validation                                  │  │
│  │  - Error Handling                                      │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬──────────────────────────────────┘
                              │
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐  ┌─────────▼────────┐  ┌───────▼────────┐
│  Route Layer   │  │  Service Layer   │  │  Data Layer     │
│  (api/routes)  │  │  (services)     │  │  (database)     │
└────────────────┘  └─────────────────┘  └────────────────┘
```

## Component Layers

### 1. Route Layer (`api/routes/`)

**Purpose**: Handle HTTP requests, validate input, and return responses.

**Responsibilities**:
- Request/response handling
- Input validation using Pydantic schemas
- Authentication/authorization checks
- Error handling and HTTP status codes
- Response serialization

**Key Files**:
- `sensor_data.py` - Sensor data endpoints
- `auth.py` - Authentication endpoints
- `predictions.py` - ML prediction endpoints
- `maintenance.py` - Maintenance management
- `control.py` - Control and change management
- `health.py` - Health check endpoints

**Example**:
```python
@router.get("/realtime", response_model=SensorDataResponse)
async def get_realtime_data(
    rig_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000)
):
    """Get real-time sensor data"""
    return data_service.get_latest_sensor_data(rig_id, limit)
```

### 2. Service Layer (`services/`)

**Purpose**: Business logic and orchestration.

**Responsibilities**:
- Business logic implementation
- Data transformation
- External service integration (Kafka, MLflow)
- Caching strategies
- Error handling and retry logic

**Key Services**:

#### DataService
- Sensor data CRUD operations
- Historical data queries
- Analytics calculations
- Data aggregation

#### AuthService
- User authentication
- Password hashing and verification
- JWT token management
- Password reset workflow
- Account lockout handling

#### PredictionService
- RUL (Remaining Useful Life) predictions
- Anomaly detection
- Model loading and inference
- Prediction history management

#### KafkaService
- Kafka producer/consumer management
- Message serialization
- Connection retry logic
- Error handling

#### DVRService
- Data validation
- Reconciliation logic
- Anomaly detection
- Data quality metrics

**Example**:
```python
class DataService:
    def get_latest_sensor_data(self, rig_id: str, limit: int):
        """Business logic for fetching latest data"""
        # Validation, query, transformation
        return processed_data
```

### 3. Data Layer (`database.py`, `api/models/`)

**Purpose**: Database access and data models.

**Responsibilities**:
- Database connection management
- ORM model definitions
- Query optimization
- Transaction management
- Migration support

**Key Components**:

#### DatabaseManager
- Connection pooling
- Session management
- Health checks
- Transaction handling

#### Models (`api/models/database_models.py`)
- SQLAlchemy ORM models
- Relationships and constraints
- Index definitions

#### Schemas (`api/models/schemas.py`)
- Pydantic models for validation
- Request/response schemas
- Serialization rules

**Example**:
```python
class SensorData(Base):
    __tablename__ = "sensor_data"
    id = Column(Integer, primary_key=True)
    rig_id = Column(String(50), index=True)
    timestamp = Column(DateTime, index=True)
    # ...
```

## Design Patterns

### 1. Dependency Injection

Services are injected into route handlers via FastAPI's dependency system:

```python
from api.dependencies import get_current_user

@router.get("/protected")
async def protected_endpoint(user: UserDB = Depends(get_current_user)):
    # User is automatically injected
    pass
```

### 2. Repository Pattern

Services act as repositories, abstracting database access:

```python
class DataService:
    def get_latest_sensor_data(self, rig_id: str):
        # Abstracts database query
        with self.db_manager.session_scope() as session:
            return session.query(SensorData).filter(...).all()
```

### 3. Service Layer Pattern

Business logic is separated from route handlers:

```python
# Route handler (thin)
@router.post("/sensor-data")
async def create_data(data: SensorDataPoint):
    return data_service.insert_sensor_data(data.dict())

# Service (thick, contains logic)
class DataService:
    def insert_sensor_data(self, data: dict):
        # Validation, transformation, business rules
        validate_sensor_data(data)
        processed = self._process_data(data)
        return self._save_to_db(processed)
```

### 4. Factory Pattern

Services use factory methods for complex object creation:

```python
class KafkaService:
    def _initialize_producer(self):
        # Factory method for creating producers
        return Producer(config)
```

## Data Flow

### Request Flow

```
1. Client Request
   ↓
2. FastAPI Router
   - Route matching
   - Dependency injection
   - Request validation
   ↓
3. Route Handler
   - Authentication check
   - Input validation
   - Service call
   ↓
4. Service Layer
   - Business logic
   - Data transformation
   - External service calls
   ↓
5. Data Layer
   - Database query
   - Data retrieval
   ↓
6. Response
   - Serialization
   - HTTP response
```

### Real-Time Data Flow

```
1. Kafka Producer (External)
   ↓
2. Kafka Topic
   ↓
3. Data Bridge Service
   - Message consumption
   - Data processing
   ↓
4. Database Storage
   ↓
5. WebSocket Manager
   - Broadcast to clients
   ↓
6. Connected Clients
```

## Security Architecture

### Authentication Flow

```
1. User Login
   ↓
2. AuthService.authenticate_user()
   - Verify credentials
   - Check account status
   - Track login attempts
   ↓
3. Generate Tokens
   - Access token (short-lived)
   - Refresh token (long-lived)
   ↓
4. Return Tokens
   ↓
5. Client Stores Tokens
   ↓
6. Subsequent Requests
   - Include access token
   - Token validation
   - User context injection
```

### Authorization

Role-based access control (RBAC):

- **Admin**: Full access
- **Engineer**: Technical operations, predictions, control
- **Operator**: Operations, monitoring
- **Maintenance**: Maintenance management
- **Viewer**: Read-only access

## Error Handling

### Error Hierarchy

```
HTTPException (FastAPI)
├── 400 Bad Request (Validation errors)
├── 401 Unauthorized (Authentication)
├── 403 Forbidden (Authorization)
├── 404 Not Found (Resource not found)
├── 422 Unprocessable Entity (Pydantic validation)
└── 500 Internal Server Error (Unexpected errors)
```

### Error Handling Strategy

1. **Route Level**: Catch and return appropriate HTTP status
2. **Service Level**: Handle business logic errors
3. **Data Level**: Handle database errors
4. **Global**: Catch-all error handler

## Caching Strategy

### Cache Layers

1. **In-Memory Cache** (Service level)
   - Frequently accessed data
   - Short TTL

2. **Redis Cache** (Optional)
   - Session data
   - Computed results
   - Rate limiting

### Cache Invalidation

- Time-based expiration
- Event-based invalidation
- Manual cache clearing

## Testing Strategy

### Test Pyramid

```
        /\
       /  \  E2E Tests (Few)
      /────\
     /      \  Integration Tests (Some)
    /────────\
   /          \  Unit Tests (Many)
  /────────────\
```

### Test Types

1. **Unit Tests**: Services, utilities, validators
2. **Integration Tests**: API endpoints, database operations
3. **E2E Tests**: Complete workflows

## Performance Considerations

### Optimization Strategies

1. **Database**:
   - Indexes on frequently queried columns
   - Query optimization
   - Connection pooling

2. **Caching**:
   - Cache expensive computations
   - Cache frequently accessed data

3. **Async Operations**:
   - Async/await for I/O operations
   - Background tasks for heavy operations

4. **Pagination**:
   - Limit result sets
   - Cursor-based pagination for large datasets

## Monitoring and Observability

### Logging

- Structured logging with levels (DEBUG, INFO, WARNING, ERROR)
- Request/response logging
- Error tracking with stack traces

### Metrics

- Request counts and latencies
- Error rates
- Database query performance
- Service health status

### Health Checks

- `/api/v1/health/` - Basic health
- `/api/v1/health/services` - Service status
- `/api/v1/health/ready` - Readiness probe
- `/api/v1/health/live` - Liveness probe

## Deployment Architecture

### Development

```
Developer Machine
├── Python Virtual Environment
├── Local PostgreSQL
├── Local Kafka (optional)
└── Local Redis (optional)
```

### Production

```
Load Balancer
    ↓
API Servers (Multiple instances)
    ↓
Database (PostgreSQL - Primary/Replica)
    ↓
Message Queue (Kafka Cluster)
    ↓
Cache (Redis Cluster)
    ↓
ML Service (MLflow Server)
```

## Future Enhancements

1. **Microservices**: Split into smaller services
2. **Event Sourcing**: For audit trails
3. **CQRS**: Separate read/write models
4. **GraphQL**: Alternative API interface
5. **gRPC**: For internal service communication

