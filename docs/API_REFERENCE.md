# 📡 مرجع API - i-Drill

راهنمای کامل API های i-Drill برای توسعه‌دهندگان.

---

## 📋 فهرست مطالب

1. [مقدمه](#مقدمه)
2. [Authentication](#authentication)
3. [Endpoints](#endpoints)
4. [WebSocket API](#websocket-api)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [Examples](#examples)

---

## مقدمه

### Base URL

```
Development: http://localhost:8001
Production: https://api.yourdomain.com
```

### API Version

همه endpoints در مسیر `/api/v1/` قرار دارند.

### Content Type

- **Request**: `application/json` یا `application/x-www-form-urlencoded`
- **Response**: `application/json`

---

## Authentication

### JWT Token Authentication

اکثر endpoints نیاز به authentication دارند. برای دریافت token:

```bash
POST /api/v1/auth/login
Content-Type: application/x-www-form-urlencoded

username=admin&password=your_password
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400,
  "refresh_token": "refresh_token_here"
}
```

### استفاده از Token

```bash
GET /api/v1/sensor-data/realtime
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Refresh Token

```bash
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "your_refresh_token"
}
```

---

## Endpoints

### 🔐 Authentication

#### Login
```http
POST /api/v1/auth/login
Content-Type: application/x-www-form-urlencoded

username=admin&password=password
```

#### Register
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "username": "newuser",
  "email": "user@example.com",
  "password": "SecurePassword123!",
  "full_name": "New User",
  "role": "viewer"
}
```

#### Get Current User
```http
GET /api/v1/auth/me
Authorization: Bearer {token}
```

#### Refresh Token
```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "refresh_token_here"
}
```

---

### 📊 Sensor Data

#### Get Real-time Data
```http
GET /api/v1/sensor-data/realtime?rig_id=RIG_01&limit=10
Authorization: Bearer {token}
```

#### Get Historical Data
```http
GET /api/v1/sensor-data/historical?rig_id=RIG_01&start_time=2024-01-01T00:00:00&end_time=2024-01-02T00:00:00&limit=100
Authorization: Bearer {token}
```

#### Create Sensor Data
```http
POST /api/v1/sensor-data/
Authorization: Bearer {token}
Content-Type: application/json

{
  "rig_id": "RIG_01",
  "timestamp": "2024-01-01T00:00:00",
  "depth": 5000.0,
  "wob": 15000.0,
  "rpm": 120.0,
  "torque": 8000.0,
  "rop": 50.0,
  "mud_flow": 500.0,
  "mud_pressure": 3000.0
}
```

---

### 🤖 Predictions

#### Get RUL Prediction
```http
POST /api/v1/predictions/rul
Authorization: Bearer {token}
Content-Type: application/json

{
  "rig_id": "RIG_01",
  "model_type": "LSTM",
  "historical_data": [...]
}
```

#### Get Anomaly Detection
```http
POST /api/v1/predictions/anomaly
Authorization: Bearer {token}
Content-Type: application/json

{
  "rig_id": "RIG_01",
  "sensor_data": {...}
}
```

---

### 🎮 Control

#### Apply Parameter Change
```http
POST /api/v1/control/apply-change
Authorization: Bearer {token}
Content-Type: application/json

{
  "rig_id": "RIG_01",
  "component": "drill",
  "parameter": "rpm",
  "new_value": 150.0,
  "metadata": {
    "reason": "Optimization",
    "operator": "admin"
  }
}
```

#### Get Change History
```http
GET /api/v1/control/history?rig_id=RIG_01&limit=10
Authorization: Bearer {token}
```

---

### 🔧 Maintenance

#### Get Maintenance Alerts
```http
GET /api/v1/maintenance/alerts?rig_id=RIG_01&status=active
Authorization: Bearer {token}
```

#### Create Maintenance Schedule
```http
POST /api/v1/maintenance/schedule
Authorization: Bearer {token}
Content-Type: application/json

{
  "rig_id": "RIG_01",
  "component": "drill_bit",
  "maintenance_type": "preventive",
  "scheduled_date": "2024-02-01T00:00:00",
  "description": "Routine maintenance"
}
```

---

### 📈 DVR (Data Validation & Reconciliation)

#### Process Record
```http
POST /api/v1/dvr/process
Authorization: Bearer {token}
Content-Type: application/json

{
  "rig_id": "RIG_01",
  "sensor_data": {...}
}
```

#### Get Statistics
```http
GET /api/v1/dvr/stats?rig_id=RIG_01
Authorization: Bearer {token}
```

---

### 🤖 Reinforcement Learning

#### Get RL State
```http
GET /api/v1/rl/state?rig_id=RIG_01
Authorization: Bearer {token}
```

#### Apply RL Action
```http
POST /api/v1/rl/action
Authorization: Bearer {token}
Content-Type: application/json

{
  "rig_id": "RIG_01",
  "action": [0.5, 0.3, 0.2]
}
```

---

## WebSocket API

### اتصال WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8001/api/v1/sensor-data/ws/RIG_01?token=YOUR_TOKEN');

ws.onopen = () => {
  console.log('Connected');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected');
};
```

### پیام‌های WebSocket

#### Sensor Data Message
```json
{
  "message_type": "sensor_data",
  "data": {
    "rig_id": "RIG_01",
    "timestamp": "2024-01-01T00:00:00",
    "depth": 5000.0,
    "wob": 15000.0,
    "rpm": 120.0
  },
  "timestamp": "2024-01-01T00:00:00"
}
```

#### Status Update
```json
{
  "message_type": "status_update",
  "data": {
    "status": "connected",
    "rig_id": "RIG_01"
  },
  "timestamp": "2024-01-01T00:00:00"
}
```

---

## Error Handling

### Error Response Format

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {}
  },
  "trace_id": "uuid",
  "timestamp": "2024-01-01T00:00:00",
  "path": "/api/v1/endpoint"
}
```

### HTTP Status Codes

- `200 OK` - Request successful
- `201 Created` - Resource created
- `400 Bad Request` - Invalid request
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `422 Unprocessable Entity` - Validation error
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

### Error Codes

- `AUTH_REQUIRED` - Authentication required
- `INVALID_CREDENTIALS` - Invalid username/password
- `TOKEN_EXPIRED` - Token has expired
- `INSUFFICIENT_PERMISSIONS` - User lacks required permissions
- `VALIDATION_ERROR` - Input validation failed
- `RESOURCE_NOT_FOUND` - Resource not found
- `RATE_LIMIT_EXCEEDED` - Too many requests

---

## Rate Limiting

### Rate Limits

- **Default**: 100 requests/minute
- **Authentication**: 5 requests/minute
- **Predictions**: 20 requests/minute
- **Sensor Data**: 200 requests/minute

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

---

## Examples

### Python Example

```python
import requests

# Login
response = requests.post(
    "http://localhost:8001/api/v1/auth/login",
    data={"username": "admin", "password": "password"}
)
token = response.json()["access_token"]

# Get sensor data
headers = {"Authorization": f"Bearer {token}"}
response = requests.get(
    "http://localhost:8001/api/v1/sensor-data/realtime",
    headers=headers,
    params={"rig_id": "RIG_01", "limit": 10}
)
data = response.json()
```

### JavaScript Example

```javascript
// Login
const loginResponse = await fetch('http://localhost:8001/api/v1/auth/login', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/x-www-form-urlencoded',
  },
  body: new URLSearchParams({
    username: 'admin',
    password: 'password'
  })
});

const { access_token } = await loginResponse.json();

// Get sensor data
const dataResponse = await fetch(
  'http://localhost:8001/api/v1/sensor-data/realtime?rig_id=RIG_01&limit=10',
  {
    headers: {
      'Authorization': `Bearer ${access_token}`
    }
  }
);

const data = await dataResponse.json();
```

### cURL Example

```bash
# Login
curl -X POST http://localhost:8001/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=password"

# Get sensor data
curl -X GET "http://localhost:8001/api/v1/sensor-data/realtime?rig_id=RIG_01&limit=10" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## 📚 منابع بیشتر

- [Interactive API Docs](http://localhost:8001/docs) - Swagger UI
- [ReDoc](http://localhost:8001/redoc) - Alternative API docs
- [OpenAPI Schema](http://localhost:8001/openapi.json) - OpenAPI specification

---

**آخرین به‌روزرسانی:** ژانویه 2025  
**نسخه API:** 1.0

