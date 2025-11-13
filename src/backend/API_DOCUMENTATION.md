# i-Drill Backend API Documentation

Comprehensive API documentation for the i-Drill drilling operations control system.

## 📋 Table of Contents

- [Overview](#overview)
- [Base URL](#base-url)
- [Authentication](#authentication)
- [API Endpoints](#api-endpoints)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [WebSocket Support](#websocket-support)
- [Examples](#examples)

## 🎯 Overview

The i-Drill API provides a comprehensive RESTful interface for:
- Real-time sensor data monitoring
- Historical data analysis
- Predictive maintenance
- Drilling optimization
- Control system management
- User authentication and authorization

### API Version

Current version: **v1**

Base path: `/api/v1`

## 🌐 Base URL

**Development**: `http://localhost:8001/api/v1`

**Production**: `https://api.idrill.example.com/api/v1`

## 🔐 Authentication

Most endpoints require authentication using JWT (JSON Web Tokens).

### Getting an Access Token

```http
POST /api/v1/auth/login
Content-Type: application/x-www-form-urlencoded

username=your_username&password=your_password
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400,
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### Using the Access Token

Include the token in the `Authorization` header:

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Token Refresh

```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

## 📡 API Endpoints

### Health Check

#### GET /health/

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### GET /health/services

Get detailed service health status.

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "database": {
      "status": "healthy",
      "response_time_ms": 5
    },
    "kafka": {
      "status": "healthy",
      "response_time_ms": 10
    }
  }
}
```

### Sensor Data

#### GET /sensor-data/realtime

Get real-time sensor data.

**Query Parameters:**
- `rig_id` (optional): Filter by rig ID
- `limit` (optional, default: 100): Number of records to return
- `offset` (optional, default: 0): Pagination offset

**Response:**
```json
{
  "success": true,
  "count": 10,
  "data": [
    {
      "id": 1,
      "rig_id": "RIG_01",
      "timestamp": "2024-01-01T00:00:00Z",
      "depth": 5000.0,
      "wob": 15000.0,
      "rpm": 100.0,
      "torque": 10000.0,
      "rop": 50.0,
      "mud_flow": 800.0,
      "mud_pressure": 3000.0
    }
  ]
}
```

#### GET /sensor-data/historical

Get historical sensor data.

**Query Parameters:**
- `rig_id` (optional): Filter by rig ID
- `start_time` (required): Start time (ISO 8601)
- `end_time` (required): End time (ISO 8601)
- `limit` (optional, default: 1000): Number of records
- `offset` (optional, default: 0): Pagination offset
- `parameters` (optional): Comma-separated list of parameters

**Example:**
```http
GET /api/v1/sensor-data/historical?start_time=2024-01-01T00:00:00Z&end_time=2024-01-02T00:00:00Z&rig_id=RIG_01
```

#### GET /sensor-data/analytics/{rig_id}

Get analytics summary for a rig.

**Path Parameters:**
- `rig_id`: Rig identifier

**Response:**
```json
{
  "rig_id": "RIG_01",
  "current_depth": 5000.0,
  "average_rop": 50.0,
  "total_power_consumption": 200000.0,
  "maintenance_alerts_count": 2,
  "total_drilling_time_hours": 120.5,
  "last_updated": "2024-01-01T00:00:00Z"
}
```

#### POST /sensor-data/

Create a new sensor data record.

**Request Body:**
```json
{
  "rig_id": "RIG_01",
  "timestamp": "2024-01-01T00:00:00Z",
  "depth": 5000.0,
  "wob": 15000.0,
  "rpm": 100.0,
  "torque": 10000.0,
  "rop": 50.0,
  "mud_flow": 800.0,
  "mud_pressure": 3000.0
}
```

**Response:**
```json
{
  "success": true,
  "message": "Sensor data created successfully",
  "data": { ... }
}
```

### Authentication

#### POST /auth/register

Register a new user.

**Request Body:**
```json
{
  "username": "newuser",
  "email": "user@example.com",
  "password": "SecurePassword123!",
  "role": "engineer"
}
```

#### POST /auth/login

Login and get access token.

**Request Body (form-data):**
```
username=your_username&password=your_password
```

#### POST /auth/logout

Logout and invalidate token.

**Headers:**
- `Authorization: Bearer <token>`

#### GET /auth/me

Get current user profile.

**Headers:**
- `Authorization: Bearer <token>`

#### POST /auth/password/reset/request

Request password reset.

**Request Body:**
```json
{
  "email": "user@example.com"
}
```

#### POST /auth/password/reset/confirm

Confirm password reset.

**Request Body:**
```json
{
  "token": "reset_token_here",
  "new_password": "NewSecurePassword123!"
}
```

### Control

#### POST /control/apply-change

Apply a change to drilling parameters.

**Request Body:**
```json
{
  "rig_id": "RIG_01",
  "change_type": "parameter",
  "component": "drilling",
  "parameter": "rpm",
  "value": 120.0,
  "auto_execute": false,
  "metadata": {}
}
```

**Response:**
```json
{
  "success": true,
  "change_id": 123,
  "status": "pending",
  "message": "Change request created, pending approval"
}
```

#### GET /control/change-history

Get change request history.

**Query Parameters:**
- `rig_id` (optional): Filter by rig ID
- `status` (optional): Filter by status (pending, approved, rejected, applied)
- `limit` (optional): Number of records
- `offset` (optional): Pagination offset

#### POST /control/change/{id}/approve

Approve a change request.

**Path Parameters:**
- `id`: Change request ID

#### POST /control/change/{id}/reject

Reject a change request.

**Request Body:**
```json
{
  "reason": "Reason for rejection"
}
```

### Maintenance

#### GET /maintenance/alerts

Get maintenance alerts.

**Query Parameters:**
- `rig_id` (optional): Filter by rig ID
- `severity` (optional): Filter by severity (low, medium, high, critical)
- `status` (optional): Filter by status (active, acknowledged, resolved)

#### GET /maintenance/schedule

Get maintenance schedule.

**Query Parameters:**
- `rig_id` (optional): Filter by rig ID
- `start_date` (optional): Start date filter
- `end_date` (optional): End date filter

### Predictions

#### POST /predictions/anomaly-detection

Detect anomalies in sensor data.

**Request Body:**
```json
{
  "rig_id": "RIG_01",
  "sensor_data": {
    "depth": 5000.0,
    "wob": 15000.0,
    "rpm": 100.0
  }
}
```

**Response:**
```json
{
  "is_anomaly": false,
  "anomaly_score": 0.15,
  "confidence": 0.95
}
```

#### POST /predictions/rul

Predict Remaining Useful Life (RUL).

**Request Body:**
```json
{
  "rig_id": "RIG_01",
  "component": "motor"
}
```

**Response:**
```json
{
  "rig_id": "RIG_01",
  "component": "motor",
  "rul_hours": 500.0,
  "confidence": 0.88,
  "prediction_date": "2024-01-01T00:00:00Z"
}
```

## 📊 Data Models

### SensorDataPoint

```json
{
  "rig_id": "string (required)",
  "timestamp": "datetime (required)",
  "depth": "float (optional)",
  "wob": "float (optional)",
  "rpm": "float (optional)",
  "torque": "float (optional)",
  "rop": "float (optional)",
  "mud_flow": "float (optional)",
  "mud_pressure": "float (optional)",
  "mud_temperature": "float (optional)",
  "bit_temperature": "float (optional)",
  "motor_temperature": "float (optional)",
  "power_consumption": "float (optional)",
  "vibration_level": "float (optional)"
}
```

### ChangeRequest

```json
{
  "rig_id": "string (required)",
  "change_type": "parameter | maintenance | configuration",
  "component": "string (required)",
  "parameter": "string (required)",
  "value": "any (required)",
  "auto_execute": "boolean (default: false)",
  "metadata": "object (optional)"
}
```

## ⚠️ Error Handling

### Error Response Format

```json
{
  "detail": "Error message",
  "status_code": 400,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### HTTP Status Codes

- `200 OK`: Request successful
- `201 Created`: Resource created
- `400 Bad Request`: Invalid request
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service unavailable

## 🚦 Rate Limiting

Rate limits (if enabled):
- **Default**: 100 requests/minute
- **Authentication**: 5 requests/minute
- **Predictions**: 20 requests/minute
- **Sensor Data**: 200 requests/minute

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## 🔌 WebSocket Support

### WebSocket Endpoint

```
ws://localhost:8001/api/v1/sensor-data/ws/{rig_id}
```

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8001/api/v1/sensor-data/ws/RIG_01');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Sensor data:', data);
};
```

### Message Format

**Server to Client:**
```json
{
  "message_type": "sensor_data",
  "data": {
    "rig_id": "RIG_01",
    "timestamp": "2024-01-01T00:00:00Z",
    "depth": 5000.0,
    ...
  }
}
```

**Client to Server:**
```json
{
  "message_type": "ping"
}
```

## 💡 Examples

### Python Example

```python
import requests

# Login
response = requests.post(
    "http://localhost:8001/api/v1/auth/login",
    data={
        "username": "your_username",
        "password": "your_password"
    }
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
  headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  body: 'username=your_username&password=your_password'
});
const { access_token } = await loginResponse.json();

// Get sensor data
const response = await fetch(
  'http://localhost:8001/api/v1/sensor-data/realtime?rig_id=RIG_01&limit=10',
  {
    headers: { 'Authorization': `Bearer ${access_token}` }
  }
);
const data = await response.json();
```

### cURL Example

```bash
# Login
TOKEN=$(curl -X POST http://localhost:8001/api/v1/auth/login \
  -d "username=your_username&password=your_password" \
  | jq -r '.access_token')

# Get sensor data
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8001/api/v1/sensor-data/realtime?rig_id=RIG_01"
```

## 📚 Additional Resources

- [Interactive API Documentation (Swagger)](http://localhost:8001/docs)
- [ReDoc Documentation](http://localhost:8001/redoc)
- [Test Suite Documentation](tests/README.md)
- [Security Guide](SECURITY.md)

## 🔄 API Versioning

The API uses URL-based versioning:
- Current version: `/api/v1`
- Future versions: `/api/v2`, `/api/v3`, etc.

Breaking changes will result in a new version number.

## 📝 Changelog

### v1.0.0 (2024-01-01)
- Initial API release
- Sensor data endpoints
- Authentication system
- Control endpoints
- Maintenance endpoints
- Prediction endpoints
