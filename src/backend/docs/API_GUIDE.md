# i-Drill API Usage Guide

Complete guide for using the i-Drill Backend API.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Authentication](#authentication)
3. [Sensor Data API](#sensor-data-api)
4. [Predictions API](#predictions-api)
5. [Maintenance API](#maintenance-api)
6. [Control API](#control-api)
7. [WebSocket API](#websocket-api)
8. [Error Handling](#error-handling)
9. [Rate Limiting](#rate-limiting)
10. [Best Practices](#best-practices)

## Getting Started

### Base URL

```
Development: http://localhost:8001
Production: https://api.yourdomain.com
```

### API Version

All endpoints are prefixed with `/api/v1`

### Content Types

- **Request**: `application/json` (for POST/PUT requests)
- **Response**: `application/json`

### Example Request

```bash
curl -X GET "http://localhost:8001/api/v1/health/" \
  -H "Content-Type: application/json"
```

## Authentication

### Overview

The API uses JWT (JSON Web Tokens) for authentication. Most endpoints require authentication.

### Register a User

```bash
POST /api/v1/auth/register

Request Body:
{
  "username": "engineer1",
  "email": "engineer@example.com",
  "password": "SecurePassword123!",
  "role": "engineer"
}

Response:
{
  "id": 1,
  "username": "engineer1",
  "email": "engineer@example.com",
  "role": "engineer",
  "is_active": true,
  "created_at": "2024-01-01T00:00:00"
}
```

### Login

```bash
POST /api/v1/auth/login

Content-Type: application/x-www-form-urlencoded

username=engineer1&password=SecurePassword123!

Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400,
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### Using the Token

Include the token in the `Authorization` header:

```bash
Authorization: Bearer <access_token>
```

### Refresh Token

```bash
POST /api/v1/auth/refresh

Request Body:
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}

Response:
{
  "access_token": "new_access_token...",
  "token_type": "bearer",
  "expires_in": 86400,
  "refresh_token": "new_refresh_token..."
}
```

### Logout

```bash
POST /api/v1/auth/logout

Headers:
Authorization: Bearer <access_token>

Response:
{
  "success": true,
  "message": "Logged out successfully"
}
```

## Sensor Data API

### Get Real-Time Data

```bash
GET /api/v1/sensor-data/realtime

Query Parameters:
- rig_id (optional): Filter by rig ID
- limit (optional): Number of records (default: 100, max: 1000)

Example:
GET /api/v1/sensor-data/realtime?rig_id=RIG_01&limit=10

Response:
{
  "success": true,
  "count": 10,
  "data": [
    {
      "id": 1,
      "rig_id": "RIG_01",
      "timestamp": "2024-01-01T12:00:00",
      "depth": 5000.0,
      "wob": 15000.0,
      "rpm": 100.0,
      "torque": 10000.0,
      "rop": 50.0,
      "mud_flow": 800.0,
      "mud_pressure": 3000.0,
      ...
    }
  ]
}
```

### Get Historical Data

```bash
GET /api/v1/sensor-data/historical

Query Parameters:
- rig_id (optional): Filter by rig ID
- start_time (required): ISO 8601 timestamp
- end_time (required): ISO 8601 timestamp
- parameters (optional): Comma-separated list of parameters
- limit (optional): Number of records (default: 1000)
- offset (optional): Pagination offset (default: 0)

Example:
GET /api/v1/sensor-data/historical?rig_id=RIG_01&start_time=2024-01-01T00:00:00&end_time=2024-01-02T00:00:00&limit=100

Response:
{
  "success": true,
  "count": 100,
  "data": [...],
  "pagination": {
    "limit": 100,
    "offset": 0,
    "total": 5000
  }
}
```

### Get Analytics Summary

```bash
GET /api/v1/sensor-data/analytics/{rig_id}

Example:
GET /api/v1/sensor-data/analytics/RIG_01

Response:
{
  "rig_id": "RIG_01",
  "current_depth": 5000.0,
  "average_rop": 12.5,
  "total_drilling_time_hours": 240.5,
  "total_power_consumption": 48000.0,
  "maintenance_alerts_count": 2,
  "last_updated": "2024-01-01T12:00:00"
}
```

### Create Sensor Data

```bash
POST /api/v1/sensor-data/

Headers:
Authorization: Bearer <access_token>
Content-Type: application/json

Request Body:
{
  "rig_id": "RIG_01",
  "timestamp": "2024-01-01T12:00:00",
  "depth": 5000.0,
  "wob": 15000.0,
  "rpm": 100.0,
  "torque": 10000.0,
  "rop": 50.0,
  "mud_flow": 800.0,
  "mud_pressure": 3000.0
}

Response:
{
  "success": true,
  "message": "Sensor data created successfully",
  "data": {...}
}
```

## Predictions API

### Predict RUL (Remaining Useful Life)

```bash
POST /api/v1/predictions/rul

Headers:
Authorization: Bearer <access_token>
Content-Type: application/json

Request Body:
{
  "rig_id": "RIG_01",
  "sensor_data": {
    "depth": 5000.0,
    "wob": 15000.0,
    "rpm": 100.0,
    "torque": 10000.0,
    "temperature": 90.0,
    "vibration": 0.8
  }
}

Response:
{
  "rig_id": "RIG_01",
  "rul_hours": 240.5,
  "confidence": 0.85,
  "prediction_date": "2024-01-01T12:00:00",
  "model_version": "v1.2.0"
}
```

### Anomaly Detection

```bash
POST /api/v1/predictions/anomaly-detection

Headers:
Authorization: Bearer <access_token>
Content-Type: application/json

Request Body:
{
  "rig_id": "RIG_01",
  "sensor_data": {
    "depth": 5000.0,
    "wob": 15000.0,
    "rpm": 100.0,
    "torque": 10000.0
  }
}

Response:
{
  "rig_id": "RIG_01",
  "is_anomaly": false,
  "anomaly_score": 0.15,
  "anomaly_type": null,
  "confidence": 0.92
}
```

### Get RUL History

```bash
GET /api/v1/predictions/rul-history/{rig_id}

Query Parameters:
- limit (optional): Number of records (default: 100)

Response:
{
  "rig_id": "RIG_01",
  "predictions": [
    {
      "rul_hours": 240.5,
      "confidence": 0.85,
      "prediction_date": "2024-01-01T12:00:00"
    }
  ]
}
```

## Maintenance API

### Get Maintenance Alerts

```bash
GET /api/v1/maintenance/alerts

Query Parameters:
- rig_id (optional): Filter by rig ID
- status (optional): Filter by status (pending, acknowledged, resolved)
- limit (optional): Number of records

Response:
{
  "success": true,
  "count": 5,
  "alerts": [
    {
      "id": 1,
      "rig_id": "RIG_01",
      "alert_type": "high_temperature",
      "severity": "high",
      "message": "Bit temperature exceeds threshold",
      "status": "pending",
      "created_at": "2024-01-01T12:00:00"
    }
  ]
}
```

### Create Maintenance Alert

```bash
POST /api/v1/maintenance/alerts

Headers:
Authorization: Bearer <access_token>
Content-Type: application/json

Request Body:
{
  "rig_id": "RIG_01",
  "alert_type": "high_temperature",
  "severity": "high",
  "message": "Bit temperature exceeds threshold",
  "component": "drill_bit"
}

Response:
{
  "success": true,
  "alert_id": 1,
  "message": "Maintenance alert created successfully"
}
```

## Control API

### Apply Change

```bash
POST /api/v1/control/apply-change

Headers:
Authorization: Bearer <access_token>
Content-Type: application/json

Request Body:
{
  "rig_id": "RIG_01",
  "change_type": "parameter",
  "component": "drilling",
  "parameter": "rpm",
  "value": 120.0,
  "auto_execute": false,
  "metadata": {
    "reason": "Optimize ROP",
    "expected_impact": "Increase ROP by 15%"
  }
}

Response:
{
  "success": true,
  "change_id": 1,
  "status": "pending",
  "message": "Change request created, pending approval"
}
```

### Get Change History

```bash
GET /api/v1/control/change-history

Query Parameters:
- rig_id (optional): Filter by rig ID
- status (optional): Filter by status
- limit (optional): Number of records

Response:
{
  "success": true,
  "count": 10,
  "changes": [...]
}
```

### Approve Change

```bash
POST /api/v1/control/change/{id}/approve

Headers:
Authorization: Bearer <access_token>

Response:
{
  "success": true,
  "message": "Change approved and applied successfully"
}
```

## WebSocket API

### Connect to Real-Time Stream

```javascript
const ws = new WebSocket('ws://localhost:8001/api/v1/sensor-data/ws/RIG_01');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Sensor data:', data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};
```

### Message Format

```json
{
  "message_type": "sensor_data",
  "data": {
    "rig_id": "RIG_01",
    "timestamp": "2024-01-01T12:00:00",
    "depth": 5000.0,
    "wob": 15000.0,
    ...
  }
}
```

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message here",
  "status_code": 400,
  "error_type": "ValidationError"
}
```

### Common Error Codes

- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

### Example Error Handling

```python
import requests

try:
    response = requests.get('http://localhost:8001/api/v1/sensor-data/realtime')
    response.raise_for_status()
    data = response.json()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 401:
        print("Authentication required")
    elif e.response.status_code == 403:
        print("Insufficient permissions")
    else:
        print(f"Error: {e}")
```

## Rate Limiting

Rate limits are applied to prevent abuse:

- **Default**: 100 requests per minute
- **Authentication**: 5 requests per minute
- **Predictions**: 20 requests per minute
- **Sensor Data**: 200 requests per minute

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1609459200
```

## Best Practices

### 1. Use Pagination

Always use pagination for large datasets:

```bash
GET /api/v1/sensor-data/historical?limit=100&offset=0
```

### 2. Handle Errors Gracefully

Always check response status and handle errors:

```python
response = requests.get(url)
if response.status_code == 200:
    data = response.json()
else:
    handle_error(response)
```

### 3. Use WebSocket for Real-Time Data

For real-time monitoring, use WebSocket instead of polling:

```javascript
// Good: WebSocket
const ws = new WebSocket('ws://...');

// Bad: Polling
setInterval(() => fetch('/api/v1/sensor-data/realtime'), 1000);
```

### 4. Cache Tokens

Store and reuse access tokens until they expire:

```python
# Store token
access_token = login_response['access_token']

# Reuse token
headers = {'Authorization': f'Bearer {access_token}'}
```

### 5. Validate Input

Always validate input before sending requests:

```python
if not rig_id or not validate_rig_id(rig_id):
    raise ValueError("Invalid rig_id")
```

### 6. Use Appropriate HTTP Methods

- `GET`: Retrieve data
- `POST`: Create resources
- `PUT`: Update resources
- `DELETE`: Delete resources

### 7. Include Request IDs

For debugging, include request IDs in headers:

```python
headers = {
    'X-Request-ID': str(uuid.uuid4()),
    'Authorization': f'Bearer {token}'
}
```

## SDK Examples

### Python SDK Example

```python
from i_drill_client import IDrillClient

client = IDrillClient(
    base_url='http://localhost:8001',
    username='engineer1',
    password='SecurePassword123!'
)

# Get real-time data
data = client.sensor_data.get_realtime(rig_id='RIG_01', limit=10)

# Get predictions
rul = client.predictions.predict_rul('RIG_01', sensor_data)

# Apply change
change = client.control.apply_change(
    rig_id='RIG_01',
    parameter='rpm',
    value=120.0
)
```

### JavaScript SDK Example

```javascript
import { IDrillClient } from '@idrill/sdk';

const client = new IDrillClient({
  baseURL: 'http://localhost:8001',
  username: 'engineer1',
  password: 'SecurePassword123!'
});

// Get real-time data
const data = await client.sensorData.getRealtime('RIG_01', { limit: 10 });

// Get predictions
const rul = await client.predictions.predictRUL('RIG_01', sensorData);

// Apply change
const change = await client.control.applyChange({
  rig_id: 'RIG_01',
  parameter: 'rpm',
  value: 120.0
});
```

## Support

For API support:
- Documentation: http://localhost:8001/docs
- Issues: [GitHub Issues](https://github.com/your-repo/issues)
- Email: support@example.com

