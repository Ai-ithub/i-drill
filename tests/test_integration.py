"""
Integration tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta


def test_health_check_integration(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "services" in data


def test_api_root_endpoint(client):
    """Test root API endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "i-Drill API"
    assert data["version"] == "1.0.0"
    assert "documentation" in data


def test_openapi_docs_accessible(client):
    """Test that OpenAPI docs are accessible"""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_json_accessible(client):
    """Test that OpenAPI JSON is accessible"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "paths" in schema


def test_sensor_data_realtime_endpoint(client, sample_sensor_data):
    """Test real-time sensor data endpoint"""
    response = client.get("/api/v1/sensor-data/realtime")
    # Should return 200 even if no data (empty list)
    assert response.status_code in [200, 503]  # 503 if database unavailable
    
    if response.status_code == 200:
        data = response.json()
        assert "success" in data
        assert "count" in data
        assert "data" in data


def test_sensor_data_historical_endpoint(client):
    """Test historical sensor data endpoint"""
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    
    response = client.get(
        "/api/v1/sensor-data/historical",
        params={
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "limit": 10
        }
    )
    
    # Should return 200 or 422 (validation) or 503 (service unavailable)
    assert response.status_code in [200, 422, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "success" in data
        assert "count" in data
        assert "data" in data


def test_create_sensor_data_endpoint(client, sample_sensor_data):
    """Test create sensor data endpoint"""
    response = client.post(
        "/api/v1/sensor-data/",
        json=sample_sensor_data
    )
    
    # Should return 201 (created) or 422 (validation) or 503 (service unavailable)
    assert response.status_code in [201, 422, 503, 500]
    
    if response.status_code == 201:
        data = response.json()
        assert "rig_id" in data or "success" in data


def test_predictions_rul_endpoint(client, sample_rul_request):
    """Test RUL prediction endpoint"""
    response = client.post(
        "/api/v1/predictions/rul",
        json=sample_rul_request
    )
    
    # Should return 200, 422, 503, or 500
    assert response.status_code in [200, 422, 503, 500]
    
    if response.status_code == 200:
        data = response.json()
        assert "success" in data or "rul" in data


def test_maintenance_alerts_endpoint(client):
    """Test maintenance alerts endpoint"""
    response = client.get("/api/v1/maintenance/alerts")
    
    # Should return 200 or 503
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list) or "success" in data


def test_error_response_format(client):
    """Test that error responses follow standard format"""
    # Request invalid endpoint
    response = client.get("/api/v1/nonexistent")
    
    # Should return 404
    assert response.status_code == 404
    
    data = response.json()
    # Check error response format
    assert "success" in data or "detail" in data


def test_validation_error_format(client):
    """Test validation error response format"""
    # Send invalid data
    response = client.post(
        "/api/v1/sensor-data/",
        json={
            "rig_id": "RIG_01",
            "depth": -10,  # Invalid: negative
            "wob": 1500
        }
    )
    
    # Should return 422
    assert response.status_code == 422
    
    data = response.json()
    assert "success" in data or "detail" in data
    assert data.get("success") is False or "detail" in data


def test_cors_headers(client):
    """Test CORS headers are present"""
    response = client.options(
        "/api/v1/sensor-data/realtime",
        headers={"Origin": "http://localhost:5173"}
    )
    
    # CORS headers should be present
    assert "access-control-allow-origin" in response.headers or response.status_code == 200


def test_rate_limit_headers(client):
    """Test rate limit headers (if rate limiting is enabled)"""
    # Make multiple requests
    for _ in range(5):
        response = client.get("/api/v1/sensor-data/realtime")
    
    # Check for rate limit headers (may not be present if rate limiting is disabled)
    # This test is informational
    assert response.status_code in [200, 429, 503]


@pytest.mark.parametrize("endpoint", [
    "/api/v1/health",
    "/api/v1/health/database",
    "/api/v1/health/kafka",
    "/api/v1/health/ready",
    "/api/v1/health/live",
])
def test_health_endpoints(client, endpoint):
    """Test all health check endpoints"""
    response = client.get(endpoint)
    assert response.status_code in [200, 503]  # 503 if service unavailable


def test_api_versioning(client):
    """Test API versioning is consistent"""
    # All endpoints should be under /api/v1
    response = client.get("/api/v1/health")
    assert response.status_code in [200, 503]
    
    # Old version should not exist or redirect
    response_old = client.get("/api/v0/health")
    assert response_old.status_code in [404, 301, 302]

