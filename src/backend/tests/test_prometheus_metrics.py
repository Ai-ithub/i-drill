"""
Unit tests for Prometheus Metrics
"""
import pytest
from fastapi import Response
from utils.prometheus_metrics import (
    get_metrics,
    metrics_response,
    http_requests_total,
    http_request_duration_seconds,
    sensor_data_points_total,
    predictions_total,
    active_websocket_connections,
    database_connections_active,
    database_query_duration_seconds,
    cache_hits_total,
    cache_misses_total
)


class TestPrometheusMetrics:
    """Tests for Prometheus Metrics"""
    
    def test_get_metrics(self):
        """Test getting metrics"""
        metrics = get_metrics()
        assert metrics is not None
        assert isinstance(metrics, bytes)
        assert len(metrics) > 0
    
    def test_metrics_response(self):
        """Test creating metrics response"""
        response = metrics_response()
        assert isinstance(response, Response)
        assert response.media_type == "text/plain; version=0.0.4; charset=utf-8"
        assert len(response.body) > 0
    
    def test_http_requests_total_counter(self):
        """Test HTTP requests counter"""
        # Increment counter
        http_requests_total.labels(method="GET", endpoint="/api/v1/health", status=200).inc()
        http_requests_total.labels(method="POST", endpoint="/api/v1/auth/login", status=401).inc()
        
        # Should not raise error
        assert True
    
    def test_http_request_duration_histogram(self):
        """Test HTTP request duration histogram"""
        # Record duration
        http_request_duration_seconds.labels(method="GET", endpoint="/api/v1/health").observe(0.123)
        http_request_duration_seconds.labels(method="POST", endpoint="/api/v1/sensor-data").observe(0.456)
        
        # Should not raise error
        assert True
    
    def test_sensor_data_points_counter(self):
        """Test sensor data points counter"""
        sensor_data_points_total.labels(rig_id="RIG_01").inc()
        sensor_data_points_total.labels(rig_id="RIG_02").inc(5)
        
        # Should not raise error
        assert True
    
    def test_predictions_counter(self):
        """Test predictions counter"""
        predictions_total.labels(model_type="LSTM", status="success").inc()
        predictions_total.labels(model_type="Transformer", status="error").inc()
        
        # Should not raise error
        assert True
    
    def test_websocket_connections_gauge(self):
        """Test WebSocket connections gauge"""
        active_websocket_connections.set(5)
        active_websocket_connections.inc()
        active_websocket_connections.dec()
        
        # Should not raise error
        assert True
    
    def test_database_connections_gauge(self):
        """Test database connections gauge"""
        database_connections_active.set(10)
        database_connections_active.inc()
        database_connections_active.dec()
        
        # Should not raise error
        assert True
    
    def test_database_query_duration_histogram(self):
        """Test database query duration histogram"""
        database_query_duration_seconds.labels(operation="SELECT").observe(0.05)
        database_query_duration_seconds.labels(operation="INSERT").observe(0.1)
        
        # Should not raise error
        assert True
    
    def test_cache_hits_counter(self):
        """Test cache hits counter"""
        cache_hits_total.labels(cache_key="sensor_data").inc()
        cache_hits_total.labels(cache_key="predictions").inc(3)
        
        # Should not raise error
        assert True
    
    def test_cache_misses_counter(self):
        """Test cache misses counter"""
        cache_misses_total.labels(cache_key="sensor_data").inc()
        cache_misses_total.labels(cache_key="predictions").inc(2)
        
        # Should not raise error
        assert True
    
    def test_metrics_with_different_status_codes(self):
        """Test recording different HTTP status codes"""
        status_codes = [200, 201, 400, 401, 403, 404, 500]
        for status in status_codes:
            http_requests_total.labels(method="GET", endpoint="/api/v1/test", status=status).inc()
        
        # Should not raise error
        assert True
    
    def test_metrics_with_different_methods(self):
        """Test recording different HTTP methods"""
        methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
        for method in methods:
            http_requests_total.labels(method=method, endpoint="/api/v1/test", status=200).inc()
        
        # Should not raise error
        assert True

