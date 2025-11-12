"""
Prometheus metrics for i-Drill API
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import time

# Request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

# Business metrics
sensor_data_points_total = Counter(
    'sensor_data_points_total',
    'Total sensor data points received',
    ['rig_id']
)

predictions_total = Counter(
    'predictions_total',
    'Total predictions made',
    ['model_type', 'status']
)

active_websocket_connections = Gauge(
    'active_websocket_connections',
    'Number of active WebSocket connections'
)

# Database metrics
database_connections_active = Gauge(
    'database_connections_active',
    'Active database connections'
)

database_query_duration_seconds = Histogram(
    'database_query_duration_seconds',
    'Database query duration in seconds',
    ['operation']
)

# Cache metrics
cache_hits_total = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_key']
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_key']
)


def get_metrics():
    """Get Prometheus metrics"""
    return generate_latest()


def metrics_response() -> Response:
    """Create FastAPI response with metrics"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

