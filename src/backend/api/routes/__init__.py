"""
API Routes package
"""
from .sensor_data import router as sensor_data_router
from .predictions import router as predictions_router
from .maintenance import router as maintenance_router
from .config import router as config_router
from .health import router as health_router
from .producer import router as producer_router

__all__ = [
    'sensor_data_router',
    'predictions_router',
    'maintenance_router',
    'config_router',
    'health_router',
    'producer_router'
]

