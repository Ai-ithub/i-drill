"""
API Routes Package
Exports all route modules
"""
from . import (
    health,
    sensor_data,
    predictions,
    maintenance,
    producer,
    config,
    auth
)

__all__ = [
    'health',
    'sensor_data',
    'predictions',
    'maintenance',
    'producer',
    'config',
    'auth'
]
