"""
Services package
"""
from .data_service import DataService
from .kafka_service import KafkaService, kafka_service

# Try to import PredictionService (optional - requires torch)
try:
    from .prediction_service import PredictionService
    PREDICTION_AVAILABLE = True
except ImportError as e:
    PREDICTION_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"PredictionService not available: {e}")
    PredictionService = None

__all__ = [
    'DataService',
    'KafkaService',
    'kafka_service'
]

if PREDICTION_AVAILABLE:
    __all__.append('PredictionService')

