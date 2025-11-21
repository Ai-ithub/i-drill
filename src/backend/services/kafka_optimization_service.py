"""
Kafka Optimization Service
Manages topic configuration, partitions, retention, and compression for high throughput
"""
import logging
from typing import Dict, Any, Optional, List
import os

try:
    from confluent_kafka.admin import AdminClient, NewTopic, ConfigResource
    from confluent_kafka import KafkaException
    KAFKA_ADMIN_AVAILABLE = True
except ImportError:
    AdminClient = NewTopic = ConfigResource = KafkaException = None
    KAFKA_ADMIN_AVAILABLE = False

from config_loader import config_loader
from services.kafka_service import kafka_service

logger = logging.getLogger(__name__)


class KafkaOptimizationService:
    """
    Service for optimizing Kafka topics and configurations.
    
    Features:
    - Topic creation with optimized partitions
    - Retention policy configuration
    - Compression settings
    - Topic configuration updates
    """
    
    def __init__(self):
        """Initialize KafkaOptimizationService."""
        self.available = KAFKA_ADMIN_AVAILABLE and kafka_service.available
        if not self.available:
            logger.warning("Kafka admin client not available")
            self.admin_client = None
            return
        
        try:
            self.kafka_config = config_loader.get_kafka_config()
            bootstrap_servers = self.kafka_config.get('bootstrap_servers', 'localhost:9092')
            
            admin_config = {
                'bootstrap.servers': bootstrap_servers,
                'client.id': 'kafka-optimization-service'
            }
            
            # Add authentication if configured
            kafka_username = os.getenv('KAFKA_USERNAME')
            kafka_password = os.getenv('KAFKA_PASSWORD')
            if kafka_username and kafka_password:
                admin_config.update({
                    'security.protocol': 'SASL_PLAINTEXT',
                    'sasl.mechanism': 'PLAIN',
                    'sasl.username': kafka_username,
                    'sasl.password': kafka_password,
                })
            
            self.admin_client = AdminClient(admin_config)
            logger.info("Kafka optimization service initialized")
        except Exception as e:
            logger.error(f"Error initializing Kafka optimization service: {e}")
            self.admin_client = None
            self.available = False
    
    def create_optimized_topic(
        self,
        topic_name: str,
        num_partitions: int = 4,
        replication_factor: int = 1,
        retention_ms: int = 2592000000,  # 30 days default
        compression_type: str = "snappy",
        segment_ms: int = 86400000,  # 1 day
        min_insync_replicas: int = 1
    ) -> Dict[str, Any]:
        """
        Create a Kafka topic with optimized settings for high throughput.
        
        Args:
            topic_name: Name of the topic to create
            num_partitions: Number of partitions (minimum 4 for sensor data)
            replication_factor: Replication factor
            retention_ms: Retention time in milliseconds (default: 30 days)
            compression_type: Compression type (none, gzip, snappy, lz4, zstd)
            segment_ms: Segment roll time in milliseconds
            min_insync_replicas: Minimum in-sync replicas
        
        Returns:
            Dictionary with success status and message
        """
        if not self.available or not self.admin_client:
            return {
                "success": False,
                "message": "Kafka admin client not available"
            }
        
        try:
            # Topic configuration
            topic_config = {
                'retention.ms': str(retention_ms),
                'compression.type': compression_type,
                'segment.ms': str(segment_ms),
                'min.insync.replicas': str(min_insync_replicas),
                # Performance optimizations
                'max.message.bytes': '10485760',  # 10MB
                'segment.bytes': '1073741824',  # 1GB
                'index.interval.bytes': '4096',
                # Producer optimizations
                'batch.size': '32768',  # 32KB
                'linger.ms': '10',
            }
            
            new_topic = NewTopic(
                topic_name,
                num_partitions=num_partitions,
                replication_factor=replication_factor,
                config=topic_config
            )
            
            # Create topic
            futures = self.admin_client.create_topics([new_topic])
            
            # Wait for result
            for topic, future in futures.items():
                try:
                    future.result()  # Wait for topic creation
                    logger.info(f"Topic {topic_name} created successfully with {num_partitions} partitions")
                    return {
                        "success": True,
                        "message": f"Topic {topic_name} created with {num_partitions} partitions",
                        "topic": topic_name,
                        "partitions": num_partitions,
                        "replication_factor": replication_factor,
                        "retention_ms": retention_ms,
                        "compression": compression_type
                    }
                except KafkaException as e:
                    if "already exists" in str(e).lower():
                        logger.warning(f"Topic {topic_name} already exists")
                        return {
                            "success": False,
                            "message": f"Topic {topic_name} already exists",
                            "error": str(e)
                        }
                    else:
                        logger.error(f"Error creating topic {topic_name}: {e}")
                        return {
                            "success": False,
                            "message": f"Failed to create topic: {e}",
                            "error": str(e)
                        }
        
        except Exception as e:
            logger.error(f"Error in create_optimized_topic: {e}")
            return {
                "success": False,
                "message": f"Error creating topic: {e}",
                "error": str(e)
            }
    
    def update_topic_config(
        self,
        topic_name: str,
        config_updates: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Update topic configuration.
        
        Args:
            topic_name: Name of the topic
            config_updates: Dictionary of config key-value pairs to update
        
        Returns:
            Dictionary with success status
        """
        if not self.available or not self.admin_client:
            return {
                "success": False,
                "message": "Kafka admin client not available"
            }
        
        try:
            config_resource = ConfigResource(
                ConfigResource.Type.TOPIC,
                topic_name,
                set_config=config_updates
            )
            
            futures = self.admin_client.alter_configs([config_resource])
            
            for resource, future in futures.items():
                try:
                    future.result()
                    logger.info(f"Topic {topic_name} configuration updated")
                    return {
                        "success": True,
                        "message": f"Topic {topic_name} configuration updated",
                        "updates": config_updates
                    }
                except KafkaException as e:
                    logger.error(f"Error updating topic config: {e}")
                    return {
                        "success": False,
                        "message": f"Failed to update topic config: {e}",
                        "error": str(e)
                    }
        
        except Exception as e:
            logger.error(f"Error in update_topic_config: {e}")
            return {
                "success": False,
                "message": f"Error updating topic config: {e}",
                "error": str(e)
            }
    
    def get_topic_config(self, topic_name: str) -> Optional[Dict[str, Any]]:
        """
        Get current topic configuration.
        
        Args:
            topic_name: Name of the topic
        
        Returns:
            Dictionary with topic configuration or None if error
        """
        if not self.available or not self.admin_client:
            return None
        
        try:
            config_resource = ConfigResource(ConfigResource.Type.TOPIC, topic_name)
            futures = self.admin_client.describe_configs([config_resource])
            
            for resource, future in futures.items():
                try:
                    config = future.result()
                    result = {}
                    for key, entry in config.items():
                        result[key] = {
                            "value": entry.value,
                            "is_default": entry.is_default,
                            "is_read_only": entry.is_read_only,
                            "is_sensitive": entry.is_sensitive
                        }
                    return result
                except KafkaException as e:
                    logger.error(f"Error getting topic config: {e}")
                    return None
        
        except Exception as e:
            logger.error(f"Error in get_topic_config: {e}")
            return None
    
    def ensure_sensor_topic_optimized(self) -> Dict[str, Any]:
        """
        Ensure sensor data topic is optimized with proper partitions and settings.
        
        Returns:
            Dictionary with status
        """
        try:
            topic_name = self.kafka_config.get('topics', {}).get('sensor_stream', 'rig.sensor.stream')
            
            # Check if topic exists and get current config
            current_config = self.get_topic_config(topic_name)
            
            if current_config is None:
                # Topic doesn't exist, create it
                logger.info(f"Creating optimized topic {topic_name}")
                return self.create_optimized_topic(
                    topic_name=topic_name,
                    num_partitions=4,
                    retention_ms=2592000000,  # 30 days
                    compression_type="snappy"
                )
            else:
                # Topic exists, check if optimization is needed
                current_partitions = self._get_topic_partitions(topic_name)
                needs_update = False
                updates = {}
                
                # Check compression
                compression = current_config.get('compression.type', {}).get('value', 'none')
                if compression == 'none':
                    updates['compression.type'] = 'snappy'
                    needs_update = True
                
                # Check retention (ensure it's set)
                retention = current_config.get('retention.ms', {}).get('value')
                if not retention or retention == '-1':
                    updates['retention.ms'] = '2592000000'  # 30 days
                    needs_update = True
                
                if needs_update:
                    logger.info(f"Updating topic {topic_name} configuration")
                    return self.update_topic_config(topic_name, updates)
                else:
                    return {
                        "success": True,
                        "message": f"Topic {topic_name} is already optimized",
                        "partitions": current_partitions,
                        "compression": compression
                    }
        
        except Exception as e:
            logger.error(f"Error ensuring topic optimization: {e}")
            return {
                "success": False,
                "message": f"Error optimizing topic: {e}",
                "error": str(e)
            }
    
    def _get_topic_partitions(self, topic_name: str) -> int:
        """Get number of partitions for a topic."""
        try:
            if not kafka_service.producer:
                return 0
            
            metadata = kafka_service.producer.list_topics(topic_name, timeout=5)
            if metadata and topic_name in metadata.topics:
                return len(metadata.topics[topic_name].partitions)
            return 0
        except Exception as e:
            logger.error(f"Error getting topic partitions: {e}")
            return 0


# Global instance
kafka_optimization_service = KafkaOptimizationService()

