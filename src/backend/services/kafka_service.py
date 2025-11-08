"""
Kafka Service for real-time data streaming
"""
import logging
from typing import Dict, Any, Optional, List

try:
    from confluent_kafka import Producer, Consumer, KafkaError  # type: ignore
    KAFKA_AVAILABLE = True
except ImportError:
    Producer = Consumer = KafkaError = None  # type: ignore
    KAFKA_AVAILABLE = False

from config_loader import config_loader
import json
import threading
from datetime import datetime

logger = logging.getLogger(__name__)


class KafkaService:
    """Service for Kafka streaming operations"""
    
    def __init__(self):
        if not KAFKA_AVAILABLE:
            logger.warning("confluent_kafka is not installed; Kafka streaming is disabled.")
            self.kafka_config = {'bootstrap_servers': 'localhost:9092'}
            self.producer = None
            self.consumer = None
            self.consumers = {}
            return
        
        try:
            self.kafka_config = config_loader.get_kafka_config()
        except Exception as e:
            logger.warning(f"Failed to load Kafka config: {e}. Using defaults...")
            self.kafka_config = {'bootstrap_servers': 'localhost:9092'}
        
        self.producer = None
        self.consumer = None
        self.consumers = {}  # Track multiple consumer instances
        self._initialize_producer()
    
    def _initialize_producer(self):
        """Initialize Kafka producer"""
        if not KAFKA_AVAILABLE:
            return
        try:
            self.producer = Producer({
                'bootstrap.servers': self.kafka_config.get('bootstrap_servers', 'localhost:9092'),
                'client.id': 'api-producer',
                'acks': 'all',
                'retries': 3,
            })
            logger.info("Kafka producer initialized")
        except Exception as e:
            logger.error(f"Error initializing Kafka producer: {e}")
            self.producer = None
    
    def produce_sensor_data(self, topic: str, data: Dict[str, Any]) -> bool:
        """
        Produce sensor data to Kafka topic
        
        Args:
            topic: Kafka topic name
            data: Sensor data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if not KAFKA_AVAILABLE:
            logger.debug("Kafka not available; skipping produce")
            return False
        
        if not self.producer:
            logger.error("Kafka producer not initialized")
            return False
        
        try:
            # Serialize data
            value = json.dumps(data, default=str)
            
            # Produce message
            self.producer.produce(
                topic,
                value=value.encode('utf-8'),
                callback=self._delivery_callback
            )
            
            # Trigger any pending callbacks
            self.producer.poll(0)
            
            logger.debug(f"Produced message to topic {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Error producing message to Kafka: {e}")
            return False
    
    def _delivery_callback(self, err, msg):
        """Callback for Kafka delivery reports"""
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")
    
    def create_consumer(self, consumer_id: str, topic: str, group_id: Optional[str] = None) -> bool:
        """
        Create a Kafka consumer
        
        Args:
            consumer_id: Unique identifier for this consumer
            topic: Topic to subscribe to
            group_id: Consumer group ID (optional)
            
        Returns:
            True if successful, False otherwise
        """
        if not KAFKA_AVAILABLE:
            logger.debug("Kafka not available; skipping consumer creation")
            return False

        try:
            group_id = group_id or f'api-consumer-{consumer_id}'
            
            consumer = Consumer({
                'bootstrap.servers': self.kafka_config.get('bootstrap_servers', 'localhost:9092'),
                'group.id': group_id,
                'auto.offset.reset': 'latest',
                'enable.auto.commit': True,
                'auto.commit.interval.ms': 1000,
            })
            
            consumer.subscribe([topic])
            self.consumers[consumer_id] = consumer
            
            logger.info(f"Consumer {consumer_id} created for topic {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating consumer: {e}")
            return False
    
    def consume_messages(self, consumer_id: str, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Consume a single message from Kafka
        
        Args:
            consumer_id: Consumer identifier
            timeout: Poll timeout in seconds
            
        Returns:
            Message data or None if timeout
        """
        if not KAFKA_AVAILABLE:
            logger.debug("Kafka not available; cannot consume messages")
            return None

        if consumer_id not in self.consumers:
            logger.error(f"Consumer {consumer_id} not found")
            return None
        
        consumer = self.consumers[consumer_id]
        
        try:
            msg = consumer.poll(timeout)
            
            if msg is None:
                return None
            
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    logger.error(f"Consumer error: {msg.error()}")
                return None
            
            # Deserialize message
            value = msg.value().decode('utf-8')
            data = json.loads(value)
            
            logger.debug(f"Consumed message from {consumer_id}")
            return data
            
        except Exception as e:
            logger.error(f"Error consuming message: {e}")
            return None
    
    def get_latest_messages(self, consumer_id: str, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get latest messages from Kafka
        
        Args:
            consumer_id: Consumer identifier
            count: Number of messages to retrieve
            
        Returns:
            List of messages
        """
        messages = []
        
        for _ in range(count):
            msg = self.consume_messages(consumer_id, timeout=0.5)
            if msg:
                messages.append(msg)
        
        return messages
    
    def close_consumer(self, consumer_id: str):
        """Close a consumer"""
        if consumer_id in self.consumers:
            self.consumers[consumer_id].close()
            del self.consumers[consumer_id]
            logger.info(f"Consumer {consumer_id} closed")
    
    def close(self):
        """Close all consumers and producer"""
        if not KAFKA_AVAILABLE:
            return
        for consumer_id in list(self.consumers.keys()):
            self.close_consumer(consumer_id)
        
        if self.producer:
            self.producer.flush()
            self.producer = None
            logger.info("Kafka producer closed")
    
    def check_connection(self) -> bool:
        """
        Check if Kafka connection is healthy
        
        Returns:
            True if Kafka is connected, False otherwise
        """
        if not KAFKA_AVAILABLE:
            return False
        
        try:
            if self.producer is None:
                return False
            
            # Try to list topics (lightweight check)
            metadata = self.producer.list_topics(timeout=2)
            return metadata is not None
            
        except Exception as e:
            logger.error(f"Kafka connection check failed: {e}")
            return False

# Global instance
kafka_service = KafkaService()

