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
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class KafkaService:
    """
    Service for Kafka streaming operations.
    
    Handles Kafka producer and consumer operations for real-time data streaming.
    Provides methods for producing sensor data to Kafka topics and consuming messages
    with automatic reconnection and error handling.
    
    Attributes:
        available: Boolean indicating if Kafka is available and initialized
        kafka_config: Dictionary containing Kafka configuration
        producer: Kafka Producer instance
        consumer: Kafka Consumer instance (deprecated, use consumers dict)
        consumers: Dictionary of consumer instances keyed by consumer_id
    """
    
    def __init__(self):
        """
        Initialize KafkaService.
        
        Attempts to load Kafka configuration and initialize the producer.
        If Kafka is not available or initialization fails, the service will
        operate in a degraded mode.
        """
        self.available = KAFKA_AVAILABLE
        if not KAFKA_AVAILABLE:
            logger.warning("confluent_kafka is not installed; Kafka streaming is disabled.")
            self.kafka_config = {'bootstrap_servers': 'localhost:9092'}
            self.producer = None
            self.consumer = None
            self.consumers = {}
            self.available = False
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
    
    def _initialize_producer(self, retry_count: int = 0, max_retries: int = 3) -> None:
        """
        Initialize Kafka producer with retry logic.
        
        Attempts to create a Kafka producer with exponential backoff retry.
        Configures producer with appropriate settings for reliability and performance.
        
        Args:
            retry_count: Current retry attempt number
            max_retries: Maximum number of retry attempts
        """
        if not KAFKA_AVAILABLE:
            return
        
        try:
            # Get Kafka authentication if configured
            producer_config = {
                'bootstrap.servers': self.kafka_config.get('bootstrap_servers', 'localhost:9092'),
                'client.id': 'api-producer',
                'acks': 'all',
                'retries': 3,
                'retry.backoff.ms': 100,
                'request.timeout.ms': 30000,
                'delivery.timeout.ms': 120000,
            }
            
            # Add authentication if configured
            kafka_username = os.getenv('KAFKA_USERNAME')
            kafka_password = os.getenv('KAFKA_PASSWORD')
            if kafka_username and kafka_password:
                producer_config.update({
                    'security.protocol': 'SASL_PLAINTEXT',
                    'sasl.mechanism': 'PLAIN',
                    'sasl.username': kafka_username,
                    'sasl.password': kafka_password,
                })
            
            self.producer = Producer(producer_config)
            logger.info("Kafka producer initialized")
            self.available = True
        except Exception as e:
            logger.error(f"Error initializing Kafka producer (attempt {retry_count + 1}/{max_retries}): {e}")
            if retry_count < max_retries:
                import time
                time.sleep(2 ** retry_count)  # Exponential backoff
                return self._initialize_producer(retry_count + 1, max_retries)
            else:
                self.producer = None
                self.available = False
    
    def produce_sensor_data(self, topic: str, data: Dict[str, Any], retry_count: int = 0, max_retries: int = 3) -> bool:
        """
        Produce sensor data to Kafka topic with retry logic
        
        Args:
            topic: Kafka topic name
            data: Sensor data dictionary
            retry_count: Current retry attempt
            max_retries: Maximum number of retries
            
        Returns:
            True if successful, False otherwise
        """
        if not KAFKA_AVAILABLE:
            logger.debug("Kafka not available; skipping produce")
            return False
        
        if not self.producer:
            # Try to reinitialize producer
            logger.warning("Kafka producer not initialized, attempting to reinitialize...")
            self._initialize_producer()
            if not self.producer:
                logger.error("Failed to reinitialize Kafka producer")
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
            logger.error(f"Error producing message to Kafka (attempt {retry_count + 1}/{max_retries}): {e}")
            
            # Retry with exponential backoff
            if retry_count < max_retries:
                import time
                time.sleep(2 ** retry_count)
                # Try to reinitialize producer if connection lost
                if "Connection" in str(e) or "Broker" in str(e):
                    self._initialize_producer()
                return self.produce_sensor_data(topic, data, retry_count + 1, max_retries)
            
            return False
    
    def _delivery_callback(self, err: Optional[Exception], msg: Any) -> None:
        """
        Callback function for Kafka message delivery reports.
        
        Logs delivery status for each message produced to Kafka.
        Called asynchronously by the Kafka producer library.
        
        Args:
            err: Error object if delivery failed, None if successful
            msg: Message object containing delivery metadata
        """
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
    
    def close_consumer(self, consumer_id: str) -> None:
        """
        Close a specific Kafka consumer.
        
        Args:
            consumer_id: Unique identifier of the consumer to close
        """
        if consumer_id in self.consumers:
            self.consumers[consumer_id].close()
            del self.consumers[consumer_id]
            logger.info(f"Consumer {consumer_id} closed")
    
    def close(self) -> None:
        """
        Close all consumers and the producer.
        
        Flushes any pending messages before closing the producer.
        Closes all active consumer instances.
        """
        if not KAFKA_AVAILABLE:
            return
        for consumer_id in list(self.consumers.keys()):
            self.close_consumer(consumer_id)
        
        if self.producer:
            self.producer.flush()
            self.producer = None
            logger.info("Kafka producer closed")
        self.available = False

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

    def is_available(self) -> bool:
        """
        Check if Kafka service is available and ready.
        
        Returns:
            True if Kafka is available and producer is initialized, False otherwise
        """
        return bool(self.available and self.producer is not None)

# Global instance
kafka_service = KafkaService()

