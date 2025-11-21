"""
Kafka Batch Consumer Service
Optimized consumer with batch processing, consumer groups, and offset management
"""
import logging
import threading
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import json

try:
    from confluent_kafka import Consumer, KafkaError, TopicPartition
    KAFKA_AVAILABLE = True
except ImportError:
    Consumer = KafkaError = TopicPartition = None
    KAFKA_AVAILABLE = False

from config_loader import config_loader
from services.kafka_monitoring_service import kafka_monitoring_service
import os

logger = logging.getLogger(__name__)


class KafkaBatchConsumer:
    """
    Optimized Kafka consumer with batch processing capabilities.
    
    Features:
    - Batch processing for improved performance
    - Consumer groups for load balancing
    - Manual offset management with commit strategies
    - Error handling and retry logic
    """
    
    def __init__(
        self,
        consumer_id: str,
        topics: List[str],
        group_id: str,
        batch_size: int = 100,
        batch_timeout: float = 1.0,
        auto_commit: bool = False,
        commit_interval: float = 5.0
    ):
        """
        Initialize batch consumer.
        
        Args:
            consumer_id: Unique identifier for this consumer
            topics: List of topics to subscribe to
            group_id: Consumer group ID
            batch_size: Maximum number of messages in a batch
            batch_timeout: Maximum time to wait for batch (seconds)
            auto_commit: Enable automatic offset commits
            commit_interval: Interval for manual commits (seconds)
        """
        if not KAFKA_AVAILABLE:
            raise RuntimeError("Kafka is not available")
        
        self.consumer_id = consumer_id
        self.topics = topics
        self.group_id = group_id
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.auto_commit = auto_commit
        self.commit_interval = commit_interval
        
        self.consumer = None
        self.running = False
        self.consumer_thread: Optional[threading.Thread] = None
        self.message_handler: Optional[Callable] = None
        self.last_commit = time.time()
        
        self._initialize_consumer()
    
    def _initialize_consumer(self) -> None:
        """Initialize Kafka consumer with optimized settings."""
        try:
            kafka_config = config_loader.get_kafka_config()
            bootstrap_servers = kafka_config.get('bootstrap_servers', 'localhost:9092')
            
            consumer_config = {
                'bootstrap.servers': bootstrap_servers,
                'group.id': self.group_id,
                'client.id': f'{self.consumer_id}-{int(time.time())}',
                'auto.offset.reset': 'latest',
                'enable.auto.commit': self.auto_commit,
                'auto.commit.interval.ms': int(self.commit_interval * 1000) if self.auto_commit else 0,
                # Batch processing optimizations
                'fetch.min.bytes': 1024,  # Wait for at least 1KB
                'fetch.wait.max.ms': int(self.batch_timeout * 1000),  # Max wait time
                'max.partition.fetch.bytes': 1048576,  # 1MB per partition
                # Session and heartbeat
                'session.timeout.ms': 30000,
                'heartbeat.interval.ms': 10000,
                # Max poll records (batch size)
                'max.poll.records': self.batch_size,
            }
            
            # Add authentication if configured
            kafka_username = os.getenv('KAFKA_USERNAME')
            kafka_password = os.getenv('KAFKA_PASSWORD')
            if kafka_username and kafka_password:
                consumer_config.update({
                    'security.protocol': 'SASL_PLAINTEXT',
                    'sasl.mechanism': 'PLAIN',
                    'sasl.username': kafka_username,
                    'sasl.password': kafka_password,
                })
            
            self.consumer = Consumer(consumer_config)
            self.consumer.subscribe(self.topics)
            
            logger.info(
                f"Batch consumer {self.consumer_id} initialized for topics {self.topics} "
                f"with batch_size={self.batch_size}, group_id={self.group_id}"
            )
        except Exception as e:
            logger.error(f"Error initializing batch consumer: {e}")
            raise
    
    def set_message_handler(self, handler: Callable[[List[Dict[str, Any]]], None]) -> None:
        """
        Set message handler for batch processing.
        
        Args:
            handler: Function that processes a list of messages
        """
        self.message_handler = handler
    
    def start(self) -> None:
        """Start consuming messages in background thread."""
        if self.running:
            logger.warning(f"Consumer {self.consumer_id} is already running")
            return
        
        if not self.message_handler:
            logger.warning(f"No message handler set for consumer {self.consumer_id}")
            return
        
        self.running = True
        self.consumer_thread = threading.Thread(
            target=self._consume_loop,
            daemon=True,
            name=f"KafkaBatchConsumer-{self.consumer_id}"
        )
        self.consumer_thread.start()
        logger.info(f"Batch consumer {self.consumer_id} started")
    
    def stop(self) -> None:
        """Stop consuming messages."""
        self.running = False
        if self.consumer_thread:
            self.consumer_thread.join(timeout=10)
        if self.consumer:
            self.consumer.close()
        logger.info(f"Batch consumer {self.consumer_id} stopped")
    
    def _consume_loop(self) -> None:
        """Main consumption loop with batch processing."""
        reconnect_delay = 1
        max_reconnect_delay = 60
        
        while self.running:
            try:
                # Poll for messages (returns up to max.poll.records)
                messages = self.consumer.poll(timeout=self.batch_timeout)
                
                if messages is None:
                    # Timeout, check if we should commit
                    if not self.auto_commit and (time.time() - self.last_commit) >= self.commit_interval:
                        self._commit_offsets()
                    continue
                
                # Handle errors
                if messages.error():
                    if messages.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    elif messages.error().code() == KafkaError._TRANSPORT:
                        logger.warning("Kafka transport error, will reconnect")
                        break
                    else:
                        logger.error(f"Consumer error: {messages.error()}")
                        kafka_monitoring_service.record_error(str(messages.error()))
                        continue
                
                # Collect batch of messages
                batch = []
                batch_start = time.time()
                
                # First message
                try:
                    value = messages.value().decode('utf-8')
                    data = json.loads(value)
                    batch.append({
                        "data": data,
                        "topic": messages.topic(),
                        "partition": messages.partition(),
                        "offset": messages.offset(),
                        "timestamp": datetime.now().isoformat()
                    })
                    kafka_monitoring_service.record_message(len(messages.value()))
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    kafka_monitoring_service.record_error(str(e))
                    continue
                
                # Try to get more messages in batch
                while len(batch) < self.batch_size and (time.time() - batch_start) < self.batch_timeout:
                    msg = self.consumer.poll(timeout=0.1)
                    if msg is None:
                        break
                    
                    if msg.error():
                        if msg.error().code() != KafkaError._PARTITION_EOF:
                            logger.error(f"Error in batch: {msg.error()}")
                            kafka_monitoring_service.record_error(str(msg.error()))
                        break
                    
                    try:
                        value = msg.value().decode('utf-8')
                        data = json.loads(value)
                        batch.append({
                            "data": data,
                            "topic": msg.topic(),
                            "partition": msg.partition(),
                            "offset": msg.offset(),
                            "timestamp": datetime.now().isoformat()
                        })
                        kafka_monitoring_service.record_message(len(msg.value()))
                    except Exception as e:
                        logger.error(f"Error processing message in batch: {e}")
                        kafka_monitoring_service.record_error(str(e))
                
                # Process batch
                if batch:
                    try:
                        self.message_handler(batch)
                        
                        # Commit offsets if manual commit
                        if not self.auto_commit:
                            if (time.time() - self.last_commit) >= self.commit_interval:
                                self._commit_offsets()
                    except Exception as e:
                        logger.error(f"Error in message handler: {e}")
                        kafka_monitoring_service.record_error(str(e))
                
                # Reset reconnect delay on success
                reconnect_delay = 1
                
            except Exception as e:
                logger.error(f"Error in consume loop: {e}")
                kafka_monitoring_service.record_error(str(e))
                
                if self.running:
                    logger.warning(f"Reconnecting in {reconnect_delay}s...")
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                    
                    # Try to reinitialize
                    try:
                        self._initialize_consumer()
                    except Exception as reconnect_error:
                        logger.error(f"Failed to reconnect: {reconnect_error}")
        
        logger.info(f"Consume loop for {self.consumer_id} stopped")
    
    def _commit_offsets(self) -> None:
        """Commit current offsets."""
        try:
            if self.consumer:
                self.consumer.commit(asynchronous=False)
                self.last_commit = time.time()
                logger.debug(f"Offsets committed for consumer {self.consumer_id}")
        except Exception as e:
            logger.error(f"Error committing offsets: {e}")
    
    def get_assignment(self) -> List[TopicPartition]:
        """Get currently assigned partitions."""
        if self.consumer:
            return self.consumer.assignment()
        return []
    
    def get_committed_offsets(self) -> Dict[str, Any]:
        """Get committed offsets for assigned partitions."""
        if not self.consumer:
            return {}
        
        try:
            assignment = self.consumer.assignment()
            if not assignment:
                return {}
            
            committed = self.consumer.committed(assignment, timeout=2.0)
            result = {}
            
            for partition, offset in committed.items():
                key = f"{partition.topic}:{partition.partition}"
                result[key] = {
                    "offset": offset.offset if offset else None,
                    "topic": partition.topic,
                    "partition": partition.partition
                }
            
            return result
        except Exception as e:
            logger.error(f"Error getting committed offsets: {e}")
            return {}

