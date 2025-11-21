"""
Kafka Monitoring Service
Monitors Kafka performance metrics: lag, throughput, error rate
"""
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict, deque
import threading

try:
    from confluent_kafka import Consumer, TopicPartition
    from confluent_kafka.admin import AdminClient
    KAFKA_MONITORING_AVAILABLE = True
except ImportError:
    Consumer = TopicPartition = AdminClient = None
    KAFKA_MONITORING_AVAILABLE = False

from config_loader import config_loader
from services.kafka_service import kafka_service
import os

logger = logging.getLogger(__name__)


class KafkaMonitoringService:
    """
    Service for monitoring Kafka performance metrics.
    
    Monitors:
    - Consumer lag
    - Throughput (messages/sec, bytes/sec)
    - Error rates
    - Partition distribution
    """
    
    def __init__(self):
        """Initialize KafkaMonitoringService."""
        self.available = KAFKA_MONITORING_AVAILABLE and kafka_service.available
        if not self.available:
            logger.warning("Kafka monitoring not available")
            self.monitoring_thread = None
            self.running = False
            return
        
        try:
            self.kafka_config = config_loader.get_kafka_config()
            self.metrics = {
                "throughput": {
                    "messages_per_sec": 0.0,
                    "bytes_per_sec": 0.0,
                    "total_messages": 0,
                    "total_bytes": 0
                },
                "consumer_lag": {},
                "error_rate": {
                    "errors_per_minute": 0.0,
                    "total_errors": 0,
                    "last_error": None
                },
                "partitions": {},
                "last_updated": None
            }
            
            # Thread-safe metrics tracking
            self.message_count = 0
            self.byte_count = 0
            self.error_count = 0
            self.last_reset = time.time()
            self.lock = threading.Lock()
            
            # Window for calculating rates (60 seconds)
            self.rate_window = 60
            self.message_history = deque(maxlen=60)  # Last 60 seconds
            self.byte_history = deque(maxlen=60)
            self.error_history = deque(maxlen=60)
            
            logger.info("Kafka monitoring service initialized")
        except Exception as e:
            logger.error(f"Error initializing Kafka monitoring service: {e}")
            self.available = False
    
    def start_monitoring(self, interval: float = 5.0) -> None:
        """
        Start background monitoring thread.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if not self.available:
            return
        
        if self.running:
            logger.warning("Monitoring is already running")
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True,
            name="KafkaMonitoring"
        )
        self.monitoring_thread.start()
        logger.info("Kafka monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Kafka monitoring stopped")
    
    def _monitoring_loop(self, interval: float) -> None:
        """Background monitoring loop."""
        while self.running:
            try:
                # Update metrics
                self._update_throughput_metrics()
                self._update_consumer_lag()
                self._update_error_rate()
                self._update_partition_metrics()
                
                # Update timestamp
                with self.lock:
                    self.metrics["last_updated"] = datetime.now().isoformat()
                
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _update_throughput_metrics(self) -> None:
        """Update throughput metrics."""
        try:
            current_time = time.time()
            
            # Calculate rates from history
            if len(self.message_history) > 1:
                time_span = current_time - self.message_history[0][0]
                if time_span > 0:
                    messages = sum(count for _, count in self.message_history)
                    bytes_data = sum(byte_count for _, byte_count in self.byte_history)
                    
                    messages_per_sec = messages / time_span
                    bytes_per_sec = bytes_data / time_span
                else:
                    messages_per_sec = 0.0
                    bytes_per_sec = 0.0
            else:
                messages_per_sec = 0.0
                bytes_per_sec = 0.0
            
            with self.lock:
                self.metrics["throughput"]["messages_per_sec"] = messages_per_sec
                self.metrics["throughput"]["bytes_per_sec"] = bytes_per_sec
                self.metrics["throughput"]["total_messages"] = self.message_count
                self.metrics["throughput"]["total_bytes"] = self.byte_count
        except Exception as e:
            logger.error(f"Error updating throughput metrics: {e}")
    
    def _update_consumer_lag(self) -> None:
        """Update consumer lag metrics."""
        try:
            lag_metrics = {}
            
            # Get lag for all consumer groups
            for consumer_id, consumer in kafka_service.consumers.items():
                try:
                    # Get assigned partitions
                    assignment = consumer.assignment()
                    if not assignment:
                        continue
                    
                    # Get committed offsets
                    committed = consumer.committed(assignment, timeout=1.0)
                    
                    # Get high water marks
                    metadata = consumer.list_topics(timeout=1.0)
                    
                    total_lag = 0
                    partition_lags = {}
                    
                    for partition in assignment:
                        committed_offset = committed.get(partition, None)
                        if committed_offset is None:
                            continue
                        
                        # Get high water mark
                        topic_metadata = metadata.topics.get(partition.topic)
                        if topic_metadata:
                            partition_metadata = topic_metadata.partitions.get(partition.partition)
                            if partition_metadata:
                                high_water = partition_metadata.leader.high_water
                                lag = high_water - committed_offset.offset
                                total_lag += lag
                                partition_lags[f"{partition.topic}:{partition.partition}"] = lag
                    
                    lag_metrics[consumer_id] = {
                        "total_lag": total_lag,
                        "partition_lags": partition_lags
                    }
                except Exception as e:
                    logger.debug(f"Error getting lag for consumer {consumer_id}: {e}")
            
            with self.lock:
                self.metrics["consumer_lag"] = lag_metrics
        except Exception as e:
            logger.error(f"Error updating consumer lag: {e}")
    
    def _update_error_rate(self) -> None:
        """Update error rate metrics."""
        try:
            current_time = time.time()
            
            # Calculate errors per minute from history
            if len(self.error_history) > 1:
                time_span = current_time - self.error_history[0][0]
                if time_span > 0:
                    errors = len(self.error_history)
                    errors_per_minute = (errors / time_span) * 60
                else:
                    errors_per_minute = 0.0
            else:
                errors_per_minute = 0.0
            
            with self.lock:
                self.metrics["error_rate"]["errors_per_minute"] = errors_per_minute
                self.metrics["error_rate"]["total_errors"] = self.error_count
        except Exception as e:
            logger.error(f"Error updating error rate: {e}")
    
    def _update_partition_metrics(self) -> None:
        """Update partition distribution metrics."""
        try:
            if not kafka_service.producer:
                return
            
            topic_name = self.kafka_config.get('topics', {}).get('sensor_stream', 'rig.sensor.stream')
            metadata = kafka_service.producer.list_topics(topic_name, timeout=2)
            
            if metadata and topic_name in metadata.topics:
                topic_metadata = metadata.topics[topic_name]
                partition_info = {}
                
                for partition_id, partition_metadata in topic_metadata.partitions.items():
                    partition_info[partition_id] = {
                        "leader": partition_metadata.leader.id if partition_metadata.leader else None,
                        "replicas": [r.id for r in partition_metadata.replicas] if partition_metadata.replicas else []
                    }
                
                with self.lock:
                    self.metrics["partitions"][topic_name] = {
                        "total_partitions": len(partition_info),
                        "partition_details": partition_info
                    }
        except Exception as e:
            logger.debug(f"Error updating partition metrics: {e}")
    
    def record_message(self, message_size: int) -> None:
        """Record a processed message for throughput calculation."""
        if not self.available:
            return
        
        current_time = time.time()
        with self.lock:
            self.message_count += 1
            self.byte_count += message_size
            self.message_history.append((current_time, 1))
            self.byte_history.append((current_time, message_size))
    
    def record_error(self, error: str) -> None:
        """Record an error for error rate calculation."""
        if not self.available:
            return
        
        current_time = time.time()
        with self.lock:
            self.error_count += 1
            self.error_history.append((current_time, error))
            self.metrics["error_rate"]["last_error"] = {
                "message": error,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current monitoring metrics."""
        with self.lock:
            return self.metrics.copy()
    
    def get_consumer_lag(self, consumer_id: Optional[str] = None) -> Dict[str, Any]:
        """Get consumer lag for specific consumer or all consumers."""
        with self.lock:
            if consumer_id:
                return self.metrics["consumer_lag"].get(consumer_id, {})
            return self.metrics["consumer_lag"].copy()
    
    def get_throughput(self) -> Dict[str, Any]:
        """Get throughput metrics."""
        with self.lock:
            return self.metrics["throughput"].copy()
    
    def get_error_rate(self) -> Dict[str, Any]:
        """Get error rate metrics."""
        with self.lock:
            return self.metrics["error_rate"].copy()
    
    def reset_metrics(self) -> None:
        """Reset all metrics counters."""
        with self.lock:
            self.message_count = 0
            self.byte_count = 0
            self.error_count = 0
            self.message_history.clear()
            self.byte_history.clear()
            self.error_history.clear()
            self.last_reset = time.time()
        logger.info("Kafka monitoring metrics reset")


# Global instance
kafka_monitoring_service = KafkaMonitoringService()

