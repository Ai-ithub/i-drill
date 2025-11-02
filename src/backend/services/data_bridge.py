"""
Data Bridge Service - Bridges Producer/Consumer with WebSocket and API
"""
import logging
import threading
from typing import Dict, Any, Optional
from confluent_kafka import Consumer, KafkaError
from config_loader import config_loader
from services.websocket_manager import websocket_manager
from services.data_service import DataService
import json
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class DataBridge:
    """
    Bridge service that connects Kafka Producer/Consumer with WebSocket and Database
    """
    
    def __init__(self):
        self.kafka_config = config_loader.get_kafka_config()
        self.data_service = DataService()
        self.running = False
        self.consumer_thread: Optional[threading.Thread] = None
        self.topic = self.kafka_config.get('topics', {}).get('sensor_stream', 'rig.sensor.stream')
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        
    def start(self, event_loop: Optional[asyncio.AbstractEventLoop] = None):
        """Start the data bridge consumer thread"""
        if self.running:
            logger.warning("Data bridge is already running")
            return
        
        # Store the event loop for async operations
        if event_loop is None:
            try:
                event_loop = asyncio.get_running_loop()
            except RuntimeError:
                # If no running loop, we'll create one in the thread
                pass
        
        self.event_loop = event_loop
        self.running = True
        self.consumer_thread = threading.Thread(
            target=self._consumer_loop,
            daemon=True,
            name="DataBridge-Consumer"
        )
        self.consumer_thread.start()
        logger.info("Data bridge started")
    
    def stop(self):
        """Stop the data bridge"""
        self.running = False
        if self.consumer_thread:
            self.consumer_thread.join(timeout=5)
        logger.info("Data bridge stopped")
    
    def _consumer_loop(self):
        """Main consumer loop that processes Kafka messages"""
        try:
            consumer = Consumer({
                'bootstrap.servers': self.kafka_config.get('bootstrap_servers', 'localhost:9092'),
                'group.id': 'data-bridge-group',
                'auto.offset.reset': 'latest',
                'enable.auto.commit': True,
                'auto.commit.interval.ms': 1000,
            })
            
            consumer.subscribe([self.topic])
            logger.info(f"Data bridge consumer subscribed to topic: {self.topic}")
            
            while self.running:
                try:
                    msg = consumer.poll(timeout=1.0)
                    
                    if msg is None:
                        continue
                    
                    if msg.error():
                        if msg.error().code() != KafkaError._PARTITION_EOF:
                            logger.error(f"Consumer error: {msg.error()}")
                        continue
                    
                    # Deserialize message
                    try:
                        value = msg.value().decode('utf-8')
                        data = json.loads(value)
                        
                        # Process the message
                        self._process_message(data)
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {e}")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        
                except Exception as e:
                    logger.error(f"Error in consumer loop: {e}")
                    if self.running:
                        import time
                        time.sleep(1)  # Wait before retrying
            
            consumer.close()
            logger.info("Data bridge consumer closed")
            
        except Exception as e:
            logger.error(f"Fatal error in data bridge consumer: {e}")
            self.running = False
    
    def _process_message(self, data: Dict[str, Any]):
        """
        Process a single Kafka message:
        1. Store in database
        2. Broadcast via WebSocket
        """
        try:
            # Store in database
            self.data_service.insert_sensor_data(data)
            
            # Broadcast to WebSocket clients
            rig_id = data.get('rig_id') or data.get('Rig_ID')
            if rig_id and self.event_loop:
                # Use asyncio to send to WebSocket clients via main event loop
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._broadcast_to_websocket(rig_id, data),
                        self.event_loop
                    )
                except Exception as e:
                    logger.error(f"Error scheduling WebSocket broadcast: {e}")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def _broadcast_to_websocket(self, rig_id: str, data: Dict[str, Any]):
        """Broadcast data to WebSocket clients"""
        try:
            message = {
                "message_type": "sensor_data",
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            await websocket_manager.send_to_rig(rig_id, message)
        except Exception as e:
            logger.error(f"Error broadcasting to WebSocket: {e}")
    
    def produce_sensor_data(self, data: Dict[str, Any]) -> bool:
        """Produce sensor data to Kafka (wrapper for kafka_service)"""
        from services.kafka_service import kafka_service
        topic = self.kafka_config.get('topics', {}).get('sensor_stream', 'rig.sensor.stream')
        return kafka_service.produce_sensor_data(topic, data)


# Global data bridge instance
data_bridge = DataBridge()

