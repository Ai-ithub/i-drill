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
from services.integration_service import integration_service
import json
import asyncio
from datetime import datetime
from services.kafka_service import kafka_service
import os

logger = logging.getLogger(__name__)


class DataBridge:
    """
    Bridge service that connects Kafka Producer/Consumer with WebSocket and Database.
    
    This service acts as a bridge between Kafka message streams and the application's
    data storage and real-time communication layers. It:
    - Consumes messages from Kafka topics
    - Stores data in the database via DataService
    - Broadcasts data to WebSocket clients for real-time updates
    
    Attributes:
        kafka_config: Kafka configuration dictionary
        data_service: DataService instance for database operations
        running: Boolean flag indicating if the bridge is active
        consumer_thread: Thread running the Kafka consumer loop
        topic: Kafka topic name to consume from
        event_loop: AsyncIO event loop for WebSocket operations
    """
    
    def __init__(self):
        """
        Initialize DataBridge service.
        
        Sets up the bridge with Kafka configuration and data service.
        The bridge must be started explicitly using the start() method.
        """
        self.kafka_config = config_loader.get_kafka_config()
        self.data_service = DataService()
        self.running = False
        self.consumer_thread: Optional[threading.Thread] = None
        self.topic = self.kafka_config.get('topics', {}).get('sensor_stream', 'rig.sensor.stream')
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        # Integration settings
        self.enable_dvr_integration = os.getenv("ENABLE_DVR_IN_BRIDGE", "true").lower() == "true"
        self.enable_rl_integration = os.getenv("ENABLE_RL_IN_BRIDGE", "false").lower() == "true"
        
    def start(self, event_loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """
        Start the data bridge consumer thread.
        
        Initializes and starts a background thread that consumes messages from Kafka
        and processes them through the database and WebSocket layers.
        
        Args:
            event_loop: Optional AsyncIO event loop for WebSocket operations.
                       If not provided, attempts to get the running loop.
        """
        if self.running:
            logger.warning("Data bridge is already running")
            return

        if kafka_service.producer is None:
            logger.warning("Kafka producer unavailable; data bridge will not start.")
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
    
    def stop(self) -> None:
        """
        Stop the data bridge service.
        
        Signals the consumer thread to stop and waits for it to finish.
        Has a timeout of 5 seconds for graceful shutdown.
        """
        self.running = False
        if self.consumer_thread:
            self.consumer_thread.join(timeout=5)
        logger.info("Data bridge stopped")
    
    def _consumer_loop(self) -> None:
        """
        Main consumer loop that processes Kafka messages with reconnection logic.
        
        Continuously polls Kafka for messages, processes them, and handles reconnection
        with exponential backoff in case of connection failures. This method runs
        in a separate thread.
        """
        import time
        reconnect_delay = 1
        max_reconnect_delay = 60
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self.running:
            consumer = None
            try:
                # Build consumer config
                consumer_config = {
                    'bootstrap.servers': self.kafka_config.get('bootstrap_servers', 'localhost:9092'),
                    'group.id': 'data-bridge-group',
                    'auto.offset.reset': 'latest',
                    'enable.auto.commit': True,
                    'auto.commit.interval.ms': 1000,
                    'session.timeout.ms': 30000,
                    'heartbeat.interval.ms': 10000,
                }
                
                # Add authentication if configured
                import os
                kafka_username = os.getenv('KAFKA_USERNAME')
                kafka_password = os.getenv('KAFKA_PASSWORD')
                if kafka_username and kafka_password:
                    consumer_config.update({
                        'security.protocol': 'SASL_PLAINTEXT',
                        'sasl.mechanism': 'PLAIN',
                        'sasl.username': kafka_username,
                        'sasl.password': kafka_password,
                    })
                
                consumer = Consumer(consumer_config)
                consumer.subscribe([self.topic])
                logger.info(f"Data bridge consumer subscribed to topic: {self.topic}")
                
                # Reset reconnect delay on successful connection
                reconnect_delay = 1
                consecutive_errors = 0
                
                # Main message processing loop
                while self.running:
                    try:
                        msg = consumer.poll(timeout=1.0)
                        
                        if msg is None:
                            continue
                        
                        if msg.error():
                            if msg.error().code() == KafkaError._PARTITION_EOF:
                                continue
                            elif msg.error().code() == KafkaError._TRANSPORT:
                                logger.warning("Kafka transport error, will reconnect")
                                break
                            else:
                                logger.error(f"Consumer error: {msg.error()}")
                                consecutive_errors += 1
                                if consecutive_errors >= max_consecutive_errors:
                                    logger.error("Too many consecutive errors, reconnecting...")
                                    break
                                continue
                        
                        # Reset error counter on successful message
                        consecutive_errors = 0
                        
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
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            break
                        time.sleep(1)
                
                # Close consumer before reconnecting
                if consumer:
                    try:
                        consumer.close()
                    except:
                        pass
                
            except Exception as e:
                logger.error(f"Fatal error in data bridge consumer: {e}")
                if consumer:
                    try:
                        consumer.close()
                    except:
                        pass
            
            # Reconnect logic
            if self.running:
                logger.warning(f"Data bridge consumer disconnected. Reconnecting in {reconnect_delay}s...")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)  # Exponential backoff
        
        logger.info("Data bridge consumer stopped")
    
    def _process_message(self, data: Dict[str, Any]) -> None:
        """
        Process a single Kafka message through integrated pipeline.
        
        Pipeline:
        1. Process through DVR (if enabled) - validates and reconciles data
        2. Optionally feed validated data to RL (if enabled)
        3. Store in database
        4. Broadcast to WebSocket clients for real-time updates
        
        Args:
            data: Dictionary containing sensor data from Kafka message
        """
        try:
            # Step 1: Process through DVR and optionally RL (if enabled)
            processed_data = data
            if self.enable_dvr_integration or self.enable_rl_integration:
                try:
                    integration_result = integration_service.process_sensor_data_for_rl(
                        sensor_record=data,
                        apply_to_rl=self.enable_rl_integration
                    )
                    
                    if integration_result.get("success"):
                        # Use validated/reconciled data if available
                        dvr_result = integration_result.get("dvr_result", {})
                        if dvr_result.get("processed_record"):
                            processed_data = dvr_result["processed_record"]
                            logger.debug(f"Data processed through DVR for rig: {data.get('rig_id') or data.get('Rig_ID')}")
                    else:
                        logger.warning(
                            f"DVR processing failed for message: {integration_result.get('message')}. "
                            "Using raw data."
                        )
                except Exception as integration_exc:
                    logger.warning(f"Integration processing failed: {integration_exc}. Using raw data.")
            
            # Step 2: Store in database
            try:
                self.data_service.insert_sensor_data(processed_data)
            except Exception as db_exc:
                logger.error(f"Database insertion failed: {db_exc}")
            
            # Step 3: Broadcast to WebSocket clients
            rig_id = processed_data.get('rig_id') or processed_data.get('Rig_ID')
            if rig_id and self.event_loop:
                # Use asyncio to send to WebSocket clients via main event loop
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._broadcast_to_websocket(rig_id, processed_data),
                        self.event_loop
                    )
                except Exception as e:
                    logger.error(f"Error scheduling WebSocket broadcast: {e}")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def _broadcast_to_websocket(self, rig_id: str, data: Dict[str, Any]) -> None:
        """
        Broadcast data to WebSocket clients for a specific rig.
        
        Sends sensor data to all connected WebSocket clients subscribed to the
        specified rig's data stream.
        
        Args:
            rig_id: Rig identifier to broadcast to
            data: Sensor data dictionary to broadcast
        """
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
        """
        Produce sensor data to Kafka topic.
        
        Wrapper method that uses kafka_service to send sensor data to the
        configured Kafka topic.
        
        Args:
            data: Sensor data dictionary to produce
            
        Returns:
            True if message was successfully produced, False otherwise
        """
        topic = self.kafka_config.get('topics', {}).get('sensor_stream', 'rig.sensor.stream')
        return kafka_service.produce_sensor_data(topic, data)


# Global data bridge instance
data_bridge = DataBridge()

