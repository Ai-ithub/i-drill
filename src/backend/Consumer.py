from confluent_kafka import Consumer, KafkaException
from src.processing.dvr import insert_message, get_history_for_anomaly, flag_anomaly
import json
import os
import logging
from config_loader import config_loader
from processing.dvr_controller import process_data
from datetime import datetime
import pandas as pd
from collections import defaultdict
import time

# --- Setup logging ---
logging_config = config_loader.get_logging_config()
logging.basicConfig(
    level=getattr(logging, logging_config.get('level', 'INFO')),
    format=logging_config.get('format', '%(asctime)s - %(levelname)s - %(message)s'),
    filename=logging_config.get('file', 'consumer.log')
)
logger = logging.getLogger(__name__)

# --- Load Kafka Configuration ---
# Use environment variable if available, otherwise use config_loader
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS')
if not KAFKA_BOOTSTRAP_SERVERS:
    try:
        kafka_config = config_loader.get_kafka_config()
        KAFKA_BOOTSTRAP_SERVERS = kafka_config.get('bootstrap_servers', 'localhost:9092')
        consumer_config = kafka_config.get('consumer', {})
        topic = kafka_config.get('topics', {}).get('sensor_stream', 'rig.sensor.stream')
    except Exception as e:
        logger.warning(f"Failed to load Kafka config: {e}. Using defaults...")
        KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
        consumer_config = {}
        topic = 'rig.sensor.stream'
else:
    try:
        kafka_config = config_loader.get_kafka_config()
        consumer_config = kafka_config.get('consumer', {})
        topic = kafka_config.get('topics', {}).get('sensor_stream', 'rig.sensor.stream')
    except Exception as e:
        logger.warning(f"Failed to load Kafka config: {e}. Using defaults...")
        consumer_config = {}
        topic = 'rig.sensor.stream'

logger.info(f"Kafka bootstrap servers: {KAFKA_BOOTSTRAP_SERVERS}")
logger.info(f"Consumer initialized with topic: {topic}")

# Kafka configuration
TOPIC_NAME = topic
CONSUMER_GROUP = consumer_config.get('group_id', 'rig-consumer-group')

def create_consumer():
    """Create and return a Confluent Kafka consumer with JSON deserializer"""
    consumer_conf = {
        'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
        'group.id': CONSUMER_GROUP,
        'auto.offset.reset': consumer_config.get('auto_offset_reset', 'earliest'),
        'enable.auto.commit': consumer_config.get('enable_auto_commit', True),
        'auto.commit.interval.ms': consumer_config.get('auto_commit_interval_ms', 1000)
    }
    consumer = Consumer(consumer_conf)
    consumer.subscribe([TOPIC_NAME])
    return consumer


logger.info(f"Consumer initialized with topic: {topic}")

logger.info(f"Listening to Kafka topic '{topic}' for RIG sensor data... (Press Ctrl+C to stop)")

# Initialize consumer
consumer = create_consumer()

try:
    while True:
        msg = consumer.poll(1.0)  # 1 second timeout
        if msg is None:
            continue
        if msg.error():
            logger.error(f"Kafka error: {msg.error()}")
            logger.error(f"Kafka error: {msg.error()}")
            continue

        try:
            # Decode JSON
            key = msg.key().decode('utf-8') if msg.key() else None
            value = json.loads(msg.value().decode('utf-8'))

            # --- Process data through DVR system ---
            processed_record = process_data(value)
            
            if processed_record is not None:
                # Log received data summary
                logger.info(
                    f"Record processed - ID: {key}, RIG: {value.get('Rig_ID')}, "
                    f"Depth: {value.get('Depth', 0):.2f}, Status: {processed_record.get('status', 'Unknown')}"
                )
                
                logger.info(f"Successfully processed record {key}")
            else:
                logger.warning(f"Record {key} failed DVR validation")
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for message {key}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing message {key}: {e}", exc_info=True)

except KeyboardInterrupt:
    logger.info("Consumer stopped by user")
finally:
    consumer.close()
    logger.info("Consumer closed")
def process_message(msg):
    """Process a single Kafka message and check for alerts"""
    try:
        data = json.loads(msg.value().decode('utf-8'))
    except Exception as e:
        logger.error(f"Error decoding message: {e}", exc_info=True)
        return None

    rig_id = data['rig_id']
    timestamp = datetime.fromisoformat(data['timestamp'])

    print(f"\n📥 Processing data from {rig_id} at {timestamp}")

    # Maintenance flag alerts
    if data['maintenance_flag'] == 1:
        print(f"🚨 MAINTENANCE ALERT: {rig_id} has {data['failure_type']}")

    history_dict, numeric_cols = get_history_for_anomaly(50)
    data = flag_anomaly(data, history_dict, numeric_cols)
    # insert_message(data)


    return data


def aggregate_data(consumer, duration_seconds=60):
    """Aggregate data over a time window and display summary"""
    print(f"\n🚀 Starting consumer. Aggregating data in {duration_seconds}-second windows...")

    try:
        window_start = time.time()
        window_data = defaultdict(list)

        while True:
            msg = consumer.poll(timeout=1.0)  # Wait for messages
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())

            data = process_message(msg)
            if not data:
                continue

            rig_id = data['rig_id']
            window_data[rig_id].append(data)

            # Check if the time window has elapsed
            if time.time() - window_start >= duration_seconds:
                analyze_window(window_data)
                window_data = defaultdict(list)  # Reset
                window_start = time.time()

    except KeyboardInterrupt:
        print("\n🛑 Stopping consumer...")
    finally:
        consumer.close()


def analyze_window(window_data):
    """Analyze aggregated data for a time window"""
    print("\n" + "=" * 50)
    print("📊 WINDOW SUMMARY ANALYSIS")
    print("=" * 50)

    for rig_id, records in window_data.items():
        if not records:
            continue

        df = pd.DataFrame(records)
        numeric_cols = [col for col in df.columns if col not in ['timestamp', 'rig_id', 'failure_type']]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        print(f"\n{rig_id} - {len(records)} records")
        print("Average values:")
        print(df[numeric_cols].mean().to_string())

        # Maintenance events in this window
        maintenance_count = df['maintenance_flag'].sum()
        if maintenance_count > 0:
            print(f"\n🔴 {maintenance_count} maintenance events detected:")
            print(df[df['maintenance_flag'] == 1][['timestamp', 'failure_type']].to_string(index=False))


if __name__ == "__main__":
    consumer = create_consumer()
    aggregate_data(consumer, duration_seconds=60)
