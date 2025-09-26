from confluent_kafka import Consumer
import json
import logging
from config_loader import config_loader
from processing.dvr_controller import process_data

# --- Setup logging ---
logging_config = config_loader.get_logging_config()
logging.basicConfig(
    level=getattr(logging, logging_config.get('level', 'INFO')),
    format=logging_config.get('format', '%(asctime)s - %(levelname)s - %(message)s'),
    filename=logging_config.get('file', 'consumer.log')
)
logger = logging.getLogger(__name__)

# --- Load Kafka Configuration ---
kafka_config = config_loader.get_kafka_config()
consumer_config = kafka_config.get('consumer', {})

conf = {
    'bootstrap.servers': kafka_config.get('bootstrap_servers', 'localhost:9092'),
    'group.id': consumer_config.get('group_id', 'rig-consumer-group'),
    'auto.offset.reset': consumer_config.get('auto_offset_reset', 'earliest'),
    'enable.auto.commit': consumer_config.get('enable_auto_commit', True),
    'auto.commit.interval.ms': consumer_config.get('auto_commit_interval_ms', 1000)
}

topic = kafka_config.get('topics', {}).get('sensor_stream', 'rig.sensor.stream')

# --- Create Consumer ---
consumer = Consumer(conf)
consumer.subscribe([topic])

logger.info(f"Consumer initialized with topic: {topic}")

print(f"📥 Listening to Kafka topic '{topic}' for RIG sensor data... (Press Ctrl+C to stop)")

try:
    while True:
        msg = consumer.poll(1.0)  # 1 second timeout
        if msg is None:
            continue
        if msg.error():
            logger.error(f"Kafka error: {msg.error()}")
            print(f"⚠️ Error: {msg.error()}")
            continue

        try:
            # Decode JSON
            key = msg.key().decode('utf-8') if msg.key() else None
            value = json.loads(msg.value().decode('utf-8'))

            # --- Process data through DVR system ---
            processed_record = process_data(value)
            
            if processed_record is not None:
                # --- Print received data summary ---
                print("────────────────────────────────────────────")
                print(f"📦 Record ID: {key}")
                print(f"🕒 Timestamp: {value.get('Timestamp')}")
                print(f"🛢  RIG: {value.get('Rig_ID')} | Depth: {value.get('Depth', 0):.2f}")
                print(f"🔧 WOB: {value.get('WOB', 0):.2f} | RPM: {value.get('RPM', 0):.2f} | Torque: {value.get('Torque', 0):.2f}")
                print(f"💧 Mud Flow: {value.get('Mud_Flow_Rate', 0):.2f} | Pressure: {value.get('Mud_Pressure', 0):.2f} psi")
                print(f"🌡  Bit Temp: {value.get('Bit_Temperature', 0):.2f} °C | Motor Temp: {value.get('Motor_Temperature', 0):.2f} °C")
                print(f"⚡ Power: {value.get('Power_Consumption', 0):.2f} kW | Vibration: {value.get('Vibration_Level', 0):.2f}")
                print(f"✅ Status: {processed_record.get('status', 'Unknown')}")
                print("────────────────────────────────────────────")
                
                logger.info(f"Successfully processed record {key}")
            else:
                logger.warning(f"Record {key} failed DVR validation")
                print(f"❌ Record {key} failed validation")
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for message {key}: {e}")
            print(f"⚠️ JSON Error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing message {key}: {e}")
            print(f"⚠️ Processing Error: {e}")

except KeyboardInterrupt:
    logger.info("Consumer stopped by user")
    print("\n⛔️ Stopped by user.")
finally:
    consumer.close()
    logger.info("Consumer closed")
