from confluent_kafka import Consumer, KafkaException
from src.processing.dvr import insert_message, get_history_for_anomaly, flag_anomaly
import json
from datetime import datetime
import pandas as pd
from collections import defaultdict
import time

# Kafka configuration
KAFKA_BROKER = 'localhost:29092'
TOPIC_NAME = 'oil_rig_sensor_data'
CONSUMER_GROUP = 'oil_rig_analytics'

def create_consumer():
    """Create and return a Confluent Kafka consumer with JSON deserializer"""
    consumer_conf = {
        'bootstrap.servers': KAFKA_BROKER,
        'group.id': CONSUMER_GROUP,
        'auto.offset.reset': 'latest',   # Start from latest messages
        'enable.auto.commit': True
    }
    consumer = Consumer(consumer_conf)
    consumer.subscribe(["oil_rig_sensor_data"])
    return consumer


def process_message(msg):
    """Process a single Kafka message and check for alerts"""
    try:
        data = json.loads(msg.value().decode('utf-8'))
    except Exception as e:
        print(f"⚠️ Error decoding message: {e}")
        return None

    rig_id = data['rig_id']
    timestamp = datetime.fromisoformat(data['timestamp'])

    print(f"\n📥 Processing data from {rig_id} at {timestamp}")

    # Maintenance flag alerts
    if data['maintenance_flag'] == 1:
        print(f"🚨 MAINTENANCE ALERT: {rig_id} has {data['failure_type']}")

    history_dict, numeric_cols = get_history_for_anomaly(50)
    data = flag_anomaly(data, history_dict, numeric_cols)
    insert_message(data)


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
