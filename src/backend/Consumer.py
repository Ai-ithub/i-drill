import logging
from confluent_kafka import Consumer, KafkaException
from src.processing.dvr import insert_message, get_history_for_anomaly, flag_anomaly
import json
from datetime import datetime
import pandas as pd
from collections import defaultdict
import time
from pathlib import Path
import yaml
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

BASE_DIR = Path(__file__).resolve().parents[2]
config_path = BASE_DIR / "config.yml"

try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    logging.critical(f"Config file not found: {config_path}")
    sys.exit(1)
except yaml.YAMLError as e:
    logging.critical(f"Error parsing config file: {e}")
    sys.exit(1)

logging.info(f"Loaded config: {config}")

KAFKA_BROKER = config["kafka"]["broker"]
TOPIC_NAME = config["kafka"]["topic"]
CONSUMER_GROUP = config["kafka"]["group"]


def create_consumer():
    try:
        consumer_conf = {
            'bootstrap.servers': KAFKA_BROKER,
            'group.id': CONSUMER_GROUP,
            'auto.offset.reset': 'latest',
            'enable.auto.commit': True
        }
        consumer = Consumer(consumer_conf)
        consumer.subscribe([TOPIC_NAME])
        return consumer
    except KafkaException as e:
        logging.critical(f"Kafka consumer creation failed: {e}")
        sys.exit(1)


def process_message(msg):
    try:
        data = json.loads(msg.value().decode('utf-8'))
    except Exception as e:
        logging.warning(f"Error decoding message: {e}")
        return None

    try:
        rig_id = data['rig_id']
        timestamp = datetime.fromisoformat(data['timestamp'])
    except KeyError as e:
        logging.warning(f"Missing expected key: {e}")
        return None
    except ValueError as e:
        logging.warning(f"Invalid timestamp format: {e}")
        return None

    logging.info(f"Processing data from {rig_id} at {timestamp}")

    if data.get('maintenance_flag') == 1:
        logging.error(f"MAINTENANCE ALERT: {rig_id} has {data.get('failure_type', 'UNKNOWN')}")

    try:
        history_dict, numeric_cols = get_history_for_anomaly(50)
        data = flag_anomaly(data, history_dict, numeric_cols)
        insert_message(data)
    except Exception as e:
        logging.error(f"Error during anomaly processing or DB insert: {e}")
        return None

    return data


def aggregate_data(consumer, duration_seconds=60):
    logging.info(f"Starting consumer. Aggregating data in {duration_seconds}-second windows...")

    try:
        window_start = time.time()
        window_data = defaultdict(list)

        while True:
            try:
                msg = consumer.poll(timeout=1.0)
            except KafkaException as e:
                logging.error(f"Kafka poll error: {e}")
                continue

            if msg is None:
                continue
            if msg.error():
                logging.error(f"Kafka message error: {msg.error()}")
                continue

            data = process_message(msg)
            if not data:
                continue

            rig_id = data['rig_id']
            window_data[rig_id].append(data)

            if time.time() - window_start >= duration_seconds:
                analyze_window(window_data)
                window_data = defaultdict(list)
                window_start = time.time()

    except KeyboardInterrupt:
        logging.info("Stopping consumer...")
    except Exception as e:
        logging.critical(f"Unexpected error in consumer loop: {e}")
    finally:
        consumer.close()


def analyze_window(window_data):
    logging.info("WINDOW SUMMARY ANALYSIS START")
    for rig_id, records in window_data.items():
        if not records:
            continue
        try:
            df = pd.DataFrame(records)
            numeric_cols = [col for col in df.columns if col not in ['timestamp', 'rig_id', 'failure_type']]
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

            logging.info(f"{rig_id} - {len(records)} records")
            logging.info("Averages:\n%s", df[numeric_cols].mean().to_string())

            maintenance_count = df['maintenance_flag'].sum()
            if maintenance_count > 0:
                logging.warning(f"{maintenance_count} maintenance events detected for {rig_id}")
                logging.warning("\n%s", df[df['maintenance_flag'] == 1][['timestamp', 'failure_type']].to_string(index=False))
        except Exception as e:
            logging.error(f"Error analyzing window for {rig_id}: {e}")
    logging.info("WINDOW SUMMARY ANALYSIS END")


if __name__ == "__main__":
    consumer = create_consumer()
    aggregate_data(consumer, duration_seconds=60)
