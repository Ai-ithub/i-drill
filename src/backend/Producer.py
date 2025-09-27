import numpy as np
from datetime import datetime, timedelta
import random
import json
from time import sleep
from confluent_kafka import Producer
from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
config_path = BASE_DIR / "config.yml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

print(config)

KAFKA_BROKER = config["kafka"]["broker"]
TOPIC_NAME = config["kafka"]["topic"]

PRODUCE_FREQUENCY = 1

# Data generation configuration
num_devices = 10
duration_days = 6 * 30
freq_seconds = 1
start_time = datetime(2025, 1, 1, 0, 0, 0)

failure_types = [
    'Motor_Failure', 'Pump_Leak', 'Hydraulic_Failure',
    'Bit_Wear', 'High_Vibration', 'Sensor_Fault', 'Compressor_Failure'
]

def json_serializer(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


def delivery_report(err, msg):
    """Callback for Kafka delivery reports"""
    if err is not None:
        print(f"❌ Delivery failed for record {msg.key()}: {err}")
    else:
        print(f"✅ Record delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")


def create_producer():
    """Create and return a Confluent Kafka producer"""
    return Producer({'bootstrap.servers': KAFKA_BROKER})


def generate_sensor_data(device_id, timestamp):
    """Generate a single record of sensor data for a device at a specific timestamp"""
    data = {
        'timestamp': timestamp.isoformat(),
        'rig_id': f'RIG_{device_id:02d}',
        'depth': max(0, np.random.normal(5000, 1000)),
        'wob': max(0, np.random.normal(1500, 100)),
        'rpm': max(0, np.random.normal(80, 5)),
        'torque': max(0, np.random.normal(400, 30)),
        'rop': max(0, np.random.normal(12, 2)),
        'mud_flow_rate': max(0, np.random.normal(1200, 100)),
        'mud_pressure': max(0, np.random.normal(3000, 200)),
        'mud_temperature': np.random.normal(60, 3),
        'mud_density': max(0, np.random.normal(1200, 50)),
        'mud_viscosity': max(0, np.random.normal(35, 5)),
        'mud_ph': max(0, np.random.normal(8.5, 0.2)),
        'gamma_ray': max(0, np.random.normal(85, 15)),
        'resistivity': max(0, np.random.normal(20, 5)),
        'pump_status': np.random.choice([0, 1], p=[0.01, 0.99]),
        'compressor_status': np.random.choice([0, 1], p=[0.02, 0.98]),
        'power_consumption': max(0, np.random.normal(200, 20)),
        'vibration_level': max(0, np.random.normal(0.8, 0.3)),
        'bit_temperature': np.random.normal(90, 5),
        'motor_temperature': np.random.normal(75, 4),
        'maintenance_flag': 0,
        'failure_type': 'None'
    }

    # Add 5% chance of failure
    if random.random() < 0.05:
        data['maintenance_flag'] = 1
        data['failure_type'] = random.choice(failure_types)

    # Add 5% chance of missing data
    for field in data:
        if field not in ['timestamp', 'rig_id', 'maintenance_flag', 'failure_type'] and random.random() < 0.05:
            data[field] = None

    # Add 3% chance of noisy data
    for field in data:
        if isinstance(data[field], (int, float)) and data[field] is not None and random.random() < 0.03:
            data[field] = data[field] * (1 + random.uniform(-0.1, 0.1))

    return data


def produce_sensor_data(producer):
    """Continuously produce sensor data for all devices"""
    try:
        current_time = start_time
        print(f"🚀 Starting Kafka producer. Sending data to topic: {TOPIC_NAME}")

        while True:
            for device_id in range(1, num_devices + 1):
                data = generate_sensor_data(device_id, current_time)
                print(type(data))
                producer.produce(
                    TOPIC_NAME,
                    value=json.dumps(data, default=json_serializer).encode('utf-8'),
                    callback=delivery_report
                )
                print(f"📤 Sent data for {data['rig_id']} at {data['timestamp']}")

            producer.poll(0)  # Trigger delivery report callbacks
            current_time += timedelta(seconds=freq_seconds)
            sleep(PRODUCE_FREQUENCY)

    except KeyboardInterrupt:
        print("\n🛑 Stopping producer...")
    finally:
        producer.flush()



if __name__ == "__main__":
    producer = create_producer()
    produce_sensor_data(producer)
