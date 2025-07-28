import json
import time
import numpy as np
from datetime import datetime, timedelta
from confluent_kafka import Producer

# --- Kafka configuration ---
producer = Producer({
    'bootstrap.servers': 'localhost:9092',
    'client.id': 'rig-stream-generator'
})
topic = 'rig.sensor.stream'

# --- Simulation configuration ---
rig_id = "RIG_01"
start_time = datetime(2025, 1, 1, 0, 0, 0)
seconds_since_start = 0

def generate_one_record(seconds_since_start):
    timestamp = start_time + timedelta(seconds=seconds_since_start)

    record = {
        'Timestamp': timestamp.isoformat(),
        'Rig_ID': rig_id,
        'Depth': float(np.random.normal(1000 + seconds_since_start * 0.002, 0.001)),
        'WOB': float(np.random.normal(1500, 100)),
        'RPM': float(np.random.normal(80, 5)),
        'Torque': float(np.random.normal(400, 30)),
        'ROP': float(np.random.normal(12, 2)),
        'Mud_Flow_Rate': float(np.random.normal(1200, 100)),
        'Mud_Pressure': float(np.random.normal(3000, 200)),
        'Mud_Temperature': float(np.random.normal(60, 3)),
        'Mud_Density': float(np.random.normal(1200, 50)),
        'Mud_Viscosity': float(np.random.normal(35, 5)),
        'Mud_PH': float(np.random.normal(8.5, 0.2)),
        'Gamma_Ray': float(np.random.normal(85, 15)),
        'Resistivity': float(np.random.normal(20, 5)),
        'Pump_Status': int(np.random.choice([0, 1], p=[0.01, 0.99])),
        'Compressor_Status': int(np.random.choice([0, 1], p=[0.02, 0.98])),
        'Power_Consumption': float(np.random.normal(200, 20)),
        'Vibration_Level': float(np.random.normal(0.8, 0.3)),
        'Bit_Temperature': float(np.random.normal(90, 5)),
        'Motor_Temperature': float(np.random.normal(75, 4))
    }

    # حذف هدف‌ها (در اینجا تولید نشده‌اند)
    return record

# --- Infinite Stream ---
print(f"📡 Sending streaming data for {rig_id} to Kafka topic '{topic}' ... (Ctrl+C to stop)")

record_id = 0
try:
    while True:
        record = generate_one_record(seconds_since_start)
        producer.produce(topic, key=str(record_id), value=json.dumps(record))
        producer.poll(0)
        print(f"[{record_id}] Sent record at {record['Timestamp']}")
        record_id += 1
        seconds_since_start += 1
        time.sleep(1)

except KeyboardInterrupt:
    print("\n⛔️ Stopped by user.")
    producer.flush()
