from confluent_kafka import Consumer
import json

# --- Kafka Consumer Configuration ---
conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'rig-consumer-group',
    'auto.offset.reset': 'earliest'  # یا latest برای فقط پیام‌های جدید
}

topic = 'rig.sensor.stream'

# --- Create Consumer ---
consumer = Consumer(conf)
consumer.subscribe([topic])

print(f"📥 Listening to Kafka topic '{topic}' for RIG sensor data... (Press Ctrl+C to stop)")

try:
    while True:
        msg = consumer.poll(1.0)  # 1 second timeout
        if msg is None:
            continue
        if msg.error():
            print(f"⚠️ Error: {msg.error()}")
            continue

        # Decode JSON
        key = msg.key().decode('utf-8') if msg.key() else None
        value = json.loads(msg.value().decode('utf-8'))

        # --- Print received data summary ---
        print("────────────────────────────────────────────")
        print(f"📦 Record ID: {key}")
        print(f"🕒 Timestamp: {value.get('Timestamp')}")
        print(f"🛢  RIG: {value.get('Rig_ID')} | Depth: {value.get('Depth'):.2f}")
        print(f"🔧 WOB: {value.get('WOB'):.2f} | RPM: {value.get('RPM'):.2f} | Torque: {value.get('Torque'):.2f}")
        print(f"💧 Mud Flow: {value.get('Mud_Flow_Rate'):.2f} | Pressure: {value.get('Mud_Pressure'):.2f} psi")
        print(f"🌡  Bit Temp: {value.get('Bit_Temperature'):.2f} °C | Motor Temp: {value.get('Motor_Temperature'):.2f} °C")
        print(f"⚡ Power: {value.get('Power_Consumption'):.2f} kW | Vibration: {value.get('Vibration_Level'):.2f}")
        print("────────────────────────────────────────────")

except KeyboardInterrupt:
    print("\n⛔️ Stopped by user.")
finally:
    consumer.close()
