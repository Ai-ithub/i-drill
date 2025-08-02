# mock_kafka_batch.py

from processing.dvr_controller import process_data

# Simulated micro-batch of incoming sensor data
data_batch = [
    {"sensor_id": 1, "value": 25},
    {"sensor_id": 2, "value": 110},   # Invalid value
    {"sensor_id": 3, "value": 70},
    {"sensor_id": 4, "value": None},  # Missing value
    {"sensor_id": 5, "value": 95}
]

# Process each data item in the batch
for data in data_batch:
    try:
        result = process_data(data)
        print("✅ Processed batch data:", result)
    except Exception as e:
        print("❌ Error:", e)
