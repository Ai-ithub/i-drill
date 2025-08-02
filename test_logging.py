from processing.dvr_controller import process_data

sample_data = {"sensor_id": 1, "value": 55}

processed = process_data(sample_data)

print("Processed data:", processed)
