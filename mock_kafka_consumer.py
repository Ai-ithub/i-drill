from processing.dvr_controller import process_data

def main():
    data_stream = [
        {"sensor_id": 1, "value": 45},
        {"sensor_id": 2, "value": 150},  # This should fail statistical check
        {"sensor_id": 3, "value": 70, "status": "BAD"},
        {"sensor_id": 4, "value": 85},  # Missing status, will be filled
    ]

    for record in data_stream:
        try:
            result = process_data(record)
            print(f"Processed data: {result}")
        except Exception as e:
            print(f"Error processing record {record}: {e}")

if __name__ == "__main__":
    main()
