import pytest
from src.i_drill.processing.dvr_controller import process_data

# Test case for valid data
def test_process_data_valid_record():
    """
    Tests if the controller correctly processes a valid data record.
    """
    record = {"sensor_id": 1, "value": 50}
    result = process_data(record)
    assert result is not None
    assert result['status'] == 'OK'

# Test case for data with an out-of-range value
def test_process_data_invalid_value():
    """
    Tests if the controller correctly handles a record with an out-of-range value.
    It should return None and not crash.
    """
    record = {"sensor_id": 2, "value": 150} # Value > 100
    result = process_data(record)
    assert result is None

# Test case for data with a missing value
def test_process_data_missing_value():
    """
    Tests if the controller correctly handles a record with a missing 'value'.
    It should return None and not crash.
    """
    record = {"sensor_id": 3, "status": "OK"} # 'value' key is missing
    result = process_data(record)
    assert result is None