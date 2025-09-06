def run_statistical_checks(data):
    """
    Performs basic statistical checks on a data record.
    Instead of raising an error, it returns a status and a reason.

    Args:
        data (dict): A dictionary representing a sensor reading.

    Returns:
        tuple: A tuple containing (is_valid, reason).
               is_valid (bool): True if data is valid, False otherwise.
               reason (str): A message explaining why the data is invalid.
    """
    value = data.get("value")
    if value is None:
        return (False, "'value' is missing from the data.")
    
    if not (0 <= value <= 100):
        return (False, f"Value {value} is out of the valid range (0-100).")
    
    return (True, "Data is valid.")