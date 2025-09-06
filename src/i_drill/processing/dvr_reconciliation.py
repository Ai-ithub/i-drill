def reconcile_data(data):
    """
    Simple reconciliation example.
    For example, fill missing 'status' field with 'OK'.
    """
    if "status" not in data:
        data["status"] = "OK"
    return data
