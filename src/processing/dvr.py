import pandas as pd
import numpy as np
import psycopg2


COLUMNS = [
    "Timestamp", "Rig_ID", "Depth", "WOB", "RPM", "Torque", "ROP", "Mud_Flow_Rate",
    "Mud_Pressure", "Mud_Temperature", "Mud_Density", "Mud_Viscosity", "Mud_PH",
    "Gamma_Ray", "Resistivity", "Pump_Status", "Compressor_Status", "Power_Consumption",
    "Vibration_Level", "Bit_Temperature", "Motor_Temperature", "Maintenance_Flag",
    "Failure_Type"
]

INSERT_QUERY = f"""
INSERT INTO sensor_data (
    {', '.join([f'"{col}"' for col in COLUMNS])}
) VALUES ({', '.join(['%s' for _ in COLUMNS])})
ON CONFLICT ("Timestamp") DO UPDATE SET
    {', '.join([f'"{col}"=EXCLUDED."{col}"' for col in COLUMNS if col != 'Timestamp'])}
"""

def insert_message(message: dict):
    """Insert a single message into PostgreSQL sensor_data table."""

    conn = psycopg2.connect(
        dbname="oilrig",
        user="postgres",
        password="1234",
        host="localhost",
        port=5432
    )

    cursor = conn.cursor()
    data = tuple(message.get(col) for col in COLUMNS)
    cursor.execute(INSERT_QUERY, data)
    conn.commit()
    cursor.close()
    conn.close()

def get_last_n_rows(n, dbname="oilrig", user="postgres", password="1234", host="localhost", port=5432):
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )
    cursor = conn.cursor()

    query = f'''
    SELECT *
    FROM sensor_data
    ORDER BY "Timestamp" DESC
    LIMIT %s;
    '''
    cursor.execute(query, (n,))
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=columns)

    cursor.close()
    conn.close()
    return df

def zscore_outlier(value, history, threshold=3.0):
    arr = np.array(history)
    mean = arr.mean()
    std = arr.std()
    if std == 0 or arr.size == 0:
        return False
    z = (value - mean) / std
    return abs(z) > threshold

def rolling_anomaly(value, history, window=50, threshold=3.0):
    if len(history) < window:
        window_data = history
    else:
        window_data = history[-window:]
    arr = np.array(window_data)
    mean = arr.mean()
    std = arr.std()
    if std == 0 or arr.size == 0:
        return False
    return abs(value - mean) > threshold * std

def pca_outlier(current_values, history, threshold=3.0):
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return False

    history_arr = np.array(history)
    if history_arr.shape[0] < 2 or history_arr.shape[1] < 2:
        return False

    pca = PCA(n_components=min(2, history_arr.shape[1]))
    pcs = pca.fit_transform(history_arr)

    mean = pcs.mean(axis=0)
    std = pcs.std(axis=0)

    std[std == 0] = 1e-9

    current_pc = pca.transform([current_values])[0]
    z = np.sqrt(np.sum(((current_pc - mean) / std) ** 2))
    return z > threshold

def get_history_for_anomaly(n=50, dbname="oilrig", user="postgres", password="1234", host="localhost", port=5432):
    df = get_last_n_rows(n, dbname, user, password, host, port)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    history_dict = {}
    for col in numeric_cols:
        history_dict[col] = df[col].dropna().tolist()
    if len(numeric_cols) >= 2:
        history_dict["multi"] = df[numeric_cols].dropna().values.tolist()
    else:
        history_dict["multi"] = []
    return history_dict, numeric_cols

def flag_anomaly(message, history_dict, numeric_cols, window=50, threshold=3.0):
    """
    Sets message["Anomaly"] = True if any feature is anomalous by z-score, rolling, or PCA.
    Otherwise sets message["Anomaly"] = False.
    """
    anomaly_found = False

    for key in numeric_cols:
        value = message.get(key)
        history = history_dict.get(key, [])
        if value is None or len(history) == 0:
            continue
        if zscore_outlier(value, history, threshold) or rolling_anomaly(value, history, window, threshold):
            anomaly_found = True
            break

    if not anomaly_found and "multi" in history_dict and len(history_dict["multi"]) > 0:
        current = [message.get(col) for col in numeric_cols if message.get(col) is not None]
        if len(current) == len(numeric_cols):
            if pca_outlier(current, history_dict["multi"], threshold):
                anomaly_found = True
    message["Anomaly"] = anomaly_found
    return message



