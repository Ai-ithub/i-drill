from contextlib import contextmanager

import pandas as pd
import numpy as np
import psycopg2
import os

COLUMNS = [
    "timestamp", "rig_id", "depth", "wob", "rpm", "torque", "rop", "mud_flow_rate",
    "mud_pressure", "mud_temperature", "mud_density", "mud_viscosity", "mud_ph",
    "gamma_ray", "resistivity", "pump_status", "compressor_status", "power_consumption",
    "vibration_level", "bit_temperature", "motor_temperature", "maintenance_flag",
    "failure_type"
]


INSERT_QUERY = f"""
INSERT INTO drilling_data (
    {', '.join([f'"{col}"' for col in COLUMNS])}
) VALUES ({', '.join(['%s' for _ in COLUMNS])})
ON CONFLICT ("timestamp") DO UPDATE SET
    {', '.join([f'"{col}"=EXCLUDED."{col}"' for col in COLUMNS if col != 'timestamp'])}
"""

# put creds in envs or a single dict (don’t hardcode in multiple places)
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "oilrig"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "1234"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
}

@contextmanager
def db_conn(**overrides):
    """
    Context manager for database connections.
    
    Yields a psycopg2 connection with automatic commit/rollback/close handling.
    On success, commits the transaction. On exception, rolls back.
    
    Args:
        **overrides: Optional database configuration overrides
        
    Yields:
        psycopg2 connection object
    """
    cfg = {**DB_CONFIG, **overrides}
    conn = psycopg2.connect(**cfg)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def insert_message(message: dict) -> None:
    """
    Insert a message into the drilling_data table.
    
    Inserts sensor data into the database using the predefined INSERT_QUERY.
    Handles conflicts by updating existing records.
    
    Args:
        message: Dictionary containing sensor data with keys matching COLUMNS
    """
    data = tuple(message.get(col) for col in COLUMNS)
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(INSERT_QUERY, data)
        conn.commit()

def get_last_n_rows(n: int, **overrides) -> pd.DataFrame:
    """
    Get the last N rows from drilling_data table.
    
    Retrieves the most recent N records ordered by timestamp descending.
    
    Args:
        n: Number of rows to retrieve
        **overrides: Optional database configuration overrides
        
    Returns:
        DataFrame containing the last N rows with all columns
    """
    with db_conn(**overrides) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT *
            FROM drilling_data
            ORDER BY "timestamp" DESC
            LIMIT %s;
        """, (n,))
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


def zscore_outlier(value: float, history: list, threshold: float = 3.0) -> bool:
    """
    Detect outlier using z-score method.
    
    Calculates z-score of value against historical data and checks if it
    exceeds the threshold (default: 3 standard deviations).
    
    Args:
        value: Current value to check
        history: List of historical values
        threshold: Z-score threshold (default: 3.0)
        
    Returns:
        True if value is an outlier, False otherwise
    """
    arr = np.array(history)
    mean = arr.mean()
    std = arr.std()
    if std == 0 or arr.size == 0:
        return False
    z = (value - mean) / std
    return abs(z) > threshold

def rolling_anomaly(value: float, history: list, window: int = 50, threshold: float = 3.0) -> bool:
    """
    Detect anomaly using rolling window method.
    
    Uses a rolling window of recent historical data to calculate mean and
    standard deviation, then checks if value exceeds threshold standard deviations.
    
    Args:
        value: Current value to check
        history: List of historical values
        window: Size of rolling window (default: 50)
        threshold: Standard deviation threshold (default: 3.0)
        
    Returns:
        True if value is an anomaly, False otherwise
    """
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

def pca_outlier(current_values: list, history: list, threshold: float = 3.0) -> bool:
    """
    Detect outlier using Principal Component Analysis (PCA).
    
    Performs PCA on historical multi-dimensional data and checks if
    the current values represent an outlier in the principal component space.
    
    Args:
        current_values: List of current feature values
        history: List of historical multi-dimensional data points
        threshold: Outlier threshold in PCA space (default: 3.0)
        
    Returns:
        True if current values are an outlier, False otherwise
    """
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

def get_history_for_anomaly(n: int = 50) -> tuple:
    """
    Get historical data for anomaly detection.
    
    Retrieves the last N rows and extracts numeric columns for use in
    anomaly detection algorithms.
    
    Args:
        n: Number of historical rows to retrieve (default: 50)
        
    Returns:
        Tuple of (history_dict, numeric_cols) where:
        - history_dict: Dictionary mapping column names to value lists,
          plus "multi" key for multi-dimensional data
        - numeric_cols: List of numeric column names
    """
    df = get_last_n_rows(n)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    history_dict = {}
    for col in numeric_cols:
        history_dict[col] = df[col].dropna().tolist()
    if len(numeric_cols) >= 2:
        history_dict["multi"] = df[numeric_cols].dropna().values.tolist()
    else:
        history_dict["multi"] = []
    return history_dict, numeric_cols

def flag_anomaly(message: dict, history_dict: dict, numeric_cols: list, 
                 window: int = 50, threshold: float = 3.0) -> dict:
    """
    Flag anomalies in a message using multiple detection methods.
    
    Checks for anomalies using z-score, rolling window, and PCA methods.
    Sets message["Anomaly"] = True if any method detects an anomaly.
    
    Args:
        message: Dictionary containing sensor data to check
        history_dict: Dictionary of historical data by column name
        numeric_cols: List of numeric column names to check
        window: Rolling window size for rolling_anomaly (default: 50)
        threshold: Threshold for all detection methods (default: 3.0)
        
    Returns:
        Message dictionary with "Anomaly" flag set
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



