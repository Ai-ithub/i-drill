from contextlib import contextmanager
import pandas as pd
import numpy as np
from psycopg2.pool import SimpleConnectionPool
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
) VALUES ({', '.join(['%s' for _ in COLUMNS])});
"""

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "oilrig"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "1234"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
}

_POOL = None
def _get_pool(cfg):
    global _POOL
    if _POOL is None:
        _POOL = SimpleConnectionPool(1, 10, **cfg)
    return _POOL

@contextmanager
def db_conn(**overrides):
    """Yields a pooled connection; commits/rolls back automatically."""
    cfg = {**DB_CONFIG, **overrides}
    pool = _get_pool(cfg)             # first call creates pool
    conn = pool.getconn()             # borrow from pool
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)

def insert_message(message: dict):
    data = tuple(message.get(col) for col in COLUMNS)
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(INSERT_QUERY, data)
        conn.commit()

def get_last_n_rows(n, **overrides):
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

def get_history_for_anomaly(n=50):
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



