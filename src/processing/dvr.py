from contextlib import contextmanager
import logging
import os

import numpy as np
import pandas as pd
from psycopg2 import Error as PsycopgError
from psycopg2.pool import SimpleConnectionPool

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
log = logging.getLogger("drilling")

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
    """Create or return the global connection pool."""
    global _POOL
    if _POOL is None:
        try:
            _POOL = SimpleConnectionPool(1, 10, **cfg)
            log.info("Created new DB connection pool.")
        except PsycopgError as e:
            log.exception("Failed to create DB connection pool.")
            raise
        except Exception as e:
            log.exception("Unexpected error creating DB pool.")
            raise
    return _POOL

@contextmanager
def db_conn(**overrides):
    """
    Yields a pooled connection; commits/rolls back automatically.
    Ensures the connection is always returned to the pool.
    """
    cfg = {**DB_CONFIG, **overrides}
    pool = _get_pool(cfg)
    conn = None
    try:
        try:
            conn = pool.getconn()
        except PsycopgError:
            log.exception("Failed to get a connection from the pool.")
            raise
        yield conn
        try:
            conn.commit()
        except PsycopgError:
            log.exception("Commit failed; rolling back.")
            conn.rollback()
            raise
    except Exception:
        # Errors within the with-block are logged by callers where helpful.
        raise
    finally:
        if conn is not None:
            try:
                pool.putconn(conn)
            except Exception:
                # Don't mask original exceptions with put-back failures.
                log.exception("Failed to return connection to the pool.")

def insert_message(message: dict):
    """Insert a single message dict into drilling_data."""
    try:
        data = tuple(message.get(col) for col in COLUMNS)
        if len(data) != len(COLUMNS):
            raise ValueError(f"Column mismatch: expected {len(COLUMNS)}, got {len(data)}")
        with db_conn() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(INSERT_QUERY, data)
            except PsycopgError:
                log.exception("Insert failed.")
                raise
    except Exception:
        log.exception("insert_message encountered an error.")
        raise

def get_last_n_rows(n, **overrides):
    """Return last n rows as a DataFrame, newest first."""
    try:
        with db_conn(**overrides) as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT *
                        FROM drilling_data
                        ORDER BY "timestamp" DESC
                        LIMIT %s;
                    """, (n,))
                    rows = cur.fetchall()
                    cols = [d[0] for d in cur.description]
            except PsycopgError:
                log.exception("Query for last n rows failed.")
                raise
        try:
            return pd.DataFrame(rows, columns=cols)
        except Exception:
            log.exception("Failed to construct DataFrame from query results.")
            raise
    except Exception:
        log.exception("get_last_n_rows encountered an error.")
        raise

def zscore_outlier(value, history, threshold=3.0):
    try:
        arr = np.array(history, dtype=float)
        if arr.size == 0:
            return False
        mean = arr.mean()
        std = arr.std()
        if std == 0:
            return False
        z = (float(value) - mean) / std
        return abs(z) > threshold
    except Exception:
        log.exception("zscore_outlier failed; returning False.")
        return False

def rolling_anomaly(value, history, window=50, threshold=3.0):
    try:
        if not history:
            return False
        window_data = history if len(history) < window else history[-window:]
        arr = np.array(window_data, dtype=float)
        if arr.size == 0:
            return False
        mean = arr.mean()
        std = arr.std()
        if std == 0:
            return False
        return abs(float(value) - mean) > threshold * std
    except Exception:
        log.exception("rolling_anomaly failed; returning False.")
        return False

def pca_outlier(current_values, history, threshold=3.0):
    try:
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            log.warning("sklearn not installed; PCA check skipped.")
            return False

        history_arr = np.array(history, dtype=float)
        if history_arr.ndim != 2 or history_arr.shape[0] < 2 or history_arr.shape[1] < 2:
            return False

        pca = PCA(n_components=min(2, history_arr.shape[1]))
        try:
            pcs = pca.fit_transform(history_arr)
        except Exception:
            log.exception("PCA fit/transform failed on history; returning False.")
            return False

        mean = pcs.mean(axis=0)
        std = pcs.std(axis=0)
        std[std == 0] = 1e-9

        try:
            current_pc = pca.transform([current_values])[0]
        except Exception:
            log.exception("PCA transform failed on current_values; returning False.")
            return False

        z = float(np.sqrt(np.sum(((current_pc - mean) / std) ** 2)))
        return z > threshold
    except Exception:
        log.exception("pca_outlier failed; returning False.")
        return False

def get_history_for_anomaly(n=50):
    """Build per-column histories and a multi-variate matrix for PCA."""
    try:
        df = get_last_n_rows(n)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        history_dict = {}
        for col in numeric_cols:
            try:
                history_dict[col] = df[col].dropna().astype(float).tolist()
            except Exception:
                log.exception("Failed to coerce column '%s' to float; skipping.", col)
                history_dict[col] = []
        if len(numeric_cols) >= 2:
            try:
                history_dict["multi"] = df[numeric_cols].dropna().astype(float).values.tolist()
            except Exception:
                log.exception("Failed to build multi-variate history; using empty list.")
                history_dict["multi"] = []
        else:
            history_dict["multi"] = []
        return history_dict, numeric_cols
    except Exception:
        log.exception("get_history_for_anomaly encountered an error.")
        raise

def flag_anomaly(message, history_dict, numeric_cols, window=50, threshold=3.0):
    """
    Sets message['Anomaly'] = True if any feature flags by z-score, rolling, or PCA.
    Otherwise sets message['Anomaly'] = False.
    """
    anomaly_found = False
    try:
        for key in numeric_cols:
            try:
                value = message.get(key)
                history = history_dict.get(key, [])
                if value is None or len(history) == 0:
                    continue
                if (
                    zscore_outlier(value, history, threshold)
                    or rolling_anomaly(value, history, window, threshold)
                ):
                    anomaly_found = True
                    break
            except Exception:
                # Log and keep evaluating other features
                log.exception("Feature-level anomaly check failed for '%s'.", key)

        if not anomaly_found and "multi" in history_dict and len(history_dict["multi"]) > 0:
            current = [message.get(col) for col in numeric_cols if message.get(col) is not None]
            if len(current) == len(numeric_cols):
                try:
                    if pca_outlier(current, history_dict["multi"], threshold):
                        anomaly_found = True
                except Exception:
                    log.exception("PCA anomaly check failed.")
        message["Anomaly"] = anomaly_found
        return message
    except Exception:
        log.exception("flag_anomaly encountered an error; marking as non-anomalous.")
        message["Anomaly"] = False
        return message
