"""
Database manager for PostgreSQL operations
"""
import psycopg2
from psycopg2 import pool
import logging
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from config_loader import config_loader

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages PostgreSQL database connections and operations"""
    
    def __init__(self):
        self.connection_pool = None
        self.available = False
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool"""
        try:
            db_config = config_loader.get_database_config()
            
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=db_config.get('pool_size', 10),
                host=db_config.get('host', 'localhost'),
                port=db_config.get('port', 5432),
                database=db_config.get('database', 'drilling_db'),
                user=db_config.get('username', 'drill_user'),
                password=db_config.get('password', ''),
            )
            logger.info("Database connection pool initialized")
            self.available = True
            
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            self.connection_pool = None
            self.available = False
    
    @contextmanager
    def get_connection(self):
        """Get database connection from pool"""
        if self.connection_pool is None:
            raise RuntimeError("Database connection pool not available")

        connection = None
        try:
            connection = self.connection_pool.getconn()
            yield connection
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if connection:
                self.connection_pool.putconn(connection)

    def is_available(self) -> bool:
        return bool(self.connection_pool is not None and self.available)

    def insert_sensor_data(self, record: Dict[str, Any]) -> bool:
        """Insert sensor data record into database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create table if not exists
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sensor_data (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP,
                        rig_id VARCHAR(50),
                        depth FLOAT,
                        wob FLOAT,
                        rpm FLOAT,
                        torque FLOAT,
                        rop FLOAT,
                        mud_flow_rate FLOAT,
                        mud_pressure FLOAT,
                        mud_temperature FLOAT,
                        mud_density FLOAT,
                        mud_viscosity FLOAT,
                        mud_ph FLOAT,
                        gamma_ray FLOAT,
                        resistivity FLOAT,
                        pump_status INTEGER,
                        compressor_status INTEGER,
                        power_consumption FLOAT,
                        vibration_level FLOAT,
                        bit_temperature FLOAT,
                        motor_temperature FLOAT,
                        status VARCHAR(20),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Insert data
                cursor.execute("""
                    INSERT INTO sensor_data (
                        timestamp, rig_id, depth, wob, rpm, torque, rop,
                        mud_flow_rate, mud_pressure, mud_temperature, mud_density,
                        mud_viscosity, mud_ph, gamma_ray, resistivity,
                        pump_status, compressor_status, power_consumption,
                        vibration_level, bit_temperature, motor_temperature, status
                    ) VALUES (
                        %(Timestamp)s, %(Rig_ID)s, %(Depth)s, %(WOB)s, %(RPM)s,
                        %(Torque)s, %(ROP)s, %(Mud_Flow_Rate)s, %(Mud_Pressure)s,
                        %(Mud_Temperature)s, %(Mud_Density)s, %(Mud_Viscosity)s,
                        %(Mud_PH)s, %(Gamma_Ray)s, %(Resistivity)s,
                        %(Pump_Status)s, %(Compressor_Status)s, %(Power_Consumption)s,
                        %(Vibration_Level)s, %(Bit_Temperature)s, %(Motor_Temperature)s,
                        %(status)s
                    )
                """, record)
                
                conn.commit()
                logger.debug(f"Inserted sensor data for RIG {record.get('Rig_ID')}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to insert sensor data: {e}")
            return False
    
    def get_historical_data(self, rig_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical sensor data for anomaly detection"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM sensor_data 
                    WHERE rig_id = %s 
                    AND created_at >= NOW() - INTERVAL '%s hours'
                    ORDER BY created_at DESC
                    LIMIT 1000
                """, (rig_id, hours))
                
                columns = [desc[0] for desc in cursor.description]
                results = []
                
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                
                logger.debug(f"Retrieved {len(results)} historical records for {rig_id}")
                return results
                
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return []
    
    def close(self):
        """Close connection pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Database connection pool closed")

# Global instance
db_manager = DatabaseManager()