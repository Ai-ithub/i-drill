"""
Data Service for managing sensor data operations
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from database_manager import db_manager
from config_loader import config_loader

logger = logging.getLogger(__name__)


class DataService:
    """Service for sensor data operations"""
    
    def __init__(self):
        self.db_manager = db_manager
        
    def get_latest_sensor_data(self, rig_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get latest sensor data records
        
        Args:
            rig_id: Optional rig ID to filter by
            limit: Maximum number of records to return
            
        Returns:
            List of sensor data records
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                if rig_id:
                    cursor.execute("""
                        SELECT * FROM sensor_data 
                        WHERE rig_id = %s 
                        ORDER BY timestamp DESC 
                        LIMIT %s
                    """, (rig_id, limit))
                else:
                    cursor.execute("""
                        SELECT * FROM sensor_data 
                        ORDER BY timestamp DESC 
                        LIMIT %s
                    """, (limit,))
                
                columns = [desc[0] for desc in cursor.description]
                results = []
                
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                
                logger.info(f"Retrieved {len(results)} latest records")
                return results
                
        except Exception as e:
            logger.error(f"Error getting latest sensor data: {e}")
            return []
    
    def get_historical_data(
        self, 
        rig_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        parameters: Optional[List[str]] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get historical sensor data with filters
        
        Args:
            rig_id: Optional rig ID to filter by
            start_time: Start time for query
            end_time: End time for query
            parameters: List of parameter names to include
            limit: Maximum number of records
            offset: Offset for pagination
            
        Returns:
            List of historical sensor data records
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Build dynamic WHERE clause
                where_conditions = []
                query_params = []
                
                if rig_id:
                    where_conditions.append("rig_id = %s")
                    query_params.append(rig_id)
                
                if start_time:
                    where_conditions.append("timestamp >= %s")
                    query_params.append(start_time)
                
                if end_time:
                    where_conditions.append("timestamp <= %s")
                    query_params.append(end_time)
                
                where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
                
                # Build SELECT clause
                if parameters:
                    select_clause = ", ".join(parameters)
                else:
                    select_clause = "*"
                
                query = f"""
                    SELECT {select_clause} FROM sensor_data 
                    WHERE {where_clause} 
                    ORDER BY timestamp DESC 
                    LIMIT %s OFFSET %s
                """
                
                query_params.extend([limit, offset])
                cursor.execute(query, tuple(query_params))
                
                columns = [desc[0] for desc in cursor.description]
                results = []
                
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                
                logger.info(f"Retrieved {len(results)} historical records")
                return results
                
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return []
    
    def get_time_series_aggregated(
        self,
        rig_id: str,
        time_bucket_seconds: int = 60,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get aggregated time series data
        
        Args:
            rig_id: Rig ID
            time_bucket_seconds: Time bucket size in seconds
            start_time: Start time
            end_time: End time
            
        Returns:
            List of aggregated data points
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Default to last 24 hours if no time specified
                if not start_time:
                    start_time = datetime.now() - timedelta(hours=24)
                if not end_time:
                    end_time = datetime.now()
                
                query = f"""
                    SELECT 
                        DATE_TRUNC('epoch', timestamp)::BIGINT / {time_bucket_seconds} * {time_bucket_seconds} AS time_bucket,
                        AVG(depth) as avg_depth,
                        AVG(wob) as avg_wob,
                        AVG(rpm) as avg_rpm,
                        AVG(torque) as avg_torque,
                        AVG(rop) as avg_rop,
                        AVG(power_consumption) as avg_power_consumption,
                        COUNT(*) as data_points_count
                    FROM sensor_data 
                    WHERE rig_id = %s 
                        AND timestamp >= %s 
                        AND timestamp <= %s
                    GROUP BY time_bucket
                    ORDER BY time_bucket ASC
                """
                
                cursor.execute(query, (rig_id, start_time, end_time))
                
                columns = [desc[0] for desc in cursor.description]
                results = []
                
                for row in cursor.fetchall():
                    row_dict = dict(zip(columns, row))
                    # Convert time_bucket to datetime
                    row_dict['time_bucket'] = datetime.fromtimestamp(row_dict['time_bucket'])
                    results.append(row_dict)
                
                logger.info(f"Retrieved {len(results)} aggregated records")
                return results
                
        except Exception as e:
            logger.error(f"Error getting time series aggregated data: {e}")
            return []
    
    def get_analytics_summary(self, rig_id: str) -> Optional[Dict[str, Any]]:
        """
        Get analytics summary for a rig
        
        Args:
            rig_id: Rig ID
            
        Returns:
            Dictionary with analytics summary
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get first and last timestamp
                cursor.execute("""
                    SELECT 
                        MIN(timestamp) as first_record,
                        MAX(timestamp) as last_record,
                        MAX(depth) as current_depth,
                        AVG(rop) as avg_rop,
                        SUM(power_consumption) as total_power
                    FROM sensor_data 
                    WHERE rig_id = %s
                """, (rig_id,))
                
                row = cursor.fetchone()
                
                if row is None:
                    return None
                
                # Calculate drilling time in hours
                first_record = row[0]
                last_record = row[1]
                drilling_time_hours = 0
                if first_record and last_record:
                    drilling_time_hours = (last_record - first_record).total_seconds() / 3600
                
                summary = {
                    'rig_id': rig_id,
                    'total_drilling_time_hours': drilling_time_hours,
                    'current_depth': row[2] or 0,
                    'average_rop': row[3] or 0,
                    'total_power_consumption': row[4] or 0,
                    'maintenance_alerts_count': 0,  # TODO: Get from alerts table
                    'last_updated': row[1] or datetime.now()
                }
                
                logger.info(f"Retrieved analytics summary for {rig_id}")
                return summary
                
        except Exception as e:
            logger.error(f"Error getting analytics summary: {e}")
            return None
    
    def insert_sensor_data(self, data: Dict[str, Any]) -> bool:
        """
        Insert sensor data record
        
        Args:
            data: Sensor data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        return self.db_manager.insert_sensor_data(data)

