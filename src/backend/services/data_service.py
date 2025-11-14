"""
Data Service for handling sensor data operations
Provides CRUD operations and analytics queries
"""
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging

from api.models.database_models import (
    SensorData, 
    MaintenanceAlertDB, 
    MaintenanceScheduleDB,
    DVRProcessHistoryDB,
    RULPredictionDB,
    AnomalyDetectionDB,
    WellProfileDB,
    DrillingSessionDB
)
from database import db_manager

logger = logging.getLogger(__name__)


class DataService:
    """
    Service for sensor data operations.
    
    Provides CRUD operations and analytics queries for sensor data.
    Handles database interactions, data validation, and error handling.
    
    Attributes:
        db_manager: Database manager instance for database operations
        
    Example:
        ```python
        service = DataService()
        latest_data = service.get_latest_sensor_data(rig_id="RIG_01", limit=10)
        historical = service.get_historical_data(
            rig_id="RIG_01",
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now()
        )
        ```
    """
    
    def __init__(self):
        """
        Initialize DataService.
        
        Sets up the database manager for data operations.
        """
        self.db_manager = db_manager
    
    # ==================== Sensor Data Operations ====================
    
    def _db_ready(self) -> bool:
        """
        Check if database is ready for operations.
        
        Returns:
            True if database is initialized and ready, False otherwise
        """
        if not getattr(self.db_manager, "_initialized", False):
            logger.debug("Database not initialized; skipping data service operation")
            return False
        return True
    
    def get_latest_sensor_data(
        self, 
        rig_id: Optional[str] = None, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get latest sensor readings
        
        Optimized with caching for frequently accessed data.
        
        Args:
            rig_id: Filter by rig ID
            limit: Number of records to return
            
        Returns:
            List of sensor data dictionaries
        """
        if not self._db_ready():
            return []
        
        # Try cache first (TTL: 10 seconds for real-time data)
        from services.cache_service import cache_service
        cache_key = f"sensor_data:latest:{rig_id or 'all'}:{limit}"
        cached_data = cache_service.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Cache hit for latest sensor data: {cache_key}")
            return cached_data
        
        try:
            with self.db_manager.session_scope() as session:
                # Optimized query: use index on (rig_id, timestamp) if available
                query = session.query(SensorData)
                
                if rig_id:
                    query = query.filter(SensorData.rig_id == rig_id)
                
                # Order by timestamp descending (most recent first)
                # Use index on timestamp for better performance
                results = query.order_by(desc(SensorData.timestamp)).limit(limit).all()
                
                data = [self._sensor_data_to_dict(record) for record in results]
                
                # Cache result for 10 seconds (real-time data changes frequently)
                cache_service.set(cache_key, data, ttl=10)
                
                return data
                
        except Exception as e:
            logger.error(f"Error getting latest sensor data: {e}")
            return []
    
    def get_historical_data(
        self,
        rig_id: Optional[str] = None,
        start_time: datetime = None,
        end_time: datetime = None,
        parameters: Optional[List[str]] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get historical sensor data with filtering
        
        Args:
            rig_id: Filter by rig ID
            start_time: Start time for query
            end_time: End time for query
            parameters: Specific parameters to include
            limit: Number of records to return
            offset: Offset for pagination
            
        Returns:
            List of sensor data dictionaries
        """
        if not self._db_ready():
            return []
        try:
            with self.db_manager.session_scope() as session:
                # Build query
                if parameters:
                    # Select specific columns
                    columns = [getattr(SensorData, param) for param in parameters if hasattr(SensorData, param)]
                    columns.insert(0, SensorData.id)
                    columns.insert(1, SensorData.rig_id)
                    columns.insert(2, SensorData.timestamp)
                    query = session.query(*columns)
                else:
                    query = session.query(SensorData)
                
                # Apply filters
                if rig_id:
                    query = query.filter(SensorData.rig_id == rig_id)
                
                if start_time:
                    query = query.filter(SensorData.timestamp >= start_time)
                
                if end_time:
                    query = query.filter(SensorData.timestamp <= end_time)
                
                # Order and paginate
                # Use index on timestamp for better performance
                query = query.order_by(SensorData.timestamp.desc())
                
                # Optimize pagination: use cursor-based pagination for large datasets
                # For now, use offset-based but with limit on offset
                if offset > 10000:
                    logger.warning(f"Large offset detected: {offset}. Consider using cursor-based pagination.")
                
                results = query.offset(offset).limit(limit).all()
                
                return [self._sensor_data_to_dict(record) for record in results]
                
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
        Get aggregated time series data for visualization
        
        Args:
            rig_id: Rig identifier
            time_bucket_seconds: Time bucket size in seconds
            start_time: Start time (defaults to 24 hours ago)
            end_time: End time (defaults to now)
            
        Returns:
            List of aggregated data points
        """
        if not self._db_ready():
            return []
        try:
            # Set default time range
            if end_time is None:
                end_time = datetime.now()
            if start_time is None:
                start_time = end_time - timedelta(hours=24)
            
            bucket_expression = func.to_timestamp(
                func.floor(func.extract('epoch', SensorData.timestamp) / time_bucket_seconds) * time_bucket_seconds
            ).label('time_bucket')

            with self.db_manager.session_scope() as session:
                # Calculate time buckets and aggregate
                # Note: This is PostgreSQL-specific. Adjust for other databases.
                query = session.query(
                    bucket_expression,
                    func.avg(SensorData.wob).label('avg_wob'),
                    func.avg(SensorData.rpm).label('avg_rpm'),
                    func.avg(SensorData.torque).label('avg_torque'),
                    func.avg(SensorData.rop).label('avg_rop'),
                    func.avg(SensorData.mud_flow).label('avg_mud_flow'),
                    func.avg(SensorData.mud_pressure).label('avg_mud_pressure'),
                    func.max(SensorData.depth).label('max_depth'),
                ).filter(
                    and_(
                        SensorData.rig_id == rig_id,
                        SensorData.timestamp >= start_time,
                        SensorData.timestamp <= end_time
                    )
                ).group_by('time_bucket').order_by('time_bucket')
                
                results = query.all()
                
                return [{
                    'timestamp': row.time_bucket.isoformat(),
                    'avg_wob': float(row.avg_wob) if row.avg_wob else None,
                    'avg_rpm': float(row.avg_rpm) if row.avg_rpm else None,
                    'avg_torque': float(row.avg_torque) if row.avg_torque else None,
                    'avg_rop': float(row.avg_rop) if row.avg_rop else None,
                    'avg_mud_flow': float(row.avg_mud_flow) if row.avg_mud_flow else None,
                    'avg_mud_pressure': float(row.avg_mud_pressure) if row.avg_mud_pressure else None,
                    'max_depth': float(row.max_depth) if row.max_depth else None,
                } for row in results]
                
        except Exception as e:
            logger.error(f"Error getting aggregated data: {e}")
            return []
    
    def get_analytics_summary(self, rig_id: str) -> Optional[Dict[str, Any]]:
        """
        Get analytics summary for a rig
        
        Args:
            rig_id: Rig identifier
            
        Returns:
            Analytics summary dictionary
        """
        if not self._db_ready():
            return None
        try:
            with self.db_manager.session_scope() as session:
                # Get latest data point
                latest = session.query(SensorData).filter(
                    SensorData.rig_id == rig_id
                ).order_by(desc(SensorData.timestamp)).first()
                
                if not latest:
                    return None
                
                # Calculate statistics from last 24 hours
                time_24h_ago = datetime.now() - timedelta(hours=24)
                
                stats = session.query(
                    func.avg(SensorData.rop).label('avg_rop'),
                    func.sum(SensorData.rop * 1.0 / 3600).label('total_distance'),  # Assuming rop is ft/hr
                    func.count(SensorData.id).label('data_points')
                ).filter(
                    and_(
                        SensorData.rig_id == rig_id,
                        SensorData.timestamp >= time_24h_ago
                    )
                ).first()
                
                # Get maintenance alerts count
                alerts_count = session.query(func.count(MaintenanceAlertDB.id)).filter(
                    and_(
                        MaintenanceAlertDB.rig_id == rig_id,
                        MaintenanceAlertDB.resolved == False
                    )
                ).scalar()
                
                # Calculate drilling time (assuming 1 record per second)
                drilling_hours = stats.data_points / 3600.0 if stats.data_points else 0
                
                # Estimate power consumption (simplified calculation)
                power_consumption = drilling_hours * 500  # kWh (rough estimate)
                
                return {
                    'rig_id': rig_id,
                    'current_depth': latest.depth,
                    'average_rop': float(stats.avg_rop) if stats.avg_rop else 0,
                    'total_drilling_time_hours': drilling_hours,
                    'total_power_consumption': power_consumption,
                    'maintenance_alerts_count': alerts_count or 0,
                    'last_updated': latest.timestamp.isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting analytics summary: {e}")
            return None
    
    def insert_sensor_data(self, data: Dict[str, Any]) -> bool:
        """
        Insert a single sensor data record
        
        Args:
            data: Sensor data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if not self._db_ready():
            return False
        try:
            with self.db_manager.session_scope() as session:
                sensor_record = SensorData(**data)
                session.add(sensor_record)
                return True
                
        except Exception as e:
            logger.error(f"Error inserting sensor data: {e}")
            return False
    
    def bulk_insert_sensor_data(self, data_list: List[Dict[str, Any]]) -> bool:
        """
        Bulk insert sensor data records
        
        Args:
            data_list: List of sensor data dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        if not self._db_ready():
            return False
        try:
            with self.db_manager.session_scope() as session:
                objects = [SensorData(**data) for data in data_list]
                session.bulk_save_objects(objects)
                logger.info(f"Bulk inserted {len(data_list)} sensor records")
                return True
                
        except Exception as e:
            logger.error(f"Error bulk inserting sensor data: {e}")
            return False
    
    # ==================== Maintenance Operations ====================
    
    def get_maintenance_alerts(
        self,
        rig_id: Optional[str] = None,
        severity: Optional[str] = None,
        resolved: Optional[bool] = None,
        limit: int = 100,
        hours: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get maintenance alerts with optional filtering.
        
        Args:
            rig_id: Filter by rig ID
            severity: Filter by severity level
            resolved: Filter by resolved status
            limit: Maximum number of alerts to return
            hours: Filter alerts created within last N hours
            
        Returns:
            List of maintenance alert dictionaries
        """
        if not self._db_ready():
            return []
        try:
            with self.db_manager.session_scope() as session:
                query = session.query(MaintenanceAlertDB)
                
                if rig_id:
                    query = query.filter(MaintenanceAlertDB.rig_id == rig_id)
                if severity:
                    query = query.filter(MaintenanceAlertDB.severity == severity)
                if resolved is not None:
                    query = query.filter(MaintenanceAlertDB.resolved == resolved)
                if hours is not None:
                    since = datetime.now() - timedelta(hours=hours)
                    query = query.filter(MaintenanceAlertDB.created_at >= since)
                
                results = query.order_by(desc(MaintenanceAlertDB.created_at)).limit(limit).all()
                
                return [self._alert_to_dict(alert) for alert in results]
                
        except Exception as e:
            logger.error(f"Error getting maintenance alerts: {e}")
            return []
    
    def get_maintenance_alert_by_id(self, alert_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a single maintenance alert by ID.
        
        Args:
            alert_id: Alert ID to retrieve
            
        Returns:
            Maintenance alert dictionary if found, None otherwise
        """
        if not self._db_ready():
            return None
        try:
            with self.db_manager.session_scope() as session:
                alert = session.query(MaintenanceAlertDB).filter(
                    MaintenanceAlertDB.id == alert_id
                ).first()
                
                if not alert:
                    return None
                
                return self._alert_to_dict(alert)
        except Exception as e:
            logger.error(f"Error getting maintenance alert by id: {e}")
            return None
    
    def create_maintenance_alert(self, data: Dict[str, Any]) -> Optional[int]:
        """
        Create a new maintenance alert.
        
        Args:
            data: Dictionary containing alert data (rig_id, component, severity, etc.)
            
        Returns:
            Created alert ID if successful, None otherwise
        """
        if not self._db_ready():
            return None
        try:
            with self.db_manager.session_scope() as session:
                alert = MaintenanceAlertDB(**data)
                session.add(alert)
                session.flush()
                return alert.id
                
        except Exception as e:
            logger.error(f"Error creating maintenance alert: {e}")
            return None

    def acknowledge_maintenance_alert(
        self,
        alert_id: int,
        acknowledged_by: str,
        notes: Optional[str] = None,
        dvr_history_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Acknowledge a maintenance alert.
        
        Marks an alert as acknowledged and optionally links it to a DVR history entry.
        
        Args:
            alert_id: ID of the alert to acknowledge
            acknowledged_by: Username of person acknowledging the alert
            notes: Optional acknowledgement notes
            dvr_history_id: Optional DVR process history ID to link
            
        Returns:
            Updated alert dictionary if successful, None otherwise
            
        Raises:
            ValueError: If DVR history entry is specified but not found
        """
        if not self._db_ready():
            return None
        try:
            with self.db_manager.session_scope() as session:
                alert = session.query(MaintenanceAlertDB).filter(MaintenanceAlertDB.id == alert_id).first()
                if alert is None:
                    return None

                if dvr_history_id is not None:
                    exists = session.query(DVRProcessHistoryDB.id).filter(DVRProcessHistoryDB.id == dvr_history_id).first()
                    if not exists:
                        raise ValueError("DVR history entry not found")
                    alert.dvr_history_id = dvr_history_id

                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                alert.acknowledgement_notes = notes
                session.flush()
                session.refresh(alert)
                return self._alert_to_dict(alert)
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error acknowledging maintenance alert {alert_id}: {e}")
            return None

    def resolve_maintenance_alert(
        self,
        alert_id: int,
        resolved_by: Optional[str] = None,
        notes: Optional[str] = None,
        dvr_history_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve a maintenance alert.
        
        Marks an alert as resolved and optionally links it to a DVR history entry.
        Automatically acknowledges the alert if not already acknowledged.
        
        Args:
            alert_id: ID of the alert to resolve
            resolved_by: Username of person resolving the alert
            notes: Optional resolution notes
            dvr_history_id: Optional DVR process history ID to link
            
        Returns:
            Updated alert dictionary if successful, None otherwise
            
        Raises:
            ValueError: If DVR history entry is specified but not found
        """
        if not self._db_ready():
            return None
        try:
            with self.db_manager.session_scope() as session:
                alert = session.query(MaintenanceAlertDB).filter(MaintenanceAlertDB.id == alert_id).first()
                if alert is None:
                    return None

                if dvr_history_id is not None:
                    exists = session.query(DVRProcessHistoryDB.id).filter(DVRProcessHistoryDB.id == dvr_history_id).first()
                    if not exists:
                        raise ValueError("DVR history entry not found")
                    alert.dvr_history_id = dvr_history_id

                alert.resolved = True
                alert.resolved_at = datetime.now()
                alert.resolved_by = resolved_by
                alert.resolution_notes = notes
                if not alert.acknowledged:
                    alert.acknowledged = True
                    alert.acknowledged_by = resolved_by or alert.acknowledged_by
                    alert.acknowledged_at = alert.acknowledged_at or datetime.now()
                session.flush()
                session.refresh(alert)
                return self._alert_to_dict(alert)
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error resolving maintenance alert {alert_id}: {e}")
            return None
    
    def create_maintenance_schedule(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create a new maintenance schedule entry.
        
        Args:
            data: Dictionary containing schedule data (rig_id, component, scheduled_date, etc.)
            
        Returns:
            Created schedule dictionary if successful, None otherwise
        """
        if not self._db_ready():
            return None
        try:
            with self.db_manager.session_scope() as session:
                schedule = MaintenanceScheduleDB(**data)
                session.add(schedule)
                session.flush()
                session.refresh(schedule)
                return self._schedule_to_dict(schedule)
        except Exception as e:
            logger.error(f"Error creating maintenance schedule: {e}")
            return None
    
    def get_maintenance_schedules(
        self,
        rig_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        until_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get maintenance schedules with optional filtering.
        
        Args:
            rig_id: Filter by rig ID
            status: Filter by schedule status
            limit: Maximum number of schedules to return
            until_date: Filter schedules scheduled before this date
            
        Returns:
            List of maintenance schedule dictionaries
        """
        if not self._db_ready():
            return []
        try:
            with self.db_manager.session_scope() as session:
                query = session.query(MaintenanceScheduleDB)
                
                if rig_id:
                    query = query.filter(MaintenanceScheduleDB.rig_id == rig_id)
                if status:
                    query = query.filter(MaintenanceScheduleDB.status == status)
                if until_date:
                    query = query.filter(MaintenanceScheduleDB.scheduled_date <= until_date)
                
                results = query.order_by(MaintenanceScheduleDB.scheduled_date).limit(limit).all()
                
                return [self._schedule_to_dict(schedule) for schedule in results]
                
        except Exception as e:
            logger.error(f"Error getting maintenance schedules: {e}")
            return []
    
    def update_maintenance_schedule(
        self, 
        schedule_id: int, 
        data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Update an existing maintenance schedule.
        
        Args:
            schedule_id: ID of the schedule to update
            data: Dictionary containing fields to update
            
        Returns:
            Updated schedule dictionary if successful, None otherwise
        """
        if not self._db_ready():
            return None
        try:
            with self.db_manager.session_scope() as session:
                schedule = session.query(MaintenanceScheduleDB).filter(
                    MaintenanceScheduleDB.id == schedule_id
                ).first()
                
                if not schedule:
                    return None
                
                for key, value in data.items():
                    if hasattr(schedule, key) and value is not None:
                        setattr(schedule, key, value)
                
                schedule.updated_at = datetime.now()
                session.flush()
                session.refresh(schedule)
                return self._schedule_to_dict(schedule)
                
        except Exception as e:
            logger.error(f"Error updating maintenance schedule: {e}")
            return None
    
    def get_maintenance_schedule_by_id(self, schedule_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a single maintenance schedule by ID.
        
        Args:
            schedule_id: Schedule ID to retrieve
            
        Returns:
            Maintenance schedule dictionary if found, None otherwise
        """
        if not self._db_ready():
            return None
        try:
            with self.db_manager.session_scope() as session:
                schedule = session.query(MaintenanceScheduleDB).filter(
                    MaintenanceScheduleDB.id == schedule_id
                ).first()
                
                if not schedule:
                    return None
                
                return self._schedule_to_dict(schedule)
        except Exception as e:
            logger.error(f"Error getting maintenance schedule by id: {e}")
            return None
    
    def delete_maintenance_schedule(self, schedule_id: int) -> bool:
        """
        Delete a maintenance schedule.
        
        Args:
            schedule_id: ID of the schedule to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if not self._db_ready():
            return False
        try:
            with self.db_manager.session_scope() as session:
                schedule = session.query(MaintenanceScheduleDB).filter(
                    MaintenanceScheduleDB.id == schedule_id
                ).first()
                
                if not schedule:
                    return False
                
                session.delete(schedule)
                return True
        except Exception as e:
            logger.error(f"Error deleting maintenance schedule: {e}")
            return False
    
    # ==================== RUL Predictions Operations ====================
    
    def save_rul_prediction(self, data: Dict[str, Any]) -> Optional[int]:
        """
        Save a RUL (Remaining Useful Life) prediction to the database.
        
        Args:
            data: Dictionary containing prediction data (rig_id, component, predicted_rul, etc.)
            
        Returns:
            Created prediction ID if successful, None otherwise
        """
        if not self._db_ready():
            return None
        try:
            with self.db_manager.session_scope() as session:
                prediction = RULPredictionDB(**data)
                session.add(prediction)
                session.flush()
                return prediction.id
                
        except Exception as e:
            logger.error(f"Error saving RUL prediction: {e}")
            return None
    
    def get_rul_predictions(
        self,
        rig_id: str,
        component: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get RUL prediction history for a rig.
        
        Args:
            rig_id: Rig identifier
            component: Optional component filter
            limit: Maximum number of predictions to return
            
        Returns:
            List of RUL prediction dictionaries
        """
        if not self._db_ready():
            return []
        try:
            with self.db_manager.session_scope() as session:
                query = session.query(RULPredictionDB).filter(
                    RULPredictionDB.rig_id == rig_id
                )
                
                if component:
                    query = query.filter(RULPredictionDB.component == component)
                
                results = query.order_by(desc(RULPredictionDB.timestamp)).limit(limit).all()
                
                return [self._rul_prediction_to_dict(pred) for pred in results]
                
        except Exception as e:
            logger.error(f"Error getting RUL predictions: {e}")
            return []
    
    # ==================== Helper Methods ====================
    
    @staticmethod
    def _sensor_data_to_dict(record: SensorData) -> Dict[str, Any]:
        """
        Convert SensorData ORM object to dictionary.
        
        Args:
            record: SensorData database model instance
            
        Returns:
            Dictionary representation of sensor data
        """
        return {
            'id': record.id,
            'rig_id': record.rig_id,
            'timestamp': record.timestamp.isoformat(),
            'depth': record.depth,
            'wob': record.wob,
            'rpm': record.rpm,
            'torque': record.torque,
            'rop': record.rop,
            'mud_flow': record.mud_flow,
            'mud_pressure': record.mud_pressure,
            'mud_temperature': record.mud_temperature,
            'gamma_ray': record.gamma_ray,
            'resistivity': record.resistivity,
            'density': record.density,
            'porosity': record.porosity,
            'hook_load': record.hook_load,
            'vibration': record.vibration,
            'status': record.status,
        }
    
    @staticmethod
    def _alert_to_dict(alert: MaintenanceAlertDB) -> Dict[str, Any]:
        """
        Convert MaintenanceAlertDB ORM object to dictionary.
        
        Args:
            alert: MaintenanceAlertDB database model instance
            
        Returns:
            Dictionary representation of maintenance alert
        """
        return {
            'id': alert.id,
            'rig_id': alert.rig_id,
            'component': alert.component,
            'alert_type': alert.alert_type,
            'severity': alert.severity,
            'message': alert.message,
            'predicted_failure_time': alert.predicted_failure_time.isoformat() if alert.predicted_failure_time else None,
            'created_at': alert.created_at.isoformat(),
            'acknowledged': alert.acknowledged,
            'acknowledged_by': alert.acknowledged_by,
            'acknowledged_at': alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
            'acknowledgement_notes': alert.acknowledgement_notes,
            'resolved': alert.resolved,
            'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None,
            'resolved_by': alert.resolved_by,
            'resolution_notes': alert.resolution_notes,
            'dvr_history_id': alert.dvr_history_id,
        }
    
    @staticmethod
    def _schedule_to_dict(schedule: MaintenanceScheduleDB) -> Dict[str, Any]:
        """
        Convert MaintenanceScheduleDB ORM object to dictionary.
        
        Args:
            schedule: MaintenanceScheduleDB database model instance
            
        Returns:
            Dictionary representation of maintenance schedule
        """
        return {
            'id': schedule.id,
            'rig_id': schedule.rig_id,
            'component': schedule.component,
            'maintenance_type': schedule.maintenance_type,
            'scheduled_date': schedule.scheduled_date.isoformat(),
            'estimated_duration_hours': schedule.estimated_duration_hours,
            'priority': schedule.priority,
            'status': schedule.status,
            'assigned_to': schedule.assigned_to,
            'notes': schedule.notes,
            'created_at': schedule.created_at.isoformat(),
            'updated_at': schedule.updated_at.isoformat(),
        }
    
    @staticmethod
    def _rul_prediction_to_dict(prediction: RULPredictionDB) -> Dict[str, Any]:
        """
        Convert RULPredictionDB ORM object to dictionary.
        
        Args:
            prediction: RULPredictionDB database model instance
            
        Returns:
            Dictionary representation of RUL prediction
        """
        return {
            'id': prediction.id,
            'rig_id': prediction.rig_id,
            'component': prediction.component,
            'predicted_rul': prediction.predicted_rul,
            'confidence': prediction.confidence,
            'timestamp': prediction.timestamp.isoformat(),
            'model_used': prediction.model_used,
            'recommendation': prediction.recommendation,
        }
