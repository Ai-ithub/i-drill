"""
SQLAlchemy Database Models
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class SensorData(Base):
    """Sensor data table"""
    __tablename__ = "sensor_data"
    
    id = Column(Integer, primary_key=True, index=True)
    rig_id = Column(String(50), index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.now, index=True, nullable=False)
    
    # Drilling Parameters
    depth = Column(Float, nullable=False)
    wob = Column(Float, nullable=False)
    rpm = Column(Float, nullable=False)
    torque = Column(Float, nullable=False)
    rop = Column(Float, nullable=False)
    
    # Mud Parameters
    mud_flow = Column(Float, nullable=False)
    mud_pressure = Column(Float, nullable=False)
    mud_temperature = Column(Float, nullable=True)
    
    # Formation Parameters
    gamma_ray = Column(Float, nullable=True)
    resistivity = Column(Float, nullable=True)
    density = Column(Float, nullable=True)
    porosity = Column(Float, nullable=True)
    
    # Equipment Health
    hook_load = Column(Float, nullable=True)
    vibration = Column(Float, nullable=True)
    
    # Status
    status = Column(String(20), default="normal")
    
    # Indexes for faster queries
    __table_args__ = (
        {'mysql_engine': 'InnoDB', 'mysql_charset': 'utf8mb4'},
    )


class MaintenanceAlertDB(Base):
    """Maintenance alerts table"""
    __tablename__ = "maintenance_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    rig_id = Column(String(50), index=True, nullable=False)
    component = Column(String(100), nullable=False)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False, index=True)
    message = Column(Text, nullable=False)
    predicted_failure_time = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.now, index=True)
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(100), nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)


class MaintenanceScheduleDB(Base):
    """Maintenance schedule table"""
    __tablename__ = "maintenance_schedules"
    
    id = Column(Integer, primary_key=True, index=True)
    rig_id = Column(String(50), index=True, nullable=False)
    component = Column(String(100), nullable=False)
    maintenance_type = Column(String(50), nullable=False)
    scheduled_date = Column(DateTime, nullable=False, index=True)
    estimated_duration_hours = Column(Float, nullable=False)
    priority = Column(String(20), nullable=False)
    status = Column(String(20), default="scheduled", index=True)
    assigned_to = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class UserDB(Base):
    """Users table"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)
    role = Column(String(20), nullable=False, default="viewer")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    last_login = Column(DateTime, nullable=True)


class RULPredictionDB(Base):
    """RUL predictions history table"""
    __tablename__ = "rul_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    rig_id = Column(String(50), index=True, nullable=False)
    component = Column(String(100), nullable=False)
    predicted_rul = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    model_used = Column(String(50), nullable=False)
    recommendation = Column(Text, nullable=True)
    actual_failure_time = Column(DateTime, nullable=True)


class AnomalyDetectionDB(Base):
    """Anomaly detection results table"""
    __tablename__ = "anomaly_detections"
    
    id = Column(Integer, primary_key=True, index=True)
    rig_id = Column(String(50), index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    is_anomaly = Column(Boolean, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    affected_parameters = Column(JSON, nullable=False)
    severity = Column(String(20), nullable=False)
    description = Column(Text, nullable=True)
    investigated = Column(Boolean, default=False)
    investigation_notes = Column(Text, nullable=True)


class DVRProcessHistoryDB(Base):
    """DVR processing history table"""
    __tablename__ = "dvr_process_history"

    id = Column(Integer, primary_key=True, index=True)
    rig_id = Column(String(50), index=True, nullable=True)
    raw_record = Column(JSON, nullable=False)
    reconciled_record = Column(JSON, nullable=True)
    is_valid = Column(Boolean, nullable=False, default=True)
    reason = Column(Text, nullable=True)
    anomaly_flag = Column(Boolean, nullable=False, default=False)
    anomaly_details = Column(JSON, nullable=True)
    status = Column(String(20), nullable=False, default="processed", index=True)
    notes = Column(Text, nullable=True)
    source = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.now, index=True)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class ModelVersionDB(Base):
    """ML Model versions table"""
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)
    file_path = Column(String(255), nullable=False)
    metrics = Column(JSON, nullable=True)
    training_date = Column(DateTime, default=datetime.now)
    is_active = Column(Boolean, default=False)
    description = Column(Text, nullable=True)
    created_by = Column(String(100), nullable=True)


class WellProfileDB(Base):
    """Well profiles table"""
    __tablename__ = "well_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    well_id = Column(String(50), unique=True, index=True, nullable=False)
    rig_id = Column(String(50), index=True, nullable=False)
    total_depth = Column(Float, nullable=False)
    kick_off_point = Column(Float, nullable=False)
    build_rate = Column(Float, nullable=False)
    max_inclination = Column(Float, nullable=False)
    target_zone_start = Column(Float, nullable=False)
    target_zone_end = Column(Float, nullable=False)
    geological_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class DrillingSessionDB(Base):
    """Drilling sessions table"""
    __tablename__ = "drilling_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    rig_id = Column(String(50), index=True, nullable=False)
    well_id = Column(String(50), index=True, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    start_depth = Column(Float, nullable=False)
    end_depth = Column(Float, nullable=True)
    average_rop = Column(Float, nullable=True)
    total_drilling_time_hours = Column(Float, nullable=True)
    status = Column(String(20), default="active")
    notes = Column(Text, nullable=True)


class SystemLogDB(Base):
    """System logs table"""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    level = Column(String(20), nullable=False, index=True)
    service = Column(String(50), nullable=False, index=True)
    message = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)

