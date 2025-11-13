"""
Pydantic Schemas for API Request/Response Models
"""
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum


# ==================== Sensor Data Schemas ====================

class SensorDataPoint(BaseModel):
    """Single sensor data point"""
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = None
    rig_id: str = Field(..., description="Rig identifier")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Drilling Parameters
    depth: float = Field(..., ge=0, description="Current depth in feet")
    wob: float = Field(..., ge=0, description="Weight on Bit in lbs")
    rpm: float = Field(..., ge=0, le=300, description="Rotary speed in RPM")
    torque: float = Field(..., ge=0, description="Torque in ft-lbs")
    rop: float = Field(..., ge=0, description="Rate of Penetration in ft/hr")
    
    # Mud Parameters
    mud_flow: float = Field(..., ge=0, description="Mud flow rate in gpm")
    mud_pressure: float = Field(..., ge=0, description="Standpipe pressure in psi")
    mud_temperature: Optional[float] = Field(None, description="Mud temperature in °C")
    
    # Formation Parameters (LWD/MWD)
    gamma_ray: Optional[float] = Field(None, ge=0, description="Gamma ray in API units")
    resistivity: Optional[float] = Field(None, ge=0, description="Resistivity in ohm-m")
    density: Optional[float] = Field(None, ge=0, description="Formation density in g/cc")
    porosity: Optional[float] = Field(None, ge=0, le=100, description="Porosity in %")
    
    # Equipment Health
    hook_load: Optional[float] = Field(None, ge=0, description="Hook load in lbs")
    vibration: Optional[float] = Field(None, ge=0, description="Vibration in g")
    
    # Status
    status: Optional[str] = Field("normal", description="Operational status")
    
    @field_validator('timestamp', mode='before')
    @classmethod
    def parse_timestamp(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v


class SensorDataResponse(BaseModel):
    """Response model for sensor data queries"""
    success: bool = True
    count: int
    data: List[SensorDataPoint]
    message: Optional[str] = None


class HistoricalDataQuery(BaseModel):
    """Query parameters for historical data"""
    rig_id: Optional[str] = None
    start_time: datetime
    end_time: datetime
    parameters: Optional[List[str]] = None
    limit: int = Field(1000, ge=1, le=10000)
    offset: int = Field(0, ge=0)


class AggregatedDataResponse(BaseModel):
    """Response for aggregated time-series data"""
    success: bool = True
    count: int
    time_bucket: str
    data: List[Dict[str, Any]]


class AnalyticsSummary(BaseModel):
    """Analytics summary for a rig"""
    rig_id: str
    current_depth: float
    average_rop: float
    total_drilling_time_hours: float
    total_power_consumption: float
    maintenance_alerts_count: int
    last_updated: datetime


# ==================== WebSocket Schemas ====================

class MessageType(str, Enum):
    """WebSocket message types"""
    SENSOR_DATA = "sensor_data"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"


class WebSocketMessage(BaseModel):
    """WebSocket message structure"""
    message_type: MessageType
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


# ==================== Prediction Schemas ====================

class ModelType(str, Enum):
    """Available model types"""
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    CNN_LSTM = "cnn_lstm"
    RANDOM_FOREST = "random_forest"
    LINEAR = "linear"


class PredictionRequest(BaseModel):
    """Request for RUL prediction"""
    rig_id: str = Field(..., description="Rig identifier")
    sensor_data: Optional[List[Dict[str, Any]]] = Field(None, description="Historical sensor data")
    model_type: ModelType = Field(ModelType.LSTM, description="Model type to use")
    lookback_window: int = Field(50, ge=10, le=200, description="Lookback window size")


# Alias for backward compatibility
RULPredictionRequest = PredictionRequest


class RULPrediction(BaseModel):
    """RUL prediction result"""
    rig_id: str
    component: str
    predicted_rul: float = Field(..., description="Predicted remaining useful life in hours")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    timestamp: datetime = Field(default_factory=datetime.now)
    model_used: str
    recommendation: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response for prediction requests"""
    success: bool = True
    predictions: List[RULPrediction]
    message: Optional[str] = None


# Alias for backward compatibility
RULPredictionResponse = PredictionResponse


class AnomalyDetectionRequest(BaseModel):
    """Request for anomaly detection"""
    rig_id: str
    sensor_data: List[Dict[str, Any]]
    threshold: float = Field(0.8, ge=0, le=1, description="Anomaly threshold")


class AnomalyDetectionResult(BaseModel):
    """Anomaly detection result"""
    rig_id: str
    timestamp: datetime
    is_anomaly: bool
    anomaly_score: float
    affected_parameters: List[str]
    severity: str = Field(..., pattern=r"^(low|medium|high|critical)$")
    description: Optional[str] = None


class AnomalyDetectionResponse(BaseModel):
    """Response for anomaly detection"""
    success: bool = True
    results: List[AnomalyDetectionResult]
    message: Optional[str] = None


class TrainingJobRequest(BaseModel):
    model_name: str
    experiment_name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class TrainingJobResponse(BaseModel):
    success: bool
    run_id: Optional[str] = None
    experiment_name: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class ModelPromotionRequest(BaseModel):
    model_name: str
    version: str
    stage: str


class OperationStatusResponse(BaseModel):
    success: bool
    message: Optional[str] = None


class ModelRegistryEntry(BaseModel):
    name: str
    creation_timestamp: Optional[int] = None
    last_updated_timestamp: Optional[int] = None
    latest_versions: List[Dict[str, Any]] = Field(default_factory=list)


class ModelRegistryResponse(BaseModel):
    success: bool
    models: List[ModelRegistryEntry]


class ModelVersionListResponse(BaseModel):
    success: bool
    versions: List[Dict[str, Any]] = Field(default_factory=list)


# ==================== Maintenance Schemas ====================

class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class MaintenanceAlert(BaseModel):
    """Maintenance alert"""
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = None
    rig_id: str
    component: str = Field(..., description="Equipment component")
    alert_type: str = Field(..., description="Type of alert")
    severity: AlertSeverity
    message: str
    predicted_failure_time: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    acknowledgement_notes: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None
    dvr_history_id: Optional[int] = None


class MaintenanceAlertResponse(BaseModel):
    """Response for maintenance alerts"""
    success: bool = True
    count: int
    alerts: List[MaintenanceAlert]


class MaintenanceSchedule(BaseModel):
    """Maintenance schedule entry"""
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = None
    rig_id: str
    component: str
    maintenance_type: str = Field(..., description="Type of maintenance")
    scheduled_date: datetime
    estimated_duration_hours: float
    priority: str = Field(..., pattern=r"^(low|medium|high)$")
    status: str = Field("scheduled", pattern=r"^(scheduled|in_progress|completed|cancelled)$")
    assigned_to: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class MaintenanceScheduleResponse(BaseModel):
    """Response for maintenance schedule"""
    success: bool = True
    count: int
    schedules: List[MaintenanceSchedule]


class CreateMaintenanceAlertRequest(BaseModel):
    rig_id: str
    component: str
    alert_type: str
    severity: AlertSeverity
    message: str
    dvr_history_id: Optional[int] = None


class UpdateMaintenanceScheduleRequest(BaseModel):
    scheduled_date: Optional[datetime] = None
    estimated_duration_hours: Optional[float] = None
    priority: Optional[str] = Field(None, pattern=r"^(low|medium|high)$")
    status: Optional[str] = Field(None, pattern=r"^(scheduled|in_progress|completed|cancelled)$")
    assigned_to: Optional[str] = None
    notes: Optional[str] = None


class MaintenanceAlertAcknowledgeRequest(BaseModel):
    acknowledged_by: str = Field(..., description="User acknowledging the alert")
    notes: Optional[str] = None
    dvr_history_id: Optional[int] = None


class MaintenanceAlertResolveRequest(BaseModel):
    resolved_by: Optional[str] = Field(None, description="User resolving the alert")
    notes: Optional[str] = None
    dvr_history_id: Optional[int] = None


# ==================== Reinforcement Learning Schemas ====================


class RLAction(BaseModel):
    wob: float = Field(..., ge=0)
    rpm: float = Field(..., ge=0)
    flow_rate: float = Field(..., ge=0)


class RLEnvironmentState(BaseModel):
    observation: List[float]
    reward: float
    done: bool
    info: Dict[str, Any] = {}
    step: int
    episode: int
    warning: Optional[str] = None
    action: Optional[Dict[str, float]] = None
    policy_mode: Optional[str] = None
    policy_loaded: Optional[bool] = None


class RLStateResponse(BaseModel):
    success: bool = True
    state: RLEnvironmentState
    message: Optional[str] = None


class RLConfigResponse(BaseModel):
    success: bool = True
    config: Dict[str, Any]


class RLHistoryResponse(BaseModel):
    success: bool = True
    history: List[RLEnvironmentState]


class RLResetRequest(BaseModel):
    random_init: bool = False


class RLPolicyMode(str, Enum):
    MANUAL = "manual"
    AUTO = "auto"


class RLPolicySource(str, Enum):
    MLFLOW = "mlflow"
    FILE = "file"


class RLPolicyStatus(BaseModel):
    mode: RLPolicyMode
    policy_loaded: bool
    source: Optional[str] = None
    identifier: Optional[str] = None
    stage: Optional[str] = None
    loaded_at: Optional[str] = None
    auto_interval_seconds: float
    message: Optional[str] = None


class RLPolicyModeRequest(BaseModel):
    mode: RLPolicyMode
    auto_interval_seconds: Optional[float] = Field(None, ge=0.5)


class RLPolicyLoadRequest(BaseModel):
    source: RLPolicySource
    model_name: Optional[str] = Field(None, description="Registered model name for MLflow sources")
    stage: Optional[str] = Field("Production", description="Model stage when loading from MLflow")
    file_path: Optional[str] = Field(None, description="Local file path when loading from file")

    @field_validator("model_name")
    @classmethod
    def _validate_model_name(cls, value, info):
        source = info.data.get("source")
        if source == RLPolicySource.MLFLOW and not value:
            raise ValueError("model_name is required when source is 'mlflow'")
        return value

    @field_validator("file_path")
    @classmethod
    def _validate_file_path(cls, value, info):
        source = info.data.get("source")
        if source == RLPolicySource.FILE and not value:
            raise ValueError("file_path is required when source is 'file'")
        return value


class RLPolicyStatusResponse(BaseModel):
    success: bool
    status: RLPolicyStatus
    message: Optional[str] = None


# ==================== DVR Schemas ====================


class DVRProcessRequest(BaseModel):
    record: Dict[str, Any]


class DVRProcessResponse(BaseModel):
    success: bool
    processed_record: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    history_id: Optional[int] = None
    anomaly_flag: Optional[bool] = None


class DVRStatsResponse(BaseModel):
    success: bool
    summary: Dict[str, Any]
    message: Optional[str] = None


class DVRAnomalyResponse(BaseModel):
    success: bool
    numeric_columns: List[str]
    history_sizes: Dict[str, int]
    message: Optional[str] = None


class DVREvaluateRequest(BaseModel):
    record: Dict[str, Any]
    history_size: int = Field(100, ge=1, le=1000)


class DVREvaluateResponse(BaseModel):
    success: bool
    record: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class DVRRecordStatus(str, Enum):
    PROCESSED = "processed"
    INVALID = "invalid"
    ERROR = "error"
    ACKNOWLEDGED = "acknowledged"


class DVRHistoryEntry(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    rig_id: Optional[str] = None
    raw_record: Dict[str, Any]
    reconciled_record: Optional[Dict[str, Any]] = None
    is_valid: bool
    reason: Optional[str] = None
    anomaly_flag: bool
    anomaly_details: Optional[Dict[str, Any]] = None
    status: DVRRecordStatus
    notes: Optional[str] = None
    source: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DVRHistoryResponse(BaseModel):
    success: bool
    count: int
    history: List[DVRHistoryEntry]


class DVRHistoryEntryResponse(BaseModel):
    success: bool
    entry: DVRHistoryEntry
    message: Optional[str] = None


class DVRHistoryUpdateRequest(BaseModel):
    status: Optional[DVRRecordStatus] = None
    notes: Optional[str] = None


# ==================== Authentication Schemas ====================

class Token(BaseModel):
    """JWT Token"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    refresh_token: Optional[str] = None


class TokenData(BaseModel):
    """Token payload data"""
    username: Optional[str] = None
    user_id: Optional[int] = None
    scopes: List[str] = []


class TokenRefreshRequest(BaseModel):
    """Token refresh request"""
    refresh_token: str


class PasswordResetRequest(BaseModel):
    """Password reset request"""
    email: str = Field(..., description="User email address")


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation"""
    token: str = Field(..., description="Password reset token")
    new_password: str = Field(..., min_length=8, description="New password")


class PasswordChangeRequest(BaseModel):
    """Password change request"""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")


class UserRole(str, Enum):
    """User roles"""
    ADMIN = "admin"
    ENGINEER = "engineer"
    OPERATOR = "operator"
    DATA_SCIENTIST = "data_scientist"
    MAINTENANCE = "maintenance"
    VIEWER = "viewer"


class User(BaseModel):
    """User model"""
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = None
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
    full_name: Optional[str] = None
    role: UserRole
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)


class UserCreate(BaseModel):
    """User creation request"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    role: UserRole = UserRole.VIEWER


class UserLogin(BaseModel):
    """User login request"""
    username: str
    password: str


class UserResponse(BaseModel):
    """User response"""
    success: bool = True
    user: User
    message: Optional[str] = None


# ==================== Configuration Schemas ====================

class WellProfileConfig(BaseModel):
    """Well profile configuration"""
    well_id: str
    total_depth: float
    kick_off_point: float
    build_rate: float
    max_inclination: float
    target_zone_start: float
    target_zone_end: float


class DrillingParametersConfig(BaseModel):
    """Drilling parameters configuration"""
    rig_id: str
    target_wob: float
    target_rpm: float
    target_mud_flow: float
    target_rop: float
    safety_limits: Dict[str, Dict[str, float]]


# ==================== Health Check Schemas ====================

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    services: Dict[str, bool] = {}
    version: str = "1.0.0"


class ServiceStatus(BaseModel):
    """Individual service status"""
    name: str
    status: str
    uptime_seconds: float
    last_check: datetime


# ==================== Error Schemas ====================

class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = False
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ValidationErrorDetail(BaseModel):
    """Validation error detail"""
    loc: List[str]
    msg: str
    type: str

