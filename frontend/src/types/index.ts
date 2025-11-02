export interface SensorDataPoint {
  timestamp: string
  rig_id: string
  depth?: number
  wob?: number
  rpm?: number
  torque?: number
  rop?: number
  mud_flow_rate?: number
  mud_pressure?: number
  mud_temperature?: number
  mud_density?: number
  mud_viscosity?: number
  mud_ph?: number
  gamma_ray?: number
  resistivity?: number
  pump_status?: number
  compressor_status?: number
  power_consumption?: number
  vibration_level?: number
  bit_temperature?: number
  motor_temperature?: number
  maintenance_flag?: number
  failure_type?: string
}

export interface SensorDataResponse {
  success: boolean
  count: number
  data: SensorDataPoint[]
}

export interface AnalyticsSummary {
  rig_id: string
  total_drilling_time_hours: number
  current_depth: number
  average_rop: number
  total_power_consumption: number
  maintenance_alerts_count: number
  last_updated: string
}

export interface RULPrediction {
  success: boolean
  rig_id: string
  predicted_rul_hours: number
  confidence_score: number
  prediction_timestamp: string
  model_used: string
}

export interface MaintenanceAlert {
  id: string
  rig_id: string
  alert_type: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  message: string
  timestamp: string
  acknowledged: boolean
}

export interface Rig {
  id: string
  name: string
  status: 'active' | 'maintenance' | 'idle'
  location?: string
}

