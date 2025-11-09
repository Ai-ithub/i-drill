import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001/api/v1'

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Sensor Data API
export const sensorDataApi = {
  getRealtime: (rigId?: string, limit = 100) =>
    api.get('/sensor-data/realtime', { params: { rig_id: rigId, limit } }),

  getHistorical: (params: {
    rig_id?: string
    start_time: string
    end_time: string
    parameters?: string
    limit?: number
    offset?: number
  }) => api.get('/sensor-data/historical', { params }),

  getAggregated: (rigId: string, timeBucketSeconds = 60, startTime?: string, endTime?: string) =>
    api.get('/sensor-data/aggregated', {
      params: {
        rig_id: rigId,
        time_bucket_seconds: timeBucketSeconds,
        start_time: startTime,
        end_time: endTime,
      },
    }),

  getAnalytics: (rigId: string) => api.get(`/sensor-data/analytics/${rigId}`),
}

// Predictions API
export const predictionsApi = {
  predictRUL: (data: {
    rig_id: string
    sensor_data?: any[]
    model_type?: string
    lookback_window?: number
  }) => api.post('/predictions/rul', data),

  predictRULAuto: (rigId: string, lookbackHours = 24, modelType = 'lstm') =>
    api.post('/predictions/rul/auto', null, {
      params: {
        rig_id: rigId,
        lookback_hours: lookbackHours,
        model_type: modelType,
      },
    }),

  detectAnomaly: (data: any) => api.post('/predictions/anomaly-detection', data),

  getAnomalyHistory: (rigId: string) =>
    api.get(`/predictions/anomaly-detection/${rigId}`),
}

// Maintenance API
export const maintenanceApi = {
  getAlerts: (rigId?: string) => api.get('/maintenance/alerts', { params: { rig_id: rigId } }),

  getSchedule: (rigId?: string) =>
    api.get('/maintenance/schedule', { params: { rig_id: rigId } }),

  createAlert: (data: any) => api.post('/maintenance/alerts', data),

  updateSchedule: (scheduleId: string, data: any) =>
    api.put(`/maintenance/schedule/${scheduleId}`, data),

  createSchedule: (data: any) => api.post('/maintenance/schedule', data),

  deleteSchedule: (scheduleId: string) => api.delete(`/maintenance/schedule/${scheduleId}`),
}

// Reinforcement Learning API
export const rlApi = {
  getConfig: () => api.get('/rl/config'),
  getState: () => api.get('/rl/state'),
  reset: (randomInit: boolean = false) => api.post('/rl/reset', { random_init: randomInit }),
  step: (data: { wob: number; rpm: number; flow_rate: number }) => api.post('/rl/step', data),
  getHistory: (limit = 50) => api.get('/rl/history', { params: { limit } }),
}

// DVR API
export const dvrApi = {
  processRecord: (record: any) => api.post('/dvr/process', { record }),
  getStats: (limit = 50) => api.get('/dvr/stats', { params: { limit } }),
  getAnomalies: (historySize = 100) =>
    api.get('/dvr/anomalies', { params: { history_size: historySize } }),
  evaluateRecord: (record: any, historySize = 100) =>
    api.post('/dvr/evaluate', { record, history_size: historySize }),
}

// Health API
export const healthApi = {
  check: () => api.get('/health'),
}

