import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001/api/v1'

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true, // Important: include cookies in all requests
})

// Request interceptor - tokens are now in httpOnly cookies, no need to add Authorization header
api.interceptors.request.use(
  (config) => {
    // Tokens are automatically sent via httpOnly cookies
    // No need to manually add Authorization header
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor to handle token refresh and connection errors
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config

    // Handle connection errors (network errors, timeouts, etc.)
    if (!error.response) {
      const connectionError = new Error(
        error.code === 'ECONNREFUSED' || error.message?.includes('Network Error')
          ? 'Unable to connect to the server. Please ensure the backend is running on ' + API_BASE_URL
          : error.message || 'Network error occurred'
      )
      return Promise.reject(connectionError)
    }

    // If error is 401 and we haven't tried to refresh yet
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true

      // Refresh token is in httpOnly cookie, just call refresh endpoint
      try {
        const response = await axios.post(
          `${API_BASE_URL}/auth/refresh`,
          {}, // No body needed, refresh token is in cookie
          { withCredentials: true }
        )

        // Tokens are now in httpOnly cookies, no need to store in localStorage
        // New cookies are automatically set by backend

        // Retry original request (cookies will be sent automatically)
        return api(originalRequest)
      } catch (refreshError) {
        // Refresh failed, clear auth
        localStorage.removeItem('i_drill_user')
        // Redirect to login if needed (handled by AuthContext)
        return Promise.reject(refreshError)
      }
    }

    return Promise.reject(error)
  }
)

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
  autoStep: () => api.post('/rl/auto-step'),
  loadPolicy: (payload: { source: 'mlflow' | 'file'; model_name?: string; stage?: string; file_path?: string }) =>
    api.post('/rl/policy/load', payload),
  setPolicyMode: (payload: { mode: 'manual' | 'auto'; auto_interval_seconds?: number }) =>
    api.post('/rl/policy/mode', payload),
  getPolicyStatus: () => api.get('/rl/policy/status'),
}

// DVR API
export const dvrApi = {
  processRecord: (record: any) => api.post('/dvr/process', { record }),
  getStats: (limit = 50) => api.get('/dvr/stats', { params: { limit } }),
  getAnomalies: (historySize = 100) =>
    api.get('/dvr/anomalies', { params: { history_size: historySize } }),
  evaluateRecord: (record: any, historySize = 100) =>
    api.post('/dvr/evaluate', { record, history_size: historySize }),
  getHistory: (params: { rig_id?: string; start_time?: string; end_time?: string; limit?: number }) =>
    api.get('/dvr/history', { params }),
  exportHistoryCsv: (params: { rig_id?: string; start_time?: string; end_time?: string }) =>
    api.get('/dvr/history/export/csv', { params, responseType: 'blob' }),
  exportHistoryPdf: (params: { rig_id?: string; start_time?: string; end_time?: string }) =>
    api.get('/dvr/history/export/pdf', { params, responseType: 'blob' }),
}

// Health API
export const healthApi = {
  check: () => api.get('/health'),
  detailed: () => api.get('/health/services'),
}

// Auth API
export const authApi = {
  login: (username: string, password: string) => {
    const formData = new FormData()
    formData.append('username', username)
    formData.append('password', password)
    return api.post('/auth/login', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      withCredentials: true, // Ensure cookies are included
    })
  },

  loginJson: (username: string, password: string) =>
    api.post('/auth/login/json', { username, password }, { withCredentials: true }),

  logout: () => api.post('/auth/logout', {}, { withCredentials: true }),

  refresh: () =>
    api.post('/auth/refresh', {}, { withCredentials: true }), // Refresh token is in cookie

  me: () => api.get('/auth/me'),

  register: (data: {
    username: string
    email: string
    password: string
    full_name?: string
    role?: string
  }) => api.post('/auth/register', data),

  updatePassword: (currentPassword: string, newPassword: string) =>
    api.put('/auth/me/password', {
      current_password: currentPassword,
      new_password: newPassword,
    }),

  requestPasswordReset: (email: string) =>
    api.post('/auth/password/reset/request', { email }),

  confirmPasswordReset: (token: string, newPassword: string) =>
    api.post('/auth/password/reset/confirm', {
      token,
      new_password: newPassword,
    }),
}

// Control API - Apply changes to drilling parameters
export const controlApi = {
  applyChange: (data: {
    rig_id: string
    change_type: 'optimization' | 'maintenance' | 'validation'
    component: string
    parameter: string
    value: number | string
    auto_execute?: boolean
  }) => api.post('/control/apply-change', data),

  getChangeHistory: (rigId?: string, limit = 50) =>
    api.get('/control/change-history', { params: { rig_id: rigId, limit } }),

  approveChange: (changeId: string) => api.post(`/control/change/${changeId}/approve`),

  rejectChange: (changeId: string) => api.post(`/control/change/${changeId}/reject`),
}

