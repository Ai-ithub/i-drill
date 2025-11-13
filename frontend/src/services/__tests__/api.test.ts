/**
 * Unit tests for API service
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import axios from 'axios'

// Mock axios before importing the API module
vi.mock('axios', () => {
  const mockAxiosInstance = {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
    interceptors: {
      request: { use: vi.fn() },
      response: { use: vi.fn() }
    }
  }
  
  return {
    default: {
      create: vi.fn(() => mockAxiosInstance),
      post: vi.fn(),
      get: vi.fn()
    }
  }
})

import {
  api,
  sensorDataApi,
  predictionsApi,
  maintenanceApi,
  rlApi,
  dvrApi,
  healthApi,
  authApi,
  controlApi
} from '../api'

const mockedAxios = axios as any

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {}
  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value.toString()
    },
    removeItem: (key: string) => {
      delete store[key]
    },
    clear: () => {
      store = {}
    }
  }
})()

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock
})

describe('API Service', () => {
  let mockApiInstance: any

  beforeEach(() => {
    vi.clearAllMocks()
    localStorageMock.clear()
    
    // Create a fresh mock instance for each test
    mockApiInstance = {
      get: vi.fn().mockResolvedValue({ data: {} }),
      post: vi.fn().mockResolvedValue({ data: {} }),
      put: vi.fn().mockResolvedValue({ data: {} }),
      delete: vi.fn().mockResolvedValue({ data: {} }),
      interceptors: {
        request: { use: vi.fn() },
        response: { use: vi.fn() }
      }
    }
    
    mockedAxios.create.mockReturnValue(mockApiInstance)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('API instance', () => {
    it('should create axios instance', () => {
      // The API instance is created on module load
      expect(mockedAxios.create).toHaveBeenCalled()
    })
  })

  describe('sensorDataApi', () => {
    it('should get realtime data', async () => {
      const mockResponse = { data: [{ rig_id: 'RIG_01', depth: 5000 }] }
      mockApiInstance.get.mockResolvedValue(mockResponse)

      const result = await sensorDataApi.getRealtime('RIG_01', 10)
      expect(result.data).toEqual(mockResponse.data)
      expect(mockApiInstance.get).toHaveBeenCalledWith(
        '/sensor-data/realtime',
        { params: { rig_id: 'RIG_01', limit: 10 } }
      )
    })

    it('should get historical data', async () => {
      const mockResponse = { data: [] }
      mockApiInstance.get.mockResolvedValue(mockResponse)

      const params = {
        rig_id: 'RIG_01',
        start_time: '2025-01-01T00:00:00Z',
        end_time: '2025-01-02T00:00:00Z'
      }
      const result = await sensorDataApi.getHistorical(params)
      expect(result.data).toEqual(mockResponse.data)
      expect(mockApiInstance.get).toHaveBeenCalledWith('/sensor-data/historical', { params })
    })

    it('should get aggregated data', async () => {
      const mockResponse = { data: [] }
      mockApiInstance.get.mockResolvedValue(mockResponse)

      const result = await sensorDataApi.getAggregated('RIG_01', 60)
      expect(result.data).toEqual(mockResponse.data)
      expect(mockApiInstance.get).toHaveBeenCalledWith(
        '/sensor-data/aggregated',
        {
          params: {
            rig_id: 'RIG_01',
            time_bucket_seconds: 60,
            start_time: undefined,
            end_time: undefined
          }
        }
      )
    })

    it('should get analytics', async () => {
      const mockResponse = { data: { rig_id: 'RIG_01', current_depth: 5000 } }
      mockApiInstance.get.mockResolvedValue(mockResponse)

      const result = await sensorDataApi.getAnalytics('RIG_01')
      expect(result.data).toEqual(mockResponse.data)
      expect(mockApiInstance.get).toHaveBeenCalledWith('/sensor-data/analytics/RIG_01')
    })
  })

  describe('predictionsApi', () => {
    it('should predict RUL', async () => {
      const mockResponse = { data: { success: true, predictions: [] } }
      mockApiInstance.post.mockResolvedValue(mockResponse)

      const data = {
        rig_id: 'RIG_01',
        sensor_data: [],
        model_type: 'lstm'
      }
      const result = await predictionsApi.predictRUL(data)
      expect(result.data).toEqual(mockResponse.data)
      expect(mockApiInstance.post).toHaveBeenCalledWith('/predictions/rul', data)
    })

    it('should predict RUL auto', async () => {
      const mockResponse = { data: { success: true, predictions: [] } }
      mockApiInstance.post.mockResolvedValue(mockResponse)

      const result = await predictionsApi.predictRULAuto('RIG_01', 24, 'lstm')
      expect(result.data).toEqual(mockResponse.data)
      expect(mockApiInstance.post).toHaveBeenCalledWith(
        '/predictions/rul/auto',
        null,
        {
          params: {
            rig_id: 'RIG_01',
            lookback_hours: 24,
            model_type: 'lstm'
          }
        }
      )
    })

    it('should detect anomaly', async () => {
      const mockResponse = { data: { has_anomaly: false } }
      mockApiInstance.post.mockResolvedValue(mockResponse)

      const result = await predictionsApi.detectAnomaly({})
      expect(result.data).toEqual(mockResponse.data)
      expect(mockApiInstance.post).toHaveBeenCalledWith('/predictions/anomaly-detection', {})
    })

    it('should get anomaly history', async () => {
      const mockResponse = { data: [] }
      mockApiInstance.get.mockResolvedValue(mockResponse)

      const result = await predictionsApi.getAnomalyHistory('RIG_01')
      expect(result.data).toEqual(mockResponse.data)
      expect(mockApiInstance.get).toHaveBeenCalledWith('/predictions/anomaly-detection/RIG_01')
    })
  })

  describe('maintenanceApi', () => {
    it('should get alerts', async () => {
      const mockResponse = { data: [] }
      mockedAxios.create.mockReturnValue({
        get: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await maintenanceApi.getAlerts('RIG_01')
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should get schedule', async () => {
      const mockResponse = { data: [] }
      mockedAxios.create.mockReturnValue({
        get: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await maintenanceApi.getSchedule('RIG_01')
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should create alert', async () => {
      const mockResponse = { data: { id: 1 } }
      mockedAxios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await maintenanceApi.createAlert({})
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should update schedule', async () => {
      const mockResponse = { data: { id: 1 } }
      mockedAxios.create.mockReturnValue({
        put: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await maintenanceApi.updateSchedule('1', {})
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should create schedule', async () => {
      const mockResponse = { data: { id: 1 } }
      mockedAxios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await maintenanceApi.createSchedule({})
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should delete schedule', async () => {
      const mockResponse = { data: {} }
      mockedAxios.create.mockReturnValue({
        delete: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await maintenanceApi.deleteSchedule('1')
      expect(result.data).toEqual(mockResponse.data)
    })
  })

  describe('rlApi', () => {
    it('should get config', async () => {
      const mockResponse = { data: {} }
      mockedAxios.create.mockReturnValue({
        get: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await rlApi.getConfig()
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should get state', async () => {
      const mockResponse = { data: {} }
      mockedAxios.create.mockReturnValue({
        get: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await rlApi.getState()
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should reset', async () => {
      const mockResponse = { data: {} }
      mockedAxios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await rlApi.reset(false)
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should step', async () => {
      const mockResponse = { data: {} }
      mockedAxios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await rlApi.step({ wob: 1500, rpm: 80, flow_rate: 1200 })
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should get history', async () => {
      const mockResponse = { data: [] }
      mockedAxios.create.mockReturnValue({
        get: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await rlApi.getHistory(50)
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should auto step', async () => {
      const mockResponse = { data: {} }
      mockedAxios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await rlApi.autoStep()
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should load policy', async () => {
      const mockResponse = { data: {} }
      mockedAxios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await rlApi.loadPolicy({ source: 'mlflow', model_name: 'test' })
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should set policy mode', async () => {
      const mockResponse = { data: {} }
      mockedAxios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await rlApi.setPolicyMode({ mode: 'auto', auto_interval_seconds: 60 })
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should get policy status', async () => {
      const mockResponse = { data: {} }
      mockedAxios.create.mockReturnValue({
        get: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await rlApi.getPolicyStatus()
      expect(result.data).toEqual(mockResponse.data)
    })
  })

  describe('dvrApi', () => {
    it('should process record', async () => {
      const mockResponse = { data: {} }
      mockedAxios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await dvrApi.processRecord({})
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should get stats', async () => {
      const mockResponse = { data: {} }
      mockedAxios.create.mockReturnValue({
        get: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await dvrApi.getStats(50)
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should get anomalies', async () => {
      const mockResponse = { data: [] }
      mockedAxios.create.mockReturnValue({
        get: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await dvrApi.getAnomalies(100)
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should evaluate record', async () => {
      const mockResponse = { data: {} }
      mockedAxios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await dvrApi.evaluateRecord({}, 100)
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should get history', async () => {
      const mockResponse = { data: [] }
      mockedAxios.create.mockReturnValue({
        get: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await dvrApi.getHistory({})
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should export history CSV', async () => {
      const mockResponse = { data: new Blob() }
      mockedAxios.create.mockReturnValue({
        get: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await dvrApi.exportHistoryCsv({})
      expect(result.data).toBeInstanceOf(Blob)
    })

    it('should export history PDF', async () => {
      const mockResponse = { data: new Blob() }
      mockedAxios.create.mockReturnValue({
        get: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await dvrApi.exportHistoryPdf({})
      expect(result.data).toBeInstanceOf(Blob)
    })
  })

  describe('healthApi', () => {
    it('should check health', async () => {
      const mockResponse = { data: { status: 'healthy' } }
      mockedAxios.create.mockReturnValue({
        get: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await healthApi.check()
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should get detailed health', async () => {
      const mockResponse = { data: { services: {} } }
      mockedAxios.create.mockReturnValue({
        get: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await healthApi.detailed()
      expect(result.data).toEqual(mockResponse.data)
    })
  })

  describe('authApi', () => {
    it('should login', async () => {
      const mockResponse = { data: { access_token: 'token' } }
      mockedAxios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await authApi.login('user', 'pass')
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should login with JSON', async () => {
      const mockResponse = { data: { access_token: 'token' } }
      mockedAxios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await authApi.loginJson('user', 'pass')
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should logout', async () => {
      const mockResponse = { data: {} }
      mockedAxios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await authApi.logout()
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should refresh token', async () => {
      const mockResponse = { data: { access_token: 'new-token' } }
      mockedAxios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await authApi.refresh('refresh-token')
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should get current user', async () => {
      const mockResponse = { data: { username: 'user' } }
      mockedAxios.create.mockReturnValue({
        get: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await authApi.me()
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should register', async () => {
      const mockResponse = { data: { id: 1 } }
      mockedAxios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await authApi.register({
        username: 'user',
        email: 'user@example.com',
        password: 'pass'
      })
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should update password', async () => {
      const mockResponse = { data: {} }
      mockedAxios.create.mockReturnValue({
        put: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await authApi.updatePassword('old', 'new')
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should request password reset', async () => {
      const mockResponse = { data: {} }
      mockedAxios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await authApi.requestPasswordReset('user@example.com')
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should confirm password reset', async () => {
      const mockResponse = { data: {} }
      mockedAxios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await authApi.confirmPasswordReset('token', 'new-pass')
      expect(result.data).toEqual(mockResponse.data)
    })
  })

  describe('controlApi', () => {
    it('should apply change', async () => {
      const mockResponse = { data: { id: 1 } }
      mockedAxios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await controlApi.applyChange({
        rig_id: 'RIG_01',
        change_type: 'optimization',
        component: 'Motor',
        parameter: 'rpm',
        value: 80
      })
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should get change history', async () => {
      const mockResponse = { data: [] }
      mockedAxios.create.mockReturnValue({
        get: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await controlApi.getChangeHistory('RIG_01', 50)
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should approve change', async () => {
      const mockResponse = { data: {} }
      mockedAxios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await controlApi.approveChange('1')
      expect(result.data).toEqual(mockResponse.data)
    })

    it('should reject change', async () => {
      const mockResponse = { data: {} }
      mockedAxios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
      })

      const result = await controlApi.rejectChange('1')
      expect(result.data).toEqual(mockResponse.data)
    })
  })
})

