import { useMemo, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { sensorDataApi, predictionsApi, maintenanceApi } from '@/services/api'
import {
  Wrench,
  AlertTriangle,
  CheckCircle2,
  Settings,
  Zap,
  TrendingUp,
  Clock,
  Shield,
  Play,
  Pause,
  X,
  Check,
  Bot,
  User,
} from 'lucide-react'
import { useChangeManager } from '@/components/ChangeManagement/ChangeManager'

interface MaintenanceRecommendation {
  id: string
  type: 'maintenance' | 'optimization'
  component: string
  title: string
  description: string
  priority: 'critical' | 'high' | 'medium' | 'low'
  estimatedRUL?: number
  confidence: number
  action: string
  expectedImpact: string
  estimatedCost?: number
  estimatedDowntime?: number
  selected: boolean
  autoExecute: boolean
}

export default function PDMTab() {
  const queryClient = useQueryClient()
  const rigId = 'RIG_01'
  const [autoExecutionEnabled, setAutoExecutionEnabled] = useState(false)
  const [selectedRecommendations, setSelectedRecommendations] = useState<Set<string>>(new Set())
  const { createChange, changes, setAutoExecutionEnabled: setChangeManagerAutoEnabled } = useChangeManager(rigId)

  // Mock data for demonstration when backend is not available
  const mockAnalyticsData = {
    current_depth: 5000,
    average_rop: 25.5,
    total_drilling_time_hours: 120,
    total_power_consumption: 500000,
    maintenance_alerts_count: 2,
  }

  const mockRealtimeData = {
    depth: 5000,
    wob: 45000,
    rpm: 150,
    torque: 8000,
    mud_flow: 650,
    mud_pressure: 2800,
    rop: 25.5,
    pump_pressure: 3200,
  }

  const mockRulData = {
    success: true,
    predicted_rul_hours: 85,
    confidence_score: 0.88,
  }

  const mockAlertsData = [
    {
      id: 'alert-1',
      component: 'Pump System',
      alert_type: 'High Pressure Warning',
      message: 'Pump pressure is elevated. Schedule maintenance.',
      severity: 'high',
    },
    {
      id: 'alert-2',
      component: 'Drill Bit',
      alert_type: 'Wear Indicator',
      message: 'Drill bit shows signs of wear. Monitor closely.',
      severity: 'medium',
    },
  ]

  // Fetch analytics data
  const { data: analyticsData } = useQuery({
    queryKey: ['analytics', rigId],
    queryFn: () => sensorDataApi.getAnalytics(rigId).then((res) => res.data.summary),
    refetchInterval: 60000,
    retry: 1,
    retryDelay: 1000,
    refetchOnWindowFocus: false,
  })

  // Fetch real-time data
  const { data: realtimeData } = useQuery({
    queryKey: ['realtime-pdm', rigId],
    queryFn: () => sensorDataApi.getRealtime(rigId, 1).then((res) => res.data.data?.[0]),
    refetchInterval: 30000,
    retry: 1,
    retryDelay: 1000,
    refetchOnWindowFocus: false,
  })

  // Fetch RUL predictions
  const { data: rulData } = useQuery({
    queryKey: ['rul-prediction', rigId],
    queryFn: () => predictionsApi.predictRULAuto(rigId, 24, 'lstm').then((res) => res.data),
    refetchInterval: 300000, // Every 5 minutes
    retry: 1,
    retryDelay: 1000,
    refetchOnWindowFocus: false,
  })

  // Fetch maintenance alerts
  const { data: alertsData } = useQuery({
    queryKey: ['maintenance-alerts-pdm', rigId],
    queryFn: () => maintenanceApi.getAlerts(rigId).then((res) => res.data),
    refetchInterval: 60000,
    retry: 1,
    retryDelay: 1000,
    refetchOnWindowFocus: false,
  })

  // Use actual data or fallback to mock data
  const effectiveAnalyticsData = analyticsData || mockAnalyticsData
  const effectiveRealtimeData = realtimeData || mockRealtimeData
  const effectiveRulData = rulData || mockRulData
  const effectiveAlertsData = alertsData || mockAlertsData

  // Generate maintenance recommendations
  const recommendations = useMemo<MaintenanceRecommendation[]>(() => {
    const recs: MaintenanceRecommendation[] = []

    // RUL-based maintenance recommendations
    if (effectiveRulData?.success && effectiveRulData.predicted_rul_hours) {
      const rulHours = effectiveRulData.predicted_rul_hours
      const confidence = effectiveRulData.confidence_score || 0.85

      if (rulHours < 100) {
        recs.push({
          id: 'rul-critical',
          type: 'maintenance',
          component: 'Drilling Equipment',
          title: 'Critical: Immediate Maintenance Required',
          description: `Predicted RUL is ${rulHours.toFixed(1)} hours. Equipment requires immediate attention to prevent failure.`,
          priority: 'critical',
          estimatedRUL: rulHours,
          confidence,
          action: 'Schedule emergency maintenance within 24 hours',
          expectedImpact: 'Prevent catastrophic failure, reduce downtime by 80%',
          estimatedCost: 50000,
          estimatedDowntime: 48,
          selected: false,
          autoExecute: false,
        })
      } else if (rulHours < 200) {
        recs.push({
          id: 'rul-high',
          type: 'maintenance',
          component: 'Drilling Equipment',
          title: 'High Priority: Preventive Maintenance Recommended',
          description: `Predicted RUL is ${rulHours.toFixed(1)} hours. Schedule preventive maintenance to avoid unexpected failures.`,
          priority: 'high',
          estimatedRUL: rulHours,
          confidence,
          action: 'Schedule preventive maintenance within 72 hours',
          expectedImpact: 'Extend equipment life, reduce unplanned downtime by 60%',
          estimatedCost: 25000,
          estimatedDowntime: 24,
          selected: false,
          autoExecute: false,
        })
      } else if (rulHours < 500) {
        recs.push({
          id: 'rul-medium',
          type: 'maintenance',
          component: 'Drilling Equipment',
          title: 'Medium Priority: Plan Maintenance Schedule',
          description: `Predicted RUL is ${rulHours.toFixed(1)} hours. Plan maintenance schedule for optimal timing.`,
          priority: 'medium',
          estimatedRUL: rulHours,
          confidence,
          action: 'Plan maintenance within next 2 weeks',
          expectedImpact: 'Optimize maintenance timing, reduce costs by 30%',
          estimatedCost: 15000,
          estimatedDowntime: 12,
          selected: false,
          autoExecute: false,
        })
      }
    }

    // Temperature-based recommendations
    if (effectiveRealtimeData) {
      const bitTemp = effectiveRealtimeData.bit_temperature || 0
      const motorTemp = effectiveRealtimeData.motor_temperature || 0
      const mudTemp = effectiveRealtimeData.mud_temperature || 0

      if (bitTemp > 120) {
        recs.push({
          id: 'temp-bit-high',
          type: 'maintenance',
          component: 'Drill Bit',
          title: 'High Bit Temperature Detected',
          description: `Bit temperature is ${bitTemp.toFixed(1)}°C, exceeding optimal range. Risk of bit damage.`,
          priority: 'high',
          confidence: 0.9,
          action: 'Reduce drilling speed, increase mud flow, inspect bit',
          expectedImpact: 'Prevent bit failure, extend bit life by 25%',
          estimatedCost: 10000,
          estimatedDowntime: 4,
          selected: false,
          autoExecute: false,
        })
      }

      if (motorTemp > 100) {
        recs.push({
          id: 'temp-motor-high',
          type: 'maintenance',
          component: 'Motor',
          title: 'High Motor Temperature Warning',
          description: `Motor temperature is ${motorTemp.toFixed(1)}°C. Risk of motor overheating and failure.`,
          priority: 'high',
          confidence: 0.85,
          action: 'Reduce load, check cooling system, schedule inspection',
          expectedImpact: 'Prevent motor failure, avoid 48+ hour downtime',
          estimatedCost: 30000,
          estimatedDowntime: 24,
          selected: false,
          autoExecute: false,
        })
      }
    }

    // Pressure-based recommendations
    if (effectiveRealtimeData) {
      const mudPressure = effectiveRealtimeData.mud_pressure || 0
      const pumpPressure = effectiveRealtimeData.pump_pressure || 0

      if (mudPressure > 3000) {
        recs.push({
          id: 'pressure-mud-high',
          type: 'maintenance',
          component: 'Mud System',
          title: 'High Mud Pressure Alert',
          description: `Mud pressure is ${mudPressure.toFixed(1)} psi. Risk of system overload.`,
          priority: 'medium',
          confidence: 0.8,
          action: 'Check mud properties, reduce flow rate if needed',
          expectedImpact: 'Prevent system damage, maintain optimal drilling',
          estimatedCost: 5000,
          estimatedDowntime: 2,
          selected: false,
          autoExecute: false,
        })
      }

      if (pumpPressure > 3500) {
        recs.push({
          id: 'pressure-pump-high',
          type: 'maintenance',
          component: 'Pump System',
          title: 'High Pump Pressure Warning',
          description: `Pump pressure is ${pumpPressure.toFixed(1)} psi. Pump may require maintenance.`,
          priority: 'high',
          confidence: 0.85,
          action: 'Inspect pump, check for blockages, schedule maintenance',
          expectedImpact: 'Prevent pump failure, avoid production loss',
          estimatedCost: 20000,
          estimatedDowntime: 8,
          selected: false,
          autoExecute: false,
        })
      }
    }

    // Optimization recommendations for drilling conditions
    if (effectiveRealtimeData && effectiveAnalyticsData) {
      const currentROP = effectiveRealtimeData.rop || effectiveAnalyticsData.average_rop || 0
      const currentWOB = effectiveRealtimeData.wob || 0
      const currentRPM = effectiveRealtimeData.rpm || 0
      const currentTorque = effectiveRealtimeData.torque || 0

      // WOB optimization
      const optimalWOB = (effectiveRealtimeData.depth || 0) * 2.5
      if (currentWOB > 0 && Math.abs(currentWOB - optimalWOB) / optimalWOB > 0.2) {
        recs.push({
          id: 'opt-wob',
          type: 'optimization',
          component: 'Drilling Parameters',
          title: 'Optimize Weight on Bit',
          description: `Current WOB (${currentWOB.toFixed(0)} lbs) is ${currentWOB < optimalWOB ? 'below' : 'above'} optimal range (${optimalWOB.toFixed(0)} lbs).`,
          priority: 'medium',
          confidence: 0.75,
          action: currentWOB < optimalWOB ? 'Increase WOB gradually' : 'Reduce WOB to prevent bit damage',
          expectedImpact: 'Improve ROP by 15-20%, reduce drilling time',
          selected: false,
          autoExecute: false,
        })
      }

      // RPM optimization
      const optimalRPM = (effectiveRealtimeData.depth || 0) < 5000 ? 180 : (effectiveRealtimeData.depth || 0) < 10000 ? 150 : 120
      if (Math.abs(currentRPM - optimalRPM) > 20) {
        recs.push({
          id: 'opt-rpm',
          type: 'optimization',
          component: 'Drilling Parameters',
          title: 'Optimize Rotary Speed',
          description: `Current RPM (${currentRPM.toFixed(0)}) is ${currentRPM > optimalRPM ? 'above' : 'below'} optimal (${optimalRPM.toFixed(0)}).`,
          priority: 'low',
          confidence: 0.7,
          action: currentRPM > optimalRPM ? 'Reduce RPM to reduce wear' : 'Increase RPM for better penetration',
          expectedImpact: 'Extend equipment life, improve efficiency by 10%',
          selected: false,
          autoExecute: false,
        })
      }

      // Torque optimization
      const optimalTorque = currentWOB * 0.15
      if (currentTorque > 0 && Math.abs(currentTorque - optimalTorque) / optimalTorque > 0.15) {
        recs.push({
          id: 'opt-torque',
          type: 'optimization',
          component: 'Drilling Parameters',
          title: 'Optimize Torque Settings',
          description: `Current torque (${currentTorque.toFixed(0)} ft-lbs) is outside optimal range.`,
          priority: 'medium',
          confidence: 0.8,
          action: 'Adjust drilling parameters to optimize torque',
          expectedImpact: 'Reduce equipment stress, improve drilling efficiency',
          selected: false,
          autoExecute: false,
        })
      }
    }

    // Maintenance alerts from API
    if (effectiveAlertsData && Array.isArray(effectiveAlertsData)) {
      effectiveAlertsData.forEach((alert: any, index: number) => {
        recs.push({
          id: `alert-${alert.id || index}`,
          type: 'maintenance',
          component: alert.component || 'Unknown',
          title: alert.alert_type || 'Maintenance Alert',
          description: alert.message || 'Maintenance required',
          priority: (alert.severity || 'medium') as any,
          confidence: 0.9,
          action: 'Review and schedule maintenance',
          expectedImpact: 'Address maintenance issue',
          selected: false,
          autoExecute: false,
        })
      })
    }

    return recs.sort((a, b) => {
      const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 }
      return priorityOrder[b.priority] - priorityOrder[a.priority]
    })
  }, [effectiveRulData, effectiveRealtimeData, effectiveAnalyticsData, effectiveAlertsData])

  // Toggle recommendation selection
  const toggleSelection = (id: string) => {
    setSelectedRecommendations((prev) => {
      const newSet = new Set(prev)
      if (newSet.has(id)) {
        newSet.delete(id)
      } else {
        newSet.add(id)
      }
      return newSet
    })
  }

  // Toggle auto-execution for a recommendation
  const toggleAutoExecute = (id: string) => {
    // In a real implementation, this would update the recommendation's autoExecute flag
    // For now, we'll just show a message
    alert(`Auto-execution for recommendation ${id} ${selectedRecommendations.has(id) ? 'enabled' : 'disabled'}`)
  }

  // Execute selected recommendations
  const executeMutation = useMutation({
    mutationFn: async (recommendationIds: string[]) => {
      // Create change requests for selected recommendations
      const selectedRecs = recommendations.filter((r) => recommendationIds.includes(r.id))
      selectedRecs.forEach((rec) => {
        if (rec.type === 'maintenance') {
          createChange({
            type: 'maintenance',
            component: rec.component,
            parameter: rec.title,
            currentValue: 'N/A',
            newValue: rec.action,
            reason: rec.description,
            priority: rec.priority,
            autoExecute: rec.autoExecute && autoExecutionEnabled,
          })
        } else if (rec.type === 'optimization') {
          // Extract numeric values if available
          const match = rec.description.match(/(\d+\.?\d*)/g)
          const currentVal = match ? parseFloat(match[0]) : 0
          const newVal = match && match.length > 1 ? parseFloat(match[1]) : currentVal
          createChange({
            type: 'optimization',
            component: rec.component,
            parameter: rec.title,
            currentValue: currentVal,
            newValue: newVal,
            reason: rec.description,
            priority: rec.priority,
            autoExecute: rec.autoExecute && autoExecutionEnabled,
          })
        }
      })
      return { success: true, executed: recommendationIds.length }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['maintenance-alerts-pdm'] })
      setSelectedRecommendations(new Set())
    },
  })

  const handleExecute = () => {
    if (selectedRecommendations.size === 0) {
      alert('Please select at least one recommendation to execute')
      return
    }
    executeMutation.mutate(Array.from(selectedRecommendations))
  }

  // Handle individual recommendation apply
  const handleApplyRecommendation = (rec: MaintenanceRecommendation, autoExecute: boolean) => {
    if (rec.type === 'maintenance') {
      createChange({
        type: 'maintenance',
        component: rec.component,
        parameter: rec.title,
        currentValue: 'N/A',
        newValue: rec.action,
        reason: rec.description,
        priority: rec.priority,
        autoExecute: autoExecute && autoExecutionEnabled,
      })
    } else {
      createChange({
        type: 'optimization',
        component: rec.component,
        parameter: rec.title,
        currentValue: 0,
        newValue: 0,
        reason: rec.description,
        priority: rec.priority,
        autoExecute: autoExecute && autoExecutionEnabled,
      })
    }
  }

  const selectedCount = selectedRecommendations.size
  const criticalCount = recommendations.filter((r) => r.priority === 'critical').length
  const highCount = recommendations.filter((r) => r.priority === 'high').length

  return (
    <div className="space-y-6 text-slate-900 dark:text-slate-100">

      {/* Auto-Execution Toggle */}
      <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`p-3 rounded-xl ${autoExecutionEnabled ? 'bg-green-500' : 'bg-slate-200 dark:bg-slate-700'}`}>
              {autoExecutionEnabled ? (
                <Bot className="w-6 h-6 text-white" />
              ) : (
                <User className="w-6 h-6 text-slate-600 dark:text-slate-300" />
              )}
            </div>
            <div>
              <h3 className="text-lg font-semibold">AI Auto-Execution Mode</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400">
                {autoExecutionEnabled
                  ? 'AI can automatically implement maintenance and optimization changes'
                  : 'Manual approval required for all changes'}
              </p>
            </div>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={autoExecutionEnabled}
              onChange={(e) => {
                setAutoExecutionEnabled(e.target.checked)
                setChangeManagerAutoEnabled(e.target.checked)
              }}
              className="sr-only peer"
            />
            <div className="w-14 h-7 bg-slate-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-cyan-300 dark:peer-focus:ring-cyan-800 rounded-full peer dark:bg-slate-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[4px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-6 after:w-6 after:transition-all dark:border-slate-600 peer-checked:bg-cyan-500"></div>
          </label>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="rounded-2xl bg-gradient-to-br from-red-500 to-rose-600 p-6 text-white shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <AlertTriangle className="w-8 h-8" />
            <span className="text-3xl font-bold">{criticalCount}</span>
          </div>
          <div className="text-sm opacity-90">Critical Alerts</div>
          <div className="text-xs opacity-75 mt-1">Critical Alerts</div>
        </div>

        <div className="rounded-2xl bg-gradient-to-br from-amber-500 to-orange-600 p-6 text-white shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <AlertTriangle className="w-8 h-8" />
            <span className="text-3xl font-bold">{highCount}</span>
          </div>
          <div className="text-sm opacity-90">High Priority</div>
          <div className="text-xs opacity-75 mt-1">High Priority</div>
        </div>

        <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
          <div className="flex items-center gap-3 mb-4">
            <CheckCircle2 className="w-6 h-6 text-green-500" />
            <div>
              <div className="text-2xl font-bold text-slate-900 dark:text-white">{selectedCount}</div>
              <div className="text-sm text-slate-500 dark:text-slate-400">Selected</div>
              <div className="text-xs text-slate-400 dark:text-slate-500">Selected</div>
            </div>
          </div>
        </div>

        <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
          <div className="flex items-center gap-3 mb-4">
            <Shield className="w-6 h-6 text-cyan-500" />
            <div>
              <div className="text-2xl font-bold text-slate-900 dark:text-white">
                {effectiveRulData?.predicted_rul_hours ? `${effectiveRulData.predicted_rul_hours.toFixed(0)}h` : '--'}
              </div>
              {!rulData && (
                <span className="text-xs text-slate-400 dark:text-slate-500">(Demo Data)</span>
              )}
              <div className="text-sm text-slate-500 dark:text-slate-400">Predicted RUL</div>
              <div className="text-xs text-slate-400 dark:text-slate-500">Predicted RUL</div>
            </div>
          </div>
        </div>
      </div>

      {/* Recommendations List */}
      <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-semibold">Maintenance & Optimization Recommendations</h3>
          {selectedCount > 0 && (
            <button
              onClick={handleExecute}
              disabled={executeMutation.isPending}
              className="flex items-center gap-2 bg-cyan-500 hover:bg-cyan-400 text-white px-4 py-2 rounded-lg font-semibold disabled:opacity-50"
            >
              <Play className="w-4 h-4" />
              Execute Selected ({selectedCount})
            </button>
          )}
        </div>

        {recommendations.length === 0 ? (
          <div className="rounded-xl border border-emerald-500/50 bg-emerald-50 dark:bg-emerald-900/10 px-6 py-8 text-center">
            <CheckCircle2 className="w-12 h-12 text-emerald-500 mx-auto mb-3" />
            <div className="text-emerald-700 dark:text-emerald-300 font-semibold mb-1">All Systems Optimal</div>
            <div className="text-sm text-emerald-600 dark:text-emerald-400">
              No maintenance recommendations at this time. All systems are operating within optimal parameters.
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {recommendations.map((rec) => {
              const isSelected = selectedRecommendations.has(rec.id)
              const priorityColors = {
                critical: 'border-red-500/50 bg-red-500/10',
                high: 'border-amber-500/50 bg-amber-500/10',
                medium: 'border-blue-500/50 bg-blue-500/10',
                low: 'border-slate-500/50 bg-slate-500/10',
              }
              const priorityIcons = {
                critical: AlertTriangle,
                high: AlertTriangle,
                medium: Settings,
                low: CheckCircle2,
              }
              const PriorityIcon = priorityIcons[rec.priority]

              return (
                <div
                  key={rec.id}
                  className={`rounded-xl border ${priorityColors[rec.priority]} p-5 transition-all hover:shadow-md ${
                    isSelected ? 'ring-2 ring-cyan-500' : ''
                  }`}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-start gap-3 flex-1">
                      <button
                        onClick={() => toggleSelection(rec.id)}
                        className={`mt-1 w-5 h-5 rounded border-2 flex items-center justify-center transition-colors ${
                          isSelected
                            ? 'bg-cyan-500 border-cyan-500'
                            : 'border-slate-300 dark:border-slate-600'
                        }`}
                      >
                        {isSelected && <Check className="w-3 h-3 text-white" />}
                      </button>
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <PriorityIcon
                            className={`w-5 h-5 ${
                              rec.priority === 'critical'
                                ? 'text-red-500'
                                : rec.priority === 'high'
                                ? 'text-amber-500'
                                : rec.priority === 'medium'
                                ? 'text-blue-500'
                                : 'text-slate-500'
                            }`}
                          />
                          <div className="font-semibold text-slate-900 dark:text-white">{rec.title}</div>
                          <span
                            className={`px-2 py-1 rounded-full text-xs font-semibold ${
                              rec.priority === 'critical'
                                ? 'bg-red-500 text-white'
                                : rec.priority === 'high'
                                ? 'bg-amber-500 text-white'
                                : rec.priority === 'medium'
                                ? 'bg-blue-500 text-white'
                                : 'bg-slate-500 text-white'
                            }`}
                          >
                            {rec.priority}
                          </span>
                          {rec.type === 'optimization' && (
                            <span className="px-2 py-1 rounded-full text-xs font-semibold bg-green-500 text-white">
                              Optimization
                            </span>
                          )}
                        </div>
                        <div className="text-sm text-slate-600 dark:text-slate-300 mb-2">
                          <span className="font-semibold">Component:</span> {rec.component}
                        </div>
                        <div className="text-sm text-slate-700 dark:text-slate-200 mb-3">{rec.description}</div>
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-3">
                    <div className="rounded-lg bg-white dark:bg-slate-800 px-4 py-3">
                      <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Recommended Action</div>
                      <div className="text-sm font-semibold text-slate-900 dark:text-white">{rec.action}</div>
                    </div>
                    <div className="rounded-lg bg-white dark:bg-slate-800 px-4 py-3">
                      <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Expected Impact</div>
                      <div className="text-sm font-semibold text-slate-900 dark:text-white">{rec.expectedImpact}</div>
                    </div>
                    {rec.estimatedRUL && (
                      <div className="rounded-lg bg-white dark:bg-slate-800 px-4 py-3">
                        <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Estimated RUL</div>
                        <div className="text-sm font-semibold text-slate-900 dark:text-white">
                          {rec.estimatedRUL.toFixed(1)} hours
                        </div>
                      </div>
                    )}
                    <div className="rounded-lg bg-white dark:bg-slate-800 px-4 py-3">
                      <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Confidence</div>
                      <div className="text-sm font-semibold text-slate-900 dark:text-white">
                        {(rec.confidence * 100).toFixed(0)}%
                      </div>
                    </div>
                    {rec.estimatedCost && (
                      <div className="rounded-lg bg-white dark:bg-slate-800 px-4 py-3">
                        <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Estimated Cost</div>
                        <div className="text-sm font-semibold text-slate-900 dark:text-white">
                          ${rec.estimatedCost.toLocaleString()}
                        </div>
                      </div>
                    )}
                    {rec.estimatedDowntime && (
                      <div className="rounded-lg bg-white dark:bg-slate-800 px-4 py-3">
                        <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Estimated Downtime</div>
                        <div className="text-sm font-semibold text-slate-900 dark:text-white">
                          {rec.estimatedDowntime} hours
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="flex items-center justify-between pt-3 border-t border-slate-200 dark:border-slate-700">
                    <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                      <Clock className="w-4 h-4" />
                      {new Date().toLocaleString('en-US')}
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => handleApplyRecommendation(rec, false)}
                        className="flex items-center gap-1 px-3 py-1.5 bg-cyan-500 hover:bg-cyan-400 text-white rounded text-xs font-semibold"
                      >
                        <Play className="w-3 h-3" />
                        Apply Now
                      </button>
                      {autoExecutionEnabled && (
                        <button
                          onClick={() => handleApplyRecommendation(rec, true)}
                          className="flex items-center gap-1 px-3 py-1.5 bg-green-500 hover:bg-green-400 text-white rounded text-xs font-semibold"
                        >
                          <Bot className="w-3 h-3" />
                          Auto-Apply
                        </button>
                      )}
                      <label className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-300 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={rec.autoExecute}
                          onChange={() => toggleAutoExecute(rec.id)}
                          className="rounded"
                        />
                        Auto-execute
                      </label>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Change History */}
      {changes.length > 0 && (
        <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Clock className="w-5 h-5 text-cyan-500" />
            Change History
          </h3>
          <div className="space-y-2">
            {changes.slice(-10).reverse().map((change) => (
              <div key={change.id} className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-800 rounded-lg text-sm">
                <div className="flex items-center gap-2">
                  {change.requestedBy === 'ai' ? (
                    <Bot className="w-4 h-4 text-cyan-500" />
                  ) : (
                    <User className="w-4 h-4 text-blue-500" />
                  )}
                  <span className="text-slate-700 dark:text-slate-300">
                    {change.component} - {change.parameter}
                  </span>
                  <span className={`px-2 py-1 rounded text-xs ${
                    change.status === 'executed' ? 'bg-green-500 text-white' :
                    change.status === 'pending' ? 'bg-amber-500 text-white' :
                    change.status === 'failed' ? 'bg-red-500 text-white' :
                    'bg-slate-500 text-white'
                  }`}>
                    {change.status}
                  </span>
                </div>
                <div className="text-slate-500 dark:text-slate-400 text-xs">
                  {new Date(change.timestamp).toLocaleTimeString('en-US')}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Information Panel */}
      <div className="rounded-2xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700 p-6">
        <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2 flex items-center gap-2">
          <Shield className="w-4 h-4 text-cyan-500" />
          How PDM Works
        </h3>
        <ul className="text-sm text-slate-600 dark:text-slate-400 space-y-1 list-disc list-inside">
          <li>
            The PDM system provides preventive maintenance recommendations based on sensor data, RUL predictions, and anomaly detection
          </li>
          <li>
            You can select recommendations and execute them manually or automatically
          </li>
          <li>
            By enabling Auto-Execution Mode, the dashboard can automatically implement selected changes
          </li>
          <li>
            Optimization recommendations for drilling conditions are provided to improve efficiency and reduce costs
          </li>
        </ul>
      </div>
    </div>
  )
}

