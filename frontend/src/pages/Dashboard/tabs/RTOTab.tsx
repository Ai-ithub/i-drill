import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { sensorDataApi } from '@/services/api'
import { TrendingUp, Zap, Gauge, Droplets, Settings, AlertCircle, CheckCircle2, ArrowUp, ArrowDown, Target, Play, Bot, User } from 'lucide-react'
import { useChangeManager } from '@/components/ChangeManagement/ChangeManager'

interface OptimizationRecommendation {
  parameter: string
  currentValue: number
  recommendedValue: number
  unit: string
  priority: 'high' | 'medium' | 'low'
  impact: string
  reason: string
  expectedImprovement: string
}

export default function RTOTab() {
  const rigId = 'RIG_01'
  const [autoExecutionEnabled, setAutoExecutionEnabled] = useState(false)
  const { createChange, changes, autoExecutionEnabled: changeManagerAutoEnabled, setAutoExecutionEnabled: setChangeManagerAutoEnabled } = useChangeManager(rigId)

  const { data: analyticsData, isLoading: analyticsLoading } = useQuery({
    queryKey: ['analytics'],
    queryFn: () => sensorDataApi.getAnalytics(rigId).then((res) => res.data.summary),
    refetchInterval: 30000, // Refresh every 30 seconds for real-time optimization
  })

  const { data: realtimeData, isLoading: realtimeLoading } = useQuery({
    queryKey: ['realtime-optimization'],
    queryFn: () => sensorDataApi.getRealtime(rigId, 1).then((res) => res.data.data?.[0]),
    refetchInterval: 10000, // Refresh every 10 seconds
  })

  const isLoading = analyticsLoading || realtimeLoading

  // Handle apply recommendation
  const handleApplyRecommendation = (rec: OptimizationRecommendation, autoExecute: boolean) => {
    createChange({
      type: 'optimization',
      component: 'Drilling Parameters',
      parameter: rec.parameter,
      currentValue: rec.currentValue,
      newValue: rec.recommendedValue,
      unit: rec.unit,
      reason: rec.reason,
      priority: rec.priority === 'high' ? 'high' : rec.priority === 'medium' ? 'medium' : 'low',
      autoExecute: autoExecute && autoExecutionEnabled,
    })
  }

  // Calculate optimization recommendations based on current data
  const recommendations = useMemo<OptimizationRecommendation[]>(() => {
    if (!realtimeData || !analyticsData) return []

    const currentROP = realtimeData.rop || analyticsData.average_rop || 0
    const currentWOB = realtimeData.wob || 0
    const currentRPM = realtimeData.rpm || 0
    const currentTorque = realtimeData.torque || 0
    const currentMudFlow = realtimeData.mud_flow || 0
    const currentMudPressure = realtimeData.mud_pressure || 0
    const currentDepth = realtimeData.depth || analyticsData.current_depth || 0

    const recs: OptimizationRecommendation[] = []

    // Mud Weight Optimization
    const optimalMudWeight = currentDepth * 0.052 + 0.5 // Simplified calculation
    const currentMudWeight = currentMudPressure / (currentDepth * 0.052) || 9.5
    if (Math.abs(currentMudWeight - optimalMudWeight) > 0.2) {
      recs.push({
        parameter: 'Mud Weight',
        currentValue: currentMudWeight,
        recommendedValue: optimalMudWeight,
        unit: 'ppg',
        priority: 'high',
        impact: 'ROP +15-25%',
        reason: currentMudWeight < optimalMudWeight 
          ? 'Mud weight is too low, increasing risk of wellbore instability'
          : 'Mud weight is too high, causing excessive pressure and reducing ROP',
        expectedImprovement: `Expected ROP increase: ${Math.abs(currentMudWeight - optimalMudWeight) * 5}%`,
      })
    }

    // Torque Optimization
    const optimalTorque = currentWOB * 0.15 // Simplified optimal torque calculation
    if (currentTorque > 0 && Math.abs(currentTorque - optimalTorque) / optimalTorque > 0.15) {
      recs.push({
        parameter: 'Torque',
        currentValue: currentTorque,
        recommendedValue: optimalTorque,
        unit: 'ft-lbs',
        priority: currentTorque > optimalTorque * 1.2 ? 'high' : 'medium',
        impact: 'Efficiency +10-20%',
        reason: currentTorque > optimalTorque
          ? 'Torque is too high, indicating potential bit balling or excessive friction'
          : 'Torque is below optimal, suggesting underutilized drilling capacity',
        expectedImprovement: `Expected efficiency gain: ${Math.abs(currentTorque - optimalTorque) / optimalTorque * 100}%`,
      })
    }

    // RPM Optimization
    const optimalRPM = currentDepth < 5000 ? 180 : currentDepth < 10000 ? 150 : 120
    if (Math.abs(currentRPM - optimalRPM) > 20) {
      recs.push({
        parameter: 'Rotary Speed (RPM)',
        currentValue: currentRPM,
        recommendedValue: optimalRPM,
        unit: 'rpm',
        priority: 'medium',
        impact: 'ROP +8-15%',
        reason: currentRPM > optimalRPM
          ? 'RPM is too high for current depth, causing excessive wear'
          : 'RPM can be increased to improve penetration rate',
        expectedImprovement: `Expected ROP improvement: ${Math.abs(currentRPM - optimalRPM) * 0.3}%`,
      })
    }

    // WOB Optimization
    const optimalWOB = currentDepth * 2.5 // Simplified calculation
    if (currentWOB > 0 && Math.abs(currentWOB - optimalWOB) / optimalWOB > 0.2) {
      recs.push({
        parameter: 'Weight on Bit (WOB)',
        currentValue: currentWOB,
        recommendedValue: optimalWOB,
        unit: 'lbs',
        priority: 'medium',
        impact: 'ROP +12-18%',
        reason: currentWOB < optimalWOB
          ? 'WOB is below optimal, reducing penetration efficiency'
          : 'WOB is too high, risking bit damage and excessive torque',
        expectedImprovement: `Expected ROP increase: ${Math.abs(currentWOB - optimalWOB) / optimalWOB * 100}%`,
      })
    }

    // Mud Flow Rate Optimization
    const optimalMudFlow = currentDepth < 5000 ? 600 : currentDepth < 10000 ? 700 : 800
    if (Math.abs(currentMudFlow - optimalMudFlow) > 50) {
      recs.push({
        parameter: 'Mud Flow Rate',
        currentValue: currentMudFlow,
        recommendedValue: optimalMudFlow,
        unit: 'gpm',
        priority: 'low',
        impact: 'Cuttings Removal +10%',
        reason: currentMudFlow < optimalMudFlow
          ? 'Flow rate is below optimal, affecting cuttings removal'
          : 'Flow rate is above optimal, increasing pump wear',
        expectedImprovement: 'Improved hole cleaning and reduced risk of stuck pipe',
      })
    }

    return recs.sort((a, b) => {
      const priorityOrder = { high: 3, medium: 2, low: 1 }
      return priorityOrder[b.priority] - priorityOrder[a.priority]
    })
  }, [realtimeData, analyticsData])

  // Calculate overall optimization score
  const optimizationScore = useMemo(() => {
    if (recommendations.length === 0) return 95
    const highPriorityCount = recommendations.filter(r => r.priority === 'high').length
    const mediumPriorityCount = recommendations.filter(r => r.priority === 'medium').length
    return Math.max(0, 100 - (highPriorityCount * 15 + mediumPriorityCount * 5))
  }, [recommendations])

  // Calculate potential improvements
  const potentialImprovements = useMemo(() => {
    if (!analyticsData || !realtimeData) return null

    const currentROP = realtimeData.rop || analyticsData.average_rop || 0
    const ropImprovement = recommendations
      .filter(r => r.impact.includes('ROP'))
      .reduce((sum, r) => {
        const match = r.impact.match(/ROP \+(\d+)-(\d+)%/)
        if (match) {
          return sum + parseFloat(match[1])
        }
        return sum
      }, 0)

    const efficiencyImprovement = recommendations
      .filter(r => r.impact.includes('Efficiency'))
      .reduce((sum, r) => {
        const match = r.impact.match(/Efficiency \+(\d+)-(\d+)%/)
        if (match) {
          return sum + parseFloat(match[1])
        }
        return sum
      }, 0)

    return {
      ropImprovement: ropImprovement > 0 ? ropImprovement : 0,
      efficiencyImprovement: efficiencyImprovement > 0 ? efficiencyImprovement : 0,
      estimatedCostReduction: (ropImprovement + efficiencyImprovement) * 0.5, // Simplified calculation
    }
  }, [recommendations, analyticsData, realtimeData])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-slate-400">Loading optimization data...</div>
      </div>
    )
  }

  return (
    <div className="space-y-6 text-slate-900 dark:text-slate-100">
      {/* Header */}
      <div className="space-y-2">
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <Target className="w-6 h-6 text-cyan-500" />
          Real-Time Optimization (RTO)
        </h2>
        <p className="text-slate-500 dark:text-slate-300">
          Real-time optimization recommendations to improve rate of penetration (ROP), reduce costs, and increase drilling efficiency
        </p>
      </div>

      {/* Auto-Execution Control */}
      <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`p-3 rounded-xl ${autoExecutionEnabled ? 'bg-green-500' : 'bg-slate-200 dark:bg-slate-700'}`}>
              {autoExecutionEnabled ? <Bot className="w-6 h-6 text-white" /> : <User className="w-6 h-6 text-slate-600 dark:text-slate-300" />}
            </div>
            <div>
              <h3 className="text-lg font-semibold">AI Auto-Execution Mode</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400">
                {autoExecutionEnabled
                  ? 'AI can automatically apply optimization changes to maintain optimal drilling conditions'
                  : 'Manual approval required for all optimization changes'}
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

      {/* Optimization Score Card */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="rounded-2xl bg-gradient-to-br from-cyan-500 to-blue-600 p-6 text-white shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <Target className="w-8 h-8" />
            <span className="text-3xl font-bold">{optimizationScore}%</span>
          </div>
          <div className="text-sm opacity-90">Optimization Score</div>
          <div className="text-xs opacity-75 mt-1">Optimization Score</div>
        </div>

        {potentialImprovements && (
          <>
            <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
              <div className="flex items-center gap-3 mb-4">
                <TrendingUp className="w-6 h-6 text-green-500" />
                <div>
                  <div className="text-2xl font-bold text-slate-900 dark:text-white">
                    +{potentialImprovements.ropImprovement.toFixed(1)}%
                  </div>
                  <div className="text-sm text-slate-500 dark:text-slate-400">Potential ROP Improvement</div>
                  <div className="text-xs text-slate-400 dark:text-slate-500">Potential ROP Improvement</div>
                </div>
              </div>
            </div>

            <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
              <div className="flex items-center gap-3 mb-4">
                <Zap className="w-6 h-6 text-yellow-500" />
                <div>
                  <div className="text-2xl font-bold text-slate-900 dark:text-white">
                    +{potentialImprovements.efficiencyImprovement.toFixed(1)}%
                  </div>
                  <div className="text-sm text-slate-500 dark:text-slate-400">Efficiency Gain</div>
                  <div className="text-xs text-slate-400 dark:text-slate-500">Efficiency Increase</div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Current Parameters */}
      {realtimeData && (
        <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Gauge className="w-5 h-5 text-cyan-500" />
            Current Drilling Parameters
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-50 dark:bg-slate-800/40">
              <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Depth</div>
              <div className="text-lg font-semibold text-slate-900 dark:text-white">
                {(realtimeData.depth || 0).toFixed(1)} ft
              </div>
            </div>
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-50 dark:bg-slate-800/40">
              <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">WOB</div>
              <div className="text-lg font-semibold text-slate-900 dark:text-white">
                {(realtimeData.wob || 0).toFixed(0)} lbs
              </div>
            </div>
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-50 dark:bg-slate-800/40">
              <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">RPM</div>
              <div className="text-lg font-semibold text-slate-900 dark:text-white">
                {(realtimeData.rpm || 0).toFixed(0)}
              </div>
            </div>
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-50 dark:bg-slate-800/40">
              <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Torque</div>
              <div className="text-lg font-semibold text-slate-900 dark:text-white">
                {(realtimeData.torque || 0).toFixed(0)} ft-lbs
              </div>
            </div>
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-50 dark:bg-slate-800/40">
              <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">ROP</div>
              <div className="text-lg font-semibold text-slate-900 dark:text-white">
                {(realtimeData.rop || 0).toFixed(1)} ft/hr
              </div>
            </div>
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-50 dark:bg-slate-800/40">
              <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Mud Flow</div>
              <div className="text-lg font-semibold text-slate-900 dark:text-white">
                {(realtimeData.mud_flow || 0).toFixed(0)} gpm
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Optimization Recommendations */}
      <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Settings className="w-5 h-5 text-cyan-500" />
          Optimization Recommendations
          <span className="text-sm font-normal text-slate-500 dark:text-slate-400">
            ({recommendations.length} recommendations)
          </span>
        </h3>

        {recommendations.length === 0 ? (
          <div className="rounded-xl border border-emerald-500/50 bg-emerald-500/10 px-6 py-8 text-center">
            <CheckCircle2 className="w-12 h-12 text-emerald-500 mx-auto mb-3" />
            <div className="text-emerald-700 dark:text-emerald-300 font-semibold mb-1">
              Optimal Configuration
            </div>
            <div className="text-sm text-emerald-600 dark:text-emerald-400">
              All parameters are within optimal range. Excellent performance!
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {recommendations.map((rec, index) => {
              const isIncrease = rec.recommendedValue > rec.currentValue
              const changePercent = ((rec.recommendedValue - rec.currentValue) / rec.currentValue) * 100
              const priorityColors = {
                high: 'border-red-500/50 bg-red-500/10',
                medium: 'border-amber-500/50 bg-amber-500/10',
                low: 'border-blue-500/50 bg-blue-500/10',
              }
              const priorityIcons = {
                high: AlertCircle,
                medium: Settings,
                low: CheckCircle2,
              }
              const PriorityIcon = priorityIcons[rec.priority]

              return (
                <div
                  key={index}
                  className={`rounded-xl border ${priorityColors[rec.priority]} p-5 transition-all hover:shadow-md`}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <PriorityIcon
                        className={`w-5 h-5 ${
                          rec.priority === 'high'
                            ? 'text-red-500'
                            : rec.priority === 'medium'
                            ? 'text-amber-500'
                            : 'text-blue-500'
                        }`}
                      />
                      <div>
                        <div className="font-semibold text-slate-900 dark:text-white">{rec.parameter}</div>
                        <div className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                          Priority: {rec.priority.toUpperCase()}
                        </div>
                      </div>
                    </div>
                    <span
                      className={`px-3 py-1 rounded-full text-xs font-semibold ${
                        rec.priority === 'high'
                          ? 'bg-red-500 text-white'
                          : rec.priority === 'medium'
                          ? 'bg-amber-500 text-white'
                          : 'bg-blue-500 text-white'
                      }`}
                    >
                      {rec.priority}
                    </span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-3">
                    <div className="rounded-lg bg-white dark:bg-slate-800 px-4 py-3">
                      <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Current Value</div>
                      <div className="text-lg font-semibold text-slate-900 dark:text-white">
                        {rec.currentValue.toFixed(2)} {rec.unit}
                      </div>
                    </div>
                    <div className="rounded-lg bg-white dark:bg-slate-800 px-4 py-3">
                      <div className="text-xs text-slate-500 dark:text-slate-400 mb-1 flex items-center gap-1">
                        Recommended Value
                        {isIncrease ? (
                          <ArrowUp className="w-4 h-4 text-green-500" />
                        ) : (
                          <ArrowDown className="w-4 h-4 text-red-500" />
                        )}
                      </div>
                      <div className="text-lg font-semibold text-slate-900 dark:text-white">
                        {rec.recommendedValue.toFixed(2)} {rec.unit}
                        <span
                          className={`ml-2 text-sm ${
                            isIncrease ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                          }`}
                        >
                          ({isIncrease ? '+' : ''}{changePercent.toFixed(1)}%)
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="rounded-lg bg-white dark:bg-slate-800 px-4 py-3 mb-3">
                    <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Expected Impact</div>
                    <div className="text-sm font-semibold text-slate-900 dark:text-white">{rec.impact}</div>
                  </div>

                  <div className="space-y-2">
                    <div className="text-sm text-slate-700 dark:text-slate-300">
                      <span className="font-semibold">Reason:</span> {rec.reason}
                    </div>
                    <div className="text-sm text-slate-600 dark:text-slate-400">
                      <span className="font-semibold">Expected Improvement:</span> {rec.expectedImprovement}
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex items-center gap-2 pt-3 border-t border-slate-200 dark:border-slate-700 mt-3">
                    <button
                      onClick={() => handleApplyRecommendation(rec, false)}
                      className="flex items-center gap-2 px-4 py-2 bg-cyan-500 hover:bg-cyan-400 text-white rounded-lg text-sm font-semibold"
                    >
                      <Play className="w-4 h-4" />
                      Apply Now
                    </button>
                    {autoExecutionEnabled && (
                      <button
                        onClick={() => handleApplyRecommendation(rec, true)}
                        className="flex items-center gap-2 px-4 py-2 bg-green-500 hover:bg-green-400 text-white rounded-lg text-sm font-semibold"
                      >
                        <Bot className="w-4 h-4" />
                        Auto-Apply
                      </button>
                    )}
                    <button
                      onClick={() => {
                        // Manual entry
                        const manualValue = prompt(`Enter new value for ${rec.parameter} (${rec.unit}):`, rec.recommendedValue.toString())
                        if (manualValue) {
                          const numValue = parseFloat(manualValue)
                          if (!isNaN(numValue)) {
                            createChange({
                              type: 'optimization',
                              component: 'Drilling Parameters',
                              parameter: rec.parameter,
                              currentValue: rec.currentValue,
                              newValue: numValue,
                              unit: rec.unit,
                              reason: `Manual adjustment: ${rec.reason}`,
                              priority: rec.priority === 'high' ? 'high' : rec.priority === 'medium' ? 'medium' : 'low',
                              autoExecute: false,
                            })
                          }
                        }
                      }}
                      className="flex items-center gap-2 px-4 py-2 bg-slate-500 hover:bg-slate-400 text-white rounded-lg text-sm font-semibold"
                    >
                      <User className="w-4 h-4" />
                      Manual Entry
                    </button>
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
            <Settings className="w-5 h-5 text-cyan-500" />
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
                    {change.parameter}: {change.currentValue} {change.unit} → {change.newValue} {change.unit}
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
                  {new Date(change.timestamp).toLocaleTimeString()}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Additional Information */}
      <div className="rounded-2xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700 p-6">
        <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2 flex items-center gap-2">
          <Droplets className="w-4 h-4 text-cyan-500" />
          Optimization Notes
        </h3>
        <ul className="text-sm text-slate-600 dark:text-slate-400 space-y-1 list-disc list-inside">
          <li>
            These recommendations are calculated based on real-time data and optimization algorithms
          </li>
          <li>
            Changes should be applied with caution and under operator supervision
          </li>
          <li>
            High-priority recommendations should be reviewed as soon as possible
          </li>
          <li>
            Predicted improvements are calculated based on statistical and empirical models
          </li>
        </ul>
      </div>
    </div>
  )
}

