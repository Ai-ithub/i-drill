import { FormEvent, useState, useMemo } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { predictionsApi } from '@/services/api'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts'
import { TrendingUp, Activity, AlertTriangle, CheckCircle2, Loader2, Clock, Zap, BarChart3 } from 'lucide-react'

interface PredictionHistory {
  id: string
  timestamp: string
  rig_id: string
  model_type: string
  predicted_rul: number
  confidence: number
  component?: string
}

export default function Predictions() {
  const [rigId, setRigId] = useState('RIG_01')
  const [lookbackHours, setLookbackHours] = useState(24)
  const [modelType, setModelType] = useState('lstm')
  const [predictionHistory, setPredictionHistory] = useState<PredictionHistory[]>([])

  const predictionMutation = useMutation({
    mutationFn: (payload: { rig_id: string; lookback_hours: number; model_type: string }) =>
      predictionsApi
        .predictRULAuto(payload.rig_id, payload.lookback_hours, payload.model_type)
        .then((res) => res.data),
  })

  // Fetch analytics for statistics
  const { data: analyticsData } = useQuery({
    queryKey: ['analytics', rigId],
    queryFn: () => predictionsApi.predictRULAuto(rigId, lookbackHours, modelType).then((res) => res.data),
    enabled: false,
  })

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    predictionMutation.mutate(
      {
        rig_id: rigId.trim(),
        lookback_hours: lookbackHours,
        model_type: modelType,
      },
      {
        onSuccess: (data) => {
          if (data?.success && data?.predictions) {
            const newPredictions = data.predictions.map((pred: any, index: number) => ({
              id: `${Date.now()}-${index}`,
              timestamp: new Date().toISOString(),
              rig_id: rigId.trim(),
              model_type: modelType,
              predicted_rul: pred.predicted_rul || 0,
              confidence: pred.confidence || 0,
              component: pred.component || 'unknown',
            }))
            setPredictionHistory((prev) => [...newPredictions, ...prev].slice(0, 50))
          }
        },
      }
    )
  }

  const latestResult = predictionMutation.data

  // Calculate statistics
  const stats = useMemo(() => {
    if (!predictionHistory.length) return null
    const predictions = predictionHistory.filter((p) => p.model_type === modelType)
    if (!predictions.length) return null

    const avgRUL = predictions.reduce((sum, p) => sum + p.predicted_rul, 0) / predictions.length
    const avgConfidence = predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length
    const minRUL = Math.min(...predictions.map((p) => p.predicted_rul))
    const maxRUL = Math.max(...predictions.map((p) => p.predicted_rul))

    return {
      avgRUL,
      avgConfidence,
      minRUL,
      maxRUL,
      count: predictions.length,
    }
  }, [predictionHistory, modelType])

  // Prepare chart data
  const chartData = useMemo(() => {
    return predictionHistory
      .filter((p) => p.model_type === modelType)
      .slice(0, 20)
      .reverse()
      .map((pred) => ({
        time: new Date(pred.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
        timestamp: pred.timestamp,
        rul: pred.predicted_rul,
        confidence: pred.confidence * 100,
      }))
  }, [predictionHistory, modelType])

  // Model comparison data
  const modelComparison = useMemo(() => {
    const models = ['lstm', 'transformer', 'cnn_lstm']
    return models.map((model) => {
      const modelPreds = predictionHistory.filter((p) => p.model_type === model)
      if (!modelPreds.length) return { model, avgRUL: 0, avgConfidence: 0, count: 0 }
      const avgRUL = modelPreds.reduce((sum, p) => sum + p.predicted_rul, 0) / modelPreds.length
      const avgConfidence = modelPreds.reduce((sum, p) => sum + p.confidence, 0) / modelPreds.length
      return {
        model: model.toUpperCase(),
        avgRUL,
        avgConfidence: avgConfidence * 100,
        count: modelPreds.length,
      }
    })
  }, [predictionHistory])

  return (
    <div className="space-y-6 text-slate-900 dark:text-slate-100">
      <div>
        <h1 className="text-3xl font-bold mb-2">Remaining Useful Life (RUL) Prediction</h1>
        <p className="text-slate-500 dark:text-slate-300">
          Run machine learning models to estimate equipment remaining useful life based on historical data
        </p>
      </div>

      {/* Statistics Cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-6 shadow-sm">
            <div className="flex items-center gap-3 mb-2">
              <Activity className="w-5 h-5 text-cyan-500" />
              <div className="text-xs text-slate-500 dark:text-slate-400">Average RUL</div>
            </div>
            <div className="text-2xl font-bold text-slate-900 dark:text-white">
              {stats.avgRUL.toFixed(1)}h
            </div>
          </div>
          <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-6 shadow-sm">
            <div className="flex items-center gap-3 mb-2">
              <TrendingUp className="w-5 h-5 text-green-500" />
              <div className="text-xs text-slate-500 dark:text-slate-400">Avg Confidence</div>
            </div>
            <div className="text-2xl font-bold text-green-600 dark:text-green-400">
              {(stats.avgConfidence * 100).toFixed(1)}%
            </div>
          </div>
          <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-6 shadow-sm">
            <div className="flex items-center gap-3 mb-2">
              <AlertTriangle className="w-5 h-5 text-amber-500" />
              <div className="text-xs text-slate-500 dark:text-slate-400">Min RUL</div>
            </div>
            <div className="text-2xl font-bold text-amber-600 dark:text-amber-400">
              {stats.minRUL.toFixed(1)}h
            </div>
          </div>
          <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-6 shadow-sm">
            <div className="flex items-center gap-3 mb-2">
              <CheckCircle2 className="w-5 h-5 text-blue-500" />
              <div className="text-xs text-slate-500 dark:text-slate-400">Max RUL</div>
            </div>
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
              {stats.maxRUL.toFixed(1)}h
            </div>
          </div>
          <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-6 shadow-sm">
            <div className="flex items-center gap-3 mb-2">
              <BarChart3 className="w-5 h-5 text-purple-500" />
              <div className="text-xs text-slate-500 dark:text-slate-400">Total Predictions</div>
            </div>
            <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
              {stats.count}
            </div>
          </div>
        </div>
      )}

      <form
        onSubmit={handleSubmit}
        className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl p-6 space-y-4 shadow-sm"
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="space-y-2">
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">Rig ID</label>
            <input
              value={rigId}
              onChange={(e) => setRigId(e.target.value)}
              className="w-full rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 px-3 py-2 text-slate-900 dark:text-white focus:border-cyan-500 focus:outline-none"
              placeholder="RIG_01"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">Data Time Range (hours)</label>
            <input
              type="number"
              min={1}
              max={168}
              value={lookbackHours}
              onChange={(e) => setLookbackHours(Number(e.target.value))}
              className="w-full rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 px-3 py-2 text-slate-900 dark:text-white focus:border-cyan-500 focus:outline-none"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">Model</label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              className="w-full rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 px-3 py-2 text-slate-900 dark:text-white focus:border-cyan-500 focus:outline-none"
            >
              <option value="lstm">LSTM</option>
              <option value="transformer">Transformer</option>
              <option value="cnn_lstm">CNN-LSTM</option>
            </select>
          </div>
        </div>

        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <span className="text-sm text-slate-500 dark:text-slate-400">
            The selected model will be loaded from MLflow using the latest available version.
          </span>
          <button
            type="submit"
            className="w-full md:w-auto h-10 px-6 rounded-md bg-cyan-500 hover:bg-cyan-600 text-white font-semibold transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            disabled={predictionMutation.isLoading}
          >
            {predictionMutation.isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Predicting...
              </>
            ) : (
              <>
                <Zap className="w-4 h-4" />
                Run Prediction
              </>
            )}
          </button>
        </div>

        {predictionMutation.isError && (
          <div className="rounded-xl border border-red-500/40 bg-red-50 dark:bg-red-900/20 px-4 py-3 text-sm text-red-700 dark:text-red-300">
            Error running prediction: {String((predictionMutation.error as Error)?.message)}
          </div>
        )}
      </form>

      {/* Latest Prediction Results */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl p-6 space-y-4 shadow-sm">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-slate-900 dark:text-white">Latest Prediction Results</h2>
          {predictionMutation.isLoading && (
            <span className="text-sm text-slate-500 dark:text-slate-400 flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              Calculating...
            </span>
          )}
        </div>

        {!latestResult ? (
          <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 px-4 py-10 text-center text-slate-500 dark:text-slate-400">
            Run a new prediction to view the output.
          </div>
        ) : latestResult.success === false ? (
          <div className="rounded-xl border border-amber-500/40 bg-amber-50 dark:bg-amber-500/10 px-4 py-4 text-amber-700 dark:text-amber-300">
            {latestResult.message || 'Model was unable to generate a prediction.'}
          </div>
        ) : (
          <div className="space-y-4">
            {latestResult.predictions?.map((prediction: any) => (
              <div
                key={`${prediction.rig_id}-${prediction.timestamp}`}
                className="rounded-xl border border-cyan-500/30 bg-cyan-50 dark:bg-slate-800/40 p-6"
              >
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                  <div>
                    <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Rig</div>
                    <div className="text-lg font-semibold text-slate-900 dark:text-white">{prediction.rig_id}</div>
                  </div>
                  <div>
                    <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Component</div>
                    <div className="text-lg font-semibold text-slate-900 dark:text-white">
                      {prediction.component || 'unknown'}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Remaining Useful Life</div>
                    <div className="text-lg font-semibold text-cyan-600 dark:text-cyan-400">
                      {prediction.predicted_rul?.toFixed?.(1) ?? prediction.predicted_rul ?? '-'}{' '}
                      hours
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Confidence</div>
                    <div className="text-lg font-semibold text-green-600 dark:text-green-400">
                      {Math.round((prediction.confidence ?? 0) * 100)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Model</div>
                    <div className="text-lg font-semibold text-slate-900 dark:text-white">{prediction.model_used || modelType.toUpperCase()}</div>
                  </div>
                </div>
                <div className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-700 flex justify-between items-center text-xs text-slate-500 dark:text-slate-400">
                  <span className="flex items-center gap-2">
                    <Clock className="w-3 h-3" />
                    Updated:{' '}
                    {prediction.timestamp
                      ? new Date(prediction.timestamp).toLocaleString('en-US')
                      : new Date().toLocaleString('en-US')}
                  </span>
                  {prediction.recommendation && (
                    <span className="text-amber-600 dark:text-amber-400">{prediction.recommendation}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Prediction History Chart */}
      {chartData.length > 0 && (
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl p-6 shadow-sm">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">Prediction History</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.2)" />
                <XAxis 
                  dataKey="time" 
                  tick={{ fontSize: 12, fill: '#64748b' }}
                  stroke="#64748b"
                />
                <YAxis 
                  tick={{ fontSize: 12, fill: '#64748b' }}
                  stroke="#64748b"
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                  }}
                  formatter={(value: number, name: string) => {
                    if (name === 'rul') return [`${value.toFixed(1)} hours`, 'RUL']
                    if (name === 'confidence') return [`${value.toFixed(1)}%`, 'Confidence']
                    return [value, name]
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="rul"
                  stroke="#06b6d4"
                  strokeWidth={2}
                  dot={false}
                  name="RUL (hours)"
                />
                <Line
                  type="monotone"
                  dataKey="confidence"
                  stroke="#22c55e"
                  strokeWidth={2}
                  dot={false}
                  name="Confidence (%)"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Model Comparison */}
      {modelComparison.some((m) => m.count > 0) && (
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl p-6 shadow-sm">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">Model Comparison</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelComparison.filter((m) => m.count > 0)} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.2)" />
                <XAxis 
                  dataKey="model" 
                  tick={{ fontSize: 12, fill: '#64748b' }}
                  stroke="#64748b"
                />
                <YAxis 
                  tick={{ fontSize: 12, fill: '#64748b' }}
                  stroke="#64748b"
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                  }}
                  formatter={(value: number, name: string) => {
                    if (name === 'avgRUL') return [`${value.toFixed(1)} hours`, 'Average RUL']
                    if (name === 'avgConfidence') return [`${value.toFixed(1)}%`, 'Average Confidence']
                    return [value, name]
                  }}
                />
                <Legend />
                <Bar dataKey="avgRUL" fill="#06b6d4" name="Average RUL (hours)" />
                <Bar dataKey="avgConfidence" fill="#22c55e" name="Average Confidence (%)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  )
}

