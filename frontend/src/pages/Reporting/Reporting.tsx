import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { FileDown, TrendingUp, DollarSign, Shield, BarChart3, Download } from 'lucide-react'
import { Card, Button, Loading, ErrorDisplay } from '@/components/UI'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  AreaChart,
  Area,
} from 'recharts'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001'

export default function Reporting() {
  const [selectedReportType, setSelectedReportType] = useState<'performance' | 'cost' | 'safety'>('performance')
  const [selectedAnalysisType, setSelectedAnalysisType] = useState<'comparison' | 'trends'>('trends')
  const [rigId, setRigId] = useState('RIG_01')
  const [timeRange, setTimeRange] = useState({
    start: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString().slice(0, 16),
    end: new Date().toISOString().slice(0, 16),
  })

  // Generate Performance Report
  const { data: performanceReport, isLoading: performanceLoading } = useQuery({
    queryKey: ['performance-report', rigId, timeRange],
    queryFn: async () => {
      const response = await fetch(`${API_BASE_URL}/api/v1/reporting/performance`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rig_id: rigId,
          start_time: new Date(timeRange.start).toISOString(),
          end_time: new Date(timeRange.end).toISOString(),
        }),
        credentials: 'include',
      })
      if (!response.ok) throw new Error('Failed to generate performance report')
      return response.json()
    },
    enabled: selectedReportType === 'performance',
    retry: 1,
  })

  // Generate Cost Report
  const { data: costReport, isLoading: costLoading } = useQuery({
    queryKey: ['cost-report', rigId, timeRange],
    queryFn: async () => {
      const response = await fetch(`${API_BASE_URL}/api/v1/reporting/cost`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rig_id: rigId,
          start_time: new Date(timeRange.start).toISOString(),
          end_time: new Date(timeRange.end).toISOString(),
        }),
        credentials: 'include',
      })
      if (!response.ok) throw new Error('Failed to generate cost report')
      return response.json()
    },
    enabled: selectedReportType === 'cost',
    retry: 1,
  })

  // Generate Safety Report
  const { data: safetyReport, isLoading: safetyLoading } = useQuery({
    queryKey: ['safety-report', rigId, timeRange],
    queryFn: async () => {
      const response = await fetch(`${API_BASE_URL}/api/v1/reporting/safety`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rig_id: rigId,
          start_time: new Date(timeRange.start).toISOString(),
          end_time: new Date(timeRange.end).toISOString(),
        }),
        credentials: 'include',
      })
      if (!response.ok) throw new Error('Failed to generate safety report')
      return response.json()
    },
    enabled: selectedReportType === 'safety',
    retry: 1,
  })

  // Export mutations
  const exportExcelMutation = useMutation({
    mutationFn: async (reportType: string) => {
      const url = `${API_BASE_URL}/api/v1/reporting/export/excel?report_type=${reportType}&rig_id=${rigId}&start_time=${new Date(timeRange.start).toISOString()}&end_time=${new Date(timeRange.end).toISOString()}`
      const response = await fetch(url, { credentials: 'include' })
      if (!response.ok) throw new Error('Failed to export to Excel')
      const blob = await response.blob()
      const downloadUrl = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = downloadUrl
      a.download = `${reportType}_report_${rigId}_${new Date().toISOString().slice(0, 10)}.xlsx`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(downloadUrl)
      document.body.removeChild(a)
    },
  })

  const exportPDFMutation = useMutation({
    mutationFn: async (reportType: string) => {
      const url = `${API_BASE_URL}/api/v1/reporting/export/pdf?report_type=${reportType}&rig_id=${rigId}&start_time=${new Date(timeRange.start).toISOString()}&end_time=${new Date(timeRange.end).toISOString()}`
      const response = await fetch(url, { credentials: 'include' })
      if (!response.ok) throw new Error('Failed to export to PDF')
      const blob = await response.blob()
      const downloadUrl = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = downloadUrl
      a.download = `${reportType}_report_${rigId}_${new Date().toISOString().slice(0, 10)}.pdf`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(downloadUrl)
      document.body.removeChild(a)
    },
  })

  const currentReport = selectedReportType === 'performance' ? performanceReport : selectedReportType === 'cost' ? costReport : safetyReport
  const isLoading = performanceLoading || costLoading || safetyLoading

  return (
    <div className="space-y-6 text-slate-900 dark:text-slate-100">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold">Reporting & Analysis</h1>
          <p className="text-slate-500 dark:text-slate-300 mt-1">
            Real-time reports and historical analysis
          </p>
        </div>

        {/* Export Buttons */}
        {currentReport?.success && (
          <div className="flex gap-2">
            <Button
              variant="secondary"
              onClick={() => exportExcelMutation.mutate(selectedReportType)}
              disabled={exportExcelMutation.isPending}
              className="flex items-center gap-2"
            >
              <FileDown className="w-4 h-4" />
              Export Excel
            </Button>
            <Button
              variant="secondary"
              onClick={() => exportPDFMutation.mutate(selectedReportType)}
              disabled={exportPDFMutation.isPending}
              className="flex items-center gap-2"
            >
              <FileDown className="w-4 h-4" />
              Export PDF
            </Button>
          </div>
        )}
      </div>

      {/* Time Range Selector */}
      <Card padding="md">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">Rig ID</label>
            <input
              type="text"
              value={rigId}
              onChange={(e) => setRigId(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-800"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Start Time</label>
            <input
              type="datetime-local"
              value={timeRange.start}
              onChange={(e) => setTimeRange({ ...timeRange, start: e.target.value })}
              className="w-full px-3 py-2 border border-slate-300 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-800"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">End Time</label>
            <input
              type="datetime-local"
              value={timeRange.end}
              onChange={(e) => setTimeRange({ ...timeRange, end: e.target.value })}
              className="w-full px-3 py-2 border border-slate-300 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-800"
            />
          </div>
          <div className="flex items-end">
            <Button
              variant="primary"
              onClick={() => {
                // Trigger refetch
                window.location.reload()
              }}
              className="w-full"
            >
              Generate Report
            </Button>
          </div>
        </div>
      </Card>

      {/* Report Type Selector */}
      <div className="flex gap-2">
        <button
          onClick={() => setSelectedReportType('performance')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2 ${
            selectedReportType === 'performance'
              ? 'bg-cyan-500 text-white'
              : 'bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300'
          }`}
        >
          <TrendingUp className="w-4 h-4" />
          Performance
        </button>
        <button
          onClick={() => setSelectedReportType('cost')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2 ${
            selectedReportType === 'cost'
              ? 'bg-cyan-500 text-white'
              : 'bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300'
          }`}
        >
          <DollarSign className="w-4 h-4" />
          Cost
        </button>
        <button
          onClick={() => setSelectedReportType('safety')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2 ${
            selectedReportType === 'safety'
              ? 'bg-cyan-500 text-white'
              : 'bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300'
          }`}
        >
          <Shield className="w-4 h-4" />
          Safety
        </button>
      </div>

      {/* Report Content */}
      {isLoading ? (
        <Card padding="md">
          <Loading.SkeletonText lines={5} />
        </Card>
      ) : currentReport?.success ? (
        <ReportContent report={currentReport.report} reportType={selectedReportType} />
      ) : (
        <Card padding="md">
          <ErrorDisplay error={new Error(currentReport?.message || 'Failed to generate report')} />
        </Card>
      )}

      {/* Historical Analysis Section */}
      <HistoricalAnalysis rigId={rigId} />
    </div>
  )
}

function ReportContent({ report, reportType }: { report: any; reportType: string }) {
  if (!report) return null

  return (
    <div className="space-y-6">
      {/* Summary */}
      <Card padding="md">
        <Card.Header title="Report Summary" />
        <Card.Content>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {report.summary?.key_insights?.map((insight: string, index: number) => (
              <div key={index} className="p-3 bg-slate-100 dark:bg-slate-800 rounded-lg">
                <div className="text-sm text-slate-500 dark:text-slate-400 mb-1">Insight {index + 1}</div>
                <div className="font-semibold">{insight}</div>
              </div>
            ))}
          </div>
        </Card.Content>
      </Card>

      {/* Performance Report */}
      {reportType === 'performance' && (
        <>
          <Card padding="md">
            <Card.Header title="Current Metrics" />
            <Card.Content>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <div className="text-sm text-slate-500 dark:text-slate-400 mb-1">ROP Efficiency</div>
                  <div className="text-2xl font-bold">{report.current_metrics?.rop_efficiency?.toFixed(1) || 0}%</div>
                </div>
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <div className="text-sm text-slate-500 dark:text-slate-400 mb-1">Energy Efficiency</div>
                  <div className="text-2xl font-bold">
                    {report.current_metrics?.energy_efficiency?.toFixed(2) || 0} m/kWh
                  </div>
                </div>
                <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                  <div className="text-sm text-slate-500 dark:text-slate-400 mb-1">Drilling Efficiency Index</div>
                  <div className="text-2xl font-bold">
                    {report.current_metrics?.drilling_efficiency_index?.toFixed(1) || 0}
                  </div>
                </div>
              </div>
            </Card.Content>
          </Card>

          {report.statistics && (
            <Card padding="md">
              <Card.Header title="Statistics" />
              <Card.Content>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-semibold mb-2">ROP Statistics</h4>
                    <div className="space-y-1 text-sm">
                      <div>Min: {report.statistics.rop_stats?.min?.toFixed(2) || 0} m/h</div>
                      <div>Max: {report.statistics.rop_stats?.max?.toFixed(2) || 0} m/h</div>
                      <div>Average: {report.statistics.rop_stats?.average?.toFixed(2) || 0} m/h</div>
                      <div>Current: {report.statistics.rop_stats?.current?.toFixed(2) || 0} m/h</div>
                    </div>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">Depth Range</h4>
                    <div className="space-y-1 text-sm">
                      <div>Min: {report.statistics.depth_range?.min?.toFixed(1) || 0} m</div>
                      <div>Max: {report.statistics.depth_range?.max?.toFixed(1) || 0} m</div>
                      <div>Current: {report.statistics.depth_range?.current?.toFixed(1) || 0} m</div>
                    </div>
                  </div>
                </div>
              </Card.Content>
            </Card>
          )}
        </>
      )}

      {/* Cost Report */}
      {reportType === 'cost' && (
        <>
          <Card padding="md">
            <Card.Header title="Cost Breakdown" />
            <Card.Content>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 bg-slate-100 dark:bg-slate-800 rounded-lg">
                  <div className="text-sm text-slate-500 dark:text-slate-400 mb-1">Time Cost</div>
                  <div className="text-2xl font-bold">
                    ${report.cost_breakdown?.time_cost?.toFixed(2) || 0}
                  </div>
                </div>
                <div className="p-4 bg-slate-100 dark:bg-slate-800 rounded-lg">
                  <div className="text-sm text-slate-500 dark:text-slate-400 mb-1">Energy Cost</div>
                  <div className="text-2xl font-bold">
                    ${report.cost_breakdown?.energy_cost?.toFixed(2) || 0}
                  </div>
                </div>
                <div className="p-4 bg-slate-100 dark:bg-slate-800 rounded-lg">
                  <div className="text-sm text-slate-500 dark:text-slate-400 mb-1">Total Cost</div>
                  <div className="text-2xl font-bold">
                    ${report.cost_breakdown?.total_cost?.toFixed(2) || 0}
                  </div>
                </div>
                <div className="p-4 bg-slate-100 dark:bg-slate-800 rounded-lg">
                  <div className="text-sm text-slate-500 dark:text-slate-400 mb-1">Cost per Meter</div>
                  <div className="text-2xl font-bold">
                    ${report.cost_breakdown?.cost_per_meter?.toFixed(2) || 0}
                  </div>
                </div>
              </div>
            </Card.Content>
          </Card>

          {report.budget_status && (
            <Card padding="md">
              <Card.Header title="Budget Status" />
              <Card.Content>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between mb-2">
                      <span>Budget Utilization</span>
                      <span className="font-semibold">{report.budget_status.utilization_percent?.toFixed(1) || 0}%</span>
                    </div>
                    <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-4">
                      <div
                        className={`h-4 rounded-full ${
                          report.budget_status.utilization_percent < 80
                            ? 'bg-green-500'
                            : report.budget_status.utilization_percent < 95
                            ? 'bg-yellow-500'
                            : 'bg-red-500'
                        }`}
                        style={{ width: `${Math.min(report.budget_status.utilization_percent || 0, 100)}%` }}
                      />
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <div className="text-slate-500 dark:text-slate-400">Total Budget</div>
                      <div className="font-semibold">${report.budget_status.budget?.toFixed(2) || 0}</div>
                    </div>
                    <div>
                      <div className="text-slate-500 dark:text-slate-400">Remaining Budget</div>
                      <div className="font-semibold">${report.budget_status.remaining_budget?.toFixed(2) || 0}</div>
                    </div>
                  </div>
                </div>
              </Card.Content>
            </Card>
          )}
        </>
      )}

      {/* Safety Report */}
      {reportType === 'safety' && (
        <>
          <Card padding="md">
            <Card.Header title="Current Safety Status" />
            <Card.Content>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
                  <div className="text-sm text-slate-500 dark:text-slate-400 mb-1">Active Alerts</div>
                  <div className="text-2xl font-bold">{report.current_status?.active_alerts_count || 0}</div>
                </div>
                <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                  <div className="text-sm text-slate-500 dark:text-slate-400 mb-1">Critical Alerts</div>
                  <div className="text-2xl font-bold">{report.current_status?.critical_alerts || 0}</div>
                </div>
                <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                  <div className="text-sm text-slate-500 dark:text-slate-400 mb-1">High Alerts</div>
                  <div className="text-2xl font-bold">{report.current_status?.high_alerts || 0}</div>
                </div>
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <div className="text-sm text-slate-500 dark:text-slate-400 mb-1">Status</div>
                  <div className="text-2xl font-bold capitalize">{report.current_status?.status || 'unknown'}</div>
                </div>
              </div>
            </Card.Content>
          </Card>

          {report.incidents_summary && (
            <Card padding="md">
              <Card.Header title="Incidents Summary" />
              <Card.Content>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-semibold mb-2">Alerts by Severity</h4>
                    <div className="space-y-1 text-sm">
                      {Object.entries(report.incidents_summary.alerts_by_severity || {}).map(([severity, count]) => (
                        <div key={severity} className="flex justify-between">
                          <span className="capitalize">{severity}:</span>
                          <span className="font-semibold">{count as number}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">Alerts by Type</h4>
                    <div className="space-y-1 text-sm">
                      {Object.entries(report.incidents_summary.alerts_by_type || {}).map(([type, count]) => (
                        <div key={type} className="flex justify-between">
                          <span className="capitalize">{type.replace('_', ' ')}:</span>
                          <span className="font-semibold">{count as number}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </Card.Content>
            </Card>
          )}

          {report.recent_incidents && report.recent_incidents.length > 0 && (
            <Card padding="md">
              <Card.Header title="Recent Incidents" />
              <Card.Content>
                <div className="space-y-2">
                  {report.recent_incidents.slice(0, 5).map((incident: any, index: number) => (
                    <div key={index} className="p-3 bg-slate-100 dark:bg-slate-800 rounded-lg">
                      <div className="font-semibold">{incident.title || incident.alert_type}</div>
                      <div className="text-sm text-slate-500 dark:text-slate-400">{incident.message}</div>
                      <div className="text-xs text-slate-400 dark:text-slate-500 mt-1">
                        {new Date(incident.timestamp).toLocaleString()}
                      </div>
                    </div>
                  ))}
                </div>
              </Card.Content>
            </Card>
          )}
        </>
      )}
    </div>
  )
}

function HistoricalAnalysis({ rigId }: { rigId: string }) {
  const [analysisType, setAnalysisType] = useState<'trends' | 'comparison'>('trends')
  const [trendParams, setTrendParams] = useState({
    parameter: 'rop',
    timeBucket: 'hour',
    start: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().slice(0, 16),
    end: new Date().toISOString().slice(0, 16),
  })

  // Trend Analysis
  const { data: trendData, isLoading: trendLoading } = useQuery({
    queryKey: ['trend-analysis', rigId, trendParams],
    queryFn: async () => {
      const response = await fetch(`${API_BASE_URL}/api/v1/reporting/trends`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rig_id: rigId,
          start_time: new Date(trendParams.start).toISOString(),
          end_time: new Date(trendParams.end).toISOString(),
          parameter: trendParams.parameter,
          time_bucket: trendParams.timeBucket,
        }),
        credentials: 'include',
      })
      if (!response.ok) throw new Error('Failed to analyze trends')
      return response.json()
    },
    enabled: analysisType === 'trends',
    retry: 1,
  })

  const exportTrendsExcel = useMutation({
    mutationFn: async () => {
      const url = `${API_BASE_URL}/api/v1/reporting/export/trends/excel?rig_id=${rigId}&start_time=${new Date(trendParams.start).toISOString()}&end_time=${new Date(trendParams.end).toISOString()}&parameter=${trendParams.parameter}&time_bucket=${trendParams.timeBucket}`
      const response = await fetch(url, { credentials: 'include' })
      if (!response.ok) throw new Error('Failed to export trends')
      const blob = await response.blob()
      const downloadUrl = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = downloadUrl
      a.download = `trends_${trendParams.parameter}_${rigId}_${new Date().toISOString().slice(0, 10)}.xlsx`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(downloadUrl)
      document.body.removeChild(a)
    },
  })

  return (
    <div className="space-y-6">
      <Card padding="md">
        <Card.Header title="Historical Analysis" />
        <Card.Content>
          <div className="flex gap-2 mb-4">
            <button
              onClick={() => setAnalysisType('trends')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2 ${
                analysisType === 'trends'
                  ? 'bg-cyan-500 text-white'
                  : 'bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300'
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              Trend Analysis
            </button>
          </div>

          {analysisType === 'trends' && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Parameter</label>
                  <select
                    value={trendParams.parameter}
                    onChange={(e) => setTrendParams({ ...trendParams, parameter: e.target.value })}
                    className="w-full px-3 py-2 border border-slate-300 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-800"
                  >
                    <option value="rop">ROP</option>
                    <option value="wob">WOB</option>
                    <option value="rpm">RPM</option>
                    <option value="torque">Torque</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Time Bucket</label>
                  <select
                    value={trendParams.timeBucket}
                    onChange={(e) => setTrendParams({ ...trendParams, timeBucket: e.target.value })}
                    className="w-full px-3 py-2 border border-slate-300 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-800"
                  >
                    <option value="hour">Hour</option>
                    <option value="day">Day</option>
                    <option value="week">Week</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Start</label>
                  <input
                    type="datetime-local"
                    value={trendParams.start}
                    onChange={(e) => setTrendParams({ ...trendParams, start: e.target.value })}
                    className="w-full px-3 py-2 border border-slate-300 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-800"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">End</label>
                  <input
                    type="datetime-local"
                    value={trendParams.end}
                    onChange={(e) => setTrendParams({ ...trendParams, end: e.target.value })}
                    className="w-full px-3 py-2 border border-slate-300 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-800"
                  />
                </div>
              </div>

              {trendLoading ? (
                <Loading.SkeletonText lines={3} />
              ) : trendData?.success ? (
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <div>
                      <div className="text-sm text-slate-500 dark:text-slate-400">Trend Direction</div>
                      <div className="text-lg font-semibold capitalize">
                        {trendData.report?.trend_summary?.direction || 'unknown'}
                      </div>
                      <div className="text-sm text-slate-500 dark:text-slate-400">
                        Change: {trendData.report?.trend_summary?.change_percent?.toFixed(2) || 0}%
                      </div>
                    </div>
                    <Button
                      variant="secondary"
                      onClick={() => exportTrendsExcel.mutate()}
                      disabled={exportTrendsExcel.isPending}
                      className="flex items-center gap-2"
                    >
                      <Download className="w-4 h-4" />
                      Export Excel
                    </Button>
                  </div>

                  {trendData.report?.trend_data && trendData.report.trend_data.length > 0 && (
                    <ResponsiveContainer width="100%" height={400}>
                      <AreaChart data={trendData.report.trend_data}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-slate-300 dark:stroke-slate-700" />
                        <XAxis dataKey="timestamp" className="text-xs" />
                        <YAxis className="text-xs" />
                        <Tooltip />
                        <Area
                          type="monotone"
                          dataKey="average"
                          stroke="#3b82f6"
                          fill="#3b82f6"
                          fillOpacity={0.3}
                          name="Average"
                        />
                        <Area
                          type="monotone"
                          dataKey="min"
                          stroke="#10b981"
                          fill="#10b981"
                          fillOpacity={0.2}
                          name="Min"
                        />
                        <Area
                          type="monotone"
                          dataKey="max"
                          stroke="#ef4444"
                          fill="#ef4444"
                          fillOpacity={0.2}
                          name="Max"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  )}
                </div>
              ) : (
                <ErrorDisplay error={new Error(trendData?.message || 'Failed to analyze trends')} />
              )}
            </div>
          )}
        </Card.Content>
      </Card>
    </div>
  )
}

