import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { QueryClientProvider } from '@tanstack/react-query'
import { queryClient } from './utils/queryClient'
import { I18nProvider } from './i18n'
import NewLayout from './components/Layout/NewLayout'
import Dashboard from './pages/Dashboard/Dashboard'
import RealTimeMonitoring from './pages/RealTimeMonitoring/RealTimeMonitoring'
import HistoricalData from './pages/HistoricalData/HistoricalData'
import Predictions from './pages/Predictions/Predictions'
import Maintenance from './pages/Maintenance/Maintenance'
import GaugePage from './pages/Gauge/GaugePage'
import SensorPage from './pages/Sensor/SensorPage'
import ControlPage from './pages/Control/ControlPage'
import RPMPage from './pages/RPM/RPMPage'
import RLControl from './pages/RL/RLControl'
import DVRMonitoring from './pages/DVR/DVRMonitoring'
import Data from './pages/Data/Data'

function App() {
  return (
    <I18nProvider>
      <QueryClientProvider client={queryClient}>
        <Router>
          <NewLayout>
            <Routes>
            <Route path="/" element={<RealTimeMonitoring />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/data" element={<Data />} />
            <Route path="/realtime" element={<RealTimeMonitoring />} />
            <Route path="/historical" element={<HistoricalData />} />
            <Route path="/predictions" element={<Predictions />} />
            <Route path="/maintenance" element={<Maintenance />} />
            <Route path="/display/gauge" element={<GaugePage />} />
            <Route path="/display/sensor" element={<SensorPage />} />
            <Route path="/display/control" element={<ControlPage />} />
            <Route path="/display/rpm" element={<RPMPage />} />
            <Route path="/display/rl" element={<RLControl />} />
            <Route path="/dvr" element={<DVRMonitoring />} />
            </Routes>
          </NewLayout>
        </Router>
      </QueryClientProvider>
    </I18nProvider>
  )
}

export default App

