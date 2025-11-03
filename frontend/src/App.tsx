import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { QueryClientProvider } from 'react-query'
import { queryClient } from './utils/queryClient'
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

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <NewLayout>
          <Routes>
            <Route path="/" element={<RealTimeMonitoring />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/realtime" element={<RealTimeMonitoring />} />
            <Route path="/historical" element={<HistoricalData />} />
            <Route path="/predictions" element={<Predictions />} />
            <Route path="/maintenance" element={<Maintenance />} />
            <Route path="/display/gauge" element={<GaugePage />} />
            <Route path="/display/sensor" element={<SensorPage />} />
            <Route path="/display/control" element={<ControlPage />} />
            <Route path="/display/rpm" element={<RPMPage />} />
          </Routes>
        </NewLayout>
      </Router>
    </QueryClientProvider>
  )
}

export default App

