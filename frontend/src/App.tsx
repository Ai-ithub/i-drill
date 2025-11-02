import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { QueryClientProvider } from 'react-query'
import { queryClient } from './utils/queryClient'
import Layout from './components/Layout/Layout'
import Dashboard from './pages/Dashboard/Dashboard'
import RealTimeMonitoring from './pages/RealTimeMonitoring/RealTimeMonitoring'
import HistoricalData from './pages/HistoricalData/HistoricalData'
import Predictions from './pages/Predictions/Predictions'
import Maintenance from './pages/Maintenance/Maintenance'

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/realtime" element={<RealTimeMonitoring />} />
            <Route path="/historical" element={<HistoricalData />} />
            <Route path="/predictions" element={<Predictions />} />
            <Route path="/maintenance" element={<Maintenance />} />
          </Routes>
        </Layout>
      </Router>
    </QueryClientProvider>
  )
}

export default App

