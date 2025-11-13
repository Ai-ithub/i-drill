import { Wifi, WifiOff } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { healthApi } from '@/services/api'

export default function Header() {
  const { data: healthData, isLoading } = useQuery({
    queryKey: ['health'],
    queryFn: healthApi.check,
    refetchInterval: 30000, // Check every 30 seconds
    retry: 1,
  })

  const isHealthy = healthData?.data?.status === 'healthy'

  return (
    <header className="bg-slate-800 border-b border-slate-700 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h2 className="text-xl font-semibold text-white">i drill Dashboard</h2>
          <div className="flex items-center gap-2">
            {isLoading ? (
              <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse" />
            ) : isHealthy ? (
              <div className="flex items-center gap-2 text-green-400">
                <Wifi className="w-4 h-4" />
                <span className="text-sm">Connected</span>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-red-400">
                <WifiOff className="w-4 h-4" />
                <span className="text-sm">Disconnected</span>
              </div>
            )}
          </div>
        </div>
        <div className="text-sm text-slate-400">
          {new Date().toLocaleString('en-US')}
        </div>
      </div>
    </header>
  )
}

