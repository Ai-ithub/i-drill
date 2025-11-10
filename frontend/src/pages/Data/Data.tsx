import { useState } from 'react'
import RealTimeDataTab from './tabs/RealTimeDataTab'
import HistoricalDataTab from './tabs/HistoricalDataTab'
import SyntheticDataTab from './tabs/SyntheticDataTab'
import { Activity, Clock, Sparkles } from 'lucide-react'

type TabType = 'realtime' | 'historical' | 'synthetic'

const tabs = [
  { id: 'realtime' as TabType, label: 'Real Time Data', icon: Activity },
  { id: 'historical' as TabType, label: 'Historical Data', icon: Clock },
  { id: 'synthetic' as TabType, label: 'Synthetic Data', icon: Sparkles },
]

export default function Data() {
  const [activeTab, setActiveTab] = useState<TabType>('realtime')

  return (
    <div className="space-y-6 text-slate-900 dark:text-slate-100">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold">Data Dashboard</h1>
        <p className="text-slate-500 dark:text-slate-300">
          Monitor and analyze drilling data in real-time, historical, and synthetic formats
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-slate-200 dark:border-slate-700">
        <nav className="flex space-x-1" aria-label="Tabs">
          {tabs.map((tab) => {
            const Icon = tab.icon
            const isActive = activeTab === tab.id
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`
                  flex items-center gap-2 px-6 py-3 text-sm font-medium transition-colors
                  ${
                    isActive
                      ? 'border-b-2 border-cyan-500 text-cyan-600 dark:text-cyan-400'
                      : 'text-slate-500 hover:text-slate-700 hover:border-slate-300 dark:text-slate-400 dark:hover:text-slate-300'
                  }
                `}
              >
                <Icon className="w-5 h-5" />
                {tab.label}
              </button>
            )
          })}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="mt-6">
        {activeTab === 'realtime' && <RealTimeDataTab />}
        {activeTab === 'historical' && <HistoricalDataTab />}
        {activeTab === 'synthetic' && <SyntheticDataTab />}
      </div>
    </div>
  )
}


