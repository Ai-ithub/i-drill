import { Link, useLocation } from 'react-router-dom'
import { LayoutDashboard, Activity, Clock, TrendingUp, Wrench } from 'lucide-react'

const menuItems = [
  { path: '/', label: 'داشبورد', icon: LayoutDashboard },
  { path: '/realtime', label: 'مانیتورینگ Real-time', icon: Activity },
  { path: '/historical', label: 'داده‌های تاریخی', icon: Clock },
  { path: '/predictions', label: 'پیش‌بینی‌ها', icon: TrendingUp },
  { path: '/maintenance', label: 'تعمیر و نگهداری', icon: Wrench },
]

export default function Sidebar() {
  const location = useLocation()

  return (
    <aside className="w-64 bg-slate-800 border-l border-slate-700 flex flex-col">
      <div className="p-6 border-b border-slate-700">
        <h1 className="text-2xl font-bold text-white">i-Drill</h1>
        <p className="text-sm text-slate-400 mt-1">Drilling Data Dashboard</p>
      </div>
      <nav className="flex-1 p-4 space-y-2">
        {menuItems.map((item) => {
          const Icon = item.icon
          const isActive = location.pathname === item.path

          return (
            <Link
              key={item.path}
              to={item.path}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                isActive
                  ? 'bg-primary-600 text-white'
                  : 'text-slate-300 hover:bg-slate-700 hover:text-white'
              }`}
            >
              <Icon className="w-5 h-5" />
              <span>{item.label}</span>
            </Link>
          )
        })}
      </nav>
    </aside>
  )
}

