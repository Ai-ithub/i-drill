import { ReactNode, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import {
  Activity,
  BarChart3,
  Settings,
  Database,
  Bell,
  Gauge,
  TrendingUp,
  Wrench,
  FileText,
  Wifi,
  Server,
  Eye,
  Clock,
  ChevronDown,
  ChevronUp
} from 'lucide-react'

interface LayoutProps {
  children: ReactNode
}

interface MenuItem {
  name: string
  nameEn: string
  path?: string
  icon: any
  submenu?: { name: string; nameEn: string; path: string }[]
}

export default function NewLayout({ children }: LayoutProps) {
  const location = useLocation()
  const [expandedMenu, setExpandedMenu] = useState<string | null>(null)

  const menuItems: MenuItem[] = [
    {
      name: 'نمایش',
      nameEn: 'display',
      icon: Eye,
      submenu: [
        { name: 'Gauge', nameEn: 'Gauge', path: '/display/gauge' },
        { name: 'RPM', nameEn: 'RPM', path: '/display/rpm' },
        { name: 'SENSOR', nameEn: 'SENSOR', path: '/display/sensor' },
        { name: 'Control', nameEn: 'Control', path: '/display/control' }
      ]
    },
    {
      name: 'لیست بررسی',
      nameEn: 'check list',
      path: '/checklist',
      icon: FileText
    },
    {
      name: 'سیستم هشدار',
      nameEn: 'Alarm Systems',
      path: '/alarms',
      icon: Bell
    },
    {
      name: 'کنترل',
      nameEn: 'Control',
      path: '/control',
      icon: Settings
    },
    {
      name: 'تحلیل نمودار',
      nameEn: 'Graph_Analysis',
      path: '/graph-analysis',
      icon: BarChart3
    },
    {
      name: 'تحلیل 3بعدی',
      nameEn: '3D_Analysis_OP',
      path: '/3d-analysis',
      icon: TrendingUp
    },
    {
      name: 'عملیات زمان‌واقعی',
      nameEn: 'REAL_TIME_OP',
      path: '/',
      icon: Clock
    },
    {
      name: 'گزارش‌دهی',
      nameEn: 'Reporting',
      path: '/reporting',
      icon: FileText
    },
    {
      name: 'اتصال',
      nameEn: 'Connection',
      path: '/connection',
      icon: Wifi
    },
    {
      name: 'ثبت داده',
      nameEn: 'Data Loggers',
      path: '/data-loggers',
      icon: Database
    },
    {
      name: 'پایگاه داده',
      nameEn: 'Databases',
      path: '/databases',
      icon: Server
    },
    {
      name: 'نگهداری',
      nameEn: 'PDM',
      path: '/maintenance',
      icon: Wrench
    }
  ]

  const toggleMenu = (menuName: string) => {
    setExpandedMenu(expandedMenu === menuName ? null : menuName)
  }

  const isActive = (path: string) => {
    return location.pathname === path
  }

  return (
    <div className="flex h-screen bg-black overflow-hidden">
      {/* Main Content */}
      <div className="flex-1 overflow-auto">
        {children}
      </div>

      {/* Right Sidebar */}
      <div className="w-48 bg-gradient-to-b from-[#8BC34A] to-[#689F38] flex flex-col">
        {/* Logo/Title */}
        <div className="p-4 bg-gradient-to-r from-[#7CB342] to-[#8BC34A] border-b-2 border-[#689F38]">
          <div className="text-white font-bold text-lg tracking-wider">
            <span className="text-sm">TURBIN</span>
            <br />
            <span className="text-xs">Generator</span>
          </div>
        </div>

        {/* Menu Items */}
        <nav className="flex-1 overflow-y-auto py-2">
          {menuItems.map((item, index) => (
            <div key={index} className="mb-1">
              {item.submenu ? (
                <>
                  <button
                    onClick={() => toggleMenu(item.nameEn)}
                    className={`w-full flex items-center justify-between px-4 py-2.5 text-white text-sm hover:bg-[#689F38] transition-colors ${
                      expandedMenu === item.nameEn ? 'bg-[#689F38]' : ''
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <item.icon className="w-4 h-4" />
                      <span className="font-medium">{item.nameEn}</span>
                    </div>
                    {expandedMenu === item.nameEn ? (
                      <ChevronUp className="w-3 h-3" />
                    ) : (
                      <ChevronDown className="w-3 h-3" />
                    )}
                  </button>
                  {expandedMenu === item.nameEn && (
                    <div className="bg-[#7CB342]">
                      {item.submenu.map((subItem, subIndex) => (
                        <Link
                          key={subIndex}
                          to={subItem.path}
                          className={`block px-8 py-2 text-white text-xs hover:bg-[#689F38] transition-colors ${
                            isActive(subItem.path) ? 'bg-[#558B2F] font-semibold' : ''
                          }`}
                        >
                          {subItem.nameEn}
                        </Link>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                <Link
                  to={item.path!}
                  className={`flex items-center gap-2 px-4 py-2.5 text-white text-sm hover:bg-[#689F38] transition-colors ${
                    isActive(item.path!) ? 'bg-[#558B2F] border-r-4 border-white' : ''
                  }`}
                >
                  <item.icon className="w-4 h-4" />
                  <span className="font-medium">{item.nameEn}</span>
                </Link>
              )}
            </div>
          ))}
        </nav>

        {/* Version/Info */}
        <div className="p-3 bg-[#558B2F] border-t-2 border-[#689F38] text-center">
          <div className="text-white text-xs">
            <div className="font-semibold">i-Drill v1.0</div>
            <div className="text-[10px] text-green-100">Real-Time System</div>
          </div>
        </div>
      </div>
    </div>
  )
}

