import { ReactNode, useEffect, useMemo, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import {
  Activity,
  BellRing,
  ChevronDown,
  ChevronUp,
  Cpu,
  Database,
  Eye,
  LayoutDashboard,
  LineChart,
  LogIn,
  Moon,
  Sun,
  Wrench,
} from 'lucide-react'
import { useThemeMode } from '@/context/ThemeContext'
import { useUserRole, UserRole } from '@/context/RoleContext'
import NotificationBadge from '@/components/Notifications/NotificationBadge'

interface LayoutProps {
  children: ReactNode
}

interface MenuItem {
  name: string
  path: string
  icon: any
  roles: UserRole[]
}

interface MenuSection {
  title: string
  items: MenuItem[]
}

const ROLE_LABEL: Record<UserRole, string> = {
  viewer: 'Viewer',
  operator: 'Operator',
  engineer: 'Engineer',
  maintenance: 'Maintenance',
}

const menuSections: MenuSection[] = [
  {
    title: 'Monitoring',
    items: [
      { name: 'Dashboard', path: '/dashboard', icon: LayoutDashboard, roles: ['viewer', 'operator', 'engineer', 'maintenance'] },
      { name: 'Real-time View', path: '/realtime', icon: Eye, roles: ['viewer', 'operator', 'engineer'] },
      { name: 'Historical Data', path: '/historical', icon: LineChart, roles: ['viewer', 'operator', 'engineer'] },
      { name: 'Predictions', path: '/predictions', icon: Activity, roles: ['engineer'] },
      { name: 'DVR Monitoring', path: '/dvr', icon: Database, roles: ['engineer', 'maintenance'] },
      { name: 'Maintenance', path: '/maintenance', icon: Wrench, roles: ['maintenance'] },
      { name: 'RL Agent', path: '/display/rl', icon: Cpu, roles: ['engineer'] },
    ],
  },
]

export default function NewLayout({ children }: LayoutProps) {
  const location = useLocation()
  const { mode, toggle } = useThemeMode()
  const { role, setRole } = useUserRole()
  const [expandedSection, setExpandedSection] = useState<string | null>('Monitoring')
  const [isMobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [isScrolled, setScrolled] = useState(false)

  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 12)
    handler()
    window.addEventListener('scroll', handler)
    return () => window.removeEventListener('scroll', handler)
  }, [])

  const filteredSections = useMemo(() => {
    return menuSections
      .map((section) => ({
        ...section,
        items: section.items.filter((item) => item.roles.includes(role)),
      }))
      .filter((section) => section.items.length > 0)
  }, [role])

  const currentRouteAllowed = useMemo(() => {
    const normalizedPath = location.pathname === '/' ? '/realtime' : location.pathname
    return menuSections.some((section) =>
      section.items.some((item) => item.path === normalizedPath && item.roles.includes(role))
    )
  }, [location.pathname, role])

  return (
    <div className="min-h-screen bg-slate-100 dark:bg-slate-950 transition-colors duration-300 flex flex-col">
      <header
        className={`sticky top-0 z-30 backdrop-blur bg-white/70 dark:bg-slate-950/70 border-b border-slate-200/60 dark:border-slate-800 transition-shadow ${
          isScrolled ? 'shadow-lg shadow-slate-900/10' : ''
        }`}
      >
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-3 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <button
              className="lg:hidden rounded-lg border border-slate-200 dark:border-slate-800 p-2"
              onClick={() => setMobileMenuOpen((prev) => !prev)}
              aria-label="toggle navigation"
            >
              <span className="sr-only">toggle menu</span>
              {isMobileMenuOpen ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
            </button>
            <div className="flex items-center gap-2">
              <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center text-white font-bold">
                ID
              </div>
              <div>
                <div className="text-sm font-semibold text-slate-900 dark:text-white">i-Drill Control Room</div>
                <div className="text-xs text-slate-500 dark:text-slate-400">Drilling Operations Control Room</div>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <NotificationBadge />

            <select
              value={role}
              onChange={(e) => setRole(e.target.value as UserRole)}
              className="hidden sm:block bg-slate-200 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-800 dark:text-slate-100"
            >
              {Object.entries(ROLE_LABEL).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>

            <button
              onClick={toggle}
              className="rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-2 text-slate-700 dark:text-slate-200"
              aria-label="toggle theme"
            >
              {mode === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>

            <button className="hidden md:flex items-center gap-2 rounded-lg border border-slate-200 dark:border-slate-700 px-3 py-2 text-sm text-slate-600 dark:text-slate-300">
              <LogIn className="w-4 h-4" />
              Login / Switch User
            </button>
          </div>
        </div>
      </header>

      <div className="flex-1 flex">
        <aside
          className={`fixed top-[64px] bottom-0 left-0 right-0 z-20 lg:z-auto lg:relative lg:w-64 w-full transform transition-transform duration-200 ${
            isMobileMenuOpen ? 'translate-x-0' : '-translate-x-full'
          } lg:translate-x-0 bg-white/95 dark:bg-slate-950/95 lg:bg-transparent lg:dark:bg-transparent border-r border-slate-200/70 dark:border-slate-800`}
        >
          <div className="h-full overflow-y-auto px-4 py-6 space-y-6">
            <div className="lg:hidden">
              <label className="block text-xs text-slate-500 dark:text-slate-400 mb-1">User Role</label>
              <select
                value={role}
                onChange={(e) => setRole(e.target.value as UserRole)}
                className="w-full bg-slate-200 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-800 dark:text-slate-100"
              >
                {Object.entries(ROLE_LABEL).map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </div>

            {filteredSections.map((section) => (
              <div key={section.title} className="space-y-2">
                <button
                  onClick={() => setExpandedSection((prev) => (prev === section.title ? null : section.title))}
                  className="flex w-full items-center justify-between text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400"
                >
                  <span>{section.title}</span>
                  {expandedSection === section.title ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>

                {expandedSection === section.title && (
                  <nav className="space-y-1">
                    {section.items.map((item) => {
                      const Icon = item.icon
                      const normalizedPath = location.pathname === '/' ? '/realtime' : location.pathname
                      const isActive = normalizedPath === item.path
                      return (
                        <Link
                          key={item.path}
                          to={item.path}
                          onClick={() => setMobileMenuOpen(false)}
                          className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-all ${
                            isActive
                              ? 'bg-cyan-500/10 text-cyan-600 dark:text-cyan-300 border border-cyan-500/40'
                              : 'text-slate-600 dark:text-slate-300 hover:bg-slate-200/70 dark:hover:bg-slate-800/70'
                          }`}
                        >
                          <Icon className="w-4 h-4" />
                          <span>{item.name}</span>
                        </Link>
                      )
                    })}
                  </nav>
                )}
              </div>
            ))}
          </div>
        </aside>

        <main className="flex-1 bg-slate-50/60 dark:bg-slate-900/50">
          <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
            {!currentRouteAllowed ? (
              <div className="rounded-2xl border border-amber-400/50 bg-amber-100/40 dark:bg-amber-400/10 px-6 py-10 text-center text-amber-700 dark:text-amber-200">
                <h2 className="text-xl font-bold mb-2">Access Denied</h2>
                <p className="text-sm">Current role ({ROLE_LABEL[role]}) does not have permission to view this section.</p>
              </div>
            ) : (
              children
            )}
          </div>
        </main>
      </div>
    </div>
  )
}

