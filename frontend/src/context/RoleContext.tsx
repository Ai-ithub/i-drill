import { createContext, useContext, useEffect, useMemo, useState, ReactNode } from 'react'

export type UserRole = 'viewer' | 'operator' | 'engineer' | 'maintenance'

const STORAGE_KEY = 'idrill-role'

interface RoleContextValue {
  role: UserRole
  setRole: (role: UserRole) => void
}

const RoleContext = createContext<RoleContextValue | undefined>(undefined)

const DEFAULT_ROLE: UserRole = 'viewer'

export function RoleProvider({ children }: { children: ReactNode }) {
  const [role, setRole] = useState<UserRole>(() => {
    if (typeof window === 'undefined') {
      return DEFAULT_ROLE
    }
    const stored = window.localStorage.getItem(STORAGE_KEY) as UserRole | null
    if (stored && ['viewer', 'operator', 'engineer', 'maintenance'].includes(stored)) {
      return stored
    }
    return DEFAULT_ROLE
  })

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, role)
  }, [role])

  const value = useMemo(() => ({ role, setRole }), [role])

  return <RoleContext.Provider value={value}>{children}</RoleContext.Provider>
}

export function useUserRole() {
  const ctx = useContext(RoleContext)
  if (!ctx) {
    throw new Error('useUserRole must be used within RoleProvider')
  }
  return ctx
}
