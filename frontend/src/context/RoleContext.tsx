import { createContext, useContext, useEffect, useMemo, useState, ReactNode } from 'react'

export type UserRole = 'viewer' | 'operator' | 'engineer' | 'maintenance'

interface RoleContextValue {
  role: UserRole
  setRole: (role: UserRole) => void
}

const STORAGE_KEY = 'idrill-role'
const DEFAULT_ROLE: UserRole = 'viewer'

const RoleContext = createContext<RoleContextValue | undefined>(undefined)

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
    if (typeof window === 'undefined') {
      return
    }
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

