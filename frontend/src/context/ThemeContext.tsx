import { createContext, useContext, useEffect, useMemo, useState, ReactNode } from 'react'

const STORAGE_KEY = 'idrill-theme'

type ThemeMode = 'dark' | 'light'

interface ThemeContextValue {
  mode: ThemeMode
  toggle: () => void
  setMode: (mode: ThemeMode) => void
}

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined)

function getInitialMode(): ThemeMode {
  if (typeof window === 'undefined') {
    return 'dark'
  }
  const stored = window.localStorage.getItem(STORAGE_KEY) as ThemeMode | null
  if (stored === 'dark' || stored === 'light') {
    return stored
  }
  const prefersDark = window.matchMedia?.('(prefers-color-scheme: dark)').matches
  return prefersDark ? 'dark' : 'light'
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<ThemeMode>(getInitialMode)

  useEffect(() => {
    const root = document.documentElement
    root.classList.remove(mode === 'dark' ? 'light' : 'dark')
    root.classList.add(mode)
    window.localStorage.setItem(STORAGE_KEY, mode)
  }, [mode])

  const value = useMemo(
    () => ({
      mode,
      toggle: () => setMode((prev) => (prev === 'dark' ? 'light' : 'dark')),
      setMode,
    }),
    [mode]
  )

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export function useThemeMode() {
  const ctx = useContext(ThemeContext)
  if (!ctx) {
    throw new Error('useThemeMode must be used within ThemeProvider')
  }
  return ctx
}
