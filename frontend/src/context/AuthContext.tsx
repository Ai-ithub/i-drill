import { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { authApi } from '@/services/api'

export interface User {
  id: number
  username: string
  email: string
  full_name?: string
  role: 'admin' | 'engineer' | 'operator' | 'data_scientist' | 'maintenance' | 'viewer'
  is_active: boolean
  created_at: string
}

interface AuthContextType {
  user: User | null
  token: string | null
  refreshToken: string | null
  isAuthenticated: boolean
  isLoading: boolean
  login: (username: string, password: string) => Promise<void>
  logout: () => Promise<void>
  refreshAccessToken: () => Promise<void>
  updateUser: (user: User) => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

const TOKEN_KEY = 'i_drill_access_token'
const REFRESH_TOKEN_KEY = 'i_drill_refresh_token'
const USER_KEY = 'i_drill_user'

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(null)
  const [refreshToken, setRefreshToken] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const navigate = useNavigate()

  // Load from localStorage on mount
  useEffect(() => {
    const storedToken = localStorage.getItem(TOKEN_KEY)
    const storedRefreshToken = localStorage.getItem(REFRESH_TOKEN_KEY)
    const storedUser = localStorage.getItem(USER_KEY)

    if (storedToken && storedUser) {
      try {
        setToken(storedToken)
        setRefreshToken(storedRefreshToken)
        setUser(JSON.parse(storedUser))
        // Verify token is still valid
        verifyToken()
      } catch (error) {
        console.error('Error loading auth state:', error)
        clearAuth()
      }
    } else {
      setIsLoading(false)
    }
  }, [])

  const verifyToken = async () => {
    try {
      const response = await authApi.me()
      if (response.data) {
        setUser(response.data)
        localStorage.setItem(USER_KEY, JSON.stringify(response.data))
        setIsLoading(false)
      }
    } catch (error: any) {
      console.error('Token verification failed:', error)
      // Try to refresh token
      if (refreshToken) {
        try {
          await refreshAccessToken()
        } catch {
          clearAuth()
        }
      } else {
        clearAuth()
      }
    }
  }

  const login = async (username: string, password: string) => {
    try {
      const formData = new FormData()
      formData.append('username', username)
      formData.append('password', password)

      const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8001/api/v1'
      const response = await fetch(`${apiBaseUrl}/auth/login`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Login failed')
      }

      const data = await response.json()
      
      setToken(data.access_token)
      setRefreshToken(data.refresh_token)
      localStorage.setItem(TOKEN_KEY, data.access_token)
      if (data.refresh_token) {
        localStorage.setItem(REFRESH_TOKEN_KEY, data.refresh_token)
      }

      // Get user info
      const userResponse = await authApi.me()
      setUser(userResponse.data)
      localStorage.setItem(USER_KEY, JSON.stringify(userResponse.data))

      navigate('/dashboard')
    } catch (error: any) {
      console.error('Login error:', error)
      throw error
    }
  }

  const logout = async () => {
    try {
      if (token) {
        // Call logout endpoint to blacklist token
        try {
          const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8001/api/v1'
          await fetch(`${apiBaseUrl}/auth/logout`, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${token}`,
            },
          })
        } catch (error) {
          console.error('Logout API call failed:', error)
        }
      }
    } catch (error) {
      console.error('Logout error:', error)
    } finally {
      clearAuth()
      navigate('/login')
    }
  }

  const refreshAccessToken = async () => {
    if (!refreshToken) {
      throw new Error('No refresh token available')
    }

    try {
      const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8001/api/v1'
      const response = await fetch(`${apiBaseUrl}/auth/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ refresh_token: refreshToken }),
      })

      if (!response.ok) {
        throw new Error('Token refresh failed')
      }

      const data = await response.json()
      setToken(data.access_token)
      setRefreshToken(data.refresh_token)
      localStorage.setItem(TOKEN_KEY, data.access_token)
      if (data.refresh_token) {
        localStorage.setItem(REFRESH_TOKEN_KEY, data.refresh_token)
      }

      return data.access_token
    } catch (error) {
      console.error('Token refresh error:', error)
      clearAuth()
      throw error
    }
  }

  const clearAuth = () => {
    setUser(null)
    setToken(null)
    setRefreshToken(null)
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(REFRESH_TOKEN_KEY)
    localStorage.removeItem(USER_KEY)
    setIsLoading(false)
  }

  const updateUser = (updatedUser: User) => {
    setUser(updatedUser)
    localStorage.setItem(USER_KEY, JSON.stringify(updatedUser))
  }

  const value: AuthContextType = {
    user,
    token,
    refreshToken,
    isAuthenticated: !!user && !!token,
    isLoading,
    login,
    logout,
    refreshAccessToken,
    updateUser,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

