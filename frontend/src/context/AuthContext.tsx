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

const USER_KEY = 'i_drill_user' // Only user data in localStorage, tokens are in httpOnly cookies

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(null) // Not used anymore, kept for backward compatibility
  const [refreshToken, setRefreshToken] = useState<string | null>(null) // Not used anymore, kept for backward compatibility
  const [isLoading, setIsLoading] = useState(true)
  const navigate = useNavigate()

  // Load user from localStorage on mount and verify authentication
  useEffect(() => {
    const storedUser = localStorage.getItem(USER_KEY)

    if (storedUser) {
      try {
        setUser(JSON.parse(storedUser))
        // Verify token is still valid (token is in httpOnly cookie)
        verifyToken()
      } catch (error) {
        console.error('Error loading auth state:', error)
        clearAuth()
      }
    } else {
      // Check if user is authenticated via cookie
      verifyToken()
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
      // Try to refresh token (refresh token is in httpOnly cookie)
      try {
        await refreshAccessToken()
      } catch {
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
        credentials: 'include', // Important: include cookies
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Login failed')
      }

      const data = await response.json()
      
      // Tokens are now in httpOnly cookies, no need to store in localStorage
      // Keep token state for backward compatibility (though not used)
      setToken(data.access_token || null)
      setRefreshToken(data.refresh_token || null)

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
      // Call logout endpoint to blacklist token and clear cookies
      try {
        const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8001/api/v1'
        await fetch(`${apiBaseUrl}/auth/logout`, {
          method: 'POST',
          credentials: 'include', // Important: include cookies
        })
      } catch (error) {
        console.error('Logout API call failed:', error)
      }
    } catch (error) {
      console.error('Logout error:', error)
    } finally {
      clearAuth()
      navigate('/login')
    }
  }

  const refreshAccessToken = async () => {
    try {
      const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8001/api/v1'
      const response = await fetch(`${apiBaseUrl}/auth/refresh`, {
        method: 'POST',
        credentials: 'include', // Important: include cookies (refresh token is in cookie)
        headers: {
          'Content-Type': 'application/json',
        },
      })

      if (!response.ok) {
        throw new Error('Token refresh failed')
      }

      const data = await response.json()
      
      // Tokens are now in httpOnly cookies, no need to store in localStorage
      // Keep token state for backward compatibility (though not used)
      setToken(data.access_token || null)
      setRefreshToken(data.refresh_token || null)

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
    localStorage.removeItem(USER_KEY)
    // Cookies are cleared by backend on logout
    setIsLoading(false)
  }

  const updateUser = (updatedUser: User) => {
    setUser(updatedUser)
    localStorage.setItem(USER_KEY, JSON.stringify(updatedUser))
  }

  const value: AuthContextType = {
    user,
    token, // Kept for backward compatibility, but tokens are in httpOnly cookies
    refreshToken, // Kept for backward compatibility, but tokens are in httpOnly cookies
    isAuthenticated: !!user, // Authentication is verified via httpOnly cookies
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

