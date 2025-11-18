import { useState, useEffect, useRef, useCallback } from 'react'

export interface WebSocketMessage {
  message_type: string
  data: any
  timestamp: string
}

export interface UseWebSocketReturn {
  data: WebSocketMessage | null
  isConnected: boolean
  error: Error | null
  sendMessage: (message: any) => void
  reconnect: () => void
}

export interface UseWebSocketOptions {
  /**
   * Access token for authentication.
   * If not provided, will try to get from cookies (httpOnly cookies are automatically sent).
   * For better security, prefer using cookies over query parameter.
   */
  token?: string | null
  /**
   * Whether to include credentials (cookies) in WebSocket connection.
   * Default: true (recommended for httpOnly cookie authentication)
   */
  withCredentials?: boolean
}

export function useWebSocket(url: string, options?: UseWebSocketOptions): UseWebSocketReturn {
  const [data, setData] = useState<WebSocketMessage | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [reconnectAttempts, setReconnectAttempts] = useState(0)
  const maxReconnectAttempts = 5

  const connect = useCallback(() => {
    try {
      // Build WebSocket URL with token if provided
      let wsUrl = url
      const token = options?.token
      
      // Add token to query parameter if provided (fallback method)
      // Note: Prefer using httpOnly cookies (automatically sent) for better security
      if (token) {
        const separator = url.includes('?') ? '&' : '?'
        wsUrl = `${url}${separator}token=${encodeURIComponent(token)}`
      }
      
      // Note: WebSocket API doesn't support withCredentials option directly
      // Cookies (including httpOnly cookies) are automatically sent by the browser
      // if the WebSocket URL is on the same origin or configured CORS allows credentials
      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        console.log('WebSocket connected:', url)
        setIsConnected(true)
        setError(null)
        setReconnectAttempts(0)
      }

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          setData(message)
        } catch (err) {
          console.error('Error parsing WebSocket message:', err)
        }
      }

      ws.onerror = (event) => {
        console.error('WebSocket error:', event)
        setError(new Error('WebSocket connection error'))
      }

      ws.onclose = () => {
        console.log('WebSocket disconnected')
        setIsConnected(false)

        // Attempt to reconnect
        if (reconnectAttempts < maxReconnectAttempts) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000)
          console.log(`Reconnecting in ${delay}ms...`)

          reconnectTimeoutRef.current = setTimeout(() => {
            setReconnectAttempts(prev => prev + 1)
            connect()
          }, delay)
        } else {
          setError(new Error('Max reconnection attempts reached'))
        }
      }

      wsRef.current = ws
    } catch (err) {
      console.error('Error creating WebSocket:', err)
      setError(err as Error)
    }
  }, [url, options?.token, reconnectAttempts])

  const reconnect = useCallback(() => {
    setReconnectAttempts(0)
    if (wsRef.current) {
      wsRef.current.close()
    }
    connect()
  }, [connect])

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message))
    } else {
      console.warn('WebSocket is not connected')
    }
  }, [])

  useEffect(() => {
    connect()

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [connect])

  return {
    data,
    isConnected,
    error,
    sendMessage,
    reconnect
  }
}

