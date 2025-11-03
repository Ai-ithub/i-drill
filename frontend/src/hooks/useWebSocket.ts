import { useState, useEffect, useRef, useCallback } from 'react'

interface WebSocketMessage {
  message_type: string
  data: any
  timestamp: string
}

interface UseWebSocketReturn {
  data: WebSocketMessage | null
  isConnected: boolean
  error: Error | null
  sendMessage: (message: any) => void
  reconnect: () => void
}

export function useWebSocket(url: string): UseWebSocketReturn {
  const [data, setData] = useState<WebSocketMessage | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const [reconnectAttempts, setReconnectAttempts] = useState(0)
  const maxReconnectAttempts = 5

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(url)

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
  }, [url, reconnectAttempts])

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

