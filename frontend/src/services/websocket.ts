import { useState, useEffect, useRef, useCallback } from 'react'

export interface WebSocketMessage {
  message_type: 'sensor_data' | 'status_update' | 'error'
  data: any
  timestamp: string
}

export interface UseWebSocketOptions {
  rigId: string
  autoConnect?: boolean
  onMessage?: (message: WebSocketMessage) => void
  onError?: (error: Event) => void
  onConnect?: () => void
  onDisconnect?: () => void
}

const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8001/api/v1'

export function useWebSocket(options: UseWebSocketOptions) {
  const { rigId, autoConnect = true, onMessage, onError, onConnect, onDisconnect } = options

  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const [error, setError] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()
  const reconnectAttemptsRef = useRef(0)
  const maxReconnectAttempts = 5

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    try {
      const wsUrl = `${WS_BASE_URL}/sensor-data/ws/${rigId}`
      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        console.log(`WebSocket connected for rig ${rigId}`)
        setIsConnected(true)
        setError(null)
        reconnectAttemptsRef.current = 0
        onConnect?.()
      }

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          setLastMessage(message)
          onMessage?.(message)
        } catch (err) {
          console.error('Error parsing WebSocket message:', err)
        }
      }

      ws.onerror = (event) => {
        console.error('WebSocket error:', event)
        setError('WebSocket connection error')
        onError?.(event)
      }

      ws.onclose = () => {
        console.log(`WebSocket disconnected for rig ${rigId}`)
        setIsConnected(false)
        onDisconnect?.()

        // Auto-reconnect logic
        if (autoConnect && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current++
          const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 30000)
          console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current})`)
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, delay)
        }
      }

      wsRef.current = ws
    } catch (err) {
      console.error('Error creating WebSocket:', err)
      setError('Failed to create WebSocket connection')
    }
  }, [rigId, autoConnect, onMessage, onError, onConnect, onDisconnect])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }

    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    setIsConnected(false)
  }, [])

  const send = useCallback((message: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(message)
    } else {
      console.warn('WebSocket is not connected')
    }
  }, [])

  useEffect(() => {
    if (autoConnect && rigId) {
      connect()
    }

    return () => {
      disconnect()
    }
  }, [rigId, autoConnect, connect, disconnect])

  return {
    isConnected,
    lastMessage,
    error,
    connect,
    disconnect,
    send,
  }
}

