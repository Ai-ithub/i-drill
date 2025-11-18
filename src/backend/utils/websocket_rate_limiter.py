"""
WebSocket Rate Limiter
Implements rate limiting for WebSocket connections to prevent abuse
"""
import logging
import time
from typing import Dict, Optional
from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketRateLimiter:
    """
    Rate limiter for WebSocket connections.
    
    Tracks connection attempts and message rates per user/IP to prevent abuse.
    Uses sliding window algorithm for rate limiting.
    """
    
    def __init__(
        self,
        max_connections_per_user: int = 5,
        max_connections_per_ip: int = 10,
        max_messages_per_minute: int = 100,
        window_seconds: int = 60
    ):
        """
        Initialize WebSocket rate limiter.
        
        Args:
            max_connections_per_user: Maximum concurrent connections per user
            max_connections_per_ip: Maximum concurrent connections per IP address
            max_messages_per_minute: Maximum messages per minute per connection
            window_seconds: Time window for rate limiting in seconds
        """
        self.max_connections_per_user = max_connections_per_user
        self.max_connections_per_ip = max_connections_per_ip
        self.max_messages_per_minute = max_messages_per_minute
        self.window_seconds = window_seconds
        
        # Track connections per user
        self.user_connections: Dict[int, set] = defaultdict(set)
        
        # Track connections per IP
        self.ip_connections: Dict[str, set] = defaultdict(set)
        
        # Track message timestamps per connection
        self.connection_messages: Dict[WebSocket, list] = {}
        
        # Track connection attempts (for rate limiting connection attempts)
        self.connection_attempts: Dict[str, list] = defaultdict(list)
        
        # Maximum connection attempts per minute per IP
        self.max_connection_attempts_per_minute = 10
    
    def _get_client_ip(self, websocket: WebSocket) -> str:
        """
        Get client IP address from WebSocket.
        
        Args:
            websocket: WebSocket instance
            
        Returns:
            IP address string
        """
        # Try to get IP from headers (when behind proxy)
        if websocket.headers:
            forwarded_for = websocket.headers.get("x-forwarded-for")
            if forwarded_for:
                return forwarded_for.split(",")[0].strip()
            
            real_ip = websocket.headers.get("x-real-ip")
            if real_ip:
                return real_ip
        
        # Fallback to client host
        if websocket.client:
            return websocket.client.host or "unknown"
        
        return "unknown"
    
    def _cleanup_old_messages(self, connection: WebSocket) -> None:
        """
        Clean up old message timestamps outside the time window.
        
        Args:
            connection: WebSocket connection
        """
        if connection not in self.connection_messages:
            return
        
        now = time.time()
        cutoff = now - self.window_seconds
        
        # Remove timestamps older than the window
        self.connection_messages[connection] = [
            ts for ts in self.connection_messages[connection]
            if ts > cutoff
        ]
    
    def check_connection_allowed(
        self,
        websocket: WebSocket,
        user_id: Optional[int] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a new WebSocket connection is allowed.
        
        Args:
            websocket: WebSocket instance
            user_id: Optional user ID for authenticated users
            
        Returns:
            Tuple of (is_allowed, reason_if_not_allowed)
        """
        ip_address = self._get_client_ip(websocket)
        
        # Check connection attempts rate limit per IP
        now = time.time()
        cutoff = now - 60  # 1 minute window
        
        # Clean up old attempts
        self.connection_attempts[ip_address] = [
            ts for ts in self.connection_attempts[ip_address]
            if ts > cutoff
        ]
        
        # Check if too many connection attempts
        if len(self.connection_attempts[ip_address]) >= self.max_connection_attempts_per_minute:
            logger.warning(
                f"WebSocket connection rate limit exceeded for IP {ip_address}: "
                f"{len(self.connection_attempts[ip_address])} attempts in last minute"
            )
            return False, "Too many connection attempts. Please try again later."
        
        # Record this connection attempt
        self.connection_attempts[ip_address].append(now)
        
        # Check per-user connection limit
        if user_id:
            user_conn_count = len(self.user_connections.get(user_id, set()))
            if user_conn_count >= self.max_connections_per_user:
                logger.warning(
                    f"WebSocket connection limit exceeded for user {user_id}: "
                    f"{user_conn_count} active connections"
                )
                return False, f"Maximum concurrent connections ({self.max_connections_per_user}) reached for this user."
        
        # Check per-IP connection limit
        ip_conn_count = len(self.ip_connections.get(ip_address, set()))
        if ip_conn_count >= self.max_connections_per_ip:
            logger.warning(
                f"WebSocket connection limit exceeded for IP {ip_address}: "
                f"{ip_conn_count} active connections"
            )
            return False, f"Maximum concurrent connections ({self.max_connections_per_ip}) reached for this IP."
        
        return True, None
    
    def register_connection(
        self,
        websocket: WebSocket,
        user_id: Optional[int] = None
    ) -> None:
        """
        Register a new WebSocket connection.
        
        Args:
            websocket: WebSocket instance
            user_id: Optional user ID
        """
        ip_address = self._get_client_ip(websocket)
        
        # Register connection
        if user_id:
            self.user_connections[user_id].add(websocket)
        
        self.ip_connections[ip_address].add(websocket)
        self.connection_messages[websocket] = []
        
        logger.debug(
            f"WebSocket connection registered: user_id={user_id}, "
            f"ip={ip_address}, total_user_conns={len(self.user_connections.get(user_id, set()))}, "
            f"total_ip_conns={len(self.ip_connections.get(ip_address, set()))}"
        )
    
    def unregister_connection(self, websocket: WebSocket) -> None:
        """
        Unregister a WebSocket connection.
        
        Args:
            websocket: WebSocket instance
        """
        # Remove from user connections
        for user_id, connections in list(self.user_connections.items()):
            connections.discard(websocket)
            if not connections:
                del self.user_connections[user_id]
        
        # Remove from IP connections
        ip_address = self._get_client_ip(websocket)
        self.ip_connections[ip_address].discard(websocket)
        if not self.ip_connections[ip_address]:
            del self.ip_connections[ip_address]
        
        # Remove message tracking
        self.connection_messages.pop(websocket, None)
    
    def check_message_allowed(self, websocket: WebSocket) -> tuple[bool, Optional[str]]:
        """
        Check if sending a message is allowed (rate limit check).
        
        Args:
            websocket: WebSocket instance
            
        Returns:
            Tuple of (is_allowed, reason_if_not_allowed)
        """
        if websocket not in self.connection_messages:
            # Connection not registered, allow but log warning
            logger.warning("Message check for unregistered WebSocket connection")
            return True, None
        
        # Clean up old messages
        self._cleanup_old_messages(websocket)
        
        # Check message rate
        message_count = len(self.connection_messages[websocket])
        if message_count >= self.max_messages_per_minute:
            logger.warning(
                f"WebSocket message rate limit exceeded: {message_count} messages in last {self.window_seconds} seconds"
            )
            return False, f"Message rate limit exceeded ({self.max_messages_per_minute} messages per minute)."
        
        # Record this message
        self.connection_messages[websocket].append(time.time())
        
        return True, None
    
    def get_stats(self) -> dict:
        """
        Get rate limiter statistics.
        
        Returns:
            Dictionary with statistics
        """
        total_connections = sum(len(conns) for conns in self.user_connections.values())
        total_ips = len(self.ip_connections)
        
        return {
            "total_connections": total_connections,
            "total_users": len(self.user_connections),
            "total_ips": total_ips,
            "max_connections_per_user": self.max_connections_per_user,
            "max_connections_per_ip": self.max_connections_per_ip,
            "max_messages_per_minute": self.max_messages_per_minute,
        }


# Global WebSocket rate limiter instance
websocket_rate_limiter = WebSocketRateLimiter(
    max_connections_per_user=int(__import__("os").getenv("WS_MAX_CONNECTIONS_PER_USER", "5")),
    max_connections_per_ip=int(__import__("os").getenv("WS_MAX_CONNECTIONS_PER_IP", "10")),
    max_messages_per_minute=int(__import__("os").getenv("WS_MAX_MESSAGES_PER_MINUTE", "100")),
)

