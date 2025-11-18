"""
WebSocket Connection Manager for managing multiple WebSocket connections
"""
import logging
from typing import Dict, Set, Optional
from fastapi import WebSocket
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections for real-time data streaming.
    
    Handles multiple WebSocket connections organized by rig ID, allowing
    broadcasting of sensor data and other real-time updates to connected clients.
    
    Attributes:
        active_connections: Dictionary mapping rig_id to set of WebSocket connections
        connection_rigs: Dictionary mapping WebSocket to rig_id for reverse lookup
        connection_users: Dictionary mapping WebSocket to user_id for authentication tracking
    """
    
    def __init__(self):
        """
        Initialize WebSocketManager.
        
        Sets up empty connection tracking dictionaries.
        """
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.connection_rigs: Dict[WebSocket, str] = {}
        self.connection_users: Dict[WebSocket, int] = {}  # Track user_id for each connection
    
    async def connect(self, websocket: WebSocket, rig_id: str, user_id: Optional[int] = None) -> None:
        """
        Accept and register a WebSocket connection.
        
        Accepts the WebSocket handshake and adds the connection to the
        appropriate rig's connection set.
        
        Args:
            websocket: FastAPI WebSocket instance
            rig_id: Rig identifier for this connection
            user_id: Optional user ID for authentication tracking
        """
        await websocket.accept()
        
        if rig_id not in self.active_connections:
            self.active_connections[rig_id] = set()
        
        self.active_connections[rig_id].add(websocket)
        self.connection_rigs[websocket] = rig_id
        
        if user_id:
            self.connection_users[websocket] = user_id
        
        logger.info(
            f"WebSocket connected for rig {rig_id}" +
            (f" (user_id: {user_id})" if user_id else "") +
            f". Total connections: {len(self.connection_rigs)}"
        )
    
    def disconnect(self, websocket: WebSocket) -> None:
        """
        Remove a WebSocket connection.
        
        Removes the connection from tracking and cleans up empty rig sets.
        
        Args:
            websocket: WebSocket instance to disconnect
        """
        rig_id = self.connection_rigs.pop(websocket, None)
        user_id = self.connection_users.pop(websocket, None)
        
        if rig_id and rig_id in self.active_connections:
            self.active_connections[rig_id].discard(websocket)
            
            if not self.active_connections[rig_id]:
                del self.active_connections[rig_id]
        
        logger.info(
            f"WebSocket disconnected for rig {rig_id}" +
            (f" (user_id: {user_id})" if user_id else "") +
            f". Total connections: {len(self.connection_rigs)}"
        )
    
    async def send_to_rig(self, rig_id: str, message: dict) -> None:
        """
        Send message to all connections for a specific rig.
        
        Broadcasts a JSON message to all WebSocket clients connected to
        the specified rig. Automatically removes disconnected connections.
        
        Args:
            rig_id: Rig identifier to send message to
            message: Dictionary to send as JSON
        """
        if rig_id not in self.active_connections:
            return
        
        disconnected = set()
        
        for connection in self.active_connections[rig_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {rig_id}: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected connections
        for conn in disconnected:
            self.disconnect(conn)
    
    async def broadcast_to_all(self, message: dict) -> None:
        """
        Broadcast message to all connected clients.
        
        Sends a JSON message to all WebSocket clients regardless of rig.
        Automatically removes disconnected connections.
        
        Args:
            message: Dictionary to send as JSON
        """
        disconnected = set()
        
        for connections in self.active_connections.values():
            for connection in connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting message: {e}")
                    disconnected.add(connection)
        
        # Clean up disconnected connections
        for conn in disconnected:
            self.disconnect(conn)
    
    def get_connected_rigs(self) -> Set[str]:
        """
        Get set of all rig IDs with active connections.
        
        Returns:
            Set of rig IDs that have at least one active WebSocket connection
        """
        return set(self.active_connections.keys())
    
    def get_connection_count(self, rig_id: str = None) -> int:
        """
        Get connection count for a specific rig or total.
        
        Args:
            rig_id: Optional rig ID to get count for. If None, returns total count.
            
        Returns:
            Number of active connections for the specified rig, or total if rig_id is None
        """
        if rig_id:
            return len(self.active_connections.get(rig_id, set()))
        return len(self.connection_rigs)
    
    def get_user_connections(self, user_id: int) -> Set[WebSocket]:
        """
        Get all WebSocket connections for a specific user.
        
        Args:
            user_id: User ID to get connections for
            
        Returns:
            Set of WebSocket connections for the user
        """
        return {
            ws for ws, uid in self.connection_users.items()
            if uid == user_id
        }
    
    async def disconnect_user(self, user_id: int) -> int:
        """
        Disconnect all WebSocket connections for a specific user.
        
        Useful for logging out a user or revoking access.
        
        Args:
            user_id: User ID to disconnect
            
        Returns:
            Number of connections disconnected
        """
        user_connections = self.get_user_connections(user_id)
        count = len(user_connections)
        
        for ws in list(user_connections):  # Create a list copy to avoid modification during iteration
            try:
                # Close the WebSocket gracefully
                await ws.close(code=1008, reason="User logged out")
            except:
                pass
            self.disconnect(ws)
        
        logger.info(f"Disconnected {count} WebSocket connection(s) for user_id: {user_id}")
        return count


# Global WebSocket manager instance
websocket_manager = WebSocketManager()

