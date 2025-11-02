"""
WebSocket Connection Manager for managing multiple WebSocket connections
"""
import logging
from typing import Dict, Set
from fastapi import WebSocket
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections for real-time data streaming"""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.connection_rigs: Dict[WebSocket, str] = {}
    
    async def connect(self, websocket: WebSocket, rig_id: str):
        """Accept and register a WebSocket connection"""
        await websocket.accept()
        
        if rig_id not in self.active_connections:
            self.active_connections[rig_id] = set()
        
        self.active_connections[rig_id].add(websocket)
        self.connection_rigs[websocket] = rig_id
        
        logger.info(f"WebSocket connected for rig {rig_id}. Total connections: {len(self.connection_rigs)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        rig_id = self.connection_rigs.pop(websocket, None)
        
        if rig_id and rig_id in self.active_connections:
            self.active_connections[rig_id].discard(websocket)
            
            if not self.active_connections[rig_id]:
                del self.active_connections[rig_id]
        
        logger.info(f"WebSocket disconnected for rig {rig_id}. Total connections: {len(self.connection_rigs)}")
    
    async def send_to_rig(self, rig_id: str, message: dict):
        """Send message to all connections for a specific rig"""
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
    
    async def broadcast_to_all(self, message: dict):
        """Broadcast message to all connected clients"""
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
        """Get set of all rig IDs with active connections"""
        return set(self.active_connections.keys())
    
    def get_connection_count(self, rig_id: str = None) -> int:
        """Get connection count for a specific rig or total"""
        if rig_id:
            return len(self.active_connections.get(rig_id, set()))
        return len(self.connection_rigs)


# Global WebSocket manager instance
websocket_manager = WebSocketManager()

