"""
Unit tests for WebSocket Manager
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import WebSocket
from services.websocket_manager import WebSocketManager


class TestWebSocketManager:
    """Tests for WebSocketManager"""
    
    @pytest.fixture
    def manager(self):
        """Create WebSocketManager instance"""
        return WebSocketManager()
    
    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket"""
        ws = AsyncMock(spec=WebSocket)
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()
        ws.close = AsyncMock()
        return ws
    
    @pytest.mark.asyncio
    async def test_connect(self, manager, mock_websocket):
        """Test connecting a WebSocket"""
        rig_id = "RIG_01"
        
        await manager.connect(mock_websocket, rig_id)
        
        # Verify WebSocket was accepted
        mock_websocket.accept.assert_called_once()
        
        # Verify connection was registered
        assert rig_id in manager.active_connections
        assert mock_websocket in manager.active_connections[rig_id]
        assert manager.connection_rigs[mock_websocket] == rig_id
    
    @pytest.mark.asyncio
    async def test_connect_multiple_rigs(self, manager):
        """Test connecting multiple WebSockets to different rigs"""
        ws1 = AsyncMock(spec=WebSocket)
        ws1.accept = AsyncMock()
        ws2 = AsyncMock(spec=WebSocket)
        ws2.accept = AsyncMock()
        
        await manager.connect(ws1, "RIG_01")
        await manager.connect(ws2, "RIG_02")
        
        assert "RIG_01" in manager.active_connections
        assert "RIG_02" in manager.active_connections
        assert len(manager.active_connections["RIG_01"]) == 1
        assert len(manager.active_connections["RIG_02"]) == 1
    
    @pytest.mark.asyncio
    async def test_connect_multiple_to_same_rig(self, manager):
        """Test connecting multiple WebSockets to same rig"""
        ws1 = AsyncMock(spec=WebSocket)
        ws1.accept = AsyncMock()
        ws2 = AsyncMock(spec=WebSocket)
        ws2.accept = AsyncMock()
        
        await manager.connect(ws1, "RIG_01")
        await manager.connect(ws2, "RIG_01")
        
        assert len(manager.active_connections["RIG_01"]) == 2
    
    def test_disconnect(self, manager, mock_websocket):
        """Test disconnecting a WebSocket"""
        rig_id = "RIG_01"
        manager.active_connections[rig_id] = {mock_websocket}
        manager.connection_rigs[mock_websocket] = rig_id
        
        manager.disconnect(mock_websocket)
        
        # Verify connection was removed
        assert mock_websocket not in manager.connection_rigs
        assert rig_id not in manager.active_connections
    
    def test_disconnect_nonexistent(self, manager, mock_websocket):
        """Test disconnecting a non-existent WebSocket"""
        # Should not raise error
        manager.disconnect(mock_websocket)
        
        assert mock_websocket not in manager.connection_rigs
    
    def test_disconnect_removes_empty_rig(self, manager, mock_websocket):
        """Test that empty rig sets are removed after disconnect"""
        rig_id = "RIG_01"
        manager.active_connections[rig_id] = {mock_websocket}
        manager.connection_rigs[mock_websocket] = rig_id
        
        manager.disconnect(mock_websocket)
        
        # Empty rig set should be removed
        assert rig_id not in manager.active_connections
    
    @pytest.mark.asyncio
    async def test_send_to_rig(self, manager, mock_websocket):
        """Test sending message to a rig"""
        rig_id = "RIG_01"
        message = {"type": "sensor_data", "data": {"temperature": 25.5}}
        
        await manager.connect(mock_websocket, rig_id)
        await manager.send_to_rig(rig_id, message)
        
        # Verify message was sent
        mock_websocket.send_json.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_send_to_rig_multiple_connections(self, manager):
        """Test sending message to multiple connections"""
        rig_id = "RIG_01"
        message = {"type": "sensor_data", "data": {"temperature": 25.5}}
        
        ws1 = AsyncMock(spec=WebSocket)
        ws1.accept = AsyncMock()
        ws1.send_json = AsyncMock()
        ws2 = AsyncMock(spec=WebSocket)
        ws2.accept = AsyncMock()
        ws2.send_json = AsyncMock()
        
        await manager.connect(ws1, rig_id)
        await manager.connect(ws2, rig_id)
        await manager.send_to_rig(rig_id, message)
        
        # Verify message was sent to both
        ws1.send_json.assert_called_once_with(message)
        ws2.send_json.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_send_to_nonexistent_rig(self, manager):
        """Test sending message to non-existent rig"""
        message = {"type": "sensor_data", "data": {"temperature": 25.5}}
        
        # Should not raise error
        await manager.send_to_rig("NONEXISTENT", message)
    
    @pytest.mark.asyncio
    async def test_send_to_rig_handles_errors(self, manager, mock_websocket):
        """Test that send_to_rig handles connection errors"""
        rig_id = "RIG_01"
        message = {"type": "sensor_data", "data": {"temperature": 25.5}}
        
        # Make send_json raise an error
        mock_websocket.send_json.side_effect = Exception("Connection error")
        
        await manager.connect(mock_websocket, rig_id)
        await manager.send_to_rig(rig_id, message)
        
        # Connection should be removed after error
        assert mock_websocket not in manager.active_connections.get(rig_id, set())
        assert mock_websocket not in manager.connection_rigs
    
    @pytest.mark.asyncio
    async def test_broadcast(self, manager):
        """Test broadcasting message to all connections"""
        message = {"type": "system_update", "data": {"status": "maintenance"}}
        
        ws1 = AsyncMock(spec=WebSocket)
        ws1.accept = AsyncMock()
        ws1.send_json = AsyncMock()
        ws2 = AsyncMock(spec=WebSocket)
        ws2.accept = AsyncMock()
        ws2.send_json = AsyncMock()
        
        await manager.connect(ws1, "RIG_01")
        await manager.connect(ws2, "RIG_02")
        await manager.broadcast(message)
        
        # Verify message was sent to all connections
        ws1.send_json.assert_called_once_with(message)
        ws2.send_json.assert_called_once_with(message)
    
    def test_get_connection_count(self, manager, mock_websocket):
        """Test getting connection count"""
        assert manager.get_connection_count() == 0
        
        manager.active_connections["RIG_01"] = {mock_websocket}
        manager.connection_rigs[mock_websocket] = "RIG_01"
        
        assert manager.get_connection_count() == 1
    
    def test_get_rig_connections(self, manager, mock_websocket):
        """Test getting connections for a rig"""
        rig_id = "RIG_01"
        manager.active_connections[rig_id] = {mock_websocket}
        manager.connection_rigs[mock_websocket] = rig_id
        
        connections = manager.get_rig_connections(rig_id)
        assert mock_websocket in connections
        assert len(connections) == 1
    
    def test_get_rig_connections_nonexistent(self, manager):
        """Test getting connections for non-existent rig"""
        connections = manager.get_rig_connections("NONEXISTENT")
        assert len(connections) == 0

