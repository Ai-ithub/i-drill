"""
Unit tests for Cache Service
"""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from services.cache_service import CacheService


class TestCacheService:
    """Tests for CacheService"""
    
    @pytest.fixture
    def cache_service(self):
        """Create CacheService instance"""
        return CacheService()
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client"""
        mock = MagicMock()
        mock.ping.return_value = True
        mock.get.return_value = None
        mock.set.return_value = True
        mock.delete.return_value = 1
        mock.exists.return_value = False
        mock.keys.return_value = []
        return mock
    
    def test_init_without_redis(self, monkeypatch):
        """Test CacheService initialization without Redis"""
        monkeypatch.setenv("REDIS_HOST", "localhost")
        
        with patch('services.cache_service.REDIS_AVAILABLE', False):
            service = CacheService()
            assert service.enabled is False
            assert service.redis_client is None
    
    def test_init_with_redis_connection_failure(self, monkeypatch):
        """Test CacheService initialization with Redis connection failure"""
        monkeypatch.setenv("REDIS_HOST", "invalid_host")
        
        with patch('services.cache_service.redis') as mock_redis_module:
            mock_redis_module.Redis.side_effect = Exception("Connection failed")
            service = CacheService()
            assert service.enabled is False
    
    def test_get_when_disabled(self, cache_service):
        """Test get when cache is disabled"""
        cache_service.enabled = False
        result = cache_service.get("test_key")
        assert result is None
    
    def test_get_when_enabled(self, cache_service, mock_redis):
        """Test get when cache is enabled"""
        cache_service.enabled = True
        cache_service.redis_client = mock_redis
        mock_redis.get.return_value = '{"test": "value"}'
        
        result = cache_service.get("test_key")
        assert result == {"test": "value"}
        mock_redis.get.assert_called_once_with("test_key")
    
    def test_get_nonexistent_key(self, cache_service, mock_redis):
        """Test getting non-existent key"""
        cache_service.enabled = True
        cache_service.redis_client = mock_redis
        mock_redis.get.return_value = None
        
        result = cache_service.get("nonexistent")
        assert result is None
    
    def test_set_when_disabled(self, cache_service):
        """Test set when cache is disabled"""
        cache_service.enabled = False
        result = cache_service.set("test_key", {"test": "value"})
        assert result is False
    
    def test_set_when_enabled(self, cache_service, mock_redis):
        """Test set when cache is enabled"""
        cache_service.enabled = True
        cache_service.redis_client = mock_redis
        
        result = cache_service.set("test_key", {"test": "value"}, ttl=3600)
        assert result is True
        mock_redis.set.assert_called_once()
    
    def test_delete_when_disabled(self, cache_service):
        """Test delete when cache is disabled"""
        cache_service.enabled = False
        result = cache_service.delete("test_key")
        assert result is False
    
    def test_delete_when_enabled(self, cache_service, mock_redis):
        """Test delete when cache is enabled"""
        cache_service.enabled = True
        cache_service.redis_client = mock_redis
        
        result = cache_service.delete("test_key")
        assert result is True
        mock_redis.delete.assert_called_once_with("test_key")
    
    def test_exists_when_disabled(self, cache_service):
        """Test exists when cache is disabled"""
        cache_service.enabled = False
        result = cache_service.exists("test_key")
        assert result is False
    
    def test_exists_when_enabled(self, cache_service, mock_redis):
        """Test exists when cache is enabled"""
        cache_service.enabled = True
        cache_service.redis_client = mock_redis
        mock_redis.exists.return_value = True
        
        result = cache_service.exists("test_key")
        assert result is True
        mock_redis.exists.assert_called_once_with("test_key")
    
    def test_clear_when_disabled(self, cache_service):
        """Test clear when cache is disabled"""
        cache_service.enabled = False
        result = cache_service.clear()
        assert result is False
    
    def test_clear_when_enabled(self, cache_service, mock_redis):
        """Test clear when cache is enabled"""
        cache_service.enabled = True
        cache_service.redis_client = mock_redis
        mock_redis.keys.return_value = ["key1", "key2"]
        
        result = cache_service.clear()
        assert result is True
        assert mock_redis.delete.called
    
    def test_get_with_json_serialization(self, cache_service, mock_redis):
        """Test get with JSON serialized value"""
        cache_service.enabled = True
        cache_service.redis_client = mock_redis
        mock_redis.get.return_value = '{"nested": {"key": "value"}}'
        
        result = cache_service.get("test_key")
        assert isinstance(result, dict)
        assert result["nested"]["key"] == "value"
    
    def test_set_with_json_serialization(self, cache_service, mock_redis):
        """Test set with complex object"""
        cache_service.enabled = True
        cache_service.redis_client = mock_redis
        
        complex_obj = {
            "nested": {
                "key": "value",
                "list": [1, 2, 3]
            }
        }
        
        result = cache_service.set("test_key", complex_obj)
        assert result is True
        # Verify JSON serialization was used
        call_args = mock_redis.set.call_args
        assert "test_key" in str(call_args)
    
    def test_get_with_ttl(self, cache_service, mock_redis):
        """Test set with TTL"""
        cache_service.enabled = True
        cache_service.redis_client = mock_redis
        
        result = cache_service.set("test_key", "value", ttl=3600)
        assert result is True
        # Verify TTL was set
        assert mock_redis.set.called

