"""
Cache Service using Redis
"""
import json
import logging
from typing import Optional, Any, Union
from datetime import timedelta
import os

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None  # type: ignore


class CacheService:
    """
    Service for caching operations using Redis.
    
    Provides key-value caching functionality with TTL support.
    Automatically handles JSON serialization/deserialization for complex objects.
    
    Attributes:
        redis_client: Redis client instance (if available)
        enabled: Boolean indicating if caching is enabled
    """
    
    def __init__(self):
        """
        Initialize CacheService.
        
        Attempts to connect to Redis using environment variables.
        Falls back to disabled state if Redis is unavailable.
        """
        self.redis_client = None
        self.enabled = False
        
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available. Caching disabled.")
            return
        
        try:
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            redis_db = int(os.getenv("REDIS_DB", "0"))
            redis_password = os.getenv("REDIS_PASSWORD")
            
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            self.enabled = True
            logger.info(f"✅ Redis cache connected: {redis_host}:{redis_port}")
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            self.redis_client = None
            self.enabled = False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            Cached value if found, None otherwise. Automatically deserializes
            JSON values if applicable.
        """
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            value = self.redis_client.get(key)
            if value is None:
                return None
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logger.error(f"Cache get error for key '{key}': {e}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """
        Set value in cache with optional TTL.
        
        Automatically serializes complex objects to JSON. Supports both
        integer seconds and timedelta objects for TTL.
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized if needed)
            ttl: Time to live in seconds (int) or timedelta object
            
        Returns:
            True if value was successfully cached, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            # Serialize value to JSON if needed
            if not isinstance(value, (str, int, float, bool)):
                value = json.dumps(value)
            
            # Convert timedelta to seconds
            if isinstance(ttl, timedelta):
                ttl = int(ttl.total_seconds())
            
            if ttl:
                self.redis_client.setex(key, ttl, value)
            else:
                self.redis_client.set(key, value)
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key '{key}': {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error for key '{key}': {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching a pattern.
        
        Uses Redis KEYS command to find matching keys, then deletes them.
        Supports Redis pattern matching (e.g., "user:*").
        
        Args:
            pattern: Redis key pattern to match
            
        Returns:
            Number of keys deleted
        """
        if not self.enabled or not self.redis_client:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache clear pattern error for '{pattern}': {e}")
            return 0
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Cache exists error for key '{key}': {e}")
            return False
    
    def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment a numeric value in cache.
        
        Atomically increments the value stored at key by the specified amount.
        If the key doesn't exist, it is initialized to 0 before incrementing.
        
        Args:
            key: Cache key to increment
            amount: Amount to increment by (default: 1)
            
        Returns:
            New value after increment, or None if operation failed
        """
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            return self.redis_client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Cache increment error for key '{key}': {e}")
            return None


# Global cache service instance
cache_service = CacheService()

