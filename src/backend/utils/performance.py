"""
Performance optimization utilities
"""
import functools
import time
import logging
from typing import Callable, Any, Optional
from datetime import timedelta

from services.cache_service import cache_service

logger = logging.getLogger(__name__)


def cache_result(
    ttl: int = 60,
    key_prefix: str = "cache",
    key_func: Optional[Callable] = None
):
    """
    Decorator to cache function results
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
        key_func: Optional function to generate cache key from arguments
    
    Example:
        @cache_result(ttl=300, key_prefix="sensor_data")
        def get_sensor_data(rig_id: str):
            # ... expensive operation
            return data
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default: use function name + args + kwargs
                key_parts = [key_prefix, func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cached_result = cache_service.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache_service.set(cache_key, result, ttl=ttl)
            logger.debug(f"Cached result for key: {cache_key} (TTL: {ttl}s)")
            
            return result
        
        return wrapper
    return decorator


def measure_time(func: Callable) -> Callable:
    """
    Decorator to measure function execution time
    
    Example:
        @measure_time
        def expensive_operation():
            # ... operation
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        logger.info(
            f"Function {func.__name__} executed in {execution_time:.3f}s"
        )
        
        return result
    
    return wrapper


def async_measure_time(func: Callable) -> Callable:
    """
    Decorator to measure async function execution time
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        logger.info(
            f"Async function {func.__name__} executed in {execution_time:.3f}s"
        )
        
        return result
    
    return wrapper


def paginate_query(query, page: int = 1, per_page: int = 100, max_per_page: int = 1000):
    """
    Optimized pagination helper for SQLAlchemy queries
    
    Args:
        query: SQLAlchemy query object
        page: Page number (1-indexed)
        per_page: Items per page
        max_per_page: Maximum items per page
    
    Returns:
        Tuple of (items, total_count, page, per_page, total_pages)
    """
    # Validate and clamp per_page
    per_page = min(per_page, max_per_page)
    per_page = max(1, per_page)
    page = max(1, page)
    
    # Calculate offset
    offset = (page - 1) * per_page
    
    # Get total count (optimized for large datasets)
    # Use COUNT(*) for better performance
    total_count = query.count()
    
    # Get paginated results
    items = query.offset(offset).limit(per_page).all()
    
    # Calculate total pages
    total_pages = (total_count + per_page - 1) // per_page if total_count > 0 else 0
    
    return {
        "items": items,
        "total": total_count,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1
    }


def optimize_query(query, use_indexes: bool = True):
    """
    Apply query optimizations
    
    Args:
        query: SQLAlchemy query object
        use_indexes: Whether to hint database to use indexes
    
    Returns:
        Optimized query
    """
    # Add query hints for index usage (PostgreSQL specific)
    if use_indexes:
        # This is a placeholder - actual implementation depends on database
        # For PostgreSQL, you might use query.with_hint()
        pass
    
    return query


class QueryPerformanceMonitor:
    """
    Monitor and log slow queries
    """
    
    def __init__(self, slow_query_threshold: float = 1.0):
        """
        Args:
            slow_query_threshold: Threshold in seconds for logging slow queries
        """
        self.slow_query_threshold = slow_query_threshold
        self.slow_queries = []
    
    def monitor(self, query_func: Callable):
        """
        Decorator to monitor query performance
        
        Args:
            query_func: Function that executes a query
        
        Returns:
            Decorated function
        """
        @functools.wraps(query_func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = query_func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if execution_time > self.slow_query_threshold:
                logger.warning(
                    f"Slow query detected: {query_func.__name__} "
                    f"took {execution_time:.3f}s (threshold: {self.slow_query_threshold}s)"
                )
                self.slow_queries.append({
                    "function": query_func.__name__,
                    "execution_time": execution_time,
                    "args": str(args),
                    "kwargs": str(kwargs)
                })
            
            return result
        
        return wrapper
    
    def get_slow_queries(self):
        """Get list of slow queries"""
        return self.slow_queries
    
    def clear_slow_queries(self):
        """Clear slow queries log"""
        self.slow_queries.clear()


# Global query performance monitor
query_monitor = QueryPerformanceMonitor(slow_query_threshold=1.0)

