import redis
import json
import pickle
from typing import Optional, Any
from datetime import timedelta

from Src.Schema.MemoryTypes import KMem, MemoryType
class RedisStorage:
    """
    Redis-based storage for working memory.
    
    Characteristics:
    - Sub-millisecond access times
    - TTL-native for automatic cleanup
    - Supports complex data structures
    - Ideal for session-scoped temporary data
    """
    
    def __init__(self, config):
        self.client = redis.Redis(
            host=config.host,
            port=config.port,
            db=config.db,
            password=config.password,
            decode_responses=False
        )
        self.default_ttl = timedelta(hours=1)  # Default session TTL
    
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        """Store a value with optional TTL."""
        try:
            serialized = pickle.dumps(value)
            ttl = ttl or self.default_ttl
            return self.client.setex(key, int(ttl.total_seconds()), serialized)
        except Exception as e:
            print(f"Redis set error: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value."""
        try:
            data = self.client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            print(f"Redis get error: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a key."""
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            print(f"Redis delete error: {e}")
            return False
    
    def set_working_memory(self, session_id: str, node: KMem, ttl: Optional[timedelta] = None) -> bool:
        """Store a working memory node."""
        key = f"working:{session_id}:{node.node_id}"
        return self.set(key, node, ttl)
    
    def get_working_memory(self, session_id: str, node_id: str) -> Optional[KMem]:
        """Retrieve a working memory node."""
        key = f"working:{session_id}:{node_id}"
        return self.get(key)
    
    def get_session_context(self, session_id: str) -> list[KMem]:
        """Get all working memory nodes for a session."""
        pattern = f"working:{session_id}:*"
        keys = self.client.keys(pattern)
        nodes = []
        for key in keys:
            node = self.get(key)
            if node:
                nodes.append(node)
        return nodes
    
    def clear_session(self, session_id: str) -> int:
        """Clear all working memory for a session."""
        pattern = f"working:{session_id}:*"
        keys = self.client.keys(pattern)
        if keys:
            return self.client.delete(*keys)
        return 0
    
    def health_check(self) -> bool:
        """Check if Redis is accessible."""
        try:
            return self.client.ping()
        except Exception:
            return False    