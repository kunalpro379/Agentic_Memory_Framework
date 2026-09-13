from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue, Range
)
from typing import Optional, List
import numpy as np

from Src.Schema.MemoryTypes import KMem, MemoryType


class QdrantStorage:
    """
    Qdrant-based storage for episodic and procedural memory.
    
    Characteristics:
    - High-performance vector similarity search
    - Advanced filtering (scope, time, metadata)
    - Supports both episodic and procedural memory
    - Production-ready with horizontal scaling
    """
    
    EMBEDDING_DIM = 768
    
    def __init__(self, config):
        self.client = QdrantClient(
            host=config.host,
            port=config.port,
            api_key=config.api_key
        )
        self._ensure_collections()
    
    def _ensure_collections(self):
        """Create collections if they don't exist."""
        collections = {
            "episodic": {
                "vector_size": self.EMBEDDING_DIM,
                "distance": Distance.COSINE
            },
            "procedural": {
                "vector_size": self.EMBEDDING_DIM,
                "distance": Distance.COSINE
            }
        }
        
        for collection_name, params in collections.items():
            existing_collections = [c.name for c in self.client.get_collections().collections]
            if collection_name not in existing_collections:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(**params)
                )
    
    def upsert(self, collection_name: str, node: KMem) -> bool:
        """Insert or update a memory node."""
        try:
            if node.embedding is None:
                return False
            
            point = PointStruct(
                id=node.node_id,
                vector=node.embedding.tolist(),
                payload={
                    "content": node.content,
                    "agent_id": node.agent_id,
                    "user_id": node.user_id,
                    "session_id": node.session_id,
                    "org_id": node.org_id,
                    "created_at": node.created_at.isoformat(),
                    "confidence": node.confidence,
                    "memory_type": node.memory_type.value,
                    "status": node.status.value,
                    "decay_rate": node.decay_rate,
                    "structured": node.structured
                }
            )
            
            self.client.upsert(collection_name=collection_name, points=[point])
            return True
        except Exception as e:
            print(f"Qdrant upsert error: {e}")
            return False
    
    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        org_id: Optional[str] = None,
        time_range_days: Optional[int] = None,
        limit: int = 10,
        min_confidence: float = 0.0
    ) -> List[tuple[KMem, float]]:
        """Search for similar memories with filters."""
        try:
            must_conditions = []
            
            # Scope filters
            if agent_id:
                must_conditions.append(
                    FieldCondition(key="agent_id", match=MatchValue(value=agent_id))
                )
            if user_id:
                must_conditions.append(
                    FieldCondition(key="user_id", match=MatchValue(value=user_id))
                )
            if session_id:
                must_conditions.append(
                    FieldCondition(key="session_id", match=MatchValue(value=session_id))
                )
            if org_id:
                must_conditions.append(
                    FieldCondition(key="org_id", match=MatchValue(value=org_id))
                )
            
            # Time range filter
            if time_range_days:
                from datetime import datetime, timedelta
                cutoff = (datetime.utcnow() - timedelta(days=time_range_days)).isoformat()
                must_conditions.append(
                    FieldCondition(key="created_at", range={"gte": cutoff})
                )
            
            # Confidence filter
            if min_confidence > 0:
                must_conditions.append(
                    FieldCondition(key="confidence", range={"gte": min_confidence})
                )
            
            search_filter = Filter(must=must_conditions) if must_conditions else None
            
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector.tolist(),
                query_filter=search_filter,
                limit=limit,
                with_payload=True
            )
            
            nodes = []
            for result in results:
                payload = result.payload
                node = KMem(
                    node_id=str(result.id),
                    memory_type=MemoryType(payload["memory_type"]),
                    content=payload["content"],
                    agent_id=payload.get("agent_id"),
                    user_id=payload.get("user_id"),
                    session_id=payload.get("session_id"),
                    org_id=payload.get("org_id"),
                    confidence=payload["confidence"],
                    created_at=payload.get("created_at"),
                    structured=payload.get("structured", {}),
                    decay_rate=payload.get("decay_rate", 0.01)
                )
                nodes.append((node, result.score))
            
            return nodes
        except Exception as e:
            print(f"Qdrant search error: {e}")
            return []
    
    def delete(self, collection_name: str, node_id: str) -> bool:
        """Delete a memory node."""
        try:
            self.client.delete(collection_name=collection_name, points_selector=[node_id])
            return True
        except Exception as e:
            print(f"Qdrant delete error: {e}")
            return False
    
    def health_check(self) -> bool:
        """Check if Qdrant is accessible."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

