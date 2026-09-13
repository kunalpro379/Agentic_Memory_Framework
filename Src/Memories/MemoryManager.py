"""
Main memory orchestration manager.
Coordinates all four storage tiers and provides unified interface.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, List
import numpy as np

from Src.Schema.MemoryTypes import KMem, MemoryType, MemoryStatus
from Src.Schema.Config import Config
from Src.Storage.RedisStorage import RedisStorage
from Src.Storage.QdrantStorage import QdrantStorage
from Src.Storage.Neo4jStorage import Neo4jStorage
from Src.Storage.PostgresStorage import PostgresStorage
from Src.LLM.EmbeddingProvider import EmbeddingProvider
from Src.LLM.LLMProvider import LLMProvider


class MemoryManager:
    """
    Single interface for all four memory tiers.
    
    Handles:
    - Write routing: which tier(s) receive a new memory
    - Read orchestration: querying multiple tiers and merging results
    - Consolidation: promoting episodic → semantic as patterns emerge
    - Decay: archiving low-salience memories on a background schedule
    - Scope enforcement: isolating memories by agent, user, session, org
    
    Example:
        manager = MemoryManager(config)
        await manager.initialize()
        await manager.remember(node)
        results = await manager.recall("client issue")
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize storage backends
        self.redis = RedisStorage(config.redis)
        self.qdrant = QdrantStorage(config.qdrant)
        self.neo4j = Neo4jStorage(config.neo4j)
        self.postgres = PostgresStorage(config.postgres)
        
        # Initialize LLM components
        self.embedder = EmbeddingProvider(config.llm.embedding_model)
        self.llm = LLMProvider(config.llm.openai_api_key, config.llm.llm_model)
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize all storage backends."""
        if not self._initialized:
            await self.postgres.initialize()
            await self.neo4j.setup_indexes()
            self._initialized = True
    
    async def close(self):
        """Close all connections."""
        await self.neo4j.close()
        await self.postgres.close()
    
# Write Operations 
    
    async def remember(self, node: KMem) -> str:
        """
        Write a memory node to the appropriate tier(s).
        
        Routing logic:
        - WORKING → Redis only (TTL = session)
        - EPISODIC → Qdrant (vector) + PostgreSQL (temporal)
        - SEMANTIC → Neo4j (graph)
        - PROCEDURAL → PostgreSQL (rules) + Qdrant (semantic match)
        """
        # Generate embedding if not already set
        if node.embedding is None:
            node.embedding = self.embedder.embed_kmem(node)
        
        if node.memory_type == MemoryType.WORKING:
            await self._write_working(node)
        elif node.memory_type == MemoryType.EPISODIC:
            await self._write_episodic(node)
        elif node.memory_type == MemoryType.SEMANTIC:
            await self._write_semantic(node)
        elif node.memory_type == MemoryType.PROCEDURAL:
            await self._write_procedural(node)
        
        return node.node_id
    
    async def _write_working(self, node: KMem) -> None:
        """Write working memory to Redis."""
        ttl = timedelta(hours=1)  # Default session TTL
        self.redis.set_working_memory(node.session_id, node, ttl)
    
    async def _write_episodic(self, node: KMem) -> None:
        """Write episodic node to Qdrant + PostgreSQL."""
        # Qdrant: vector search index
        self.qdrant.upsert("episodic", node)
        # PostgreSQL: temporal index
        await self.postgres.insert_episodic(node)
    
    async def _write_semantic(self, node: KMem) -> None:
        """Write semantic node to Neo4j knowledge graph."""
        await self.neo4j.create_node(node)
    
    async def _write_procedural(self, node: KMem) -> None:
        """Write procedural node to PostgreSQL + Qdrant."""
        await self.postgres.insert_procedural(node)
        self.qdrant.upsert("procedural", node)

# READ OPERATIONS    

    async def recall(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        org_id: Optional[str] = None,
        time_range_days: Optional[int] = None,
        top_k: int = 10,
        min_salience: float = 0.1
    ) -> List[KMem]:
        """
        Retrieve memories relevant to a query, across all applicable tiers.
        
        The recall pipeline:
        1. Embed the query
        2. Query each relevant tier with scope filters
        3. Merge and deduplicate results
        4. Re-rank by composite score: salience X semantic_similarity
        5. Return top_k results, update access counts
        """
        query_embedding = self.embedder.embed(query)
        
        memory_types = memory_types or [
            MemoryType.EPISODIC,
            MemoryType.SEMANTIC,
            MemoryType.PROCEDURAL
        ]
        
        results = []
        
        if MemoryType.EPISODIC in memory_types:
            episodic = await self._recall_episodic(
                query_embedding, agent_id, user_id, session_id, org_id,
                time_range_days, top_k * 2
            )
            results.extend(episodic)
        
        if MemoryType.SEMANTIC in memory_types:
            semantic = await self._recall_semantic(
                query, agent_id, user_id, org_id, top_k
            )
            results.extend(semantic)
        
        if MemoryType.PROCEDURAL in memory_types:
            procedural = await self._recall_procedural(
                query_embedding, agent_id, user_id, org_id, top_k
            )
            results.extend(procedural)
        
        # Deduplicate and rank
        return self._rank_and_deduplicate(
            results, query_embedding, top_k, min_salience
        )
    
    async def _recall_episodic(
        self,
        query_embedding: np.ndarray,
        agent_id: Optional[str],
        user_id: Optional[str],
        session_id: Optional[str],
        org_id: Optional[str],
        time_range_days: Optional[int],
        limit: int
    ) -> List[KMem]:
        """Retrieve episodic memories from Qdrant with scope + time filters."""
        hits = self.qdrant.search(
            "episodic", query_embedding,
            agent_id=agent_id, user_id=user_id, session_id=session_id,
            org_id=org_id, time_range_days=time_range_days, limit=limit
        )
        return [node for node, _ in hits]
    
    async def _recall_semantic(
        self,
        query: str,
        agent_id: Optional[str],
        user_id: Optional[str],
        org_id: Optional[str],
        limit: int
    ) -> List[KMem]:
        """Retrieve semantic memories from Neo4j using full-text search."""
        hits = await self.neo4j.fulltext_search(query, agent_id or user_id, limit)
        return [node for node, _ in hits]
    
    async def _recall_procedural(
        self,
        query_embedding: np.ndarray,
        agent_id: Optional[str],
        user_id: Optional[str],
        org_id: Optional[str],
        limit: int
    ) -> List[KMem]:
        """Retrieve procedural memories from Qdrant."""
        hits = self.qdrant.search(
            "procedural", query_embedding,
            agent_id=agent_id, user_id=user_id, org_id=org_id, limit=limit
        )
        return [node for node, _ in hits]
    
    def _rank_and_deduplicate(
        self,
        results: List[KMem],
        query_embedding: np.ndarray,
        top_k: int,
        min_salience: float
    ) -> List[KMem]:
        """Deduplicate and rank results by composite score."""
        seen = set()
        unique_results = []
        
        for node in results:
            if node.node_id not in seen:
                seen.add(node.node_id)
                unique_results.append(node)
        
        # Composite ranking: salience × cosine similarity
        now = datetime.utcnow()
        scored = []
        
        for node in unique_results:
            if node.embedding is not None:
                sim = self.embedder.similarity(query_embedding, node.embedding)
            else:
                sim = 0.5
            
            sal = node.salience(now)
            if sal >= min_salience:
                scored.append((node, sal * sim))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        top_results = [node for node, _ in scored[:top_k]]
        
        # Update access counts asynchronously
        for node in top_results:
            node.access()
            asyncio.create_task(self._update_access(node))
        
        return top_results
    
    async def _update_access(self, node: KMem) -> None:
        """Update access statistics in storage."""
        if node.memory_type == MemoryType.EPISODIC:
            await self.postgres.update_access(node.node_id, MemoryType.EPISODIC)
        elif node.memory_type == MemoryType.PROCEDURAL:
            await self.postgres.update_access(node.node_id, MemoryType.PROCEDURAL)
    
    
    async def health_check(self) -> dict:
        """Check health of all storage backends."""
        return {
            "redis": self.redis.health_check(),
            "qdrant": self.qdrant.health_check(),
            "neo4j": await self.neo4j.health_check(),
            "postgres": await self.postgres.health_check()
        }

