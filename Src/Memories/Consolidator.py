from datetime import datetime, timedelta
from typing import Optional, List
import numpy as np
from sklearn.cluster import DBSCAN

from Src.Schema.MemoryTypes import KMem, MemoryType, MemoryStatus
from Src.Manager.MemoryManager import MemoryManager


class MemoryConsolidator:
    """
    Promotes episodic memories into semantic knowledge through pattern detection.
    
    The consolidation pipeline:
    1. Cluster recent episodic memories by semantic similarity
    2. For each cluster above the density threshold, extract the common pattern
    3. Check if the pattern contradicts existing semantic memories
    4. If novel and consistent: create a new semantic KMem node
    5. If contradictory: flag both nodes for human review or update semantic
    6. Archive the consolidated episodic nodes (reduce salience, don't delete)
    
    Runs as a scheduled background task — typically nightly or hourly
    for high-volume agents.
    """
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        min_cluster_size: int = 3,
        similarity_threshold: float = 0.75
    ):
        self.mm = memory_manager
        self.min_cluster_size = min_cluster_size
        self.sim_threshold = similarity_threshold
    
    async def consolidate(
        self,
        agent_id: str,
        lookback_days: int = 7
    ) -> dict:
        """
        Run the consolidation pipeline for a specific agent.
        
        Returns statistics on consolidation results.
        """
        # 1. Fetch recent episodic memories
        recent_episodic = await self._fetch_recent_episodic(
            agent_id, lookback_days
        )
        
        if len(recent_episodic) < self.min_cluster_size:
            return {
                "status": "skipped",
                "reason": "insufficient episodic memories"
            }
        
        # 2. Cluster by semantic similarity
        clusters = self._cluster_episodic(recent_episodic)
        
        consolidated_count = 0
        contradiction_count = 0
        
        for cluster in clusters:
            if len(cluster) < self.min_cluster_size:
                continue
            
            # 3. Extract semantic pattern from cluster
            pattern = await self._extract_pattern(cluster)
            if not pattern:
                continue
            
            # 4. Check for contradictions with existing semantic memory
            existing = await self.mm.recall(
                query=pattern['content'],
                memory_types=[MemoryType.SEMANTIC],
                agent_id=agent_id,
                top_k=3
            )
            
            contradiction = self._detect_contradiction(pattern, existing)
            
            if contradiction:
                # Flag for review rather than auto-updating
                await self._flag_contradiction(pattern, contradiction)
                contradiction_count += 1
            else:
                # Create new semantic memory node
                semantic_node = KMem(
                    memory_type=MemoryType.SEMANTIC,
                    content=pattern['content'],
                    structured=pattern.get('structured', {}),
                    confidence=pattern.get('confidence', 0.8),
                    agent_id=agent_id,
                    source="consolidation",
                    parent_nodes=[n.node_id for n in cluster],
                    decay_rate=0.001  # Semantic memory decays very slowly
                )
                await self.mm.remember(semantic_node)
                consolidated_count += 1
                
                # Archive source episodic nodes
                for episodic_node in cluster:
                    episodic_node.status = MemoryStatus.CONSOLIDATED
                    episodic_node.decay_rate *= 3.0  # Accelerate decay
                    await self.mm._update_status(episodic_node)
        
        return {
            "status": "completed",
            "episodic_processed": len(recent_episodic),
            "clusters_found": len(clusters),
            "consolidated": consolidated_count,
            "contradictions_flagged": contradiction_count
        }
    
    async def _fetch_recent_episodic(
        self,
        agent_id: str,
        lookback_days: int
    ) -> List[KMem]:
        """Fetch recent episodic memories for consolidation."""
        return await self.mm.postgres.query_by_time_range(
            agent_id=agent_id,
            days_back=lookback_days,
            limit=1000
        )
    
    def _cluster_episodic(self, nodes: List[KMem]) -> List[List[KMem]]:
        """Cluster episodic nodes by embedding similarity."""
        if not nodes:
            return []
        
        embeddings = np.array([
            node.embedding for node in nodes
            if node.embedding is not None
        ])
        
        if len(embeddings) < 2:
            return [nodes]
        
        clustering = DBSCAN(
            eps=1 - self.sim_threshold,  # Cosine distance threshold
            min_samples=self.min_cluster_size,
            metric='cosine'
        ).fit(embeddings)
        
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label == -1:  # Noise point
                continue
            clusters.setdefault(label, []).append(nodes[i])
        
        return list(clusters.values())
    
    async def _extract_pattern(self, cluster: List[KMem]) -> Optional[dict]:
        """
        Use LLM to extract the common semantic pattern from a cluster
        of episodic memories.
        """
        return await self.mm.llm.extract_pattern(cluster)
    
    def _detect_contradiction(
        self,
        pattern: dict,
        existing: List[KMem]
    ) -> bool:
        """Detect if pattern contradicts existing semantic memories."""
        # Simple implementation: check for high similarity with different content
        pattern_content = pattern['content'].lower()
        
        for existing_node in existing:
            existing_content = existing_node.content.lower()
            # If very similar but different, potential contradiction
            if (pattern_content != existing_content and 
                len(set(pattern_content.split()) & set(existing_content.split())) > 3):
                return True
        
        return False
    
    async def _flag_contradiction(
        self,
        pattern: dict,
        existing: List[KMem]
    ) -> None:
        """Flag contradictory memories for review."""
        # In production, this would send to a review queue
        print(f"Contradiction flagged: {pattern['content']}")
        for node in existing:
            print(f"  Conflicts with: {node.content}")