
"""
Multi-agent memory synchronization bus.
Coordinates memory access across multiple agents.
"""

from typing import Optional, List, Callable

from Src.Schema.MemoryTypes import KMem
from Src.Manager.MemoryManager import MemoryManager


class MultiAgentMemoryBus:
    """
    Coordinates memory access across a multi-agent system.
    
    Memory visibility rules:
    - org_id scope: visible to all agents in the organization
    - user_id scope: visible to all agents serving this user
    - agent_id scope: visible only to the specific agent
    - session_id scope: visible only within the current session
    
    Write propagation:
    - Agent writes to its own scope
    - Consolidation promotes to user or org scope based on generality
    - Coordinator agent can explicitly promote memories to wider scope
    """
    
    def __init__(self, memory_manager: MemoryManager):
        self.mm = memory_manager
        self._subscriptions: dict[str, List[Callable]] = {}
    
    async def broadcast_memory(
        self,
        node: KMem,
        target_scope: str,  # 'agent' | 'user' | 'org'
        source_agent_id: str
    ) -> None:
        """
        Broadcast a memory to a wider scope.
        
        Used when a specialist agent learns something that all agents should know.
        """
        broadcast_node = KMem(
            memory_type=node.memory_type,
            content=node.content,
            structured=node.structured,
            confidence=node.confidence * 0.9,  # Slight confidence reduction
            source=f"broadcast_from:{source_agent_id}",
            parent_nodes=[node.node_id],
            decay_rate=node.decay_rate,
            org_id=node.org_id,
            user_id=node.user_id
        )
        
        if target_scope == 'org':
            broadcast_node.agent_id = None  # Org-wide
        elif target_scope == 'user':
            broadcast_node.agent_id = None  # User-wide
        
        await self.mm.remember(broadcast_node)
        
        # Notify subscribed agents
        for callback in self._subscriptions.get(target_scope, []):
            await callback(broadcast_node)
    
    def subscribe(self, scope: str, callback: Callable) -> None:
        """Subscribe an agent to memory broadcasts at a given scope."""
        self._subscriptions.setdefault(scope, []).append(callback)
    
    async def resolve_conflict(
        self,
        node_a: KMem,
        node_b: KMem,
        resolution_strategy: str = "confidence_weighted"
    ) -> KMem:
        """
        Resolve conflicting memories from different agents.
        
        Strategies:
        - confidence_weighted: weight content by confidence scores
        - recency: prefer the more recent memory
        - authority: prefer the memory from the designated authoritative agent
        """
        if resolution_strategy == "confidence_weighted":
            if node_a.confidence >= node_b.confidence:
                winner = node_a
                loser = node_b
            else:
                winner = node_b
                loser = node_a
            
            # Mark loser as deprecated but don't delete
            loser.status = MemoryStatus.DEPRECATED
            loser.contradicts.append(winner.node_id)
            await self.mm._update_status(loser)
            return winner
        
        elif resolution_strategy == "recency":
            return node_a if node_a.created_at > node_b.created_at else node_b
        
        return node_a  # Fallback