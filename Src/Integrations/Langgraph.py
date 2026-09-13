"""
LangGraph integration for the memory system.
Provides memory state management for LangGraph workflows.
"""

from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END

from Src.Schema.MemoryTypes import KMem, MemoryType
from Src.Manager.MemoryManager import MemoryManager


class MemoryState(TypedDict):
    """State for LangGraph memory integration."""
    query: str
    memories: Annotated[Sequence[KMem], operator.add]
    response: str



class LangGraphMemory:
    """
    LangGraph integration for memory system.
    
    Provides stateful memory management within LangGraph workflows,
    enabling agents to maintain context across graph executions.
    
    Usage:
        memory_integration = LangGraphMemory(memory_manager)
        graph = memory_integration.create_graph()
    """
    
    def __init__(self, memory_manager: MemoryManager):
        self.mm = memory_manager
    
    async def retrieve_memories(self, state: MemoryState) -> MemoryState:
        """Retrieve relevant memories based on query."""
        memories = await self.mm.recall(
            query=state["query"],
            top_k=5
        )
        return {"memories": memories}
    
    async def generate_response(self, state: MemoryState) -> MemoryState:
        """Generate response using retrieved memories."""
        # This would integrate with your LLM
        memory_context = "\n".join([m.content for m in state["memories"]])
        response = f"Based on {len(state['memories'])} memories: {memory_context}"
        return {"response": response}
    
    async def save_interaction(self, state: MemoryState) -> MemoryState:
        """Save the interaction to episodic memory."""
        node = KMem(
            memory_type=MemoryType.EPISODIC,
            content=f"Q: {state['query']} A: {state['response']}",
            agent_id="langgraph_agent",
            decay_rate=0.02
        )
        await self.mm.remember(node)
        return state
    
    def create_graph(self) -> StateGraph:
        """Create a LangGraph with memory integration."""
        workflow = StateGraph(MemoryState)
        
        workflow.add_node("retrieve", self.retrieve_memories)
        workflow.add_node("respond", self.generate_response)
        workflow.add_node("save", self.save_interaction)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "respond")
        workflow.add_edge("respond", "save")
        workflow.add_edge("save", END)
        
        return workflow.compile()
