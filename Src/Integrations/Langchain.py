
"""
LangChain integration for the memory system.
Provides a drop-in replacement for LangChain's BaseMemory.
"""
from langchain.memory import BaseMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from typing import Dict, List, Any, Optional
import asyncio

from Src.Schema.MemoryTypes import KMem, MemoryType
from Src.Manager.MemoryManager import MemoryManager


class KMemMemory(BaseMemory):
    """
    LangChain-compatible memory class backed by the full KMem architecture.
    
    Replaces LangChain's built-in memory types with the four-tier
    KMem system. Drop-in replacement — existing LangChain chains and
    agents work without modification.
    
    Usage:
        memory = KMemMemory(
            memory_manager=manager,
            agent_id="sales_agent_001",
            user_id="user_abc",
            session_id="session_xyz"
        )
        chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    """
    
    memory_manager: MemoryManager
    agent_id: str
    user_id: str
    session_id: str
    memory_key: str = "chat_history"
    return_messages: bool = True
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load relevant memories for the current input.
        Called by LangChain before each LLM invocation.
        """
        query = inputs.get("input", inputs.get("question", ""))
        
        # Run async recall in sync context
        loop = asyncio.get_event_loop()
        memories = loop.run_until_complete(
            self.memory_manager.recall(
                query=query,
                agent_id=self.agent_id,
                user_id=self.user_id,
                session_id=self.session_id,
                top_k=8
            )
        )
        
        # Format as LangChain messages
        messages = []
        for mem in memories:
            if mem.structured.get("role") == "human":
                messages.append(HumanMessage(content=mem.content))
            else:
                messages.append(AIMessage(content=mem.content))
        
        return {self.memory_key: messages}
    
    def save_context(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, str]
    ) -> None:
        """
        Save the current interaction to episodic memory.
        Called by LangChain after each LLM invocation.
        """
        loop = asyncio.get_event_loop()
        
        human_input = inputs.get("input", inputs.get("question", ""))
        ai_output = outputs.get("output", outputs.get("response", ""))
        
        # Save human turn
        human_node = KMem(
            memory_type=MemoryType.EPISODIC,
            content=human_input,
            agent_id=self.agent_id,
            user_id=self.user_id,
            session_id=self.session_id,
            structured={"role": "human"},
            decay_rate=0.02
        )
        loop.run_until_complete(self.memory_manager.remember(human_node))
        
        # Save AI turn
        ai_node = KMem(
            memory_type=MemoryType.EPISODIC,
            content=ai_output,
            agent_id=self.agent_id,
            user_id=self.user_id,
            session_id=self.session_id,
            structured={"role": "ai"},
            decay_rate=0.02,
            parent_nodes=[human_node.node_id]
        )
        loop.run_until_complete(self.memory_manager.remember(ai_node))
    
    def clear(self) -> None:
        """Clear working memory for the current session."""
        self.memory_manager.redis.clear_session(self.session_id)
