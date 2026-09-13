"""
Main entry point with usage examples.
"""

import asyncio
from Src.Schema.Config import Config
from Src.Schema.MemoryTypes import KMem, MemoryType
from Src.Manager.MemoryManager import MemoryManager
from Src.Manager.Consolidator import MemoryConsolidator
from Src.Integration.LangChainMemory import KMemMemory


async def basic_usage_example():
    """Basic usage example."""
    print("=== Basic Usage Example ===\n")
    
    # Initialize
    config = Config.from_env()
    manager = MemoryManager(config)
    await manager.initialize()
    
    # Create and store a memory
    node = KMem(
        memory_type=MemoryType.EPISODIC,
        content="Client called about Q3 report issues",
        agent_id="support_agent_001",
        user_id="user_123",
        session_id="session_abc",
        structured={"category": "support", "priority": "high"},
        decay_rate=0.02
    )
    
    node_id = await manager.remember(node)
    print(f"Stored memory: {node_id}")
    
    # Retrieve memories
    results = await manager.recall(
        query="client report issue",
        agent_id="support_agent_001",
        top_k=5
    )
    
    print(f"\nRetrieved {len(results)} memories:")
    for mem in results:
        print(f"  - {mem.content[:60]}... (salience: {mem.salience(manager.embedder)})")
    
    await manager.close()


async def consolidation_example():
    """Memory consolidation example."""
    print("\n=== Consolidation Example ===\n")
    
    config = Config.from_env()
    manager = MemoryManager(config)
    await manager.initialize()
    
    # Add multiple similar episodic memories
    for i in range(5):
        node = KMem(
            memory_type=MemoryType.EPISODIC,
            content=f"Client prefers email communication for project updates {i}",
            agent_id="support_agent_001",
            user_id="user_123",
            decay_rate=0.02
        )
        await manager.remember(node)
    
    # Run consolidation
    consolidator = MemoryConsolidator(manager)
    results = await consolidator.consolidate(agent_id="support_agent_001")
    
    print(f"Consolidation results: {results}")
    
    await manager.close()


async def langchain_integration_example():
    """LangChain integration example."""
    print("\n=== LangChain Integration Example ===\n")
    
    config = Config.from_env()
    manager = MemoryManager(config)
    await manager.initialize()
    
    # Create LangChain-compatible memory
    memory = KMemMemory(
        memory_manager=manager,
        agent_id="langchain_agent",
        user_id="user_123",
        session_id="session_xyz"
    )
    
    # Use with LangChain
    # memory.save_context({"input": "Hello"}, {"output": "Hi there!"})
    # loaded = memory.load_memory_variables({"input": "Hello"})
    
    print("LangChain memory integration ready")
    print(f"Memory variables: {memory.memory_variables}")
    
    await manager.close()


async def multi_agent_example():
    """Multi-agent memory synchronization example."""
    print("\n=== Multi-Agent Example ===\n")
    
    from Src.Manager.MultiAgentBus import MultiAgentMemoryBus
    
    config = Config.from_env()
    manager = MemoryManager(config)
    await manager.initialize()
    
    bus = MultiAgentMemoryBus(manager)
    
    # Agent A learns something
    node_a = KMem(
        memory_type=MemoryType.SEMANTIC,
        content="Client budget is $50K for Q4",
        agent_id="agent_a",
        user_id="user_123",
        confidence=0.9
    )
    await manager.remember(node_a)
    
    # Broadcast to organization
    await bus.broadcast_memory(node_a, "org", "agent_a")
    
    print("Multi-agent memory broadcast complete")
    
    await manager.close()


async def main():
    """Run all examples."""
    await basic_usage_example()
    await consolidation_example()
    await langchain_integration_example()
    await multi_agent_example()


if __name__ == "__main__":
    asyncio.run(main())
