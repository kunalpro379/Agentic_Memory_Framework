"""
Performance benchmarking script.
Tests memory system performance against standard benchmarks.
"""

import asyncio
import time
from Src.Schema.Config import Config
from Src.Manager.MemoryManager import MemoryManager
from Src.Schema.MemoryTypes import KMem, MemoryType


async def benchmark_recall_latency(manager: MemoryManager, queries: list):
    """Benchmark recall latency."""
    latencies = []
    
    for query in queries:
        start = time.time()
        await manager.recall(query, top_k=10)
        latency = time.time() - start
        latencies.append(latency)
    
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    
    return {
        "avg_latency": avg_latency,
        "p95_latency": p95_latency,
        "min_latency": min(latencies),
        "max_latency": max(latencies)
    }


async def benchmark_write_throughput(manager: MemoryManager, count: int):
    """Benchmark write throughput."""
    start = time.time()
    
    for i in range(count):
        node = KMem(
            memory_type=MemoryType.EPISODIC,
            content=f"Benchmark memory {i}",
            agent_id="benchmark_agent"
        )
        await manager.remember(node)
    
    duration = time.time() - start
    throughput = count / duration
    
    return {
        "writes_per_second": throughput,
        "total_duration": duration,
        "total_writes": count
    }


async def run_benchmarks():
    """Run all benchmarks."""
    config = Config.from_env()
    manager = MemoryManager(config)
    await manager.initialize()
    
    print("Running benchmarks...")
    
    # Health check
    health = await manager.health_check()
    print(f"Health check: {health}")
    
    # Recall latency benchmark
    queries = ["client issue", "budget", "deadline", "meeting", "project"]
    recall_stats = await benchmark_recall_latency(manager, queries)
    print(f"\nRecall Latency:")
    print(f"  Average: {recall_stats['avg_latency']:.4f}s")
    print(f"  P95: {recall_stats['p95_latency']:.4f}s")
    
    # Write throughput benchmark
    write_stats = await benchmark_write_throughput(manager, 100)
    print(f"\nWrite Throughput:")
    print(f"  Writes/sec: {write_stats['writes_per_second']:.2f}")
    print(f"  Duration: {write_stats['total_duration']:.4f}s")
    
    await manager.close()


if __name__ == "__main__":
    asyncio.run(run_benchmarks())