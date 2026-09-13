import time
from typing import Dict, List
from collections import defaultdict


class MemoryMetrics:
    """Collect and track performance metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
    
    def record_latency(self, operation: str, duration: float):
        """Record operation latency."""
        self.metrics[f"{operation}_latency"].append(duration)
    
    def record_throughput(self, operation: str, count: int):
        """Record operation throughput."""
        self.counters[f"{operation}_count"] += count
    
    def get_stats(self) -> Dict:
        """Get statistics summary."""
        stats = {}
        for key, values in self.metrics.items():
            if values:
                stats[key] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        stats.update(dict(self.counters))
        return stats
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.counters.clear()