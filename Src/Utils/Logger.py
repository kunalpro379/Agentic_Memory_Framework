
import logging
from typing import Optional
from datetime import datetime


class MemoryLogger:
    """Centralized logging for memory operations."""
    
    def __init__(self, name: str = "AgenticMemory", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_write(self, node_id: str, memory_type: str, tier: str):
        """Log a write operation."""
        self.logger.info(f"WRITE: {node_id} | Type: {memory_type} | Tier: {tier}")
    
    def log_read(self, query: str, results_count: int, tiers: list):
        """Log a read operation."""
        self.logger.info(
            f"READ: Query='{query[:50]}...' | Results: {results_count} | Tiers: {tiers}"
        )
    
    def log_consolidation(self, stats: dict):
        """Log consolidation results."""
        self.logger.info(f"CONSOLIDATION: {stats}")
    
    def log_error(self, operation: str, error: str):
        """Log an error."""
        self.logger.error(f"ERROR in {operation}: {error}")
