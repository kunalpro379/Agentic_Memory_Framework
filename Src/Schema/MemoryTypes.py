from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from enum import Enum
import uuid
import numpy as np

class MemoryType(Enum):
    EPISODIC   = "episodic"    
    SEMANTIC   = "semantic"  #DOmain Knowledge
    PROCEDURAL = "procedural" #Patterns
    WORKING    = "working"     # Short term context
class MemoryStatus(Enum):
    ACTIVE      = "active"
    CONSOLIDATED = "consolidated"  # episodic → semantic
    DEPRECATED  = "deprecated"   
    ARCHIVED    = "archived" 
@dataclass
class KMem:
    # Identity
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: MemoryType = MemoryType.EPISODIC
    status: MemoryStatus = MemoryStatus.ACTIVE
    # Content
    content: str = ""                       
    structured: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None     # dense vector for similarity search
    # Temporal
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    decay_rate: float = 0.01 #decay older memory
    # Scope
    agent_id: Optional[str] = None  
    session_id: Optional[str] = None 
    user_id: Optional[str] = None 
    org_id: Optional[str] = None     
    # Relational
    parent_nodes: list[str] = field(default_factory=list)   # generalized from
    child_nodes: list[str] = field(default_factory=list)    # more specific than
    related_nodes: list[str] = field(default_factory=list)  # associated with
    # Epistemic
    confidence: float = 1.0       
    source: str = ""                  # where this memory came from
    contradicts: list[str] = field(default_factory=list)  # conflicting node IDs
    def salience(self, current_time: datetime) -> float:
        """
        S(t) = confidence * access_boost * exp(-decay_rate * days_since_access)
        """
        days_since_access = (
            current_time - self.last_accessed
        ).total_seconds() / 86400
        access_boost = 1.0 + 0.5 * np.log1p(self.access_count)
        recency_factor = np.exp(-self.decay_rate * days_since_access)
        return float(self.confidence * access_boost * recency_factor)
    def access(self) -> None:
        self.last_accessed = datetime.utcnow()
        self.access_count += 1