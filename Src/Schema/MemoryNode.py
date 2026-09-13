@dataclass
class KMem:
    """
        Memory-Node Encapsulation: the atomic unit of agentic memory.
        Each node represents a single memory — an event, a fact, a procedure,
        or a working context item — with full provenance, temporal indexing,
        relationship pointers, and a decay model.
        The four components:
        1. Content: the memory itself (text + structured metadata)
        2. Temporal: when it was created, last accessed, and how it decays
        3. Relational: connections to other nodes in the memory graph
        4. Epistemic: confidence, source quality, and contradiction flags
    """

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
        """Record an access — updates recency and count."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "memory_type": self.memory_type.value,
            "status": self.status.value,
            "content": self.content,
            "structured": self.structured,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "decay_rate": self.decay_rate,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "org_id": self.org_id,
            "parent_nodes": self.parent_nodes,
            "child_nodes": self.child_nodes,
            "related_nodes": self.related_nodes,
            "confidence": self.confidence,
            "source": self.source,
            "contradicts": self.contradicts
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'KMem':
        """Create KMem from dictionary."""
        embedding = np.array(data["embedding"]) if data.get("embedding") else None
        return cls(
            node_id=data["node_id"],
            memory_type=MemoryType(data["memory_type"]),
            status=MemoryStatus(data["status"]),
            content=data["content"],
            structured=data["structured"],
            embedding=embedding,
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data["access_count"],
            decay_rate=data["decay_rate"],
            agent_id=data.get("agent_id"),
            session_id=data.get("session_id"),
            user_id=data.get("user_id"),
            org_id=data.get("org_id"),
            parent_nodes=data["parent_nodes"],
            child_nodes=data["child_nodes"],
            related_nodes=data["related_nodes"],
            confidence=data["confidence"],
            source=data["source"],
            contradicts=data["contradicts"]
        )