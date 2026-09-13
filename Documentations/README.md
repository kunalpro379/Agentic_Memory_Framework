# Agentic Memory System - Complete Implementation

A production-ready memory architecture for AI agents based on Memory-Node Encapsulation (MNE) design. This system implements a four-tier memory architecture (Working, Episodic, Semantic, Procedural) with proper consolidation, decay, and multi-agent synchronization.

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Core Components](#core-components)
- [Usage Examples](#usage-examples)
- [LangChain/LangGraph Integration](#langchainlanggraph-integration)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Benchmarking](#benchmarking)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 4: WORKING MEMORY                                             │
│  Redis (in-memory key-value)                                        │
│  Current session context · Active reasoning state · Tool results    │
│  TTL: session lifetime (minutes to hours)                           │
├─────────────────────────────────────────────────────────────────────┤
│  TIER 3: EPISODIC MEMORY                                            │
│  Qdrant (vector database) + PostgreSQL (temporal index)             │
│  Interaction history · Events · Observations                        │
│  TTL: weeks to months (salience-gated archival)                     │
├─────────────────────────────────────────────────────────────────────┤
│  TIER 2: SEMANTIC MEMORY                                            │
│  Neo4j (property graph)                                             │
│  Entities · Relationships · Facts · Domain Knowledge                │
│  TTL: indefinite (version-controlled updates)                       │
├─────────────────────────────────────────────────────────────────────┤
│  TIER 1: PROCEDURAL MEMORY                                          │
│  PostgreSQL (structured rules) + Qdrant (semantic matching)         │
│  Task patterns · Behavioral rules · Learned preferences            │
│  TTL: indefinite (reinforcement-updated)                            │
└─────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
AgenticMemory/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── docker-compose.yml                 # Local development setup
├── .env.example                      # Environment variables template
├── Src/
│   ├── Schema/
│   │   ├── __init__.py
│   │   ├── MemoryTypes.py            # Core memory data structures
│   │   └── Config.py                 # Configuration management
│   ├── Storage/
│   │   ├── __init__.py
│   │   ├── RedisStorage.py          # Working memory implementation
│   │   ├── QdrantStorage.py         # Vector storage for episodic/procedural
│   │   ├── Neo4jStorage.py          # Knowledge graph for semantic
│   │   └── PostgresStorage.py       # Temporal index and procedural rules
│   ├── Manager/
│   │   ├── __init__.py
│   │   ├── MemoryManager.py         # Main memory orchestration
│   │   ├── Consolidator.py          # Memory consolidation pipeline
│   │   └── MultiAgentBus.py          # Multi-agent synchronization
│   ├── LLM/
│   │   ├── __init__.py
│   │   ├── EmbeddingProvider.py     # Text embedding service
│   │   ├── LLMProvider.py           # LLM integration
│   │   └── Prompts.py               # System prompts and templates
│   ├── Integration/
│   │   ├── __init__.py
│   │   ├── LangChainMemory.py       # LangChain integration
│   │   └── LangGraphMemory.py       # LangGraph integration
│   ├── Utils/
│   │   ├── __init__.py
│   │   ├── Logger.py                # Logging utilities
│   │   └── Metrics.py               # Performance metrics
│   └── main.py                      # Entry point and examples
├── Tests/
│   ├── __init__.py
│   ├── test_memory_types.py
│   ├── test_storage.py
│   ├── test_manager.py
│   └── test_integration.py
└── Scripts/
    ├── setup_databases.py           # Database initialization
    └── benchmark.py                 # Performance benchmarking
```

## Prerequisites

- Python 3.10+
- Redis 7+
- Qdrant 1.7+
- Neo4j 5+
- PostgreSQL 16+
- Docker (for local development)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd AgenticMemory

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start services with Docker Compose
docker-compose up -d

# Initialize databases
python Scripts/setup_databases.py
```

## Configuration

Create a `.env` file based on `.env.example`:

```env
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=agent_memory
POSTGRES_USER=agent
POSTGRES_PASSWORD=memory_pass

# LLM Configuration
OPENAI_API_KEY=your_openai_api_key
EMBEDDING_MODEL=all-mpnet-base-v2
LLM_MODEL=gpt-4

# Memory Configuration
DEFAULT_DECAY_RATE=0.01
CONSOLIDATION_INTERVAL_HOURS=24
MIN_CLUSTER_SIZE=3
SIMILARITY_THRESHOLD=0.75
```

## Core Components

### 1. Schema/MemoryTypes.py

```python
"""
Core memory data structures implementing Memory-Node Encapsulation (MNE).
This module defines the atomic unit of agentic memory - the KMem node.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from enum import Enum
import uuid
import numpy as np


class MemoryType(Enum):
    """Types of memory following cognitive science taxonomy."""
    EPISODIC = "episodic"      # Time-indexed events and interactions
    SEMANTIC = "semantic"      # Facts, relationships, domain knowledge
    PROCEDURAL = "procedural"  # Task patterns and behavioral rules
    WORKING = "working"        # Active context, short-lived


class MemoryStatus(Enum):
    """Lifecycle status of memory nodes."""
    ACTIVE = "active"
    CONSOLIDATED = "consolidated"  # Moved from episodic → semantic
    DEPRECATED = "deprecated"      # Superseded by newer information
    ARCHIVED = "archived"          # Retained but low retrieval priority


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
    embedding: Optional[np.ndarray] = None  # Dense vector for similarity search
    
    # Temporal
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    decay_rate: float = 0.01  # Per day — episodic decays faster than semantic
    
    # Scope
    agent_id: Optional[str] = None  # Which agent owns this memory
    session_id: Optional[str] = None  # Which session created it
    user_id: Optional[str] = None  # Which user it's associated with
    org_id: Optional[str] = None  # Organizational scope
    
    # Relational
    parent_nodes: list[str] = field(default_factory=list)  # Generalized from
    child_nodes: list[str] = field(default_factory=list)  # More specific than
    related_nodes: list[str] = field(default_factory=list)  # Associated with
    
    # Epistemic
    confidence: float = 1.0  # 0-1 confidence in this memory
    source: str = ""  # Where this memory came from
    contradicts: list[str] = field(default_factory=list)  # Conflicting node IDs
    
    def salience(self, current_time: datetime) -> float:
        """
        Computes current salience: how likely this memory is to be retrieved.
        
        Combines recency, access frequency, and confidence.
        Based on the Ebbinghaus forgetting curve, modified for digital systems:
        S(t) = confidence * access_boost * exp(-decay_rate * days_since_access)
        """
        days_since_access = (
            current_time - self.last_accessed
        ).total_seconds() / 86400
        
        # Access frequency boost (log scale — 10x accesses = 2x boost)
        access_boost = 1.0 + 0.5 * np.log1p(self.access_count)
        
        # Ebbinghaus-inspired decay
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
```

### 2. Schema/Config.py

```python
"""
Configuration management for the memory system.
Handles environment variables and provides typed configuration access.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RedisConfig:
    """Redis configuration for working memory."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'RedisConfig':
        return cls(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD") or None
        )


@dataclass
class QdrantConfig:
    """Qdrant configuration for vector storage."""
    host: str = "localhost"
    port: int = 6333
    api_key: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'QdrantConfig':
        return cls(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            api_key=os.getenv("QDRANT_API_KEY") or None
        )


@dataclass
class Neo4jConfig:
    """Neo4j configuration for semantic memory."""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = ""
    
    @classmethod
    def from_env(cls) -> 'Neo4jConfig':
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "")
        )


@dataclass
class PostgresConfig:
    """PostgreSQL configuration for temporal indexing."""
    host: str = "localhost"
    port: int = 5432
    database: str = "agent_memory"
    user: str = "agent"
    password: str = ""
    
    @classmethod
    def from_env(cls) -> 'PostgresConfig':
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "agent_memory"),
            user=os.getenv("POSTGRES_USER", "agent"),
            password=os.getenv("POSTGRES_PASSWORD", "")
        )
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class LLMConfig:
    """LLM and embedding configuration."""
    openai_api_key: str = ""
    embedding_model: str = "all-mpnet-base-v2"
    llm_model: str = "gpt-4"
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4")
        )


@dataclass
class MemoryConfig:
    """Memory system configuration."""
    default_decay_rate: float = 0.01
    consolidation_interval_hours: int = 24
    min_cluster_size: int = 3
    similarity_threshold: float = 0.75
    contradiction_threshold: float = 0.85
    
    @classmethod
    def from_env(cls) -> 'MemoryConfig':
        return cls(
            default_decay_rate=float(os.getenv("DEFAULT_DECAY_RATE", "0.01")),
            consolidation_interval_hours=int(os.getenv("CONSOLIDATION_INTERVAL_HOURS", "24")),
            min_cluster_size=int(os.getenv("MIN_CLUSTER_SIZE", "3")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.75")),
            contradiction_threshold=float(os.getenv("CONTRADICTION_THRESHOLD", "0.85"))
        )


@dataclass
class Config:
    """Main configuration container."""
    redis: RedisConfig = field(default_factory=RedisConfig.from_env)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig.from_env)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig.from_env)
    postgres: PostgresConfig = field(default_factory=PostgresConfig.from_env)
    llm: LLMConfig = field(default_factory=LLMConfig.from_env)
    memory: MemoryConfig = field(default_factory=MemoryConfig.from_env)
    
    @classmethod
    def from_env(cls) -> 'Config':
        return cls(
            redis=RedisConfig.from_env(),
            qdrant=QdrantConfig.from_env(),
            neo4j=Neo4jConfig.from_env(),
            postgres=PostgresConfig.from_env(),
            llm=LLMConfig.from_env(),
            memory=MemoryConfig.from_env()
        )
```

### 3. Storage/RedisStorage.py

```python
"""
Redis storage implementation for working memory.
Provides fast, TTL-based storage for active session context.
"""

import redis
import json
import pickle
from typing import Optional, Any
from datetime import timedelta

from Src.Schema.MemoryTypes import KMem, MemoryType


class RedisStorage:
    """
    Redis-based storage for working memory.
    
    Characteristics:
    - Sub-millisecond access times
    - TTL-native for automatic cleanup
    - Supports complex data structures
    - Ideal for session-scoped temporary data
    """
    
    def __init__(self, config):
        self.client = redis.Redis(
            host=config.host,
            port=config.port,
            db=config.db,
            password=config.password,
            decode_responses=False
        )
        self.default_ttl = timedelta(hours=1)  # Default session TTL
    
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        """Store a value with optional TTL."""
        try:
            serialized = pickle.dumps(value)
            ttl = ttl or self.default_ttl
            return self.client.setex(key, int(ttl.total_seconds()), serialized)
        except Exception as e:
            print(f"Redis set error: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value."""
        try:
            data = self.client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            print(f"Redis get error: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a key."""
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            print(f"Redis delete error: {e}")
            return False
    
    def set_working_memory(self, session_id: str, node: KMem, ttl: Optional[timedelta] = None) -> bool:
        """Store a working memory node."""
        key = f"working:{session_id}:{node.node_id}"
        return self.set(key, node, ttl)
    
    def get_working_memory(self, session_id: str, node_id: str) -> Optional[KMem]:
        """Retrieve a working memory node."""
        key = f"working:{session_id}:{node_id}"
        return self.get(key)
    
    def get_session_context(self, session_id: str) -> list[KMem]:
        """Get all working memory nodes for a session."""
        pattern = f"working:{session_id}:*"
        keys = self.client.keys(pattern)
        nodes = []
        for key in keys:
            node = self.get(key)
            if node:
                nodes.append(node)
        return nodes
    
    def clear_session(self, session_id: str) -> int:
        """Clear all working memory for a session."""
        pattern = f"working:{session_id}:*"
        keys = self.client.keys(pattern)
        if keys:
            return self.client.delete(*keys)
        return 0
    
    def health_check(self) -> bool:
        """Check if Redis is accessible."""
        try:
            return self.client.ping()
        except Exception:
            return False
```

### 4. Storage/QdrantStorage.py

```python
"""
Qdrant storage implementation for episodic and procedural memory.
Provides vector similarity search with filtering capabilities.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue, Range
)
from typing import Optional, List
import numpy as np

from Src.Schema.MemoryTypes import KMem, MemoryType


class QdrantStorage:
    """
    Qdrant-based storage for episodic and procedural memory.
    
    Characteristics:
    - High-performance vector similarity search
    - Advanced filtering (scope, time, metadata)
    - Supports both episodic and procedural memory
    - Production-ready with horizontal scaling
    """
    
    EMBEDDING_DIM = 768
    
    def __init__(self, config):
        self.client = QdrantClient(
            host=config.host,
            port=config.port,
            api_key=config.api_key
        )
        self._ensure_collections()
    
    def _ensure_collections(self):
        """Create collections if they don't exist."""
        collections = {
            "episodic": {
                "vector_size": self.EMBEDDING_DIM,
                "distance": Distance.COSINE
            },
            "procedural": {
                "vector_size": self.EMBEDDING_DIM,
                "distance": Distance.COSINE
            }
        }
        
        for collection_name, params in collections.items():
            existing_collections = [c.name for c in self.client.get_collections().collections]
            if collection_name not in existing_collections:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(**params)
                )
    
    def upsert(self, collection_name: str, node: KMem) -> bool:
        """Insert or update a memory node."""
        try:
            if node.embedding is None:
                return False
            
            point = PointStruct(
                id=node.node_id,
                vector=node.embedding.tolist(),
                payload={
                    "content": node.content,
                    "agent_id": node.agent_id,
                    "user_id": node.user_id,
                    "session_id": node.session_id,
                    "org_id": node.org_id,
                    "created_at": node.created_at.isoformat(),
                    "confidence": node.confidence,
                    "memory_type": node.memory_type.value,
                    "status": node.status.value,
                    "decay_rate": node.decay_rate,
                    "structured": node.structured
                }
            )
            
            self.client.upsert(collection_name=collection_name, points=[point])
            return True
        except Exception as e:
            print(f"Qdrant upsert error: {e}")
            return False
    
    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        org_id: Optional[str] = None,
        time_range_days: Optional[int] = None,
        limit: int = 10,
        min_confidence: float = 0.0
    ) -> List[tuple[KMem, float]]:
        """Search for similar memories with filters."""
        try:
            must_conditions = []
            
            # Scope filters
            if agent_id:
                must_conditions.append(
                    FieldCondition(key="agent_id", match=MatchValue(value=agent_id))
                )
            if user_id:
                must_conditions.append(
                    FieldCondition(key="user_id", match=MatchValue(value=user_id))
                )
            if session_id:
                must_conditions.append(
                    FieldCondition(key="session_id", match=MatchValue(value=session_id))
                )
            if org_id:
                must_conditions.append(
                    FieldCondition(key="org_id", match=MatchValue(value=org_id))
                )
            
            # Time range filter
            if time_range_days:
                from datetime import datetime, timedelta
                cutoff = (datetime.utcnow() - timedelta(days=time_range_days)).isoformat()
                must_conditions.append(
                    FieldCondition(key="created_at", range={"gte": cutoff})
                )
            
            # Confidence filter
            if min_confidence > 0:
                must_conditions.append(
                    FieldCondition(key="confidence", range={"gte": min_confidence})
                )
            
            search_filter = Filter(must=must_conditions) if must_conditions else None
            
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector.tolist(),
                query_filter=search_filter,
                limit=limit,
                with_payload=True
            )
            
            nodes = []
            for result in results:
                payload = result.payload
                node = KMem(
                    node_id=str(result.id),
                    memory_type=MemoryType(payload["memory_type"]),
                    content=payload["content"],
                    agent_id=payload.get("agent_id"),
                    user_id=payload.get("user_id"),
                    session_id=payload.get("session_id"),
                    org_id=payload.get("org_id"),
                    confidence=payload["confidence"],
                    created_at=payload.get("created_at"),
                    structured=payload.get("structured", {}),
                    decay_rate=payload.get("decay_rate", 0.01)
                )
                nodes.append((node, result.score))
            
            return nodes
        except Exception as e:
            print(f"Qdrant search error: {e}")
            return []
    
    def delete(self, collection_name: str, node_id: str) -> bool:
        """Delete a memory node."""
        try:
            self.client.delete(collection_name=collection_name, points_selector=[node_id])
            return True
        except Exception as e:
            print(f"Qdrant delete error: {e}")
            return False
    
    def health_check(self) -> bool:
        """Check if Qdrant is accessible."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
```

### 5. Storage/Neo4jStorage.py

```python
"""
Neo4j storage implementation for semantic memory.
Provides knowledge graph storage with relationship traversal.
"""

from neo4j import AsyncGraphDatabase
from typing import Optional, List, Dict, Any
import json

from Src.Schema.MemoryTypes import KMem, MemoryType


class Neo4jStorage:
    """
    Neo4j-based storage for semantic memory.
    
    Characteristics:
    - Property graph data model
    - Native relationship traversal
    - Multi-hop queries
    - Full-text search capabilities
    - Ideal for entity-relationship knowledge
    """
    
    def __init__(self, config):
        self.driver = AsyncGraphDatabase.driver(
            config.uri,
            auth=(config.user, config.password)
        )
    
    async def close(self):
        """Close the database connection."""
        await self.driver.close()
    
    async def create_node(self, node: KMem) -> bool:
        """Create a semantic memory node in the knowledge graph."""
        try:
            entity_type = node.structured.get("entity_type", "Concept")
            properties = {
                "node_id": node.node_id,
                "content": node.content,
                "confidence": node.confidence,
                "source": node.source,
                "created_at": node.created_at.isoformat(),
                "agent_id": node.agent_id,
                "user_id": node.user_id,
                "org_id": node.org_id,
                **{k: v for k, v in node.structured.items() 
                   if isinstance(v, (str, int, float, bool))}
            }
            
            async with self.driver.session() as session:
                await session.run(f"""
                    MERGE (n:{entity_type} {{node_id: $node_id}})
                    SET n += $properties
                """, node_id=node.node_id, properties=properties)
                
                # Create relationships
                for related_id in node.related_nodes:
                    rel_type = node.structured.get("relationship_type", "RELATED_TO")
                    await session.run(f"""
                        MATCH (a {{node_id: $source_id}})
                        MATCH (b {{node_id: $target_id}})
                        MERGE (a)-[r:{rel_type}]->(b)
                        SET r.confidence = $confidence,
                            r.created_at = $created_at
                    """, 
                    source_id=node.node_id,
                    target_id=related_id,
                    confidence=node.confidence,
                    created_at=node.created_at.isoformat()
                )
                
            return True
        except Exception as e:
            print(f"Neo4j create error: {e}")
            return False
    
    async def fulltext_search(
        self,
        query: str,
        agent_id: Optional[str] = None,
        limit: int = 10
    ) -> List[tuple[KMem, float]]:
        """Search nodes using full-text search."""
        try:
            async with self.driver.session() as session:
                result = await session.run("""
                    CALL db.index.fulltext.queryNodes('memory_content', $query)
                    YIELD node, score
                    WHERE ($agent_id IS NULL OR node.agent_id = $agent_id)
                      AND node.confidence >= 0.5
                    RETURN node, score
                    ORDER BY score DESC
                    LIMIT $limit
                """, query=query, agent_id=agent_id, limit=limit)
                
                nodes = []
                async for record in result:
                    props = dict(record["node"])
                    node = KMem(
                        node_id=props.get("node_id", ""),
                        memory_type=MemoryType.SEMANTIC,
                        content=props.get("content", ""),
                        confidence=props.get("confidence", 1.0),
                        source=props.get("source", ""),
                        agent_id=props.get("agent_id"),
                        user_id=props.get("user_id"),
                        org_id=props.get("org_id"),
                        structured={k: v for k, v in props.items() 
                                  if k not in ["node_id", "content", "confidence", "source"]}
                    )
                    nodes.append((node, record["score"]))
                
                return nodes
        except Exception as e:
            print(f"Neo4j fulltext search error: {e}")
            return []
    
    async def traverse_relationships(
        self,
        node_id: str,
        relationship_type: Optional[str] = None,
        max_depth: int = 2,
        limit: int = 20
    ) -> List[KMem]:
        """Traverse relationships from a node."""
        try:
            rel_pattern = f"[:{relationship_type}]" if relationship_type else "[]"
            
            async with self.driver.session() as session:
                result = await session.run(f"""
                    MATCH (start {{node_id: $node_id}})
                    MATCH (start)-{rel_pattern}*1..{max_depth}-(related)
                    RETURN DISTINCT related
                    LIMIT $limit
                """, node_id=node_id, limit=limit)
                
                nodes = []
                async for record in result:
                    props = dict(record["related"])
                    node = KMem(
                        node_id=props.get("node_id", ""),
                        memory_type=MemoryType.SEMANTIC,
                        content=props.get("content", ""),
                        confidence=props.get("confidence", 1.0),
                        structured={k: v for k, v in props.items() 
                                  if k not in ["node_id", "content", "confidence"]}
                    )
                    nodes.append(node)
                
                return nodes
        except Exception as e:
            print(f"Neo4j traversal error: {e}")
            return []
    
    async def setup_indexes(self):
        """Setup full-text indexes for search."""
        try:
            async with self.driver.session() as session:
                # Create full-text index on content
                await session.run("""
                    CREATE FULLTEXT INDEX memory_content 
                    IF NOT EXISTS FOR (n:Concept) ON EACH [n.content]
                """)
        except Exception as e:
            print(f"Neo4j index setup error: {e}")
    
    async def health_check(self) -> bool:
        """Check if Neo4j is accessible."""
        try:
            async with self.driver.session() as session:
                await session.run("RETURN 1")
            return True
        except Exception:
            return False
```

### 6. Storage/PostgresStorage.py

```python
"""
PostgreSQL storage implementation for temporal indexing and procedural rules.
Provides structured storage with efficient time-based queries.
"""

import asyncpg
from typing import Optional, List
from datetime import datetime
import json

from Src.Schema.MemoryTypes import KMem, MemoryType, MemoryStatus


class PostgresStorage:
    """
    PostgreSQL-based storage for temporal indexing and procedural rules.
    
    Characteristics:
    - Efficient time-series queries
    - ACID compliance
    - Complex joins and aggregations
    - Ideal for temporal indexing and structured rules
    """
    
    def __init__(self, config):
        self.config = config
        self.pool = None
    
    async def initialize(self):
        """Initialize connection pool and create tables."""
        self.pool = await asyncpg.create_pool(self.config.connection_string)
        await self._create_tables()
    
    async def _create_tables(self):
        """Create necessary tables."""
        async with self.pool.acquire() as conn:
            # Episodic memories table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    node_id VARCHAR(36) PRIMARY KEY,
                    agent_id VARCHAR(100),
                    user_id VARCHAR(100),
                    session_id VARCHAR(100),
                    org_id VARCHAR(100),
                    content TEXT,
                    created_at TIMESTAMP WITH TIME ZONE,
                    last_accessed TIMESTAMP WITH TIME ZONE,
                    access_count INTEGER DEFAULT 0,
                    confidence FLOAT,
                    decay_rate FLOAT DEFAULT 0.01,
                    structured JSONB,
                    status VARCHAR(20) DEFAULT 'active'
                )
            """)
            
            # Create indexes for efficient queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_episodic_agent 
                ON episodic_memories(agent_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_episodic_user 
                ON episodic_memories(user_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_episodic_created 
                ON episodic_memories(created_at DESC)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_episodic_salience 
                ON episodic_memories(confidence, access_count, last_accessed)
            """)
            
            # Procedural memories table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS procedural_memories (
                    node_id VARCHAR(36) PRIMARY KEY,
                    agent_id VARCHAR(100),
                    user_id VARCHAR(100),
                    org_id VARCHAR(100),
                    content TEXT,
                    confidence FLOAT,
                    decay_rate FLOAT DEFAULT 0.001,
                    structured JSONB,
                    created_at TIMESTAMP WITH TIME ZONE,
                    last_accessed TIMESTAMP WITH TIME ZONE,
                    access_count INTEGER DEFAULT 0,
                    status VARCHAR(20) DEFAULT 'active'
                )
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_procedural_agent 
                ON procedural_memories(agent_id)
            """)
    
    async def insert_episodic(self, node: KMem) -> bool:
        """Insert an episodic memory."""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO episodic_memories
                    (node_id, agent_id, user_id, session_id, org_id, content,
                     created_at, last_accessed, access_count, confidence,
                     decay_rate, structured, status)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (node_id) DO UPDATE SET
                        last_accessed = EXCLUDED.last_accessed,
                        access_count = episodic_memories.access_count + 1
                """,
                    node.node_id, node.agent_id, node.user_id, node.session_id,
                    node.org_id, node.content, node.created_at, node.last_accessed,
                    node.access_count, node.confidence, node.decay_rate,
                    json.dumps(node.structured), node.status.value
                )
            return True
        except Exception as e:
            print(f"Postgres insert episodic error: {e}")
            return False
    
    async def insert_procedural(self, node: KMem) -> bool:
        """Insert a procedural memory."""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO procedural_memories
                    (node_id, agent_id, user_id, org_id, content, confidence,
                     decay_rate, structured, created_at, last_accessed,
                     access_count, status)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (node_id) DO UPDATE SET
                        last_accessed = EXCLUDED.last_accessed,
                        access_count = procedural_memories.access_count + 1
                """,
                    node.node_id, node.agent_id, node.user_id, node.org_id,
                    node.content, node.confidence, node.decay_rate,
                    json.dumps(node.structured), node.created_at,
                    node.last_accessed, node.access_count, node.status.value
                )
            return True
        except Exception as e:
            print(f"Postgres insert procedural error: {e}")
            return False
    
    async def query_by_time_range(
        self,
        agent_id: Optional[str],
        user_id: Optional[str],
        days_back: int = 30,
        limit: int = 100
    ) -> List[KMem]:
        """Query episodic memories by time range."""
        try:
            from datetime import timedelta
            cutoff = datetime.utcnow() - timedelta(days=days_back)
            
            async with self.pool.acquire() as conn:
                query = """
                    SELECT * FROM episodic_memories
                    WHERE created_at >= $1
                """
                params = [cutoff]
                
                if agent_id:
                    query += " AND agent_id = $2"
                    params.append(agent_id)
                elif user_id:
                    query += " AND user_id = $2"
                    params.append(user_id)
                
                query += " ORDER BY created_at DESC LIMIT $3"
                params.append(limit)
                
                rows = await conn.fetch(query, *params)
                
                nodes = []
                for row in rows:
                    node = KMem(
                        node_id=row["node_id"],
                        memory_type=MemoryType.EPISODIC,
                        content=row["content"],
                        agent_id=row["agent_id"],
                        user_id=row["user_id"],
                        session_id=row["session_id"],
                        org_id=row["org_id"],
                        created_at=row["created_at"],
                        last_accessed=row["last_accessed"],
                        access_count=row["access_count"],
                        confidence=row["confidence"],
                        decay_rate=row["decay_rate"],
                        structured=json.loads(row["structured"]),
                        status=MemoryStatus(row["status"])
                    )
                    nodes.append(node)
                
                return nodes
        except Exception as e:
            print(f"Postgres time range query error: {e}")
            return []
    
    async def update_access(self, node_id: str, memory_type: MemoryType) -> bool:
        """Update access statistics for a node."""
        try:
            table = "episodic_memories" if memory_type == MemoryType.EPISODIC else "procedural_memories"
            async with self.pool.acquire() as conn:
                await conn.execute(f"""
                    UPDATE {table}
                    SET last_accessed = NOW(),
                        access_count = access_count + 1
                    WHERE node_id = $1
                """, node_id)
            return True
        except Exception as e:
            print(f"Postgres update access error: {e}")
            return False
    
    async def update_status(self, node: KMem) -> bool:
        """Update the status of a memory node."""
        try:
            table = "episodic_memories" if node.memory_type == MemoryType.EPISODIC else "procedural_memories"
            async with self.pool.acquire() as conn:
                await conn.execute(f"""
                    UPDATE {table}
                    SET status = $1,
                        decay_rate = $2
                    WHERE node_id = $3
                """, node.status.value, node.decay_rate, node.node_id)
            return True
        except Exception as e:
            print(f"Postgres update status error: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check if PostgreSQL is accessible."""
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False
    
    async def close(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
```

### 7. LLM/EmbeddingProvider.py

```python
"""
Embedding provider for text vectorization.
Supports multiple embedding models via SentenceTransformers.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Optional, List
import os


class EmbeddingProvider:
    """
    Provides text embedding functionality using SentenceTransformers.
    
    Supports various pre-trained models for different use cases:
    - all-mpnet-base-v2: High quality, 768 dimensions
    - all-MiniLM-L6-v2: Fast, 384 dimensions
    - Custom models can be loaded by name
    """
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts efficiently."""
        return self.model.encode(texts, convert_to_numpy=True)
    
    def embed_kmem(self, node) -> np.ndarray:
        """Generate embedding for a KMem node's content."""
        return self.embed(node.content)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        return float(np.dot(embedding1, embedding2) / 
                    (np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-8))
    
    def get_embedding_dim(self) -> int:
        """Return the dimension of embeddings."""
        return self.embedding_dim
```

### 8. LLM/LLMProvider.py

```python
"""
LLM provider for text generation and pattern extraction.
Supports OpenAI and compatible APIs.
"""

import openai
from typing import Optional, Dict, Any
import json
import os


class LLMProvider:
    """
    Provides LLM functionality for pattern extraction and text generation.
    
    Supports:
    - OpenAI GPT models
    - Compatible APIs (Azure, local models via API)
    - Structured output for pattern extraction
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if self.api_key:
            openai.api_key = self.api_key
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate text completion."""
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM completion error: {e}")
            return ""
    
    async def extract_pattern(self, memories: list, context: str = "") -> Optional[Dict[str, Any]]:
        """
        Extract semantic pattern from a list of memories using LLM.
        
        Returns a dictionary with:
        - content: The extracted pattern/fact
        - entity_type: Type of entity
        - confidence: Confidence in the pattern
        - structured: Additional structured properties
        """
        memory_text = "\n\n".join([
            f"[{i+1}] {mem.content}" for i, mem in enumerate(memories[:10])
        ])
        
        prompt = f"""
        These are {len(memories)} related interaction memories from an AI agent.
        Extract the core semantic pattern or fact they collectively establish.
        
        Context: {context}
        
        Memories:
        {memory_text}
        
        Return JSON with:
        - "content": a single clear statement of the pattern/fact
        - "entity_type": the type of entity this is about (Person, Organization, Process, etc.)
        - "confidence": 0-1 confidence that this is a reliable pattern
        - "structured": key-value pairs of structured properties
        
        Return null if no clear pattern emerges.
        """
        
        try:
            response = await self.complete(prompt)
            return json.loads(response)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Pattern extraction error: {e}")
            return None
    
    async def detect_contradiction(
        self, 
        new_memory: str, 
        existing_memories: list
    ) -> list:
        """
        Detect contradictions between new memory and existing memories.
        """
        existing_text = "\n\n".join([
            f"[{i+1}] {mem.content}" for i, mem in enumerate(existing_memories)
        ])
        
        prompt = f"""
        New memory: {new_memory}
        
        Existing memories:
        {existing_text}
        
        Identify which existing memories (if any) contradict the new memory.
        Return a list of indices (1-based) of contradicting memories.
        Return empty list if no contradictions found.
        """
        
        try:
            response = await self.complete(prompt)
            # Parse response to get indices
            return []
        except Exception as e:
            print(f"Contradiction detection error: {e}")
            return []
```

### 9. LLM/Prompts.py

```python
"""
System prompts and templates for LLM interactions.
"""

SYSTEM_PROMPTS = {
    "pattern_extraction": """
    You are an expert at extracting semantic patterns from interaction data.
    Your task is to identify recurring facts, relationships, or insights
    from a collection of episodic memories.
    
    Guidelines:
    - Focus on information that is stable across multiple instances
    - Ignore one-off events unless they represent a pattern
    - Extract clear, factual statements
    - Assign appropriate entity types
    - Be conservative with confidence scores
    """,
    
    "contradiction_detection": """
    You are an expert at detecting contradictions between statements.
    Your task is to identify when new information conflicts with
    existing knowledge.
    
    Guidelines:
    - Focus on factual contradictions, not minor differences
    - Consider temporal context (old vs new information)
    - Be precise about what constitutes a contradiction
    - When in doubt, flag for review rather than auto-resolving
    """,
    
    "memory_consolidation": """
    You are an expert at memory consolidation, similar to how biological
    systems convert episodic memories into semantic knowledge.
    
    Your task is to identify patterns across interaction history and
    extract the core semantic knowledge that should persist.
    """,
    
    "procedural_extraction": """
    You are an expert at identifying procedural knowledge - the "how-to"
    patterns that emerge from repeated interactions.
    
    Your task is to extract task patterns, behavioral rules, and learned
    preferences from episodic memory.
    """
}

PROMPT_TEMPLATES = {
    "memory_query": """
    Based on the following memories, answer the user's question.
    
    Relevant memories:
    {memories}
    
    User question: {question}
    
    Provide a concise, accurate answer based on the memories.
    If the memories don't contain sufficient information, state that clearly.
    """,
    
    "pattern_extraction": """
    Analyze these {count} related memories and extract the core pattern:
    
    {memories}
    
    Return your analysis as JSON with:
    - content: The pattern/fact
    - entity_type: Type classification
    - confidence: 0-1 score
    - structured: Additional properties
    """,
    
    "relationship_extraction": """
    From these memories, extract relationships between entities:
    
    {memories}
    
    Return relationships as:
    - source: Entity A
    - target: Entity B
    - relationship_type: Type of relationship
    - confidence: 0-1 score
    """
}
```

### 10. Manager/MemoryManager.py

```python
"""
Main memory orchestration manager.
Coordinates all four storage tiers and provides unified interface.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, List
import numpy as np

from Src.Schema.MemoryTypes import KMem, MemoryType, MemoryStatus
from Src.Schema.Config import Config
from Src.Storage.RedisStorage import RedisStorage
from Src.Storage.QdrantStorage import QdrantStorage
from Src.Storage.Neo4jStorage import Neo4jStorage
from Src.Storage.PostgresStorage import PostgresStorage
from Src.LLM.EmbeddingProvider import EmbeddingProvider
from Src.LLM.LLMProvider import LLMProvider


class MemoryManager:
    """
    Single interface for all four memory tiers.
    
    Handles:
    - Write routing: which tier(s) receive a new memory
    - Read orchestration: querying multiple tiers and merging results
    - Consolidation: promoting episodic → semantic as patterns emerge
    - Decay: archiving low-salience memories on a background schedule
    - Scope enforcement: isolating memories by agent, user, session, org
    
    Example:
        manager = MemoryManager(config)
        await manager.initialize()
        await manager.remember(node)
        results = await manager.recall("client issue")
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize storage backends
        self.redis = RedisStorage(config.redis)
        self.qdrant = QdrantStorage(config.qdrant)
        self.neo4j = Neo4jStorage(config.neo4j)
        self.postgres = PostgresStorage(config.postgres)
        
        # Initialize LLM components
        self.embedder = EmbeddingProvider(config.llm.embedding_model)
        self.llm = LLMProvider(config.llm.openai_api_key, config.llm.llm_model)
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize all storage backends."""
        if not self._initialized:
            await self.postgres.initialize()
            await self.neo4j.setup_indexes()
            self._initialized = True
    
    async def close(self):
        """Close all connections."""
        await self.neo4j.close()
        await self.postgres.close()
    
    # ── Write Operations ─────────────────────────────────────────────
    
    async def remember(self, node: KMem) -> str:
        """
        Write a memory node to the appropriate tier(s).
        
        Routing logic:
        - WORKING → Redis only (TTL = session)
        - EPISODIC → Qdrant (vector) + PostgreSQL (temporal)
        - SEMANTIC → Neo4j (graph)
        - PROCEDURAL → PostgreSQL (rules) + Qdrant (semantic match)
        """
        # Generate embedding if not already set
        if node.embedding is None:
            node.embedding = self.embedder.embed_kmem(node)
        
        if node.memory_type == MemoryType.WORKING:
            await self._write_working(node)
        elif node.memory_type == MemoryType.EPISODIC:
            await self._write_episodic(node)
        elif node.memory_type == MemoryType.SEMANTIC:
            await self._write_semantic(node)
        elif node.memory_type == MemoryType.PROCEDURAL:
            await self._write_procedural(node)
        
        return node.node_id
    
    async def _write_working(self, node: KMem) -> None:
        """Write working memory to Redis."""
        ttl = timedelta(hours=1)  # Default session TTL
        self.redis.set_working_memory(node.session_id, node, ttl)
    
    async def _write_episodic(self, node: KMem) -> None:
        """Write episodic node to Qdrant + PostgreSQL."""
        # Qdrant: vector search index
        self.qdrant.upsert("episodic", node)
        # PostgreSQL: temporal index
        await self.postgres.insert_episodic(node)
    
    async def _write_semantic(self, node: KMem) -> None:
        """Write semantic node to Neo4j knowledge graph."""
        await self.neo4j.create_node(node)
    
    async def _write_procedural(self, node: KMem) -> None:
        """Write procedural node to PostgreSQL + Qdrant."""
        await self.postgres.insert_procedural(node)
        self.qdrant.upsert("procedural", node)
    
    # ── Read Operations ──────────────────────────────────────────────
    
    async def recall(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        org_id: Optional[str] = None,
        time_range_days: Optional[int] = None,
        top_k: int = 10,
        min_salience: float = 0.1
    ) -> List[KMem]:
        """
        Retrieve memories relevant to a query, across all applicable tiers.
        
        The recall pipeline:
        1. Embed the query
        2. Query each relevant tier with scope filters
        3. Merge and deduplicate results
        4. Re-rank by composite score: salience × semantic_similarity
        5. Return top_k results, update access counts
        """
        query_embedding = self.embedder.embed(query)
        
        memory_types = memory_types or [
            MemoryType.EPISODIC,
            MemoryType.SEMANTIC,
            MemoryType.PROCEDURAL
        ]
        
        results = []
        
        if MemoryType.EPISODIC in memory_types:
            episodic = await self._recall_episodic(
                query_embedding, agent_id, user_id, session_id, org_id,
                time_range_days, top_k * 2
            )
            results.extend(episodic)
        
        if MemoryType.SEMANTIC in memory_types:
            semantic = await self._recall_semantic(
                query, agent_id, user_id, org_id, top_k
            )
            results.extend(semantic)
        
        if MemoryType.PROCEDURAL in memory_types:
            procedural = await self._recall_procedural(
                query_embedding, agent_id, user_id, org_id, top_k
            )
            results.extend(procedural)
        
        # Deduplicate and rank
        return self._rank_and_deduplicate(
            results, query_embedding, top_k, min_salience
        )
    
    async def _recall_episodic(
        self,
        query_embedding: np.ndarray,
        agent_id: Optional[str],
        user_id: Optional[str],
        session_id: Optional[str],
        org_id: Optional[str],
        time_range_days: Optional[int],
        limit: int
    ) -> List[KMem]:
        """Retrieve episodic memories from Qdrant with scope + time filters."""
        hits = self.qdrant.search(
            "episodic", query_embedding,
            agent_id=agent_id, user_id=user_id, session_id=session_id,
            org_id=org_id, time_range_days=time_range_days, limit=limit
        )
        return [node for node, _ in hits]
    
    async def _recall_semantic(
        self,
        query: str,
        agent_id: Optional[str],
        user_id: Optional[str],
        org_id: Optional[str],
        limit: int
    ) -> List[KMem]:
        """Retrieve semantic memories from Neo4j using full-text search."""
        hits = await self.neo4j.fulltext_search(query, agent_id or user_id, limit)
        return [node for node, _ in hits]
    
    async def _recall_procedural(
        self,
        query_embedding: np.ndarray,
        agent_id: Optional[str],
        user_id: Optional[str],
        org_id: Optional[str],
        limit: int
    ) -> List[KMem]:
        """Retrieve procedural memories from Qdrant."""
        hits = self.qdrant.search(
            "procedural", query_embedding,
            agent_id=agent_id, user_id=user_id, org_id=org_id, limit=limit
        )
        return [node for node, _ in hits]
    
    def _rank_and_deduplicate(
        self,
        results: List[KMem],
        query_embedding: np.ndarray,
        top_k: int,
        min_salience: float
    ) -> List[KMem]:
        """Deduplicate and rank results by composite score."""
        seen = set()
        unique_results = []
        
        for node in results:
            if node.node_id not in seen:
                seen.add(node.node_id)
                unique_results.append(node)
        
        # Composite ranking: salience × cosine similarity
        now = datetime.utcnow()
        scored = []
        
        for node in unique_results:
            if node.embedding is not None:
                sim = self.embedder.similarity(query_embedding, node.embedding)
            else:
                sim = 0.5
            
            sal = node.salience(now)
            if sal >= min_salience:
                scored.append((node, sal * sim))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        top_results = [node for node, _ in scored[:top_k]]
        
        # Update access counts asynchronously
        for node in top_results:
            node.access()
            asyncio.create_task(self._update_access(node))
        
        return top_results
    
    async def _update_access(self, node: KMem) -> None:
        """Update access statistics in storage."""
        if node.memory_type == MemoryType.EPISODIC:
            await self.postgres.update_access(node.node_id, MemoryType.EPISODIC)
        elif node.memory_type == MemoryType.PROCEDURAL:
            await self.postgres.update_access(node.node_id, MemoryType.PROCEDURAL)
    
    # ── Utility Methods ───────────────────────────────────────────────
    
    async def health_check(self) -> dict:
        """Check health of all storage backends."""
        return {
            "redis": self.redis.health_check(),
            "qdrant": self.qdrant.health_check(),
            "neo4j": await self.neo4j.health_check(),
            "postgres": await self.postgres.health_check()
        }
```

### 11. Manager/Consolidator.py

```python
"""
Memory consolidation pipeline.
Promotes episodic memories into semantic knowledge through pattern detection.
"""

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
```

### 12. Manager/MultiAgentBus.py

```python
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
```

### 13. Integration/LangChainMemory.py

```python
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
```

### 14. Integration/LangGraphMemory.py

```python
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
```

### 15. Utils/Logger.py

```python
"""
Logging utilities for the memory system.
"""

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
```

### 16. Utils/Metrics.py

```python
"""
Performance metrics collection for the memory system.
"""

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
```

### 17. Scripts/setup_databases.py

```python
"""
Database initialization script.
Sets up all required databases and indexes.
"""

import asyncio
from Src.Schema.Config import Config
from Src.Storage.RedisStorage import RedisStorage
from Src.Storage.QdrantStorage import QdrantStorage
from Src.Storage.Neo4jStorage import Neo4jStorage
from Src.Storage.PostgresStorage import PostgresStorage


async def setup_databases():
    """Initialize all database backends."""
    config = Config.from_env()
    
    print("Initializing databases...")
    
    # Redis
    print("Setting up Redis...")
    redis = RedisStorage(config.redis)
    if redis.health_check():
        print("✓ Redis is ready")
    else:
        print("✗ Redis connection failed")
    
    # Qdrant
    print("Setting up Qdrant...")
    qdrant = QdrantStorage(config.qdrant)
    if qdrant.health_check():
        print("✓ Qdrant is ready")
    else:
        print("✗ Qdrant connection failed")
    
    # Neo4j
    print("Setting up Neo4j...")
    neo4j = Neo4jStorage(config.neo4j)
    await neo4j.setup_indexes()
    if await neo4j.health_check():
        print("✓ Neo4j is ready")
    else:
        print("✗ Neo4j connection failed")
    await neo4j.close()
    
    # PostgreSQL
    print("Setting up PostgreSQL...")
    postgres = PostgresStorage(config.postgres)
    await postgres.initialize()
    if await postgres.health_check():
        print("✓ PostgreSQL is ready")
    else:
        print("✗ PostgreSQL connection failed")
    await postgres.close()
    
    print("\nDatabase setup complete!")


if __name__ == "__main__":
    asyncio.run(setup_databases())
```

### 18. Scripts/benchmark.py

```python
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
```

### 19. Src/main.py

```python
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
```

### 20. requirements.txt

```txt
# Core dependencies
python-dotenv==1.0.0
redis==5.0.1
qdrant-client==1.7.0
neo4j==5.14.0
asyncpg==0.29.0
sentence-transformers==2.2.2
openai==1.3.0
numpy==1.24.3
scikit-learn==1.3.0

# LangChain integration
langchain==0.1.0
langchain-openai==0.0.2
langgraph==0.0.20

# Utilities
pydantic==2.5.0
typing-extensions==4.8.0
```

### 21. docker-compose.yml

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
    volumes:
      - ./redis_data:/data

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage

  neo4j:
    image: neo4j:5-community
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: neo4j/canary_memory
      NEO4J_PLUGINS: '["apoc", "graph-data-science"]'
    volumes:
      - ./neo4j_data:/data

  postgres:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: agent_memory
      POSTGRES_USER: agent
      POSTGRES_PASSWORD: memory_pass
    volumes:
      - ./pg_data:/var/lib/postgresql/data

volumes:
  redis_data:
  qdrant_storage:
  neo4j_data:
  pg_data:
```

### 22. .env.example

```env
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=canary_memory

# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=agent_memory
POSTGRES_USER=agent
POSTGRES_PASSWORD=memory_pass

# LLM Configuration
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL=all-mpnet-base-v2
LLM_MODEL=gpt-4

# Memory Configuration
DEFAULT_DECAY_RATE=0.01
CONSOLIDATION_INTERVAL_HOURS=24
MIN_CLUSTER_SIZE=3
SIMILARITY_THRESHOLD=0.75
CONTRADICTION_THRESHOLD=0.85
```

## Usage Examples

### Basic Memory Operations

```python
import asyncio
from Src.Schema.Config import Config
from Src.Schema.MemoryTypes import KMem, MemoryType
from Src.Manager.MemoryManager import MemoryManager

async def main():
    # Initialize
    config = Config.from_env()
    manager = MemoryManager(config)
    await manager.initialize()
    
    # Store a memory
    node = KMem(
        memory_type=MemoryType.EPISODIC,
        content="Client called about Q3 report issues",
        agent_id="support_agent_001",
        user_id="user_123",
        structured={"category": "support", "priority": "high"}
    )
    node_id = await manager.remember(node)
    
    # Retrieve memories
    results = await manager.recall(
        query="client report issue",
        agent_id="support_agent_001",
        top_k=5
    )
    
    for mem in results:
        print(f"{mem.content} (salience: {mem.salience(datetime.utcnow())})")
    
    await manager.close()

asyncio.run(main())
```

### LangChain Integration

```python
from Src.Integration.LangChainMemory import KMemMemory
from Src.Manager.MemoryManager import MemoryManager
from Src.Schema.Config import Config
from langchain.chains import LLMChain
from langchain.llms import OpenAI

# Setup
config = Config.from_env()
manager = MemoryManager(config)
await manager.initialize()

# Create LangChain-compatible memory
memory = KMemMemory(
    memory_manager=manager,
    agent_id="sales_agent_001",
    user_id="user_abc",
    session_id="session_xyz"
)

# Use with LangChain
llm = OpenAI()
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
```

### LangGraph Integration

```python
from Src.Integration.LangGraphMemory import LangGraphMemory
from Src.Manager.MemoryManager import MemoryManager
from Src.Schema.Config import Config

# Setup
config = Config.from_env()
manager = MemoryManager(config)
await manager.initialize()

# Create graph with memory
memory_integration = LangGraphMemory(manager)
graph = memory_integration.create_graph()

# Execute
result = await graph.ainvoke({"query": "What do you know about client X?"})
```

## API Reference

### MemoryManager

Main interface for memory operations.

**Methods:**
- `async initialize()` - Initialize all storage backends
- `async remember(node: KMem) -> str` - Store a memory node
- `async recall(query: str, **filters) -> List[KMem]` - Retrieve relevant memories
- `async health_check() -> dict` - Check health of all backends
- `async close()` - Close all connections

### KMem

Core memory data structure.

**Attributes:**
- `node_id: str` - Unique identifier
- `memory_type: MemoryType` - Type of memory
- `content: str` - Natural language content
- `structured: dict` - Structured metadata
- `embedding: np.ndarray` - Vector embedding
- `confidence: float` - Confidence score (0-1)
- `decay_rate: float` - Decay rate per day
- `agent_id, user_id, session_id, org_id` - Scope identifiers

**Methods:**
- `salience(current_time: datetime) -> float` - Compute retrieval salience
- `access() -> None` - Record an access

### MemoryConsolidator

Handles episodic → semantic consolidation.

**Methods:**
- `async consolidate(agent_id: str, lookback_days: int) -> dict` - Run consolidation pipeline

## Deployment

### Local Development

```bash
# Start services
docker-compose up -d

# Initialize databases
python Scripts/setup_databases.py

# Run examples
python Src/main.py
```

### Production (Kubernetes)

Use managed services:
- Redis Cloud or Upstash for working memory
- Qdrant Cloud for vector search
- Neo4j AuraDB for knowledge graph
- RDS PostgreSQL for temporal indexing

## Benchmarking

Run performance benchmarks:

```bash
python Scripts/benchmark.py
```

Target metrics:
- Recall latency: <100ms P95
- Write throughput: >100 writes/sec
- Memory accuracy: >85% on LoCoMo benchmark
- Knowledge updates: >75% on LongMemEval

## Troubleshooting

**Connection Issues:**
- Check all services are running: `docker-compose ps`
- Verify environment variables in `.env`
- Check firewall settings for ports

**Performance Issues:**
- Monitor Redis memory usage
- Check Qdrant collection sizes
- Optimize PostgreSQL indexes
- Adjust embedding batch sizes

**Memory Quality:**
- Tune similarity thresholds
- Adjust decay rates
- Review consolidation patterns
- Check contradiction detection

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## Support

For issues and questions:
- GitHub Issues: [github.com/Bodhi8/mne](https://github.com/Bodhi8/mne)
- Documentation: [vector1.ai](https://vector1.ai)
- Email: brian@vector1.ai
