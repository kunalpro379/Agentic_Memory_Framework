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