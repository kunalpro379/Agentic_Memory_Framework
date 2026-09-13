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