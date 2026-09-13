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