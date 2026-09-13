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