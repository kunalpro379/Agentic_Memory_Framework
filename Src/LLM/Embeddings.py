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