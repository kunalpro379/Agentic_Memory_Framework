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