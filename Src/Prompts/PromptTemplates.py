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