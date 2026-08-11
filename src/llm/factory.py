# src/llm/factory.py
from src.llm.base import LLMClient

def get_llm(provider: str = "claude") -> LLMClient:
    if provider == "claude":
        from .claude import ClaudeClient
        return ClaudeClient()
    if provider == "groq":
        from .groq_client import GroqClient
        return GroqClient()
    raise ValueError(f"Unknown provider: {provider}")