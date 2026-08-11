# src/llm/base.py
from abc import ABC, abstractmethod
from typing import Iterator

class LLMClient(ABC):
    @abstractmethod
    def stream(self, system: str, messages: list[dict], max_tokens: int = 1500) -> Iterator[str]:
        """Yield text chunks as they arrive."""
        ...

    @abstractmethod
    def complete(self, system: str, messages: list[dict], max_tokens: int = 1500) -> str:
        """Non-streaming for eval and structured tasks."""
        ...