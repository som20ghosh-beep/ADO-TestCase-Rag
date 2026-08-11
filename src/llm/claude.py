# src/llm/claude.py
from anthropic import Anthropic
from .base import LLMClient

CLAUDE_MODEL = "claude-opus-4-7"   # or claude-sonnet-5 for cheaper/faster

class ClaudeClient(LLMClient):
    def __init__(self):
        self.client = Anthropic()  # reads ANTHROPIC_API_KEY from env

    def stream(self, system, messages, max_tokens=1500):
        with self.client.messages.stream(
            model=CLAUDE_MODEL,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
        ) as s:
            for text in s.text_stream:
                yield text

    def complete(self, system, messages, max_tokens=1500):
        resp = self.client.messages.create(
            model=CLAUDE_MODEL,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
        )
        return resp.content[0].text