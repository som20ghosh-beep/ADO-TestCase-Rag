# src/llm/groq_client.py
import os
from groq import Groq
from .base import LLMClient

GROQ_MODEL = "llama-3.3-70b-versatile"

class GroqClient(LLMClient):
    def __init__(self):
        self.client = Groq(api_key=os.environ["GROQ_KEY"])

    def stream(self, system, messages, max_tokens=1500):
        stream = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system}, *messages],
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def complete(self, system, messages, max_tokens=1500):
        resp = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system}, *messages],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
