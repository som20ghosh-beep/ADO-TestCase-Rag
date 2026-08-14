# src/chat/pipeline.py
import httpx
from typing import Iterator

from src.llm.base import LLMClient
from src.chat.prompts import build_messages

MIN_SCORE = 0.4

def _filter_weak_results(results: list[dict]) -> list[dict]:
    """Drop results scoring below the relevance threshold."""
    return [r for r in results if r.get("score", 0) >= MIN_SCORE]

class ChatPipeline:
    def __init__(self, llm: LLMClient, search_api_url: str = "http://localhost:8001"):
        self.llm = llm
        self.search_api_url = search_api_url
        self.http = httpx.Client(timeout=60)

    def answer_stream(
        self,
        question: str,
        history: list[dict] | None = None,
        top_k: int = 8,
        filters: dict | None = None,
    ) -> Iterator[dict]:
        # Rewrite follow-up questions into a standalone query for retrieval
        retrieval_question = question
        if history:
            retrieval_question = self.llm.complete(
                system="Rewrite this follow-up question as a standalone search query, using context from history.",
                messages=history + [{"role": "user", "content": question}],
                max_tokens=100,
            )

        # 1. Call your Phase 3 search API
        payload = {"query": retrieval_question, "top_k": top_k, **(filters or {})}
        resp = self.http.post(f"{self.search_api_url}/search", json=payload)
        resp.raise_for_status()
        search_result = resp.json()

        if not search_result["results"]:
            yield {"type": "text", "data": "I couldn't find any test cases matching your query. Try rephrasing, or add filters like `feature:X`."}
            return

        # Drop results below the relevance threshold
        results = _filter_weak_results(search_result["results"])
        if not results:
            yield {"type": "text", "data": "I couldn't find any test cases matching your query. Try rephrasing, or add filters like `feature:X`."}
            return

        # Yield retrieval results first so UI can show "sources" before text streams
        yield {"type": "sources", "data": results}

        # 2. Build prompt
        system, messages = build_messages(
            question=question,
            results=results,
            history=history,
        )

        # 3. Stream LLM
        full_text = ""
        for chunk in self.llm.stream(system=system, messages=messages):
            full_text += chunk
            yield {"type": "text", "data": chunk}

        # 4. Emit final done event with full text for logging
        yield {
            "type": "done",
            "data": {
                "answer": full_text,
                "retrieved_ids": [r["test_case_id"] for r in results],
                "candidate_count": search_result["candidate_count"],
            },
        }