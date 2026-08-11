# src/retrieval/service.py
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import HybridRetriever


class SearchService:
    def __init__(self, retriever: HybridRetriever, reranker: Reranker):
        self.retriever = retriever
        self.reranker = reranker

    def search(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 8,
        rerank: bool = True,
    ) -> dict:
        candidates = self.retriever.retrieve(query, top_k=40, filters=filters)
        if rerank and candidates:
            results = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            results = candidates[:top_k]

        return {
            "query": query,
            "filters": filters or {},
            "reranked": rerank,
            "candidate_count": len(candidates),
            "results": [
                {
                    "test_case_id": r["id"],
                    "title": r["title"],
                    "feature": r["feature"],
                    "state": r["payload"].get("state"),
                    "searchable_text": r["searchable_text"],
                    "score": r.get("rerank_score", r["score"]),
                    "retrieval_score": r.get("retrieval_score"),
                    "why": self._explain(r),
                }
                for r in results
            ],
        }

    def _explain(self, r: dict) -> str:
        """Simple reasoning trace — Phase 4's LLM will make this richer."""
        return f"Matched via hybrid search (retrieval: {r.get('retrieval_score', 0):.3f}, rerank: {r.get('rerank_score', 0):.3f})"