# src/retrieval/service.py
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import HybridRetriever


RERANK_POOL_SIZE = 10


class SearchService:
    def __init__(self, retriever, reranker, rewriter=None):
        self.retriever = retriever
        self.reranker = reranker
        self.rewriter = rewriter

    def search(self, query, filters=None, top_k=8, rerank=True, rewrite=True):
        rewrite_info = None
        search_query = query

        if rewrite and self.rewriter:
            rewrite_info = self.rewriter.rewrite(query)
            search_query = rewrite_info["rewritten"]

        candidates = self.retriever.retrieve(search_query, top_k=40, filters=filters)

        if rerank and candidates:
            # IMPORTANT: rerank against the ORIGINAL query, not the rewrite.
            # Only rerank the top slice of the retrieval pool (already sorted by
            # retrieval score) to bound cross-encoder latency on CPU.
            rerank_pool = candidates[:max(top_k, RERANK_POOL_SIZE)]
            results = self.reranker.rerank(query, rerank_pool, top_k=top_k)
        else:
            results = candidates[:top_k]

        return {
            "query": query,
            "search_query": search_query,
            "rewrite": rewrite_info,
            "filters": filters or {},
            "reranked": rerank,
            "candidate_count": len(candidates),
            "results": [self._to_result(r) for r in results],
        }

    def _to_result(self, r: dict) -> dict:
        return {
            "test_case_id": r["id"],
            "title": r["title"],
            "feature": r["feature"],
            "state": r["payload"].get("state"),
            "searchable_text": r["searchable_text"],
            "score": r.get("rerank_score", r["score"]),
            "retrieval_score": r.get("retrieval_score"),
            "why": self._explain(r),
        }

    def _explain(self, r: dict) -> str:
        """Simple reasoning trace — Phase 4's LLM will make this richer."""
        return f"Matched via hybrid search (retrieval: {r.get('retrieval_score', 0):.3f}, rerank: {r.get('rerank_score', 0):.3f})"