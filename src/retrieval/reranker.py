# src/retrieval/reranker.py
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "BAAI/bge-reranker-base"

class Reranker:
    def __init__(self):
        self.model = CrossEncoder(RERANKER_MODEL, max_length=512)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 8,
    ) -> list[dict]:
        if not candidates:
            return []
        pairs = [[query, c["searchable_text"]] for c in candidates]
        scores = self.model.predict(pairs, batch_size=16, show_progress_bar=False)

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
            c["retrieval_score"] = c["score"]

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]