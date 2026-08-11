# src/retrieval/retriever.py
from qdrant_client.models import (
    Prefetch, FusionQuery, Fusion, Filter, FieldCondition, MatchValue,
    SparseVector,
)

from src.embedding.collection import COLLECTION_NAME, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

class HybridRetriever:
    def __init__(self, qdrant, embedder, sparse_encoder):
        self.qdrant = qdrant
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder

    def retrieve(
        self,
        query: str,
        top_k: int = 40,
        filters: dict | None = None,
    ) -> list[dict]:
        dense_vec = self.embedder.embed_batch([BGE_QUERY_PREFIX + query])[0]
        sparse_vec = self.sparse_encoder.encode_query(query)

        qdrant_filter = self._build_filter(filters) if filters else None

        result = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(
                    query=dense_vec.tolist(),
                    using=DENSE_VECTOR_NAME,
                    limit=top_k * 2,
                    filter=qdrant_filter,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist(),
                    ),
                    using=SPARSE_VECTOR_NAME,
                    limit=top_k * 2,
                    filter=qdrant_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "id": p.id,
                "score": p.score,
                "title": p.payload["title"],
                "feature": p.payload["feature"],
                "searchable_text": p.payload["searchable_text"],
                "payload": p.payload,
            }
            for p in result.points
        ]

    def _build_filter(self, filters: dict) -> Filter:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items() if v is not None
        ]
        return Filter(must=conditions) if conditions else None