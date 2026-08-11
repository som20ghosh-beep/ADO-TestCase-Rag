# src/embedding/sparse.py
from fastembed import SparseTextEmbedding

SPARSE_MODEL = "Qdrant/bm25"

class SparseEncoder:
    def __init__(self):
        self.model = SparseTextEmbedding(model_name=SPARSE_MODEL)

    def encode_batch(self, texts: list[str]) -> list:
        return list(self.model.embed(texts))

    def encode_query(self, text: str):
        return next(iter(self.model.query_embed([text])))