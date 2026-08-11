# src/embedding/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_VERSION = f"{MODEL_NAME.split('/')[-1]}:v1"

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.model.max_seq_length = 512

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # critical for cosine distance
            convert_to_numpy=True,
        )