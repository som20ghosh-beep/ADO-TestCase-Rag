# src/embedding/qdrant.py
from functools import lru_cache

from qdrant_client import QdrantClient

from src.config import QDRANT_HOST, QDRANT_PORT


@lru_cache
def get_qdrant() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
