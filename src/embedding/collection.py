# src/embedding/collection.py — updated
import logging

from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, PayloadSchemaType
)

log = logging.getLogger(__name__)

COLLECTION_NAME = "test_cases"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
EMBEDDING_DIM = 768

def ensure_collection(client):
    if client.collection_exists(COLLECTION_NAME):
        # If existing collection has old schema, recreate. Detect via config.
        info = client.get_collection(COLLECTION_NAME)
        if DENSE_VECTOR_NAME not in (info.config.params.vectors or {}):
            log.warning("recreating_collection: reason=schema_upgrade")
            client.delete_collection(COLLECTION_NAME)
        else:
            return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(),
        },
    )
    # ... payload indexes as before