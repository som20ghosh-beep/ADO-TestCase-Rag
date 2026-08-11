# src/embedding/indexer.py
import logging
from datetime import datetime

from qdrant_client.models import PointStruct, SparseVector
from sqlmodel import Session, select

from src.db import engine
from src.models import TestCase
from src.embedding.collection import (
    COLLECTION_NAME, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, ensure_collection,
)
from src.embedding.embedder import EMBEDDING_VERSION, Embedder
from src.embedding.qdrant import get_qdrant
from src.embedding.sparse import SparseEncoder
from src.embedding.text_builder import build_searchable_text, compute_content_hash

log = logging.getLogger(__name__)


def index_all(mode: str = "incremental") -> None:
    qdrant = get_qdrant()
    ensure_collection(qdrant)

    embedder = Embedder()
    sparse_encoder = SparseEncoder()

    with Session(engine) as session:
        test_cases = session.exec(select(TestCase)).all()

        to_embed = []
        for tc in test_cases:
            searchable_text = build_searchable_text(tc)
            content_hash = compute_content_hash(searchable_text)

            stale = (
                mode == "full"
                or tc.embedded_at is None
                or tc.content_hash != content_hash
                or tc.embedding_version != EMBEDDING_VERSION
            )
            if not stale:
                continue

            tc.searchable_text = searchable_text
            tc.content_hash = content_hash
            to_embed.append(tc)

        if not to_embed:
            log.info("index_all: nothing to embed (mode=%s)", mode)
            return

        log.info("index_all: embedding %d test case(s) (mode=%s)", len(to_embed), mode)

        texts = [tc.searchable_text for tc in to_embed]
        dense_vecs = embedder.embed_batch(texts)
        sparse_vecs = sparse_encoder.encode_batch(texts)

        points = [
            PointStruct(
                id=tc.id,
                vector={
                    DENSE_VECTOR_NAME: dense_vecs[i].tolist(),
                    SPARSE_VECTOR_NAME: SparseVector(
                        indices=sparse_vecs[i].indices.tolist(),
                        values=sparse_vecs[i].values.tolist(),
                    ),
                },
                payload={
                    "title": tc.title,
                    "feature": tc.area_path.split("\\")[-1] if tc.area_path else None,
                    "state": tc.state,
                    "priority": tc.priority,
                    "automation_status": tc.automation_status,
                    "searchable_text": tc.searchable_text,
                },
            )
            for i, tc in enumerate(to_embed)
        ]

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

        now = datetime.utcnow()
        for tc in to_embed:
            tc.embedding_version = EMBEDDING_VERSION
            tc.embedded_at = now
            session.add(tc)
        session.commit()

    log.info("index_all: done")
