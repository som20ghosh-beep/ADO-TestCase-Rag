"""Duplicate detection: pairwise cosine similarity over Qdrant vectors."""
import numpy as np
from datetime import datetime
from sqlmodel import Session, select

from src.embedding.collection import COLLECTION_NAME, DENSE_VECTOR_NAME
from src.models import DuplicatePair

DEFAULT_THRESHOLD = 0.92
SCROLL_BATCH = 500


def fetch_all_vectors(qdrant) -> tuple[list[int], np.ndarray]:
    """Pull every dense vector from Qdrant via scroll pagination."""
    ids: list[int] = []
    vectors: list[list[float]] = []
    offset = None

    while True:
        points, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=SCROLL_BATCH,
            offset=offset,
            with_vectors=[DENSE_VECTOR_NAME],
            with_payload=False,
        )
        for p in points:
            vec = p.vector[DENSE_VECTOR_NAME] if isinstance(p.vector, dict) else p.vector
            ids.append(p.id)
            vectors.append(vec)
        if offset is None:
            break

    if not ids:
        return [], np.empty((0, 0), dtype=np.float32)
    return ids, np.asarray(vectors, dtype=np.float32)


def find_similar_pairs(
    ids: list[int],
    vecs: np.ndarray,
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Return (id_a, id_b, similarity) for all pairs >= threshold. a < b always."""
    if len(ids) < 2:
        return []

    # Normalize so dot product == cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.clip(norms, 1e-8, None)
    sim_matrix = vecs @ vecs.T

    iu = np.triu_indices(len(ids), k=1)  # upper triangle, excludes diagonal
    sims = sim_matrix[iu]
    mask = sims >= threshold

    return [
        (min(ids[i], ids[j]), max(ids[i], ids[j]), float(s))
        for i, j, s in zip(iu[0][mask], iu[1][mask], sims[mask])
    ]


def scan_for_duplicates(
    qdrant,
    engine,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Full scan: fetch vectors, find pairs, upsert into DuplicatePair table."""
    ids, vecs = fetch_all_vectors(qdrant)
    pairs = find_similar_pairs(ids, vecs, threshold)

    new_count, updated_count = 0, 0
    with Session(engine) as s:
        for a, b, sim in pairs:
            existing = s.exec(
                select(DuplicatePair).where(
                    DuplicatePair.test_case_a == a,
                    DuplicatePair.test_case_b == b,
                )
            ).first()
            if existing:
                existing.similarity = sim  # refresh score, preserve status
                updated_count += 1
            else:
                s.add(DuplicatePair(
                    test_case_a=a,
                    test_case_b=b,
                    similarity=sim,
                    scanned_at=datetime.utcnow(),
                ))
                new_count += 1
        s.commit()

    return {
        "total_cases": len(ids),
        "pairs_found": len(pairs),
        "new_pairs": new_count,
        "updated_pairs": updated_count,
        "threshold": threshold,
    }