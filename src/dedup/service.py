"""Review workflow + query helpers for duplicate pairs."""
from datetime import datetime
from sqlmodel import Session, select

from src.models import DuplicatePair, TestCase  # adjust TestCase import to your model name

VALID_DECISIONS = {"confirmed", "dismissed"}


def review_pair(engine, pair_id: int, decision: str) -> bool:
    if decision not in VALID_DECISIONS:
        raise ValueError(f"decision must be one of {VALID_DECISIONS}")
    with Session(engine) as s:
        pair = s.get(DuplicatePair, pair_id)
        if pair is None:
            return False
        pair.status = decision
        pair.reviewed_at = datetime.utcnow()
        s.commit()
        return True


def get_pairs_with_titles(engine, status: str = "pending", limit: int = 50) -> list[dict]:
    """Pairs enriched with both test case titles, for CLI and API."""
    with Session(engine) as s:
        pairs = s.exec(
            select(DuplicatePair)
            .where(DuplicatePair.status == status)
            .order_by(DuplicatePair.similarity.desc())
            .limit(limit)
        ).all()

        out = []
        for p in pairs:
            # NOTE: if your TestCase primary key isn't the ADO ID,
            # swap s.get() for a select on your ado_id column.
            ta = s.get(TestCase, p.test_case_a)
            tb = s.get(TestCase, p.test_case_b)
            out.append({
                "pair_id": p.id,
                "similarity": round(p.similarity, 4),
                "status": p.status,
                "test_case_a": {
                    "id": p.test_case_a,
                    "title": ta.title if ta else "(not found)",
                },
                "test_case_b": {
                    "id": p.test_case_b,
                    "title": tb.title if tb else "(not found)",
                },
                "scanned_at": p.scanned_at.isoformat(),
                "reviewed_at": p.reviewed_at.isoformat() if p.reviewed_at else None,
            })
        return out