"""Recording and aggregating feedback on search results."""
from datetime import datetime
from typing import Optional

from sqlmodel import Session, func, select

from src.models import QueryLog, ResultFeedback

VALID_VERDICTS = {"up", "down"}


def record_feedback(
    engine,
    query_log_id: int,
    test_case_id: int,
    verdict: str,
    rank_position: Optional[int] = None,
    user: Optional[str] = None,
) -> ResultFeedback:
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"verdict must be one of {VALID_VERDICTS}")

    with Session(engine) as s:
        if s.get(QueryLog, query_log_id) is None:
            raise LookupError(f"query_log {query_log_id} not found")

        existing = s.exec(
            select(ResultFeedback).where(
                ResultFeedback.query_log_id == query_log_id,
                ResultFeedback.test_case_id == test_case_id,
            )
        ).first()

        if existing is not None:
            existing.verdict = verdict
            existing.rank_position = rank_position
            existing.user = user
            existing.created_at = datetime.utcnow()
            s.add(existing)
            s.commit()
            s.refresh(existing)
            return existing

        fb = ResultFeedback(
            query_log_id=query_log_id,
            test_case_id=test_case_id,
            verdict=verdict,
            rank_position=rank_position,
            user=user,
            created_at=datetime.utcnow(),
        )
        s.add(fb)
        s.commit()
        s.refresh(fb)
        return fb


def feedback_stats(engine) -> dict:
    with Session(engine) as s:
        total = s.exec(select(func.count()).select_from(ResultFeedback)).one()
        up = s.exec(
            select(func.count()).select_from(ResultFeedback).where(ResultFeedback.verdict == "up")
        ).one()
        down = s.exec(
            select(func.count()).select_from(ResultFeedback).where(ResultFeedback.verdict == "down")
        ).one()

        worst = s.exec(
            select(ResultFeedback, QueryLog.query)
            .join(QueryLog, ResultFeedback.query_log_id == QueryLog.id)
            .where(ResultFeedback.rank_position == 1, ResultFeedback.verdict == "down")
            .order_by(ResultFeedback.created_at.desc())
            .limit(10)
        ).all()

        worst_failures = [
            {
                "query_log_id": fb.query_log_id,
                "query": query_text,
                "test_case_id": fb.test_case_id,
                "user": fb.user,
                "created_at": fb.created_at.isoformat(),
            }
            for fb, query_text in worst
        ]

        return {
            "total": total,
            "up": up,
            "down": down,
            "worst_failures": worst_failures,
        }
