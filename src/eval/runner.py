# src/eval/runner.py
from sqlmodel import Session, select

from src.db import engine
from src.models import EvalQuery

variants = [
    ("baseline",        {"rerank": False, "rewrite": False}),
    ("rerank_only",     {"rerank": True,  "rewrite": False}),
    ("rewrite_only",    {"rerank": False, "rewrite": True}),
    ("rewrite_rerank",  {"rerank": True,  "rewrite": True}),
]


def _parse_correct_ids(correct_ids: str) -> set[int]:
    return {int(p) for p in correct_ids.split(",") if p.strip()}


def _score_query(retrieved_ids: list[int], correct_ids: set[int]) -> dict:
    hit_set = [1 if rid in correct_ids else 0 for rid in retrieved_ids]
    num_correct_retrieved = sum(hit_set)

    reciprocal_rank = 0.0
    for rank, hit in enumerate(hit_set, start=1):
        if hit:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "hit": 1.0 if num_correct_retrieved > 0 else 0.0,
        "precision": num_correct_retrieved / len(retrieved_ids) if retrieved_ids else 0.0,
        "recall": num_correct_retrieved / len(correct_ids) if correct_ids else 0.0,
        "reciprocal_rank": reciprocal_rank,
    }


def run_eval(search_service, top_k: int = 8) -> list[dict]:
    """Run every (rerank, rewrite) variant against every eval_query row and
    return aggregate precision@k / recall@k / MRR / hit-rate per variant."""
    with Session(engine) as session:
        queries = session.exec(select(EvalQuery)).all()

    if not queries:
        return []

    report = []
    for variant_name, opts in variants:
        scores = []
        for eq in queries:
            correct_ids = _parse_correct_ids(eq.correct_ids)
            if not correct_ids:
                continue

            result = search_service.search(
                eq.query,
                top_k=top_k,
                rerank=opts["rerank"],
                rewrite=opts["rewrite"],
            )
            retrieved_ids = [r["test_case_id"] for r in result["results"]]
            scores.append(_score_query(retrieved_ids, correct_ids))

        n = len(scores)
        report.append({
            "variant": variant_name,
            "n": n,
            "hit_rate": sum(s["hit"] for s in scores) / n if n else 0.0,
            "precision": sum(s["precision"] for s in scores) / n if n else 0.0,
            "recall": sum(s["recall"] for s in scores) / n if n else 0.0,
            "mrr": sum(s["reciprocal_rank"] for s in scores) / n if n else 0.0,
        })

    return report
