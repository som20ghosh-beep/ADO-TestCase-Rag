# src/cli.py
import csv
from datetime import datetime

import typer
from sqlmodel import Session, func, select

from .db import engine, get_session, init_db
from .models import EvalQuery, SyncRun, TestCase
from .sync import run_sync
from src.dedup.scanner import scan_for_duplicates
from src.dedup.service import get_pairs_with_titles, review_pair
from src.embedding.indexer import index_all
from src.embedding.embedder import Embedder
from src.embedding.qdrant import get_qdrant
from src.embedding.collection import COLLECTION_NAME

app = typer.Typer()

@app.command()
def sync(full: bool = False):
    """Sync test cases from ADO."""
    mode = "full" if full else "incremental"
    run = run_sync(mode=mode)
    typer.echo(
        f"mode={run.mode} fetched={run.items_fetched} "
        f"upserted={run.items_upserted} failed={run.items_failed} error={run.error}"
    )

@app.command()
def stats():
    """Show DB stats."""
    session = get_session()

    total = session.exec(select(func.count()).select_from(TestCase)).one()
    typer.echo(f"total test cases: {total}")

    typer.echo("by state:")
    by_state = session.exec(
        select(TestCase.state, func.count()).group_by(TestCase.state).order_by(func.count().desc())
    ).all()
    for state, count in by_state:
        typer.echo(f"  {state}: {count}")

    typer.echo("by area path:")
    by_area = session.exec(
        select(TestCase.area_path, func.count()).group_by(TestCase.area_path).order_by(func.count().desc())
    ).all()
    for area_path, count in by_area:
        typer.echo(f"  {area_path}: {count}")

    last_run = session.exec(select(SyncRun).order_by(SyncRun.started_at.desc())).first()
    if last_run:
        typer.echo(
            f"last sync: mode={last_run.mode} started={last_run.started_at} "
            f"finished={last_run.finished_at} fetched={last_run.items_fetched} "
            f"upserted={last_run.items_upserted} failed={last_run.items_failed} "
            f"error={last_run.error}"
        )
    else:
        typer.echo("last sync: none")

@app.command()
def show(id: int):
    """Print a single test case by ID."""
    session = get_session()
    tc = session.get(TestCase, id)
    if not tc:
        typer.echo(f"no test case with id {id}")
        raise typer.Exit(code=1)

    typer.echo(f"[{tc.id}] {tc.title}")
    typer.echo(f"state: {tc.state}    priority: {tc.priority}    automation_status: {tc.automation_status}")
    typer.echo(f"area_path: {tc.area_path}")
    typer.echo(f"iteration_path: {tc.iteration_path}")
    typer.echo(f"tags: {tc.tags}")
    typer.echo(f"assigned_to: {tc.assigned_to}")
    typer.echo(f"revision: {tc.revision}    created: {tc.created_date}    changed: {tc.changed_date}")
    typer.echo(f"synced_at: {tc.synced_at}")
    if tc.preconditions:
        typer.echo(f"preconditions: {tc.preconditions}")
    typer.echo("steps:")
    for step in tc.steps_json:
        typer.echo(f"  {step.get('step')}. {step.get('action')}")
        if step.get("expected"):
            typer.echo(f"     expected: {step.get('expected')}")
    if tc.expected_result:
        typer.echo(f"expected_result: {tc.expected_result}")

@app.command()
def index(full: bool = False):
    """Sync SQLite → Qdrant embeddings."""
    index_all(mode="full" if full else "incremental")

@app.command()
def search(query: str, k: int = 10, feature: str = None):
    """Smoke test: run a vector search and print top-K results."""
    qdrant = get_qdrant()
    embedder = Embedder()
    q_text = f"Represent this sentence for searching relevant passages: {query}"
    q_vec = embedder.embed_batch([q_text])[0]

    filter_ = None
    if feature:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        filter_ = Filter(must=[FieldCondition(key="feature", match=MatchValue(value=feature))])

    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec.tolist(),
        limit=k,
        query_filter=filter_,
    ).points
    for h in hits:
        print(f"[{h.score:.3f}] TC-{h.id}: {h.payload['title']}  ({h.payload['feature']})")

@app.command()
def index_stats():
    """Show Qdrant collection stats + SQLite embedding coverage."""
    qdrant = get_qdrant()
    info = qdrant.get_collection(COLLECTION_NAME)
    print(f"Points in Qdrant: {info.points_count}")
    with Session(engine) as s:
        embedded = s.exec(select(func.count(TestCase.id)).where(TestCase.embedded_at.isnot(None))).one()
        total = s.exec(select(func.count(TestCase.id))).one()
        print(f"Embedded in SQLite: {embedded}/{total}")

@app.command("eval-import")
def eval_import(csv_path: str):
    """Load eval queries (query, correct_ids, notes) from a CSV into the eval_query table."""
    init_db()
    session = get_session()
    imported = 0
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                query = (row.get("query") or "").strip()
                if not query:
                    continue
                correct_ids = (row.get("correct_ids") or "").strip().strip('"')
                correct_ids = ",".join(p.strip() for p in correct_ids.split(",") if p.strip())
                session.add(EvalQuery(
                    query=query,
                    correct_ids=correct_ids,
                    notes=(row.get("notes") or "").strip() or None,
                    created_at=datetime.utcnow(),
                ))
                imported += 1
        session.commit()
    finally:
        session.close()
    typer.echo(f"imported {imported} eval queries from {csv_path}")

@app.command()
def dupes_scan(threshold: float = 0.92):
    """Scan all indexed test cases for near-duplicate pairs."""
    init_db()
    result = scan_for_duplicates(get_qdrant(), engine, threshold)
    print(
        f"Scanned {result['total_cases']} cases -> "
        f"{result['pairs_found']} pairs >= {threshold} "
        f"({result['new_pairs']} new)"
    )

@app.command()
def dupes_list(status: str = "pending", limit: int = 20):
    """List duplicate pairs for review, highest similarity first."""
    init_db()
    pairs = get_pairs_with_titles(engine, status=status, limit=limit)
    if not pairs:
        print(f"No pairs with status '{status}'.")
        return
    for p in pairs:
        print(f"\n[{p['similarity']:.3f}] pair #{p['pair_id']} ({p['status']})")
        print(f"  TC-{p['test_case_a']['id']}: {p['test_case_a']['title']}")
        print(f"  TC-{p['test_case_b']['id']}: {p['test_case_b']['title']}")

@app.command()
def dupes_review(pair_id: int, decision: str):
    """Mark a pair as confirmed or dismissed."""
    ok = review_pair(engine, pair_id, decision)
    print(f"Pair #{pair_id} -> {decision}" if ok else f"Pair #{pair_id} not found.")

if __name__ == "__main__":
    app()