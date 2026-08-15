# src/api/main.py
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.db import engine, get_session, init_db
from src.embedding.embedder import Embedder
from src.embedding.qdrant import get_qdrant
from src.embedding.sparse import SparseEncoder
from src.models import QueryLog
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import HybridRetriever
from src.retrieval.service import SearchService
from src.dedup.service import review_pair, get_pairs_with_titles
from src.dedup.scanner import scan_for_duplicates

log = logging.getLogger(__name__)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 8
    feature: str | None = None
    state: str | None = None
    rerank: bool = True

class SearchResult(BaseModel):
    test_case_id: int
    title: str
    feature: str | None
    state: str | None
    searchable_text: str
    score: float
    retrieval_score: float | None
    why: str

class SearchResponse(BaseModel):
    query: str
    reranked: bool
    candidate_count: int
    results: list[SearchResult]
    latency_ms: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Load heavy objects once
    app.state.embedder = Embedder()
    app.state.sparse_encoder = SparseEncoder()
    app.state.reranker = Reranker()
    app.state.qdrant = get_qdrant()
    retriever = HybridRetriever(app.state.qdrant, app.state.embedder, app.state.sparse_encoder)
    app.state.search_service = SearchService(retriever, app.state.reranker)
    log.info("api_ready")
    yield

app = FastAPI(title="ADO Test Case Retrieval", lifespan=lifespan)

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    start = time.perf_counter()
    try:
        filters = {"feature": req.feature, "state": req.state}
        filters = {k: v for k, v in filters.items() if v is not None}
        result = app.state.search_service.search(
            query=req.query,
            filters=filters or None,
            top_k=req.top_k,
            rerank=req.rerank,
        )
        latency = int((time.perf_counter() - start) * 1000)
        log.info("search: query=%r latency_ms=%d results=%d", req.query, latency, len(result["results"]))
        _log_query(req, result, latency)  # persist to SQLite for observability
        return {**result, "latency_ms": latency}
    except Exception as e:
        log.error("search_failed: query=%r error=%s", req.query, e)
        raise HTTPException(500, str(e))


def _log_query(req: SearchRequest, result: dict, latency_ms: int) -> None:
    session = get_session()
    try:
        session.add(QueryLog(
            query=req.query,
            filters=result.get("filters", {}),
            top_k=req.top_k,
            reranked=result["reranked"],
            candidate_count=result["candidate_count"],
            result_count=len(result["results"]),
            latency_ms=latency_ms,
            created_at=datetime.utcnow(),
        ))
        session.commit()
    finally:
        session.close()

@app.get("/health")
def health():
    return {"status": "ok"}

class ReviewRequest(BaseModel):
    decision: str  # "confirmed" | "dismissed"


@app.get("/duplicates")
def list_duplicates(status: str = "pending", limit: int = 50):
    return {
        "status_filter": status,
        "pairs": get_pairs_with_titles(engine, status=status, limit=limit),
    }


@app.post("/duplicates/{pair_id}/review")
def review_duplicate(pair_id: int, req: ReviewRequest):
    if req.decision not in ("confirmed", "dismissed"):
        raise HTTPException(400, "decision must be 'confirmed' or 'dismissed'")
    ok = review_pair(engine, pair_id, req.decision)
    if not ok:
        raise HTTPException(404, f"pair {pair_id} not found")
    return {"pair_id": pair_id, "status": req.decision}


@app.post("/duplicates/scan")
def trigger_scan(threshold: float = 0.92):
    """Manual trigger — for 5K cases this runs in seconds, fine to keep synchronous."""
    return scan_for_duplicates(app.state.qdrant, engine, threshold)