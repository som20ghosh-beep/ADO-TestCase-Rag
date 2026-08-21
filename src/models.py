# src/models.py
from sqlmodel import SQLModel, Field, JSON, Column
from datetime import datetime
from typing import Optional

class TestCase(SQLModel, table=True):
    id: int = Field(primary_key=True)              # ADO work item ID
    title: str
    state: str                                     # Design, Ready, Closed, etc.
    priority: Optional[int] = None
    area_path: str
    iteration_path: Optional[str] = None
    tags: str = ""                                 # Semicolon-separated in ADO
    preconditions: Optional[str] = None            # From description or custom field
    steps_json: list = Field(sa_column=Column(JSON))  # Parsed test steps
    expected_result: Optional[str] = None          # Aggregated or last-step expected
    automation_status: Optional[str] = None
    assigned_to: Optional[str] = None
    created_date: datetime
    changed_date: datetime                         # Critical for incremental sync
    revision: int                                  # ADO revision number
    synced_at: datetime                            # When we last pulled it
    raw_fields: dict = Field(sa_column=Column(JSON))  # Full field dump for future use

    searchable_text: Optional[str] = None
    content_hash: Optional[str] = None            # SHA256 of searchable_text
    embedding_version: Optional[str] = None       # e.g. "bge-base-en-v1.5:v1"
    embedded_at: Optional[datetime] = None

class SyncRun(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    started_at: datetime
    finished_at: Optional[datetime] = None
    mode: str                                      # "full" or "incremental"
    items_fetched: int = 0
    items_upserted: int = 0
    items_failed: int = 0
    last_changed_date_seen: Optional[datetime] = None
    error: Optional[str] = None

class QueryLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    query: str
    filters: dict = Field(default_factory=dict, sa_column=Column(JSON))
    top_k: int
    reranked: bool
    candidate_count: int
    result_count: int
    latency_ms: int
    created_at: datetime

class EvalQuery(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    query: str
    correct_ids: str  # comma-separated ADO IDs
    notes: Optional[str] = None
    created_at: datetime

    # src/models.py — add
class ChatFeedback(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    query: str
    answer: str
    retrieved_ids: str      # comma-separated
    cited_ids: str          # comma-separated
    hallucinated_ids: str   # comma-separated
    is_grounded: bool
    feedback: str           # 'up' / 'down' / null
    comment: Optional[str] = None
    llm_provider: str
    created_at: datetime

class DuplicatePair(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    test_case_a: int = Field(index=True)   # always the smaller ADO ID
    test_case_b: int = Field(index=True)   # always the larger ADO ID
    similarity: float
    status: str = Field(default="pending", index=True)  # pending | confirmed | dismissed
    scanned_at: datetime
    reviewed_at: Optional[datetime] = None

class ResultFeedback(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    query_log_id: int = Field(index=True, foreign_key="querylog.id")
    test_case_id: int = Field(index=True)      # the ADO ID of the result being judged
    verdict: str = Field(index=True)           # "up" | "down"
    rank_position: Optional[int] = None        # where it appeared in results (1-based)
    user: Optional[str] = None                 # who gave feedback (Teams user, etc.)
    created_at: datetime