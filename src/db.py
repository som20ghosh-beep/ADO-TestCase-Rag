# src/db.py
from sqlalchemy import inspect, text
from sqlmodel import Session, SQLModel, create_engine

from .config import DB_PATH

engine = create_engine(f"sqlite:///{DB_PATH}")

_NEW_TESTCASE_COLUMNS = [
    ("searchable_text", "TEXT"),
    ("content_hash", "TEXT"),
    ("embedding_version", "TEXT"),
    ("embedded_at", "DATETIME"),
]


def _add_missing_columns() -> None:
    inspector = inspect(engine)
    if "testcase" not in inspector.get_table_names():
        return
    existing = {col["name"] for col in inspector.get_columns("testcase")}
    missing = [(name, col_type) for name, col_type in _NEW_TESTCASE_COLUMNS if name not in existing]
    if not missing:
        return
    with engine.connect() as conn:
        for name, col_type in missing:
            conn.execute(text(f"ALTER TABLE testcase ADD COLUMN {name} {col_type}"))
        conn.commit()


def init_db() -> None:
    SQLModel.metadata.create_all(engine)
    _add_missing_columns()


def get_session() -> Session:
    return Session(engine, expire_on_commit=False)
