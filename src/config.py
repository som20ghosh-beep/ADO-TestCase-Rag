# src/config.py
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

ADO_ORG_URL = os.environ["ADO_ORG_URL"]
ADO_PAT = os.environ["ADO_PAT"]
ADO_PROJECT = os.environ["ADO_PROJECT"]
ADO_AREA_PATH = os.environ["ADO_AREA_PATH"]
GROQ_KEY = os.environ.get("GROQ_KEY")

DB_PATH = BASE_DIR / "data" / "testcases.db"

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
REWRITE_MODEL = os.getenv("REWRITE_MODEL", "qwen2.5:3b")
REWRITE_TIMEOUT_S = float(os.getenv("REWRITE_TIMEOUT_S", "5.0"))