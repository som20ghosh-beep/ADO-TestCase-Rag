# ado-testcase-rag

A tool that pulls **Test Case** work items out of Azure DevOps (ADO), stores them locally, and makes them searchable and chattable using a hybrid RAG (Retrieval-Augmented Generation) pipeline.

In plain terms: instead of a QA engineer manually hunting through ADO's Test Plans for "which test case covers the incentive dropdown?", they can either run a search command or ask a chatbot in plain English, and get back the actual test case IDs, titles, and steps — with citations, not guesses.

---

## 1. What this project actually does

The project is built in four layers, each of which maps to a folder under `src/`:

1. **Sync** (`src/sync.py`, `src/ado_client.py`, `src/parser.py`) — Pulls Test Case work items from Azure DevOps via WIQL query + batched work-item fetch, parses ADO's test-step XML into clean `[{step, action, expected}, ...]` data, and upserts everything into a local SQLite database (`data/testcases.db`).
2. **Index** (`src/embedding/`) — Converts each test case into a block of searchable text, embeds it with a dense embedding model (for semantic similarity) and a sparse BM25-style encoder (for keyword matching), and upserts both vectors into a Qdrant collection.
3. **Retrieve** (`src/retrieval/`) — For a query, runs dense + sparse search in parallel, fuses the results with Reciprocal Rank Fusion (RRF), then re-ranks the top candidates with a cross-encoder reranker for higher precision.
4. **Chat** (`src/chat/`, `src/llm/`) — Wraps retrieval behind an LLM (Claude or Groq/Llama), which answers the user's question using *only* the retrieved test cases and must cite every claim as `[TC-<id>]`. A validator checks the answer for citations that don't actually exist in the retrieved set ("hallucination" detection).

These layers are exposed in three ways:

- A **Typer CLI** (`src/cli.py`) for running sync/index/search by hand or from scripts.
- A **FastAPI search service** (`src/api/main.py`) exposing `POST /search`, used internally by the chat layer and usable directly by anything else.
- A **Chainlit chat UI** (`src/ui/app.py`) — a browser chat window backed by the pipeline above.

### End-to-end flow

```
   Azure DevOps                     Local SQLite                    Qdrant
  (Test Case work   --sync-->     data/testcases.db   --index-->   (dense + sparse
   items, source                  (source of truth                  vectors, one
   of truth)                       for this tool)                   point per TC)
                                                                          |
                                                                          v
 User query --> Hybrid retrieve (RRF fusion) --> Cross-encoder rerank --> top-K test cases
                                                                          |
                                                          (chat only)     v
                                                              LLM answer, grounded,
                                                              cited as [TC-1234],
                                                              checked for hallucination
```

Nothing is ever written back to Azure DevOps — the data flow is strictly one-directional (ADO → SQLite → Qdrant → search/chat).

---

## 2. Project structure

```
ado-testcase-rag/
├── src/
│   ├── config.py            # Loads .env, exposes ADO/Groq/Qdrant settings
│   ├── db.py                 # SQLite engine/session + schema init & column migration
│   ├── models.py             # SQLModel tables: TestCase, SyncRun, QueryLog, EvalQuery, ChatFeedback
│   ├── ado_client.py          # ADO WIQL query + batched work-item fetch (retried)
│   ├── parser.py               # Parses ADO's test-step XML into structured steps
│   ├── sync.py                  # Orchestrates full/incremental sync ADO -> SQLite
│   ├── cli.py                    # Typer CLI: sync, stats, show, index, search, index-stats, eval-import
│   ├── embedding/
│   │   ├── text_builder.py        # Builds the text block that gets embedded, + content hash
│   │   ├── embedder.py             # Dense embedder (BAAI/bge-base-en-v1.5)
│   │   ├── sparse.py                # Sparse BM25 encoder (Qdrant/bm25 via fastembed)
│   │   ├── collection.py             # Qdrant collection schema + auto-migration
│   │   ├── qdrant.py                  # Cached Qdrant client
│   │   └── indexer.py                  # SQLite -> Qdrant sync (only re-embeds stale rows)
│   ├── retrieval/
│   │   ├── retriever.py             # Hybrid dense+sparse search with RRF fusion
│   │   ├── reranker.py               # Cross-encoder reranking (BAAI/bge-reranker-base)
│   │   └── service.py                 # Combines retriever + reranker into one search() call
│   ├── chat/
│   │   ├── pipeline.py               # Orchestrates: rewrite query -> search -> prompt -> stream LLM
│   │   ├── prompts.py                 # System prompt + context formatting for the LLM
│   │   └── validator.py                # Detects citations not present in the retrieved set
│   ├── llm/
│   │   ├── base.py                    # LLMClient abstract interface (stream/complete)
│   │   ├── factory.py                  # Picks Claude or Groq based on LLM_PROVIDER
│   │   ├── claude.py                    # Anthropic client wrapper
│   │   └── groq_client.py                # Groq client wrapper
│   ├── api/
│   │   └── main.py                      # FastAPI: POST /search, GET /health, query logging
│   └── ui/
│       └── app.py                       # Chainlit chat app (calls the API, streams answers)
├── scripts/
│   └── warm_model.py                    # Pre-downloads/loads the dense embedding model
├── tests/
│   ├── test_parser.py                   # (currently empty — see "Known gaps")
│   └── fixtures/                         # (currently empty)
├── data/
│   ├── testcases.db                      # SQLite database (created on first sync)
│   ├── qdrant/                            # Qdrant's on-disk storage (docker volume)
│   └── dataeval_queries.csv               # Sample eval queries for eval-import
├── docker-compose.yml                     # Qdrant service definition
├── pyproject.toml                          # Dependencies (managed with uv)
├── uv.lock                                  # Locked dependency versions
├── .env / .env.example                       # Configuration (never commit .env)
└── chainlit.md                                # Chainlit welcome-screen markdown
```

---

## 3. Prerequisites

- **Python 3.13+** (declared in `pyproject.toml` as `requires-python = ">=3.13"`)
- **[uv](https://docs.astral.sh/uv/)** — the project has a `uv.lock`, so `uv` is the intended package manager. `pip` works too if you prefer, using `pyproject.toml` directly.
- **Docker** (or an existing Qdrant instance) — Qdrant is the vector database and is run via `docker-compose.yml`.
- An **Azure DevOps Personal Access Token (PAT)** with at least read access to Work Items in the target project.
- An API key for whichever LLM provider you plan to chat with — **Anthropic** (Claude) and/or **Groq** are supported out of the box.

---

## 4. Setup — step by step

### Step 1 — Get into the project directory

```
cd "ado-testcase-rag"
```

### Step 2 — Create the environment and install dependencies

With `uv` (recommended, matches the committed `uv.lock`):

```
uv sync
```

This creates `.venv/` and installs everything pinned in `uv.lock`. Without `uv`:

```
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### Step 3 — Configure `.env`

Create a `.env` file in the project root (the loader in [src/config.py](src/config.py) reads it automatically). The variables actually used across the codebase are:

| Variable | Required | Used by | Description |
|---|---|---|---|
| `ADO_ORG_URL` | Yes | `config.py`, `sync` | Full org URL, e.g. `https://dev.azure.com/<org>/` |
| `ADO_PAT` | Yes | `config.py`, `sync` | Azure DevOps Personal Access Token (Work Items: Read) |
| `ADO_PROJECT` | Yes | `config.py`, `sync` | ADO project name that owns the test cases |
| `ADO_AREA_PATH` | Yes | `config.py`, `sync` | Area path to scope the sync to, e.g. `MyProject\Team\Feature` |
| `ADO_ORG` | No | present in `.env`, not currently read by any module | Kept for reference/future use |
| `QDRANT_HOST` | No (default `localhost`) | `config.py`, Qdrant client | Hostname of the Qdrant instance |
| `QDRANT_PORT` | No (default `6333`) | `config.py`, Qdrant client | Qdrant REST port |
| `LLM_PROVIDER` | No (default `claude`) | `src/ui/app.py`, `src/llm/factory.py` | `claude` or `groq` — which LLM answers chat questions |
| `GROQ_KEY` | Required if `LLM_PROVIDER=groq` | `src/llm/groq_client.py` | Groq API key |
| `ANTHROPIC_API_KEY` | Required if `LLM_PROVIDER=claude` | `src/llm/claude.py` (read implicitly by the `anthropic` SDK) | Anthropic API key — not read via `config.py`, must be a real environment variable |

> **Security note:** `.env` holds live secrets (an ADO PAT and one or more LLM API keys). The current `.gitignore` in this project is empty, so if you ever turn this folder into a git repo, add `.env`, `.venv/`, `data/`, `__pycache__/`, and `*.log` to it **before** your first commit.

### Step 4 — Start Qdrant

```
docker compose up -d
```

This starts Qdrant on `localhost:6333` (REST) / `6334` (gRPC) and persists its data to `./data/qdrant`.

### Step 5 — (Optional) Warm the embedding model

The first time the dense embedder runs, it downloads `BAAI/bge-base-en-v1.5` from Hugging Face, which can be slow. Pre-warm it once so later steps aren't waiting on a download mid-command:

```
python scripts/warm_model.py
```

### Step 6 — First sync: pull test cases from ADO into SQLite

```
python -m src.cli sync --full
```

This creates `data/testcases.db` (via `init_db()`) if it doesn't exist yet, then pulls every Test Case work item under `ADO_AREA_PATH`.

### Step 7 — Build the vector index

```
python -m src.cli index --full
```

This embeds every test case (dense + sparse) and upserts them into the Qdrant `test_cases` collection.

### Step 8 — Verify everything landed

```
python -m src.cli stats
python -m src.cli index-stats
python -m src.cli search "some query you expect to match"
```

At this point the local system is fully populated and ready to serve searches or chat.

> **Windows note:** running `--help` (or any command whose docstring contains a non-ASCII character, such as the `index` command's "SQLite → Qdrant") can crash with `UnicodeEncodeError` on the default Windows console codepage. If you hit that, prefix commands with `PYTHONIOENCODING=utf-8` (bash) or run `chcp 65001` first (PowerShell/cmd).

---

## 5. Running the services

### CLI command reference

All commands are run as `python -m src.cli <command>` from the project root.

| Command | Example | What it does |
|---|---|---|
| `sync` | `python -m src.cli sync [--full/--no-full]` | Pulls test cases from ADO into SQLite. Default is incremental (since the last successful run); `--full` re-pulls everything. |
| `stats` | `python -m src.cli stats` | Prints total test case count, breakdown by state and area path, and the last sync run's summary. |
| `show` | `python -m src.cli show <id>` | Prints one test case's fields and steps in full. |
| `index` | `python -m src.cli index [--full/--no-full]` | Embeds SQLite test cases into Qdrant. Default only re-embeds rows that are new, changed (content hash differs), or on an old embedding version; `--full` re-embeds everything. |
| `search` | `python -m src.cli search "<query>" [--k 10] [--feature <name>]` | A smoke-test vector search straight against Qdrant (dense only, no rerank) — useful for a quick sanity check without starting the API. |
| `index-stats` | `python -m src.cli index-stats` | Shows the Qdrant point count and how many SQLite rows are embedded vs. total. |
| `eval-import` | `python -m src.cli eval-import <csv_path>` | Loads a CSV of `query,correct_ids,notes` rows into the `eval_query` table for retrieval evaluation (see §7). |

### Search API (FastAPI)

```
uvicorn src.api.main:app --port 8001 --reload
```

- `POST /search` — body: `{"query": "...", "top_k": 8, "feature": null, "state": null, "rerank": true}`. Returns ranked results with scores and a one-line "why" explanation, and logs every request to the `QueryLog` table for observability.
- `GET /health` — liveness check.

Port **8001** matters: the Chainlit chat pipeline is hardcoded to call `http://localhost:8001` (see `ChatPipeline.__init__` in [src/chat/pipeline.py](src/chat/pipeline.py)), so start the API on that port before starting the chat UI.

### Chat UI (Chainlit)

```
chainlit run src/ui/app.py -w
```

Opens on `http://localhost:8000`. Type a question in plain English, or prefix it with inline filters, e.g.:

```
feature:Payments state:Active how does refund validation work
```

The UI streams the LLM's answer, shows the retrieved test cases as collapsible source cards, and appends a grounding badge (✅ verified / ⚠️ unverified IDs) based on citation validation. Thumbs up/down feedback is persisted to the `ChatFeedback` table.

---

## 6. Adding new test cases — step by step

**Azure DevOps is the source of truth.** This tool never creates or edits test cases — it only reads them. So "adding a test case" always starts in ADO, and only becomes searchable here after you re-run sync and index:

1. **Create or edit the Test Case work item in Azure DevOps**, under the area path configured in `ADO_AREA_PATH`. Fill in at minimum: Title, Steps (Action + Expected Result per step), State, and Area Path — these are exactly the fields `src/parser.py` and `src/sync.py` read.
2. **Pull it into the local database:**
   ```
   python -m src.cli sync
   ```
   Incremental sync is watermarked on `System.ChangedDate`, so a brand-new or just-edited test case will be picked up automatically. Use `--full` if you're not confident the incremental watermark covers it (e.g. clock skew, or you changed `ADO_AREA_PATH`/`ADO_PROJECT` — see §8.D).
3. **Confirm it landed:**
   ```
   python -m src.cli show <id>
   python -m src.cli stats
   ```
4. **Embed it into the search index:**
   ```
   python -m src.cli index
   ```
   Only the new/changed test case gets embedded — `index_all()` compares a SHA-256 hash of the built text plus the embedding model version, and skips anything unchanged. This makes `index` (without `--full`) cheap to run often.
5. **Confirm it's searchable:**
   ```
   python -m src.cli index-stats
   python -m src.cli search "something from the new test case"
   ```
6. **Use it** — via the CLI `search`, a `POST /search` call, or by asking the Chainlit chat UI.

There is currently no scheduler wired up (no cron job, no Task Scheduler entry, no background worker) — steps 2 and 4 are manual or need to be automated externally (e.g. a scheduled `sync && index` script) if you want near-real-time freshness.

---

## 7. Evaluation

`data/dataeval_queries.csv` holds hand-written evaluation queries in the form:

```
query,correct_ids,notes
text search for my employers,60071,employer search feature
```

- `query` — a natural-language query a QA engineer might type.
- `correct_ids` — comma-separated ADO test case IDs that should be retrieved for that query (ground truth).
- `notes` — optional free-text context.

Load them into the `eval_query` table with:

```
python -m src.cli eval-import data/dataeval_queries.csv
```

There is no automated scorer wired up yet — the table exists so retrieval quality (precision/recall against `correct_ids`) can be measured, but that comparison is currently a manual/future step.

---

## 8. Database & index migrations

This project does **not** use Alembic or any migration framework. Schema evolution is handled by small, purpose-built mechanisms in the code. Know these before you change `src/models.py` or the embedding model.

### A. Adding or changing the SQLite schema

`init_db()` in [src/db.py](src/db.py) does two things every time it runs (which is on every CLI command and on API startup):

1. `SQLModel.metadata.create_all(engine)` — creates any **brand-new table** that exists in `src/models.py` but not yet in the database. Nothing to do manually here.
2. `_add_missing_columns()` — a hand-maintained list, `_NEW_TESTCASE_COLUMNS`, of `(column_name, sql_type)` pairs that get added to the existing `testcase` table via `ALTER TABLE ... ADD COLUMN` if missing.

**To add a new table** (e.g. a new SQLModel class): just define it in `src/models.py`. The next `init_db()` call creates it automatically — no other step needed.

**To add a new column to the existing `TestCase` table:**

1. Add the field to the `TestCase` class in [src/models.py](src/models.py).
2. Add a matching `("column_name", "SQL_TYPE")` tuple to `_NEW_TESTCASE_COLUMNS` in [src/db.py](src/db.py).
3. Run any command that touches the DB (e.g. `python -m src.cli stats`), or explicitly:
   ```
   python -c "from src.db import init_db; init_db()"
   ```

**Limitations to be aware of:** this mechanism is additive-only. It does not rename or drop columns, does not touch any table other than `testcase`, and does not backfill data for the new column on existing rows (they'll be `NULL`). If you need any of that, write a one-off script.

### B. Migrating the Qdrant vector index (e.g. changing the embedding model)

[src/embedding/collection.py](src/embedding/collection.py)'s `ensure_collection()` detects a schema mismatch (the expected dense vector name isn't present in the existing collection config) and **automatically deletes and recreates the whole collection** — which discards every point currently stored.

To change the embedding model or its dimensionality:

1. Update `MODEL_NAME` (and therefore `EMBEDDING_VERSION`) in [src/embedding/embedder.py](src/embedding/embedder.py).
2. If the new model's output dimension differs from 768, update `EMBEDDING_DIM` in [src/embedding/collection.py](src/embedding/collection.py) to match.
3. Run a full reindex:
   ```
   python -m src.cli index --full
   ```
   The first call to `ensure_collection()` inside this run will detect the mismatch, drop the old collection, recreate it with the new vector config, and then `index_all(mode="full")` re-embeds and re-upserts every test case from SQLite.
4. Verify with `python -m src.cli index-stats` — points count should match total test case count again.

Skipping `--full` after a model change is not safe: since the collection gets wiped, an incremental run would leave most test cases unembedded until their content happens to change again.

### C. Moving to a new machine / environment

All durable state lives in exactly two places: `data/testcases.db` (SQLite) and `data/qdrant/` (Qdrant's storage, mounted by `docker-compose.yml`). You have two options:

- **Copy state:** copy both `data/testcases.db` and `data/qdrant/` to the new machine, bring up `docker compose up -d` there, and you're running with the exact same data.
- **Rebuild from source of truth:** on a fresh machine, just run Setup steps 6–7 again (`sync --full` then `index --full`). Since ADO is the source of truth, this reconstructs everything — at the cost of re-fetching from ADO and re-embedding all test cases.

### D. Switching ADO project / area path / org

If you repoint `.env` at a different `ADO_PROJECT`, `ADO_AREA_PATH`, or `ADO_ORG_URL`, **always follow up with a full sync**:

```
python -m src.cli sync --full
```

Reason: incremental sync's WIQL query combines the *current* `ADO_AREA_PATH` with a `changed_since` watermark from the last successful run (`SyncRun.last_changed_date_seen`). If you switch scope, older test cases that fall in the new area path but haven't changed recently would be silently skipped by an incremental run, since the watermark has no notion of "scope changed."

---

## 9. Observability

Every search and every chat turn is logged to SQLite, alongside the sync history:

- **`SyncRun`** — one row per `sync` invocation: mode, counts fetched/upserted/failed, the changed-date watermark, and any error.
- **`QueryLog`** — one row per `POST /search` call: query, filters, whether reranking was used, candidate/result counts, and latency in ms.
- **`ChatFeedback`** — one row per thumbs up/down in the Chainlit UI: the question, the answer, retrieved vs. cited vs. hallucinated IDs, whether the answer was grounded, and which LLM provider answered.

All three are plain SQLModel tables in `data/testcases.db`, so they can be queried directly with any SQLite client for debugging or reporting.

---

## 10. Known gaps / current limitations

- **No automated tests yet** — `tests/test_parser.py` and `tests/fixtures/` both exist but are currently empty.
- **No auth on the FastAPI search endpoint** — it's intended for localhost use behind the chat UI; don't expose it publicly as-is.
- **No scheduler** — sync and index are on-demand only (§6); set up your own cron/Task Scheduler job if you want freshness without manual runs.
- **No migration framework** — schema changes are manual, see §8. Get comfortable with that section before altering `src/models.py`.
- **Heavy models loaded per process** — the FastAPI app loads the embedder, sparse encoder, and reranker once at startup (`lifespan` in `src/api/main.py`), so the very first request after a restart is slow; `scripts/warm_model.py` only pre-warms the dense embedder, not the reranker or sparse encoder.
- **`chainlit.md`** still has Chainlit's default placeholder welcome text — customize it if you want a project-specific welcome screen.
- **`ADO_ORG`** is present in `.env` but not currently read anywhere in the code (`ADO_ORG_URL` is what's actually used).
