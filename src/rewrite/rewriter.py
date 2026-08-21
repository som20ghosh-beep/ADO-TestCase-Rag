"""Query rewriting via local Ollama LLM. Fails open: any error returns the raw query."""
import json
import time

from ollama import Client

from src.config import OLLAMA_HOST, REWRITE_MODEL, REWRITE_TIMEOUT_S
from src.rewrite.prompts import SYSTEM_PROMPT, build_user_prompt
# use your existing structlog/log setup
import structlog
log = structlog.get_logger()


class QueryRewriter:
    def __init__(self):
        self.client = Client(host=OLLAMA_HOST, timeout=REWRITE_TIMEOUT_S)
        self.available = self._check_available()

    def _check_available(self) -> bool:
        try:
            # Use a throwaway client, not self.client: reusing the same
            # connection for list() then chat() makes the first chat() call
            # hang until the read timeout (observed against Ollama on Windows).
            probe = Client(host=OLLAMA_HOST, timeout=REWRITE_TIMEOUT_S)
            models = [m.model for m in probe.list().models]
            ok = any(REWRITE_MODEL in m for m in models)
            if not ok:
                log.warning("rewrite_model_missing", model=REWRITE_MODEL)
            return ok
        except Exception as e:
            log.warning("ollama_unreachable", error=str(e))
            return False

    def rewrite(self, query: str) -> dict:
        """Returns {"rewritten": str, "keywords": list, "used_llm": bool, "latency_ms": int}.
        NEVER raises — falls back to the raw query on any failure."""
        fallback = {"rewritten": query, "keywords": [], "used_llm": False, "latency_ms": 0}

        if not self.available or len(query.split()) >= 25:
            # Long queries are already specific — rewriting adds little
            return fallback

        start = time.perf_counter()
        try:
            resp = self.client.chat(
                model=REWRITE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(query)},
                ],
                options={"temperature": 0.2, "num_predict": 150},
                format="json",          # Ollama enforces valid JSON output
            )
            data = json.loads(resp.message.content)
            rewritten = str(data.get("rewritten", "")).strip()
            keywords = [str(k) for k in data.get("keywords", [])][:6]

            # Sanity guards against a misbehaving small model
            if not rewritten or len(rewritten) > 400:
                raise ValueError("rewrite failed sanity check")

            latency = int((time.perf_counter() - start) * 1000)
            log.info("query_rewritten", original=query, rewritten=rewritten, latency_ms=latency)
            return {
                "rewritten": rewritten,
                "keywords": keywords,
                "used_llm": True,
                "latency_ms": latency,
            }
        except Exception as e:
            log.warning("rewrite_failed_fallback", query=query, error=str(e))
            return fallback