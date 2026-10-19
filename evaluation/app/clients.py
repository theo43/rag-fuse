import time
from datetime import datetime, timezone

import httpx
from datasets import load_dataset

from app.config import Settings

DATASET_NAME = "mteb/SyntecRetrieval"
TIMEOUT = 180.0  # seconds; longer than the receiver's own 120 s
RATE_LIMIT_RETRIES = 10


def load_questions() -> list[dict]:
    """The dataset's 100 questions, each with the id of the one article that answers it."""
    gold = {row["query-id"]: row["corpus-id"] for row in load_dataset(DATASET_NAME, "qrels", split="test")}
    queries = load_dataset(DATASET_NAME, "queries", split="test")
    return [{"id": q["_id"], "question": q["text"], "gold_doc_id": gold[q["_id"]]} for q in queries]


def load_articles() -> dict[str, str]:
    """Full text of every article (title + text), by doc id: the reference the judge compares against."""
    return {doc["_id"]: f"{doc['title']}\n{doc['text']}" for doc in load_dataset(DATASET_NAME, "corpus", split="test")}


def make_receiver(settings: Settings) -> httpx.Client:
    return httpx.Client(base_url=settings.receiver_url, timeout=TIMEOUT)


def make_llm(settings: Settings) -> httpx.Client:
    return httpx.Client(base_url=settings.llm_url, timeout=TIMEOUT)


def judge(llm: httpx.Client, prompt: str, system: str, schema: dict) -> dict:
    """Structured call to the llm service. No trace header: judge calls stay out of the evaluated traces."""
    response = llm.post("/structured", json={"prompt": prompt, "system": system, "schema": schema})
    response.raise_for_status()
    return response.json()["data"]


def ask(receiver: httpx.Client, question: str) -> tuple[dict, float]:
    """The receiver's response and its latency in seconds (end-to-end, as a user sees it)."""
    start = time.perf_counter()
    response = receiver.post("/ask", json={"question": question})
    response.raise_for_status()
    return response.json(), time.perf_counter() - start


def fetch_observations(settings: Settings, trace_id: str, since: datetime) -> list[dict]:
    """All observations of a trace (Langfuse v2 API: the legacy GET /traces/{id} is off for new orgs).

    The Hobby plan rate-limits the public API: on 429, wait (Retry-After if given) and try again.
    """
    for _ in range(RATE_LIMIT_RETRIES):
        response = httpx.get(
            f"{settings.langfuse_base_url}/api/public/v2/observations",
            params={"traceId": trace_id, "fromStartTime": since.isoformat(),
                    "toStartTime": datetime.now(timezone.utc).isoformat(), "limit": 100, "fields": "core,basic,usage"},
            auth=(settings.langfuse_public_key, settings.langfuse_secret_key),
            timeout=30,
        )
        if response.status_code != 429:
            break
        time.sleep(float(response.headers.get("Retry-After", 10)))
    response.raise_for_status()
    return response.json()["data"]
