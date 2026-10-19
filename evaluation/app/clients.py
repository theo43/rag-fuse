import time
from datetime import datetime, timezone

import httpx
from datasets import load_dataset

from app.config import Settings

DATASET_NAME = "mteb/SyntecRetrieval"
TIMEOUT = 180.0  # seconds; longer than the receiver's own 120 s


def load_questions() -> list[dict]:
    """The dataset's 100 questions, each with the id of the one article that answers it."""
    gold = {row["query-id"]: row["corpus-id"] for row in load_dataset(DATASET_NAME, "qrels", split="test")}
    queries = load_dataset(DATASET_NAME, "queries", split="test")
    return [{"id": q["_id"], "question": q["text"], "gold_doc_id": gold[q["_id"]]} for q in queries]


def make_receiver(settings: Settings) -> httpx.Client:
    return httpx.Client(base_url=settings.receiver_url, timeout=TIMEOUT)


def ask(receiver: httpx.Client, question: str) -> tuple[dict, float]:
    """The receiver's response and its latency in seconds (end-to-end, as a user sees it)."""
    start = time.perf_counter()
    response = receiver.post("/ask", json={"question": question})
    response.raise_for_status()
    return response.json(), time.perf_counter() - start


def fetch_observations(settings: Settings, trace_id: str, since: datetime) -> list[dict]:
    """All observations of a trace (Langfuse v2 API: the legacy GET /traces/{id} is off for new orgs)."""
    response = httpx.get(
        f"{settings.langfuse_base_url}/api/public/v2/observations",
        params={"traceId": trace_id, "fromStartTime": since.isoformat(),
                "toStartTime": datetime.now(timezone.utc).isoformat(), "limit": 100, "fields": "core,basic,usage"},
        auth=(settings.langfuse_public_key, settings.langfuse_secret_key),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["data"]
