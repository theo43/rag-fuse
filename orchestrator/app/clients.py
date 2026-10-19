import httpx
from opentelemetry import trace

from app.core import Search

TIMEOUT = 60.0  # seconds; an LLM answer takes longer than httpx's 5 s default


def trace_headers(trace_id: str | None) -> dict:
    """Propagate the request's trace to the downstream services.

    `traceparent` (W3C) also carries the id of the current span: the Langfuse span of the calling node,
    so that the callee's spans nest under it. Format: version-trace_id-parent_span_id-flags.
    """
    if not trace_id:
        return {}
    span_id = trace.get_current_span().get_span_context().span_id
    return {"X-Trace-Id": trace_id, "traceparent": f"00-{trace_id}-{span_id:016x}-01"}


class HttpLLMClient:
    """`LLMClient` backed by the llm service."""

    def __init__(self, base_url: str):
        self.http = httpx.Client(base_url=base_url, timeout=TIMEOUT)

    def generate(self, prompt: str, system: str, trace_id: str | None) -> str:
        response = self.http.post("/generate", json={"prompt": prompt, "system": system},
                                  headers=trace_headers(trace_id))
        response.raise_for_status()
        return response.json()["text"]


def make_search(base_url: str) -> Search:
    """`search(query, top_k, trace_id)` backed by the retriever service."""
    http = httpx.Client(base_url=base_url, timeout=TIMEOUT)

    def search(query: str, top_k: int, trace_id: str | None) -> list[dict]:
        response = http.post("/search", json={"query": query, "top_k": top_k}, headers=trace_headers(trace_id))
        response.raise_for_status()
        return response.json()["results"]

    return search
