import httpx

from app.core import Search

TIMEOUT = 60.0  # seconds; an LLM answer takes longer than httpx's 5 s default


class HttpLLMClient:
    """`LLMClient` backed by the llm service."""

    def __init__(self, base_url: str):
        self.http = httpx.Client(base_url=base_url, timeout=TIMEOUT)

    def generate(self, prompt: str, system: str) -> str:
        response = self.http.post("/generate", json={"prompt": prompt, "system": system})
        response.raise_for_status()
        return response.json()["text"]


def make_search(base_url: str) -> Search:
    """`search(query, top_k)` backed by the retriever service."""
    http = httpx.Client(base_url=base_url, timeout=TIMEOUT)

    def search(query: str, top_k: int) -> list[dict]:
        response = http.post("/search", json={"query": query, "top_k": top_k})
        response.raise_for_status()
        return response.json()["results"]

    return search
