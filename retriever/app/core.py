from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

QUERY_PREFIX = "query: "  # e5 convention for search queries (ingestion uses "passage: ")


@dataclass
class Passage:
    id: str
    doc_id: str
    title: str
    content: str
    score: float  # cosine similarity, higher is better


class VectorStore(Protocol):
    def search(self, embedding: list[float], top_k: int) -> list[Passage]: ...


def search(
    query: str,
    top_k: int,
    embed: Callable[[str], list[float]],
    store: VectorStore,
) -> list[Passage]:
    """Embed the query with the e5 prefix, then return the `top_k` closest passages."""
    return store.search(embed(QUERY_PREFIX + query), top_k)
