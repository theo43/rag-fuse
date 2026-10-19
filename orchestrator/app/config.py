from pydantic_settings import BaseSettings

from app.core import GraphMode


class Settings(BaseSettings):
    """Read from environment variables (names are case-insensitive).

    LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL (Secret) are read by the Langfuse SDK itself.
    """

    retriever_url: str = "http://retriever"
    llm_url: str = "http://llm"
    top_k: int = 5
    max_attempts: int = 2  # query rewrites before giving up (fallback answer)
    graph_mode: GraphMode = "agentic"  # "linear" = step 1 graph, kept to compare both (phase 7)
