from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Read from environment variables (names are case-insensitive).

    LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL (Secret) are read by the Langfuse SDK itself.
    """

    retriever_url: str = "http://retriever"
    llm_url: str = "http://llm"
    top_k: int = 5
