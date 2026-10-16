from pydantic_settings import BaseSettings

DATASET_NAME = "mteb/SyntecRetrieval"
MAX_CHUNK_CHARS = 1500  # ~400 tokens of French text, under e5-small's 512-token limit


class Settings(BaseSettings):
    """Read from environment variables (names are case-insensitive)."""

    # From the embedding-config ConfigMap
    embedding_model: str
    embedding_dim: int

    # From the postgres-credentials Secret
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str = "postgres"
    postgres_port: int = 5432
