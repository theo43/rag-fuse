from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Read from environment variables (names are case-insensitive)."""

    # From the embedding-config ConfigMap (must match what ingestion used)
    embedding_model: str
    embedding_dim: int

    # From the postgres-credentials Secret
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str = "postgres"
    postgres_port: int = 5432
