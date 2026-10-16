from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Read from environment variables (names are case-insensitive)."""

    orchestrator_url: str = "http://orchestrator"
