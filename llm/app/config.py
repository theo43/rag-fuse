from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Read from environment variables (names are case-insensitive)."""

    # From the llm-config ConfigMap. ANTHROPIC_API_KEY (Secret) is read by the SDK itself.
    anthropic_model: str
