from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Read from environment variables (names are case-insensitive).

    The services are reached through `kubectl port-forward` (see README). The Langfuse keys are the ones of
    the `langfuse-credentials` Secret, exported in the shell.
    """

    receiver_url: str = "http://localhost:8080"  # kubectl port-forward svc/receiver 8080:80
    llm_url: str = "http://localhost:8081"  # kubectl port-forward svc/llm 8081:80 (the judge)
    langfuse_base_url: str
    langfuse_public_key: str
    langfuse_secret_key: str
