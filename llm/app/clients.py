import anthropic


def make_client() -> anthropic.Anthropic:
    # Reads ANTHROPIC_API_KEY from the environment. The SDK retries connection errors,
    # 408/409/429 and 5xx itself (2 retries, exponential backoff).
    return anthropic.Anthropic()


def create_message(client: anthropic.Anthropic, params: dict) -> anthropic.types.Message:
    return client.messages.create(**params)
