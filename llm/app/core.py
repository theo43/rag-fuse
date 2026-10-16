from dataclasses import dataclass

DEFAULT_MAX_TOKENS = 16000


class LLMError(Exception):
    """The model answered, but not with something usable (refusal, truncated output)."""


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


def build_params(
    model: str,
    prompt: str,
    system: str | None,
    max_tokens: int,
    schema: dict | None = None,
) -> dict:
    """Arguments for `messages.create`. With a `schema`, the API guarantees JSON matching it."""
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        params["system"] = system
    if schema is not None:
        params["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
    return params


def read_result(response) -> LLMResult:
    """Extract the answer from a Messages API response, or raise `LLMError`."""
    if response.stop_reason == "refusal":
        category = response.stop_details.category if response.stop_details else None
        raise LLMError(f"model refused to answer (category: {category})")
    if response.stop_reason == "max_tokens":
        raise LLMError("output truncated: max_tokens reached")

    # The response can start with (empty) thinking blocks: keep only the text.
    text = "".join(block.text for block in response.content if block.type == "text")
    return LLMResult(text, response.model, response.usage.input_tokens, response.usage.output_tokens)
