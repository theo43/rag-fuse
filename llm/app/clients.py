import anthropic
from langfuse import get_client


def make_client() -> anthropic.Anthropic:
    # Reads ANTHROPIC_API_KEY from the environment. The SDK retries connection errors,
    # 408/409/429 and 5xx itself (2 retries, exponential backoff).
    return anthropic.Anthropic()


def create_message(client: anthropic.Anthropic, params: dict, trace_id: str | None) -> anthropic.types.Message:
    """Call Claude. With a trace id, also record the call as a Langfuse generation in that trace."""
    if trace_id is None:
        return client.messages.create(**params)

    with get_client().start_as_current_observation(
        trace_context={"trace_id": trace_id},
        name="claude",
        as_type="generation",
        model=params["model"],
        input={"system": params.get("system"), "messages": params["messages"]},
    ) as generation:
        response = client.messages.create(**params)
        generation.update(
            output=[block.model_dump() for block in response.content],
            usage_details={"input": response.usage.input_tokens, "output": response.usage.output_tokens},
            metadata={"stop_reason": response.stop_reason},
        )
        return response
