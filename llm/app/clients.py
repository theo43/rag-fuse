import anthropic
from langfuse import get_client


def make_client() -> anthropic.Anthropic:
    # Reads ANTHROPIC_API_KEY from the environment. The SDK retries connection errors,
    # 408/409/429 and 5xx itself (2 retries, exponential backoff).
    return anthropic.Anthropic()


def create_message(client: anthropic.Anthropic, params: dict, trace_id: str | None) -> anthropic.types.Message:
    """Call Claude. With a trace id, the call is recorded in that Langfuse trace.

    The Langfuse SDK instruments the Anthropic SDK by itself: every `messages.create` becomes a generation
    (model, input/output, tokens incl. thinking, cost). This span only attaches it to the request's trace.
    """
    if trace_id is None:
        return client.messages.create(**params)

    with get_client().start_as_current_observation(trace_context={"trace_id": trace_id}, name="claude"):
        return client.messages.create(**params)
