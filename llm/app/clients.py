import anthropic
from langfuse import get_client


def make_client() -> anthropic.Anthropic:
    # Reads ANTHROPIC_API_KEY from the environment. The SDK retries connection errors,
    # 408/409/429 and 5xx itself (2 retries, exponential backoff).
    return anthropic.Anthropic()


def create_message(
    client: anthropic.Anthropic, params: dict, trace_context: dict | None
) -> anthropic.types.Message:
    """Call Claude. With a trace context, the call is recorded in that Langfuse trace, under its parent span.

    The Langfuse SDK instruments the Anthropic SDK by itself: every `messages.create` becomes a generation
    (model, input/output, tokens incl. thinking, cost). This span only attaches it to the caller's span.
    """
    if trace_context is None:
        return client.messages.create(**params)

    with get_client().start_as_current_observation(trace_context=trace_context, name="claude"):
        return client.messages.create(**params)
