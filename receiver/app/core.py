import uuid


def new_trace_id() -> str:
    """32 lowercase hex characters: the W3C trace id format Langfuse expects."""
    return uuid.uuid4().hex
