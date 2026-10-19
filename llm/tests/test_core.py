from types import SimpleNamespace

import pytest

from app.core import LLMError, LLMResult, build_params, parse_traceparent, read_result


def fake_response(content, stop_reason="end_turn", stop_details=None):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        stop_details=stop_details,
        model="claude-haiku-5-5",
        usage=SimpleNamespace(input_tokens=12, output_tokens=34),
    )


def text(value):
    return SimpleNamespace(type="text", text=value)


THINKING = SimpleNamespace(type="thinking", thinking="")


def test_build_params_minimal():
    assert build_params("m", "Bonjour", None, 100) == {
        "model": "m",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Bonjour"}],
    }


def test_build_params_with_system_and_schema():
    schema = {"type": "object"}
    params = build_params("m", "q", "Tu es un assistant RH.", 100, schema)
    assert params["system"] == "Tu es un assistant RH."
    assert params["output_config"] == {"format": {"type": "json_schema", "schema": schema}}


def test_read_result_skips_thinking_blocks():
    result = read_result(fake_response([THINKING, text("Réponse")]))
    assert result == LLMResult("Réponse", "claude-haiku-5-5", 12, 34)


def test_refusal_raises_with_category():
    response = fake_response([], "refusal", SimpleNamespace(category="cyber"))
    with pytest.raises(LLMError, match="cyber"):
        read_result(response)


def test_truncated_output_raises():
    with pytest.raises(LLMError, match="max_tokens"):
        read_result(fake_response([text('{"relevant": tr')], "max_tokens"))


def test_parse_traceparent():
    traceparent = "00-0123456789abcdef0123456789abcdef-89abcdef01234567-01"
    assert parse_traceparent(traceparent) == {
        "trace_id": "0123456789abcdef0123456789abcdef",
        "parent_span_id": "89abcdef01234567",
    }
