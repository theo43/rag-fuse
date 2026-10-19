from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan, SpanContext

from app.clients import trace_headers

TRACE_ID = "0123456789abcdef0123456789abcdef"


def test_trace_headers_carry_the_current_span_as_parent():
    span = NonRecordingSpan(SpanContext(trace_id=int(TRACE_ID, 16), span_id=0x89ABCDEF01234567, is_remote=False))
    with trace.use_span(span):
        headers = trace_headers(TRACE_ID)
    assert headers == {"X-Trace-Id": TRACE_ID, "traceparent": f"00-{TRACE_ID}-89abcdef01234567-01"}


def test_no_trace_id_no_headers():
    assert trace_headers(None) == {}
