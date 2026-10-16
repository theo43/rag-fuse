import json
import re

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client_with_fake_orchestrator():
    """Returns a function: handler -> TestClient whose orchestrator calls go to `handler`, not the network."""
    with TestClient(app) as test_client:  # runs the lifespan

        def make(handler) -> TestClient:
            app.state.http = httpx.Client(base_url="http://orchestrator", transport=httpx.MockTransport(handler))
            return test_client

        yield make


def test_trace_id_is_generated_and_sent_to_the_orchestrator(client_with_fake_orchestrator):
    seen = {}

    def orchestrator(request: httpx.Request) -> httpx.Response:
        seen["trace_id"] = request.headers["X-Trace-Id"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"answer": "Oui.", "sources": ["article-10#0"],
                                         "trace_id": seen["trace_id"]})

    response = client_with_fake_orchestrator(orchestrator).post("/ask", json={"question": "CDD ?"})

    assert response.status_code == 200
    assert re.fullmatch(r"[0-9a-f]{32}", seen["trace_id"])
    assert seen["body"] == {"question": "CDD ?"}
    assert response.json() == {"answer": "Oui.", "sources": ["article-10#0"], "trace_id": seen["trace_id"]}


def test_each_request_gets_a_new_trace_id(client_with_fake_orchestrator):
    ids = []

    def orchestrator(request):
        ids.append(request.headers["X-Trace-Id"])
        return httpx.Response(200, json={"answer": "a", "sources": [], "trace_id": ids[-1]})

    client = client_with_fake_orchestrator(orchestrator)
    client.post("/ask", json={"question": "q"})
    client.post("/ask", json={"question": "q"})
    assert ids[0] != ids[1]


def test_orchestrator_error_becomes_502(client_with_fake_orchestrator):
    client = client_with_fake_orchestrator(lambda request: httpx.Response(500))
    response = client.post("/ask", json={"question": "q"})
    assert response.status_code == 502


def test_question_is_validated(client_with_fake_orchestrator):
    client = client_with_fake_orchestrator(lambda request: httpx.Response(500))
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 1001}).status_code == 422
