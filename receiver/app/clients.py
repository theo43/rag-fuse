import httpx

TIMEOUT = 120.0  # seconds; the orchestrator itself waits up to 60 s per upstream call


def make_http(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=TIMEOUT)


def ask_orchestrator(http: httpx.Client, question: str, trace_id: str) -> dict:
    response = http.post("/ask", json={"question": question}, headers={"X-Trace-Id": trace_id})
    response.raise_for_status()
    return response.json()
