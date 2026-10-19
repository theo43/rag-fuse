from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field

from app import clients, core
from app.config import Settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.graph = core.build_graph(
        clients.HttpLLMClient(settings.llm_url),
        clients.make_search(settings.retriever_url),
        settings.top_k,
        settings.max_attempts,
        settings.graph_mode,
    )
    yield
    get_client().flush()  # traces are sent in the background: send what is left before exiting


app = FastAPI(title="orchestrator", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]  # ids of the chunks the answer is based on (empty for the fallback answer)
    trace_id: str | None  # Langfuse trace of this run


@app.post("/ask")
def ask(body: AskRequest, request: Request, x_trace_id: str | None = Header(default=None)) -> AskResponse:
    # One handler per request: records every node as a span. With the receiver's X-Trace-Id the spans go
    # into that trace; without it (orchestrator called directly), Langfuse creates a new trace id.
    langfuse_handler = CallbackHandler(trace_context={"trace_id": x_trace_id} if x_trace_id else None)
    try:
        final = request.app.state.graph.invoke(
            core.initial_state(body.question),
            config={
                "callbacks": [langfuse_handler],
                "run_name": "ask",
                "configurable": {"trace_id": x_trace_id},  # forwarded by the nodes to retriever and llm
            },
        )
    except httpx.HTTPError as e:  # retriever or llm unreachable or returned an error
        raise HTTPException(status_code=502, detail=f"upstream service error: {e}")
    return AskResponse(
        answer=final["answer"],
        sources=[doc["id"] for doc in final["docs"]],
        trace_id=langfuse_handler.last_trace_id,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
