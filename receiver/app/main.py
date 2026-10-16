"""Public entry point: validates the request, creates the trace id, forwards to the orchestrator."""

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app import clients, core
from app.config import Settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = clients.make_http(Settings().orchestrator_url)
    yield


app = FastAPI(title="receiver", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    trace_id: str


@app.post("/ask")
def ask(body: AskRequest, request: Request) -> AskResponse:
    trace_id = core.new_trace_id()
    try:
        result = clients.ask_orchestrator(request.app.state.http, body.question, trace_id)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"orchestrator error (trace_id={trace_id}): {e}")
    return AskResponse(**result)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
