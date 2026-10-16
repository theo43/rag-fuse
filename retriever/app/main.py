from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from app import clients, core
from app.config import Settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loaded once at startup: the model takes a few seconds to load.
    settings = Settings()
    model = clients.load_model(settings.embedding_model, settings.embedding_dim)
    app.state.embed = lambda text: clients.embed_query(model, text)
    app.state.store = clients.PgVectorStore(settings)
    yield


app = FastAPI(title="retriever", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResponse(BaseModel):
    results: list[core.Passage]  # chunks, so one article can appear several times


@app.post("/search")
def search(body: SearchRequest, request: Request) -> SearchResponse:
    passages = core.search(body.query, body.top_k, request.app.state.embed, request.app.state.store)
    return SearchResponse(results=passages)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
