import json
from contextlib import asynccontextmanager

import anthropic
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app import clients, core
from app.config import Settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = Settings().anthropic_model
    app.state.client = clients.make_client()
    yield


app = FastAPI(title="llm", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    system: str | None = None
    max_tokens: int = Field(default=core.DEFAULT_MAX_TOKENS, ge=1, le=core.DEFAULT_MAX_TOKENS)


class StructuredRequest(GenerateRequest):
    schema_: dict = Field(alias="schema")  # JSON Schema the answer must follow


class Usage(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int


class GenerateResponse(Usage):
    text: str


class StructuredResponse(Usage):
    data: dict


def call_claude(request: Request, body: GenerateRequest, schema: dict | None = None) -> core.LLMResult:
    params = core.build_params(request.app.state.model, body.prompt, body.system, body.max_tokens, schema)
    try:
        return core.read_result(clients.create_message(request.app.state.client, params))
    except core.LLMError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except anthropic.BadRequestError as e:  # e.g. an invalid JSON schema
        raise HTTPException(status_code=400, detail=e.message)


@app.post("/generate")
def generate(body: GenerateRequest, request: Request) -> GenerateResponse:
    result = call_claude(request, body)
    return GenerateResponse(text=result.text, model=result.model,
                            input_tokens=result.input_tokens, output_tokens=result.output_tokens)


@app.post("/structured")
def structured(body: StructuredRequest, request: Request) -> StructuredResponse:
    result = call_claude(request, body, body.schema_)
    return StructuredResponse(data=json.loads(result.text), model=result.model,
                              input_tokens=result.input_tokens, output_tokens=result.output_tokens)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
