FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv

WORKDIR /app

# Dependencies first, so code changes do not reinstall them
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
ENV PATH="/app/.venv/bin:$PATH"

# Embedding model baked into the image. Must match EMBEDDING_MODEL in k8s/embedding-config.yaml.
ARG EMBEDDING_MODEL
RUN test -n "$EMBEDDING_MODEL" \
    && python -c "import sys; from sentence_transformers import SentenceTransformer; SentenceTransformer(sys.argv[1])" "$EMBEDDING_MODEL"
# Never download at runtime: a model mismatch with the ConfigMap fails at startup instead.
ENV HF_HUB_OFFLINE=1

COPY app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
