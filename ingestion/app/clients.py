import numpy as np
import psycopg
from datasets import load_dataset
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from app.config import DATASET_NAME, Settings
from app.core import Chunk


def load_corpus() -> list[dict]:
    """SyntecRetrieval articles, as dicts with keys `_id`, `title`, `text`."""
    return list(load_dataset(DATASET_NAME, "corpus", split="test"))


def load_model(name: str, expected_dim: int) -> SentenceTransformer:
    model = SentenceTransformer(name)
    dim = model.get_sentence_embedding_dimension()
    if dim != expected_dim:
        raise ValueError(f"{name} produces {dim}-d vectors, config says {expected_dim}")
    return model


def embed(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=True)


def insert_chunks(settings: Settings, chunks: list[Chunk], embeddings: np.ndarray) -> None:
    """Insert chunks; existing ids are skipped, so running the Job twice is harmless."""
    with psycopg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        dbname=settings.postgres_db,
    ) as conn:
        register_vector(conn)
        conn.cursor().executemany(
            """
            INSERT INTO chunks (id, doc_id, title, content, embedding)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            [
                (c.id, c.doc_id, c.title, c.content, emb)
                for c, emb in zip(chunks, embeddings, strict=True)
            ],
        )
