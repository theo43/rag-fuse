import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from app.config import Settings
from app.core import Passage

# `<=>` is pgvector's cosine distance (uses the HNSW index); similarity = 1 - distance.
SEARCH_SQL = """
    SELECT id, doc_id, title, content, 1 - (embedding <=> %(embedding)s) AS score
    FROM chunks
    ORDER BY embedding <=> %(embedding)s
    LIMIT %(top_k)s
"""


def load_model(name: str, expected_dim: int) -> SentenceTransformer:
    model = SentenceTransformer(name)
    dim = model.get_embedding_dimension()
    if dim != expected_dim:
        raise ValueError(f"{name} produces {dim}-d vectors, config says {expected_dim}")
    return model


def embed_query(model: SentenceTransformer, text: str) -> list[float]:
    # Normalized, like the passages at ingestion
    return model.encode(text, normalize_embeddings=True).tolist()


class PgVectorStore:
    """`VectorStore` backed by the `chunks` table. One connection per search: simple, fast enough locally."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def search(self, embedding: list[float], top_k: int) -> list[Passage]:
        with psycopg.connect(
            host=self.settings.postgres_host,
            port=self.settings.postgres_port,
            user=self.settings.postgres_user,
            password=self.settings.postgres_password,
            dbname=self.settings.postgres_db,
        ) as conn:
            register_vector(conn)
            rows = conn.execute(SEARCH_SQL, {"embedding": Vector(embedding), "top_k": top_k}).fetchall()
        return [Passage(*row) for row in rows]
