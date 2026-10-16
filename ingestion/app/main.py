"""One-shot ingestion: load the corpus, chunk, embed, insert. Run with `python -m app.main`."""

from app import clients, core
from app.config import MAX_CHUNK_CHARS, Settings


def main() -> None:
    settings = Settings()

    corpus = clients.load_corpus()
    chunks = [
        chunk
        for doc in corpus
        for chunk in core.split_into_chunks(doc["_id"], doc["title"], doc["text"], MAX_CHUNK_CHARS)
    ]
    print(f"{len(corpus)} documents -> {len(chunks)} chunks")

    model = clients.load_model(settings.embedding_model, settings.embedding_dim)
    embeddings = clients.embed(model, [core.passage_text(c) for c in chunks])

    clients.insert_chunks(settings, chunks, embeddings)
    print("Done")


if __name__ == "__main__":
    main()
