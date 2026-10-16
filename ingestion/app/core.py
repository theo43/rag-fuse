from dataclasses import dataclass

PASSAGE_PREFIX = "passage: "  # e5 convention for indexed documents


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    content: str


def split_into_chunks(doc_id: str, title: str, text: str, max_chars: int) -> list[Chunk]:
    """Group consecutive paragraphs into chunks of at most `max_chars` characters.

    Paragraphs are never cut: one longer than `max_chars` becomes a chunk on its own.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    groups: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + 1 + len(paragraph) > max_chars:
            groups.append(current)
            current = paragraph
        else:
            current = f"{current}\n{paragraph}" if current else paragraph
    if current:
        groups.append(current)

    return [Chunk(f"{doc_id}#{i}", doc_id, title, group) for i, group in enumerate(groups)]


def passage_text(chunk: Chunk) -> str:
    """Text given to the embedding model: prefix + title (for context) + chunk content."""
    return f"{PASSAGE_PREFIX}{chunk.title}\n{chunk.content}"
