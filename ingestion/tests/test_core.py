from app.core import Chunk, passage_text, split_into_chunks


def test_short_text_gives_one_chunk():
    chunks = split_into_chunks("article-1", "Article 1", "Un.\nDeux.", max_chars=100)
    assert chunks == [Chunk("article-1#0", "article-1", "Article 1", "Un.\nDeux.")]


def test_paragraphs_are_grouped_up_to_the_limit():
    text = "aaaa\nbbbb\ncccc"  # "aaaa\nbbbb" is 9 chars
    chunks = split_into_chunks("d", "t", text, max_chars=9)
    assert [c.content for c in chunks] == ["aaaa\nbbbb", "cccc"]
    assert [c.id for c in chunks] == ["d#0", "d#1"]


def test_long_paragraph_is_kept_whole():
    chunks = split_into_chunks("d", "t", "short\n" + "x" * 50, max_chars=10)
    assert [c.content for c in chunks] == ["short", "x" * 50]


def test_blank_lines_are_ignored():
    chunks = split_into_chunks("d", "t", "\n  \nUn.\n\n", max_chars=100)
    assert [c.content for c in chunks] == ["Un."]


def test_passage_text_has_e5_prefix_and_title():
    chunk = Chunk("d#0", "d", "Article 10", "Contenu")
    assert passage_text(chunk) == "passage: Article 10\nContenu"
