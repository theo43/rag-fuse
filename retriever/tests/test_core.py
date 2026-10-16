from app.core import Passage, search

PASSAGES = [
    Passage("article-1#0", "article-1", "Article 1", "Champ d'application", 0.9),
    Passage("article-2#0", "article-2", "Article 2", "Définitions", 0.8),
]


class FakeStore:
    def __init__(self):
        self.calls = []

    def search(self, embedding, top_k):
        self.calls.append((embedding, top_k))
        return PASSAGES[:top_k]


def fake_embed(text):
    fake_embed.last_text = text
    return [0.1, 0.2]


def test_query_gets_e5_prefix():
    search("Que veut-dire IC ?", 5, fake_embed, FakeStore())
    assert fake_embed.last_text == "query: Que veut-dire IC ?"


def test_embedding_and_top_k_are_forwarded_to_the_store():
    store = FakeStore()
    search("question", 1, fake_embed, store)
    assert store.calls == [([0.1, 0.2], 1)]


def test_results_come_back_in_store_order():
    assert search("question", 2, fake_embed, FakeStore()) == PASSAGES
