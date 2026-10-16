from app.core import SYSTEM_PROMPT, build_graph, build_prompt, initial_state

DOCS = [{"id": "article-10#0", "doc_id": "article-10", "title": "Article 10 : CDD",
         "content": "La convention s'applique aux CDD.", "score": 0.9}]


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, system, trace_id):
        self.calls.append((prompt, system, trace_id))
        return "Oui (Article 10)."


class FakeSearch:
    def __init__(self):
        self.calls = []

    def __call__(self, query, top_k, trace_id):
        self.calls.append((query, top_k, trace_id))
        return DOCS


def test_retrieve_searches_with_query_not_question():
    search = FakeSearch()
    graph = build_graph(FakeLLM(), search, top_k=3)
    graph.invoke({**initial_state("question d'origine"), "query": "requête réécrite"})
    assert search.calls == [("requête réécrite", 3, None)]


def test_generate_gets_excerpts_and_question():
    llm = FakeLLM()
    build_graph(llm, FakeSearch(), top_k=3).invoke(initial_state("Les CDD sont-ils couverts ?"))
    [(prompt, system, _)] = llm.calls
    assert prompt == build_prompt("Les CDD sont-ils couverts ?", DOCS)
    assert "[article-10#0] Article 10 : CDD" in prompt
    assert system == SYSTEM_PROMPT


def test_final_state_has_docs_and_answer():
    final = build_graph(FakeLLM(), FakeSearch(), top_k=3).invoke(initial_state("q"))
    assert final["docs"] == DOCS
    assert final["answer"] == "Oui (Article 10)."
    assert final["query"] == "q"


def test_trace_id_from_run_config_reaches_search_and_llm():
    llm, search = FakeLLM(), FakeSearch()
    build_graph(llm, search, top_k=3).invoke(initial_state("q"), config={"configurable": {"trace_id": "abc123"}})
    assert search.calls[0][2] == "abc123"
    assert llm.calls[0][2] == "abc123"
