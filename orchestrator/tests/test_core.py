from app.core import (
    FALLBACK_ANSWER,
    GRADE_SCHEMA,
    REWRITE_SCHEMA,
    SYSTEM_PROMPT,
    build_graph,
    build_prompt,
    initial_state,
    route_after_grade,
)

DOCS = [{"id": "article-10#0", "doc_id": "article-10", "title": "Article 10 : CDD",
         "content": "La convention s'applique aux CDD.", "score": 0.9}]


class FakeLLM:
    """`grades`: the answers of the successive `grade_docs` calls (True = relevant)."""

    def __init__(self, grades=(True,)):
        self.grades = list(grades)
        self.calls = []  # (method, prompt, trace_id)

    def generate(self, prompt, system, trace_id):
        self.calls.append(("generate", prompt, trace_id))
        assert system == SYSTEM_PROMPT
        return "Oui (Article 10)."

    def structured(self, prompt, system, schema, trace_id):
        if schema == GRADE_SCHEMA:
            self.calls.append(("grade", prompt, trace_id))
            return {"relevant": self.grades.pop(0)}
        assert schema == REWRITE_SCHEMA
        self.calls.append(("rewrite", prompt, trace_id))
        return {"query": f"réécriture {sum(c[0] == 'rewrite' for c in self.calls)}"}


class FakeSearch:
    def __init__(self):
        self.calls = []

    def __call__(self, query, top_k, trace_id):
        self.calls.append((query, top_k, trace_id))
        return DOCS


def run(llm, search, question="q", max_attempts=2, config=None):
    return build_graph(llm, search, top_k=3, max_attempts=max_attempts).invoke(initial_state(question), config=config)


def test_route_after_grade():
    state = initial_state("q")
    assert route_after_grade({**state, "docs_relevant": True, "attempts": 2}, max_attempts=2) == "generate"
    assert route_after_grade({**state, "docs_relevant": False, "attempts": 1}, max_attempts=2) == "rewrite_query"
    assert route_after_grade({**state, "docs_relevant": False, "attempts": 2}, max_attempts=2) == "fallback"


def test_relevant_docs_go_straight_to_generate():
    llm, search = FakeLLM(grades=[True]), FakeSearch()
    final = run(llm, search, question="Les CDD sont-ils couverts ?")
    assert [c[0] for c in llm.calls] == ["grade", "generate"]
    assert llm.calls[1][1] == build_prompt("Les CDD sont-ils couverts ?", DOCS)
    assert search.calls == [("Les CDD sont-ils couverts ?", 3, None)]
    assert final["answer"] == "Oui (Article 10)."
    assert final["attempts"] == 0


def test_irrelevant_docs_rewrite_the_query_and_search_again():
    llm, search = FakeLLM(grades=[False, True]), FakeSearch()
    final = run(llm, search)
    assert [c[0] for c in llm.calls] == ["grade", "rewrite", "grade", "generate"]
    assert [query for query, _, _ in search.calls] == ["q", "réécriture 1"]  # retrieve uses the rewritten query
    assert final["question"] == "q"
    assert final["attempts"] == 1
    assert final["answer"] == "Oui (Article 10)."


def test_fallback_after_max_attempts():
    llm, search = FakeLLM(grades=[False, False, False]), FakeSearch()
    final = run(llm, search, max_attempts=2)
    assert [c[0] for c in llm.calls] == ["grade", "rewrite", "grade", "rewrite", "grade"]
    assert len(search.calls) == 3
    assert final["attempts"] == 2
    assert final["docs_relevant"] is False
    assert final["answer"] == FALLBACK_ANSWER


def test_trace_id_from_run_config_reaches_search_and_llm():
    llm, search = FakeLLM(grades=[False, True]), FakeSearch()
    run(llm, search, config={"configurable": {"trace_id": "abc123"}})
    assert {trace_id for _, _, trace_id in search.calls} == {"abc123"}
    assert {trace_id for _, _, trace_id in llm.calls} == {"abc123"}
