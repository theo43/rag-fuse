from app.core import SAMPLE_SIZE, answer_row, sample_questions, summarize, to_markdown, trace_stats

QUESTION = {"id": "7", "question": "Les CDD sont-ils couverts ?", "gold_doc_id": "article-10"}


def test_sample_is_fixed_and_sorted():
    questions = [{"id": str(i)} for i in range(100)]
    subset = sample_questions(questions)
    assert len(subset) == SAMPLE_SIZE
    assert subset == sample_questions(questions)  # same questions on every run
    assert [int(q["id"]) for q in subset] == sorted(int(q["id"]) for q in subset)


def test_answer_row_hit_is_document_level():
    response = {"answer": "Oui.", "sources": ["article-3#0", "article-10#2"], "trace_id": "t"}
    row = answer_row(QUESTION, response, 4.567)
    assert row["hit"] is True
    assert row["fallback"] is False
    assert row["latency_s"] == 4.57


def test_answer_row_fallback_has_no_hit():
    row = answer_row(QUESTION, {"answer": "Pas trouvé.", "sources": [], "trace_id": "t"}, 1.0)
    assert row["hit"] is False
    assert row["fallback"] is True


def test_trace_stats_counts_generations_and_searches():
    observations = [
        {"type": "CHAIN", "name": "retrieve"},
        {"type": "CHAIN", "name": "retrieve"},
        {"type": "SPAN", "name": "claude"},
        {"type": "GENERATION", "name": "anthropic.messages.create", "inputUsage": 100, "outputUsage": 10,
         "totalCost": 0.001},
        {"type": "GENERATION", "name": "anthropic.messages.create", "inputUsage": 200, "outputUsage": 20,
         "totalCost": 0.002},
    ]
    stats = trace_stats(observations)
    assert stats == {"llm_calls": 2, "searches": 2, "input_tokens": 300, "output_tokens": 30, "cost_usd": 0.003}


def test_summary_table():
    rows = [
        {"hit": True, "fallback": False, "latency_s": 2.0, "llm_calls": 1, "searches": 1,
         "input_tokens": 1000, "output_tokens": 200, "cost_usd": 0.001},
        {"hit": False, "fallback": True, "latency_s": 4.0, "llm_calls": 5, "searches": 3,
         "input_tokens": 3000, "output_tokens": 600, "cost_usd": 0.003},
    ]
    summary = summarize(rows)
    assert summary["hit rate"] == 0.5
    assert summary["latency median (s)"] == 3.0
    table = to_markdown({"linear": summary, "agentic": summary})
    assert table.splitlines()[0] == "| metric | linear | agentic |"
    assert "| hit rate | 50% | 50% |" in table
    assert "| tokens / question | 2,400 | 2,400 |" in table
