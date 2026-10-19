"""Evaluation metrics. No direct I/O."""

import random
from statistics import mean, median

SAMPLE_SIZE = 30
SEED = 0  # fixed: every run evaluates the same questions

VERDICTS = ["correct", "partial", "incorrect", "not_found"]
JUDGE_SYSTEM_PROMPT = """Tu évalues les réponses d'un assistant RH sur la convention collective Syntec.
On te donne la question, l'article de la convention qui y répond (la référence) et la réponse de l'assistant.
Verdict :
- correct : la réponse est juste et cohérente avec l'article de référence ;
- partial : la réponse est en partie juste, ou incomplète sur un point important ;
- incorrect : la réponse contredit l'article de référence ou invente des informations ;
- not_found : l'assistant dit ne pas avoir trouvé l'information.
Justifie en une phrase."""
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": VERDICTS}, "reason": {"type": "string"}},
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}


def sample_questions(questions: list[dict]) -> list[dict]:
    """A fixed random subset. Random, not the first ones: the questions are ordered by article."""
    subset = random.Random(SEED).sample(questions, SAMPLE_SIZE)
    return sorted(subset, key=lambda q: int(q["id"]))


def doc_id_of(chunk_id: str) -> str:
    return chunk_id.split("#")[0]  # chunk ids are "<doc_id>#<n>"


def answer_row(question: dict, response: dict, latency_s: float) -> dict:
    """One evaluated question: what was asked, what came back, and the retrieval metrics."""
    sources = response["sources"]
    return {
        **question,
        "answer": response["answer"],
        "sources": sources,
        "trace_id": response["trace_id"],
        "latency_s": round(latency_s, 2),
        "hit": question["gold_doc_id"] in {doc_id_of(s) for s in sources},  # gold article among the sources
        "fallback": not sources,  # no passage judged good enough
    }


def build_judge_prompt(question: str, reference: str, answer: str) -> str:
    return f"Question : {question}\n\nArticle de référence :\n{reference}\n\nRéponse de l'assistant :\n{answer}"


def trace_stats(observations: list[dict]) -> dict:
    """LLM calls, searches, tokens and cost of one question, from its Langfuse observations."""
    generations = [o for o in observations if o["type"] == "GENERATION"]
    return {
        "llm_calls": len(generations),
        "searches": sum(o["name"] == "retrieve" for o in observations),
        "input_tokens": sum(o.get("inputUsage") or 0 for o in generations),
        "output_tokens": sum(o.get("outputUsage") or 0 for o in generations),
        "cost_usd": sum(o.get("totalCost") or 0 for o in generations),
    }


def summarize(rows: list[dict]) -> dict:
    latencies = [r["latency_s"] for r in rows]
    return {
        "questions": len(rows),
        **{f"{verdict} rate": mean(r["verdict"] == verdict for r in rows) for verdict in VERDICTS},
        "hit rate": mean(r["hit"] for r in rows),
        "fallback rate": mean(r["fallback"] for r in rows),
        "latency mean (s)": mean(latencies),
        "latency median (s)": median(latencies),
        "LLM calls / question": mean(r["llm_calls"] for r in rows),
        "searches / question": mean(r["searches"] for r in rows),
        "tokens / question": mean(r["input_tokens"] + r["output_tokens"] for r in rows),
        "total cost (USD)": sum(r["cost_usd"] for r in rows),
    }


def to_markdown(summaries: dict[str, dict]) -> str:
    """One column per mode, one line per metric."""
    modes = list(summaries)
    lines = ["| metric | " + " | ".join(modes) + " |", "|---" * (len(modes) + 1) + "|"]
    for metric in summaries[modes[0]]:
        cells = [format_value(metric, summaries[mode][metric]) for mode in modes]
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_value(metric: str, value: float) -> str:
    if "rate" in metric:
        return f"{value:.0%}"
    return f"{value:,.0f}" if value >= 100 else f"{value:.3g}"  # 3,090 tokens; 4.2 s; 0.0123 USD
