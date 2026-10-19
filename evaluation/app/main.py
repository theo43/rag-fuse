"""Evaluate the deployed RAG on a fixed subset of SyntecRetrieval questions.

    uv run python -m app.main run --mode linear    # after: kubectl set env deployment/orchestrator GRAPH_MODE=linear
    uv run python -m app.main run --mode agentic   # after: kubectl set env deployment/orchestrator GRAPH_MODE=agentic
    uv run python -m app.main summary              # compares every results/<mode>.jsonl
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app import clients, core
from app.config import Settings

RESULTS_DIR = Path(__file__).parent.parent / "results"
LANGFUSE_DELAY = 20  # seconds; traces are sent in the background, give Langfuse time to ingest the last ones


def run(mode: str) -> None:
    """`mode` only names the results file: the orchestrator's GRAPH_MODE decides which graph runs."""
    settings = Settings()
    receiver, llm = clients.make_receiver(settings), clients.make_llm(settings)
    questions = core.sample_questions(clients.load_questions())
    articles = clients.load_articles()
    started = datetime.now(timezone.utc)

    rows = []
    for i, question in enumerate(questions, 1):
        try:
            response, latency_s = clients.ask(receiver, question["question"])
            row = core.answer_row(question, response, latency_s)
            prompt = core.build_judge_prompt(question["question"], articles[question["gold_doc_id"]], row["answer"])
            row.update(clients.judge(llm, prompt, core.JUDGE_SYSTEM_PROMPT, core.JUDGE_SCHEMA))
        except httpx.HTTPError as e:
            print(f"[{i}/{len(questions)}] question {question['id']} skipped: {e}")
            continue
        rows.append(row)
        print(f"[{i}/{len(questions)}] {latency_s:.1f}s hit={row['hit']} {row['verdict']}: {question['question'][:60]}")

    path = RESULTS_DIR / f"{mode}.jsonl"
    write_rows(path, rows)  # answers and verdicts are kept even if fetching the stats fails

    time.sleep(LANGFUSE_DELAY)
    for row in rows:
        row.update(core.trace_stats(clients.fetch_observations(settings, row["trace_id"], started)))
        if row["llm_calls"] == 0:
            print(f"warning: trace {row['trace_id']} has no LLM call in Langfuse (not ingested yet?)")

    write_rows(path, rows)
    print(f"{len(rows)} results written to {path}")


def write_rows(path: Path, rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def summary() -> None:
    summaries = {}
    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        summaries[path.stem] = core.summarize(rows)
    table = core.to_markdown(summaries)
    (RESULTS_DIR / "summary.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run").add_argument("--mode", required=True, choices=["linear", "agentic"])
    commands.add_parser("summary")
    args = parser.parse_args()
    run(args.mode) if args.command == "run" else summary()
