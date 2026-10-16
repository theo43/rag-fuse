"""The LangGraph graph (step 1, linear): START -> retrieve -> generate -> END. No direct I/O."""

from collections.abc import Callable
from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

SYSTEM_PROMPT = """Tu es un assistant RH spécialiste de la convention collective Syntec.
Réponds en français, uniquement à partir des extraits fournis, et cite les articles utilisés (ex. « Article 10 »).
Si les extraits ne permettent pas de répondre, dis-le clairement au lieu d'inventer."""


class State(TypedDict):
    question: str  # the user's question, never modified
    query: str  # what `retrieve` searches with (starts as the question; rewritten in step 2)
    docs: list[dict]  # passages from the retriever: id, doc_id, title, content, score
    answer: str


class LLMClient(Protocol):
    def generate(self, prompt: str, system: str) -> str: ...


Search = Callable[[str, int], list[dict]]  # (query, top_k) -> passages


def initial_state(question: str) -> State:
    return {"question": question, "query": question, "docs": [], "answer": ""}


def build_prompt(question: str, docs: list[dict]) -> str:
    excerpts = "\n\n".join(f"[{d['id']}] {d['title']}\n{d['content']}" for d in docs)
    return f"Extraits de la convention :\n\n{excerpts}\n\nQuestion : {question}"


def build_graph(llm: LLMClient, search: Search, top_k: int) -> CompiledStateGraph:
    def retrieve(state: State) -> dict:
        return {"docs": search(state["query"], top_k)}

    def generate(state: State) -> dict:
        prompt = build_prompt(state["question"], state["docs"])
        return {"answer": llm.generate(prompt, SYSTEM_PROMPT)}

    graph = StateGraph(State)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
