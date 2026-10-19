"""The LangGraph graph (step 2): grade the retrieved passages, rewrite the query and retry if needed.

START -> retrieve -> grade_docs -(relevant)-> generate -> END
             ^           |-(not relevant)-> rewrite_query -> retrieve
                         |-(attempts >= max_attempts)-> fallback -> END

No direct I/O.
"""

from collections.abc import Callable
from typing import Literal, Protocol, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

SYSTEM_PROMPT = """Tu es un assistant RH spécialiste de la convention collective Syntec.
Réponds en français, uniquement à partir des extraits fournis, et cite les articles utilisés (ex. « Article 10 »).
Si les extraits ne permettent pas de répondre, dis-le clairement au lieu d'inventer."""

GRADE_SYSTEM_PROMPT = """Tu évalues des extraits de la convention collective Syntec.
Indique si au moins un extrait contient l'information nécessaire pour répondre à la question."""
GRADE_SCHEMA = {
    "type": "object",
    "properties": {"relevant": {"type": "boolean"}},
    "required": ["relevant"],
    "additionalProperties": False,
}

REWRITE_SYSTEM_PROMPT = """Tu reformules des requêtes de recherche dans la convention collective Syntec.
La requête précédente n'a pas trouvé d'extraits pertinents : propose une nouvelle requête en français,
avec d'autres termes (vocabulaire de la convention, synonymes), sans changer le sens de la question."""
REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}

FALLBACK_ANSWER = (
    "Je n'ai pas trouvé dans la convention collective Syntec d'extraits permettant de répondre à cette question."
)


class State(TypedDict):
    question: str  # the user's question, never modified
    query: str  # what `retrieve` searches with (starts as the question, then rewritten)
    docs: list[dict]  # passages from the retriever: id, doc_id, title, content, score
    docs_relevant: bool  # set by `grade_docs`
    attempts: int  # number of query rewrites so far
    answer: str


class LLMClient(Protocol):
    def generate(self, prompt: str, system: str, trace_id: str | None) -> str: ...

    def structured(self, prompt: str, system: str, schema: dict, trace_id: str | None) -> dict:
        """JSON object matching `schema` (a JSON Schema)."""
        ...


Search = Callable[[str, int, str | None], list[dict]]  # (query, top_k, trace_id) -> passages
GraphMode = Literal["linear", "agentic"]


def initial_state(question: str) -> State:
    return {"question": question, "query": question, "docs": [], "docs_relevant": False, "attempts": 0, "answer": ""}


def build_prompt(question: str, docs: list[dict]) -> str:
    excerpts = "\n\n".join(f"[{d['id']}] {d['title']}\n{d['content']}" for d in docs)
    return f"Extraits de la convention :\n\n{excerpts}\n\nQuestion : {question}"


def build_rewrite_prompt(question: str, query: str) -> str:
    return f"Question : {question}\nRequête précédente : {query}"


def route_after_grade(state: State, max_attempts: int) -> str:
    """Next node after `grade_docs`. Only reads the state."""
    if state["docs_relevant"]:
        return "generate"
    if state["attempts"] >= max_attempts:
        return "fallback"
    return "rewrite_query"


def trace_id_of(config: RunnableConfig) -> str | None:
    """The request's trace id: request metadata, passed in the run config rather than the State."""
    return config.get("configurable", {}).get("trace_id")


def build_graph(
    llm: LLMClient, search: Search, top_k: int, max_attempts: int, mode: GraphMode = "agentic"
) -> CompiledStateGraph:
    """`agentic`: the step 2 graph above. `linear`: step 1, START -> retrieve -> generate -> END (for evaluation)."""

    # LangGraph passes the run config to nodes that declare a `config` parameter.
    def retrieve(state: State, config: RunnableConfig) -> dict:
        return {"docs": search(state["query"], top_k, trace_id_of(config))}

    def grade_docs(state: State, config: RunnableConfig) -> dict:
        prompt = build_prompt(state["question"], state["docs"])
        grade = llm.structured(prompt, GRADE_SYSTEM_PROMPT, GRADE_SCHEMA, trace_id_of(config))
        return {"docs_relevant": grade["relevant"]}

    def rewrite_query(state: State, config: RunnableConfig) -> dict:
        prompt = build_rewrite_prompt(state["question"], state["query"])
        rewrite = llm.structured(prompt, REWRITE_SYSTEM_PROMPT, REWRITE_SCHEMA, trace_id_of(config))
        return {"query": rewrite["query"], "attempts": state["attempts"] + 1}

    def generate(state: State, config: RunnableConfig) -> dict:
        prompt = build_prompt(state["question"], state["docs"])
        return {"answer": llm.generate(prompt, SYSTEM_PROMPT, trace_id_of(config))}

    def fallback(state: State) -> dict:
        return {"answer": FALLBACK_ANSWER, "docs": []}  # the answer is based on no passage

    graph = StateGraph(State)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("generate", END)
    if mode == "linear":
        graph.add_edge("retrieve", "generate")
        return graph.compile()

    graph.add_node("grade_docs", grade_docs)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("fallback", fallback)
    graph.add_edge("retrieve", "grade_docs")
    graph.add_conditional_edges("grade_docs", lambda state: route_after_grade(state, max_attempts),
                                ["generate", "rewrite_query", "fallback"])
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("fallback", END)
    return graph.compile()
