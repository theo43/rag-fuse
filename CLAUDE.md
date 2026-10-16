# CLAUDE.md
 
Personal learning project: a local **agentic** RAG built with LangGraph, monitored with Langfuse, split into microservices and deployed on local Kubernetes (minikube).
Data: the **SyntecRetrieval** dataset (French Syntec collective agreement, HR questions in French), stored in PostgreSQL + pgvector.
 
The goal is learning, not shipping a product: every piece of code must be readable and understandable by a human in a few minutes.
 
---
 
## Working rules (take priority)
 
- **Small increments.** One phase at a time (see "Phased plan"). Do not start the next phase without explicit approval.
- **No speculative code.** No feature, config option, abstraction layer or exotic error case that was not asked for. When in doubt, ask.
- **Limited code volume.** One service = a few hundred lines at most. If a change exceeds ~150 lines, stop and propose a split before writing. This limit excludes generated files (`uv.lock` and similar)
- **Explain before writing** for any structural choice (new dependency, new file, schema change, new k8s manifest).
- Follow the Git workflow below: one branch per phase, delivered as a pull request that the user reviews and merges.
- Secrets (Anthropic API key, Langfuse keys, Postgres password) are never written in code, manifests or the repository.
---
 
## Git workflow
 
The repository is hosted on the user's GitHub account (remote `origin`). Each phase of the "Phased plan" is a feature.
 
1. **Start of a phase**: update `main` (`git pull`), then create a branch `phase-<N>-<short-name>` from it (e.g. `phase-2-retriever`).
2. **During the phase**: one logical change per commit (e.g. schema, then endpoint, then tests) on the phase branch, using Conventional Commits (`feat:`, `fix:`, `test:`, `chore:`, `docs:`, `refactor:`). Commit without asking, but never on `main` directly.
3. **End of the phase**: run the tests and the manual check defined for the phase, then:
   - `git push -u origin phase-<N>-<short-name>`
   - open a pull request against `main` with `gh pr create`, titled `Phase <N>: <short-name>`, whose description covers what was built, how it was verified, and anything left open
   - give the user the PR link and a short summary
4. **Review**: the user reviews and merges the pull request on GitHub. **Never merge it yourself** (no local merge into `main`, no `gh pr merge`). Review feedback is addressed with new commits on the phase branch, pushed to the same PR.
5. Only once the user says the PR is merged, start the next phase (back to step 1).
Rules:
- Before every commit, check that no secret or `.env` file is staged (`git status`, `git diff --cached`).
- Never force-push, rewrite history on `main`, or delete branches without an explicit request.
- If `origin` is not configured, a push fails (authentication, rejected) or `gh` cannot create the PR, stop and tell the user instead of working around it.
---
 
## Architecture
 
```
client ─► receiver ─► orchestrator (LangGraph) ─┬─► retriever ─► PostgreSQL + pgvector
                                                 └─► llm ─► Claude API
                          │
                          └─► Langfuse (traces)
```
 
| Service        | Role |
|----------------|------|
| `receiver`     | HTTP entry point. Validates the request, generates a `trace_id`, calls the orchestrator. No business logic. |
| `orchestrator` | Holds the LangGraph graph. The only service that makes decisions. Calls `retriever` and `llm` over HTTP. |
| `retriever`    | Query embedding + vector search. **The only service allowed to access the database.** |
| `llm`          | Thin wrapper around the Claude API: API key, retries, structured output (JSON schema), generation tracing. |
| `ingestion`    | One-shot Kubernetes Job: loads SyntecRetrieval, computes embeddings, inserts into pgvector. Not a service. |
 
### Database
 
- PostgreSQL runs in **its own pod** (StatefulSet + PVC), image `pgvector/pgvector:pg16`. Never as a sidecar of another service.
- *Database per service* pattern: only `retriever` (read) and the `ingestion` Job (write) connect to it. The orchestrator always goes through the retriever API.
- Initial schema (`CREATE EXTENSION vector`, chunks table, HNSW index) in an SQL script mounted via ConfigMap into `/docker-entrypoint-initdb.d`.
- minikube reminder: the PVC survives `minikube stop`, not `minikube delete`.
### Embeddings
 
- Claude does not provide embeddings: use a local model, `intfloat/multilingual-e5-small` (dimension 384, runs on CPU).
- **The same model** in `ingestion` and `retriever`. Model name and dimension live in shared config.
- Respect the e5 prefixes: `"passage: "` at ingestion, `"query: "` at search time.
### Observability (Langfuse)
 
- Langfuse Cloud (free Hobby tier), no self-hosting for now.
- Credentials via environment variables from a Kubernetes Secret: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`.
- In the orchestrator: Langfuse `CallbackHandler` passed when invoking the graph.
- The `trace_id` generated by `receiver` is propagated in the `X-Trace-Id` HTTP header to `orchestrator`, `retriever` and `llm`, to get an end-to-end trace.
---
 
## Code architecture
 
No full hexagonal architecture. Every service follows the same light structure:
 
```
<service>/
  app/
    main.py      # FastAPI: routes + Pydantic schemas
    core.py      # business logic, no direct I/O
    clients.py   # external access (HTTP, database, embedding model)
    config.py    # pydantic-settings, reads environment variables
  tests/
  Dockerfile
  pyproject.toml
```
 
Allowed ports (abstractions), **and only these**:
- `LLMClient` (`typing.Protocol`) in the orchestrator, so the `llm` service can be mocked in tests.
- `VectorStore` (`typing.Protocol`) in the retriever, so pgvector can be mocked.
No other interface, factory or base class without an explicit request.
 
### Stack
 
- Python 3.12, dependencies managed with `uv`
- FastAPI, `httpx`, `pydantic-settings`
- `psycopg` (v3) + `pgvector`
- `sentence-transformers` (retriever, ingestion)
- `langgraph`, `langfuse` (orchestrator)
- `anthropic` SDK (llm); the Claude model name comes from the `ANTHROPIC_MODEL` environment variable, never hard-coded
- `pytest`
---
 
## The LangGraph graph
 
**Always written with LangGraph, even in the linear version**: a `TypedDict` state, every step is a node. No ad hoc pipeline chaining functions, otherwise moving to agentic behavior becomes a refactor.
 
### Step 1 — linear
 
```
START → retrieve → generate → END
```
 
State: `question`, `query`, `docs`, `answer`. The `retrieve` node always searches with `state["query"]` (initialized to the question), never directly with `question`.
 
### Step 2 — a single decision: grading + query rewriting
 
```
START → retrieve → grade_docs ─(relevant)──────────► generate → END
            ▲           ├─(not relevant)─► rewrite_query
            └───────────┘
                        └─(attempts ≥ MAX)─► fallback → END
```
 
State additions: `docs_relevant: bool`, `attempts: int`.
- `grade_docs`: asks the LLM, with structured output, whether the passages answer the question.
- `rewrite_query`: rewrites the query and increments `attempts`.
- `fallback`: answers honestly that the information was not found.
- The routing function only reads the state. `MAX_ATTEMPTS` is in config (default: 2).
Further decisions (entry routing "is retrieval needed?", checking that the answer is grounded) come later, **one at a time**, and only on request.
 
---
 
## Kubernetes
 
- Plain YAML manifests in `k8s/` (no Helm, no Kustomize for now).
- Images built directly inside minikube (`minikube image build` or `eval $(minikube docker-env)`), `imagePullPolicy: Never`.
- Secrets created by hand with `kubectl create secret`, referenced in manifests, never committed.
- Postgres: StatefulSet + Service + PVC + ConfigMap (init SQL script).
- Ingestion: `Job`.
- The four services: `Deployment` + `Service` (ClusterIP). Only `receiver` is exposed (NodePort or `minikube service`).
### Target layout
 
```
.
├── CLAUDE.md
├── receiver/
├── orchestrator/
├── retriever/
├── llm/
├── ingestion/
└── k8s/
```
 
---
 
## Phased plan
 
Each phase ends with something that runs and can be checked by hand.
 
0. **Database** — Postgres manifests + init SQL script; check with `psql` that the `vector` extension is enabled.
1. **Ingestion** — Job that loads SyntecRetrieval (Hugging Face, MTEB dataset; verify the exact identifier before coding), embeds and inserts; check the row count.
2. **Retriever** — `POST /search` endpoint (query → top-k passages with score); test on a few questions from the dataset.
3. **LLM** — generation endpoint + structured output endpoint; test with a simple prompt.
4. **Linear orchestrator** — step 1 graph + Langfuse traces.
5. **Receiver** — entry point + `trace_id` generation and propagation; end-to-end test.
6. **First decision** — grading + rewriting loop (step 2 of the graph).
7. **Evaluation** — compare linear vs agentic on the SyntecRetrieval questions: answer quality, latency, number of LLM calls.
 