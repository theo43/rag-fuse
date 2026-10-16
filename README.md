# rag-fuse

Learning project: a RAG over the French Syntec collective agreement (dataset `mteb/SyntecRetrieval`), built with LangGraph as microservices on local Kubernetes (minikube), traced with Langfuse. Design and decisions: [CLAUDE.md](CLAUDE.md) (its "Status" section records what each phase built).

## Repository

| Path | What it is |
|------|------------|
| `k8s/` | All Kubernetes manifests (plain YAML) |
| `ingestion/` | One-shot Job: loads the corpus, chunks, embeds (`multilingual-e5-small`), inserts into Postgres + pgvector |
| `retriever/` | `POST /search`: query → closest chunks with a score. Only service reading the database |
| `llm/` | `POST /generate` and `POST /structured` (JSON-schema output): thin wrapper around the Claude API |
| `orchestrator/` | `POST /ask`: question → answer. LangGraph graph `retrieve → generate`, traced in Langfuse |
| `receiver/` | Public entry point (`POST /ask`): creates the trace id, sent as `X-Trace-Id` to every service |

Prerequisites on the Mac: `brew install minikube kubectl uv`.

## Cluster: on, off, alive?

```bash
minikube start      # switch on: pods, images and the database come back by themselves
minikube status     # is it running?
kubectl get pods    # all Running and READY 1/1 (the ingestion pod: Completed)
minikube stop       # switch off; the database survives
```

`minikube delete` wipes everything, database included: then redo the first-time setup.

## Ask a question

Only the receiver is exposed (NodePort). With minikube's Docker driver on a Mac the node is not reachable directly, so open a tunnel first:

```bash
minikube service receiver --url     # tab 1: prints e.g. http://127.0.0.1:56350, keep it open
```

The port changes every time: copy the printed URL. Then, in another tab:

```bash
URL=http://127.0.0.1:56350          # the URL printed above
curl -s $URL/ask -H 'content-type: application/json' \
  -d '{"question": "Y a-t-il un examen médical obligatoire au retour d’un déplacement à l’étranger ?"}' \
  | python3 -c 'import json, sys; r = json.load(sys.stdin); print(r["answer"], "\n", r["sources"], r["trace_id"])'
```

The response JSON has the `answer`, the `sources` (chunk ids given to Claude) and the `trace_id`. Open https://cloud.langfuse.com → Tracing and search for that id: one trace holds the graph steps (`ask` → `retrieve`, `generate`) and the Claude call (`claude` → `anthropic.messages.create`, with tokens and cost). The retriever logs the same id: `kubectl logs deployment/retriever | grep <trace_id>`. Interactive docs: `$URL/docs`.

Prefer a fixed port? `kubectl port-forward svc/receiver 8080:80` works too (it bypasses the NodePort). Internal services can be port-forwarded the same way for debugging (`svc/orchestrator`, `svc/retriever`, `svc/llm`).

## First-time setup (fresh cluster)

From the repository root.

```bash
# 1. Secrets, never committed. `read -s` keeps values off screen and out of shell history.
#    User/database names are examples.
read -s "PGPASS?Postgres password: "
kubectl create secret generic postgres-credentials \
  --from-literal=POSTGRES_USER=rag --from-literal=POSTGRES_PASSWORD="$PGPASS" --from-literal=POSTGRES_DB=rag
read -s "KEY?Anthropic API key: "
kubectl create secret generic anthropic-credentials --from-literal=ANTHROPIC_API_KEY="$KEY"
read -s "PK?Langfuse public key: "; read -s "SK?Langfuse secret key: "
kubectl create secret generic langfuse-credentials --from-literal=LANGFUSE_PUBLIC_KEY="$PK" \
  --from-literal=LANGFUSE_SECRET_KEY="$SK" --from-literal=LANGFUSE_BASE_URL=https://cloud.langfuse.com
unset PGPASS KEY PK SK

# 2. Images, built inside minikube
minikube image build -t ingestion:latest ingestion/
minikube image build -t retriever:latest \
  --build-opt=opt=build-arg:EMBEDDING_MODEL=intfloat/multilingual-e5-small retriever/
minikube image build -t llm:latest llm/
minikube image build -t orchestrator:latest orchestrator/
minikube image build -t receiver:latest receiver/

# 3. Database, then ingestion, then the services
kubectl apply -f k8s/embedding-config.yaml -f k8s/postgres-init-configmap.yaml \
  -f k8s/postgres-service.yaml -f k8s/postgres-statefulset.yaml
kubectl rollout status statefulset/postgres
kubectl apply -f k8s/ingestion-job.yaml
kubectl wait --for=condition=complete job/ingestion --timeout=10m
kubectl apply -f k8s/
```

## Everyday commands

```bash
# After changing a service's code: rebuild its image, restart it (retriever also needs the build-opt above)
minikube image build -t orchestrator:latest orchestrator/
kubectl rollout restart deployment/orchestrator

uv --directory orchestrator run pytest      # unit tests of a service

kubectl logs deployment/orchestrator        # logs
kubectl describe pod -l app=orchestrator    # why a pod does not start

kubectl exec -it postgres-0 -- sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"'   # SELECT count(*) FROM chunks; → 124

kubectl delete job ingestion && kubectl apply -f k8s/ingestion-job.yaml            # re-run the ingestion
```

The Claude model is set in `k8s/llm-config.yaml` (`ANTHROPIC_MODEL`): edit, `kubectl apply -f k8s/llm-config.yaml`, then `kubectl rollout restart deployment/llm`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Pod in `ErrImageNeverPull` | Image missing in minikube: run its `minikube image build` |
| `/ask` returns 502 `upstream service error` | `retriever` or `llm` is down or failing: `kubectl get pods`, then its logs |
| `llm` logs show `401 invalid x-api-key` | Wrong key: recreate `anthropic-credentials`, then restart `llm` |
| Changed a Secret or ConfigMap, nothing happens | Pods read them only at start: `kubectl rollout restart deployment/<name>` |
| `connection refused` on the receiver URL | The `minikube service` tunnel (or port-forward) is closed, or broke after a restart: start it again and use the new URL |
