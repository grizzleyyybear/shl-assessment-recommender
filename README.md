# SHL Conversational Assessment Recommender

A stateless FastAPI service that holds a multi-turn conversation with a recruiter and recommends
relevant assessments from the SHL Individual Test Solutions catalog (377 products). It clarifies
vague requests, recommends, refines on follow-ups, compares named assessments, and refuses
off-topic / prompt-injection input — all within an 8-turn, 30s-per-turn budget.

## API

### `GET /health`
Cheap liveness probe (never touches the LLM or index).
```json
{ "status": "ok" }
```

### `POST /chat`
Stateless: the **full** conversation history is sent on every call; the server keeps no per-session
state of its own.

Request:
```json
{
  "messages": [
    { "role": "user", "content": "I'm hiring senior Java backend engineers." }
  ]
}
```

Response:
```json
{
  "reply": "For senior Java backend engineers, consider ...",
  "recommendations": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```
- `recommendations` is an empty array `[]` while clarifying or refusing.
- `test_type` is the SHL test-type letter (A/B/C/D/E/K/P/S).
- Every `name`/`url`/`test_type` is looked up from the catalog, so a hallucinated link is
  structurally impossible (see "Anti-hallucination" below).

## Architecture

```
POST /chat
  -> derive_state()        # LLM (+ deterministic fallback): intent, constraints, search query
  -> route on intent       # clarify | recommend | refine | compare | off_topic | injection | closing
       recommend/refine:
         retriever.candidate_pool()   # BM25 over the catalog + flagship anchors + report families
         select (LLM, ID-only)        # picks catalog IDs from the fixed candidate list
         _augment_with_anchors()      # deterministically guarantees OPQ32r + Verify G+ (recall lever)
       compare:
         catalog.match_by_name() -> grounded compare over catalog facts only
  -> guardrails.validate_recommendations()   # drop anything not in the catalog, clamp to 10
```

Key design choices (rationale in `APPROACH.md`):
- **ID-only structured output** — the LLM only ever returns catalog IDs; the service projects
  name/url/test_type from the catalog. URLs cannot be hallucinated.
- **BM25 retrieval, no embedding model** — on this lexical catalog BM25 beat a `bge-small`
  semantic index in offline tests (0.51 vs 0.42 candidate recall@15), so we skip the ~400MB model
  and its cold-start cost. The LLM is the semantic re-ranking layer.
- **Anchors + report families** — the reference traces consistently complement role-specific tests
  with SHL flagships (OPQ32r, Verify G+, Graduate Scenarios) and their report variants. We seed the
  candidate pool with these and deterministically guarantee the two most frequent ones.
- **Stateless re-derivation** — every turn rebuilds cumulative constraints from the whole history,
  so "refine" needs no special machinery and the service survives the grader's independent calls.
- **Deterministic fallback everywhere** — any LLM failure/timeout degrades to a schema-valid,
  BM25 + anchor shortlist, so the service never 500s and always commits within budget.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env          # then put your real GROQ_API_KEY in .env
uvicorn app.main:app --reload --port 8000

curl localhost:8000/health
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring call-centre agents, strong spoken English."}]}'
```

### Rebuild the catalog (only if the raw file changes)
```bash
python data/build_catalog.py     # data/shl_product_catalog_raw.json -> app/data/catalog.json
```
The generated `app/data/catalog.json` is committed, so the service needs no build step at deploy.

## Evaluation

```bash
python -m eval.replay_harness    # Recall@10 over the 10 reference conversations
python -m eval.run_eval          # replay + behavior probes -> eval/eval_report.md
```
`EVAL_THROTTLE` (seconds between turns) spaces calls out under the Groq free-tier rate limits.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Groq API key. |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Selection model. 8B has ~5x the free-tier daily token budget of 70B; swap to `llama-3.3-70b-versatile` for higher quality if you have budget. |
| `GROQ_STATE_MODEL` | `llama-3.1-8b-instant` | Lightweight state-extraction model. |
| `GROQ_TIMEOUT` | `12` | Per-LLM-call timeout (seconds). |
| `CHAT_BUDGET_SECONDS` | `26` | Hard wall-clock cap per `/chat`, under the 30s turn limit. |

## Deploy to Render

1. Push this folder to a GitHub repo.
2. In Render, **New > Blueprint** and point at the repo (uses `render.yaml`, Docker runtime, free
   plan, health check `/health`).
3. Set the `GROQ_API_KEY` environment variable in the Render dashboard (it is `sync: false`, so it
   is never committed).
4. Deploy. The health check passes immediately; `/chat` is ready once the container is up.

## Tests
```bash
python -m pytest tests/ -q
```
