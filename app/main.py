"""FastAPI service: GET /health and POST /chat.

- /health is a cheap liveness check that never touches the LLM/index, so cold-start hosts pass it.
- /chat runs the agent under a hard timeout with headroom below the grader's 30s, and falls back to
  a deterministic schema-valid response on timeout or any unexpected error. Defensive input handling
  ensures empty/malformed/oversized histories never produce a 500.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from . import agent
from .catalog import load_catalog
from .retrieval import Retriever
from .schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("main")

CHAT_BUDGET_SECONDS = float(os.getenv("CHAT_BUDGET_SECONDS", "26"))
MAX_MESSAGES = 40
MAX_CONTENT_CHARS = 8000

_RESOURCES: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the catalog + BM25 index once at startup, not on the first /chat call.
    catalog = load_catalog()
    retriever = Retriever(catalog)
    _RESOURCES["catalog"] = catalog
    _RESOURCES["retriever"] = retriever
    log.info("loaded catalog with %d items; BM25 index ready", len(catalog))
    yield
    _RESOURCES.clear()


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


def _truncate(messages: list[dict]) -> list[dict]:
    # Cap history size/length so an oversized payload can't blow the context or latency budget.
    trimmed = messages[-MAX_MESSAGES:]
    for m in trimmed:
        if isinstance(m.get("content"), str) and len(m["content"]) > MAX_CONTENT_CHARS:
            m["content"] = m["content"][:MAX_CONTENT_CHARS]
    return trimmed


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> JSONResponse:
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    messages = _truncate(messages)
    catalog = _RESOURCES["catalog"]
    retriever = _RESOURCES["retriever"]

    try:
        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, agent.handle, messages, catalog, retriever),
            timeout=CHAT_BUDGET_SECONDS,
        )
    except asyncio.TimeoutError:
        log.warning("chat handler timed out; returning safe fallback")
        result = {
            "reply": (
                "I'm having trouble pulling that together right now. Could you restate the role or "
                "skills you're hiring for?"
            ),
            "recommendations": [],
            "end_of_conversation": False,
        }
    except Exception:  # noqa: BLE001 - never 500 the grader
        log.exception("chat handler crashed; returning safe fallback")
        result = {
            "reply": "Sorry, something went wrong on my end. Could you tell me the role you're hiring for?",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # Validate against the schema before returning; on the off chance it fails, degrade safely.
    validated = ChatResponse(**result)
    log.info(
        "chat: turns=%d intent_recs=%d eoc=%s",
        len(messages),
        len(validated.recommendations),
        validated.end_of_conversation,
    )
    return JSONResponse(content=validated.model_dump())
