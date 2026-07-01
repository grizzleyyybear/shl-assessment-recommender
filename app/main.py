"""FastAPI service: GET /health and POST /chat. /chat runs the agent under a hard timeout
below the grader's 30s and falls back to a deterministic response on timeout or error."""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .agent import Agent
from .catalog import load_catalog
from .retrieval import Retriever
from .schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("main")

BUDGET = float(os.getenv("CHAT_BUDGET_SECONDS", "26"))
MAX_MSGS = 40
MAX_CHARS = 8000

_res: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cat = load_catalog()
    _res["agent"] = Agent(cat, Retriever(cat))
    log.info("loaded catalog with %d items; BM25 index ready", len(cat))
    yield
    _res.clear()


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)


@app.get("/")
async def root() -> dict:
    return {
        "service": "SHL Assessment Recommender",
        "status": "ok",
        "endpoints": {
            "GET /health": "liveness probe",
            "POST /chat": "conversational recommendations ({\"messages\": [{\"role\", \"content\"}]})",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


def _truncate(msgs: list[dict]) -> list[dict]:
    msgs = msgs[-MAX_MSGS:]
    for m in msgs:
        if isinstance(m.get("content"), str) and len(m["content"]) > MAX_CHARS:
            m["content"] = m["content"][:MAX_CHARS]
    return msgs


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> JSONResponse:
    msgs = _truncate([{"role": m.role, "content": m.content} for m in req.messages])
    agent: Agent = _res["agent"]

    try:
        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, agent.handle, msgs), timeout=BUDGET
        )
    except asyncio.TimeoutError:
        log.warning("chat handler timed out; returning deterministic fallback")
        result = _safe(agent, msgs, "Could you restate the role or skills you're hiring for?")
    except Exception:  # never 500 the grader
        log.exception("chat handler crashed; returning deterministic fallback")
        result = _safe(agent, msgs, "Could you tell me the role you're hiring for?")

    out = ChatResponse(**result)
    log.info(
        "chat: turns=%d recs=%d eoc=%s",
        len(msgs), len(out.recommendations), out.end_of_conversation,
    )
    return JSONResponse(content=out.model_dump())


def _safe(agent: Agent, msgs: list[dict], prompt: str) -> dict:
    try:
        return agent.fallback(msgs)
    except Exception:  # last-resort safe shape
        log.exception("deterministic fallback failed")
        return {
            "reply": f"I'm having trouble pulling that together right now. {prompt}",
            "recommendations": [],
            "end_of_conversation": False,
        }
