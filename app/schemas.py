"""Pydantic models mirroring the non-negotiable API contract exactly.

Response shape (verbatim from the spec):
    { "reply": str,
      "recommendations": [ {"name": str, "url": str, "test_type": str}, ... ],
      "end_of_conversation": bool }
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Message(BaseModel):
    # role is kept permissive (str, not Literal) so a malformed history never 422s the endpoint;
    # we sanitize roles in the agent instead of rejecting the request.
    role: str = ""
    content: str = ""


class ChatRequest(BaseModel):
    messages: list[Message] = Field(default_factory=list)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
