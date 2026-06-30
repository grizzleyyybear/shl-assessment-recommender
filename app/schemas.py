"""API request/response models."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Message(BaseModel):
    # role stays a plain str so a malformed history never 422s; the agent sanitizes it.
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
