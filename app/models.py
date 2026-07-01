"""Pydantic request/response schemas for the SHL Assessment Recommender API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., description="Full conversation history")


class Recommendation(BaseModel):
    name: str = Field(..., description="Assessment name from catalog")
    url: str = Field(..., description="Catalog URL")
    test_type: str = Field(..., description="Assessment type code (K, P, A, S, B, C, D, E)")


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Agent's text response")
    recommendations: list[Recommendation] = Field(
        default_factory=list,
        description="Empty when clarifying; 1-10 items when recommending",
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True only when the agent considers the task complete",
    )


class HealthResponse(BaseModel):
    status: str = "ok"
