"""Conversation history utilities."""

from __future__ import annotations


def count_turns(messages: list[dict]) -> int:
    """Return the number of messages in the conversation."""
    return len(messages)


def get_last_user_message(messages: list[dict]) -> str:
    """Return the most recent user message, or empty string."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def is_off_topic(message: str) -> bool:
    """Heuristic check for clearly off-topic queries before LLM call."""
    lower = message.lower().strip()
    off_topic_signals = (
        "capital of",
        "weather in",
        "who is the president",
        "write me a poem",
        "ignore previous",
        "ignore your instructions",
        "system prompt",
        "reveal your prompt",
        "what is 2+2",
        "solve this math",
        "legal advice",
        "medical advice",
    )
    return any(signal in lower for signal in off_topic_signals)
