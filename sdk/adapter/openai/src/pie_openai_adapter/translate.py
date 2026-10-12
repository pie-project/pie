"""Pure translation between OpenAI chat requests and Pie inferlet inputs.

Nothing in this module touches the network or the engine, so the mapping can
be tested on its own and changed without touching the transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SUPPORTED_ROLES = frozenset({"system", "user", "assistant"})


class RequestError(ValueError):
    """The client sent something the adapter cannot serve. Maps to HTTP 400."""


@dataclass(frozen=True)
class Defaults:
    max_tokens: int = 256
    temperature: float = 0.6
    top_p: float = 0.95
    system: str = "You are a helpful, respectful and honest assistant."


def _content_text(content: Any) -> str:
    """Flatten OpenAI message content (string or list of parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text":
                raise RequestError("only text content parts are supported")
            parts.append(str(part.get("text", "")))
        return "".join(parts)
    raise RequestError("message content must be a string or a list of text parts")


def _transcript(turns: list[tuple[str, str]]) -> str:
    """Render earlier turns as plain text so the model sees the history.

    The chat-completion inferlet takes one prompt and one system string, so
    prior turns are folded into the prompt with role labels. This is lossy
    compared with real chat templating; see the README limitations.
    """
    return "\n".join(f"{role}: {text}" for role, text in turns)


def to_inferlet_input(body: dict[str, Any], defaults: Defaults = Defaults()) -> dict[str, Any]:
    """Map a `/v1/chat/completions` body to the chat-completion inferlet input."""
    if not isinstance(body, dict):
        raise RequestError("request body must be a JSON object")

    if body.get("tools"):
        raise RequestError("tool calling is not supported by this adapter yet")
    n = body.get("n")
    if n is not None and n != 1:
        raise RequestError("only n=1 is supported")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty list")

    system_parts: list[str] = []
    turns: list[tuple[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise RequestError("each message must be an object")
        role = message.get("role")
        if role not in SUPPORTED_ROLES:
            raise RequestError(f"unsupported message role: {role!r}")
        text = _content_text(message.get("content"))
        if role == "system":
            system_parts.append(text)
        else:
            turns.append((role, text))

    if not turns or turns[-1][0] != "user":
        raise RequestError("the last message must be from the user")

    *history, (_, prompt) = turns
    if history:
        prompt = _transcript(history) + "\nuser: " + prompt

    # Explicit None checks, not `or`: a client sending 0 must get a 400, not a
    # silent fallback to the default.
    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")
    if max_tokens is None:
        max_tokens = defaults.max_tokens
    temperature = body.get("temperature", defaults.temperature)
    top_p = body.get("top_p", defaults.top_p)
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise RequestError("max_tokens must be a positive integer")
    if not isinstance(temperature, (int, float)) or temperature < 0:
        raise RequestError("temperature must be a non-negative number")
    if not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
        raise RequestError("top_p must be in (0, 1]")

    system = "\n".join(system_parts) if system_parts else defaults.system
    return {
        "prompt": prompt,
        "system": system,
        "max_tokens": max_tokens,
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
