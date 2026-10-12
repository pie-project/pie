"""OpenAI-compatible HTTP surface: `/v1/chat/completions` and `/v1/models`.

Any harness that can point at a custom OpenAI base URL (OpenCode, Aider, or a
GUI of our own) can use this without knowing anything about Pie.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from aiohttp import web

from .backend import Backend, BackendError
from .translate import Defaults, RequestError, to_inferlet_input

BACKEND_KEY = web.AppKey("backend", object)
MODEL_KEY = web.AppKey("model_name", str)
DEFAULTS_KEY = web.AppKey("defaults", Defaults)


def _error(status: int, message: str, kind: str) -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": kind}}, status=status
    )


def _chunk(completion_id: str, model: str, delta: dict[str, Any], finish: str | None) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


async def chat_completions(request: web.Request) -> web.StreamResponse:
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _error(400, "request body is not valid JSON", "invalid_request_error")

    model = str(body.get("model") or request.app[MODEL_KEY])
    try:
        inferlet_input = to_inferlet_input(body, request.app[DEFAULTS_KEY])
    except RequestError as exc:
        return _error(400, str(exc), "invalid_request_error")

    backend: Backend = request.app[BACKEND_KEY]
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    if not body.get("stream"):
        parts: list[str] = []
        try:
            async for delta in backend.generate(inferlet_input):
                parts.append(delta)
        except BackendError as exc:
            return _error(502, f"engine error: {exc}", "server_error")
        return web.json_response(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "".join(parts)},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    response = web.StreamResponse(
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
    )
    await response.prepare(request)
    await response.write(
        _chunk(completion_id, model, {"role": "assistant", "content": ""}, None).encode()
    )
    try:
        async for delta in backend.generate(inferlet_input):
            await response.write(_chunk(completion_id, model, {"content": delta}, None).encode())
        await response.write(_chunk(completion_id, model, {}, "stop").encode())
    except BackendError as exc:
        # Headers are already sent, so the error travels in the stream.
        error = {"error": {"message": f"engine error: {exc}", "type": "server_error"}}
        await response.write(f"data: {json.dumps(error)}\n\n".encode())
    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()
    return response


async def list_models(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "object": "list",
            "data": [
                {
                    "id": request.app[MODEL_KEY],
                    "object": "model",
                    "created": 0,
                    "owned_by": "pie",
                }
            ],
        }
    )


async def health(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


def create_app(
    backend: Backend,
    model_name: str,
    defaults: Defaults = Defaults(),
) -> web.Application:
    app = web.Application()
    app[BACKEND_KEY] = backend
    app[MODEL_KEY] = model_name
    app[DEFAULTS_KEY] = defaults
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_get("/v1/models", list_models)
    app.router.add_get("/healthz", health)
    return app
