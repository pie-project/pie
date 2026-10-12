import asyncio
import json

from aiohttp.test_utils import TestClient, TestServer

from pie_openai_adapter.backend import BackendError
from pie_openai_adapter.server import create_app


class FakeBackend:
    def __init__(self, deltas=("Hel", "lo"), error=None):
        self.deltas = deltas
        self.error = error
        self.seen = None

    async def generate(self, inferlet_input):
        self.seen = inferlet_input
        for d in self.deltas:
            yield d
        if self.error:
            raise BackendError(self.error)


def run(coro):
    return asyncio.run(coro)


async def _with_client(backend, fn):
    app = create_app(backend, model_name="test-model")
    async with TestClient(TestServer(app)) as client:
        return await fn(client)


def test_non_streaming_completion():
    backend = FakeBackend()

    async def go(client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status == 200
        return await resp.json()

    body = run(_with_client(backend, go))
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Hello"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert backend.seen["prompt"] == "hi"


def test_streaming_completion_emits_sse_and_done():
    backend = FakeBackend()

    async def go(client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"stream": True, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/event-stream")
        return (await resp.read()).decode()

    raw = run(_with_client(backend, go))
    events = [line[len("data: "):] for line in raw.split("\n\n") if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == "Hello"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_bad_request_is_400_openai_error_shape():
    async def go(client):
        resp = await client.post("/v1/chat/completions", json={"messages": []})
        return resp.status, await resp.json()

    status, body = run(_with_client(FakeBackend(), go))
    assert status == 400
    assert body["error"]["type"] == "invalid_request_error"


def test_invalid_json_is_400():
    async def go(client):
        resp = await client.post("/v1/chat/completions", data=b"{nope", headers={"Content-Type": "application/json"})
        return resp.status

    assert run(_with_client(FakeBackend(), go)) == 400


def test_engine_error_before_output_is_502():
    backend = FakeBackend(deltas=(), error="boom")

    async def go(client):
        resp = await client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
        return resp.status, await resp.json()

    status, body = run(_with_client(backend, go))
    assert status == 502
    assert "boom" in body["error"]["message"]


def test_models_endpoint_lists_configured_model():
    async def go(client):
        resp = await client.get("/v1/models")
        return await resp.json()

    body = run(_with_client(FakeBackend(), go))
    assert [m["id"] for m in body["data"]] == ["test-model"]


def test_health():
    async def go(client):
        return (await client.get("/healthz")).status

    assert run(_with_client(FakeBackend(), go)) == 200
