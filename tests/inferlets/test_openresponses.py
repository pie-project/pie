"""E2E test for openresponses inferlet (HTTP daemon).

Unlike other inferlet tests, this one launches a daemon and validates
the OpenResponses HTTP API via httpx requests.

Usage::

    uv run python tests/inferlets/test_openresponses.py --dummy
"""
from __future__ import annotations

import asyncio
import json
import socket
import time
import tomllib
from pathlib import Path

import httpx

from conftest import INFERLETS_DIR, run_tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 15) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def _parse_sse_events(text: str) -> list[dict | str]:
    events: list[dict | str] = []
    current_data = None
    for line in text.split("\n"):
        if line.startswith("event: "):
            pass  # we track data, not event names
        elif line.startswith("data: "):
            current_data = line[6:].strip()
        elif line == "" and current_data is not None:
            if current_data == "[DONE]":
                events.append("[DONE]")
            else:
                try:
                    events.append(json.loads(current_data))
                except json.JSONDecodeError:
                    events.append(current_data)
            current_data = None
    return events


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


async def test_openresponses(client, args):
    """Install, launch daemon, and run full HTTP test suite."""

    name = "openresponses"
    wasm_name = name.replace("-", "_")
    # Accept either debug or release builds
    wasm_release = INFERLETS_DIR / name / "target" / "wasm32-wasip2" / "release" / f"{wasm_name}.wasm"
    wasm_debug = INFERLETS_DIR / name / "target" / "wasm32-wasip2" / "debug" / f"{wasm_name}.wasm"
    wasm_path = wasm_release if wasm_release.exists() else wasm_debug
    manifest_path = INFERLETS_DIR / name / "Pie.toml"

    if not wasm_path.exists():
        raise FileNotFoundError(f"No WASM binary for {name}")

    # Install
    manifest = tomllib.loads(manifest_path.read_text())
    inferlet_id = f"{manifest['package']['name']}@{manifest['package']['version']}"
    await client.install_program(wasm_path, manifest_path)

    # Launch daemon on a free port
    port = _find_free_port()
    base = f"http://127.0.0.1:{port}"
    await client.launch_daemon(inferlet_id, port)

    if not _wait_for_port(port):
        raise RuntimeError(f"Daemon did not start on port {port}")

    # Run all HTTP tests
    errors: list[str] = []

    async with httpx.AsyncClient(timeout=30) as http:
        # --- Non-streaming response structure ---
        resp = await http.post(f"{base}/responses", json={
            "model": "auto",
            "input": [{"type": "message", "role": "user", "content": "Hi"}],
            "stream": False,
            "max_output_tokens": 8,
        })
        assert resp.status_code == 200, f"response structure: status {resp.status_code}"
        body = resp.json()
        assert body.get("type") == "response", f"type != response"
        assert body.get("status") in ("completed", "incomplete")
        assert isinstance(body.get("output"), list) and len(body["output"]) > 0
        item = body["output"][0]
        assert item.get("type") == "message"
        assert item.get("role") == "assistant"
        assert isinstance(item.get("content"), list) and len(item["content"]) > 0
        assert item["content"][0].get("type") == "output_text"
        assert len(item["content"][0].get("text", "")) > 0

        # --- Content-Type header ---
        resp = await http.post(f"{base}/responses", json={
            "model": "auto",
            "input": [{"type": "message", "role": "user", "content": "Hello"}],
            "stream": False, "max_output_tokens": 4,
        })
        assert "application/json" in resp.headers.get("content-type", "")

        # --- System instructions ---
        resp = await http.post(f"{base}/responses", json={
            "model": "auto",
            "input": [{"type": "message", "role": "user", "content": "What is 2+2?"}],
            "instructions": "You are a helpful math tutor.",
            "stream": False, "max_output_tokens": 16,
        })
        assert resp.status_code == 200

        # --- Multi-turn input ---
        resp = await http.post(f"{base}/responses", json={
            "model": "auto",
            "input": [
                {"type": "message", "role": "user", "content": "My name is Alice."},
                {"type": "message", "role": "user", "content": "What is my name?"},
            ],
            "stream": False, "max_output_tokens": 16,
        })
        assert resp.status_code == 200
        assert len(resp.json().get("output", [])) > 0

        # --- Content as string ---
        resp = await http.post(f"{base}/responses", json={
            "model": "auto",
            "input": [{"type": "message", "role": "user", "content": "Hello string"}],
            "stream": False, "max_output_tokens": 4,
        })
        assert resp.status_code == 200

        # --- Content as array ---
        resp = await http.post(f"{base}/responses", json={
            "model": "auto",
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "Hello "},
                {"type": "input_text", "text": "array"},
            ]}],
            "stream": False, "max_output_tokens": 4,
        })
        assert resp.status_code == 200

        # --- max_output_tokens ---
        resp = await http.post(f"{base}/responses", json={
            "model": "auto",
            "input": [{"type": "message", "role": "user", "content": "Write a long essay about cats"}],
            "stream": False, "max_output_tokens": 1,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("status") == "incomplete"

        # --- Developer role ---
        resp = await http.post(f"{base}/responses", json={
            "model": "auto",
            "input": [
                {"type": "message", "role": "developer", "content": "You are a pirate."},
                {"type": "message", "role": "user", "content": "Say hello"},
            ],
            "stream": False, "max_output_tokens": 16,
        })
        assert resp.status_code == 200

        # --- Function call output ---
        resp = await http.post(f"{base}/responses", json={
            "model": "auto",
            "input": [
                {"type": "message", "role": "user", "content": "What is the weather?"},
                {"type": "function_call", "call_id": "call_123", "name": "get_weather", "arguments": '{"city": "SF"}'},
                {"type": "function_call_output", "call_id": "call_123", "output": "72°F and sunny"},
                {"type": "message", "role": "user", "content": "So what's the weather?"},
            ],
            "stream": False, "max_output_tokens": 16,
        })
        assert resp.status_code == 200

        # --- Streaming SSE ---
        async with http.stream("POST", f"{base}/responses", json={
            "model": "auto",
            "input": [{"type": "message", "role": "user", "content": "Count to 3"}],
            "stream": True, "max_output_tokens": 16,
        }) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            raw = await resp.aread()
            text = raw.decode("utf-8")

        events = _parse_sse_events(text)
        assert len(events) > 0, "No SSE events"
        json_events = [e for e in events if isinstance(e, dict)]
        assert events[-1] == "[DONE]", "Missing [DONE]"

        # Sequence numbers monotonically increase
        seq_nums = [e.get("sequence_number", -1) for e in json_events]
        assert all(seq_nums[i] < seq_nums[i + 1] for i in range(len(seq_nums) - 1))

        # Event type order
        types = [e.get("type", "") for e in json_events]
        expected_prefix = ["response.created", "response.in_progress",
                           "response.output_item.added", "response.content_part.added"]
        assert types[:4] == expected_prefix, f"Wrong prefix: {types[:4]}"
        expected_suffix = ["response.output_text.done", "response.content_part.done",
                           "response.output_item.done", "response.completed"]
        assert types[-4:] == expected_suffix, f"Wrong suffix: {types[-4:]}"

        # Deltas exist
        deltas = [e for e in json_events if e.get("type") == "response.output_text.delta"]
        assert len(deltas) > 0

        # Delta concat == final text
        concat = "".join(e.get("delta", "") for e in deltas)
        done_events = [e for e in json_events if e.get("type") == "response.output_text.done"]
        if done_events:
            assert concat == done_events[0].get("text", "")

        # --- Error handling ---
        resp = await http.post(f"{base}/responses",
                               content=b"not valid json{{{",
                               headers={"Content-Type": "application/json"})
        assert resp.status_code == 400
        assert resp.json().get("error", {}).get("type") == "invalid_request"

        resp = await http.post(f"{base}/responses", json={"model": "auto", "input": []})
        assert resp.status_code == 400

        resp = await http.post(f"{base}/responses", json={"model": "auto"})
        assert resp.status_code == 400

        # --- 404 and wrong method ---
        resp = await http.get(f"{base}/unknown/path")
        assert resp.status_code == 404

        resp = await http.get(f"{base}/responses")
        assert resp.status_code == 404

        # --- CORS ---
        resp = await http.options(f"{base}/responses")
        assert resp.status_code == 200
        headers_lower = {k.lower() for k in resp.headers.keys()}
        assert "access-control-allow-origin" in headers_lower
        assert "access-control-allow-methods" in headers_lower

        # --- Server info ---
        resp = await http.get(f"{base}/")
        assert resp.status_code == 200
        body = resp.json()
        assert "name" in body and "version" in body and "endpoints" in body


if __name__ == "__main__":
    run_tests([test_openresponses], description="OpenResponses E2E Test")
