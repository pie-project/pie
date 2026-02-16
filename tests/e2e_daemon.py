#!/usr/bin/env python3
"""
Comprehensive E2E tests for the openresponses-server daemon.

Tests cover the OpenResponses specification:
  - Non-streaming response structure validation
  - Streaming SSE event order and consistency
  - Input variations (string/array content, multi-turn, instructions, developer role)
  - Error handling (invalid JSON, empty input, missing fields, 404)
  - HTTP protocol (Content-Type headers, CORS)

Lifecycle:
  1. Start `pie serve --dummy --port 9999 --no-auth`
  2. Connect via PieClient, authenticate, install WASM, launch daemon
  3. Run all test cases against the daemon on port 9998
  4. Tear down
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
WASM_PATH = (
    REPO
    / "std"
    / "openresponses"
    / "target"
    / "wasm32-wasip2"
    / "debug"
    / "openresponses.wasm"
)
MANIFEST_PATH = REPO / "std" / "openresponses" / "Pie.toml"

PIE_PORT = 9999
DAEMON_PORT = 9998
BASE_URL = f"http://127.0.0.1:{DAEMON_PORT}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
passed = 0
failed = 0
errors: list[str] = []


def log(msg: str) -> None:
    print(f"[e2e] {msg}", flush=True)


def wait_for_port(port: int, host: str = "127.0.0.1", timeout: float = 60) -> bool:
    import socket

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        log(f"  ✓ {name}")
    else:
        failed += 1
        msg = f"  ✗ {name}" + (f" — {detail}" if detail else "")
        log(msg)
        errors.append(msg)


def parse_sse_events(text: str) -> list[dict | str]:
    """Parse SSE text into a list of parsed JSON events or raw strings (for [DONE])."""
    events = []
    current_event = None
    current_data = None

    for line in text.split("\n"):
        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: "):
            current_data = line[6:].strip()
        elif line == "" and current_data is not None:
            if current_data == "[DONE]":
                events.append("[DONE]")
            else:
                try:
                    parsed = json.loads(current_data)
                    events.append(parsed)
                except json.JSONDecodeError:
                    events.append(current_data)
            current_event = None
            current_data = None

    return events


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


async def test_response_structure(http: httpx.AsyncClient):
    """Test 1: Non-streaming response has all required fields per spec."""
    log("Test: Response structure validation")
    resp = await http.post(f"{BASE_URL}/responses", json={
        "model": "auto",
        "input": [{"type": "message", "role": "user", "content": "Hi"}],
        "stream": False,
        "max_output_tokens": 8,
    })
    check("status 200", resp.status_code == 200, f"got {resp.status_code}")
    body = resp.json()

    # Required top-level fields
    check("has 'id'", "id" in body)
    check("has 'type'='response'", body.get("type") == "response", f"got {body.get('type')}")
    check("has 'status'", body.get("status") in ("completed", "incomplete"), f"got {body.get('status')}")
    check("has 'model' field", "model" in body and isinstance(body["model"], str))
    check("has 'output' array", isinstance(body.get("output"), list))

    # Output item structure
    if body.get("output"):
        item = body["output"][0]
        check("item type='message'", item.get("type") == "message")
        check("item has 'id'", "id" in item)
        check("item role='assistant'", item.get("role") == "assistant")
        check("item status", item.get("status") in ("completed", "incomplete"))
        check("item has 'content' array", isinstance(item.get("content"), list))

        if item.get("content"):
            part = item["content"][0]
            check("content type='output_text'", part.get("type") == "output_text")
            check("content has 'text'", "text" in part)
            check("text is non-empty", len(part.get("text", "")) > 0)


async def test_content_type_header(http: httpx.AsyncClient):
    """Test 18: Non-streaming returns application/json Content-Type."""
    log("Test: Content-Type headers")
    resp = await http.post(f"{BASE_URL}/responses", json={
        "model": "auto",
        "input": [{"type": "message", "role": "user", "content": "Hello"}],
        "stream": False,
        "max_output_tokens": 4,
    })
    ct = resp.headers.get("content-type", "")
    check("non-streaming Content-Type is application/json", "application/json" in ct, f"got '{ct}'")


async def test_system_instructions(http: httpx.AsyncClient):
    """Test 2: System instructions via the 'instructions' field."""
    log("Test: System instructions field")
    resp = await http.post(f"{BASE_URL}/responses", json={
        "model": "auto",
        "input": [{"type": "message", "role": "user", "content": "What is 2+2?"}],
        "instructions": "You are a helpful math tutor.",
        "stream": False,
        "max_output_tokens": 16,
    })
    check("instructions: status 200", resp.status_code == 200)
    body = resp.json()
    check("instructions: status ok", body.get("status") in ("completed", "incomplete"))


async def test_multi_turn_input(http: httpx.AsyncClient):
    """Test 3: Multiple user messages in input."""
    log("Test: Multi-turn input")
    resp = await http.post(f"{BASE_URL}/responses", json={
        "model": "auto",
        "input": [
            {"type": "message", "role": "user", "content": "My name is Alice."},
            {"type": "message", "role": "user", "content": "What is my name?"},
        ],
        "stream": False,
        "max_output_tokens": 16,
    })
    check("multi-turn: status 200", resp.status_code == 200)
    body = resp.json()
    check("multi-turn: has output", len(body.get("output", [])) > 0)


async def test_content_as_string(http: httpx.AsyncClient):
    """Test 4a: Content as plain string."""
    log("Test: Content as string")
    resp = await http.post(f"{BASE_URL}/responses", json={
        "model": "auto",
        "input": [{"type": "message", "role": "user", "content": "Hello string"}],
        "stream": False,
        "max_output_tokens": 4,
    })
    check("string content: status 200", resp.status_code == 200)


async def test_content_as_array(http: httpx.AsyncClient):
    """Test 4b: Content as array of input_text parts."""
    log("Test: Content as array of parts")
    resp = await http.post(f"{BASE_URL}/responses", json={
        "model": "auto",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Hello "},
                {"type": "input_text", "text": "array"},
            ],
        }],
        "stream": False,
        "max_output_tokens": 4,
    })
    check("array content: status 200", resp.status_code == 200)


async def test_max_output_tokens(http: httpx.AsyncClient):
    """Test 5: max_output_tokens=1 should produce very short output and 'incomplete' status."""
    log("Test: max_output_tokens parameter + incomplete status")
    resp = await http.post(f"{BASE_URL}/responses", json={
        "model": "auto",
        "input": [{"type": "message", "role": "user", "content": "Write a long essay about cats"}],
        "stream": False,
        "max_output_tokens": 1,
    })
    check("max_tokens=1: status 200", resp.status_code == 200)
    body = resp.json()
    # Per spec: when max_output_tokens is exhausted, status should be 'incomplete'
    check("max_tokens=1: response status='incomplete'",
          body.get("status") == "incomplete", f"got '{body.get('status')}'")
    if body.get("output"):
        item = body["output"][0]
        check("max_tokens=1: item status='incomplete'",
              item.get("status") == "incomplete", f"got '{item.get('status')}'")
        text = item.get("content", [{}])[0].get("text", "")
        check("max_tokens=1: output is short", len(text) < 100, f"len={len(text)}")


async def test_developer_role(http: httpx.AsyncClient):
    """Test 6: Developer role treated as system message."""
    log("Test: Developer role")
    resp = await http.post(f"{BASE_URL}/responses", json={
        "model": "auto",
        "input": [
            {"type": "message", "role": "developer", "content": "You are a pirate."},
            {"type": "message", "role": "user", "content": "Say hello"},
        ],
        "stream": False,
        "max_output_tokens": 16,
    })
    check("developer role: status 200", resp.status_code == 200)
    body = resp.json()
    check("developer role: status ok", body.get("status") in ("completed", "incomplete"))


async def test_function_call_output_input(http: httpx.AsyncClient):
    """Test 7: function_call_output items included as context."""
    log("Test: Function call output in input")
    resp = await http.post(f"{BASE_URL}/responses", json={
        "model": "auto",
        "input": [
            {"type": "message", "role": "user", "content": "What is the weather?"},
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "get_weather",
                "arguments": '{"city": "SF"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "72°F and sunny",
            },
            {"type": "message", "role": "user", "content": "So what's the weather?"},
        ],
        "stream": False,
        "max_output_tokens": 16,
    })
    check("function_call_output: status 200", resp.status_code == 200)


async def test_streaming_event_order(http: httpx.AsyncClient):
    """Test 8-12: Streaming SSE event order, types, sequence numbers, text consistency."""
    log("Test: Streaming SSE compliance")

    async with http.stream("POST", f"{BASE_URL}/responses", json={
        "model": "auto",
        "input": [{"type": "message", "role": "user", "content": "Count to 3"}],
        "stream": True,
        "max_output_tokens": 16,
    }) as resp:
        check("streaming: status 200", resp.status_code == 200)

        ct = resp.headers.get("content-type", "")
        check("streaming: Content-Type is text/event-stream", "text/event-stream" in ct, f"got '{ct}'")

        raw = await resp.aread()
        text = raw.decode("utf-8")

    events = parse_sse_events(text)
    check("streaming: has events", len(events) > 0, f"got {len(events)} events")

    if not events:
        return

    # Separate JSON events from [DONE]
    json_events = [e for e in events if isinstance(e, dict)]
    done_markers = [e for e in events if e == "[DONE]"]

    # Test 12: [DONE] terminal marker
    check("streaming: ends with [DONE]", len(done_markers) > 0)
    if events:
        check("streaming: [DONE] is last event", events[-1] == "[DONE]")

    # Test 10: sequence_number monotonically increases
    seq_nums = [e.get("sequence_number", -1) for e in json_events if isinstance(e, dict)]
    is_monotonic = all(seq_nums[i] < seq_nums[i + 1] for i in range(len(seq_nums) - 1))
    check("streaming: sequence_number monotonically increases", is_monotonic, f"got {seq_nums}")

    # Extract event types in order
    event_types = [e.get("type", "") for e in json_events]

    # Test 8: Event order validation
    expected_prefix = [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
    ]
    actual_prefix = event_types[:4]
    check("streaming: event order prefix", actual_prefix == expected_prefix,
          f"expected {expected_prefix}, got {actual_prefix}")

    # After deltas, should end with: output_text.done, content_part.done, output_item.done, response.completed
    expected_suffix = [
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    actual_suffix = event_types[-4:]
    check("streaming: event order suffix", actual_suffix == expected_suffix,
          f"expected {expected_suffix}, got {actual_suffix}")

    # Verify deltas exist between prefix and suffix
    delta_events = [e for e in json_events if e.get("type") == "response.output_text.delta"]
    check("streaming: has delta events", len(delta_events) > 0, f"got {len(delta_events)}")

    # Test 9: event type field matches the event header
    # (We check that each JSON event has a 'type' field and it looks like an SSE event name)
    for e in json_events:
        t = e.get("type", "")
        check(f"streaming: event has type field ({t})", t.startswith("response."))

    # Test 11: Final text consistency — output_text.done text == concat of deltas
    delta_texts = [e.get("delta", "") for e in delta_events]
    concatenated = "".join(delta_texts)

    done_events = [e for e in json_events if e.get("type") == "response.output_text.done"]
    if done_events:
        final_text = done_events[0].get("text", "")
        check("streaming: delta concat == done text", concatenated == final_text,
              f"concat='{concatenated[:50]}…' vs done='{final_text[:50]}…'")


async def test_invalid_json(http: httpx.AsyncClient):
    """Test 13: Invalid JSON body returns 400."""
    log("Test: Invalid JSON body")
    resp = await http.post(
        f"{BASE_URL}/responses",
        content=b"not valid json{{{",
        headers={"Content-Type": "application/json"},
    )
    check("invalid JSON: status 400", resp.status_code == 400)
    body = resp.json()
    check("invalid JSON: has error object", "error" in body)
    err = body.get("error", {})
    check("invalid JSON: error type is invalid_request", err.get("type") == "invalid_request",
          f"got '{err.get('type')}'")
    # Spec requires code and param fields (may be null)
    check("invalid JSON: error has 'code' key", "code" in err)
    check("invalid JSON: error has 'param' key", "param" in err)
    check("invalid JSON: error has 'message' key", "message" in err and len(err["message"]) > 0)


async def test_empty_input(http: httpx.AsyncClient):
    """Test 14: Empty input array returns 400 (no user message)."""
    log("Test: Empty input array")
    resp = await http.post(f"{BASE_URL}/responses", json={
        "model": "auto",
        "input": [],
        "stream": False,
    })
    check("empty input: status 400", resp.status_code == 400)


async def test_missing_input_field(http: httpx.AsyncClient):
    """Test 15: Missing required 'input' field returns 400."""
    log("Test: Missing input field")
    resp = await http.post(f"{BASE_URL}/responses", json={
        "model": "auto",
    })
    check("missing input: status 400", resp.status_code == 400)


async def test_404_unknown_endpoint(http: httpx.AsyncClient):
    """Test 16: Unknown endpoint returns 404."""
    log("Test: 404 for unknown endpoint")
    resp = await http.get(f"{BASE_URL}/unknown/path")
    check("unknown endpoint: status 404", resp.status_code == 404)
    body = resp.json()
    err_type = body.get("error", {}).get("type", "")
    check("unknown endpoint: error type is not_found", err_type == "not_found", f"got '{err_type}'")


async def test_wrong_method(http: httpx.AsyncClient):
    """Test 17: GET /responses returns 404 (only POST is valid)."""
    log("Test: Wrong HTTP method")
    resp = await http.get(f"{BASE_URL}/responses")
    check("wrong method: status 404", resp.status_code == 404)


async def test_cors_preflight(http: httpx.AsyncClient):
    """Test 19: OPTIONS request returns CORS headers."""
    log("Test: CORS preflight")
    resp = await http.options(f"{BASE_URL}/responses")
    check("CORS: status 200", resp.status_code == 200)
    check("CORS: Access-Control-Allow-Origin",
          "access-control-allow-origin" in {k.lower() for k in resp.headers.keys()})
    check("CORS: Access-Control-Allow-Methods",
          "access-control-allow-methods" in {k.lower() for k in resp.headers.keys()})


async def test_server_info(http: httpx.AsyncClient):
    """Test 20: GET / returns server info with expected structure."""
    log("Test: Server info endpoint")
    resp = await http.get(f"{BASE_URL}/")
    check("GET /: status 200", resp.status_code == 200)
    body = resp.json()
    check("GET /: has 'name'", "name" in body)
    check("GET /: has 'version'", "version" in body)
    check("GET /: has 'endpoints'", "endpoints" in body)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_response_structure,
    test_content_type_header,
    test_system_instructions,
    test_multi_turn_input,
    test_content_as_string,
    test_content_as_array,
    test_max_output_tokens,
    test_developer_role,
    test_function_call_output_input,
    test_streaming_event_order,
    test_invalid_json,
    test_empty_input,
    test_missing_input_field,
    test_404_unknown_endpoint,
    test_wrong_method,
    test_cors_preflight,
    test_server_info,
]


async def main() -> int:
    global passed, failed

    # --- Sanity checks ---
    if not WASM_PATH.exists():
        log(f"WASM not found at {WASM_PATH}")
        log("Build first:  cargo build --target wasm32-wasip2  (from std/openresponses-server)")
        return 1
    if not MANIFEST_PATH.exists():
        log(f"Manifest not found at {MANIFEST_PATH}")
        return 1

    log(f"WASM: {WASM_PATH}")
    log(f"Manifest: {MANIFEST_PATH}")

    # --- 1. Start Pie server ---
    log(f"Starting pie serve --dummy on port {PIE_PORT} …")
    server_proc = subprocess.Popen(
        [
            "uv", "run", "pie",
            "serve",
            "--dummy",
            "--port", str(PIE_PORT),
            "--no-auth",
        ],
        cwd=str(REPO / "pie"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )

    try:
        log("Waiting for Pie server …")
        if not wait_for_port(PIE_PORT, timeout=120):
            log("❌ Timed out waiting for Pie server")
            return 1
        log("✓ Pie server is up")

        # --- 2. Connect + Auth + Install + Launch ---
        from pie_client import PieClient

        client = PieClient(f"ws://127.0.0.1:{PIE_PORT}")
        await client.connect()
        log("✓ PieClient connected")

        await client.authenticate("test")
        log("✓ Authenticated")

        log("Installing openresponses-server …")
        await client.install_program(str(WASM_PATH), str(MANIFEST_PATH), force_overwrite=True)
        log("✓ Installed")

        log(f"Launching daemon on port {DAEMON_PORT} …")
        await client.launch_daemon("openresponses@0.1.0", DAEMON_PORT)
        log("✓ Daemon launched")

        if not wait_for_port(DAEMON_PORT, timeout=15):
            log("❌ Timed out waiting for daemon")
            return 1
        log("✓ Daemon port is open")

        # --- 3. Run all tests ---
        log("=" * 60)
        log(f"Running {len(ALL_TESTS)} test suites …")
        log("=" * 60)

        async with httpx.AsyncClient(timeout=30) as http:
            for test_fn in ALL_TESTS:
                try:
                    await test_fn(http)
                except Exception as exc:
                    failed += 1
                    msg = f"  ✗ {test_fn.__name__} EXCEPTION: {exc}"
                    log(msg)
                    errors.append(msg)

        # --- 4. Report ---
        log("=" * 60)
        total = passed + failed
        log(f"Results: {passed}/{total} passed, {failed} failed")
        if errors:
            log("Failures:")
            for e in errors:
                log(e)
        if failed == 0:
            log("ALL TESTS PASSED ✓")
        log("=" * 60)
        return 0 if failed == 0 else 1

    finally:
        log("Shutting down Pie server …")
        try:
            os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        server_proc.wait(timeout=10)
        log("Done.")


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
