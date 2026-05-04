"""E2E test for http-server inferlet (HTTP daemon).

Unlike standard inferlet tests, this one launches a daemon and validates
the HTTP endpoints via httpx requests.

Usage::

    uv run python tests/inferlets/test_http_server.py --dummy
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


async def test_http_server(client, args):
    """Install, launch daemon, and validate HTTP endpoints."""

    name = "http-server"
    wasm_name = name.replace("-", "_")
    wasm_release = INFERLETS_DIR / name / "target" / "wasm32-wasip2" / "release" / f"{wasm_name}.wasm"
    wasm_debug = INFERLETS_DIR / name / "target" / "wasm32-wasip2" / "debug" / f"{wasm_name}.wasm"
    wasm_path = wasm_release if wasm_release.exists() else wasm_debug
    manifest_path = INFERLETS_DIR / name / "Pie.toml"

    if not wasm_path.exists():
        raise FileNotFoundError(f"No WASM binary for {name}")

    # Install
    manifest = tomllib.loads(manifest_path.read_text())
    inferlet_id = f"{manifest['package']['name']}@{manifest['package']['version']}"
    await client.install_program(wasm_path, manifest_path, force_overwrite=True)

    # Launch daemon on a free port
    port = _find_free_port()
    base = f"http://127.0.0.1:{port}"
    await client.launch_daemon(inferlet_id, port)

    if not _wait_for_port(port):
        raise RuntimeError(f"Daemon did not start on port {port}")

    async with httpx.AsyncClient(timeout=15) as http:
        # --- Home page ---
        resp = await http.get(f"{base}/")
        assert resp.status_code == 200, f"Home: status {resp.status_code}"
        assert "Hello from the Pie HTTP Server Inferlet" in resp.text

        # --- Wait endpoint ---
        resp = await http.get(f"{base}/wait")
        assert resp.status_code == 200, f"Wait: status {resp.status_code}"
        assert "Slept for" in resp.text

        # --- Echo endpoint ---
        resp = await http.post(f"{base}/echo", content=b"test payload")
        assert resp.status_code == 200, f"Echo: status {resp.status_code}"
        assert resp.text == "test payload"

        # --- Info endpoint (JSON) ---
        resp = await http.get(f"{base}/info")
        assert resp.status_code == 200, f"Info: status {resp.status_code}"
        body = resp.json()
        assert body.get("message") == "Server inferlet running successfully!"

        # --- 404 for unknown paths ---
        resp = await http.get(f"{base}/nonexistent")
        assert resp.status_code == 404


if __name__ == "__main__":
    run_tests([test_http_server], description="HTTP Server E2E Test")
