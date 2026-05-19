"""E2E test for pie-control inferlet (engine control plane).

Validates /healthz + /v1/models against a live pie engine using the
`launch_daemon` WS path. Uses the dummy driver so no real model weights
are loaded — only the model NAME is needed (sourced from config).

Usage::

    uv run python tests/inferlets/test_pie_control.py --driver dummy
"""
from __future__ import annotations

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


async def test_pie_control(client, args):
    """Install pie-control, launch daemon, validate endpoints."""

    name = "pie-control"
    # Prefer locally built binary; fall back to checked-in prebuilt.
    wasm_release = INFERLETS_DIR / name / "target" / "wasm32-wasip2" / "release" / "pie_control.wasm"
    wasm_debug = INFERLETS_DIR / name / "target" / "wasm32-wasip2" / "debug" / "pie_control.wasm"
    wasm_prebuilt = INFERLETS_DIR / name / "prebuilt" / "pie-control.wasm"
    wasm_path = next(
        (p for p in (wasm_release, wasm_debug, wasm_prebuilt) if p.exists()),
        None,
    )
    if wasm_path is None:
        raise FileNotFoundError(f"No WASM binary for {name}")

    manifest_path = INFERLETS_DIR / name / "Pie.toml"
    manifest = tomllib.loads(manifest_path.read_text())
    inferlet_id = f"{manifest['package']['name']}@{manifest['package']['version']}"

    await client.install_program(wasm_path, manifest_path, force_overwrite=True)

    port = _find_free_port()
    base = f"http://127.0.0.1:{port}"
    await client.launch_daemon(inferlet_id, port)

    if not _wait_for_port(port):
        raise RuntimeError(f"Daemon did not start on port {port}")

    async with httpx.AsyncClient(timeout=15) as http:
        # --- /healthz ---
        resp = await http.get(f"{base}/healthz")
        assert resp.status_code == 200, f"healthz status {resp.status_code}"
        body = resp.json()
        assert body == {"status": "ok"}, f"healthz body {body!r}"

        # --- /v1/models ---
        resp = await http.get(f"{base}/v1/models")
        assert resp.status_code == 200, f"models status {resp.status_code}"
        body = resp.json()
        assert body.get("object") == "list", f"models object {body!r}"
        data = body.get("data") or []
        assert len(data) >= 1, f"models data empty: {body!r}"
        first = data[0]
        assert first.get("object") == "model"
        assert first.get("owned_by") == "pie"
        assert isinstance(first.get("id"), str) and first["id"], f"missing id: {first!r}"

        # --- 404 ---
        resp = await http.get(f"{base}/nonexistent")
        assert resp.status_code == 404, f"unknown path status {resp.status_code}"


if __name__ == "__main__":
    run_tests([test_pie_control], description="pie-control E2E Test")
