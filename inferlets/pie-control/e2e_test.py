"""End-to-end smoke test for pie-control inferlet.

Boots `pie serve --config /tmp/pie-test-dummy.toml` as a subprocess, parses
its stdout for the bound WS URL + internal token, installs the prebuilt
pie-control.wasm via WS, launches it as a daemon on a free port, and
hits `/healthz` + `/v1/models`.

Requires:
  * pre-built pie binary at `target/release/pie`
  * pre-built `inferlets/pie-control/prebuilt/pie-control.wasm`
  * Qwen/Qwen3-0.6B in `~/.cache/huggingface/hub` (or another HF model that
    the dummy driver can resolve config for)

Usage::

    uv run python inferlets/pie-control/e2e_test.py
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from pie_client import PieClient

ROOT = Path(__file__).resolve().parent.parent.parent
PIE_BIN = ROOT / "target" / "release" / "pie"
INFERLET_DIR = ROOT / "inferlets" / "pie-control"
WASM_PATH = INFERLET_DIR / "prebuilt" / "pie-control.wasm"
MANIFEST_PATH = INFERLET_DIR / "Pie.toml"
CONFIG_TOML = """
[server]
host = "127.0.0.1"
port = 0

[auth]
enabled = false

[telemetry]
enabled = false

[runtime]
allow_fs = false
allow_network = true

[[model]]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.scheduler]
batch_policy = "adaptive"
request_timeout_secs = 60
default_endowment_pages = 4
admission_oversubscription_factor = 8.0
restore_pause_at_utilization = 0.85

[model.driver]
type = "dummy"
device = ["cpu"]

[model.driver.options]
vocab_size = 32000
arch_name = "test"
"""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _parse_handshake(proc: subprocess.Popen, timeout: float) -> tuple[str, str]:
    """Read pie stdout until we have `pie-server serving on <host>:<port>` + `internal token: <tok>`."""
    url_re = re.compile(r"pie-server serving on ([^\s]+:[0-9]+)")
    tok_re = re.compile(r"internal token: ([^\s]+)")
    url: str | None = None
    token: str | None = None
    deadline = time.monotonic() + timeout
    loop = asyncio.get_event_loop()
    while time.monotonic() < deadline and (url is None or token is None):
        if proc.poll() is not None:
            raise RuntimeError(f"pie exited early (code={proc.returncode})")
        line = await loop.run_in_executor(None, proc.stdout.readline)
        if not line:
            await asyncio.sleep(0.05)
            continue
        sys.stdout.write(f"[pie] {line}")
        sys.stdout.flush()
        if url is None:
            m = url_re.search(line)
            if m:
                url = m.group(1)
        if token is None:
            m = tok_re.search(line)
            if m:
                token = m.group(1)
    if url is None or token is None:
        raise RuntimeError(f"timeout parsing pie handshake (url={url!r} token={token!r})")
    return url, token


async def _drain_stdout(proc: subprocess.Popen) -> None:
    """Keep reading pie's stdout in background so the pipe buffer doesn't fill."""
    loop = asyncio.get_event_loop()
    while proc.poll() is None:
        line = await loop.run_in_executor(None, proc.stdout.readline)
        if not line:
            await asyncio.sleep(0.05)
            continue
        sys.stdout.write(f"[pie] {line}")
        sys.stdout.flush()


def _wait_for_port(port: int, timeout: float = 15) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


async def main() -> int:
    assert PIE_BIN.exists(), f"missing pie binary at {PIE_BIN}"
    assert WASM_PATH.exists(), f"missing wasm at {WASM_PATH}"
    assert MANIFEST_PATH.exists(), f"missing manifest at {MANIFEST_PATH}"

    with tempfile.TemporaryDirectory(prefix="pie-control-e2e-") as tmp:
        tmp = Path(tmp)
        cfg = tmp / "config.toml"
        cfg.write_text(CONFIG_TOML)
        pie_home = tmp / "home"
        pie_home.mkdir()

        env = {
            **os.environ,
            "PIE_HOME": str(pie_home),
            "PIE_SHMEM_NAME": f"/pie_e2e_{os.getpid()}",
        }
        proc = subprocess.Popen(
            [str(PIE_BIN), "serve", "--config", str(cfg), "--no-auth", "--debug"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        try:
            ws_addr, token = await _parse_handshake(proc, timeout=30)
            ws_url = f"ws://{ws_addr}"
            print(f"[harness] engine ws={ws_url} token={token[:8]}…")

            drain_task = asyncio.create_task(_drain_stdout(proc))
            try:
                client = PieClient(ws_url)
                await client.connect()
                await client.auth_by_token(token)

                await client.install_program(WASM_PATH, MANIFEST_PATH, force_overwrite=True)
                print("[harness] installed pie-control@0.1.0")

                http_port = _free_port()
                base = f"http://127.0.0.1:{http_port}"
                await client.launch_daemon("pie-control@0.1.0", http_port)
                print(f"[harness] launched daemon on {base}")

                if not _wait_for_port(http_port, timeout=15):
                    raise RuntimeError(f"daemon never bound port {http_port}")

                failures: list[str] = []
                async with httpx.AsyncClient(timeout=15) as http:
                    # /healthz
                    r = await http.get(f"{base}/healthz")
                    print(f"[harness] GET /healthz -> {r.status_code} {r.text!r}")
                    if r.status_code != 200:
                        failures.append(f"/healthz status {r.status_code}")
                    try:
                        body = r.json()
                    except Exception as e:
                        failures.append(f"/healthz not json: {e}")
                        body = None
                    if body != {"status": "ok"}:
                        failures.append(f"/healthz body {body!r}")

                    # /v1/models
                    r = await http.get(f"{base}/v1/models")
                    print(f"[harness] GET /v1/models -> {r.status_code} {r.text!r}")
                    if r.status_code != 200:
                        failures.append(f"/v1/models status {r.status_code}")
                    try:
                        body = r.json()
                    except Exception as e:
                        failures.append(f"/v1/models not json: {e}")
                        body = None
                    if not body or body.get("object") != "list":
                        failures.append(f"/v1/models object {body!r}")
                    data = (body or {}).get("data") or []
                    if not data:
                        failures.append(f"/v1/models data empty: {body!r}")
                    else:
                        first = data[0]
                        if first.get("object") != "model":
                            failures.append(f"/v1/models[0].object {first!r}")
                        if first.get("owned_by") != "pie":
                            failures.append(f"/v1/models[0].owned_by {first!r}")
                        if not isinstance(first.get("id"), str) or not first.get("id"):
                            failures.append(f"/v1/models[0].id {first!r}")

                    # 404
                    r = await http.get(f"{base}/nonexistent")
                    print(f"[harness] GET /nonexistent -> {r.status_code}")
                    if r.status_code != 404:
                        failures.append(f"/nonexistent status {r.status_code}")

                await client.close()

                if failures:
                    print("[harness] FAILURES:")
                    for f in failures:
                        print(f"  - {f}")
                    return 1
                print("[harness] PASS")
                return 0
            finally:
                drain_task.cancel()
                try:
                    await drain_task
                except (asyncio.CancelledError, Exception):
                    pass
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
