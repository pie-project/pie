"""`pie.server.Server` — async context manager around the embedded engine.

Mirrors the legacy `pie-server` Python wheel's `Server` class so existing
test fixtures (`tests/inferlets/conftest.py::_run`, `benches/*`) keep
working unchanged.

    async with Server(cfg) as server:
        client = await server.connect()
        ...

Lifecycle:
  * `__aenter__`: serialize the `Config` to TOML and hand it to the pyo3
    `bootstrap`. If `ServerConfig.port == 0`, Rust asks the OS for an
    ephemeral port and returns the bound URL. The pyo3 layer blocks until
    drivers + WS listener are up, then returns a handle. We run that on a
    thread (`asyncio.to_thread`) so the asyncio loop isn't blocked.
  * `connect()`: build a `pie_client.PieClient` against the bound URL.
    Each call returns a fresh client; the user is responsible for
    closing them.
  * `__aexit__`: closes any connect()'d clients, then shuts the engine
    down (also off-thread). The pyo3 handle's `Drop` is the safety net
    if `__aexit__` doesn't run (interpreter exit, hard crash) — combined
    with `PR_SET_PDEATHSIG` on subprocess drivers, this means "script
    ends → server is gone, no orphans".
"""

from __future__ import annotations

import asyncio
import copy
import contextlib
import os
import shutil
import socket
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pie.config import Config

if TYPE_CHECKING:
    from pie_client import PieClient


def _ensure_py_runtime() -> None:
    """Best-effort install of the Python WASM runtime (componentize-py).

    R3: provisioning moved out of the Rust engine into the ``pie`` CLI; the
    canonical installer is ``bakery.py_runtime``. Failures (offline, or bakery
    not installed in this environment) are swallowed — only Python inferlets
    need the runtime, and non-Python inferlets still work without it.
    """
    try:
        from bakery.py_runtime import ensure_installed

        ensure_installed(quiet=True)
    except Exception:
        pass


def _find_pie_binary() -> str | None:
    """Locate a `pie` CLI binary for the subprocess fallback.

    Checked in order: ``$PIE_BIN``, the repo's cargo target dirs (so an
    in-tree ``cargo build`` is picked up without configuration), then ``PATH``.
    """
    if env_bin := os.environ.get("PIE_BIN"):
        if Path(env_bin).exists():
            return env_bin
    # sdk/python-server/python/pie/server.py -> repo root is 4 levels up.
    root = Path(__file__).resolve().parents[4]
    for profile in ("release", "release-min", "debug"):
        candidate = root / "target" / profile / "pie"
        if candidate.exists():
            return str(candidate)
    return shutil.which("pie")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Server:
    """Async context manager that owns a Pie runtime.

    Compatible drop-in for the legacy `pie.server.Server` from the
    deleted `pie-server` Python wheel. Same constructor + `connect()`
    + `__aenter__/__aexit__` shape.

    Usage::

        from pie.server import Server
        from pie.config import (
            Config, ServerConfig, AuthConfig, ModelConfig, DriverConfig,
        )

        cfg = Config(
            server=ServerConfig(port=0),
            auth=AuthConfig(enabled=False),
            model=ModelConfig(
                name="default",
                hf_repo="Qwen/Qwen3-0.6B",
                driver=DriverConfig(type="dev", device=["cuda:0"]),
            ),
        )
        async with Server(cfg) as server:
            client = await server.connect()
    """

    def __init__(self, config: Config):
        # Copy so the user's object isn't mutated by the auto-port lookup.
        self._config = copy.deepcopy(config)

        self._handle: Any = None
        self._clients: list[Any] = []
        # Subprocess fallback state (used when `pie._engine` is unavailable).
        self._proc: Any = None
        self._proc_url: str | None = None
        self._cfg_path: Path | None = None
        self._drain_task: Any = None
        self._log_lines: list[str] = []

    @property
    def config(self) -> Config:
        return self._config

    @property
    def url(self) -> str:
        if self._proc_url is not None:
            return self._proc_url
        if self._handle is None:
            return f"ws://{self._config.server.host or '127.0.0.1'}:{self._config.server.port}"
        return self._handle.url

    async def __aenter__(self) -> "Server":
        # Provision the Python WASM runtime if missing (R3: the Rust engine no
        # longer auto-installs it). Best-effort, off-thread.
        await asyncio.to_thread(_ensure_py_runtime)

        try:
            from pie import _engine  # the pyo3 module
        except ImportError:
            # No in-process engine for this build (e.g. the Metal driver is
            # linked into the `pie` binary and ships no maturin extension).
            # Drive the CLI instead so the same `Server` API keeps working.
            await self._start_subprocess()
            return self

        toml_str = self._config.to_engine_toml()
        self._handle = await asyncio.to_thread(_engine.bootstrap, toml_str)
        return self

    async def _start_subprocess(self) -> None:
        pie_bin = _find_pie_binary()
        if pie_bin is None:
            raise RuntimeError(
                "the in-process engine (`pie._engine`) is unavailable and no `pie` "
                "binary was found. Set $PIE_BIN, or build one, e.g.:\n"
                "  cargo build --release -p pie-bin --no-default-features "
                "--features driver-metal"
            )

        # The CLI needs a concrete port; `port = 0` is an in-process-only affordance.
        if not self._config.server.port:
            self._config.server.port = _free_port()
        host = self._config.server.host or "127.0.0.1"

        # Readiness is read off the startup banner, and `log_serving`
        # (`worker/src/engine.rs`) only prints it when `server.verbose` is set
        # -- which defaults to false, and which `pie config init` writes as
        # false. Without this the wait below never matches and every spawn ends
        # in the 300 s timeout, against a server that came up fine. The output
        # is drained and kept for error messages either way.
        self._config.server.verbose = True

        tmp_dir = Path(tempfile.mkdtemp(prefix="pie-server-"))
        self._cfg_path = tmp_dir / "config.toml"
        self._cfg_path.write_text(self._config.to_toml())

        self._proc = await asyncio.create_subprocess_exec(
            pie_bin, "--config", str(self._cfg_path), "serve",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        assert self._proc.stdout is not None
        timeout = float(os.environ.get("PIE_SERVER_STARTUP_TIMEOUT", "300"))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        ready = False
        while loop.time() < deadline:
            try:
                line = await asyncio.wait_for(
                    self._proc.stdout.readline(),
                    timeout=max(0.1, deadline - loop.time()),
                )
            except asyncio.TimeoutError:
                break
            if not line:
                raise RuntimeError(
                    "`pie serve` exited before startup completed:\n"
                    + "".join(self._log_lines[-80:])
                )
            text = line.decode("utf-8", errors="replace")
            self._log_lines.append(text)
            del self._log_lines[:-200]
            # The banner's scheme moved from `ws://` to `gateway://` when the
            # gateway edge landed; match the phrase, not the scheme. The
            # `pie standalone serving` line is the tracing-side statement of the
            # same event, kept as a second reading so a build that suppresses
            # the banner still starts rather than hanging to the timeout.
            if "Server ready at " in text or "standalone serving" in text:
                ready = True
                break

        if not ready:
            await self._stop_subprocess()
            raise TimeoutError(
                f"timed out after {timeout}s waiting for `pie serve` startup:\n"
                + "".join(self._log_lines[-80:])
            )

        self._proc_url = f"ws://{host}:{self._config.server.port}"
        self._drain_task = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        """Keep reading stdout so the child never blocks on a full pipe."""
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                return
            self._log_lines.append(line.decode("utf-8", errors="replace"))
            del self._log_lines[:-200]

    async def _stop_subprocess(self) -> None:
        if self._drain_task is not None:
            self._drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._drain_task
            self._drain_task = None

        if self._proc is not None:
            proc, self._proc = self._proc, None
            if proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=10)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()

        if self._cfg_path is not None:
            shutil.rmtree(self._cfg_path.parent, ignore_errors=True)
            self._cfg_path = None
        self._proc_url = None

    async def __aexit__(self, exc_type, exc, tb):
        # Close any clients the user opened via `connect()`.
        for client in self._clients:
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()

        # Shut down the engine (blocking; off-thread).
        if self._handle is not None:
            handle, self._handle = self._handle, None
            await asyncio.to_thread(handle.shutdown)

        await self._stop_subprocess()
        return False

    async def connect(self) -> "PieClient":
        """Build a `PieClient` against the running engine."""
        if self._handle is None and self._proc_url is None:
            raise RuntimeError("server is not started; use `async with Server(cfg) as server:`")
        from pie_client import PieClient
        client = PieClient(self.url)
        await client.connect()
        self._clients.append(client)
        return client
