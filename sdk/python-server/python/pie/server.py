"""`pie.server.Server` — async context manager around the embedded engine.

Mirrors the legacy `pie-server` Python wheel's `Server` class so existing
test fixtures (`tests/inferlets/conftest.py::_run`, `benches/*`,
`sdk/demo/zo-training/*`) keep working unchanged.

    async with Server(cfg) as server:
        client = await server.connect()
        ...

Lifecycle:
  * `__aenter__`: optionally auto-pick a free port (`ServerConfig.port == 0`),
    serialize the `Config` to TOML, hand it to the pyo3 `bootstrap`. The
    pyo3 layer blocks until drivers + WS listener are up, then returns
    a handle. We run that on a thread (`asyncio.to_thread`) so the
    asyncio loop isn't blocked.
  * `connect()`: build a `pie_client.PieClient` against the bound URL +
    auth-token-handshake using the engine's internal token. Each call
    returns a fresh client; the user is responsible for closing them.
  * `__aexit__`: closes any connect()'d clients, then shuts the engine
    down (also off-thread). The pyo3 handle's `Drop` is the safety net
    if `__aexit__` doesn't run (interpreter exit, hard crash) — combined
    with `PR_SET_PDEATHSIG` on subprocess drivers, this means "script
    ends → server is gone, no orphans".
"""

from __future__ import annotations

import asyncio
import copy
import socket
from typing import TYPE_CHECKING, Any

from pie.config import Config

if TYPE_CHECKING:
    from pie_client import PieClient


def _find_free_port() -> int:
    """Bind a fresh socket to port 0 to discover a free local port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


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
            models=[ModelConfig(
                name="default",
                hf_repo="Qwen/Qwen3-0.6B",
                driver=DriverConfig(type="dev", device=["cuda:0"]),
            )],
        )
        async with Server(cfg) as server:
            client = await server.connect()
    """

    def __init__(self, config: Config):
        # Copy so the user's object isn't mutated by the auto-port lookup.
        self._config = copy.deepcopy(config)

        # Auto-assign a free port if the user requested one (matches the
        # legacy `pie-server.Server.__init__` behavior).
        if self._config.server.port == 0:
            self._config.server.port = _find_free_port()

        self._handle: Any = None
        self._clients: list[Any] = []

    @property
    def config(self) -> Config:
        return self._config

    @property
    def url(self) -> str:
        if self._handle is None:
            return f"ws://{self._config.server.host or '127.0.0.1'}:{self._config.server.port}"
        return self._handle.url

    @property
    def token(self) -> str:
        if self._handle is None:
            raise RuntimeError("server is not started; use `async with Server(cfg) as server:`")
        return self._handle.token

    async def __aenter__(self) -> "Server":
        from pie import _engine  # the pyo3 module
        toml_str = self._config.to_toml()
        self._handle = await asyncio.to_thread(_engine.bootstrap, toml_str)
        return self

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
        return False

    async def connect(self) -> "PieClient":
        """Build a `PieClient` against the running engine."""
        if self._handle is None:
            raise RuntimeError("server is not started; use `async with Server(cfg) as server:`")
        from pie_client import PieClient
        client = PieClient(self._handle.url)
        await client.connect()
        await client.auth_by_token(self._handle.token)
        self._clients.append(client)
        return client
