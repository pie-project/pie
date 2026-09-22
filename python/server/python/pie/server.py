"""`pie.server.Server`: what `pie serve` boots, as an async context manager.

    async with Server(cfg) as server:
        client = await server.connect()
        ...

`__aenter__` renders the `Config` to the TOML `pie serve --config` reads and
hands it to `pie._engine.bootstrap`, off-thread since the boot blocks. With
`ServerConfig.port == 0` the OS picks the port and `url` says which.
`__aexit__` closes the clients from `connect()` and shuts the engine down;
the handle's own `Drop` does the same if `__aexit__` never runs.
"""

from __future__ import annotations

import asyncio
import copy
from typing import TYPE_CHECKING, Any

from pie.config import Config

if TYPE_CHECKING:
    from pie_client import PieClient


class Server:
    def __init__(self, config: Config):
        # A copy, so the port the OS picks is never written back to the caller's object.
        self._config = copy.deepcopy(config)
        self._handle: Any = None
        self._clients: list[Any] = []

    @property
    def config(self) -> Config:
        return self._config

    @property
    def url(self) -> str:
        """`ws://host:port` the engine listens on."""
        if self._handle is None:
            return f"ws://{self._config.server.host or '127.0.0.1'}:{self._config.server.port}"
        return self._handle.url

    @property
    def http_url(self) -> str:
        """`http://host:port`, where the OpenAI / Anthropic / Gemini routes live."""
        url = self.url
        return "http" + url[2:] if url.startswith("ws") else url

    @property
    def running(self) -> bool:
        """True until `shutdown()` returns."""
        return self._handle is not None and self._handle.is_running()

    async def __aenter__(self) -> "Server":
        try:
            from pie import _engine
        except ImportError as error:
            raise ImportError(
                f"pie.server: the embedded engine (pie._engine) did not load: {error}; "
                "build the wheel with `maturin develop` in python/server"
            ) from error
        self._handle = await asyncio.to_thread(_engine.bootstrap, self._config.to_toml())
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.shutdown()
        return False

    async def connect(self) -> "PieClient":
        """A `PieClient` connected to this engine; closed by `shutdown()`."""
        if not self.running:
            raise RuntimeError("server is not started; use `async with Server(cfg) as server:`")
        from pie_client import PieClient

        client = PieClient(self.url)
        await client.connect()
        self._clients.append(client)
        return client

    async def shutdown(self) -> None:
        """Close the clients from `connect()`, then stop the engine. Idempotent."""
        for client in self._clients:
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()
        if self._handle is not None:
            handle, self._handle = self._handle, None
            await asyncio.to_thread(handle.shutdown)
