"""The only module that talks to Pie. Everything above it speaks OpenAI.

`Backend` is the seam: the server depends on the protocol below, so tests can
substitute a fake and a future transport (HTTP, gRPC) only touches this file.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from pie_client import Event, PieClient


class BackendError(RuntimeError):
    """The engine reported an error while running the inferlet. Maps to HTTP 502."""


class Backend(Protocol):
    def generate(self, inferlet_input: dict[str, Any]) -> AsyncIterator[str]:
        """Yield text deltas for one completion."""
        ...


class PieBackend:
    """Runs the chat inferlet once per request through `pie_client`.

    One WebSocket connection per request keeps failure isolation simple: a
    dropped connection affects one request, not every request behind it.
    """

    def __init__(
        self,
        server_uri: str,
        inferlet: str = "chat-completion@0.1.0",
        identity: str = "default/adapter",
    ) -> None:
        self.server_uri = server_uri
        self.inferlet = inferlet
        self.identity = identity

    async def generate(self, inferlet_input: dict[str, Any]) -> AsyncIterator[str]:
        async with PieClient(self.server_uri, identity=self.identity) as client:
            process = await client.launch_process(
                self.inferlet, inferlet_input, capture_outputs=True
            )
            finished = False
            streamed = False
            try:
                while True:
                    event, value = await process.recv()
                    if event is Event.Stdout:
                        if value:
                            streamed = True
                            yield str(value)
                    elif event is Event.Error:
                        finished = True
                        raise BackendError(str(value))
                    elif event is Event.Return:
                        finished = True
                        # The inferlet returns the full text. Only use it when
                        # no deltas arrived, so text is never emitted twice.
                        if not streamed and value:
                            yield str(value)
                        return
            finally:
                if not finished:
                    # The caller stopped reading (client disconnect or error).
                    # Stop the inferlet so it does not keep consuming the GPU.
                    await process.terminate()
