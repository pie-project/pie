"""Local bridge from this Python client to MCP servers.

Mirrors the Rust client's bridge: spawns child processes for `stdio`
transport, multiplexes JSON-RPC requests over the single connection, and
performs the MCP `initialize` handshake at registration time so the
engine-side registration is only announced if the server actually came up.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any


PROTOCOL_VERSION = "2024-11-05"
CLIENT_INFO = {"name": "pie-client-python", "version": "0.2.0"}


class JsonRpcError(Exception):
    """A JSON-RPC error returned by an MCP server."""

    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"JSON-RPC error {code}: {message}")


class StdioServer:
    """A connection to a single stdio MCP server.

    Cheap to keep around; many inferlet calls multiplex over the same
    connection using distinct request ids.
    """

    def __init__(self, name: str, process: asyncio.subprocess.Process):
        self.name = name
        self.process = process
        self._next_id = 1
        self._pending: dict[int, asyncio.Future] = {}
        self._dead = False
        self._reader_task = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._stderr_loop())

    @classmethod
    async def spawn(cls, name: str, command: str, args: list[str]) -> "StdioServer":
        process = await asyncio.create_subprocess_exec(
            command, *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return cls(name, process)

    async def handshake(self) -> None:
        await self.call("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": CLIENT_INFO,
        })
        await self.notify("notifications/initialized", {})

    async def call(self, method: str, params: Any) -> Any:
        if self._dead:
            raise JsonRpcError(-32000, f"MCP server '{self.name}' is no longer running")
        request_id = self._next_id
        self._next_id += 1
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = fut
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        try:
            self.process.stdin.write(json.dumps(request).encode() + b"\n")
            await self.process.stdin.drain()
        except Exception as e:
            self._pending.pop(request_id, None)
            raise JsonRpcError(-32000, f"MCP server '{self.name}' input closed: {e}")
        return await fut

    async def notify(self, method: str, params: Any) -> None:
        request = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            self.process.stdin.write(json.dumps(request).encode() + b"\n")
            await self.process.stdin.drain()
        except Exception:
            # Notifications are best-effort.
            pass

    async def _read_loop(self) -> None:
        try:
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode())
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"[mcp:{self.name}] non-JSON line ({e}): {line!r}", flush=True)
                    continue
                # We only handle responses (id present + result|error).
                request_id = msg.get("id")
                if request_id is None or not isinstance(request_id, int):
                    continue
                fut = self._pending.pop(request_id, None)
                if fut is None or fut.done():
                    continue
                if "error" in msg:
                    err = msg["error"] or {}
                    fut.set_exception(JsonRpcError(
                        code=err.get("code", -32000),
                        message=err.get("message", "MCP error"),
                        data=err.get("data"),
                    ))
                else:
                    fut.set_result(msg.get("result"))
        finally:
            self._dead = True
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(JsonRpcError(
                        -32000,
                        f"MCP server '{self.name}' died before response",
                    ))
            self._pending.clear()

    async def _stderr_loop(self) -> None:
        try:
            while True:
                line = await self.process.stderr.readline()
                if not line:
                    break
                print(f"[mcp:{self.name}] {line.decode(errors='replace').rstrip()}", flush=True)
        except Exception:
            pass

    async def close(self) -> None:
        # Graceful shutdown ladder: close stdin (most MCP servers exit on
        # EOF) → SIGTERM → SIGKILL. Each step gives the server a brief
        # window to exit before escalating.
        if self.process.returncode is None:
            try:
                if self.process.stdin is not None and not self.process.stdin.is_closing():
                    self.process.stdin.close()
            except Exception:
                pass
            if not await self._wait_with_timeout(1.0):
                try:
                    self.process.terminate()
                except ProcessLookupError:
                    pass
                if not await self._wait_with_timeout(1.0):
                    try:
                        self.process.kill()
                    except ProcessLookupError:
                        pass
                    try:
                        await self.process.wait()
                    except Exception:
                        pass
        self._reader_task.cancel()
        self._stderr_task.cancel()

    async def _wait_with_timeout(self, timeout: float) -> bool:
        """Wait up to `timeout` seconds for the process to exit; True on exit."""
        try:
            await asyncio.wait_for(self.process.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


class BridgeRegistry:
    """Per-`PieClient` registry of locally-spawned MCP servers."""

    def __init__(self):
        self._servers: dict[str, StdioServer] = {}

    async def register_stdio(self, name: str, command: str, args: list[str]) -> None:
        if name in self._servers:
            raise Exception(f"MCP server '{name}' already registered")
        server = await StdioServer.spawn(name, command, args)
        try:
            await server.handshake()
        except Exception:
            await server.close()
            raise
        self._servers[name] = server

    def get(self, name: str) -> StdioServer | None:
        return self._servers.get(name)

    async def close_all(self) -> None:
        for server in list(self._servers.values()):
            await server.close()
        self._servers.clear()
