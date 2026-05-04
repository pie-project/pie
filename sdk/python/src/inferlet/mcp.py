"""
MCP client wrapping ``pie:mcp/client``.

Lets inferlets discover and call MCP servers. All response payloads are
returned as raw JSON strings — the WIT contract stays stable as MCP
evolves; you parse the JSON yourself with whatever shape your inferlet
needs.
"""

from __future__ import annotations

from wit_world.imports import client as _client


def available_servers() -> list[str]:
    """Discover available MCP servers."""
    return list(_client.available_servers())


def connect(server_name: str) -> McpSession:
    """Open a session to a registered MCP server.

    The MCP `initialize` handshake is performed by the host at
    registration time; this is just a typed-handle constructor.
    """
    handle = _client.connect(server_name)
    return McpSession(handle)


class McpSession:
    """An active connection to an MCP server.

    All methods return the raw JSON-RPC ``result`` field as a string. Use
    ``json.loads(...)`` to inspect — particularly the ``isError`` /
    ``content`` / ``structuredContent`` fields of a ``call_tool`` response.

    Usage::

        import json
        session = mcp.connect("my-mcp-server")
        tools = json.loads(session.list_tools())["tools"]
        result = json.loads(session.call_tool("search", '{"query": "hi"}'))
        if result.get("isError"):
            ...
    """

    __slots__ = ("_handle",)

    def __init__(self, handle: _client.Session) -> None:
        self._handle = handle

    def list_tools(self) -> str:
        """Raw `tools/list` JSON-RPC result."""
        return self._handle.list_tools()

    def call_tool(self, name: str, args: str) -> str:
        """Raw `tools/call` JSON-RPC result. Includes `isError` / `content`."""
        return self._handle.call_tool(name, args)

    def list_resources(self) -> str:
        """Raw `resources/list` JSON-RPC result."""
        return self._handle.list_resources()

    def read_resource(self, uri: str) -> str:
        """Raw `resources/read` JSON-RPC result."""
        return self._handle.read_resource(uri)

    def list_prompts(self) -> str:
        """Raw `prompts/list` JSON-RPC result."""
        return self._handle.list_prompts()

    def get_prompt(self, name: str, args: str) -> str:
        """Raw `prompts/get` JSON-RPC result."""
        return self._handle.get_prompt(name, args)

    def __enter__(self) -> McpSession:
        return self

    def __exit__(self, *args) -> None:
        pass

    def __repr__(self) -> str:
        return f"McpSession({id(self._handle):#x})"
