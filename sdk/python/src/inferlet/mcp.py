"""
MCP client wrapping ``pie:mcp/client``.

Lets inferlets discover and call MCP servers.
"""

from __future__ import annotations

from wit_world.imports import client as _client
from wit_world.imports.pie_mcp_types import Content


def available_servers() -> list[str]:
    """Discover available MCP servers."""
    return list(_client.available_servers())


def connect(server_name: str) -> McpSession:
    """Connect to a named MCP server.

    Performs the MCP handshake and returns a session handle.
    """
    handle = _client.connect(server_name)
    return McpSession(handle)


class McpSession:
    """An active connection to an MCP server.

    Usage::

        session = mcp.connect("my-mcp-server")
        tools = session.list_tools()
        result = session.call_tool("search", '{"query": "hello"}')
    """

    __slots__ = ("_handle",)

    def __init__(self, handle: _client.Session) -> None:
        self._handle = handle

    def list_tools(self) -> str:
        """List tools exposed by this server (JSON)."""
        return self._handle.list_tools()

    def call_tool(self, name: str, args: str) -> list[Content]:
        """Call a tool by name with JSON arguments.

        Returns a list of Content items (Text, Image, or EmbeddedResource).
        """
        return self._handle.call_tool(name, args)

    def list_resources(self) -> str:
        """List resources exposed by this server (JSON)."""
        return self._handle.list_resources()

    def read_resource(self, uri: str) -> list[Content]:
        """Read a resource by URI.

        Returns a list of Content items.
        """
        return self._handle.read_resource(uri)

    def list_prompts(self) -> str:
        """List prompts exposed by this server (JSON)."""
        return self._handle.list_prompts()

    def get_prompt(self, name: str, args: str) -> str:
        """Render a prompt template with the given arguments (JSON)."""
        return self._handle.get_prompt(name, args)

    def __enter__(self) -> McpSession:
        return self

    def __exit__(self, *args) -> None:
        pass

    def __repr__(self) -> str:
        return f"McpSession({id(self._handle):#x})"
