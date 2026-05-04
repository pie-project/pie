"""Python mirror of inferlets/mcp-example.

Connects to an MCP server registered on the client session, lists its
tools, calls one, and demonstrates the canonical pattern for handling
the raw JSON responses returned by ``pie:mcp/client``.
"""

import json

from inferlet import mcp


async def main(input: dict) -> dict:
    server_name = input.get("server", "demo")
    tool = input.get("tool", "echo")
    text = input.get("text", "hello")

    servers = mcp.available_servers()
    print(f"available servers: {servers}")

    if server_name not in servers:
        raise Exception(
            f"MCP server '{server_name}' is not registered (have: {servers})"
        )

    session = mcp.connect(server_name)

    # tools/list response is opaque JSON: `{"tools": [{"name": ..., ...}, ...]}`.
    tools_json = session.list_tools()
    print(f"tools: {tools_json}")

    # tools/call: arguments must be JSON. The response carries
    # `content[]` and `isError`; we have to inspect both.
    args_json = json.dumps({"text": text})
    raw = session.call_tool(tool, args_json)

    result = _extract_tool_result(tool, raw)
    print(f"result: {result}")

    return {
        "servers": servers,
        "tools_json": tools_json,
        "result": result,
    }


def _extract_tool_result(tool: str, raw: str) -> str:
    """Pull the user-visible string out of a ``tools/call`` response.

    Honors ``isError: true`` by raising. For success, returns the text
    of the first ``text``-typed content item.
    """
    try:
        v = json.loads(raw)
    except json.JSONDecodeError as e:
        raise Exception(f"call_tool('{tool}') returned non-JSON: {e}")

    def first_text() -> str:
        for item in v.get("content") or []:
            if item.get("type") == "text":
                return item.get("text", "")
        return "<no text content>"

    if v.get("isError") is True:
        raise Exception(f"tool '{tool}' reported failure: {first_text()}")
    return first_text()
