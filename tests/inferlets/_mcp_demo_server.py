"""Hermetic stdio MCP server used by the MCP E2E tests.

Speaks line-delimited JSON-RPC 2.0 on stdin/stdout. No external dependencies
beyond the standard library so it can be spawned by the Python client's MCP
bridge without pulling in the official `mcp` SDK.

Tools exposed:
- `echo(text)`  → returns `text` as a single text content item.
- `add(a, b)`   → returns `str(a + b)`.
- `fail()`      → returns a JSON-RPC error so we can exercise the error path.
"""
from __future__ import annotations

import json
import sys


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def reply_result(rid, result) -> None:
    emit({"jsonrpc": "2.0", "id": rid, "result": result})


def reply_error(rid, code: int, message: str, data=None) -> None:
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    emit({"jsonrpc": "2.0", "id": rid, "error": err})


TOOLS = [
    {
        "name": "echo",
        "description": "Returns its input verbatim.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "add",
        "description": "Returns a + b as a string.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
    {
        "name": "fail",
        "description": "Always returns a JSON-RPC error.",
        "inputSchema": {"type": "object"},
    },
    {
        "name": "tool_error",
        "description": "Returns success at the JSON-RPC layer but isError=true at the tool layer.",
        "inputSchema": {"type": "object"},
    },
]


def handle(msg: dict) -> None:
    method = msg.get("method")
    rid = msg.get("id")
    params = msg.get("params") or {}

    # Notifications (no id) — process and don't reply.
    if rid is None:
        return

    if method == "initialize":
        reply_result(rid, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "pie-mcp-demo", "version": "0.1.0"},
        })
    elif method == "tools/list":
        reply_result(rid, {"tools": TOOLS})
    elif method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "echo":
            reply_result(rid, {"content": [{"type": "text", "text": args.get("text", "")}]})
        elif name == "add":
            try:
                total = args["a"] + args["b"]
            except (KeyError, TypeError) as e:
                reply_error(rid, -32602, f"Invalid arguments to 'add': {e}")
                return
            reply_result(rid, {"content": [{"type": "text", "text": str(total)}]})
        elif name == "fail":
            reply_error(rid, -32001, "tool 'fail' always fails", {"why": "on purpose"})
        elif name == "tool_error":
            reply_result(rid, {
                "content": [{"type": "text", "text": "tool reported failure"}],
                "isError": True,
            })
        else:
            reply_error(rid, -32601, f"Unknown tool '{name}'")
    else:
        reply_error(rid, -32601, f"Method not found: {method}")


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            handle(msg)
        except Exception as e:
            rid = msg.get("id")
            if rid is not None:
                reply_error(rid, -32603, f"Internal server error: {e}")


if __name__ == "__main__":
    main()
