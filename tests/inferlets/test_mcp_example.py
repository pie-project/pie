"""E2E tests for the mcp-example inferlets (Rust, Python, JS variants).

Spawns a self-contained stdio MCP server (`_mcp_demo_server.py`), registers
it with the running Pie engine through the Python client's bridge, runs the
inferlet, and asserts the output round-trips through every layer.

Each variant runs the same four checks (echo, list-tools, unknown-server,
tool-error). Variants whose WASM hasn't been built will SKIP via the
runner's `FileNotFoundError` handling.
"""
import sys
from pathlib import Path

from conftest import run_inferlet, run_tests

DEMO_SERVER = Path(__file__).parent / "_mcp_demo_server.py"

# Inferlet name → suffix used in generated test function names.
VARIANTS: list[tuple[str, str]] = [
    ("mcp-example",     ""),       # Rust
    ("mcp-example-py",  "_py"),
    ("mcp-example-js",  "_js"),
]


async def _register_demo(client) -> None:
    """Register the demo MCP server under the name 'demo' if not already."""
    # Tests share a client — only the first call registers; subsequent ones
    # see "already registered" and skip.
    try:
        await client.register_mcp_server(
            name="demo",
            transport="stdio",
            command=sys.executable,
            args=[str(DEMO_SERVER)],
        )
    except Exception as e:
        if "already registered" not in str(e):
            raise


async def _check_echo(client, args, inferlet: str) -> None:
    await _register_demo(client)
    out = await run_inferlet(
        client, inferlet,
        {"server": "demo", "tool": "echo", "text": "ping"},
        timeout=args.timeout,
    )
    assert "available servers: " in out, f"Missing servers list in:\n{out}"
    assert "\"demo\"" in out, f"'demo' not listed in:\n{out}"
    assert "result: ping" in out, f"Echo result not found in:\n{out}"


async def _check_list_tools(client, args, inferlet: str) -> None:
    await _register_demo(client)
    out = await run_inferlet(
        client, inferlet,
        {"server": "demo", "tool": "echo", "text": "x"},
        timeout=args.timeout,
    )
    assert "\"echo\"" in out and "\"add\"" in out and "\"fail\"" in out, \
        f"Expected all three tools to appear in tools_json:\n{out}"


async def _check_unknown_server(client, args, inferlet: str) -> None:
    # Don't register — expect a clean error from the inferlet's
    # available_servers check (so we exercise the not-registered path
    # without spawning the bridge).
    try:
        await run_inferlet(
            client, inferlet,
            {"server": "nope-not-a-server"},
            timeout=args.timeout,
        )
    except RuntimeError as e:
        assert "not registered" in str(e), f"Unexpected error message: {e}"
        return
    raise AssertionError("expected error for unregistered MCP server")


async def _check_tool_error(client, args, inferlet: str) -> None:
    """isError=true at the MCP tool layer must surface as an inferlet error.

    Regression guard: an earlier version of the runtime ignored isError
    and returned the content array as a normal success.
    """
    await _register_demo(client)
    try:
        await run_inferlet(
            client, inferlet,
            {"server": "demo", "tool": "tool_error", "text": "n/a"},
            timeout=args.timeout,
        )
    except RuntimeError as e:
        assert "tool reported failure" in str(e) or "reported failure" in str(e), \
            f"Unexpected error message: {e}"
        return
    raise AssertionError("expected isError=true to surface as an error")


# Generate one test function per (variant, check) pair so the runner
# reports each independently and skips missing WASM binaries cleanly.
def _make_test(check, inferlet: str, name: str):
    async def test(client, args):
        await check(client, args, inferlet)
    test.__name__ = name
    test.__qualname__ = name
    return test


_TESTS = []
for inferlet, suffix in VARIANTS:
    _TESTS.extend([
        _make_test(_check_echo,            inferlet, f"test_mcp_echo{suffix}"),
        _make_test(_check_list_tools,      inferlet, f"test_mcp_list_tools{suffix}"),
        _make_test(_check_unknown_server,  inferlet, f"test_mcp_unknown_server{suffix}"),
        _make_test(_check_tool_error,      inferlet, f"test_mcp_tool_error{suffix}"),
    ])

# Expose each generated test at module top-level so external runners can
# import them by name (mirrors the original explicit-function layout).
for _t in _TESTS:
    globals()[_t.__name__] = _t


if __name__ == "__main__":
    run_tests(_TESTS)
