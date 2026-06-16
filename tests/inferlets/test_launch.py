"""E2E test for the v2 runtime::launch API.

launch-caller invokes launch-callee via `inferlet::launch(...)?.await`. The
caller prints the child's pid and the returned text; the test checks both.
"""
import tomllib

from conftest import INFERLETS_DIR, run_inferlet, run_tests


async def _install(client, name: str) -> str:
    """Install an inferlet by name; return its `name@version` id."""
    wasm_name = name.replace("-", "_")
    inferlet_dir = INFERLETS_DIR / name
    candidates = [
        inferlet_dir / "target" / "wasm32-wasip2" / "release" / f"{wasm_name}.wasm",
        inferlet_dir / "target" / "wasm32-wasip2" / "debug" / f"{wasm_name}.wasm",
    ]
    wasm_path = next((p for p in candidates if p.exists()), None)
    manifest_path = inferlet_dir / "Pie.toml"
    if wasm_path is None:
        raise FileNotFoundError(f"No WASM binary for {name}")

    manifest = tomllib.loads(manifest_path.read_text())
    inferlet_id = f"{manifest['package']['name']}@{manifest['package']['version']}"
    await client.install_program(wasm_path, manifest_path, force_overwrite=True)
    return inferlet_id


async def test_launch(client, args):
    callee_id = await _install(client, "launch-callee")
    output = await run_inferlet(
        client,
        "launch-caller",
        {"callee": callee_id, "prompt": "What is 2+2?"},
        timeout=args.timeout,
    )
    assert "[caller] launching" in output, output
    assert "[caller] child pid:" in output, output
    assert "[caller] child returned:" in output, output
    # Pid line must be a 36-char UUID with 4 dashes.
    pid_line = next(
        line for line in output.splitlines() if line.startswith("[caller] child pid:")
    )
    pid = pid_line.split(":", 1)[1].strip()
    assert len(pid) == 36 and pid.count("-") == 4, f"pid not a UUID: {pid!r}"
    tail = output.split("[caller] child returned:", 1)[1].strip()
    assert len(tail) > 0, f"empty callee output:\n{output}"


if __name__ == "__main__":
    run_tests([test_launch])
