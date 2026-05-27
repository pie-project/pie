"""Stress / corner-case tests for the v2 runtime::launch API.

The launch-stress inferlet hosts caller and callee logic in one binary. Each
test runs the caller role with a specific scenario; the caller exercises the
launch API and prints a pass/fail summary the harness parses from stdout.
"""
import tomllib

from conftest import INFERLETS_DIR, run_inferlet, run_tests


async def _install_self(client) -> str:
    wasm = INFERLETS_DIR / "launch-stress" / "target" / "wasm32-wasip2" / "release" / "launch_stress.wasm"
    manifest = INFERLETS_DIR / "launch-stress" / "Pie.toml"
    if not wasm.exists():
        raise FileNotFoundError(f"missing wasm: {wasm}")
    info = tomllib.loads(manifest.read_text())
    pkg = info["package"]
    await client.install_program(wasm, manifest, force_overwrite=True)
    return f"{pkg['name']}@{pkg['version']}"


async def _run_scenario(client, args, scenario: str, **extra) -> str:
    payload = {
        "role": "caller",
        "scenario": scenario,
        "callee": await _install_self(client),
        **extra,
    }
    return await run_inferlet(client, "launch-stress", payload, timeout=args.timeout)


# ── Ported v1-era correctness scenarios ───────────────────────────────────


async def test_fanout_echo_small(client, args):
    out = await _run_scenario(client, args, "fanout_echo", n=10)
    assert "all 10 concurrent echoes returned correctly" in out, out


async def test_fanout_echo_large(client, args):
    out = await _run_scenario(client, args, "fanout_echo", n=50)
    assert "all 50 concurrent echoes returned correctly" in out, out


async def test_sequential_echo(client, args):
    out = await _run_scenario(client, args, "sequential_echo", n=20)
    assert "all 20 sequential echoes returned correctly" in out, out


async def test_error_propagation(client, args):
    out = await _run_scenario(client, args, "error_propagation")
    assert "got expected error" in out, out
    assert "deliberate failure from callee" in out, out


async def test_invalid_program_format(client, args):
    out = await _run_scenario(client, args, "invalid_program_format")
    assert "rejected with" in out, out


async def test_missing_program(client, args):
    out = await _run_scenario(client, args, "missing_program")
    assert "rejected with" in out, out


async def test_nested_chain(client, args):
    out = await _run_scenario(client, args, "nested_chain")
    assert "outer(nested[hello])" in out, out


async def test_multiline(client, args):
    out = await _run_scenario(client, args, "multiline")
    assert "multiline round-trip preserved newlines" in out, out


async def test_unicode(client, args):
    out = await _run_scenario(client, args, "unicode")
    assert "✓ héllo → 漢字 🎉" in out, out


async def test_username_inherited(client, args):
    out = await _run_scenario(client, args, "username_inherited")
    assert "username inherited correctly" in out, out


async def test_mixed_outcomes(client, args):
    out = await _run_scenario(client, args, "mixed_outcomes", n=20)
    assert "mixed_outcomes: 10 ok, 10 err (n=20)" in out, out


async def test_infer_once(client, args):
    out = await _run_scenario(client, args, "infer_once")
    assert "generated:" in out, out
    assert "4" in out, out


async def test_fanout_giant(client, args):
    out = await _run_scenario(client, args, "fanout_giant", n=200)
    assert "giant fanout: all 200 returned correctly" in out, out


async def test_fanout_xl(client, args):
    out = await _run_scenario(client, args, "fanout_giant", n=500)
    assert "giant fanout: all 500 returned correctly" in out, out


async def test_long_payload_64k(client, args):
    out = await _run_scenario(client, args, "long_payload", n=64 * 1024)
    assert "long payload round-trip OK at len=65536" in out, out


async def test_long_payload_1mb(client, args):
    out = await _run_scenario(client, args, "long_payload", n=1024 * 1024)
    assert "long payload round-trip OK at len=1048576" in out, out


async def test_long_payload_8mb(client, args):
    out = await _run_scenario(client, args, "long_payload", n=8 * 1024 * 1024)
    assert "long payload round-trip OK at len=8388608" in out, out


async def test_deep_nesting(client, args):
    out = await _run_scenario(client, args, "deep_nesting", n=5)
    assert "deep_nesting depth=5 returned correctly" in out, out


async def test_deeper_nesting(client, args):
    out = await _run_scenario(client, args, "deep_nesting", n=10)
    assert "deep_nesting depth=10 returned correctly" in out, out


async def test_repeated_fanout(client, args):
    out = await _run_scenario(client, args, "repeated_fanout", n=5)
    assert "5 rounds × 20 = 100 calls all clean" in out, out


async def test_sequential_burst(client, args):
    out = await _run_scenario(client, args, "sequential_echo", n=500)
    assert "all 500 sequential echoes returned correctly" in out, out


async def test_fanout_infer(client, args):
    out = await _run_scenario(client, args, "fanout_infer", n=8)
    assert "fanout_infer n=8 all generated" in out, out


async def test_nested_fanout(client, args):
    out = await _run_scenario(client, args, "nested_fanout", n=5)
    assert "nested_fanout: 5×5 = 25 processes all correct" in out, out


async def test_concurrent_self_in_callee(client, args):
    out = await _run_scenario(client, args, "concurrent_self_in_callee")
    assert (
        "concurrent_self_in_callee: 8×4 = 40 processes (parallel callees, each with parallel grandchildren)"
        in out
    ), out


# ── v2-only scenarios (handle features) ───────────────────────────────────


async def test_pid_is_uuid(client, args):
    out = await _run_scenario(client, args, "pid_is_uuid")
    assert "pid was " in out, out


async def test_drop_detach(client, args):
    out = await _run_scenario(client, args, "drop_detach")
    assert "ran to completion after parent dropped handle" in out, out


async def test_cancel_mid_flight(client, args):
    out = await _run_scenario(client, args, "cancel_mid_flight")
    assert "got expected err: cancelled" in out, out


async def test_timeout_then_cancel(client, args):
    out = await _run_scenario(client, args, "timeout_then_cancel")
    assert "cancelled with: cancelled" in out, out


async def test_cancel_after_done(client, args):
    out = await _run_scenario(client, args, "cancel_after_done")
    assert "child finished, cancel() would be a no-op" in out, out


async def test_fanout_with_cancel(client, args):
    out = await _run_scenario(client, args, "fanout_with_cancel", n=10)
    # Most children should be cancelled; we tolerate a few sneaking through
    # as Ok if the sleep race fell their way.
    assert "fanout_with_cancel n=10" in out, out


def tests():
    return [
        # ported v1 correctness
        test_fanout_echo_small,
        test_fanout_echo_large,
        test_sequential_echo,
        test_error_propagation,
        test_invalid_program_format,
        test_missing_program,
        test_nested_chain,
        test_multiline,
        test_unicode,
        test_username_inherited,
        test_mixed_outcomes,
        test_infer_once,
        # heavier
        test_fanout_giant,
        test_fanout_xl,
        test_long_payload_64k,
        test_long_payload_1mb,
        test_long_payload_8mb,
        test_deep_nesting,
        test_deeper_nesting,
        test_repeated_fanout,
        test_sequential_burst,
        test_fanout_infer,
        test_nested_fanout,
        test_concurrent_self_in_callee,
        # v2-only
        test_pid_is_uuid,
        test_drop_detach,
        test_cancel_mid_flight,
        test_timeout_then_cancel,
        test_cancel_after_done,
        test_fanout_with_cancel,
    ]


if __name__ == "__main__":
    run_tests(tests())
