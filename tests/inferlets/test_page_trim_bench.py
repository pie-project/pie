"""E2E test for the page-trim-bench inferlet.

Verifies the inferlet runs to completion under both modes (mask on / off) and
that the timing summary it prints contains the expected key=value lines. We
keep the workload small here (short prompt, few decode steps) so the test is
fast. Use the canonical scripts in ``benches/`` for engine-level benchmark
numbers.
"""
import re

from conftest import run_inferlet, run_tests


def _parse_summary(output: str) -> dict[str, str]:
    """Extract the trailing key=value summary lines from the inferlet's stdout."""
    pairs: dict[str, str] = {}
    for line in output.splitlines():
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)=([-+0-9.eE]+|true|false)$", line.strip())
        if m:
            pairs[m.group(1)] = m.group(2)
    return pairs


def _required_keys() -> set[str]:
    return {
        "prompt_tokens",
        "decode_steps",
        "sink_size",
        "window_size",
        "use_mask",
        "page_size",
        "prefill_ms",
        "decode_ms",
        "decode_per_step_ms",
        "decode_tokens_per_sec",
    }


async def test_page_trim_bench_with_mask(client, args):
    output = await run_inferlet(
        client,
        "page-trim-bench",
        {
            "prompt_tokens": 256,
            "decode_steps": 16,
            "sink_size": 4,
            "window_size": 32,
            "use_mask": True,
        },
        timeout=args.timeout,
    )
    assert "page-trim-bench" in output, f"Missing header in output:\n{output}"

    summary = _parse_summary(output)
    missing = _required_keys() - summary.keys()
    assert not missing, f"Missing summary keys: {missing}\nOutput:\n{output}"

    assert summary["use_mask"] == "true"
    assert int(summary["prompt_tokens"]) == 256
    assert int(summary["decode_steps"]) == 16
    assert float(summary["decode_per_step_ms"]) > 0


async def test_page_trim_bench_without_mask(client, args):
    output = await run_inferlet(
        client,
        "page-trim-bench",
        {
            "prompt_tokens": 256,
            "decode_steps": 16,
            "use_mask": False,
        },
        timeout=args.timeout,
    )
    summary = _parse_summary(output)
    missing = _required_keys() - summary.keys()
    assert not missing, f"Missing summary keys: {missing}\nOutput:\n{output}"
    assert summary["use_mask"] == "false"


if __name__ == "__main__":
    run_tests([test_page_trim_bench_with_mask, test_page_trim_bench_without_mask])
