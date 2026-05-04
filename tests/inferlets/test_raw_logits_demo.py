"""E2E test for the raw-logits-demo inferlet.

Verifies that:
  1. `Sampler::raw_logits()` round-trips through every layer (worker → IPC →
     scheduler → WIT → SDK) and returns a vocab_size * 4 byte buffer.
  2. argmax(raw_logits) matches the token a greedy sampler would have picked,
     proving the logits really are the model's pre-softmax output (and not,
     say, a stale buffer or a permuted view).

Also reports the round-trip overhead of returning the full logit vector
versus returning a single sampled token id, so you can see what the new path
costs in practice.

Run::

    uv run python tests/inferlets/test_raw_logits_demo.py --model Qwen/Qwen3-0.6B
"""
from __future__ import annotations

import re

from conftest import run_inferlet, run_tests


_KV_RE = re.compile(r"^([A-Z0-9_]+)=(.+)$", re.MULTILINE)


def _parse(output: str) -> dict[str, str]:
    return dict(_KV_RE.findall(output))


async def test_raw_logits_demo(client, args):
    # 5 paired iterations is enough to see a stable average without making
    # the test slow on small models.
    output = await run_inferlet(
        client,
        "raw-logits-demo",
        extra_args={"iters": 50},
        timeout=args.timeout,
    )

    fields = _parse(output)
    required = {
        "VOCAB_SIZE", "ITERS",
        "GREEDY_AVG_MS", "GREEDY_MIN_MS",
        "RAW_LOGITS_AVG_MS", "RAW_LOGITS_MIN_MS",
        "OVERHEAD_AVG_MS", "OVERHEAD_MIN_MS",
        "PAYLOAD_BYTES", "ARGMAX_MATCHES_GREEDY",
    }
    missing = required - set(fields)
    assert not missing, f"Inferlet output missing keys {missing}\n----\n{output}"

    vocab_size = int(fields["VOCAB_SIZE"])
    iters = int(fields["ITERS"])
    payload_bytes = int(fields["PAYLOAD_BYTES"])
    greedy_avg = float(fields["GREEDY_AVG_MS"])
    greedy_min = float(fields["GREEDY_MIN_MS"])
    raw_avg = float(fields["RAW_LOGITS_AVG_MS"])
    raw_min = float(fields["RAW_LOGITS_MIN_MS"])
    overhead_avg = float(fields["OVERHEAD_AVG_MS"])
    overhead_min = float(fields["OVERHEAD_MIN_MS"])

    # ---- Correctness ------------------------------------------------------
    assert vocab_size > 1000, f"Implausible vocab_size={vocab_size}"
    assert payload_bytes == vocab_size * 4, (
        f"PAYLOAD_BYTES={payload_bytes} != vocab_size*4={vocab_size * 4}"
    )

    matches_str = fields["ARGMAX_MATCHES_GREEDY"]  # e.g. "5/5"
    matched, total = (int(x) for x in matches_str.split("/"))
    assert total == iters
    assert matched == total, (
        f"argmax(raw_logits) disagreed with greedy on {total - matched}/{total} "
        f"iterations — raw logits may not be the same tensor the sampler sees.\n"
        f"Output:\n{output}"
    )

    # ---- Overhead reporting (informational, not a hard assertion) ---------
    # The MIN values are the steady-state cost — independent of one-time JIT
    # compile in the first iteration. AVG includes that and is sensitive to
    # warmup, so we report both but believe MIN.
    def _gbps(extra_ms: float) -> str:
        if extra_ms <= 0:
            return "N/A (no measurable overhead)"
        return f"{payload_bytes / (extra_ms * 1e-3) / 1e9:.2f} GB/s"

    pct_min = (overhead_min / greedy_min * 100.0) if greedy_min > 0 else float("nan")
    print(
        f"\n  raw-logits overhead (min):  {overhead_min:+.3f} ms/call "
        f"({pct_min:+.1f}% vs greedy {greedy_min:.3f} ms)  "
        f"payload {payload_bytes / 1024:.1f} KiB  effective {_gbps(overhead_min)}"
    )
    print(
        f"  raw-logits overhead (avg):  {overhead_avg:+.3f} ms/call "
        f"vs greedy {greedy_avg:.3f} ms"
    )

    # Soft sanity bound on the steady-state numbers: raw-logits shouldn't be
    # more than 10x slower than greedy. If it is, the wire path likely
    # regressed (e.g. someone changed `list<list<u8>>` back to `list<f32>`).
    if greedy_min > 0:
        ratio = raw_min / greedy_min
        assert ratio < 10.0, (
            f"raw_logits is {ratio:.1f}x slower than greedy "
            f"({raw_min:.3f} vs {greedy_min:.3f} ms) — investigate wire path."
        )


def tests():
    return [test_raw_logits_demo]


if __name__ == "__main__":
    run_tests(tests(), description="Raw-logits inferlet E2E test")
