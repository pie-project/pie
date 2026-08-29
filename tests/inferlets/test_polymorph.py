"""Concurrent polymorphic-batching E2E: many DIFFERENT inferlets in flight at once.

The curated suite (`test_curated.py`) proves each inferlet alone against a
quiet server; every launch has the batch to itself. This file proves the other
half of the serving story: one server, one model, and a fleet of heterogeneous
programs whose fires must share waves — plain decode next to beam forks next to
grammar-masked rows next to speculative verify windows. What is asserted is
liveness (everyone finishes, nothing vanishes) plus each program's own cheap
correctness gate, because a batcher that corrupts a neighbours' rows shows up
exactly there.

Usage::

    sdk/server/python/.venv/bin/python tests/inferlets/test_polymorph.py
    ... --model Qwen/Qwen3-0.6B --timeout 300
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import tomllib
from pathlib import Path

os.environ.setdefault("PIE_CUDA_KV_ENVELOPES", "1")

from pie_client import Event  # noqa: E402

from conftest import INFERLETS_DIR, run_tests  # noqa: E402


# ---------------------------------------------------------------------------
# Install-once / launch-many plumbing.
#
# `conftest.run_inferlet` installs on every call, which is correct for a
# sequential suite and a race for a concurrent one (two `force_overwrite`
# installs of the SAME program in flight). Here programs are installed once,
# up front, and launches then share the installed id.
# ---------------------------------------------------------------------------

_installed: dict[str, str] = {}  # name -> inferlet_id


def _resolve(name: str) -> tuple[Path, Path]:
    wasm_name = name.replace("-", "_")
    inferlet_dir = INFERLETS_DIR / name
    candidates = [
        INFERLETS_DIR / "target" / "wasm32-wasip2" / "release" / f"{wasm_name}.wasm",
        INFERLETS_DIR / "target" / "wasm32-wasip2" / "debug" / f"{wasm_name}.wasm",
        inferlet_dir / "target" / "wasm32-wasip2" / "release" / f"{wasm_name}.wasm",
        inferlet_dir / "target" / "wasm32-wasip2" / "debug" / f"{wasm_name}.wasm",
        inferlet_dir / "target" / f"{wasm_name}.wasm",
    ]
    present = [p for p in candidates if p.exists()]
    wasm_path = max(present, key=lambda p: p.stat().st_mtime, default=None)
    manifest_path = inferlet_dir / "Pie.toml"
    if wasm_path is None:
        raise FileNotFoundError(f"no wasm for {name}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"no Pie.toml for {name}")
    return wasm_path, manifest_path


async def _ensure_installed(client, name: str) -> str:
    if name in _installed:
        return _installed[name]
    wasm_path, manifest_path = _resolve(name)
    manifest = tomllib.loads(manifest_path.read_text())
    inferlet_id = f"{manifest['package']['name']}@{manifest['package']['version']}"
    await client.install_program(wasm_path, manifest_path, force_overwrite=True)
    _installed[name] = inferlet_id
    return inferlet_id


async def _launch(client, name: str, inputs: dict, *, timeout: float,
                  delay: float = 0.0) -> dict:
    """Launch one already-installed inferlet and collect its output.

    Returns a record rather than raising, so one failure cannot cancel the
    rest of a `gather` and the report can show every outcome side by side.
    """
    if delay:
        await asyncio.sleep(delay)
    inferlet_id = _installed[name]
    start = time.time()
    record = {"name": name, "status": "?", "wall": 0.0, "output": "", "detail": ""}
    try:
        process = await client.launch_process(inferlet_id, input=inputs)
        parts: list[str] = []
        while True:
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                raise RuntimeError("TIMEOUT")
            event, msg = await asyncio.wait_for(process.recv(), timeout=remaining)
            if event in (Event.Stdout, Event.Message):
                parts.append(msg)
            elif event == Event.Return:
                parts.append(msg)
                record.update(status="ok", output="".join(parts))
                break
            elif event == Event.Error:
                raise RuntimeError(str(msg)[:500])
    except (RuntimeError, asyncio.TimeoutError) as e:
        record.update(status="fail", detail=str(e) or "TIMEOUT",
                      output="")
    record["wall"] = time.time() - start
    return record


# ---------------------------------------------------------------------------
# Per-program correctness gates — cheap, and aimed at cross-contamination.
# ---------------------------------------------------------------------------

def _gate_attends(record: dict) -> None:
    lowered = record["output"].lower()
    assert "france" in lowered or "paris" in lowered, (
        f"{record['name']} did not attend its prompt: {record['output'][:200]!r}")


def _gate_nonempty(record: dict) -> None:
    assert record["output"].strip(), f"{record['name']} returned empty output"


def _gate_json(key: str):
    def gate(record: dict) -> None:
        start = record["output"].find("{")
        assert start >= 0, f"{record['name']} returned no JSON: {record['output'][:200]!r}"
        report = json.loads(record["output"][start:])
        assert key in report, f"{record['name']} report lacks {key!r}: {report}"
    return gate


def _gate_contains(needle: str):
    def gate(record: dict) -> None:
        assert needle in record["output"], (
            f"{record['name']} output lacks {needle!r}: {record['output'][:200]!r}")
    return gate


# ---------------------------------------------------------------------------
# The mixes. Each entry: (name, inputs, gate, stagger-delay-seconds).
# ---------------------------------------------------------------------------

ATTEND = "The capital of France is"

# Heterogeneous fleet: every entry brings a different fire shape to the batch.
POLYMORPH_FLEET = [
    # plain host-driven greedy decode, with the attention gate
    ("text-completion", {"prompt": ATTEND, "max_tokens": 12}, _gate_attends, 0.0),
    # device-carried decode
    ("naive-baseline", {"max_tokens": 16}, _gate_json("sampler"), 0.0),
    # chat template + decode
    ("chat-completion", {"prompt": "Say hello.", "max_tokens": 12}, _gate_nonempty, 0.0),
    # logit-mixing sampler (two contexts per step)
    ("xtc-sampling", {"max_tokens": 8, "probability": 0.5}, _gate_nonempty, 0.1),
    # sorted-tail samplers (top_k over large k)
    ("locally-typical-sampling", {"max_tokens": 8}, _gate_nonempty, 0.1),
    ("tail-free-sampling", {"max_tokens": 8}, _gate_nonempty, 0.1),
    # grammar-masked rows
    ("json-schema-constrained-decoding",
     {"prompt": "Return an object with an integer field named value.",
      "schema": ('{"type":"object","properties":{"value":{"type":"integer"}},'
                 '"required":["value"],"additionalProperties":false}'),
      "max_tokens": 64},
     _gate_contains("value"), 0.2),
    # speculative verify windows (rows-per-lane varies step to step)
    ("cacheback-speculative-decoding",
     {"prompt": "Explain in detail why the sky appears blue during the day.",
      "max_tokens": 24, "max_ngram": 4, "draft_length": 4},
     _gate_json("verification_steps"), 0.2),
    # forked prefixes sharing KV
    ("prefix-tree-kv-cache", {"num_tokens": 2}, _gate_contains("city at dawn:"), 0.3),
    # logit-surgery sampler
    ("repetition-penalty",
     {"max_tokens": 8, "repetition_penalty": 1.5, "frequency_penalty": 0.1},
     _gate_json("mean_penalized"), 0.3),
    # prompt re-tokenization boundary work
    ("token-healing", {"prompt": "The capital of Fra", "max_tokens": 6},
     _gate_json("healed"), 0.4),
]


async def _run_fleet(client, args, fleet, *, label: str) -> None:
    for name, *_ in fleet:
        await _ensure_installed(client, name)
    t0 = time.time()
    records = await asyncio.gather(*[
        _launch(client, name, inputs, timeout=args.timeout, delay=delay)
        for name, inputs, _gate, delay in fleet
    ])
    wall = time.time() - t0
    failures: list[str] = []
    print(f"\n    [{label}] fleet={len(fleet)} wall={wall:.1f}s")
    for (name, _inputs, gate, _delay), record in zip(fleet, records):
        line = f"      {record['name']:34s} {record['status']:4s} {record['wall']:6.1f}s"
        if record["status"] == "ok":
            try:
                gate(record)
            except AssertionError as e:
                record["status"] = "gate"
                failures.append(f"{name}: {e}")
                line += "  GATE FAILED"
        else:
            failures.append(f"{name}: {record['detail'][:300]}")
            line += f"  {record['detail'][:120]}"
        print(line)
    assert not failures, f"[{label}] {len(failures)} failed:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# The tests, in escalation order.
# ---------------------------------------------------------------------------

async def test_warmup_single(client, args):
    """One inferlet alone — the boot smoke. Everything after shares its server."""
    await _run_fleet(client, args, [POLYMORPH_FLEET[0]], label="warmup")


async def test_homogeneous_burst(client, args):
    """Six copies of one program at once: same shape, different prompts/lengths."""
    fleet = [
        ("text-completion", {"prompt": ATTEND, "max_tokens": 8}, _gate_attends, 0.0),
        ("text-completion", {"prompt": ATTEND, "max_tokens": 24}, _gate_attends, 0.0),
        ("text-completion", {"prompt": "Water is made of", "max_tokens": 16},
         _gate_nonempty, 0.0),
        ("text-completion", {"prompt": "1, 2, 3, 4,", "max_tokens": 16},
         _gate_nonempty, 0.05),
        ("text-completion", {"prompt": "Roses are red,", "max_tokens": 32},
         _gate_nonempty, 0.05),
        ("text-completion", {"prompt": ATTEND, "max_tokens": 48}, _gate_attends, 0.1),
    ]
    await _run_fleet(client, args, fleet, label="burst")


async def test_polymorphic_mix(client, args):
    """The full heterogeneous fleet, launched into one server within ~0.4s."""
    await _run_fleet(client, args, POLYMORPH_FLEET, label="mix")


async def test_churn_two_waves(client, args):
    """A second wave arrives while the first is mid-decode.

    Admission, growth and the runahead frontier all see joiners and leavers
    mid-flight rather than a fixed fleet.
    """
    wave1 = [(n, i, g, d) for n, i, g, d in POLYMORPH_FLEET[:6]]
    wave2 = [(n, i, g, d + 1.5) for n, i, g, d in POLYMORPH_FLEET[5:]]
    await _run_fleet(client, args, wave1 + wave2, label="churn")


# The masked family needs a model text that DECLARES `attention.masked` —
# only gemma4 does (crates/engine-cuda/tests/masked_axis.rs pins this). On a
# qwen load these lanes are refused as `Fault::Maskless`, correctly: the qwen
# text bakes no masked arm, so there is nowhere for the bits to go. Run this
# file twice: once with the default qwen model (the fleets above), once with
# --model google/gemma-4-E4B-it, which selects the masked fleet instead.
MASKED_FLEET = [
    ("text-completion", {"prompt": ATTEND, "max_tokens": 12}, _gate_attends, 0.0),
    ("sliding-window-attention",
     {"prompt": "Count upward.", "max_tokens": 8, "window_size": 2}, _gate_nonempty, 0.0),
    ("attention-sink",
     {"prompt": "Count upward.", "max_tokens": 8, "sink_size": 1, "window_size": 2},
     _gate_nonempty, 0.1),
    ("beam-search", {"max_tokens": 4, "beams": 3}, _gate_contains("[beam] width=3"), 0.1),
    ("consensus-decoding",
     {"question": "What is 2 + 2?", "num_candidates": 2, "max_tokens": 4},
     _gate_nonempty, 0.1),
    ("contrastive-decoding", {"max_tokens": 4}, _gate_nonempty, 0.2),
]


async def test_masked_family_mix(client, args):
    """The per-row-mask fleet, concurrent, on a model text with a masked arm."""
    await _run_fleet(client, args, MASKED_FLEET, label="masked-mix")


def tests():
    import sys
    argv = " ".join(sys.argv)
    if "gemma" in argv:
        return [test_masked_family_mix]
    return [
        test_warmup_single,
        test_homogeneous_burst,
        test_polymorphic_mix,
        test_churn_two_waves,
    ]


if __name__ == "__main__":
    run_tests(tests(), description="Concurrent polymorphic-batching E2E")
