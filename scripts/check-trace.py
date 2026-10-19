#!/usr/bin/env python3
"""Check a `.pie` agent trace against TRACE_FORMAT.md and summarize it.

    python3 scripts/check-trace.py examples/traces/agent-3.pie

Exits non-zero on the first file that breaks the format.
"""

import json
import sys

AGENT_SPANS = {"prefill", "decode", "tool", "queued"}
COUNTERS = {"kv_pages", "gpu_busy", "gpu_share", "memory"}
META_KEYS = [
    "pie_version", "commit", "started_at", "model", "scheme", "device",
    "hardware", "kv_page_bytes", "kv_pages_total", "weights_bytes",
]
TEXT_KEYS = {"text", "prompt", "output", "content", "arguments", "completion", "message", "input"}


def fail(path, why):
    print(f"{path}: {why}")
    sys.exit(1)


def keys(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from keys(v)
    elif isinstance(value, list):
        for v in value:
            yield from keys(v)


def check(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("format") != "pie-trace" or doc.get("version") != 1:
        fail(path, "not a pie-trace version 1")
    for key in META_KEYS:
        if key not in doc["meta"]:
            fail(path, f"meta lacks {key}")
    for key in ("name", "memory_gb", "bandwidth_gbs"):
        if key not in doc["meta"]["hardware"]:
            fail(path, f"meta.hardware lacks {key}")
    banned = TEXT_KEYS & set(keys(doc))
    if banned:
        fail(path, f"fields that could carry text: {sorted(banned)}")

    events = doc["traceEvents"]
    names = {}
    agent_spans = {}
    steps, gpus, busy, share = [], [], [], []
    for e in events:
        ph = e["ph"]
        if ph == "M":
            if e["name"] == "thread_name":
                names[(e["pid"], e["tid"])] = e["args"]["name"]
            continue
        if not isinstance(e["ts"], int) or e["ts"] < 0:
            fail(path, f"bad ts in {e}")
        if ph == "X":
            if not isinstance(e["dur"], int) or e["dur"] < 0:
                fail(path, f"bad dur in {e}")
            if e["pid"] == 1:
                if e["name"] not in AGENT_SPANS:
                    fail(path, f"agent span {e['name']!r}")
                a = e["args"]
                if e["name"] == "prefill" and set(a) != {"tokens", "cached_tokens"}:
                    fail(path, f"prefill args {a}")
                if e["name"] == "decode" and set(a) != {"tokens"}:
                    fail(path, f"decode args {a}")
                if e["name"] == "tool" and set(a) != {"tool"}:
                    fail(path, f"tool args {a}")
                agent_spans.setdefault(e["tid"], []).append(e)
            elif (e["pid"], e["tid"]) == (2, 1):
                if e["name"] != "step":
                    fail(path, f"scheduler span {e['name']!r}")
                a = e["args"]
                if set(a) != {"step", "prefill_tokens", "decode_tokens", "agents", "gpu_us"}:
                    fail(path, f"step args {a}")
                if sum(v["prefill"] for v in a["agents"].values()) != a["prefill_tokens"]:
                    fail(path, f"step {a['step']} prefill rows do not add up")
                if sum(v["decode"] for v in a["agents"].values()) != a["decode_tokens"]:
                    fail(path, f"step {a['step']} decode rows do not add up")
                steps.append(e)
            elif (e["pid"], e["tid"]) == (2, 2):
                if e["name"] != "gpu" or set(e["args"]) != {"step"}:
                    fail(path, f"gpu span {e}")
                gpus.append(e)
            else:
                fail(path, f"span on an unknown track {e}")
        elif ph == "C":
            if e["pid"] != 2 or e["name"] not in COUNTERS:
                fail(path, f"counter {e}")
            if e["name"] == "gpu_busy":
                busy.append(e)
            if e["name"] == "gpu_share":
                share.append(e)
            if e["name"] == "kv_pages" and "free" in e["args"] and doc["meta"]["kv_pages_total"]:
                used = sum(v for k, v in e["args"].items() if k.startswith("agent-"))
                if used + e["args"]["free"] != doc["meta"]["kv_pages_total"]:
                    fail(path, f"kv_pages do not add up at {e['ts']}")
        elif ph == "i":
            if e.get("s") != "t" or e["name"] not in {"start", "end", "evict"}:
                fail(path, f"instant {e}")
        else:
            fail(path, f"phase {ph!r}")

    for tid, spans in agent_spans.items():
        if (1, tid) not in names:
            fail(path, f"agent {tid} has no thread name")
        spans.sort(key=lambda e: e["ts"])
        for a, b in zip(spans, spans[1:]):
            if b["ts"] < a["ts"] + a["dur"]:
                fail(path, f"agent {tid}: {a['name']} and {b['name']} overlap at {b['ts']}")
    for track, spans in (("step", steps), ("gpu", gpus)):
        for a, b in zip(spans, spans[1:]):
            if b["ts"] < a["ts"] + a["dur"]:
                fail(path, f"{track} spans overlap at {b['ts']}")
    if len(busy) != len(share):
        fail(path, "gpu_busy and gpu_share are not paired")
    for b, s in zip(busy, share):
        total = sum(s["args"].values())
        if b["ts"] != s["ts"] or abs(total - b["args"]["pct"]) > 0.01:
            fail(path, f"gpu_share {total} != gpu_busy {b['args']['pct']} at {b['ts']}")

    # Summary.
    end = max((e["ts"] + e.get("dur", 0) for e in events if "ts" in e), default=0)
    gpu_us = sum(e["dur"] for e in gpus)
    print(f"{path}: ok — {len(agent_spans)} agent(s), {len(steps)} steps, "
          f"{len(gpus)} gpu spans, {len(events)} events, {end / 1e6:.1f} s, "
          f"GPU busy {100 * gpu_us / max(end, 1):.1f}%")
    m = doc["meta"]
    print(f"  {m['model']} {m['scheme']} on {m['hardware']['name']} "
          f"({m['kv_pages_total']} kv pages × {m['kv_page_bytes']} B)")
    for tid in sorted(agent_spans):
        totals = {}
        for e in agent_spans[tid]:
            totals.setdefault(e["name"], [0, 0])
            totals[e["name"]][0] += e["dur"]
            totals[e["name"]][1] += e["args"].get("tokens", 0)
        parts = ", ".join(
            f"{k} {v[0] / 1e6:.2f}s" + (f" ({v[1]} tok)" if v[1] else "")
            for k, v in sorted(totals.items())
        )
        print(f"  {names[(1, tid)]}: {parts}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    for p in sys.argv[1:]:
        check(p)
