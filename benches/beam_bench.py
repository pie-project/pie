"""Beam-search benchmark for the `beam-search` inferlet.

Sibling of `pie_bench.py` / `vllm_bench.py` / `sglang_bench.py`: same
`common.py` metric definitions, one engine per file. This one drives the
on-device Design B beam (ancestry encoded as a dense per-beam attention mask
over a shared fixed page pool) and nothing else.

Two things make a beam benchmark different from every other file here, and both
are structural rather than stylistic.

**A beam run has two token counts, and neither one alone is honest.** The engine
advances `beam_width` hypotheses on every step; exactly one of them is ever
returned. Report the first and the search looks free. Report the second and the
hardware looks idle. So every row carries both — `engine_output_tok_per_s` and
`goodput_output_tok_per_s` — and the shared `output_tok_per_s` is goodput,
because `RequestResult.output_tokens` counts what the request emitted. A
non-beam baseline is therefore comparable against `output_tok_per_s` directly,
and is never accidentally compared against the engine figure.

**A dropped mask is invisible.** `ModelCapabilities` has no
`supports_custom_mask` flag, and a model family whose forward never reads
`custom_mask_d` does not error — it returns fluent, plausible output at an
entirely ordinary speed. On this branch that is not hypothetical. The `kimi`
(deepseek_v2 / deepseek_v3 / kimi_k2) and `glm5` families reach
`dispatch_attention_mla_bf16` — its only two call sites are
`kimi_forward.cpp:767` and `glm5_forward.cpp:472` — whose sole mask argument is
the DSA index mask, honoured only on the naive sm100 fallback and dropped on the
FlashInfer FA2 path. `deepseek_v4` is excluded for a different reason: it attends
through `dispatch_attention_flashinfer_prefill_bf16` /
`launch_attention_compressed_paged_bf16`, not the MLA dispatch, and `custom_mask`
appears nowhere under `driver/cuda/src/model/deepseek_v4/`. Timing any of them
would produce a real number for a search that never happened. So the gate below
runs FIRST, and a configuration that fails it is struck rather than timed.

Ladder: Qwen3-0.6B only. The MoE rungs are kernel-blocked, not capacity-blocked
— see the family note above — so they are not attempted here.

Run from the repo root:

    uv --project sdk/python-server run python benches/beam_bench.py \
      --model Qwen/Qwen3-0.6B --device cuda:0 \
      --json-out beam.json 2>&1 | tee test-$(date +%Y%m%d-%H%M%S).log
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    BenchSummary,
    RequestResult,
    hash_output_tokens,
    print_summary,
    resolve_local_model,
    summarize,
)

INFERLET = "beam-search"
INFERLETS_DIR = Path(__file__).resolve().parent.parent / "tests" / "inferlets"

# ── The matrix ──────────────────────────────────────────────────────────────
#
# `beam-search` holds a fixed pool of POOL_PAGES(8) * PAGE_T(16) = 128 flat
# cells and refuses `max_tokens > (POOL - 1) / beams`. So the budget a width can
# reach is 127 / 63 / 31 / 15 for widths 1 / 2 / 4 / 8.
#
# 31 is the largest budget widths 1, 2 and 4 all share, so those three are
# budget-matched and belong in one table. Width 8 cannot reach it — its ceiling
# is 15 — so it is reported as its own row and never placed alongside the
# others. Comparing throughput across different token budgets would fold the
# budget difference into the width effect.
PRIMARY_WIDTHS = (1, 2, 4)
PRIMARY_MAX_TOKENS = 31
EXTRA_POINTS = ((8, 15),)


def inferlet_paths() -> tuple[Path, Path, str]:
    """Locate the built wasm + manifest, mirroring the curated suite's search."""
    wasm_name = INFERLET.replace("-", "_")
    inferlet_dir = INFERLETS_DIR / INFERLET
    candidates = [
        INFERLETS_DIR / "target" / "wasm32-wasip2" / "release" / f"{wasm_name}.wasm",
        INFERLETS_DIR / "target" / "wasm32-wasip2" / "debug" / f"{wasm_name}.wasm",
        inferlet_dir / "target" / "wasm32-wasip2" / "release" / f"{wasm_name}.wasm",
    ]
    wasm = next((c for c in candidates if c.exists()), None)
    if wasm is None:
        raise FileNotFoundError(
            f"no wasm for {INFERLET}; build with: cd {INFERLETS_DIR} && "
            f"cargo build --target wasm32-wasip2 --release -p {INFERLET}"
        )
    manifest = inferlet_dir / "Pie.toml"
    pkg = tomllib.loads(manifest.read_text())["package"]
    return wasm, manifest, f"{pkg['name']}@{pkg['version']}"


@dataclasses.dataclass
class BeamRun:
    """One `beam-search` invocation."""

    wall_s: float
    width: int
    max_tokens: int
    mask: bool
    returned_tokens: list[int]
    kv_cells_occupied_peak: int
    best_score: float
    greedy_mismatches: int

    @property
    def digest(self) -> str:
        # The shared helper, so this digest is byte-identical to the one every
        # other harness records in `RequestResult.output_token_sha256`.
        return hash_output_tokens(self.returned_tokens)


def parse_output(text: str, wall_s: float) -> BeamRun:
    """Read the inferlet's two `[beam]` report lines.

    Token ids come from the inferlet directly. Re-tokenizing the decoded text
    would not be equivalent: detokenize/tokenize is not the identity, so a
    digest taken that way could agree across two different token streams —
    exactly the false negative the gate exists to prevent.
    """
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("[beam] "):
            for token in line.removeprefix("[beam] ").split():
                key, _, value = token.partition("=")
                if value:
                    fields[key] = value
    missing = {
        "width",
        "steps",
        "best_score",
        "greedy_mismatches",
        "mask",
        "kv_cells_occupied_peak",
        "returned_tokens",
    } - fields.keys()
    if missing:
        raise RuntimeError(f"inferlet output missing {sorted(missing)}: {text!r}")
    raw = fields["returned_tokens"]
    return BeamRun(
        wall_s=wall_s,
        width=int(fields["width"]),
        max_tokens=int(fields["steps"]),
        mask=fields["mask"] == "true",
        returned_tokens=[int(t) for t in raw.split(",") if t],
        kv_cells_occupied_peak=int(fields["kv_cells_occupied_peak"]),
        best_score=float(fields["best_score"]),
        greedy_mismatches=int(fields["greedy_mismatches"]),
    )


async def launch(client, pkg: str, *, width: int, max_tokens: int, mask: bool,
                 timeout: int) -> BeamRun:
    from pie_client import Event

    params = {"beams": width, "max_tokens": max_tokens, "mask": mask}
    parts: list[str] = []
    start = time.perf_counter()
    process = await client.launch_process(pkg, input=params)
    while True:
        if time.perf_counter() - start > timeout:
            raise RuntimeError(f"timeout: {params}")
        event, msg = await asyncio.wait_for(process.recv(), timeout=timeout)
        if event in (Event.Stdout, Event.Message):
            parts.append(msg)
        elif event == Event.Return:
            parts.append(msg)
            return parse_output("".join(parts), time.perf_counter() - start)
        elif event == Event.Error:
            raise RuntimeError(msg)


async def gate(client, pkg: str, width: int, max_tokens: int,
               timeout: int) -> dict[str, Any]:
    """Ancestry mask vs attend-all mask, by returned-beam token digest.

    Runs before timing. Both arms bind the AttnMask port; only its CONTENTS
    differ — per-beam ancestry against all-true. Equal digests therefore mean
    attention did not act on the contents, i.e. the "beam" was one
    undifferentiated pool read.

    The contents have to be the variable, not the binding. Omitting the port for
    the control would leave `has_user_mask` false, keep the batch in
    `single_token_mode`, and send the fire down the driver's XQA decode path
    with a different KV-write path too. Two arms differing in kernel diverge on
    floating-point accumulation order alone, and over dozens of autoregressive
    steps one flipped near-tie in `top_k(log_softmax(logits) + scores)` is
    enough — which would read as `diverged` whether or not the mask was ever
    honoured, biasing the gate toward false admission.

    One exception is a property of beam search rather than of any driver: at
    width 1 a single beam's ancestry already IS the whole filled span, so the
    two masks are the same mask and the digests must agree. That is correct
    behaviour, not a dropped mask, and it is reported as `vacuous` so the
    distinction is never silently collapsed into a pass or a strike.
    """
    on = await launch(client, pkg, width=width, max_tokens=max_tokens, mask=True,
                      timeout=timeout)
    off = await launch(client, pkg, width=width, max_tokens=max_tokens, mask=False,
                       timeout=timeout)
    diverged = on.digest != off.digest
    verdict = "vacuous" if width == 1 else ("diverged" if diverged else "matched")
    return {
        "width": width,
        "max_tokens": max_tokens,
        "mask_on_sha256": on.digest,
        "mask_off_sha256": off.digest,
        "diverged": diverged,
        "verdict": verdict,
        "greedy_mismatches": on.greedy_mismatches,
        "best_score": on.best_score,
    }


async def measure(client, pkg: str, width: int, max_tokens: int, reps: int,
                  timeout: int, model: str) -> tuple[BenchSummary, list[RequestResult]]:
    """Timed mask-on reps for one admitted configuration."""
    await launch(client, pkg, width=width, max_tokens=max_tokens, mask=True,
                 timeout=timeout)  # discarded warm-up
    runs: list[BeamRun] = []
    start = time.perf_counter()
    for _ in range(reps):
        runs.append(await launch(client, pkg, width=width, max_tokens=max_tokens,
                                 mask=True, timeout=timeout))
    wall_s = time.perf_counter() - start

    results = [
        RequestResult(
            ok=True,
            latency_s=r.wall_s,
            # Goodput by construction: what the caller received is the returned
            # beam, so `output_tok_per_s` derived from this is the goodput
            # figure and stays comparable to a non-beam baseline's.
            output_tokens=len(r.returned_tokens),
            prompt_tokens=1,  # the fixed BOS seed; there is no prompt parameter
            output_token_sha256=r.digest,
            output_token_ids=r.returned_tokens,
        )
        for r in runs
    ]
    summary = summarize(
        mode="beam",
        engine="pie",
        model=model,
        results=results,
        wall_s=wall_s,
        config={
            "inferlet": INFERLET,
            "beam_width": width,
            "max_tokens": max_tokens,
            "reps": reps,
            "budget_matched": max_tokens == PRIMARY_MAX_TOKENS,
        },
    )
    delivered = sum(len(r.returned_tokens) for r in runs)
    summary = dataclasses.replace(
        summary,
        beam_width=width,
        # Exact, not estimated: a fixed-width beam with no early termination
        # advances exactly `width` hypotheses on each of `max_tokens` steps, so
        # the engine's token count is the delivered count times the width.
        engine_output_tok_per_s=(delivered * width / wall_s) if wall_s > 0 else 0.0,
        goodput_output_tok_per_s=(delivered / wall_s) if wall_s > 0 else 0.0,
        kv_cells_occupied_peak=max(r.kv_cells_occupied_peak for r in runs),
    )
    return summary, results


async def run(args: argparse.Namespace) -> int:
    from pie.config import (
        AuthConfig,
        Config,
        DriverConfig,
        ModelConfig,
        ServerConfig,
        TelemetryConfig,
    )
    from pie.server import Server

    wasm, manifest, pkg = inferlet_paths()
    model = resolve_local_model(args.model)
    devices = [d.strip() for d in args.device.split(",")]
    cfg = Config(
        server=ServerConfig(port=0),
        auth=AuthConfig(enabled=False),
        telemetry=TelemetryConfig(),
        model=ModelConfig(
            name="default",
            hf_repo=model,
            driver=DriverConfig(type=args.driver, device=devices),
        ),
    )

    points = [(w, PRIMARY_MAX_TOKENS) for w in args.widths] + list(EXTRA_POINTS)
    gates: list[dict[str, Any]] = []
    rows: list[tuple[BenchSummary, list[RequestResult]]] = []

    async with Server(cfg) as server:
        client = await server.connect()
        await client.install_program(wasm, manifest, force_overwrite=True)

        # ── Gate, for every point, before any timing ────────────────────────
        print("correctness gate: mask-on vs mask-off, by returned-beam digest")
        for width, max_tokens in points:
            g = await gate(client, pkg, width, max_tokens, args.timeout)
            gates.append(g)
            mark = {"diverged": "ok", "vacuous": "vacuous", "matched": "STRUCK"}[g["verdict"]]
            print(
                f"  width={width:<2} tokens={max_tokens:<3} {mark:<8} "
                f"on={g['mask_on_sha256'][:12]} off={g['mask_off_sha256'][:12]}"
            )
            if g["verdict"] == "matched":
                print(
                    "    struck: mask-on and mask-off produced identical tokens, "
                    "so attention never read the ancestry mask. Not timed."
                )

        admitted = [
            (g["width"], g["max_tokens"]) for g in gates if g["verdict"] != "matched"
        ]
        if not admitted:
            print("\nno configuration passed the gate; nothing to time")
            return 1

        # ── Timing, admitted configurations only ───────────────────────────
        print(f"\ntiming {len(admitted)} admitted configuration(s), reps={args.reps}")
        for width, max_tokens in admitted:
            summary, results = await measure(
                client, pkg, width, max_tokens, args.reps, args.timeout, model
            )
            rows.append((summary, results))
            print_summary(summary)

    report(gates, rows, args)
    return 0


def report(gates: list[dict[str, Any]], rows: list[tuple[BenchSummary, list[RequestResult]]],
           args: argparse.Namespace) -> None:
    matched = [r for r, _ in rows if r.config["budget_matched"]]
    unmatched = [r for r, _ in rows if not r.config["budget_matched"]]

    def table(title: str, summaries: list[BenchSummary]) -> None:
        if not summaries:
            return
        print(f"\n{title}")
        print(f"  {'width':>5}  {'tokens':>6}  {'engine tok/s':>12}  "
              f"{'goodput tok/s':>13}  {'kv cells':>8}  {'lat p50 ms':>10}")
        for s in summaries:
            print(
                f"  {s.beam_width:>5}  {s.config['max_tokens']:>6}  "
                f"{s.engine_output_tok_per_s:>12.1f}  {s.goodput_output_tok_per_s:>13.1f}  "
                f"{s.kv_cells_occupied_peak:>8}  "
                f"{(s.latency_p50_ms or 0.0):>10.1f}"
            )

    table(f"budget-matched at {PRIMARY_MAX_TOKENS} tokens", matched)
    for s in unmatched:
        table(
            f"NOT budget-matched to the table above "
            f"({s.config['max_tokens']} tokens, pool ceiling at this width)",
            [s],
        )

    if args.json_out:
        out = {
            "gate": gates,
            "rows": [
                {"summary": dataclasses.asdict(s),
                 "requests": [dataclasses.asdict(r) for r in rs]}
                for s, rs in rows
            ],
        }
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Beam-search benchmark (pie, Design B)")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--device", default="cuda:0", help="comma-separated")
    p.add_argument("--driver", default="cuda_native")
    p.add_argument("--reps", type=int, default=7,
                   help="timed repetitions per configuration, after a discarded warm-up")
    p.add_argument("--widths", type=int, nargs="+", default=list(PRIMARY_WIDTHS),
                   help=f"budget-matched widths, all run at {PRIMARY_MAX_TOKENS} tokens")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--json-out", default=None)
    return p


if __name__ == "__main__":
    sys.exit(asyncio.run(run(build_parser().parse_args())))
