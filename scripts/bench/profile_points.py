#!/usr/bin/env python3
"""Where a fixed point's device time goes, pie and vLLM alike, from nsys.

`run` drives one engine through one point under `nsys profile`, the measured
bursts set apart from the warmup and from each other by idle gaps:

    prefill   warmup, gap, 32 x 4096 -> 1
    decodeB   warmup, gap, B x 512 -> 1, gap, B x 512 -> 256

`report` reads the exported sqlite back, cuts the kernel timeline at those
gaps, and states per kernel (and per op class) the device time of one
prefill, or of one decode step as (K[256] - K[1]) / 255; the share of the
window the device was busy; and how the window's kernels were launched (in a
CUDA graph or one by one).

    nsys profile -t cuda,nvtx -s none --cpuctxsw=none -o OUT \\
        python profile_points.py run pie  --point decode64 --model ART.zt --hf SNAP
    nsys profile -t cuda,nvtx -s none --cpuctxsw=none -o OUT \\
        python profile_points.py run vllm --point decode64 --model SNAP
    nsys export --type sqlite -o OUT.sqlite OUT.nsys-rep
    python profile_points.py report OUT.sqlite --point decode64 [--json OUT.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import shlex
import sqlite3
import sys
import time
from collections import defaultdict

from common import FIXED_DECODE_INPUT, FIXED_DECODE_OUTPUT, FIXED_PREFILL, fixed_prompt_ids

GAP_S = 3.0
SPLIT_S = 1.5


def shape(point: str) -> tuple[int, int, list[int]]:
    """(batch, input_len, the measured bursts' output lengths)."""
    if point == "prefill":
        return FIXED_PREFILL["batch"], FIXED_PREFILL["input_len"], [1]
    b = int(point.removeprefix("decode"))
    return b, FIXED_DECODE_INPUT, [1, FIXED_DECODE_OUTPUT]


def run_pie(args: argparse.Namespace) -> None:
    import pie_baseline
    from common import tokenizer_vocab_size

    batch, inp, outs = shape(args.point)
    vocab, rng = tokenizer_vocab_size(args.hf), random.Random(0)

    async def go() -> None:
        async with pie_baseline.pie_session(args, os.path.join(args.out, f"{args.point}_engine.log")) as (burst, _):
            await burst(fixed_prompt_ids(rng, vocab, batch, inp), outs[-1])
            for out in outs:
                await asyncio.sleep(GAP_S)
                s, _ = await burst(fixed_prompt_ids(rng, vocab, batch, inp), out)
                print(f"[profile] pie {args.point} out={out}: {s:.3f}s", flush=True)
            await asyncio.sleep(GAP_S)

    asyncio.run(go())


def run_vllm(args: argparse.Namespace) -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    batch, inp, outs = shape(args.point)
    p = EngineArgs.add_cli_args(FlexibleArgumentParser())
    eargs = EngineArgs.from_cli_args(p.parse_args([
        "--model", args.model, "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_mem_util), "--seed", "0"] + shlex.split(args.extra)))
    llm = LLM(**vars(eargs))
    vocab, rng = llm.get_tokenizer().vocab_size, random.Random(0)

    def go(out: int) -> float:
        sp = SamplingParams(temperature=0.0, max_tokens=out, min_tokens=out, ignore_eos=True)
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids in fixed_prompt_ids(rng, vocab, batch, inp)]
        t0 = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        return time.perf_counter() - t0

    go(outs[-1])
    for out in outs:
        time.sleep(GAP_S)
        print(f"[profile] vllm {args.point} out={out}: {go(out):.3f}s", flush=True)
    time.sleep(GAP_S)


CLASSES = [
    # Order matters: the first pattern a kernel name matches is its class.
    ("attention", r"fmha|xqa|batchdecode|batchprefill"),
    ("gdn/ssm", r"gdn|ssm|conv1d|chunk_|delta|recurrent|fla_|gated_delta|l2norm|solve_tril|kkt"),
    ("moe", r"moe|expert|fused_moe|topk|router|grouped|marlin|swiglu_oai|mxfp4|bmm_.*E2m1|routing"),
    ("attention", r"attn|attention|flash|fmha|xqa|mla|paged|decode_kernel|prefill_kernel|kv_cache|reshape_and_cache|rope|rotary"),
    ("gemm", r"gemm|matmul|cutlass|nvjet|sm100_|sm90_|cublas|linear|skinny|tcgen05|xmma|splitk|wgmma|hmma|mma"),
    ("norm", r"norm"),
    ("sampling", r"sampl|argmax|softmax|logit|gumbel|top_p|topp"),
    ("elementwise/copy", r"elementwise|copy|cast|convert|fill|add|mul|act|silu|gelu|scatter|gather|index|embed|cat|memcpy|reduce"),
]


def classify(name: str) -> str:
    low = name.lower()
    for cls, pat in CLASSES:
        if re.search(pat, low):
            return cls
    return "other"


def clusters(rows: list[tuple[int, int, int]], split_ns: int) -> list[list[tuple[int, int, int]]]:
    out: list[list[tuple[int, int, int]]] = []
    end = None
    for r in rows:
        if end is None or r[0] - end > split_ns:
            out.append([])
        out[-1].append(r)
        end = r[1] if end is None else max(end, r[1])
    return out


def busy_ns(rows) -> int:
    total, cur_s, cur_e = 0, None, None
    for s, e, *_ in sorted(rows):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + ((cur_e - cur_s) if cur_e is not None else 0)


def report(args: argparse.Namespace) -> dict:
    db = sqlite3.connect(args.sqlite)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    cols = {r[1] for r in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    graph = "graphNodeId" if "graphNodeId" in cols else "NULL"
    kern = [(s, e, n, g) for s, e, n, g in db.execute(
        f"SELECT start, end, COALESCE(demangledName, shortName), {graph} FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")]
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables and not any(g for *_, g in kern):
        # Traced a graph at a time (`--cuda-graph-trace=graph`): a replay is
        # one interval whose kernels the trace does not name.
        names[-1] = "[a CUDA graph replay, kernels unnamed]"
        kern = sorted(kern + [(s, e, -1, 1) for s, e in db.execute(
            "SELECT start, end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE")])
    api = list(db.execute("SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start"))
    groups = clusters(kern, int(SPLIT_S * 1e9))
    _, _, outs = shape(args.point)
    measured = groups[-len(outs):]

    def per_kernel(rows) -> dict[str, float]:
        acc: dict[str, float] = defaultdict(float)
        for s, e, n, _ in rows:
            acc[names.get(n, str(n))] += (e - s) / 1e3
        return acc

    def launches(rows) -> dict[str, int]:
        lo, hi = rows[0][0], rows[-1][1]
        acc: dict[str, int] = defaultdict(int)
        for s, _, n in api:
            if lo - 50_000_000 <= s <= hi:
                nm = names.get(n, "")
                if "aunch" in nm:
                    acc[nm.split("_v")[0]] += 1
        return dict(acc)

    if len(outs) == 1:
        rows = measured[0]
        k = per_kernel(rows)
        scale, label = 1.0, "one prefill (32 x 4096)"
        window = (rows[-1][1] - rows[0][0]) / 1e3
        busy = busy_ns(rows) / 1e3
        graphed = sum((e - s) for s, e, _, g in rows if g) / 1e3
        launch = launches(rows)
    else:
        k1, k256 = per_kernel(measured[0]), per_kernel(measured[1])
        scale, label = 1.0 / (outs[1] - 1), f"one decode step at {args.point}"
        k = {n: k256.get(n, 0.0) - k1.get(n, 0.0) for n in set(k1) | set(k256)}
        rows = measured[1]
        w1 = (measured[0][-1][1] - measured[0][0][0]) / 1e3
        window = ((rows[-1][1] - rows[0][0]) / 1e3 - w1)
        busy = (busy_ns(rows) - busy_ns(measured[0])) / 1e3
        graphed = (sum((e - s) for s, e, _, g in rows if g) - sum((e - s) for s, e, _, g in measured[0] if g)) / 1e3
        l1, l256 = launches(measured[0]), launches(rows)
        launch = {n: (l256.get(n, 0) - l1.get(n, 0)) / (outs[1] - 1) for n in set(l1) | set(l256)}
    total = sum(k.values())
    by_class: dict[str, float] = defaultdict(float)
    for n, us in k.items():
        by_class[classify(n)] += us
    top = sorted(k.items(), key=lambda kv: -kv[1])[: args.top]
    res = dict(
        sqlite=args.sqlite, point=args.point, unit=label,
        kernel_us=total * scale, window_us=window * scale, busy_us=busy * scale,
        idle_us=(window - busy) * scale, graphed_kernel_us=graphed * scale,
        launches=launch if len(outs) == 1 else {n: round(v, 1) for n, v in launch.items()},
        by_class={c: us * scale for c, us in sorted(by_class.items(), key=lambda kv: -kv[1])},
        top_kernels=[dict(name=n[:160], us=us * scale, cls=classify(n)) for n, us in top],
        clusters=[dict(kernels=len(g), span_ms=(g[-1][1] - g[0][0]) / 1e6) for g in groups],
    )
    print(f"{args.sqlite}: {label}")
    print(f"  window {res['window_us']:.0f} us, device busy {res['busy_us']:.0f} us, idle {res['idle_us']:.0f} us,"
          f" kernel sum {res['kernel_us']:.0f} us, in graphs {res['graphed_kernel_us']:.0f} us")
    print(f"  launches: {res['launches']}")
    for c, us in res["by_class"].items():
        print(f"  {c:<18}{us:10.1f} us  {100 * us / max(res['kernel_us'], 1e-9):5.1f}%")
    for t in res["top_kernels"]:
        print(f"    {t['us']:9.1f} us  [{t['cls']}] {t['name'][:120]}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("engine", choices=["pie", "vllm"])
    r.add_argument("--point", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--hf", default=None)
    r.add_argument("--out", default=".")
    r.add_argument("--max-model-len", type=int, default=8192)
    r.add_argument("--gpu-mem-util", type=float, default=0.90)
    r.add_argument("--request-timeout", type=float, default=1800)
    r.add_argument("--startup-timeout", type=float, default=3600)
    from common import ROOT
    r.add_argument("--inferlet-dir", default=str(ROOT / "examples" / "text-completion-bench"))
    r.add_argument("--pie-bin", default=str(ROOT / "target" / "release" / "pie"))
    r.add_argument("--engine-option", action="append", default=[])
    r.add_argument("--precision", default=None, help="pie: [model] precision, e.g. bf16+nvfp4a4")
    r.add_argument("--kv", default=None, help="pie: [model] kv, e.g. e4m3 (DeepSeek-V4's fp8 rows)")
    r.add_argument("--extra", default="", help="further vLLM engine flags, one shell-quoted string")
    q = sub.add_parser("report")
    q.add_argument("sqlite")
    q.add_argument("--point", required=True)
    q.add_argument("--top", type=int, default=25)
    q.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.cmd == "report":
        report(args)
    elif args.engine == "pie":
        os.makedirs(args.out, exist_ok=True)
        run_pie(args)
    else:
        run_vllm(args)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
