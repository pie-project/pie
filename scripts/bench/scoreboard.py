#!/usr/bin/env python3
"""pie against vLLM at the fixed points, model by model, into one table.

For every model asked for, each engine is run at `offline` (prefill and the
three decode batches) and at `serve` (`vllm_baseline.py` for vLLM,
`pie_baseline.py` for pie), one engine at a time on a device it has to
itself: before each run the driver waits for the device to be idle
(`common.wait_for_idle_gpu`), and a run during which another process held the
device (`contended` in its result) is kept aside and run again, up to
`--attempts` times.

    /root/baselines/scoreboard/<model>/{vllm,pie}/{offline,serve}.json  one run's raw result
    /root/baselines/scoreboard/<model>/{vllm,pie}.json                  the two modes merged
    /root/baselines/scoreboard/scoreboard.md                            the table

    python scripts/bench/scoreboard.py --models qwen38-27b,gemma4-31b [--engines pie,vllm] [--modes offline,serve]
    python scripts/bench/scoreboard.py --table-only

Ratios are pie / vLLM in the direction where above 1 is pie ahead: throughput
over throughput, and vLLM's time over pie's for TPOT and TTFT.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from common import ROOT, wait_for_idle_gpu

HUB = Path(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")).expanduser()
OUT = Path("/root/baselines/scoreboard")
VLLM_PY = "/root/venvs/vllm/bin/python"
PIE_PY = os.environ.get("PIE_PY", "/root/venvs/piebench/bin/python")
PIE_HOME = os.environ.get("PIE_HOME", "/root/pie-homes/bench")


@dataclass
class Model:
    key: str
    repo: str
    # vLLM's precision and pie's, as the table states them.
    vllm_precision: str
    pie_precision: str | None
    # The pie deployment the artifact was imported as; None runs vLLM alone.
    pie_deployment: str | None
    vllm_flags: list[str] = field(default_factory=list)
    # `--engine-option KEY=VALUE`s pie is served with.
    pie_options: list[str] = field(default_factory=list)
    # `[model] precision`, for a deployment that shares another's artifact.
    pie_precision_flag: str | None = None
    # The device memory share pie is served at, when its vLLM row ran at
    # another `gpu_memory_utilization` than the driver's 0.90 (both rows at
    # one budget, so the KV pages each fits are compared like for like).
    pie_mem_share: float | None = None
    # `[model] kv`: the cache rows' dtype over the artifact's own, so pie
    # serves the KV format vLLM's row does (DeepSeek-V4: fp8).
    pie_kv: str | None = None
    # The artifact, when it was imported into another home under another
    # deployment's name.
    pie_artifact: str | None = None
    # A vLLM result measured before this driver existed, read as it stands.
    vllm_dir: str | None = None
    note: str = ""

    def snapshot(self) -> str:
        repo = HUB / f"models--{self.repo.replace('/', '--')}"
        return str(repo / "snapshots" / (repo / "refs" / "main").read_text().strip())

    def artifact(self) -> str:
        if self.pie_artifact:
            return self.pie_artifact
        rev = Path(self.snapshot()).name
        return f"{PIE_HOME}/artifacts/{rev}/{rev}.{self.pie_deployment}.cuda.zt"


# A hybrid trunk holds two recurrent-state slots a sequence. Before cb4c39af
# pie's default 256 slots seated 128, so its serving burst ran at most 128
# wide where vLLM admits all 256; the default is now two slots a lane.
SEATS = ("config: pie before cb4c39af seated 128 sequences (max_state_slots 256 / 2 slots each), "
         "since then all 256 as vLLM")

MODELS = {m.key: m for m in [
    Model("qwen38-27b", "Qwen/Qwen3.8-27B", "bf16 / kv bf16", "bf16 / kv bf16", "qwen38-27b-bf16-kv-bf16",
          note=SEATS),
    Model("gemma4-31b", "google/gemma-4-31B-it", "bf16 / kv bf16", "bf16 / kv bf16", "gemma4-31b-bf16-kv-bf16"),
    # vLLM 0.31's default mxfp4 MoE on this B200 (Triton) fails its own
    # warmup with CUBLAS_STATUS_EXECUTION_FAILED, and both FlashInfer kinds
    # refuse the device; Marlin (w4a16, as pie's) is what serves.
    Model("gptoss-120b", "openai/gpt-oss-120b", "experts mxfp4 (w4a16), rest bf16 / kv bf16",
          "experts mxfp4 (w4a16), rest bf16 / kv bf16", "gptoss-120b-bf16-mxfp4-kv-bf16",
          vllm_flags=["--extra", "--moe-backend marlin"],
          note="vLLM: default Triton mxfp4 MoE crashes at warmup on this build; --moe-backend marlin"),
    Model("gptoss-20b", "openai/gpt-oss-20b", "experts mxfp4 (w4a16), rest bf16 / kv bf16",
          "experts mxfp4 (w4a16), rest bf16 / kv bf16", "gptoss-20b-bf16-mxfp4-kv-bf16",
          vllm_flags=["--extra", "--moe-backend marlin"],
          note="vLLM: --moe-backend marlin, as gpt-oss-120b"),
    Model("gemma4-e4b", "google/gemma-4-E4B-it", "bf16 / kv bf16", "bf16 / kv bf16", "gemma4-e4b-bf16-kv-bf16"),
    # Before the tcgen05 grouped select, a fire past 8191 tokens routed more
    # rows than the bf16 select GEMV takes, and the before runs held fires to
    # 4096 tokens; the select has no such limit now.
    Model("qwen36-35b-a3b", "Qwen/Qwen3.6-35B-A3B", "bf16 / kv bf16", "bf16 / kv bf16",
          "qwen36-35b-a3b-bf16-kv-bf16",
          note="pie before: max_forward_tokens=4096 (wider fires refused: bf16 MoE select route run > 65535); "
               + SEATS),
    Model("qwen38-flash-next-nvfp4", "nvidia/Qwen3.8-Flash-Next-NVFP4",
          "experts nvfp4 W4A4 (CUTLASS) / kv bf16", "experts nvfp4 W4A4 / kv bf16",
          "qwen38-flash-next-bf16-nvfp4a4-kv-bf16", vllm_dir="/root/baselines/qwen38-flash-next-nvfp4",
          note=SEATS),
    Model("qwen38-flash-next-nvfp4-w4a16", "nvidia/Qwen3.8-Flash-Next-NVFP4",
          "experts nvfp4 W4A16 (Marlin) / kv bf16", "experts nvfp4 W4A16 / kv bf16",
          "qwen38-flash-next-bf16-nvfp4-kv-bf16", vllm_flags=["--extra", "--moe-backend marlin"],
          note="vLLM serves this checkpoint W4A4 by default; this pair holds both to W4A16 (Marlin); " + SEATS),
    Model("deepseek-v4-flash", "deepseek-ai/DeepSeek-V4-Flash", "fp8 dense + mxfp4 experts / kv fp8_ds_mla",
          "fp8b128 dense W8A8 + mxfp4 experts W4A8 / kv fp8 (DeepSeek's fp8 rows)",
          "dsv4-flash-fp8b128a8-mxfp4a8-kv-bf16",
          vllm_dir="/root/baselines/deepseek-v4-flash", pie_mem_share=0.95, pie_kv="e4m3",
          note="kv fp8 on both: vLLM fp8_ds_mla, pie the same layout ([model] kv = \"e4m3\" over the kv-bf16 "
               "artifact); pie's artifact is the #903 import of the official release; "
               "both rows at a 0.95 memory share (vLLM's run used gpu_memory_utilization 0.95; the weights "
               "take ~157 GB of 180, and at 0.85-0.90 pie's KV pool starves bs64)"),
]}


def log(*a, **kw) -> None:
    print(time.strftime("[%H:%M:%S]"), *a, **kw, flush=True)


def run_once(model: Model, engine: str, mode: str, out: Path) -> dict | None:
    out.mkdir(parents=True, exist_ok=True)
    if engine == "vllm":
        cmd = [VLLM_PY, "vllm_baseline.py", mode, "--model", model.snapshot(), "--out", str(out)]
        cmd += model.vllm_flags
    else:
        cmd = [PIE_PY, "pie_baseline.py", mode, "--model", model.artifact(), "--hf", model.snapshot(),
               "--out", str(out)]
        for opt in model.pie_options:
            cmd += ["--engine-option", opt]
        if model.pie_precision_flag:
            cmd += ["--precision", model.pie_precision_flag]
        if model.pie_mem_share is not None:
            cmd += ["--gpu-mem-util", str(model.pie_mem_share)]
        if model.pie_kv:
            cmd += ["--kv", model.pie_kv]
    env = {**os.environ, "PIE_HOME": PIE_HOME}
    log("+", " ".join(cmd))
    with open(out / f"{mode}_driver.log", "w") as f:
        rc = subprocess.run(cmd, cwd=Path(__file__).parent, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    path = out / f"{mode}.json"
    if rc != 0 or not path.exists():
        log(f"  {engine} {mode} failed (rc={rc}); see {out}/{mode}_driver.log")
        return None
    return json.loads(path.read_text())


def measure(model: Model, engine: str, mode: str, attempts: int) -> dict | None:
    out = OUT / model.key / engine
    for attempt in range(1, attempts + 1):
        waited = wait_for_idle_gpu(log)
        if waited > 1:
            log(f"  device idle after {waited:.0f}s")
        res = run_once(model, engine, mode, out)
        if res is not None and not res.get("contended"):
            return res
        # Keep the contended (or failed: a neighbour taking memory between
        # the idle check and the load fails it) run aside and go again.
        aside = out / f"{'contended' if res else 'failed'}_{mode}_{attempt}"
        aside.mkdir(exist_ok=True)
        for f in glob.glob(str(out / f"*{mode}*")):
            if os.path.isfile(f):
                shutil.move(f, aside / os.path.basename(f))
        log(f"  {engine} {mode} {'contended by pids ' + str(res.get('foreign_pids')) if res else 'failed'};"
            f" kept in {aside}, again")
    log(f"  {engine} {mode}: no attempt was clean")
    return None


def contention(path: Path) -> int:
    """Samples of a run's memory log, about a second each, that show a
    process besides the engine's own."""
    doc = json.loads(path.read_text())
    if "foreign_samples" in doc:
        return doc["foreign_samples"] if doc.get("contended") else 0
    csv = path.parent / f"gpu_mem_{path.stem}.csv"
    if not doc.get("contended") or not csv.exists():
        return 0
    rows = [line.split(",") for line in csv.read_text().splitlines()[1:]]
    return sum(1 for r in rows if len(r) > 3 and len(r[3].split()) > 1)


def merged(model: Model, engine: str) -> dict | None:
    """Each mode's clean run, or, when every attempt was contended, the
    least contended one, marked so. `engine` may name a kept-aside run
    (`pie-before`)."""
    return _merged(model, engine)


def _merged(model: Model, engine: str) -> dict | None:
    """Each mode's clean run, or, when every attempt was contended, the
    least contended one, marked so."""
    base = Path(model.vllm_dir) if (engine == "vllm" and model.vllm_dir) else OUT / model.key / engine
    doc = {}
    for mode in ("offline", "serve"):
        hits = sorted(base.glob(f"**/{mode}.json"),
                      key=lambda p: (contention(p), "contended" in str(p), len(p.parts)))
        if hits:
            doc[mode] = json.loads(hits[0].read_text())
            doc[mode]["path"] = str(hits[0])
            doc[mode]["contended_samples"] = contention(hits[0])
    return doc or None


def fmt(x, unit="", nd=0) -> str:
    return "–" if x is None else f"{x:,.{nd}f}{unit}"


def ratio(a, b) -> str:
    return "–" if a is None or b is None or b == 0 else f"**{a / b:.2f}**"


def points(doc: dict | None) -> dict:
    """The table's values out of one engine's merged result."""
    p: dict = {}
    if not doc:
        return p
    off = doc.get("offline") or {}
    if "prefill" in off:
        p["prefill_tok_s"] = off["prefill"]["prefill_tok_s"]
        p["prefill_s"] = off["prefill"]["median_s"]
    for d in off.get("decode", []):
        p[f"tpot{d['batch']}"] = d["tpot_ms"]
        p[f"dec{d['batch']}"] = d["decode_tok_s"]
    srv = (doc.get("serve") or {}).get("result") or {}
    for k in ("output_throughput", "total_token_throughput", "duration", "median_ttft_ms", "median_tpot_ms",
              "mean_ttft_ms", "mean_tpot_ms"):
        if k in srv:
            p[k] = srv[k]
    return p


ROWS = [
    # label, key, unit, digits, higher is better
    ("prefill 32x4096 (tok/s)", "prefill_tok_s", "", 0, True),
    ("decode bs1 TPOT (ms)", "tpot1", "", 2, False),
    ("decode bs8 TPOT (ms)", "tpot8", "", 2, False),
    ("decode bs64 TPOT (ms)", "tpot64", "", 2, False),
    ("decode bs64 (tok/s)", "dec64", "", 0, True),
    ("serve output (tok/s)", "output_throughput", "", 0, True),
    ("serve total (tok/s)", "total_token_throughput", "", 0, True),
    ("serve TTFT median (ms)", "median_ttft_ms", "", 0, False),
    ("serve TPOT median (ms)", "median_tpot_ms", "", 1, False),
]


def table() -> str:
    lines = [
        "# pie vs vLLM scoreboard (one B200, TP=1)",
        "",
        "Conditions: random token-id prompts (seed 0), greedy, ignore_eos, no prefix-cache hits; offline points 1 warmup",
        "+ 3 timed, median; decode TPOT = (t[256 out] - t[1 out]) / 255 at 512 in; serve = 256 requests 1024 in / 256 out",
        "offered at once (vLLM: `vllm serve` + `vllm bench serve --dataset-name random`, OpenAI HTTP; pie: `pie serve` +",
        "text-completion-bench processes over pie's own protocol, so vLLM's HTTP/SSE/detokenize cost is in its numbers",
        "and not in pie's). max_model_len 8192, vLLM gpu_memory_utilization 0.90, pie device_memory_share 0.90",
        "(vLLM 0.31.0; pie at the commit in each result's argv). Ratio = pie/vLLM, > 1 means pie is ahead",
        "(throughput: pie/vLLM; latency: vLLM/pie).",
        "",
    ]
    for model in MODELS.values():
        vdoc, pdoc = merged(model, "vllm"), merged(model, "pie")
        if not vdoc and not pdoc:
            continue
        v, p = points(vdoc), points(pdoc)
        before = points(merged(model, "pie-before"))
        # A parenthesis names the kernel, not the precision.
        same = model.pie_precision and (re.sub(r" \(.*?\)", "", model.pie_precision)
                                        == re.sub(r" \(.*?\)", "", model.vllm_precision))
        lines += [f"## {model.repo} ({model.key})", "",
                  f"precision — vLLM: {model.vllm_precision}; pie: {model.pie_precision or '–'}"
                  + (" (match)" if same else
                     " (MISMATCH)" if model.pie_precision else ""),
                  *( [f"note: {model.note}"] if model.note else []), "",
                  "| point | vLLM | pie | pie/vLLM | pie commit | pie before | before/vLLM |",
                  "|---|---:|---:|---:|---|---:|---:|"]
        for engine, doc in (("vLLM", vdoc), ("pie", pdoc)):
            for mode, run in (doc or {}).items():
                if run.get("contended_samples"):
                    lines.insert(len(lines) - 3, f"contended: {engine} {mode} saw a neighbour process in "
                                 f"{run['contended_samples']} one-second samples (every attempt was contended)")
        for label, key, unit, nd, higher in ROWS:
            a, b = p.get(key), v.get(key)
            r = ratio(a, b) if higher else ratio(b, a)
            run = (pdoc or {}).get("serve" if label.startswith("serve") else "offline") or {}
            commit = run.get("pie_commit", "011c5aba" if run else "–")
            c = before.get(key)
            rb = ratio(c, b) if higher else ratio(b, c)
            lines.append(f"| {label} | {fmt(b, unit, nd)} | {fmt(a, unit, nd)} | {r} | {commit} "
                         f"| {fmt(c, unit, nd)} | {rb} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(k for k, m in MODELS.items() if m.pie_deployment))
    ap.add_argument("--engines", default="vllm,pie")
    ap.add_argument("--modes", default="offline,serve")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--table-only", action="store_true")
    args = ap.parse_args()

    if not args.table_only:
        for key in args.models.split(","):
            model = MODELS[key]
            for engine in args.engines.split(","):
                if engine == "pie" and not model.pie_deployment:
                    continue
                if engine == "vllm" and model.vllm_dir:
                    continue
                for mode in args.modes.split(","):
                    measure(model, engine, mode, args.attempts)
            for engine in ("vllm", "pie"):
                doc = merged(model, engine)
                if doc:
                    (OUT / model.key).mkdir(parents=True, exist_ok=True)
                    (OUT / model.key / f"{engine}.json").write_text(json.dumps(doc, indent=2))
    md = table()
    (OUT / "scoreboard.md").write_text(md)
    print(md)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts" / "bench"))
    main()
