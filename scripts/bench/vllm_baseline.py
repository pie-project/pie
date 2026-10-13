#!/usr/bin/env python3
"""A fixed vLLM baseline: one model load, the points every engine is held to.

    offline   prefill: 32 prompts x 4096 tokens, 1 output token, one batch
              decode:  batch 1, 8, 64; 512-token prompts, 256 output tokens,
                       timed once with 256 outputs and once with 1, so the
                       per-step decode time is (t256 - t1) / 255
    serve     `vllm serve` + `vllm bench serve`: the random dataset, 256
              prompts, 1024 in / 256 out, every request offered at once

Prompts are random token ids (seeded) and `ignore_eos` holds every output to
its length, as `vllm bench latency` does. Each point is warmed once and timed
`--iters` times; the median is reported. A sampler records the device's
memory in use (nvidia-smi) for the whole run; the engine log is kept so the
kernels vLLM picked (MoE backend, attention backend) can be read back.

    python scripts/bench/vllm_baseline.py offline --model M --out DIR [vLLM flags]
    python scripts/bench/vllm_baseline.py serve   --model M --out DIR [vLLM flags]

Everything lands in DIR: `offline.json` / `serve.json`, the raw engine logs,
and `gpu_mem.csv`. Run with the vLLM venv's interpreter.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

PREFILL = dict(batch=32, input_len=4096, output_len=1)
DECODE_BATCHES = (1, 8, 64)
DECODE_INPUT, DECODE_OUTPUT = 512, 256
SERVE = dict(num_prompts=256, input_len=1024, output_len=256)


class MemSampler:
    """nvidia-smi memory.used and utilization every `period` seconds, into
    `path`, with every compute process's pid:MiB, so a neighbour sharing the
    device during a measurement shows."""

    def __init__(self, path: str, period: float = 1.0) -> None:
        self.path, self.period = path, period
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        with open(self.path, "w") as f:
            f.write("t,memory_used_mib,util_pct,processes\n")
            t0 = time.time()
            while not self._stop.is_set():
                try:
                    used, util = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=10).stdout.split(",")[:2]
                    procs = subprocess.run(
                        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=10).stdout.split("\n")
                    procs = " ".join(p.replace(", ", ":") for p in procs if p.strip())
                    used = int(used)
                    self.peak = max(self.peak, used)
                    f.write(f"{time.time() - t0:.1f},{used},{util.strip()},{procs}\n")
                    f.flush()
                except Exception:
                    pass
                self._stop.wait(self.period)

    def __enter__(self) -> "MemSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


def engine_args(args: argparse.Namespace) -> list[str]:
    flags = [
        "--tensor-parallel-size", str(args.tp),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--seed", "0",
    ]
    if args.max_num_seqs:
        flags += ["--max-num-seqs", str(args.max_num_seqs)]
    if args.trust_remote_code:
        flags.append("--trust-remote-code")
    return flags + shlex.split(args.extra)


def offline(args: argparse.Namespace) -> dict:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    p = EngineArgs.add_cli_args(FlexibleArgumentParser())
    eargs = EngineArgs.from_cli_args(p.parse_args(["--model", args.model] + engine_args(args)))
    llm = LLM(**vars(eargs))
    vocab = llm.get_tokenizer().vocab_size
    rng = random.Random(0)

    def prompts(batch: int, input_len: int) -> list:
        return [TokensPrompt(prompt_token_ids=[rng.randrange(100, vocab - 100) for _ in range(input_len)])
                for _ in range(batch)]

    def timed(batch: int, input_len: int, output_len: int) -> dict:
        sp = SamplingParams(temperature=0.0, max_tokens=output_len, min_tokens=output_len, ignore_eos=True)
        llm.generate(prompts(batch, input_len), sp, use_tqdm=False)
        times = []
        for _ in range(args.iters):
            batch_prompts = prompts(batch, input_len)  # fresh ids: no prefix-cache hits
            t0 = time.perf_counter()
            outs = llm.generate(batch_prompts, sp, use_tqdm=False)
            times.append(time.perf_counter() - t0)
            assert all(len(o.outputs[0].token_ids) == output_len for o in outs)
        return dict(batch=batch, input_len=input_len, output_len=output_len,
                    seconds=times, median_s=statistics.median(times))

    res = {}
    pf = timed(**PREFILL)
    pf["prefill_tok_s"] = PREFILL["batch"] * PREFILL["input_len"] / pf["median_s"]
    res["prefill"] = pf
    res["decode"] = []
    for b in DECODE_BATCHES:
        full = timed(b, DECODE_INPUT, DECODE_OUTPUT)
        first = timed(b, DECODE_INPUT, 1)
        step = (full["median_s"] - first["median_s"]) / (DECODE_OUTPUT - 1)
        res["decode"].append(dict(
            batch=b, e2e=full, prefill_only=first,
            tpot_ms=step * 1e3,
            decode_tok_s=b / step,
            e2e_output_tok_s=b * DECODE_OUTPUT / full["median_s"],
        ))
    return res


def serve(args: argparse.Namespace) -> dict:
    vllm = os.path.join(os.path.dirname(sys.executable), "vllm")
    port = str(args.port)
    cmd = [vllm, "serve", args.model, "--port", port] + engine_args(args)
    log = open(os.path.join(args.out, "serve_engine.log"), "w")
    print("+", shlex.join(cmd), flush=True)
    server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.time() + args.startup_timeout
        while True:
            if server.poll() is not None:
                raise SystemExit(f"vllm serve exited with {server.returncode}; see serve_engine.log")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
                break
            except Exception:
                if time.time() > deadline:
                    raise SystemExit("vllm serve did not come up")
                time.sleep(5)
        bench = [
            vllm, "bench", "serve", "--backend", "vllm", "--model", args.model,
            "--port", port, "--dataset-name", "random",
            "--random-input-len", str(SERVE["input_len"]),
            "--random-output-len", str(SERVE["output_len"]),
            "--num-prompts", str(SERVE["num_prompts"]),
            "--request-rate", "inf", "--ignore-eos", "--seed", "0",
            "--save-result", "--result-dir", args.out, "--result-filename", "serve_bench.json",
        ]
        print("+", shlex.join(bench), flush=True)
        with open(os.path.join(args.out, "serve_bench.log"), "w") as blog:
            subprocess.run(bench, stdout=blog, stderr=subprocess.STDOUT, check=True)
        with open(os.path.join(args.out, "serve_bench.json")) as f:
            return dict(engine_cmd=shlex.join(cmd), bench_cmd=shlex.join(bench), result=json.load(f))
    finally:
        os.killpg(server.pid, signal.SIGINT)
        try:
            server.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["offline", "serve"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--startup-timeout", type=float, default=3600)
    ap.add_argument("--extra", default="", help="further vLLM engine flags, one shell-quoted string")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with MemSampler(os.path.join(args.out, f"gpu_mem_{args.mode}.csv")) as mem:
        t0 = time.time()
        res = offline(args) if args.mode == "offline" else serve(args)
    res.update(mode=args.mode, model=args.model, argv=shlex.join(sys.argv),
               engine_flags=shlex.join(engine_args(args)), wall_s=time.time() - t0,
               peak_gpu_mem_mib=mem.peak)
    path = os.path.join(args.out, f"{args.mode}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k not in ("result",)}, indent=2)[:4000])
    print("wrote", path)


if __name__ == "__main__":
    main()
