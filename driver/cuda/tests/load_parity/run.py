"""Unified load-parity runner.

Composes synthetic checkpoints (spec.py — named real layouts or random ones),
materializes them with the real loader, and checks parity three ways:

  --mode differential : loader must agree with ITSELF across orthogonal paths
                        (reader on/off, fused/unfused FP8->MXFP4) — byte-identical.
  --mode absolute     : tp=1 materialized form vs the generic byte-reconstruction
                        oracle (oracle.py: direct / fusion / split, quant skipped).
  --mode tp           : tp=2 per-rank shards reassemble to the source (oracle).
  --mode all          : differential + absolute (+ tp when >=2 GPUs), the default.

Recipes: positional names pick named recipes (default: all). --random N appends N
random compositions (--seed makes them reproducible). Exit 0 = all checks pass.

  python run.py                       # all named, all modes
  python run.py --random 20 --seed 7  # 24 named + 20 random
  python run.py --mode tp glm_moe_dsa kimi_k2
"""

from __future__ import annotations

import argparse
import hashlib
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import oracle  # noqa: E402
import spec  # noqa: E402
from gen import write_checkpoint  # noqa: E402
from parse_cache import read_cache, read_safetensors  # noqa: E402

import numpy as np  # noqa: E402

def _find_pie_bin() -> Path:
    """Walk up to the repo root (the dir holding target/release/pie) rather than
    hardcoding a directory depth."""
    for d in Path(__file__).resolve().parents:
        cand = d / "target" / "release" / "pie"
        if cand.exists():
            return cand
    return Path(__file__).resolve().parents[4] / "target" / "release" / "pie"


PIE_BIN = _find_pie_bin()


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _toml(snap: Path, runtime_quant: str, port: int, devices: list[str], tp: int) -> str:
    rq = f'runtime_quant = "{runtime_quant}"\n' if runtime_quant else ""
    devs = ", ".join(f'"{d}"' for d in devices)
    return f"""[server]
host = "127.0.0.1"
port = {port}
verbose = true
[auth]
enabled = false
[runtime]
wasm_max_instances = 64
[[model]]
name = "default"
hf_repo = "{snap}"
[model.driver]
type = "cuda_native"
device = [{devs}]
tensor_parallel_size = {tp}
[model.driver.options]
gpu_mem_utilization = 0.30
{rq}ready_timeout_s = 120.0
[model.scheduler]
batch_policy = "adaptive"
"""


def materialize(snap: Path, runtime_quant: str, env_extra: dict, cache_dir: Path,
                tp: int = 1, timeout_s: int = 180) -> list[Path] | None:
    """Run a load until `tp` rank caches are written; return the cache paths."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    devices = [f"cuda:{i}" for i in range(tp)]
    port = _free_port()
    toml = cache_dir.parent / f"serve_{cache_dir.name}.toml"
    toml.write_text(_toml(snap, runtime_quant, port, devices, tp))
    env = dict(os.environ)
    env["PIE_CUDA_WEIGHT_CACHE_DIR"] = str(cache_dir)
    env.update(env_extra)
    proc = subprocess.Popen([str(PIE_BIN), "serve", "--config", str(toml)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    deadline, lines = time.time() + timeout_s, []
    try:
        assert proc.stdout is not None
        os.set_blocking(proc.stdout.fileno(), False)
        while time.time() < deadline and proc.poll() is None:
            # Completion = `tp` atomically-renamed .weights files exist (so a
            # file's presence means it is fully written). Check first, every
            # iteration, so we wait out a slower second rank under tp>1.
            if len(list(cache_dir.glob("*.weights"))) >= tp:
                break
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.02)
                continue
            s = line.decode("utf-8", "replace")
            lines.append(s)
            # Keep draining stdout the whole time: if we stop reading, a verbose
            # server can fill its stdout pipe, block on write(), and then never
            # finish a rank's cache write. Only stop early on a real hard error
            # (a benign line containing "error" must not abort the wait).
            if ("panic" in s.lower() or "fatal" in s.lower()
                    or "did not cover" in s or "CUDA error" in s):
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    files = sorted(cache_dir.glob("*.weights"))
    if len(files) < tp:
        sys.stderr.write("".join(lines[-12:]))
        return None
    return files[:tp]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- modes ----------------------------------------------------------------- #
def run_differential(snap: Path, r: spec.Recipe, work: Path) -> str:
    variants = {"base": {}, "reader_off": {"PIE_CUDA_WEIGHT_READER_THREADS": "0"}}
    if r.runtime_quant == "mxfp4":
        variants["unfused"] = {"PIE_CUDA_DISABLE_FUSED_TRANSCODE": "1"}
    hashes = {}
    for v, env in variants.items():
        files = materialize(snap, r.runtime_quant, env, work / f"c_{r.name}_{v}")
        hashes[v] = _sha(files[0]) if files else None
    if hashes["base"] is None:
        return f"FAIL  {r.name}: base load did not materialize"
    bad = [v for v, h in hashes.items() if h != hashes["base"]]
    if bad:
        return f"FAIL  {r.name}: differs across {bad}"
    return f"PASS  {r.name}: {len(hashes)} variants byte-identical ({hashes['base'][:12]})"


def run_absolute(snap: Path, r: spec.Recipe, work: Path) -> str:
    files = materialize(snap, r.runtime_quant, {}, work / f"abs_{r.name}", tp=1)
    if not files:
        return f"FAIL  {r.name}: tp=1 materialize failed"
    final = read_cache(str(files[0]))
    src = read_safetensors(str(snap / "model-00001-of-00001.safetensors"))
    res = oracle.classify(final, src, r.prefix)
    n_ok, kinds, fails = oracle.summarize(res)
    if fails:
        return f"FAIL  {r.name}: " + "; ".join(fails[:4])
    summary = ", ".join(f"{v}×{k}" for k, v in sorted(kinds.items()))
    return f"PASS  {r.name}: {n_ok} tensors verified ({summary})"


def run_tp(snap: Path, r: spec.Recipe, work: Path) -> str:
    if not r.tp_engine_ok:
        return f"SKIP  {r.name}: engine gates tp>1 (linear-attn hybrid)"
    files = materialize(snap, r.runtime_quant, {}, work / f"tp_{r.name}", tp=2)
    if not files:
        return f"FAIL  {r.name}: tp=2 materialize failed"
    a, b = read_cache(str(files[0])), read_cache(str(files[1]))
    src = read_safetensors(str(snap / "model-00001-of-00001.safetensors"))
    res, n_sharded = oracle.verify_tp(a, b, src, r.prefix)
    n_ok, kinds, fails = oracle.summarize(res)
    if fails:
        return f"FAIL  {r.name}: " + "; ".join(fails[:4])
    summary = ", ".join(f"{v}×{k}" for k, v in sorted(kinds.items()))
    warn = "  [WARN: nothing sharded]" if n_sharded == 0 else ""
    return f"PASS  {r.name}: {n_ok} tensors reassemble to source ({summary}){warn}"


def _gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        return sum(1 for ln in out.stdout.splitlines() if ln.startswith("GPU "))
    except Exception:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="named recipes (default: all)")
    ap.add_argument("--mode", choices=["differential", "absolute", "tp", "all"], default="all")
    ap.add_argument("--random", type=int, default=0, metavar="N", help="append N random recipes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not PIE_BIN.exists():
        sys.stderr.write(f"missing {PIE_BIN}; build with: cargo build -p pie-server "
                         "--release --no-default-features --features driver-cuda\n")
        return 2

    named = spec.named_recipes()
    recipes = [named[n] for n in (args.names or named)]
    if args.random:
        rng = np.random.default_rng(args.seed)
        recipes += [spec.random_recipe(rng, i) for i in range(args.random)]

    have_tp = _gpu_count() >= 2
    modes = ([args.mode] if args.mode != "all"
             else ["differential", "absolute"] + (["tp"] if have_tp else []))
    fns = {"differential": run_differential, "absolute": run_absolute, "tp": run_tp}

    failures = 0
    with tempfile.TemporaryDirectory(prefix="load-parity-") as td:
        work = Path(td)
        for r in recipes:
            snap = write_checkpoint(spec.build(r), work / f"ckpt_{r.name}", seed=args.seed + 1)
            for m in modes:
                line = fns[m](snap, r, work)
                print(f"[{m[:4]}] {line}", flush=True)
                if line.lstrip().startswith("FAIL"):
                    failures += 1
    print(f"\nload-parity: {failures} failure(s) across {len(recipes)} recipes × {len(modes)} mode(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
