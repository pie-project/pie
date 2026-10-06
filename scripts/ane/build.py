#!/usr/bin/env python3
"""Build the Neural Engine half of each dense MLP for pie's Metal engine.

pie splits a dense SwiGLU MLP along its intermediate axis on long prefills:
the GPU keeps columns [0, inter - ane) and the Neural Engine runs the last
`ane` columns as a CoreML program, both at once. This script writes one
multifunction CoreML model per layer (one function per row bucket, weights
shared) plus `meta.json`; pie picks the build up on its own.

The Neural Engine half runs W8A8: int8 weights and int8 activations, with
both activations rotated by a block Hadamard transform first so outliers
spread out and one static scale per tensor holds. The scales are calibrated
on the model's own activations. Without `--ane`, the split is chosen for
this Mac by timing the GPU and the Neural Engine on one layer.

    python scripts/ane/build.py mlx-community/Qwen3.8-27B-4bit

Needs `coremltools`, `mlx` and `mlx-lm`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import coremltools as ct
import coremltools.optimize.coreml as cto
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

BLOCK = 128  # Hadamard block; hidden and the Neural Engine's columns are multiples
STEP = 256  # the split moves in whole quant groups on both sides

CALIBRATION = [
    "The history of the printing press begins long before Gutenberg. In the ninth century, Chinese "
    "craftsmen carved whole pages into wooden blocks, inked them, and pressed paper against the wood. "
    "Movable type made of fired clay followed, and later of bronze in Korea, where a Buddhist text was "
    "printed in 1377. What changed in fifteenth century Mainz was an alloy that cast cleanly, an oil based "
    "ink that clung to metal, and a screw press borrowed from the wine makers of the Rhine valley.",
    "def merge_intervals(intervals):\n    intervals.sort(key=lambda pair: pair[0])\n    merged = []\n"
    "    for start, end in intervals:\n        if merged and start <= merged[-1][1]:\n"
    "            merged[-1][1] = max(merged[-1][1], end)\n        else:\n            merged.append([start, end])\n"
    "    return merged\n\nclass LRUCache:\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n"
    "        self.items = {}\n",
    "User: My tomato plants have yellow leaves at the bottom and brown spots spreading upward. "
    "I water them every evening. What is going on?\nAssistant: Yellowing lower leaves with brown spots "
    "that climb the plant usually point to early blight, a fungal disease that thrives when leaves stay wet. "
    "Watering in the evening keeps the foliage damp overnight, so move watering to the morning and water at the base.",
    "Theorem. Every bounded monotone sequence of real numbers converges. Proof sketch: let (a_n) be "
    "increasing and bounded above, and let L be the supremum of its terms, which exists by completeness. "
    "For any epsilon > 0, L - epsilon is not an upper bound, so some a_N exceeds it; since the sequence "
    "increases, every later term lies within epsilon of L. Hence a_n tends to L.",
    "Le marché couvert ouvre à sept heures. Les maraîchers installent leurs étals de poireaux, de "
    "carottes et de pommes, tandis que le fromager découpe une meule de comté. 東京の朝は早い。電車は"
    "時間通りに到着し、人々は静かに乗り換える。 Die Bibliothek bleibt am Sonntag geschlossen.",
]


def snapshot(repo: str) -> Path:
    if Path(repo).is_dir():
        return Path(repo)
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    found = sorted(glob.glob(str(hub / f"models--{repo.replace('/', '--')}" / "snapshots" / "*")))
    if not found:
        sys.exit(f"{repo} is not in the Hugging Face cache; pull it first")
    return Path(found[-1])


def default_out(repo: str, ane: int) -> Path:
    return Path.home() / ".cache/pie/ane" / repo.replace("/", "--") / f"a{ane}-w8a8"


class Checkpoint:
    def __init__(self, root: Path):
        self.root = root
        index = root / "model.safetensors.index.json"
        if index.exists():
            self.where = json.loads(index.read_text())["weight_map"]
        else:
            self.where = {k: "model.safetensors" for k in mx.load(str(root / "model.safetensors"))}
        self.files: dict[str, dict] = {}
        config = json.loads((root / "config.json").read_text())
        self.config = config.get("text_config", config)
        self.quant = config.get("quantization") or config.get("quantization_config")

    def get(self, name: str) -> mx.array:
        file = self.where[name]
        if file not in self.files:
            self.files = {file: mx.load(str(self.root / file))}
        return self.files[file][name]

    def has(self, name: str) -> bool:
        return name in self.where

    def dense(self, stem: str) -> np.ndarray:
        w = self.get(stem + ".weight")
        if self.has(stem + ".scales"):
            q = self.quant
            w = mx.dequantize(w, self.get(stem + ".scales"), self.get(stem + ".biases"),
                              group_size=q["group_size"], bits=q["bits"])
        return np.array(w.astype(mx.float32))


def prefix(ck: Checkpoint) -> str:
    for p in ("language_model.model.layers.", "model.language_model.layers.", "model.layers."):
        if any(k.startswith(p) for k in ck.where):
            return p
    sys.exit("no `layers.N` prefix in this checkpoint")


def hadamard(n: int) -> np.ndarray:
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(n)).astype(np.float32)


def rotated(w: np.ndarray) -> np.ndarray:
    """`w @ blockdiag(H)` along the contraction: `x w^T == (x H)(w H)^T`."""
    rows, cols = w.shape
    return (w.reshape(rows, cols // BLOCK, BLOCK) @ hadamard(BLOCK)).reshape(rows, cols)


def program(gate: np.ndarray, up: np.ndarray, down: np.ndarray, rows: int):
    width, cols = gate.shape[1], gate.shape[0]
    h = hadamard(BLOCK).astype(np.float16)
    gate, up, down = (rotated(w).astype(np.float16) for w in (gate, up, down))

    @mb.program(input_specs=[mb.TensorSpec(shape=(rows, width), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def mlp(x):
        x = mb.reshape(x=mb.matmul(x=mb.reshape(x=x, shape=(rows, width // BLOCK, BLOCK)), y=h),
                       shape=(rows, width))
        g = mb.linear(x=x, weight=gate)
        u = mb.linear(x=x, weight=up)
        a = mb.mul(x=mb.silu(x=g), y=u)
        a = mb.reshape(x=mb.matmul(x=mb.reshape(x=a, shape=(rows, cols // BLOCK, BLOCK)), y=h),
                       shape=(rows, cols))
        return mb.linear(x=a, weight=down, name="y")

    return ct.convert(mlp, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                      compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE)


def w8a8(model, samples: list[np.ndarray]):
    w8 = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel"))
    model = cto.linear_quantize_weights(model, w8)
    a8 = cto.OptimizationConfig(op_type_configs={
        "linear": cto.experimental.OpActivationLinearQuantizerConfig(mode="linear_symmetric")})
    return cto.experimental.linear_quantize_activations(model, a8, [{"x": s.astype(np.float16)} for s in samples])


def tiles(rows: np.ndarray, count: int, n: int) -> list[np.ndarray]:
    reps = -(-count * n // len(rows))
    tiled = np.concatenate([rows] * reps)[: count * n]
    return list(tiled.reshape(n, count, -1))


def capture(repo: str, layers: list[int]) -> dict[int, np.ndarray]:
    """Each dense layer's MLP inputs over the calibration texts, from mlx-lm."""
    from mlx_lm import load
    import mlx.nn as nn

    model, tokenizer = load(repo)
    blocks = None
    for path in ("language_model.model.layers", "model.layers", "layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            blocks = obj
            break
        except AttributeError:
            continue
    if blocks is None:
        sys.exit("cannot find the decoder layers in the mlx-lm model")
    seen: dict[int, list[np.ndarray]] = {l: [] for l in layers}

    class Tap(nn.Module):
        def __init__(self, inner, at):
            super().__init__()
            self.inner, self.at = inner, at

        def __call__(self, x):
            seen[self.at].append(np.array(x.reshape(-1, x.shape[-1]).astype(mx.float32)))
            return self.inner(x)

    for l in layers:
        blocks[l].mlp = Tap(blocks[l].mlp, l)
    for text in CALIBRATION:
        ids = mx.array([tokenizer.encode(text)])
        mx.eval(model(ids))
    out = {l: np.concatenate(v) for l, v in seen.items()}
    del model
    mx.clear_cache()
    return out


def seconds(fn, n=6) -> float:
    for _ in range(2):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def calibrate_split(ck: Checkpoint, stem: str, layer: int, rows: np.ndarray, buckets: list[int]) -> int:
    """Columns that balance this Mac's Neural Engine against its GPU."""
    inter, hidden = ck.config["intermediate_size"], ck.config["hidden_size"]
    t = max(buckets)
    q = ck.quant or {"group_size": 64, "bits": 4}
    x = mx.random.normal((t, hidden)).astype(mx.bfloat16)
    ws = [mx.quantize(mx.random.normal(s).astype(mx.bfloat16) * 0.02, group_size=q["group_size"], bits=q["bits"])
          for s in [(2 * inter, hidden), (hidden, inter)]]

    def gpu():
        g = mx.quantized_matmul(x, *ws[0], transpose=True, group_size=q["group_size"], bits=q["bits"])
        a = g[:, :inter] * mx.sigmoid(g[:, :inter]) * g[:, inter:]
        mx.eval(mx.quantized_matmul(a, *ws[1], transpose=True, group_size=q["group_size"], bits=q["bits"]))

    # pie's GPU path runs these matmuls slower than MLX does (41 vs 55 TF/s
    # measured on M5 Max), so the GPU's share is costed accordingly.
    gpu_per_col = seconds(gpu) / inter * 1.3
    probe = 4096 if inter > 8192 else inter // 2
    base = f"{stem}{layer}.mlp"
    gate = ck.dense(f"{base}.gate_proj")[-probe:]
    up = ck.dense(f"{base}.up_proj")[-probe:]
    down = ck.dense(f"{base}.down_proj")[:, -probe:]
    model = w8a8(program(gate, up, down, t), tiles(rows, t, 2))
    xin = {"x": rows[:t].astype(np.float16) if len(rows) >= t else tiles(rows, t, 1)[0].astype(np.float16)}
    ane_per_col = seconds(lambda: model.predict(xin)) / probe
    # The hand-off costs a little on both sides; leave the Neural Engine slack.
    ane = int(inter * gpu_per_col / (gpu_per_col + ane_per_col) * 0.92) // STEP * STEP
    print(f"calibration: GPU {gpu_per_col * 1e6:.2f} us/col, Neural Engine {ane_per_col * 1e6:.2f} us/col "
          f"at {t} rows -> {ane} of {inter} columns", flush=True)
    return max(STEP, min(ane, inter - STEP))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="Hugging Face repo id (in the local cache)")
    ap.add_argument("--ane", type=int, default=None,
                    help="intermediate columns the Neural Engine takes (a multiple of 256); "
                         "omit to calibrate for this Mac")
    ap.add_argument("--buckets", default="512,2048", help="row counts compiled per layer")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--layers", default=None, help="only these layers, e.g. 0-3 (for a quick check)")
    args = ap.parse_args()

    ck = Checkpoint(snapshot(args.model))
    cfg = ck.config
    hidden, inter, count = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
    if hidden % BLOCK:
        sys.exit(f"hidden size {hidden} is not a multiple of the {BLOCK}-wide Hadamard block")
    buckets = [int(b) for b in args.buckets.split(",")]
    stem = prefix(ck)
    wanted = range(count)
    if args.layers:
        lo, _, hi = args.layers.partition("-")
        wanted = range(int(lo), int(hi or lo) + 1)
    dense = [l for l in wanted if ck.has(f"{stem}{l}.mlp.gate_proj.weight")]
    if not dense:
        sys.exit("no dense MLP layer to split (MoE layers stay on the GPU)")

    t0 = time.time()
    rows = capture(args.model, dense)
    print(f"captured {len(next(iter(rows.values())))} rows per layer in {time.time() - t0:.0f}s", flush=True)
    ane = args.ane or calibrate_split(ck, stem, dense[0], rows[dense[0]], buckets)
    if ane % STEP or ane >= inter:
        sys.exit(f"--ane must be a multiple of {STEP} below {inter}")
    keep = inter - ane
    out = args.out or default_out(args.model, ane)
    out.mkdir(parents=True, exist_ok=True)

    layers = []
    for l in dense:
        target = out / f"layer{l:03d}.mlmodelc"
        if target.exists():
            layers.append(l)
            continue
        t0 = time.time()
        base = f"{stem}{l}.mlp"
        gate = ck.dense(f"{base}.gate_proj")[keep:]
        up = ck.dense(f"{base}.up_proj")[keep:]
        down = ck.dense(f"{base}.down_proj")[:, keep:]
        x = rows[l]
        held, fit = x[: len(x) // 8], x[len(x) // 8:]
        staging = out / f".layer{l:03d}"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        desc = ct.utils.MultiFunctionDescriptor()
        for n in buckets:
            path = staging / f"t{n}.mlpackage"
            model = w8a8(program(gate, up, down, n), tiles(fit, n, 2))
            model.save(str(path))
            desc.add_function(str(path), src_function_name="main", target_function_name=f"t{n}")
            if n == min(buckets):
                probe = tiles(held, n, 1)[0]
                want = (probe @ gate.T) / (1 + np.exp(-(probe @ gate.T))) * (probe @ up.T) @ down.T
                got = model.predict({"x": probe.astype(np.float16)})["y"].astype(np.float32)
                err = np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-9)
        desc.default_function_name = f"t{max(buckets)}"
        package = staging / "layer.mlpackage"
        ct.utils.save_multifunction(desc, str(package))
        shutil.move(ct.models.utils.compile_model(str(package)), target)
        shutil.rmtree(staging, ignore_errors=True)
        layers.append(l)
        print(f"layer {l}: {time.time() - t0:.1f}s, relative error {err:.4f} on held-out rows", flush=True)

    known = json.loads((out / "meta.json").read_text())["layers"] if (out / "meta.json").exists() else []
    meta = {
        "model": args.model,
        "hidden": hidden,
        "intermediate": inter,
        "ane": ane,
        "buckets": buckets,
        "input": "x",
        "output": "y",
        "precision": "w8a8-hadamard",
        "layers": sorted(set(layers) | set(known)),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(out)


if __name__ == "__main__":
    main()
