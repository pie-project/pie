#!/usr/bin/env python3
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

BLOCK = 128
STEP = 256



def snapshot(repo: str) -> Path:
    if Path(repo).is_dir():
        return Path(repo)
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    found = sorted(glob.glob(str(hub / f"models--{repo.replace('/', '--')}" / "snapshots" / "*")))
    if not found:
        sys.exit(f"{repo} is not in the Hugging Face cache; pull it first")
    return Path(found[-1])


def default_out(repo: str, ane: int) -> Path:
    return Path.home() / ".cache/pie/ane" / repo.replace("/", "--") / f"a{ane}"


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


def fingerprint(ck: Checkpoint, stem: str, layers: int) -> str:
    first = next(l for l in range(layers) if ck.has(f"{stem}{l}.mlp.gate_proj.weight"))
    codes = np.array(ck.get(f"{stem}{first}.mlp.gate_proj.weight")[:64])
    h = 0xCBF29CE484222325
    for b in codes.tobytes():
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def hadamard(n: int) -> np.ndarray:
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(n)).astype(np.float32)


def rotated(w: np.ndarray) -> np.ndarray:
    rows, cols = w.shape
    return (w.reshape(rows, cols // BLOCK, BLOCK) @ hadamard(BLOCK)).reshape(rows, cols)


def per_token(v):
    peak = mb.maximum(x=mb.reduce_max(x=mb.abs(x=v), axes=[1], keep_dims=True), y=np.float16(1e-4))
    q = mb.quantize(input=mb.mul(x=v, y=mb.real_div(x=np.float16(127.0), y=peak)),
                    scale=np.float16(1.0), output_dtype="int8")
    return mb.dequantize(input=q, scale=np.float16(1.0)), mb.mul(x=peak, y=np.float16(1 / 127))


def program(gate: np.ndarray, up: np.ndarray, down: np.ndarray, rows: int):
    width, cols = gate.shape[1], gate.shape[0]
    h = hadamard(BLOCK).astype(np.float16)
    gate, up, down = (rotated(w).astype(np.float16) for w in (gate, up, down))

    @mb.program(input_specs=[mb.TensorSpec(shape=(rows, width), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def mlp(x):
        x = mb.reshape(x=mb.matmul(x=mb.reshape(x=x, shape=(rows, width // BLOCK, BLOCK)), y=h),
                       shape=(rows, width))
        xq, xs = per_token(x)
        g = mb.mul(x=mb.linear(x=xq, weight=gate), y=xs)
        u = mb.mul(x=mb.linear(x=xq, weight=up), y=xs)
        half = mb.mul(x=g, y=np.float16(0.5))
        a = mb.mul(x=mb.mul(x=half, y=mb.add(x=mb.tanh(x=half), y=np.float16(1.0))), y=u)
        a = mb.reshape(x=mb.matmul(x=mb.reshape(x=a, shape=(rows, cols // BLOCK, BLOCK)), y=h),
                       shape=(rows, cols))
        aq, scale = per_token(a)
        return mb.mul(x=mb.linear(x=aq, weight=down), y=scale, name="y")

    model = ct.convert(mlp, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE)
    w8 = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel"))
    return cto.linear_quantize_weights(model, w8)


def seconds(fn, n=6) -> float:
    for _ in range(2):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def calibrate_split(ck: Checkpoint, stem: str, layer: int, buckets: list[int]) -> int:
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

    gpu_per_col = seconds(gpu) / inter * 1.3
    probe = 4096 if inter > 8192 else inter // 2
    base = f"{stem}{layer}.mlp"
    gate = ck.dense(f"{base}.gate_proj")[-probe:]
    up = ck.dense(f"{base}.up_proj")[-probe:]
    down = ck.dense(f"{base}.down_proj")[:, -probe:]
    model = program(gate, up, down, t)
    xin = {"x": np.random.default_rng(0).standard_normal((t, hidden)).astype(np.float16)}
    ane_per_col = seconds(lambda: model.predict(xin)) / probe
    ane = int(inter * gpu_per_col / (gpu_per_col + ane_per_col) * 0.92) // STEP * STEP
    print(f"calibration: GPU {gpu_per_col * 1e6:.2f} us/col, Neural Engine {ane_per_col * 1e6:.2f} us/col "
          f"at {t} rows -> {ane} of {inter} columns", flush=True)
    return max(STEP, min(ane, inter - STEP))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Neural Engine half of each dense MLP for pie's Metal engine.")
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

    ane = args.ane or calibrate_split(ck, stem, dense[0], buckets)
    if ane % STEP or ane >= inter:
        sys.exit(f"--ane must be a multiple of {STEP} below {inter}")
    keep = inter - ane
    out = args.out or default_out(args.model, ane)
    stamp = fingerprint(ck, stem, count)
    expected = {"fingerprint": stamp, "hidden": hidden, "intermediate": inter, "ane": ane,
                "buckets": buckets, "precision": "w8a8-hadamard-per-token"}
    previous = out / "meta.json"
    if out.exists() and (not previous.exists() or any(
            json.loads(previous.read_text()).get(key) != value for key, value in expected.items())):
        shutil.rmtree(out)
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
        staging = out / f".layer{l:03d}"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        desc = ct.utils.MultiFunctionDescriptor()
        for n in buckets:
            path = staging / f"t{n}.mlpackage"
            program(gate, up, down, n).save(str(path))
            desc.add_function(str(path), src_function_name="main", target_function_name=f"t{n}")
        desc.default_function_name = f"t{max(buckets)}"
        package = staging / "layer.mlpackage"
        ct.utils.save_multifunction(desc, str(package))
        shutil.move(ct.models.utils.compile_model(str(package)), target)
        shutil.rmtree(staging, ignore_errors=True)
        layers.append(l)
        print(f"layer {l}: {time.time() - t0:.1f}s", flush=True)

    known = json.loads((out / "meta.json").read_text())["layers"] if (out / "meta.json").exists() else []
    meta = {
        "model": args.model,
        "hidden": hidden,
        "intermediate": inter,
        "ane": ane,
        "buckets": buckets,
        "input": "x",
        "output": "y",
        "precision": "w8a8-hadamard-per-token",
        "fingerprint": stamp,
        "layers": sorted(set(layers) | set(known)),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(out)


if __name__ == "__main__":
    main()
