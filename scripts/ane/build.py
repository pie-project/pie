#!/usr/bin/env python3
"""Build the Neural Engine half of each dense MLP for pie's Metal engine.

pie splits a dense SwiGLU MLP along its intermediate axis on long prefills:
the GPU keeps columns [0, inter - ane) and the Neural Engine runs the last
`ane` columns as a CoreML program, both at once. This script writes one
multifunction CoreML model per layer (one function per row bucket, weights
shared) plus `meta.json`, which the engine reads when `PIE_ANE` is set.

    python scripts/ane/build.py mlx-community/Qwen3.8-27B-4bit --ane 3584

Needs `coremltools` and `mlx` (to dequantize MLX affine checkpoints).
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
        if ck.has(f"{p}0.mlp.gate_proj.weight") or any(k.startswith(p) for k in ck.where):
            return p
    sys.exit("no `layers.N` prefix in this checkpoint")


def program(gate: np.ndarray, up: np.ndarray, down: np.ndarray, rows: int):
    width = gate.shape[1]

    @mb.program(input_specs=[mb.TensorSpec(shape=(rows, width), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def mlp(x):
        g = mb.linear(x=x, weight=gate.astype(np.float16))
        u = mb.linear(x=x, weight=up.astype(np.float16))
        return mb.linear(x=mb.mul(x=mb.silu(x=g), y=u), weight=down.astype(np.float16), name="y")

    model = ct.convert(mlp, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT16)
    w8 = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel"))
    return cto.linear_quantize_weights(model, w8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="Hugging Face repo id (in the local cache) or a checkpoint directory")
    ap.add_argument("--ane", type=int, required=True,
                    help="intermediate columns the Neural Engine takes (a multiple of 256)")
    ap.add_argument("--buckets", default="512,1024,2048", help="row counts compiled per layer")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--layers", default=None, help="only these layers, e.g. 0-3 (for a quick check)")
    args = ap.parse_args()
    if args.ane % 256:
        sys.exit("--ane must be a multiple of 256 so both halves keep whole quant groups")

    ck = Checkpoint(snapshot(args.model))
    cfg = ck.config
    hidden, inter, count = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
    if args.ane >= inter:
        sys.exit(f"--ane {args.ane} leaves the GPU nothing of {inter}")
    keep = inter - args.ane
    buckets = [int(b) for b in args.buckets.split(",")]
    out = args.out or default_out(args.model, args.ane)
    out.mkdir(parents=True, exist_ok=True)
    stem = prefix(ck)
    wanted = range(count)
    if args.layers:
        lo, _, hi = args.layers.partition("-")
        wanted = range(int(lo), int(hi or lo) + 1)

    layers = []
    for l in wanted:
        base = f"{stem}{l}.mlp"
        if not ck.has(f"{base}.gate_proj.weight"):
            continue  # a routed (MoE) layer stays on the GPU
        target = out / f"layer{l:03d}.mlmodelc"
        if target.exists():
            layers.append(l)
            continue
        t0 = time.time()
        gate = ck.dense(f"{base}.gate_proj")[keep:]
        up = ck.dense(f"{base}.up_proj")[keep:]
        down = ck.dense(f"{base}.down_proj")[:, keep:]
        staging = out / f".layer{l:03d}"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        desc = ct.utils.MultiFunctionDescriptor()
        for rows in buckets:
            path = staging / f"t{rows}.mlpackage"
            program(gate, up, down, rows).save(str(path))
            desc.add_function(str(path), src_function_name="main", target_function_name=f"t{rows}")
        desc.default_function_name = f"t{buckets[-1]}"
        package = staging / "layer.mlpackage"
        ct.utils.save_multifunction(desc, str(package))
        compiled = ct.models.utils.compile_model(str(package))
        shutil.move(compiled, target)
        shutil.rmtree(staging, ignore_errors=True)
        layers.append(l)
        print(f"layer {l}: {time.time() - t0:.1f}s", flush=True)

    meta = {
        "model": args.model,
        "hidden": hidden,
        "intermediate": inter,
        "ane": args.ane,
        "buckets": buckets,
        "input": "x",
        "output": "y",
        "layers": sorted(set(layers) | set(json.loads((out / "meta.json").read_text())["layers"]
                                            if (out / "meta.json").exists() else [])),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(out)


if __name__ == "__main__":
    main()
