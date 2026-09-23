#!/usr/bin/env python3
"""
flux2_klein_parity.py -- drive pie's `flux2-klein-4b` row against the diffusers golden.

The reference and its dump come from `flux2_golden.py --full` (see README).
This script is the other half: it turns the golden's *inputs* into the case
the `flux2-klein-parity` inferlet reads, runs it, turns its answers back into
an `.npz` under the golden's own key names, diffs the two with `compare.py`,
and decodes both final latents with the diffusers VAE to PNGs.

    # everything, one command (GPU for pie AND the VAE decode)
    CUDA_VISIBLE_DEVICES=3 python flux2_klein_parity.py all \\
        --out /tmp/flux2-klein-parity --config ~/.pie/config.flux2.toml

    # or step by step
    python flux2_klein_parity.py case    --out /tmp/flux2-klein-parity
    python flux2_klein_parity.py run     --out /tmp/flux2-klein-parity --config ~/.pie/config.flux2.toml
    python flux2_klein_parity.py collect --out /tmp/flux2-klein-parity
    python flux2_klein_parity.py compare --out /tmp/flux2-klein-parity
    python flux2_klein_parity.py decode  --out /tmp/flux2-klein-parity

`case` and `decode` import torch (the `context_embedder` fold reads a bf16
safetensors plane; the decode runs `AutoencoderKLFlux2`); the rest is numpy.

WHAT IS COMPARED
----------------
Three readings, in the order the golden was made, every one gated on cosine
>= 0.999 (a 27-layer bf16 trunk feeding a 4B bf16 DiT: `compare.py`'s
`--cos-tol`; the max-abs and rel columns are reported, not gated):

  text.hidden    pie's `text` reading over the family's template (the
                 chat surface's `first_user` + `cue`) against the golden's
                 `prompt_embeds[:L]` folded through `context_embedder` in
                 numpy (the golden dumps the raw `[512, 7680]` stack; pie
                 exports the fold). `L` is the unpadded length -- the
                 golden pads to 512 under a key mask, so its first `L`
                 rows are what an unpadded prefill computes. The token ids
                 are checked against `text.input_ids` exactly.
  dit.step0.out  one `denoise` step over the golden's own step-0 inputs
                 (its prompt embeds folded, its noise, sigma 1, its ids).
  probe.step0..3 the same one-fire step from the REFERENCE's own latent at
                 every sigma (`dit.step{k}.in.hidden_states`), so each
                 step's velocity is read under the reference's state
                 instead of pie's drifting one. This is the model check.
  sched.x1..x4   the latent after every Euler step, `latent.final` = x4:
                 the four pinned sigmas, `euler_step` in the image lane's
                 epilogue. REPORTED, NOT GATED -- see NUMERICS.

NUMERICS: WHY THE TRAJECTORY IS NOT GATED
-----------------------------------------
klein-4B is distilled to four steps, and its last one takes sigma 0.767 to
0, so the trajectory amplifies a velocity difference by ~6x. bf16 alone
moves the step-0 velocity by 4e-4 in cosine: running the SAME diffusers
transformer in fp32 and comparing to the bf16 golden reads

    velocity  fp32 vs golden bf16   cos 0.999618
    latent x4 fp32 vs golden bf16   cos 0.997510

so no bf16 engine reaches 0.999 on the final latent against this dump --
an exact fp32 computation does not either. The per-step velocities are
gated at 0.999 instead, and the final latent is reported beside the fp32
number above and checked as PSNR after the VAE decode.

Then `decode` writes `golden.png` / `pie.png` (both final latents through
the same VAE) and reports PSNR between them, plus PSNR of `golden.png`
against the pipeline's own `flux2_golden.png` (the decode path's own check).

With `--native` the guest also runs the trajectory over pie's OWN text rows
(unpadded, `L` rows: the reference's DiT saw 512 rows, pads included, so
this is a different conditioning and is reported, not gated):
`native.latent.final`, decoded as `native.png`.

THE SUBMIT DEADLINE IS NOT A KNOB THIS HARNESS TOUCHES
-----------------------------------------------------
It used to be. The two lanes of a denoise group compose into one fire only
if the scheduler seals them into one frame, and that wait was once leashed
by `[runtime] submit_deadline` (50 ms): a 1024^2 job's first submit carries
a 6 MB seeded `context` channel and a 2 MB `latents` one across the guest
boundary, took longer than the leash under load, and the group's first frame
sealed with the IMAGE lane alone -- an UNCONDITIONED velocity reported as a
success (cos ~0.535 against the golden, cos 0.9999 against a reference run
with no text at all). This script demanded `submit_deadline = "10s"` to
paper over it.

The runtime now keeps the promise instead: a STATED cohort fires whole or
not at all, so the deadline cannot buy or cost correctness here (see
`crates/runtime/src/scheduler/frame.rs`, `group_short` / `cohort_verdict`).
The demand is gone, and this harness runs at the 50 ms DEFAULT on purpose --
if a partial group is ever sealed again, this parity is what says so.

THE CASE CROSSES AS FILES
-------------------------
Eight megabytes of f32 is past argv. The sandbox mounts
`<fs_scratch_dir>/<instance-id>` at `/scratch` when the instance starts, so
`run` watches the config's `fs_scratch_dir` for the new directory, drops the
case there, and the guest (which polls for it) reads on. Answers come back
through `session::send_file`, which `pie run -o DIR` writes as
`file-NNNN.bin` in send order; the guest's JSON names them.
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
import tomllib

import numpy as np

DEFAULT_GOLDEN = os.path.join(
    os.environ.get("PIE_IMAGEGEN_GOLDEN", "/root/.cache/pie-imagegen/golden"), "flux2"
)
DEFAULT_SNAPSHOT = os.path.join(
    os.path.expanduser("~/.cache/huggingface/hub"),
    "models--black-forest-labs--FLUX.2-klein-4B/snapshots/*/",
)
DEFAULT_SKU = "flux2-klein-4b-bf16-kv-bf16"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

TOLERANCES = ["--cos-tol", "0.999"]

GATED_KEYS = ["text.hidden", "dit.step0.out", "probe.step0.out", "probe.step1.out",
              "probe.step2.out", "probe.step3.out"]
TRAJECTORY_KEYS = ["sched.x1", "sched.x2", "sched.x3", "sched.x4", "latent.final"]

def snapshot(args) -> str:
    found = sorted(glob.glob(args.snapshot))
    if not found:
        raise SystemExit(f"{args.snapshot}: no snapshot; `hf download black-forest-labs/FLUX.2-klein-4B`")
    return found[-1]

def context_embedder(snap: str) -> np.ndarray:
    """`transformer/context_embedder.weight` `[3072, 7680]` (bf16) as float32."""
    import torch
    from safetensors import safe_open

    for path in sorted(glob.glob(os.path.join(snap, "transformer", "*.safetensors"))):
        with safe_open(path, "pt") as st:
            if "context_embedder.weight" in st.keys():
                w = st.get_tensor("context_embedder.weight")
                assert w.shape == (3072, 7680), w.shape
                return w.to(torch.float32).numpy()
    raise SystemExit(f"{snap}/transformer: no context_embedder.weight")

def fold(embeds: np.ndarray, w: np.ndarray) -> np.ndarray:
    """`context_embedder(cat(h9, h18, h27))`: `[L, 7680] @ W.T`, no bias, fp32."""
    return (embeds.astype(np.float32) @ w.T).astype(np.float32)

def case(args) -> None:
    dump = np.load(os.path.join(args.golden, "flux2_golden.npz"))
    with open(os.path.join(args.golden, "MANIFEST.json")) as f:
        prompt = json.load(f)["prompt"]
    if "text.input_ids" not in dump.files:
        raise SystemExit("the golden has no `text.input_ids`; regenerate it: "
                         "`flux2_golden.py --full`")
    mask = dump["text.attention_mask"]
    ids = dump["text.input_ids"][mask > 0]
    pe = dump["prompt_embeds"][0]
    noise = dump["dit.step0.in.hidden_states"][0]
    init = dump["noise.init.0"][0]
    if not np.array_equal(noise, init):
        print(f"[case] note: step-0 hidden_states differ from noise.init.0 (max {np.abs(noise - init).max():.3g}); "
              f"the step-0 input is what is fed")
    img_ids = dump["dit.step0.in.img_ids"][0]
    txt_ids = dump["dit.step0.in.txt_ids"][0]
    sigmas = dump["sigmas"].astype(np.float32)
    assert sigmas[-1] == 0.0, sigmas
    timestep = float(dump["dit.step0.in.timestep"][0])
    assert abs(timestep - sigmas[0]) < 1e-6, (timestep, sigmas[0])

    w = context_embedder(snapshot(args))
    ctx = fold(pe, w)

    os.makedirs(args.out, exist_ok=True)

    def raw(name: str, arr: np.ndarray) -> str:
        np.ascontiguousarray(arr, dtype=np.float32).tofile(os.path.join(args.out, name))
        return name

    probe = np.stack([dump[f"dit.step{k}.in.hidden_states"][0]
                      for k in range(len(sigmas) - 1)])
    doc = {
        "prompt": prompt,
        "context_file": raw("context.f32", ctx),
        "probe_latents_file": raw("probe.f32", probe),
        "text_rows": int(ctx.shape[0]),
        "context_width": int(ctx.shape[1]),
        "latents_file": raw("noise.f32", noise),
        "image_rows": int(noise.shape[0]),
        "channels": int(noise.shape[1]),
        "text_positions_file": raw("txt_pos.f32", txt_ids),
        "image_positions_file": raw("img_pos.f32", img_ids),
        "sigmas": [float(s) for s in sigmas[:-1]],
        "native": bool(args.native),
    }
    with open(os.path.join(args.out, "case.json"), "w") as f:
        json.dump(doc, f, indent=1)
    np.savez(os.path.join(args.out, "expected.npz"),
             **{"text.input_ids": ids, "text.hidden": ctx[: len(ids)]})
    print(f"[case] prompt {prompt!r}: {len(ids)} ids {ids.tolist()}")
    print(f"[case] context {ctx.shape}, noise {noise.shape}, sigmas {doc['sigmas']} -> {args.out}")

def wasm(inferlet: str) -> str:
    """The newest `.wasm` a build left for `inferlet`, building one first."""
    name = os.path.basename(os.path.normpath(inferlet))
    stem = name.replace("-", "_")
    workspace = os.path.dirname(os.path.normpath(inferlet))
    if not os.environ.get("PIE_INFERLETS_NO_BUILD"):
        done = subprocess.run(
            ["cargo", "build", "-p", name, "--target", "wasm32-wasip2"],
            cwd=workspace, capture_output=True, text=True,
        )
        if done.returncode != 0:
            sys.stderr.write(done.stderr)
            raise SystemExit(f"building {name} for wasm32-wasip2 failed")
    candidates = [
        os.path.join(workspace, "target/wasm32-wasip2/release", f"{stem}.wasm"),
        os.path.join(workspace, "target/wasm32-wasip2/debug", f"{stem}.wasm"),
    ]
    present = [path for path in candidates if os.path.exists(path)]
    if not present:
        raise SystemExit(f"no wasm for {name}; tried {', '.join(candidates)}")
    return max(present, key=os.path.getmtime)

def scratch_base(config: str) -> str:
    with open(os.path.expanduser(config), "rb") as f:
        cfg = tomllib.load(f)
    sandbox = cfg.get("sandbox", {})
    if not sandbox.get("allow_fs"):
        raise SystemExit(f"{config}: `[sandbox] allow_fs = true` is needed; the case crosses as files")
    base = sandbox.get("fs_scratch_dir")
    if not base:
        raise SystemExit(f"{config}: set `[sandbox] fs_scratch_dir` to a directory of its own")
    os.makedirs(base, exist_ok=True)
    return base

def run(args) -> None:
    case_path = os.path.join(args.out, "case.json")
    if not os.path.exists(case_path):
        raise SystemExit(f"{case_path}: no case; run `case` first")
    if not args.config:
        raise SystemExit("`run` needs --config (its `[model] model` is the imported artifact)")
    pie = args.pie or shutil.which("pie") or os.path.join(REPO, "target/debug/pie")
    if not os.path.exists(pie):
        raise SystemExit(f"{pie}: no pie binary. Build one with `cargo build -p pie --features cuda`, or pass --pie.")
    binary = wasm(args.inferlet)
    base = scratch_base(args.config)
    files_dir = os.path.join(args.out, "pie_files")
    shutil.rmtree(files_dir, ignore_errors=True)
    with open(case_path) as f:
        doc = json.load(f)
    payload = ["case.json", doc["context_file"], doc["latents_file"],
               doc["text_positions_file"], doc["image_positions_file"]]
    if doc.get("probe_latents_file"):
        payload.append(doc["probe_latents_file"])

    cmd = [pie, "--config", os.path.expanduser(args.config), "run", "--path", binary,
           "-o", files_dir, "--", "--case_file", "case.json",
           "--wait_secs", str(args.wait)]
    print(f"[run] {' '.join(cmd)}")
    before = set(os.listdir(base))
    stdout_path = os.path.join(args.out, "pie.stdout")
    stderr_path = os.path.join(args.out, "pie.stderr")
    with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, cwd=REPO, text=True)
        deadline = time.time() + args.wait
        target = None
        while proc.poll() is None and time.time() < deadline:
            fresh = sorted(
                name for name in set(os.listdir(base)) - before
                if os.path.isdir(os.path.join(base, name))
            )
            if fresh:
                target = os.path.join(base, fresh[-1])
                break
            time.sleep(0.2)
        if target is None:
            proc.wait()
            raise SystemExit(f"no scratch dir appeared under {base} (pie exited {proc.returncode}); "
                             f"see {stderr_path}")
        for name in payload[1:] + payload[:1]:
            tmp = os.path.join(target, f".{name}.tmp")
            shutil.copyfile(os.path.join(args.out, name), tmp)
            os.replace(tmp, os.path.join(target, name))
        print(f"[run] case dropped into {target}")
        proc.wait()
    if proc.returncode != 0:
        sys.stderr.write(open(stdout_path).read())
        sys.stderr.write(open(stderr_path).read())
        raise SystemExit(f"pie run failed ({proc.returncode})")
    print(f"[run] -> {stdout_path}, files in {files_dir}")

def document(path: str) -> dict:
    """`pie run` prints a human header before the document; take the JSON."""
    lines = [line for line in open(path).read().splitlines() if line.startswith("{")]
    if not lines:
        raise SystemExit(f"{path}: no JSON document (did the run fail?)")
    doc = json.loads(lines[-1])
    if "result" in doc and isinstance(doc["result"], (dict, str)):
        doc = doc["result"]
    if isinstance(doc, str):
        doc = json.loads(doc)
    return doc

def collect(args) -> None:
    doc = document(os.path.join(args.out, "pie.stdout"))
    files_dir = os.path.join(args.out, "pie_files")
    rows, width = doc["image_rows"], doc["channels"]
    text_rows, hidden_width = doc["text_rows"], doc["hidden_width"]

    def blob(index: int, shape: tuple[int, ...]) -> np.ndarray:
        path = os.path.join(files_dir, f"file-{index:04d}.bin")
        arr = np.fromfile(path, dtype="<f4")
        if arr.size != int(np.prod(shape)):
            raise SystemExit(f"{path}: {arr.size} f32, expected {shape}")
        return arr.reshape(shape)

    names = doc["files"]
    shapes = {"hidden.f32": (text_rows, hidden_width)}
    mine, native = {}, {}
    steps = 0
    for i, name in enumerate(names):
        shape = shapes.get(name, (rows, width))
        arr = blob(i, shape)
        if name == "hidden.f32":
            mine["text.hidden"] = arr
        elif name == "velocity0.f32":
            mine["dit.step0.out"] = arr
        elif (m := re.fullmatch(r"latent(\d+)\.f32", name)):
            mine[f"sched.x{m.group(1)}"] = arr
            steps = max(steps, int(m.group(1)))
        elif (m := re.fullmatch(r"probe_velocity(\d+)\.f32", name)):
            mine[f"probe.step{m.group(1)}.out"] = arr
        elif name == "native_velocity0.f32":
            native["native.dit.step0.out"] = arr
        elif (m := re.fullmatch(r"native_latent(\d+)\.f32", name)):
            native[f"native.sched.x{m.group(1)}"] = arr
    mine["latent.final"] = mine[f"sched.x{steps}"]
    if native:
        native["native.latent.final"] = native[f"native.sched.x{steps}"]

    expected = np.load(os.path.join(args.out, "expected.npz"))
    ids = np.asarray(doc["token_ids"], dtype=np.int64)
    want = expected["text.input_ids"]
    if not np.array_equal(ids, want):
        print(f"[collect] TOKEN IDS DIFFER\n  pie   {ids.tolist()}\n  golden {want.tolist()}")
    else:
        print(f"[collect] token ids match the pipeline's ({len(ids)} ids)")
    if mine["text.hidden"].shape[0] != len(want):
        raise SystemExit(f"pie encoded {mine['text.hidden'].shape[0]} rows; the golden has {len(want)} real tokens")

    dump = np.load(os.path.join(args.golden, "flux2_golden.npz"))
    theirs = {"text.hidden": expected["text.hidden"], "dit.step0.out": dump["dit.step0.out"][0],
              "latent.final": dump["latent.final"][0]}
    for k in range(1, steps + 1):
        theirs[f"sched.x{k}"] = dump[f"sched.x{k}"][0]
    for k in range(steps):
        if f"probe.step{k}.out" in mine:
            theirs[f"probe.step{k}.out"] = dump[f"dit.step{k}.out"][0]

    pie_npz = os.path.join(args.out, "flux2_klein_pie.npz")
    target_npz = os.path.join(args.out, "flux2_klein_target.npz")
    traj_npz = os.path.join(args.out, "flux2_klein_pie_trajectory.npz")
    traj_target_npz = os.path.join(args.out, "flux2_klein_target_trajectory.npz")
    gated = [k for k in GATED_KEYS if k in mine]
    np.savez(pie_npz, **{k: mine[k].astype(np.float32) for k in gated})
    np.savez(target_npz, **{k: theirs[k].astype(np.float32) for k in gated})
    traj = [k for k in TRAJECTORY_KEYS if k in mine]
    np.savez(traj_npz, **{k: mine[k].astype(np.float32) for k in traj})
    np.savez(traj_target_npz, **{k: theirs[k].astype(np.float32) for k in traj})
    print(f"[collect] gated {gated} -> {pie_npz}; golden rows -> {target_npz}")
    print(f"[collect] trajectory {traj} -> {traj_npz}")
    if native:
        native_npz = os.path.join(args.out, "flux2_klein_native.npz")
        np.savez(native_npz, **{k: v.astype(np.float32) for k, v in native.items()})
        print(f"[collect] native trajectory -> {native_npz}")
    with open(os.path.join(args.out, "ids.json"), "w") as f:
        json.dump({"pie": ids.tolist(), "golden": want.tolist(), "match": bool(np.array_equal(ids, want))}, f)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

def compare(args) -> int:
    mine = os.path.join(args.out, "flux2_klein_pie.npz")
    theirs = os.path.join(args.out, "flux2_klein_target.npz")
    for path in (mine, theirs):
        if not os.path.exists(path):
            raise SystemExit(f"{path}: missing; run `collect` first")
    ids = json.load(open(os.path.join(args.out, "ids.json")))
    status = 0
    if not ids["match"]:
        print("[compare] FAIL: the token ids are not the pipeline's")
        status = 1
    cmd = [sys.executable, os.path.join(HERE, "compare.py"), mine, theirs, "--sort-by", "key", *TOLERANCES]
    print(f"[compare] {' '.join(cmd)}")
    status = subprocess.call(cmd) or status
    walk = os.path.join(args.out, "flux2_klein_pie_trajectory.npz")
    walk_target = os.path.join(args.out, "flux2_klein_target_trajectory.npz")
    if os.path.exists(walk):
        print("[compare] the Euler trajectory (reported, not gated -- see NUMERICS):")
        a, b = np.load(walk), np.load(walk_target)
        for key in TRAJECTORY_KEYS:
            if key in a.files:
                print(f"  {key:<16} cos {cosine(a[key], b[key]):.6f}")
        print("  for scale: an fp32 diffusers run of the same model reads cos 0.997510 on "
              "`latent.final` against this bf16 golden")
    native = os.path.join(args.out, "flux2_klein_native.npz")
    if os.path.exists(native):
        n = np.load(native)
        dump = np.load(os.path.join(args.golden, "flux2_golden.npz"))
        print("[compare] native trajectory (pie's own text rows, unpadded; reported, not gated):")
        for key, golden in (("native.dit.step0.out", dump["dit.step0.out"][0]),
                            ("native.latent.final", dump["latent.final"][0])):
            if key in n.files:
                print(f"  {key:<28} cos {cosine(n[key], golden):.6f} vs the golden's")
    return status

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)

def decode(args) -> None:
    import torch
    from diffusers import AutoencoderKLFlux2
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
    from PIL import Image

    snap = snapshot(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    vae = AutoencoderKLFlux2.from_pretrained(snap, subfolder="vae", torch_dtype=dtype).to(device)
    dump = np.load(os.path.join(args.golden, "flux2_golden.npz"))
    ids = torch.from_numpy(dump["noise.init.1"]).to(device)
    size = int(json.load(open(os.path.join(args.golden, "MANIFEST.json")))["size"])
    scale = 2 ** (len(vae.config.block_out_channels) - 1)
    lat_h, lat_w = 2 * (size // (scale * 2)), 2 * (size // (scale * 2))
    processor = VaeImageProcessor(vae_scale_factor=scale * 2)

    def to_png(latent: np.ndarray, name: str) -> np.ndarray:
        x = torch.from_numpy(latent[None]).to(device, dtype)
        x = Flux2KleinPipeline._unpack_latents_with_ids(x, ids, lat_h // 2, lat_w // 2)
        mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device, dtype)
        std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(device, dtype)
        x = x * std + mean
        x = Flux2KleinPipeline._unpatchify_latents(x)
        with torch.no_grad():
            img = vae.decode(x, return_dict=False)[0]
        pil = processor.postprocess(img, output_type="pil")[0]
        path = os.path.join(args.out, name)
        pil.save(path)
        print(f"[decode] {path}")
        return np.asarray(pil)

    golden_png = to_png(dump["latent.final"][0], "golden.png")
    mine = np.load(os.path.join(args.out, "flux2_klein_pie_trajectory.npz"))
    pie_png = to_png(mine["latent.final"], "pie.png")
    ref = np.asarray(Image.open(os.path.join(args.golden, "flux2_golden.png")).convert("RGB"))
    print(f"[decode] PSNR(golden.png, flux2_golden.png) = {psnr(golden_png, ref):.2f} dB  (the decode path itself)")
    print(f"[decode] PSNR(pie.png, golden.png)          = {psnr(pie_png, golden_png):.2f} dB")
    native = os.path.join(args.out, "flux2_klein_native.npz")
    if os.path.exists(native):
        native_png = to_png(np.load(native)["native.latent.final"], "native.png")
        print(f"[decode] PSNR(native.png, golden.png)       = {psnr(native_png, golden_png):.2f} dB  (unpadded text rows; not gated)")

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["case", "run", "collect", "compare", "decode", "all"])
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--snapshot", default=DEFAULT_SNAPSHOT, help="the HF snapshot dir (glob ok)")
    ap.add_argument("--out", default="/tmp/flux2-klein-parity")
    ap.add_argument("--inferlet", default=os.path.join(REPO, "examples/flux2-klein-parity"))
    ap.add_argument("--config", default=None,
                    help=f"the serving config; its `[model] model` must be the artifact `{DEFAULT_SKU}` imported")
    ap.add_argument("--pie", default=None, help="the pie binary (default: PATH, else target/debug)")
    ap.add_argument("--wait", type=int, default=600, help="seconds to wait for the instance to start")
    ap.add_argument("--native", action="store_true", help="also run the trajectory over pie's own text rows")
    args = ap.parse_args()

    if args.cmd == "case":
        case(args)
    elif args.cmd == "run":
        run(args)
    elif args.cmd == "collect":
        collect(args)
    elif args.cmd == "compare":
        return compare(args)
    elif args.cmd == "decode":
        decode(args)
    else:
        case(args)
        run(args)
        collect(args)
        status = compare(args)
        decode(args)
        return status
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
