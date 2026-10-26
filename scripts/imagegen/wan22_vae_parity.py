#!/usr/bin/env python3
"""
wan22_vae_parity.py -- decode a Wan 2.2 latent THROUGH THE RUNTIME, frame by
frame as `examples/text-to-media` does, and score each chunk against the
diffusers reference.

    python wan22_golden.py --vae-ref        # the reference (vae/* only)
    python wan22_vae_parity.py all --out /tmp/wan22-vae --config <config.wan.toml>

Gate: `cos >= 0.9999` per chunk, and the fork's last chunk equal to the parent's.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

from zimage_vae_parity import document, f32s, pieces, score

DEFAULT_GOLDEN = os.path.join(
    os.environ.get("PIE_IMAGEGEN_GOLDEN", "/root/.cache/pie-imagegen/golden"),
    "wan22",
    "wan22_vae_ref",
)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
WORKSPACE = os.path.join(REPO, "examples")

COS_TOL = 0.9999
FORK_TOL = 0.0

def shapes(golden: str) -> dict:
    path = os.path.join(golden, "shapes.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path}: no golden. Run `python scripts/imagegen/wan22_golden.py --vae-ref`.")
    return json.load(open(path))

def case(args) -> str:
    box = shapes(args.golden)["latent"]
    latent = f32s(os.path.join(args.golden, "latent.f32"))
    assert len(latent) == box["t"] * box["h"] * box["w"] * box["channels"]
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "case.json")
    with open(path, "w") as f:
        json.dump({"latent": [round(v, 7) for v in latent], **{k: box[k] for k in
                   ("t", "h", "w", "channels")}}, f)
    print(f"[case] {box['t']}x{box['h']}x{box['w']}x{box['channels']} latent -> {path}")
    return path

def wasm() -> str:
    if not os.environ.get("PIE_INFERLETS_NO_BUILD"):
        done = subprocess.run(
            ["cargo", "build", "-p", "wan2-vae-parity", "--target", "wasm32-wasip3", "--release"],
            cwd=WORKSPACE, capture_output=True, text=True)
        if done.returncode != 0:
            sys.stderr.write(done.stderr)
            raise SystemExit("building wan2-vae-parity for wasm32-wasip3 failed")
    path = os.path.join(WORKSPACE, "target/wasm32-wasip3/release/wan2_vae_parity.wasm")
    if not os.path.exists(path):
        raise SystemExit(f"{path}: no wasm")
    return path

def run(args) -> None:
    path = os.path.join(args.out, "case.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path}: no case JSON; run `case` first")
    pie = args.pie or shutil.which("pie") or os.path.join(REPO, "target/debug/pie")
    cmd = [pie]
    if args.config:
        cmd += ["--config", os.path.expanduser(args.config)]
    cmd += ["run", "--path", wasm(), "-o", args.out, "--"]
    for i, piece in enumerate(pieces(open(path).read())):
        cmd += [f"--case_{i}", piece]
    print(f"[run] {' '.join(cmd[:6])} ...")
    done = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    out = os.path.join(args.out, "pie.json")
    with open(out, "w") as f:
        f.write(done.stdout)
    with open(out[:-5] + ".stderr", "w") as f:
        f.write(done.stderr)
    if done.returncode != 0:
        sys.stderr.write(done.stdout[-4000:] + done.stderr[-8000:])
        raise SystemExit(f"pie run failed ({done.returncode}); see {out[:-5]}.stderr")
    print(f"[run] -> {out}")

def compare(args) -> int:
    doc = document(os.path.join(args.out, "pie.json"))
    meta = shapes(args.golden)
    want = f32s(os.path.join(args.golden, "pixels.f32"))
    got = f32s(os.path.join(args.out, "file-0000.bin"))
    box = meta["pixels"]
    print(f"[compare] fires {doc.get('fires')} -> frames {doc.get('chunk_frames')}, "
          f"{doc.get('pixel_h')}x{doc.get('pixel_w')}")
    if len(got) != len(want) or (doc.get("pixel_h"), doc.get("pixel_w")) != (box["h"], box["w"]):
        print(f"[compare] FAIL: {len(got)} numbers back, the golden holds {len(want)}")
        return 1
    frame = box["h"] * box["w"] * box["channels"]
    chunks = meta["chunks"]
    ok = True
    for k, (a, b) in enumerate(zip(chunks, chunks[1:])):
        cos, mean_err, max_err = score(got[a * frame:b * frame], want[a * frame:b * frame])
        good = cos >= COS_TOL
        ok &= good
        print(f"[compare] chunk {k} (frames {a}..{b - 1}): cos {cos:.6f} (>= {COS_TOL}), "
              f"mean |err| {mean_err:.5f}, max |err| {max_err:.5f}  --  "
              f"{'PASS' if good else 'FAIL'}")
    # The last chunk was decoded on a fork too, and must match the parent.
    fork = doc.get("fork_max_abs_diff")
    good = fork is not None and fork <= FORK_TOL
    ok &= good
    print(f"[compare] fork: the last chunk on a fork, max |diff| from the parent's {fork} "
          f"(<= {FORK_TOL})  --  {'PASS' if good else 'FAIL'}")
    print(f"[compare] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["case", "run", "compare", "all"])
    ap.add_argument("--out", default="/tmp/wan22-vae", help="working directory")
    ap.add_argument("--golden", default=DEFAULT_GOLDEN, help="the reference dump")
    ap.add_argument("--config", default=None, help="pie config (its `[model] source` is Wan 2.2)")
    ap.add_argument("--pie", default=None, help="the pie binary")
    args = ap.parse_args()
    args.golden = os.path.expanduser(args.golden)
    if args.stage in ("case", "all"):
        case(args)
    if args.stage in ("run", "all"):
        run(args)
    if args.stage in ("compare", "all"):
        return compare(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
