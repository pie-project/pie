#!/usr/bin/env python3
"""
zimage_vae_parity.py -- drive the `vae.decode` reading FROM A GUEST and diff
its pixels against the reference dump.

The GPU gate `engine-cuda/tests/the_z_image_vae_answers_the_reference` already
proves the VAE's numbers, but it feeds the port from the host, through a door
no guest can reach. This is the other half: the same golden latent, bound as a
`Voxels` PORT CHANNEL by the `zimage-vae-parity` inferlet, fired through the
real runtime, read back through the `pixels()` intrinsic, and — the point —
sent to the client as a PNG that `pie run -o DIR` writes to disk.

    # 1. the case the inferlet reads (the golden latent, as JSON)
    python zimage_vae_parity.py case    --out /tmp/zimage-vae

    # 2. run it (the config's `[model] model` is the imported artifact)
    python zimage_vae_parity.py run     --out /tmp/zimage-vae \
        --config ~/.pie/config.zimage-vae.toml

    # 3. the diff against `pixels.f32`
    python zimage_vae_parity.py compare --out /tmp/zimage-vae

    # or all three
    python zimage_vae_parity.py all     --out /tmp/zimage-vae \
        --config ~/.pie/config.zimage-vae.toml

The golden is `scripts/imagegen/zimage_golden.py --vae`'s dump:
`$PIE_IMAGEGEN_GOLDEN/z-image/zimage_vae/{latent,pixels,mean}.f32` beside a
`shapes.json` that states the boxes. Nothing here imports torch.

THE GATE
--------
`cos >= 0.999` on the pixels, which is the tolerance the contract states for
this reading (§6, "the first real VAE"): the device runs the VAE in bf16 and
the reference in fp32, so the two differ by one rounding per launch and by
nothing else. A tighter gate would be measuring the rounding.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys

DEFAULT_GOLDEN = os.path.join(
    os.environ.get("PIE_IMAGEGEN_GOLDEN", "/root/.cache/pie-imagegen/golden"),
    "z-image",
    "zimage_vae",
)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
INFERLET = os.path.join(REPO, "examples/zimage-vae-parity")

COS_TOL = 0.999
ABS_TOL = 0.05

def f32s(path: str) -> list[float]:
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) % 4:
        raise SystemExit(f"{path}: {len(raw)} bytes is not whole f32")
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))

def shapes(golden: str) -> dict:
    path = os.path.join(golden, "shapes.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"{path}: no golden. Run `python scripts/imagegen/zimage_golden.py --vae` "
            f"first (see scripts/imagegen/README.md)."
        )
    return json.load(open(path))

def case(args) -> str:
    """The golden latent as the inferlet's case JSON.

    The latent dump is `[h, w, channels]` row-major — the same order the
    engine's own parity gate feeds it in, and the same order a `Voxels` port
    channel declares — so this is a copy, not a transpose.
    """
    meta = shapes(args.golden)
    box = meta["latent"]
    latent = f32s(os.path.join(args.golden, "latent.f32"))
    want = box["h"] * box["w"] * box["channels"]
    if len(latent) != want:
        raise SystemExit(
            f"latent.f32 holds {len(latent)} numbers and shapes.json says "
            f"{box['h']}x{box['w']}x{box['channels']} = {want}"
        )
    pixels = meta["pixels"]
    compression = pixels["h"] // box["h"]
    if compression * box["h"] != pixels["h"] or compression * box["w"] != pixels["w"]:
        raise SystemExit(
            f"the golden's boxes are not one compression apart: latent "
            f"{box['h']}x{box['w']}, pixels {pixels['h']}x{pixels['w']}"
        )
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "case.json")
    with open(path, "w") as f:
        json.dump(
            {
                "latent": [round(v, 7) for v in latent],
                "h": box["h"],
                "w": box["w"],
                "channels": box["channels"],
                "compression": compression,
            },
            f,
        )
    print(
        f"[case] {box['h']}x{box['w']}x{box['channels']} latent -> "
        f"{pixels['h']}x{pixels['w']}x{pixels['channels']} pixels ({os.path.getsize(path)} B) "
        f"-> {path}"
    )
    return path

def wasm() -> str:
    """The newest `.wasm` a build left for the fixture, building one first.

    Newest wins, not first: a release artifact built once by hand would
    otherwise shadow every debug rebuild afterwards, silently.
    """
    workspace = os.path.dirname(os.path.normpath(INFERLET))
    if not os.environ.get("PIE_INFERLETS_NO_BUILD"):
        done = subprocess.run(
            ["cargo", "build", "-p", "zimage-vae-parity", "--target", "wasm32-wasip2"],
            cwd=workspace, capture_output=True, text=True,
        )
        if done.returncode != 0:
            sys.stderr.write(done.stderr)
            raise SystemExit("building zimage-vae-parity for wasm32-wasip2 failed")
    candidates = [
        os.path.join(workspace, "target/wasm32-wasip2/release", "zimage_vae_parity.wasm"),
        os.path.join(workspace, "target/wasm32-wasip2/debug", "zimage_vae_parity.wasm"),
    ]
    present = [path for path in candidates if os.path.exists(path)]
    if not present:
        raise SystemExit(f"no wasm; tried {', '.join(candidates)}")
    return max(present, key=os.path.getmtime)

PIECES = 16

def pieces(text: str, n: int = PIECES) -> list[str]:
    """Split `text` into `n` argv pieces, each starting with a comma.

    NOT an even split. `pie run` types an argument by what it looks like and
    treats a token that starts with `-` and does not parse as a number as the
    NEXT FLAG (`src/ops/run.rs::is_flag`) — so a piece that happens to begin
    at a negative number (`-0.31,0.4,...`, which a latent is full of) would
    turn its own `--case_k` into a bare boolean flag and vanish. Every
    boundary is therefore nudged forward to the next `,`, which no rule reads
    as anything but a string.
    """
    step = -(-len(text) // n)
    cuts = [0]
    for i in range(1, n):
        at = min(i * step, len(text))
        comma = text.find(",", at)
        cuts.append(len(text) if comma < 0 else comma)
    cuts.append(len(text))
    out = [text[a:b] for a, b in zip(cuts, cuts[1:]) if b > a]
    assert "".join(out) == text, "the pieces do not reassemble the case"
    for piece in out[1:]:
        assert piece.startswith(","), "a piece starts with something `pie run` may read as a flag"
    return out

def run(args) -> None:
    path = os.path.join(args.out, "case.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path}: no case JSON; run `case` first")
    pie = args.pie or shutil.which("pie") or os.path.join(REPO, "target/debug/pie")
    if not os.path.exists(pie):
        raise SystemExit(
            f"{pie}: no pie binary. Build one with "
            f"`cargo build -p pie --features cuda`, or pass --pie."
        )
    binary = wasm()
    cmd = [pie]
    if args.config:
        cmd += ["--config", os.path.expanduser(args.config)]
    cmd += ["run", "--path", binary]
    if args.png_dir:
        os.makedirs(args.png_dir, exist_ok=True)
        cmd += ["-o", args.png_dir]
    cmd += ["--"]
    parts = pieces(open(path).read())
    for i, piece in enumerate(parts):
        cmd += [f"--case_{i}", piece]
    if args.png:
        cmd += ["--png", args.png]
    if not args.no_pixels:
        cmd += ["--pixels_file", "true"]
    print(f"[run] {' '.join(cmd[:6])} ... ({len(parts)} case pieces)")
    done = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    out = os.path.join(args.out, "pie.json")
    with open(out, "w") as f:
        f.write(done.stdout)
    with open(out[:-5] + ".stderr", "w") as f:
        f.write(done.stderr)
    if done.returncode != 0:
        sys.stderr.write(done.stdout[-4000:])
        sys.stderr.write(done.stderr[-8000:])
        raise SystemExit(f"pie run failed ({done.returncode}); see {out[:-5]}.stderr")
    print(f"[run] -> {out}")

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

def score(got: list[float], want: list[float]) -> tuple[float, float, float]:
    """Cosine, mean |err|, max |err| — the same three the engine gate prints."""
    dot = sum(a * b for a, b in zip(got, want))
    na = sum(a * a for a in got) ** 0.5
    nb = sum(b * b for b in want) ** 0.5
    cos = dot / (na * nb) if na > 0 and nb > 0 else 0.0
    errs = [abs(a - b) for a, b in zip(got, want)]
    return cos, sum(errs) / max(len(errs), 1), max(errs, default=0.0)

def compare(args) -> int:
    doc = document(os.path.join(args.out, "pie.json"))
    meta = shapes(args.golden)
    want = f32s(os.path.join(args.golden, "pixels.f32"))
    got = doc.get("pixels") or []
    if not got and doc.get("pixels_bytes"):
        dump = os.path.join(args.png_dir, "file-0000.bin")
        if not os.path.exists(dump):
            raise SystemExit(
                f"{dump}: the run says it sent {doc['pixels_bytes']} bytes of pixels and "
                f"nothing wrote them; was `--png-dir` the `pie run -o` directory?"
            )
        got = f32s(dump)
    print(
        f"[compare] reading `{doc.get('reading')}` port `{doc.get('port')}`: "
        f"{doc.get('latent_h')}x{doc.get('latent_w')}x{doc.get('latent_channels')} -> "
        f"{doc.get('pixel_h')}x{doc.get('pixel_w')}x{doc.get('pixel_channels')}, "
        f"{doc.get('rows')} rows"
    )
    print(
        f"[compare] pie: mean {doc.get('mean'):+.4f} std {doc.get('std'):.4f} "
        f"min {doc.get('min'):+.4f} max {doc.get('max'):+.4f} "
        f"non-finite {doc.get('non_finite')}"
    )
    if doc.get("png"):
        print(f"[compare] sent as `{doc['png']}` (pie run -o wrote it)")
    if not got:
        print("[compare] the run took the zero-copy road and lifted no rows; "
              "nothing to diff (drop --no-pixels to diff)")
        return 0
    if len(got) != len(want):
        print(
            f"[compare] FAIL: {len(got)} numbers back and the golden holds {len(want)}",
            file=sys.stderr,
        )
        return 1
    box = meta["pixels"]
    if (doc.get("pixel_h"), doc.get("pixel_w")) != (box["h"], box["w"]):
        print(
            f"[compare] FAIL: the answer's box is {doc.get('pixel_h')}x{doc.get('pixel_w')} "
            f"and the golden's is {box['h']}x{box['w']}",
            file=sys.stderr,
        )
        return 1
    cos, mean_err, max_err = score(got, want)
    ok = cos >= COS_TOL and mean_err <= ABS_TOL
    print(
        f"[compare] cos {cos:.6f} (>= {COS_TOL}), mean |err| {mean_err:.5f} "
        f"(<= {ABS_TOL}), max |err| {max_err:.5f}  --  {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["case", "run", "compare", "all"])
    ap.add_argument("--out", default="/tmp/zimage-vae", help="working directory")
    ap.add_argument("--golden", default=DEFAULT_GOLDEN, help="the reference dump")
    ap.add_argument("--config", default=None, help="pie config (its `[model] model` is the artifact)")
    ap.add_argument("--pie", default=None, help="the pie binary")
    ap.add_argument("--png", default="zimage-vae.png",
                    help="send the pixels to the client under this name; empty to skip")
    ap.add_argument("--png-dir", default=None,
                    help="`pie run -o DIR`: where the client writes what it is sent")
    ap.add_argument("--no-pixels", action="store_true",
                    help="take the zero-copy road only (no rows in the answer, nothing to diff)")
    args = ap.parse_args()
    if args.png_dir is None:
        args.png_dir = args.out
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
