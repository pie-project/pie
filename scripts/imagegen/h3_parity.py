#!/usr/bin/env python3
"""
h3_parity.py -- drive pie's `minimax-h3-mini` row against the M5 miniature golden.

The reference and its dump come from `h3_golden.py --mini` (see README).
This script is the other half: it turns the golden's *inputs* into the case
JSON the `h3-parity` inferlet takes, runs it, turns its JSON answer back
into an `.npz` under the golden's own key names, and diffs the two with
`compare.py`.

    # 0. the golden (CPU, seconds; no checkpoint, no sglang)
    python h3_golden.py --mini

    # 1. import the miniature it wrote, and point a PRIVATE config at it
    pie model import $PIE_IMAGEGEN_GOLDEN/minimax_h3/h3_mini.safetensors \\
        --sku minimax-h3-mini-bf16-kv-bf16
    #    ~/.pie/config.h3-mini.toml with its own [server] port and
    #    [engine] max_model_len >= packed rows x submit depth

    # 2. the case, the run, the pie-side npz, the diff
    python h3_parity.py case    --out /tmp/h3-parity
    python h3_parity.py run     --out /tmp/h3-parity --config ~/.pie/config.h3-mini.toml
    python h3_parity.py collect --out /tmp/h3-parity
    python h3_parity.py compare --out /tmp/h3-parity

    # or all four
    python h3_parity.py all --out /tmp/h3-parity --config ~/.pie/config.h3-mini.toml

Nothing here imports torch: the reference numbers are already on disk.

THE GUEST'S SHAPE
-----------------
One `refine` pass (the text lane alone) and then one `denoise` step of
FOUR lanes in one attention group -- text, video, audio, reference --
each on its own pipeline. The packed row order is `[text | video | audio |
reference]`, by stream code, which is the order the golden dumps in.

WHAT IS COMPARED
----------------
    mini.out.refined_text   the refine reading's answer vs the reference's
                            `refine_prompt_embeds`
    mini.out.video          the video head's velocity  [video_rows, 96]
    mini.out.audio          the audio head's velocity  [audio_rows, 32]

The golden is fp32 (`h3_golden.py --mini` runs on CPU in float32); pie runs
the same weights cast to bf16 with bf16 activations, so the bar is the
bf16 drift of a two-block trunk -- the same order as mini-dit's and
wan_2's gates (README section 3).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

DEFAULT_GOLDEN = os.path.join(
    os.environ.get("PIE_IMAGEGEN_GOLDEN", "/root/.cache/pie-imagegen/golden"), "minimax_h3"
)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

TOLERANCES = ["--tol", "0.1", "--rel-tol", "0.02", "--cos-tol", "0.9999"]

def config(golden: str) -> dict:
    with open(os.path.join(golden, "h3_mini_config.json")) as f:
        return json.load(f)

def numbered(out: str, stem: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(stem)}_(\d+)\.json$")
    found = []
    for name in os.listdir(out):
        match = pattern.match(name)
        if match:
            found.append((int(match.group(1)), os.path.join(out, name)))
    return [path for _, path in sorted(found)]

def cases(args) -> list[str]:
    cfg = config(args.golden)
    dump = np.load(os.path.join(args.golden, "h3_mini.npz"))
    layout = cfg["layout"]
    order = cfg["order"]
    assert order == ["text", "video", "audio", "reference"], order

    text_hidden = dump["mini.in.text_hidden"]
    refined = dump["mini.in.refined_text"]
    video = dump["mini.in.video_rows"]
    audio = dump["mini.in.audio_rows"]
    reference = dump["mini.in.reference_rows"]
    positions = dump["mini.in.positions"]
    timesteps = dump["mini.in.unique_timesteps"]

    rows = sum(layout[name]["rows"] for name in order)
    assert positions.shape == (rows, 3), (positions.shape, rows)

    case = {
        "text_hidden": text_hidden.reshape(-1).astype(np.float32).tolist(),
        "text_rows": int(text_hidden.shape[0]),
        "text_dim": int(text_hidden.shape[1]),
        "refined_text": refined.reshape(-1).astype(np.float32).tolist(),
        "dim": int(refined.shape[1]),
        "video": video.reshape(-1).astype(np.float32).tolist(),
        "video_rows": int(video.shape[0]),
        "video_features": int(video.shape[1]),
        "audio": audio.reshape(-1).astype(np.float32).tolist(),
        "audio_rows": int(audio.shape[0]),
        "audio_channels": int(audio.shape[1]),
        "reference": reference.reshape(-1).astype(np.float32).tolist(),
        "reference_rows": int(reference.shape[0]),
        "positions": positions.reshape(-1).astype(np.float32).tolist(),
        "timesteps": timesteps.reshape(-1).astype(np.float32).tolist(),
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "case_0.json")
    with open(path, "w") as f:
        json.dump(case, f)
    print(
        f"[case] {rows} packed rows "
        f"(text {case['text_rows']}, video {case['video_rows']}, "
        f"audio {case['audio_rows']}, reference {case['reference_rows']}) -> {path}"
    )
    return [path]

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

def run(args) -> None:
    paths = numbered(args.out, "case")
    if not paths:
        raise SystemExit(f"{args.out}: no case JSON; run `case` first")
    pie = args.pie or shutil.which("pie") or os.path.join(REPO, "target/debug/pie")
    if not os.path.exists(pie):
        raise SystemExit(
            f"{pie}: no pie binary. Build one with "
            f"`cargo build -p pie --features cuda`, or pass --pie."
        )
    binary = wasm(args.inferlet)
    for b, case in enumerate(paths):
        out = os.path.join(args.out, f"pie_{b}.json")
        cmd = [pie]
        if args.config:
            cmd += ["--config", args.config]
        cmd += ["run", "--path", binary, "--"]
        text = ""
        if args.case_file:
            cmd += ["--case_file", os.path.basename(case)]
        else:
            text = open(case).read()
            n = 8
            step = -(-len(text) // n)
            for i in range(n):
                cmd += [f"--case_{i}", text[i * step:(i + 1) * step]]
        print(f"[run] {' '.join(cmd[:8])} ... ({len(text)} bytes of case)")
        done = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        with open(out[:-5] + ".stderr", "w") as f:
            f.write(done.stderr)
        if done.returncode != 0:
            sys.stderr.write(done.stdout)
            sys.stderr.write(done.stderr)
            raise SystemExit(f"pie run failed ({done.returncode})")
        with open(out, "w") as f:
            f.write(done.stdout)
        print(f"[run] case {b} -> {out}")

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

KEYS = {
    "refined": ("mini.out.refined_text", "text_rows", "dim"),
    "velocity": ("mini.out.video", "video_rows", "video_features"),
    "audio": ("mini.out.audio", "audio_rows", "audio_channels"),
}

def collect(args) -> str:
    dump = np.load(os.path.join(args.golden, "h3_mini.npz"))
    paths = numbered(args.out, "pie")
    if not paths:
        raise SystemExit(f"{args.out}: no pie answer; run `run` first")
    doc = document(paths[0])
    mine, theirs = {}, {}
    golden_of = {
        "mini.out.refined_text": "mini.in.refined_text",
        "mini.out.video": "mini.out.video",
        "mini.out.audio": "mini.out.audio",
    }
    for field, (key, rows_key, width_key) in KEYS.items():
        rows, width = int(doc[rows_key]), int(doc[width_key])
        mine[key] = np.asarray(doc[field], dtype=np.float32).reshape(rows, width)
        theirs[key] = dump[golden_of[key]].astype(np.float32)
        assert theirs[key].shape == mine[key].shape, (key, theirs[key].shape, mine[key].shape)
    path = os.path.join(args.out, "h3_mini_pie.npz")
    target = os.path.join(args.out, "h3_mini_target.npz")
    np.savez(path, **mine)
    np.savez(target, **theirs)
    for key in mine:
        a = mine[key].reshape(-1).astype(np.float64)
        b = theirs[key].reshape(-1).astype(np.float64)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        print(f"[collect] {key:<24} {mine[key].shape}  cos {cos:.8f}")
    print(f"[collect] {path}; golden -> {target}")
    return path

def compare(args) -> int:
    mine = os.path.join(args.out, "h3_mini_pie.npz")
    theirs = os.path.join(args.out, "h3_mini_target.npz")
    for path in (mine, theirs):
        if not os.path.exists(path):
            raise SystemExit(f"{path}: missing; run `collect` first")
    cmd = [
        sys.executable, os.path.join(HERE, "compare.py"), mine, theirs,
        "--sort-by", "rel", *TOLERANCES,
    ]
    print(f"[compare] {' '.join(cmd)}")
    return subprocess.call(cmd)

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["case", "run", "collect", "compare", "all"])
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--out", default="/tmp/h3-parity")
    ap.add_argument("--inferlet", default=os.path.join(REPO, "examples/h3-parity"))
    ap.add_argument("--config", default=None,
                    help="the serving config; its `[model] model` must be the imported miniature")
    ap.add_argument("--pie", default=None, help="the pie binary (default: PATH, else target/debug)")
    ap.add_argument("--case_file", action="store_true",
                    help="pass the case as a scratch file instead of argv pieces")
    args = ap.parse_args()

    if args.cmd == "case":
        cases(args)
    elif args.cmd == "run":
        run(args)
    elif args.cmd == "collect":
        collect(args)
    elif args.cmd == "compare":
        return compare(args)
    else:
        cases(args)
        run(args)
        collect(args)
        return compare(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
