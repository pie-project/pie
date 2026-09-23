#!/usr/bin/env python3
"""
ltx2_parity.py -- drive pie's `ltx25-mini` row against the M4 miniature golden.

The reference and its dump come from `ltx2_golden.py --mini` (see README).
This script is the other half: it turns the golden's *inputs* into the case
JSON the `ltx2-parity` inferlet takes, runs it, turns its JSON answer back
into an `.npz` under the golden's own key names, and diffs the two with
`compare.py`.

    # 0. the artifact (one file, already carrying the pipeline's prefixes)
    pie model import /root/.cache/pie-imagegen/golden/ltx25 \
        --sku ltx25-mini-bf16-kv-bf16

    # 1. the case the inferlet reads
    python ltx2_parity.py case --out /tmp/ltx2-parity            # the joint step
    python ltx2_parity.py case --out /tmp/ltx2-parity --refine   # the connectors

    # 2. run it, 3. collect, 4. compare  (or `all`)
    python ltx2_parity.py all --out /tmp/ltx2-parity --config ~/.pie/config.ltx2.toml

THE GUEST'S SHAPE
-----------------
The joint step is FOUR lanes of one group, each on its own pipeline: a
`Video` lane (72 latent rows, three rope coordinates), an `Audio` lane (8
rows, one coordinate), and the two text contexts (`Context` at 4096-scale
and `Reference` at 2048-scale in the flagship; 256 and 128 here). Every lane
carries a timestep cell, the context lanes included -- they modulate their
own rows.

The connector case is two passes, `refine.video` and `refine.audio`, over one
rectangle of packed trunk rows.

POSITIONS
---------
Nothing here recomputes the rope. The golden dumps the coordinates ALREADY
normalised -- `(2 * midpoint / max - 1) * pi/2`, in seconds and pixels -- and
the pie port takes exactly that, because `RopeForm::SplitLadder` multiplies
it by `theta^(f/(F-1))` and nothing else.
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
    os.environ.get("PIE_IMAGEGEN_GOLDEN", "/root/.cache/pie-imagegen/golden"), "ltx25"
)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

TOLERANCES = ["--tol", "0.1", "--rel-tol", "0.02", "--cos-tol", "0.9999"]
PIECES = 12

def config(golden: str) -> dict:
    with open(os.path.join(golden, "ltx2_mini_config.json")) as f:
        return json.load(f)

def numbered(out: str, stem: str, tail: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(stem)}{re.escape(tail)}_(\d+)\.json$")
    found = []
    for name in os.listdir(out):
        match = pattern.match(name)
        if match:
            found.append((int(match.group(1)), os.path.join(out, name)))
    return [path for _, path in sorted(found)]

def suffix(args) -> str:
    return "_refine" if args.refine else ""

def cases(args) -> list[str]:
    dump = np.load(os.path.join(args.golden, "ltx2_mini.npz"))
    os.makedirs(args.out, exist_ok=True)
    written = []
    if args.refine:
        text = dump["mini.conn.in.text"]
        pos = dump["mini.conn.in.positions"]
        for i in range(text.shape[0]):
            case = {
                "kind": "refine",
                "text": text[i].reshape(-1).astype(np.float32).tolist(),
                "text_rows": int(text.shape[1]),
                "text_width": int(text.shape[2]),
                "text_positions": pos.reshape(-1).astype(np.float32).tolist(),
            }
            path = os.path.join(args.out, f"case{suffix(args)}_{i}.json")
            with open(path, "w") as f:
                json.dump(case, f)
            written.append(path)
        print(f"[case] {len(written)} connector case(s), {text.shape[1]} text rows -> {args.out}")
        return written

    x_v = dump["mini.dit.in.latents"]
    x_a = dump["mini.dit.in.audio_latents"]
    ctx = dump["mini.dit.in.context"]
    actx = dump["mini.dit.in.audio_context"]
    pos = dump["mini.dit.in.positions"]
    apos = dump["mini.dit.in.audio_positions"]
    t_v = dump["mini.dit.in.timestep"]
    t_a = dump["mini.dit.in.audio_timestep"]
    for i in range(x_v.shape[0]):
        case = {
            "kind": "denoise",
            "latents": x_v[i].reshape(-1).astype(np.float32).tolist(),
            "rows": int(x_v.shape[1]),
            "channels": int(x_v.shape[2]),
            "audio_latents": x_a[i].reshape(-1).astype(np.float32).tolist(),
            "audio_rows": int(x_a.shape[1]),
            "context": ctx[i].reshape(-1).astype(np.float32).tolist(),
            "context_rows": int(ctx.shape[1]),
            "context_width": int(ctx.shape[2]),
            "audio_context": actx[i].reshape(-1).astype(np.float32).tolist(),
            "audio_context_width": int(actx.shape[2]),
            "positions": pos[i].reshape(-1).astype(np.float32).tolist(),
            "audio_positions": apos[i].reshape(-1).astype(np.float32).tolist(),
            "timestep": float(t_v[i] if t_v.ndim else t_v),
            "audio_timestep": float(t_a[i] if t_a.ndim else t_a),
        }
        path = os.path.join(args.out, f"case{suffix(args)}_{i}.json")
        with open(path, "w") as f:
            json.dump(case, f)
        written.append(path)
    print(
        f"[case] {len(written)} denoise case(s): {x_v.shape[1]} video rows, "
        f"{x_a.shape[1]} audio rows, {ctx.shape[1]} text rows -> {args.out}"
    )
    return written

def split(text: str, pieces: int) -> list[str]:
    """`pieces` roughly equal cuts of `text`, none of them starting with `-`.

    A case is a few hundred KB of floats and goes in as argv; an argv value
    that starts with a minus is read as another FLAG, and the parameter
    before it silently becomes a boolean (`invalid type: boolean true,
    expected a string`). Every cut is therefore nudged forward off a minus
    sign, which a JSON number list has one of every few characters.
    """
    step = -(-len(text) // pieces)
    bounds = [0]
    for i in range(1, pieces):
        at = min(i * step, len(text))
        while at < len(text) and text[at] == "-":
            at += 1
        bounds.append(max(at, bounds[-1]))
    bounds.append(len(text))
    return [text[bounds[i] : bounds[i + 1]] for i in range(pieces)]

def wasm(inferlet: str) -> str:
    """The newest `.wasm` a build left for `inferlet`, building one first."""
    name = os.path.basename(os.path.normpath(inferlet))
    stem = name.replace("-", "_")
    workspace = os.path.dirname(os.path.normpath(inferlet))
    if not os.environ.get("PIE_INFERLETS_NO_BUILD"):
        done = subprocess.run(
            ["cargo", "build", "-p", name, "--target", "wasm32-wasip2"],
            cwd=workspace,
            capture_output=True,
            text=True,
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
    paths = numbered(args.out, "case", suffix(args))
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
        out = os.path.join(args.out, f"pie{suffix(args)}_{b}.json")
        cmd = [pie]
        if args.config:
            cmd += ["--config", args.config]
        cmd += ["run", "--path", binary, "--"]
        text = ""
        if args.case_file:
            cmd += ["--case_file", os.path.basename(case)]
        else:
            text = open(case).read()
            for i, piece in enumerate(split(text, PIECES)):
                cmd += [f"--case_{i}", piece]
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
        print(f"[run] batch {b} -> {out}")

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

def collect(args) -> str:
    dump = np.load(os.path.join(args.golden, "ltx2_mini.npz"))
    paths = numbered(args.out, "pie", suffix(args))
    if not paths:
        raise SystemExit(f"{args.out}: no pie answer; run `run` first")
    docs = [document(path) for path in paths]
    keys = (
        ("mini.conn.out.video", "mini.conn.out.audio")
        if args.refine
        else ("mini.dit.out.velocity", "mini.dit.out.audio_velocity")
    )
    mine, theirs = {}, {}
    for key, side in zip(keys, ("video", "audio")):
        rows = docs[0][f"{side}_rows"]
        width = docs[0][f"{side}_width"]
        mine[key] = np.stack(
            [np.asarray(doc[side], dtype=np.float32).reshape(rows, width) for doc in docs]
        )
        theirs[key] = dump[key][: len(docs)].astype(np.float32)
    path = os.path.join(args.out, f"ltx2_mini_pie{suffix(args)}.npz")
    target = os.path.join(args.out, f"ltx2_mini_target{suffix(args)}.npz")
    np.savez(path, **mine)
    np.savez(target, **theirs)
    shapes = {k: v.shape for k, v in mine.items()}
    print(f"[collect] {shapes} from {len(docs)} case(s) -> {path}; golden -> {target}")
    return path

def one_run(args, case: dict) -> tuple[np.ndarray, np.ndarray]:
    """One pie run of `case`, in memory — the video and audio answers."""
    pie = args.pie or shutil.which("pie") or os.path.join(REPO, "target/debug/pie")
    binary = wasm(args.inferlet)
    cmd = [pie]
    if args.config:
        cmd += ["--config", args.config]
    cmd += ["run", "--path", binary, "--"]
    for i, piece in enumerate(split(json.dumps(case), PIECES)):
        cmd += [f"--case_{i}", piece]
    done = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    if done.returncode != 0:
        sys.stderr.write(done.stderr[-2000:])
        raise SystemExit(f"pie run failed ({done.returncode})")
    path = os.path.join(args.out, "matters.json")
    with open(path, "w") as f:
        f.write(done.stdout)
    doc = document(path)
    v = np.asarray(doc["video"], dtype=np.float32).reshape(doc["video_rows"], -1)
    a = np.asarray(doc["audio"], dtype=np.float32).reshape(doc["audio_rows"], -1)
    return v, a

def matters(args) -> int:
    """**EVERY CONDITIONING STREAM MOVES THE ANSWER.**

    A parity gate can pass on a fixture whose cross-attentions read nothing:
    a lane that never joined the fire's attention group contributes zero, and
    on random-init weights zero is within tolerance. So perturb each
    conditioning stream in turn and demand the velocity move by MORE than the
    gate's own tolerance — the video by its text context and by the audio
    stream (the a2v fold), the audio by its own text context and by the video
    stream (the v2a fold).
    """
    paths = numbered(args.out, "case", "")
    if not paths:
        cases(args)
        paths = numbered(args.out, "case", "")
    base = json.load(open(paths[0]))
    rng = np.random.default_rng(7)

    def jolt(field: str) -> dict:
        case = dict(base)
        case[field] = (
            np.asarray(base[field], dtype=np.float32)
            + rng.normal(0, 1, len(base[field])).astype(np.float32)
        ).tolist()
        return case

    v0, a0 = one_run(args, base)
    floor = 1.0 - float(TOLERANCES[TOLERANCES.index("--cos-tol") + 1])
    print(f"[matters] the gate's own slack is {floor:.1e} of cosine; a stream that matters "
          f"must move its answer by more")
    ok = True
    for field, moves in [
        ("context", ("video",)),
        ("audio_context", ("audio",)),
        ("audio_latents", ("audio", "video")),
        ("latents", ("video", "audio")),
    ]:
        v, a = one_run(args, jolt(field))
        for side, (mine, was) in {"video": (v, v0), "audio": (a, a0)}.items():
            moved = 1.0 - float(
                (mine.ravel() @ was.ravel())
                / (np.linalg.norm(mine) * np.linalg.norm(was))
            )
            want = side in moves
            good = moved > 10 * floor if want else True
            ok = ok and good
            print(
                f"  {field:<14} -> {side:<5} 1 - cos = {moved:.2e}"
                f"{'   MUST MOVE' if want else ''}{'' if good else '   FAILED'}"
            )
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1

def compare(args) -> int:
    mine = os.path.join(args.out, f"ltx2_mini_pie{suffix(args)}.npz")
    theirs = os.path.join(args.out, f"ltx2_mini_target{suffix(args)}.npz")
    for path in (mine, theirs):
        if not os.path.exists(path):
            raise SystemExit(f"{path}: missing; run `collect` first")
    cmd = [
        sys.executable,
        os.path.join(HERE, "compare.py"),
        mine,
        theirs,
        "--sort-by",
        "rel",
        *TOLERANCES,
    ]
    print(f"[compare] {' '.join(cmd)}")
    return subprocess.call(cmd)

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["case", "run", "collect", "compare", "all", "matters"])
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--out", default="/tmp/ltx2-parity")
    ap.add_argument(
        "--refine",
        action="store_true",
        help="the two connector passes instead of the joint denoise step",
    )
    ap.add_argument("--inferlet", default=os.path.join(REPO, "examples/ltx2-parity"))
    ap.add_argument(
        "--config",
        default=None,
        help="the serving config; its `[model] model` must be the imported miniature",
    )
    ap.add_argument("--pie", default=None, help="the pie binary (default: PATH, else target/debug)")
    ap.add_argument(
        "--case_file",
        action="store_true",
        help="pass the case as a scratch file instead of argv pieces",
    )
    args = ap.parse_args()

    if args.cmd == "matters":
        return matters(args)
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
