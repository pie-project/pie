#!/usr/bin/env python3
"""
flux2_parity.py -- drive pie's `flux2-mini` row against the M2 miniature golden.

The reference and its dump come from `flux2_golden.py --mini` (see README).
This script is the other half: it turns the golden's *inputs* into the case
JSON the `flux2-parity` inferlet takes, runs it, turns its JSON answer back
into an `.npz` under the golden's own key names, and diffs the two with
`compare.py`.

    # 1. the case the inferlet reads
    python flux2_parity.py case  --out /tmp/flux2-parity

    # 2. run it (the config's `[model] source` is the imported artifact:
    #    `pie model import $PIE_IMAGEGEN_GOLDEN/flux2/ --deployment flux2-mini-bf16-kv-bf16`)
    python flux2_parity.py run   --out /tmp/flux2-parity --config ~/.pie/config.flux2-mini.toml

    # 3. the pie-side npz, then the diff
    python flux2_parity.py collect --out /tmp/flux2-parity
    python flux2_parity.py compare --out /tmp/flux2-parity

    # or all four
    python flux2_parity.py all --out /tmp/flux2-parity

Nothing here imports torch: the reference numbers are already on disk.

WHAT IS COMPARED
----------------
The golden is ONE transformer call at `timestep = 0.5`, `guidance = 4.0`,
over `[target (64) ‖ ref_0 (64) ‖ ref_1 (64)]` image tokens and 32 text rows,
`mini.out.0` being the prediction for all 192 image rows. pie's head runs on
the target lane alone (the reference discards `pred[:, S_img:]`), so the diff
is over `mini.out.0[:, :64]`, which `collect` writes beside pie's answer as
`flux2_mini_target.npz` under the same key.

THE LATENT IS THE VAE'S
-----------------------
The golden holds packed tokens (`[S, C*4]`, a 2x2 block of VAE cells in
`(c, ph, pw)` order, rows row-major over the token grid its ids state). The
guest takes the VAE's own `[2h, 2w, C]` and reads the velocity out the same
way, so `case` unpacks each picture (the target, and each reference split
off by its own `T` id) and `collect` packs the answer back. No ids cross:
the model computes every position, a reference's from its ordinal.
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
    os.environ.get("PIE_IMAGEGEN_GOLDEN", "/root/.cache/pie-imagegen/golden"), "flux2"
)
DEFAULT_DEPLOYMENT = "flux2-mini-bf16-kv-bf16"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

TOLERANCES = ["--tol", "0.1", "--rel-tol", "0.02", "--cos-tol", "0.9999"]

def config(golden: str) -> dict:
    with open(os.path.join(golden, "flux2_mini_config.json")) as f:
        return json.load(f)

def numbered(out: str, stem: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(stem)}_(\d+)\.json$")
    found = []
    for name in os.listdir(out):
        match = pattern.match(name)
        if match:
            found.append((int(match.group(1)), os.path.join(out, name)))
    return [path for _, path in sorted(found)]

PACK = 2

def pictures(ids: np.ndarray) -> list[tuple[int, int, int, int]]:
    """The pictures `ids` lays out, in row order: `(first row, rows, h, w)`
    for each run of one `T`, its rows a row-major `h x w` token grid."""
    out, start = [], 0
    while start < len(ids):
        t = ids[start, 0]
        end = start
        while end < len(ids) and ids[end, 0] == t:
            end += 1
        run = ids[start:end]
        h, w = int(run[:, 1].max()) + 1, int(run[:, 2].max()) + 1
        grid = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing="ij"), -1).reshape(-1, 2)
        assert np.array_equal(run[:, 1:3].astype(np.int64), grid), "ids are not a row-major grid"
        out.append((start, end - start, h, w))
        start = end
    return out

def to_voxels(tokens: np.ndarray, h: int, w: int) -> np.ndarray:
    """`[h*w, C*4]` packed tokens -> the VAE latent `[2h, 2w, C]`."""
    c = tokens.shape[-1] // (PACK * PACK)
    x = tokens.reshape(h, w, c, PACK, PACK).transpose(0, 3, 1, 4, 2)
    return np.ascontiguousarray(x.reshape(h * PACK, w * PACK, c))

def to_tokens(voxels: np.ndarray, h: int, w: int) -> np.ndarray:
    """The VAE latent's `2h*2w*C` values -> `[h*w, C*4]` packed tokens."""
    c = voxels.size // (h * PACK * w * PACK)
    x = voxels.reshape(h, PACK, w, PACK, c).transpose(0, 2, 4, 1, 3)
    return np.ascontiguousarray(x.reshape(h * w, c * PACK * PACK))

def latent(tokens: np.ndarray, h: int, w: int) -> dict:
    v = to_voxels(tokens, h, w)
    return {"latents": v.reshape(-1).astype(np.float32).tolist(),
            "height": int(v.shape[0]), "width": int(v.shape[1])}

def cases(args) -> list[str]:
    """Write one `case_{b}.json` per batch element; answer their paths."""
    cfg = config(args.golden)
    dump = np.load(os.path.join(args.golden, "flux2_mini.npz"))
    h, w = cfg["img_hw"]
    s_img = h * w
    hs = dump["mini.in.hidden_states"]
    img_ids = dump["mini.in.img_ids"]
    txt_ids = dump["mini.in.txt_ids"]
    ctx = dump["mini.in.encoder_hidden_states"]
    timestep = dump["mini.in.timestep"]
    guidance = dump["mini.in.guidance"]
    refs = int(dump["mini.in.num_ref_tokens"])
    assert hs.shape[1] == s_img + refs, (hs.shape, s_img, refs)
    assert cfg["config"]["guidance_embeds"], "the miniature carries a guidance embedder"

    os.makedirs(args.out, exist_ok=True)
    written = []
    for b in range(hs.shape[0]):
        laid = pictures(img_ids[b])
        (first, rows, th, tw), refs_laid = laid[0], laid[1:]
        assert (first, rows, th * tw) == (0, s_img, s_img), laid
        # Reference `k` sits at `T = 10 (k + 1)`: the order the model's
        # ordinal input recomputes.
        assert [int(img_ids[b, at, 0]) for at, *_ in refs_laid] == \
            [10 * (k + 1) for k in range(len(refs_laid))], refs_laid
        assert np.array_equal(txt_ids[b][:, 3], np.arange(ctx.shape[1])), "text ids are (0, 0, 0, j)"
        case = {
            "target": latent(hs[b, :s_img], th, tw),
            "channels": int(hs.shape[2]) // (PACK * PACK),
            "references": [latent(hs[b, at:at + n], h, w) for at, n, h, w in refs_laid],
            "context": ctx[b].reshape(-1).astype(np.float32).tolist(),
            "text_rows": int(ctx.shape[1]),
            "context_width": int(ctx.shape[2]),
            "timestep": float(timestep[b]) * 1000.0,
            "guidance": float(guidance[b]),
        }
        path = os.path.join(args.out, f"case_{b}.json")
        with open(path, "w") as f:
            json.dump(case, f)
        written.append(path)
    print(f"[case] {len(written)} batch element(s), {s_img} target + {refs} reference rows -> {args.out}")
    return written

def wasm(inferlet: str) -> str:
    """The newest `.wasm` a build left for `inferlet`, building one first."""
    name = os.path.basename(os.path.normpath(inferlet))
    stem = name.replace("-", "_")
    workspace = os.path.dirname(os.path.normpath(inferlet))
    if not os.environ.get("PIE_INFERLETS_NO_BUILD"):
        done = subprocess.run(
            ["cargo", "build", "-p", name, "--target", "wasm32-wasip3"],
            cwd=workspace, capture_output=True, text=True,
        )
        if done.returncode != 0:
            sys.stderr.write(done.stderr)
            raise SystemExit(f"building {name} for wasm32-wasip3 failed")
    candidates = [
        os.path.join(workspace, "target/wasm32-wasip3/release", f"{stem}.wasm"),
        os.path.join(workspace, "target/wasm32-wasip3/debug", f"{stem}.wasm"),
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
        if args.case_file:
            cmd += ["--case_file", os.path.basename(case)]
        else:
            text = open(case).read()
            n = 8
            step = -(-len(text) // n)
            for i in range(n):
                cmd += [f"--case_{i}", text[i * step:(i + 1) * step]]
        print(f"[run] {' '.join(cmd[:8])} ... ({len(text) if not args.case_file else 0} bytes of case)")
        done = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
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
    paths = numbered(args.out, "pie")
    if not paths:
        raise SystemExit(f"{args.out}: no pie answer; run `run` first")
    docs = [document(path) for path in paths]
    h, w = docs[0]["height"] // PACK, docs[0]["width"] // PACK
    rows = h * w
    mine = np.stack(
        [to_tokens(np.asarray(doc["velocity"], dtype=np.float32), h, w) for doc in docs]
    )
    path = os.path.join(args.out, "flux2_mini_pie.npz")
    np.savez(path, **{"mini.out.0": mine})

    dump = np.load(os.path.join(args.golden, "flux2_mini.npz"))
    theirs = dump["mini.out.0"][:, :rows]
    target = os.path.join(args.out, "flux2_mini_target.npz")
    np.savez(target, **{"mini.out.0": theirs.astype(np.float32)})
    print(f"[collect] {mine.shape} from {len(docs)} batch element(s) -> {path}; golden target rows -> {target}")
    return path

def compare(args) -> int:
    mine = os.path.join(args.out, "flux2_mini_pie.npz")
    theirs = os.path.join(args.out, "flux2_mini_target.npz")
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
    ap.add_argument("--out", default="/tmp/flux2-parity")
    ap.add_argument("--inferlet", default=os.path.join(REPO, "examples/flux2-parity"))
    ap.add_argument("--config", default=None,
                    help=f"the serving config; its `[model] source` must be the artifact "
                         f"`{DEFAULT_DEPLOYMENT}` imported")
    ap.add_argument("--pie", default=None, help="the pie binary (default: PATH, else target/debug)")
    ap.add_argument("--case_file", action="store_true", help="pass the case as a scratch file instead of argv pieces")
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
