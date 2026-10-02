#!/usr/bin/env python3
"""
wan22_parity.py -- drive pie's `wan22-*` rows against the M3 golden.

The reference and its dump come from `wan22_golden.py` (see README): the two
miniatures under `--mini`, the real `Wan2.2-TI2V-5B` step-0 forward under
`--full`. This script is the other half: it turns the golden's *inputs* into
the case JSON the `wan2-parity` inferlet takes, runs it, turns its JSON
answer back into an `.npz` under the golden's own key names, and diffs the
two with `compare.py`.

    # 1. the case the inferlet reads (scalar timestep, or TI2V's per-token one)
    python wan22_parity.py case  --out /tmp/wan22-parity
    python wan22_parity.py case  --out /tmp/wan22-parity --pertoken

    # 2. run it (the config's `[model] model` is the imported artifact:
    #    `pie model import <dir with wan22_mini_d128.safetensors> --sku wan22-mini-d128-bf16-kv-bf16`)
    python wan22_parity.py run   --out /tmp/wan22-parity --config ~/.pie/config.wan22-mini.toml

    # 3. the pie-side npz, then the diff
    python wan22_parity.py collect --out /tmp/wan22-parity
    python wan22_parity.py compare --out /tmp/wan22-parity

    # 4. the claim that the PROMPT MATTERS: the same step with the context
    #    zeroed must answer something else (a context lane that is not in the
    #    video lanes' fire conditions nothing, and a miniature's tolerance
    #    will not notice)
    python wan22_parity.py conditioning --out /tmp/wan22-parity --config ...

    # or all of it
    python wan22_parity.py all --out /tmp/wan22-parity [--pertoken]

    # the real row: 1950 rows x 192 features and a 512x4096 context, whose
    # case is 15 MB as JSON numbers and 2.3 MB with `--compact` (base64 f32,
    # and only the context rows the prompt actually filled) -- which is what
    # argv can carry
    python wan22_parity.py all --variant ti2v-5b --compact --out /tmp/wan22-full \
        --config ~/.pie/config.wan22-ti2v.toml

Nothing here imports torch: the reference numbers are already on disk, and
patchify is a reshape.

THE GUEST'S SHAPE
-----------------
One step is a context lane plus one video lane (scalar timestep) or two
video lanes (`--pertoken`: the first latent frame's tokens at timestep 0,
the rest at `t` -- `wan_2`'s per-lane form of TI2V's per-token timestep),
all in one attention group, each on ITS OWN pipeline.

LAYOUTS
-------
Patch rows are `(c, ph, pw)`, the transformer's `patch_embedding` input
order, and the family hands its velocity back in the same order (`proj_out`
is permuted at import), so `unpatchify` here is the inverse of `patchify`.
Token order is `t * (H/2) * (W/2) + h * (W/2) + w`; positions are `(t, h, w)`
in patch units.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import resource
import shutil
import subprocess
import sys

import numpy as np

DEFAULT_GOLDEN = os.path.join(
    os.environ.get("PIE_IMAGEGEN_GOLDEN", "/root/.cache/pie-imagegen/golden"), "wan22"
)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

ROWS = {
    "d128": dict(npz="wan22_mini.npz", stem="mini.d128", out="mini.d128.out.0",
                 out_pertoken="mini.d128.out_pertoken.0", cos="0.9999", move=0.002),
    "nano": dict(npz="wan22_mini.npz", stem="mini.nano", out="mini.nano.out.0",
                 out_pertoken="mini.nano.out_pertoken.0", cos="0.9999", move=0.002),
    "ti2v-5b": dict(npz="wan22_golden.npz", stem="dit.step0", out="dit.step0.out",
                    out_pertoken=None, cos="0.999", move=0.05),
}

PIECE = 120 * 1024
ARGV_CEILING = 3 * 1024 * 1024
CHILD_STACK = 256 * 1024 * 1024

def tolerances(args) -> list[str]:
    return ["--tol", "0.1", "--rel-tol", "0.02", "--cos-tol", ROWS[args.variant]["cos"]]

def b64(a: np.ndarray) -> str:
    """A float rectangle as the guest's `*_b64`: little-endian f32."""
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()

def patchify(latent: np.ndarray, p: int) -> np.ndarray:
    """[B,C,T,H,W] -> [B, T*(H/p)*(W/p), C*p*p], feature order (c, ph, pw)."""
    b, c, t, h, w = latent.shape
    x = latent.reshape(b, c, t, h // p, p, w // p, p)
    x = x.transpose(0, 2, 3, 5, 1, 4, 6)
    return np.ascontiguousarray(x.reshape(b, t * (h // p) * (w // p), c * p * p))

def unpatchify(tokens: np.ndarray, c: int, t: int, h: int, w: int, p: int) -> np.ndarray:
    b = tokens.shape[0]
    x = tokens.reshape(b, t, h // p, w // p, c, p, p)
    x = x.transpose(0, 4, 1, 2, 5, 3, 6)
    return np.ascontiguousarray(x.reshape(b, c, t, h, w))

def positions(t: int, hp: int, wp: int) -> np.ndarray:
    """`(t, h, w)` per token, in token order."""
    tt, hh, ww = np.meshgrid(np.arange(t), np.arange(hp), np.arange(wp), indexing="ij")
    return np.stack([tt, hh, ww], axis=-1).reshape(-1, 3).astype(np.float32)

def suffix(args) -> str:
    return "_pertoken" if args.pertoken else ""

def numbered(out: str, stem: str, tail: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(stem)}{re.escape(tail)}_(\d+)\.json$")
    found = []
    for name in os.listdir(out):
        match = pattern.match(name)
        if match:
            found.append((int(match.group(1)), os.path.join(out, name)))
    return [path for _, path in sorted(found)]

def row(args) -> dict:
    return ROWS[args.variant]

def dump_of(args) -> np.lib.npyio.NpzFile:
    return np.load(os.path.join(args.golden, row(args)["npz"]))

def out_key(args) -> str:
    """The golden key this run's answer stands against."""
    r = row(args)
    if not args.pertoken:
        return r["out"]
    if r["out_pertoken"] is None:
        raise SystemExit(f"`{args.variant}` has no per-token forward in its golden")
    return r["out_pertoken"]

def timesteps(args, dump) -> np.ndarray:
    """The golden's timestep for this run, `[B]` or `[B, S]` per token."""
    r = row(args)
    if args.pertoken:
        return dump[f"{r['stem']}.in.timestep_pertoken"]
    return dump[f"{r['stem']}.in.timestep"]

def lane_cut(pt: np.ndarray) -> tuple[int, float]:
    """A per-token timestep as the two lanes `wan_2` takes: how many leading
    rows are the conditioning frame's (timestep 0), and the rest's timestep."""
    zeros = np.flatnonzero(pt == 0.0)
    cond_rows = int(zeros.size)
    assert np.array_equal(zeros, np.arange(cond_rows)), "the zeros are a prefix"
    rest = pt[cond_rows:]
    assert rest.size and np.all(rest == rest[0]), "one timestep past the prefix"
    return cond_rows, float(rest[0])

def cases(args) -> list[str]:
    """Write one `case[_pertoken]_{b}.json` per batch element; answer their paths."""
    r = row(args)
    dump = dump_of(args)
    hs = dump[f"{r['stem']}.in.hidden_states"]
    ctx = dump[f"{r['stem']}.in.encoder_hidden_states"]
    ts = timesteps(args, dump)
    b, c, t, h, w = hs.shape
    tokens = patchify(hs, 2)
    pos = positions(t, h // 2, w // 2)
    assert tokens.shape[1] == pos.shape[0]

    os.makedirs(args.out, exist_ok=True)
    written = []
    for i in range(b):
        cond_rows = 0
        if ts.ndim == 1:
            timestep = float(ts[i])
        else:
            assert ts.shape[1] == tokens.shape[1], (ts.shape, tokens.shape)
            cond_rows, timestep = lane_cut(ts[i])
        case = {
            "rows": int(tokens.shape[1]),
            "patch_features": int(tokens.shape[2]),
            "cond_rows": cond_rows,
            "context_rows": int(ctx.shape[1]),
            "context_width": int(ctx.shape[2]),
            "timestep": timestep,
        }
        rows = {"latents": tokens[i], "context": ctx[i], "positions": pos}
        if args.zero_context:
            rows["context"] = np.zeros_like(ctx[i])
        if args.compact:
            real = int(np.flatnonzero(np.abs(ctx[i]).sum(-1) > 0).max() + 1)
            rows["context"] = rows["context"][:real]
            case |= {f"{name}_b64": b64(a) for name, a in rows.items()}
        else:
            case |= {name: a.reshape(-1).astype(np.float32).tolist()
                     for name, a in rows.items()}
        path = os.path.join(args.out, f"case{suffix(args)}_{i}.json")
        with open(path, "w") as f:
            json.dump(case, f)
        written.append(path)
    lanes = f"cond {cond_rows} + rest" if cond_rows else "one lane"
    biggest = max(os.path.getsize(path) for path in written)
    print(f"[case] {len(written)} batch element(s), {tokens.shape[1]} rows "
          f"({lanes}), {biggest / (1 << 20):.1f} MiB each -> {args.out}")
    return written

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

def roomy_argv() -> None:
    """The child's argv budget is a quarter of its stack limit; buy room for
    a real row's case. Runs between fork and exec (`preexec_fn`)."""
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    want = CHILD_STACK if hard == resource.RLIM_INFINITY else min(CHILD_STACK, hard)
    if soft == resource.RLIM_INFINITY or soft >= want:
        return
    resource.setrlimit(resource.RLIMIT_STACK, (want, hard))

def scratch_dir(config: str | None) -> str | None:
    """The sandbox scratch a `--case_file` name resolves under, off the config."""
    if not config:
        return None
    with open(os.path.expanduser(config)) as f:
        found = re.search(r'^\s*fs_scratch_dir\s*=\s*"([^"]*)"', f.read(), re.M)
    return found.group(1) if found else None

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
    scratch = scratch_dir(args.config)
    for b, case in enumerate(paths):
        out = os.path.join(args.out, f"pie{suffix(args)}_{b}.json")
        cmd = [pie]
        if args.config:
            cmd += ["--config", args.config]
        cmd += ["run", "--path", binary, "--"]
        text = ""
        if args.case_file:
            if scratch and os.path.abspath(scratch) != os.path.abspath(args.out):
                os.makedirs(scratch, exist_ok=True)
                shutil.copyfile(case, os.path.join(scratch, os.path.basename(case)))
            cmd += ["--case_file", os.path.basename(case)]
        else:
            text = open(case).read()
            if len(text) > ARGV_CEILING:
                raise SystemExit(
                    f"{case}: {len(text)} bytes is past what argv carries; "
                    f"write the case with `--compact`"
                )
            for i in range(-(-len(text) // PIECE)):
                cmd += [f"--case_{i}", text[i * PIECE:(i + 1) * PIECE]]
        print(f"[run] {' '.join(cmd[:8])} ... ({len(text)} bytes of case in "
              f"{-(-len(text) // PIECE)} piece(s))")
        done = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO,
                              preexec_fn=roomy_argv)
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

def answers(args) -> tuple[str, str]:
    """Where this run's answer and the golden it stands against are written."""
    stem = os.path.join(args.out, f"wan22_{args.variant}")
    return f"{stem}_pie{suffix(args)}.npz", f"{stem}_target{suffix(args)}.npz"

def collect(args) -> str:
    dump = dump_of(args)
    hs = dump[f"{row(args)['stem']}.in.hidden_states"]
    _, c, t, h, w = hs.shape
    paths = numbered(args.out, "pie", suffix(args))
    if not paths:
        raise SystemExit(f"{args.out}: no pie answer; run `run` first")
    docs = [document(path) for path in paths]
    rows, width = docs[0]["rows"], docs[0]["patch_features"]
    tokens = np.stack(
        [np.asarray(doc["velocity"], dtype=np.float32).reshape(rows, width) for doc in docs]
    )
    key = out_key(args)
    mine = unpatchify(tokens, c, t, h, w, 2)
    path, target = answers(args)
    np.savez(path, **{key: mine})
    theirs = dump[key][: len(docs)]
    np.savez(target, **{key: theirs.astype(np.float32)})
    print(f"[collect] {mine.shape} from {len(docs)} batch element(s) -> {path}; golden -> {target}")
    return path

def compare(args) -> int:
    mine, theirs = answers(args)
    for path in (mine, theirs):
        if not os.path.exists(path):
            raise SystemExit(f"{path}: missing; run `collect` first")
    cmd = [
        sys.executable, os.path.join(HERE, "compare.py"), mine, theirs,
        "--sort-by", "rel", *tolerances(args),
    ]
    print(f"[compare] {' '.join(cmd)}")
    return subprocess.call(cmd)

def conditioning(args) -> int:
    """THE PROMPT MUST MATTER. Runs the same step twice — once with the
    golden's umT5 context, once with that context zeroed — and demands the
    velocity move by more than the gate could hide.

    A cross-attention context is a lane of its own, and it conditions the
    video rows only if the two lanes are members of ONE fire. Put them on one
    pipeline and the scheduler seats them in different steps: the model then
    attends its video rows alone, reads whatever the context rectangle held,
    and answers something that a miniature's tolerance will happily pass. So
    the harness asks the question directly: drop the prompt, and if the answer
    does not move, the lanes were never in one attention. Lanes seated apart
    read the same context rectangle in both runs, so the move is EXACTLY zero.
    """
    move = args.conditioning_move
    if move is None:
        move = row(args)["move"]
    answers = {}
    for zero in (False, True):
        run_args = argparse.Namespace(**vars(args))
        run_args.zero_context = zero
        run_args.out = os.path.join(args.out, "zeroctx" if zero else "prompt")
        cases(run_args)
        run(run_args)
        paths = numbered(run_args.out, "pie", suffix(args))
        docs = [document(path) for path in paths]
        answers[zero] = np.stack([
            np.asarray(doc["velocity"], dtype=np.float32) for doc in docs
        ])
    with_prompt, without = answers[False], answers[True]
    scale = float(np.linalg.norm(without))
    moved = float(np.linalg.norm(with_prompt - without)) / max(scale, 1e-30)
    verdict = "PASS" if moved > move else "FAIL"
    print(f"[conditioning] dropping the prompt moves the velocity by {moved:.4f} "
          f"(needs > {move}) — {verdict}")
    if verdict == "FAIL":
        print("[conditioning] the context lane is not in the video lanes' fire: "
              "each lane needs its OWN pipeline (a pipeline is serial, and the "
              "scheduler never seats two of its passes in one step).")
        return 1
    return 0

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["case", "run", "collect", "compare",
                                    "conditioning", "all"])
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--out", default="/tmp/wan22-parity")
    ap.add_argument("--variant", choices=sorted(ROWS), default="d128")
    ap.add_argument("--pertoken", action="store_true",
                    help="the miniatures' TI2V per-token-timestep forward: two video lanes")
    ap.add_argument("--inferlet", default=os.path.join(REPO, "examples/wan2-parity"))
    ap.add_argument("--config", default=None,
                    help="the serving config; its `[model] model` must be the imported row")
    ap.add_argument("--pie", default=None, help="the pie binary (default: PATH, else target/debug)")
    ap.add_argument("--compact", action="store_true",
                    help="state the case's rectangles as base64 f32 and give the context "
                         "only its real rows: what a real row needs to fit in argv at all")
    ap.add_argument("--case_file", action="store_true",
                    help="pass the case as a `/scratch` file name instead of argv pieces")
    ap.add_argument("--zero-context", action="store_true",
                    help="write the case with a zeroed umT5 context (see `conditioning`)")
    ap.add_argument("--conditioning-move", type=float, default=None,
                    help="how far dropping the prompt must move the velocity "
                         "(default: the row's own floor)")
    args = ap.parse_args()

    if args.cmd == "case":
        cases(args)
    elif args.cmd == "run":
        run(args)
    elif args.cmd == "collect":
        collect(args)
    elif args.cmd == "compare":
        return compare(args)
    elif args.cmd == "conditioning":
        return conditioning(args)
    else:
        cases(args)
        run(args)
        collect(args)
        gate = compare(args)
        return conditioning(args) or gate

if __name__ == "__main__":
    raise SystemExit(main())
