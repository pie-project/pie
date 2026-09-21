#!/usr/bin/env python3
"""
mini_dit_parity.py -- drive pie's `mini-dit` row against the M0 golden.

The reference and its dumps come from `mini_dit_ref.py` (see README §3).  This
script is the other half: it turns the golden's *inputs* into the case JSON the
`mini-dit-parity` inferlet takes, runs it, turns its JSON answer back into an
`.npz` with the golden's own key names, and diffs the two with `compare.py`.

    # 1. the case the inferlet reads (one file per batch element)
    python mini_dit_parity.py case  --out /tmp/mini-dit-parity
    python mini_dit_parity.py case  --out /tmp/mini-dit-parity --euler

    # 2. run it (the config's `[model] model` is the imported artifact; the
    #    case files must be reachable as /scratch/<name> inside the sandbox)
    python mini_dit_parity.py run   --out /tmp/mini-dit-parity --config ~/.pie/config.toml

    # 3. the pie-side npz, then the diff
    python mini_dit_parity.py collect --out /tmp/mini-dit-parity
    python mini_dit_parity.py compare --out /tmp/mini-dit-parity

    # or all four
    python mini_dit_parity.py all --out /tmp/mini-dit-parity

Nothing here imports torch: the reference numbers are already on disk, and
patchify is a reshape.

THE GUEST'S SHAPE
-----------------
One step is three passes (caption, image, context) in one attention group,
each on ITS OWN pipeline: the scheduler seals a frame from every live
pipeline and never seats two passes of one pipeline in one step, so three
passes down one pipeline would be three fires, each lane attending alone.
The runtime holds a fresh group's first frame for its cohort (`FireRequest::
cohort`), so the three seal together from the first step -- and keeps holding
it: a stated cohort is sealed whole or the request dies by name, never as a
fire over whichever lanes were quick enough.

BISECTION
---------
`--tap <dump key>` (e.g. `b0.attn_out`) re-imports nothing: the family reads
`PIE_MINI_DIT_TAP` at IMPORT time and plants the velocity export on that
intermediate, so re-import the artifact under the env first, then `run
--tap KEY`, `collect --tap KEY`, `compare --tap KEY` diff that one tensor.
`pie_<b>.stderr` is kept beside each answer.
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
    os.environ.get("PIE_IMAGEGEN_GOLDEN", "/root/.cache/pie-imagegen/golden"), "mini-dit"
)
DEFAULT_SKU = "mini-dit-bf16-kv-bf16"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

TOLERANCES = ["--tol", "0.1", "--rel-tol", "0.02", "--cos-tol", "0.9999"]

def patchify(latent: np.ndarray, p: int) -> np.ndarray:
    """[B,C,H,W] -> [B, (H/p)*(W/p), C*p*p], feature order (c, ph, pw)."""
    b, c, hs, ws = latent.shape
    x = latent.reshape(b, c, hs // p, p, ws // p, p)
    x = x.transpose(0, 2, 4, 1, 3, 5)
    return np.ascontiguousarray(x.reshape(b, (hs // p) * (ws // p), c * p * p))

def unpatchify(tokens: np.ndarray, c: int, hs: int, ws: int, p: int) -> np.ndarray:
    b = tokens.shape[0]
    x = tokens.reshape(b, hs // p, ws // p, c, p, p)
    x = x.transpose(0, 3, 1, 4, 2, 5)
    return np.ascontiguousarray(x.reshape(b, c, hs, ws))

def loop_of(args) -> str | None:
    """The loop a run integrates: `euler`, a `--solver`, or none (one step)."""
    return getattr(args, "solver", None) or ("euler" if args.euler else None)


def numbered(out: str, stem: str, euler) -> list[str]:
    """`<out>/<stem>[_euler]_<b>.json` for every batch element `b`, in order.

    Matched by pattern and not by prefix: `pie_euler_0.json` starts with
    `pie_`, so a prefix test folds the four-step run's files into the
    one-step run's and stacks a batch of four against a golden of two.
    """
    suffix = f"_{euler}" if isinstance(euler, str) else ("_euler" if euler else "")
    pattern = re.compile(rf"^{re.escape(stem)}{suffix}_(\d+)\.json$")
    found = []
    for name in os.listdir(out):
        match = pattern.match(name)
        if match:
            found.append((int(match.group(1)), os.path.join(out, name)))
    return [path for _, path in sorted(found)]

def config(golden: str) -> dict:
    with open(os.path.join(golden, "config.json")) as f:
        return json.load(f)

def cases(args) -> list[str]:
    """Write one `case_{b}.json` per batch element; answer their paths."""
    cfg = config(args.golden)
    dump = np.load(os.path.join(args.golden, f"mini_dit_dump_{args.policy}.npz"))
    p = cfg["patch_size"]
    c, hs, ws = cfg["latent_shape"]

    text = dump["in.text"]
    context = dump["in.context"]
    timestep = dump["in.timestep"]
    txt_pos = dump["in.txt_pos"]
    img_pos = dump["in.img_pos"]
    patches = patchify(dump["in.latent"], p)
    assert np.array_equal(patches, dump["patches"]), "patchify disagrees with the reference"

    if args.euler:
        euler = np.load(os.path.join(args.golden, f"mini_dit_euler_{args.policy}.npz"))
        latents = patchify(euler["euler.x_init"], p)
        sigmas = euler["euler.sigmas"].tolist()
    else:
        latents = patches
        sigmas = []

    os.makedirs(args.out, exist_ok=True)
    written = []
    for b in range(latents.shape[0]):
        case = {
            "latents": latents[b].reshape(-1).astype(np.float32).tolist(),
            "image_rows": int(latents.shape[1]),
            "patch_features": int(latents.shape[2]),
            "text": text[b].reshape(-1).astype(np.float32).tolist(),
            "text_rows": int(text.shape[1]),
            "text_width": int(text.shape[2]),
            "context": context[b].reshape(-1).astype(np.float32).tolist(),
            "context_rows": int(context.shape[1]),
            "context_width": int(context.shape[2]),
            "text_positions": txt_pos.reshape(-1).astype(np.float32).tolist(),
            "image_positions": img_pos.reshape(-1).astype(np.float32).tolist(),
            "timestep": float(timestep[b]),
            "sigmas": [float(s) for s in sigmas],
            "t_scale": float(cfg["euler_t_scale"]),
            "steps": int(cfg["euler_steps"]) if args.euler else 0,
        }
        path = os.path.join(args.out, f"case{'_' + loop_of(args) if loop_of(args) else ''}_{b}.json")
        with open(path, "w") as f:
            json.dump(case, f)
        written.append(path)
    print(f"[case] {len(written)} batch element(s), {latents.shape[1]} image rows -> {args.out}")
    return written

def wasm(inferlet: str) -> str:
    """The newest `.wasm` a build left for `inferlet`, building one first.

    Newest wins, not first, for `tests/examples/conftest.py`'s reason: a
    release artifact built once by hand would otherwise shadow every debug
    rebuild afterwards, silently.
    """
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
        os.path.join(inferlet, "target/wasm32-wasip2/release", f"{stem}.wasm"),
        os.path.join(inferlet, "target/wasm32-wasip2/debug", f"{stem}.wasm"),
    ]
    present = [path for path in candidates if os.path.exists(path)]
    if not present:
        raise SystemExit(f"no wasm for {name}; tried {', '.join(candidates)}")
    return max(present, key=os.path.getmtime)

def run(args) -> None:
    paths = numbered(args.out, "case", loop_of(args))
    if not paths:
        raise SystemExit(f"{args.out}: no case JSON; run `case` first")
    pie = args.pie or shutil.which("pie") or os.path.join(REPO, "target/debug/pie")
    if not os.path.exists(pie):
        raise SystemExit(
            f"{pie}: no pie binary. Build one with "
            f"`cargo build -p pie --features cuda`, or pass --pie."
        )
    binary = wasm(args.inferlet)
    manifest = os.path.join(args.inferlet, "Pie.toml")
    for b, case in enumerate(paths):
        out = os.path.join(args.out, f"pie{'_' + loop_of(args) if loop_of(args) else ''}_{b}.json")
        cmd = [pie]
        if args.config:
            cmd += ["--config", args.config]
        cmd += ["run", "--path", binary, "--manifest", manifest, "--"]
        if args.case_file:
            cmd += ["--case_file", os.path.basename(case)]
        else:
            text = open(case).read()
            n = 8
            step = -(-len(text) // n)
            for i in range(n):
                cmd += [f"--case_{i}", text[i * step:(i + 1) * step]]
        if args.solver:
            cmd += ["--solver", args.solver]
        elif args.euler:
            cmd += ["--euler", "true"]
        if getattr(args, "cfg", False):
            cmd += ["--cfg", "true", "--cfg_scale", str(args.cfg_scale)]
        if getattr(args, "classes", None):
            cmd += ["--classes", args.classes]
        if getattr(args, "sweep", None):
            cmd += ["--sweep", str(args.sweep)]
        print(f"[run] {' '.join(cmd)}")
        done = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        if done.returncode != 0:
            sys.stderr.write(done.stdout)
            sys.stderr.write(done.stderr)
            raise SystemExit(
                f"pie run failed ({done.returncode}). If it refused an op by name, the "
                f"engine arms for that op are not wired yet — see this file's header."
            )
        with open(out, "w") as f:
            f.write(done.stdout)
        with open(out[:-5] + ".stderr", "w") as f:
            f.write(done.stderr)
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

def classes(args) -> int:
    """A STRUCTURED ATTENTION MASK, STATED BY THE GUEST — does it mask, and
    only what it says? One step in four modes of `--classes`, all 3-lane
    fires of one shape, so every claim is bit-for-bit within its own kind:

      all vs none    ->  two classes whose table lets everything see
                         everything must land the plain step EXACTLY: the
                         class arm of the kernel adds a predicate and no
                         arithmetic.
      split vs none  ->  an identity table over a left/right split of the
                         image grid and a first/last split of the text rows
                         must MOVE the velocity: the mask does something.
      split-swap     ->  negating the class-1 text rows must leave every
                         class-0 (left-half) image row EXACTLY where `split`
                         put it, and move the class-1 rows: rows see only
                         their own class, through every block, and the
                         context lane (class -1) stays visible to both.
    """
    bad = 0
    answers = {}
    for mode in ("none", "all", "split", "split-swap"):
        run_args = argparse.Namespace(**vars(args))
        run_args.classes = mode
        run_args.euler = False
        run_args.solver = None
        run_args.out = os.path.join(args.out, mode)
        cases(run_args)
        run(run_args)
        answers[mode] = [document(path) for path in numbered(run_args.out, "pie", False)]
    batches = len(answers["none"])
    if not batches:
        print("[classes] FAIL the guest published nothing")
        return 1
    for at in range(batches):
        v = {mode: np.asarray(answers[mode][at]["velocity"], dtype=np.float32) for mode in answers}
        rows = int(answers["none"][at]["image_rows"])
        width = int(answers["none"][at]["patch_features"])
        side = int(round(rows ** 0.5))
        left = np.array([(i % side) < side // 2 for i in range(rows)])
        rel = lambda a, b: float(np.linalg.norm(a - b)) / max(float(np.linalg.norm(b)), 1e-30)
        if np.array_equal(v["all"], v["none"]):
            print(f"[classes] batch {at}: PASS an all-ones table is the plain step, exactly")
        else:
            print(f"[classes] batch {at}: FAIL an all-ones table moved the step by rel "
                  f"{rel(v['all'], v['none']):.3g}")
            bad += 1
        moved = rel(v["split"], v["none"])
        if moved > 1e-3:
            print(f"[classes] batch {at}: PASS the split table moves the velocity by rel {moved:.4f}")
        else:
            print(f"[classes] batch {at}: FAIL the split table left the velocity at rel {moved:.3g}")
            bad += 1
        s = v["split"].reshape(rows, width)
        w = v["split-swap"].reshape(rows, width)
        if np.array_equal(s[left], w[left]):
            print(f"[classes] batch {at}: PASS class-0 image rows ignore the class-1 text, exactly")
        else:
            print(f"[classes] batch {at}: FAIL class-0 image rows moved by rel "
                  f"{rel(w[left], s[left]):.3g} when only class-1 text changed")
            bad += 1
        right = rel(w[~left], s[~left])
        if right > 1e-3:
            print(f"[classes] batch {at}: PASS class-1 image rows follow the class-1 text (rel {right:.4f})")
        else:
            print(f"[classes] batch {at}: FAIL class-1 image rows ignored their own text (rel {right:.3g})")
            bad += 1
    if bad:
        print(f"[classes] FAIL {bad} claim(s) failed")
        return 1
    print("[classes] PASS")
    return 0


def sweep(args) -> int:
    """N REQUESTS IN ONE FIRE. `--sweep N` submits N attention groups (3N
    lanes) together, each group on pipelines of its own, and reads N
    velocities back. Two claims, no golden:

      g1 == g0 exactly  ->  group 1 repeats group 0's inputs. Bit-for-bit
                            equality is only possible inside ONE fire: a
                            lane's bf16 answer depends on the fire's
                            composition (documented: rel 0.0056 between a
                            six-lane and a three-lane fire), so this is the
                            proof the runtime packed the groups together.
      g2 != g0          ->  group 2 scaled its latent, so its velocity
                            must move: the groups are independent lanes,
                            not one lane read N times.

    The lone 3-lane step is run too and its distance from g0 is REPORTED
    (expected small and non-zero: same math, another fire's accumulation).
    """
    n = args.sweep or 4
    run_args = argparse.Namespace(**vars(args))
    run_args.sweep = n
    run_args.euler = False
    run_args.solver = None
    run_args.out = os.path.join(args.out, f"sweep{n}")
    cases(run_args)
    run(run_args)
    swept = [document(path) for path in numbered(run_args.out, "pie", False)]
    lone_args = argparse.Namespace(**vars(args))
    lone_args.sweep = None
    lone_args.euler = False
    lone_args.solver = None
    lone_args.out = os.path.join(args.out, "lone")
    cases(lone_args)
    run(lone_args)
    lone = [document(path) for path in numbered(lone_args.out, "pie", False)]
    if not swept:
        print("[sweep] FAIL the guest published nothing")
        return 1
    bad = 0
    rel = lambda a, b: float(np.linalg.norm(a - b)) / max(float(np.linalg.norm(b)), 1e-30)
    for at, doc in enumerate(swept):
        vs = [np.asarray(v, dtype=np.float32) for v in doc["sweep_velocity"]]
        if len(vs) != n:
            print(f"[sweep] batch {at}: FAIL {len(vs)} velocities for {n} groups")
            bad += 1
            continue
        if np.array_equal(vs[0], vs[1]):
            print(f"[sweep] batch {at}: PASS group 1 repeats group 0 exactly — one fire")
        else:
            print(f"[sweep] batch {at}: FAIL group 1 differs from group 0 by rel "
                  f"{rel(vs[1], vs[0]):.3g}; the groups did not share a fire")
            bad += 1
        moved = rel(vs[2], vs[0]) if n > 2 else 1.0
        if moved > 1e-3:
            print(f"[sweep] batch {at}: PASS group 2 is its own request (rel {moved:.4f})")
        else:
            print(f"[sweep] batch {at}: FAIL group 2 landed group 0's velocity (rel {moved:.3g})")
            bad += 1
        if at < len(lone):
            alone = np.asarray(lone[at]["velocity"], dtype=np.float32)
            print(f"[sweep] batch {at}: lone 3-lane fire vs group 0 of the {3 * n}-lane fire: "
                  f"rel {rel(vs[0], alone):.3g} (reported)")
    if bad:
        print(f"[sweep] FAIL {bad} claim(s) failed")
        return 1
    print("[sweep] PASS")
    return 0


def guidance(args) -> int:
    """CLASSIFIER-FREE GUIDANCE, ON THE DEVICE — does it run, and is it right?

    Six lanes, two attention groups, one fire: group 0 holds the case's
    context, group 1 a zeroed one, and group 0's image lane names group 1 as
    its peer, so its epilogue holds BOTH branches' velocities and combines
    them. That combine is what every guest did on the host until now.

    Checked with three claims, all WITHIN one fire's own answers and needing
    no golden. Same-fire matters: a six-lane fire and a three-lane one give
    the same lane slightly different bf16 answers (measured: rel 0.0056),
    because the composition differs and so does the accumulation order — so
    an identity that compares across the two is not an identity at all, and
    an earlier version of this check failed intermittently for exactly that
    reason.

      s = 0  ->  the combine IS the unconditional branch, exactly. That
                 branch publishes its own velocity from the SAME fire, so
                 this is bit-for-bit. It is the claim that proves the peer
                 bind pointed at the OTHER group and not at its own rows —
                 the one failure a plausible-looking picture would hide.
      s = 1 vs 0  ->  the two must differ by a real margin. Without this the
                 other two claims hold trivially: a lane guided by itself
                 answers the same thing at every scale.
      s = 2  ->  the combine is AFFINE in s: `g(2) = 2 g(1) - g(0)`, since
                 `g(s) = u + s(c - u)`. Proves the arithmetic, and every
                 term comes from a fire of the same shape.
    """
    bad = 0
    answers = {}
    for scale in (0.0, 1.0, 2.0):
        run_args = argparse.Namespace(**vars(args))
        run_args.cfg = True
        run_args.cfg_scale = scale
        run_args.euler = False
        run_args.out = os.path.join(args.out, f"s{scale:g}")
        cases(run_args)
        run(run_args)
        answers[scale] = [document(path) for path in numbered(run_args.out, "pie", False)]

    batches = len(answers[0.0])
    if not batches:
        print("[guidance] FAIL the guest published nothing")
        return 1

    for at in range(batches):
        g0 = np.asarray(answers[0.0][at]["cfg_guided"], dtype=np.float32)
        g1 = np.asarray(answers[1.0][at]["cfg_guided"], dtype=np.float32)
        g2 = np.asarray(answers[2.0][at]["cfg_guided"], dtype=np.float32)
        unc = np.asarray(answers[0.0][at]["cfg_uncond"], dtype=np.float32)
        if min(g0.size, g1.size, g2.size, unc.size) == 0:
            print(f"[guidance] batch {at}: FAIL the guest published no velocities")
            bad += 1
            continue

        if np.array_equal(g0, unc):
            print(f"[guidance] batch {at}: PASS s=0 the combine is the unconditional "
                  f"branch, exactly — the peer is the other group")
        else:
            off = float(np.linalg.norm(g0 - unc)) / max(float(np.linalg.norm(unc)), 1e-30)
            print(f"[guidance] batch {at}: FAIL s=0 the combine differs from the "
                  f"unconditional branch of its OWN fire by rel {off:.3g}")
            bad += 1

        spread = float(np.linalg.norm(g1 - g0)) / max(float(np.linalg.norm(g0)), 1e-30)
        if spread > 1e-3:
            print(f"[guidance] batch {at}: PASS guidance moves the velocity by rel "
                  f"{spread:.4f} between s=0 and s=1")
        else:
            print(f"[guidance] batch {at}: FAIL s=0 and s=1 agree to rel {spread:.3g}; "
                  f"the branches never conditioned separately, so every other claim "
                  f"here holds trivially")
            bad += 1

        want = 2.0 * g1 - g0
        off = float(np.linalg.norm(g2 - want)) / max(float(np.linalg.norm(want)), 1e-30)
        if off < 1e-5:
            print(f"[guidance] batch {at}: PASS s=2 is 2*s1 - s0 to rel {off:.3g} — "
                  f"the combine is affine in the scale")
        else:
            print(f"[guidance] batch {at}: FAIL s=2 is not 2*s1 - s0: rel {off:.3g}")
            bad += 1
    return 1 if bad else 0

def collect(args) -> str:
    cfg = config(args.golden)
    p = cfg["patch_size"]
    c, hs, ws = cfg["latent_shape"]
    paths = numbered(args.out, "pie", loop_of(args))
    if not paths:
        raise SystemExit(f"{args.out}: no pie answer; run `run` first")
    docs = [document(path) for path in paths]
    rows, width = docs[0]["image_rows"], docs[0]["patch_features"]

    def stack(key: str, at: int | None = None) -> np.ndarray:
        tokens = np.stack(
            [
                np.asarray(doc[key] if at is None else doc[key][at], dtype=np.float32)
                .reshape(rows, width)
                for doc in docs
            ]
        )
        return unpatchify(tokens, c, hs, ws, p)

    out: dict[str, np.ndarray] = {}
    if loop_of(args):
        name = loop_of(args)
        steps = len(docs[0]["euler_v"])
        for i in range(steps):
            out[f"{name}.v{i}"] = stack("euler_v", i)
            out[f"{name}.x{i + 1}"] = stack("euler_x", i)
        out[f"{name}.latent"] = out[f"{name}.x{steps}"]
    elif args.tap:
        planes = [np.asarray(doc["velocity"], dtype=np.float32) for doc in docs]
        tw = docs[0].get("velocity_width", width)
        out[args.tap] = np.stack([plane.reshape(-1, tw) for plane in planes]).squeeze()
    else:
        out["velocity"] = stack("velocity")
        out["final.tokens"] = np.stack(
            [np.asarray(doc["velocity"], dtype=np.float32).reshape(rows, width) for doc in docs]
        )

    path = os.path.join(args.out, f"mini_dit_pie{'_' + loop_of(args) if loop_of(args) else ''}.npz")
    np.savez(path, **out)
    print(f"[collect] {len(out)} tensors from {len(docs)} batch element(s) -> {path}")
    return path

def compare(args) -> int:
    mine = os.path.join(args.out, f"mini_dit_pie{'_' + loop_of(args) if loop_of(args) else ''}.npz")
    if not os.path.exists(mine):
        raise SystemExit(f"{mine}: no pie npz; run `collect` first")
    theirs = os.path.join(
        args.golden,
        f"mini_dit_{loop_of(args) or 'dump'}_{args.policy}.npz",
    )
    cmd = [
        sys.executable, os.path.join(HERE, "compare.py"), mine, theirs,
        "--allow-missing", "--sort-by", "rel", *TOLERANCES,
    ]
    if args.keys:
        for glob in args.keys:
            cmd += ["--keys", glob]
    if args.tap:
        cmd += ["--keys", args.tap]
    print(f"[compare] {' '.join(cmd)}")
    return subprocess.call(cmd)

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["case", "run", "collect", "compare", "all", "guidance", "classes", "sweep"])
    ap.add_argument("--sweep", type=int, default=None, help="groups to pack into one fire (`sweep`)")
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--out", default="/tmp/mini-dit-parity")
    ap.add_argument("--policy", choices=["bf16", "fp32"], default="bf16",
                    help="which emulation of the reference to diff against")
    ap.add_argument("--solver", choices=["dpm2m"], default=None,
                    help="integrate the Euler golden's schedule with this solver on the device")
    ap.add_argument("--euler", action="store_true",
                    help="the four-step schedule instead of one step")
    ap.add_argument("--inferlet", default=os.path.join(REPO, "examples/mini-dit-parity"))
    ap.add_argument("--config", default=None,
                    help=f"the serving config; its `[model] model` must be the artifact "
                         f"`{DEFAULT_SKU}` imported")
    ap.add_argument("--pie", default=None, help="the pie binary (default: PATH, else target/debug)")
    ap.add_argument("--case_file", action="store_true", help="pass the case as a scratch file instead of argv pieces")
    ap.add_argument("--keys", action="append", default=None)
    ap.add_argument("--tap", default=None,
                    help="the dump key the artifact was imported to export (PIE_MINI_DIT_TAP)")
    ap.add_argument("--cfg", action="store_true",
                    help="classifier-free guidance ON THE DEVICE: six lanes, two "
                         "attention groups, one fire, combined in the epilogue")
    ap.add_argument("--cfg-scale", dest="cfg_scale", type=float, default=1.0)
    args = ap.parse_args()
    if args.solver:
        args.euler = True

    if args.cmd == "guidance":
        return guidance(args)
    if args.cmd == "classes":
        return classes(args)
    if args.cmd == "sweep":
        return sweep(args)
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
