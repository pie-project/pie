#!/usr/bin/env python3
"""capture.py <out dir> [name,...]: the wafer program outputs of every config in configs.json (or the named ones),
for comparing two trees (compare.py): before and after a change that must not change them.

A config's directory holds:
- plan/       wafer-compile plan's plan.json and tiling.json (the config's model or shape, preset and overrides);
- prep/       (tiny only) the prep the tree's prep script writes from the frozen tiny checkpoint for that plan;
- build/      wafer-compile build's output for the plan in compiled/ (layout/ with layout.csl and one source a PE kind, the artifact) and
              params.used (the cslc params compare.py evaluates the layout with: none, every param a literal);
- host/       the Rust host (examples/artifact_baseline.rs: the artifact read back, ArtifactModel::open + one greedy launch
              on a recording device): pe.tsv (every PE's words of every symbol, hashed), rcfg.txt (every PE's rcfg
              words), calls.log (every device call), derived.json (the model's public fields).
manifest.json maps every file to its sha256 (compare.py also reads a file gzipped beside its name, as a reference
keeps its big text files).

Env: BENCH_JOB (staged full preps, default spatial/bench-job), TINY_CKPTS (frozen tiny checkpoints, default
scratch/refactor/baseline/tiny_hf), ARTIFACT_BASELINE_BIN (the artifact_baseline binary) and WAFER_COMPILE (the wafer-compile
binary; default for both: cargo build --release into $CARGO_TARGET_DIR), PY (a python with numpy for the tree's
Python tools; default this one), WAFER_TOOLS (the tree's tools/wafer; default this file's parent's parent)."""
import hashlib, json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.environ.get("WAFER_TOOLS", os.path.dirname(HERE))
CRATE = os.path.abspath(os.path.join(TOOLS, "..", ".."))
ROOT = "/cra-1900/ing-cloud/pie-bringup-20261027"
JOB = os.environ.get("BENCH_JOB", os.path.join(ROOT, "spatial", "bench-job"))
TINY = os.environ.get("TINY_CKPTS", os.path.join(ROOT, "scratch", "refactor", "baseline", "tiny_hf"))
PY = os.environ.get("PY", sys.executable)
ENV = dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")


def sh(cmd, cwd=None, log=None):
    r = subprocess.run(cmd, cwd=cwd, env=ENV, capture_output=True, text=True)
    if log:
        open(log, "w").write(" ".join(cmd) + "\n" + r.stdout + r.stderr)
    if r.returncode:
        raise RuntimeError(f"{os.path.basename(cmd[1] if cmd[1] != '-c' else cmd[0])} rc {r.returncode}: {(r.stderr or r.stdout)[-800:]}")
    return r.stdout


def binaries():
    """artifact_baseline and wafer-compile, built unless named."""
    t = os.environ.get("CARGO_TARGET_DIR", os.path.join(CRATE, "..", "..", "target"))
    b, g = os.environ.get("ARTIFACT_BASELINE_BIN"), os.environ.get("WAFER_COMPILE")
    if not b:
        sh(["cargo", "build", "-q", "--release", "-p", "engine-cerebras", "--example", "artifact_baseline"], cwd=CRATE)
        b = os.path.join(t, "release", "examples", "artifact_baseline")
    if not g:
        sh(["cargo", "build", "-q", "--release", "-p", "kernels-cerebras", "--bin", "wafer-compile"], cwd=CRATE)
        g = os.path.join(t, "release", "wafer-compile")
    ENV["WAFER_COMPILE"] = g
    return b, g


def fill(args, **kw):
    return [a.format(**kw) for a in args]


def capture(c, top, B, G):
    d = os.path.join(top, c["name"])
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    # the plan
    pl = os.path.join(d, "plan")
    src = ["--model", c["model"]] if "model" in c else ["--shape", os.path.join(HERE, c["shape"])]
    pos = ["--positions", str(c["positions"])] if c.get("positions") else []
    sh([G, "plan"] + src + ["--preset", c["preset"], "--set", c["set"]] + pos + ["--out", pl], log=os.path.join(d, "plan.log"))
    if c["kind"] == "full":
        prep, prefix = os.path.join(JOB, c["prep"]), c["prefix"]
    else:
        prep, prefix = os.path.join(d, "prep"), c["prefix"]
        ck = os.path.join(TINY, c["ckpt"])
        sh([PY] + fill([os.path.join(TOOLS, c["prep"][0])] + c["prep"][1:], ckpt=ck, out=prep, plan=pl),
           cwd=os.path.dirname(os.path.join(TOOLS, c["prep"][0])), log=os.path.join(d, "prep.log"))
    # the build
    b = os.path.join(d, "build")
    lev = ["--levers", c["levers"]] if c["levers"] else []
    sh([G, "build", "--plan", os.path.join(pl, "plan.json"), "--out", os.path.join(b, "compiled")] + lev)
    open(os.path.join(b, "params.used"), "w").write("\n")  # (every param a literal: the link takes none)
    # the Rust host, on the build's artifact
    sh([B, os.path.join(b, "compiled"), prep, prefix, str(c["vocab"]), os.path.join(d, "host")], log=os.path.join(d, "host.log"))


def manifest(top):
    m = {}
    for r, _, fs in os.walk(top):
        for f in sorted(fs):
            p = os.path.join(r, f)
            rel = os.path.relpath(p, top)
            if f.endswith((".log", ".gz")) and f != "calls.log" or f == "manifest.json":
                continue
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(blk)
            m[rel] = h.hexdigest()
    return m


def main():
    top = os.path.abspath(sys.argv[1])
    only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
    cfgs = [c for c in json.load(open(os.path.join(HERE, "configs.json")))["configs"] if not only or c["name"] in only]
    os.makedirs(top, exist_ok=True)
    B, G = binaries()
    for c in cfgs:
        try:
            capture(c, top, B, G)
            print(c["name"], "ok", flush=True)
        except Exception as e:
            print(c["name"], "FAILED", e, flush=True)
            open(os.path.join(top, c["name"], "FAILED"), "w").write(str(e) + "\n")
    m = manifest(top)
    json.dump(m, open(os.path.join(top, "manifest.json"), "w"), indent=0, sort_keys=True)
    print("manifest", len(m), "files")


if __name__ == "__main__":
    main()
