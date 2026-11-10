#!/usr/bin/env python3
"""capture.py <out dir> [name,...]: the role-split outputs of every config in configs.json (or the named ones),
for comparing two trees (compare.py): before and after a change that must not change them.

A config's directory holds:
- prep/       (tiny only) the prep the tree's prep script writes from the frozen tiny checkpoint;
- host/       the Rust host (examples/split_baseline.rs: SplitModel::open + one greedy launch on a recording
              device): pe.tsv (every PE's words of every symbol, hashed), rcfg.txt (every PE's rcfg words),
              calls.log (every device call), derived.json (the model's public fields);
- build/      the family's mkfull output (layout.csl, one source a PE kind) and params.used (the cslc params);
- meta.json, params.txt   the prep's (staged for full configs);
- py_rcfg.txt (where the tree has a Python geometry) every PE's rcfg words from it, "x y w.." as host/rcfg.txt.
manifest.json maps every file to its sha256 (big text files are also gzipped; the hash is of the plain text).

Env: SPLIT_JOB (staged full preps, default spatial/bench-job), SPLIT_TINY (frozen tiny checkpoints, default
<this dir>/../../../../../scratch/refactor/baseline/tiny_hf, else scratch/split), SPLIT_BASELINE_BIN (the
split_baseline binary; default: cargo build --release --example split_baseline into $CARGO_TARGET_DIR),
PY (a python with numpy for the tree's Python tools; default this one), SPLIT_TOOLS (the tree's tools/split;
default this file's parent's parent)."""
import gzip, hashlib, json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.environ.get("SPLIT_TOOLS", os.path.dirname(HERE))
CRATE = os.path.abspath(os.path.join(TOOLS, "..", ".."))
ROOT = "/cra-1900/ing-cloud/pie-bringup-20261027"
JOB = os.environ.get("SPLIT_JOB", os.path.join(ROOT, "spatial", "bench-job"))
TINY = os.environ.get("SPLIT_TINY", os.path.join(ROOT, "scratch", "refactor", "baseline", "tiny_hf"))
PY = os.environ.get("PY", sys.executable)
ENV = dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")
BIG = ("pe.tsv", "rcfg.txt", "py_rcfg.txt", "calls.log")


def sh(cmd, cwd=None, log=None):
    r = subprocess.run(cmd, cwd=cwd, env=ENV, capture_output=True, text=True)
    if log:
        open(log, "w").write(" ".join(cmd) + "\n" + r.stdout + r.stderr)
    if r.returncode:
        raise RuntimeError(f"{os.path.basename(cmd[1] if cmd[1] != '-c' else cmd[0])} rc {r.returncode}: {(r.stderr or r.stdout)[-800:]}")
    return r.stdout


def binary():
    b = os.environ.get("SPLIT_BASELINE_BIN")
    if b:
        return b
    sh(["cargo", "build", "-q", "--release", "-p", "engine-cerebras", "--example", "split_baseline"], cwd=CRATE)
    t = os.environ.get("CARGO_TARGET_DIR", os.path.join(CRATE, "..", "..", "target"))
    return os.path.join(t, "release", "examples", "split_baseline")


def merge(*parts):
    """K:V lists, later ones overriding (link.sh's merge)."""
    d = {}
    for p in parts:
        for kv in filter(None, p.strip().split(",")):
            k, v = kv.split(":")
            d[k] = v
    return ",".join(f"{k}:{v}" for k, v in d.items())


def fill(args, **kw):
    return [a.format(**kw) for a in args]


def py_rcfg(c, prep, prefix, out):
    """The tree's Python geometry's rcfg words of every PE, as host/rcfg.txt ("x y w..", rows then columns)."""
    fam = c["family"]
    meta = json.load(open(os.path.join(prep, prefix + "meta.json")))
    ov = {k: int(v) for k, v in (kv.split("=") for kv in filter(None, c["split"].split(",")))}
    M = dict(meta, **ov)
    code = None
    if fam == "q35":
        code = f"""
import sys; sys.path.insert(0, {TOOLS!r}); import split_geo as G
import json; M = json.loads({json.dumps(M)!r})
g = G.Geo(M["PX"], M["PY"], M["UNX"], M["UNY"], M["HC"], M["HR"], M["Hc"], M["SWA"], M["SWG"], M["SWV"], M["DKG"], M["DKH"],
          G2=M.get("G2", 0), G2X=M.get("G2X"), G2Y=M.get("G2Y"), G2H=M.get("G2H"), WL=M.get("WL", 0), G2W=M.get("G2W", 24))
g.build(M["has_att"], M["has_gdn"])
L = []
for y in range(g.H):
    for x in range(g.W):
        k = g.kind(x, y)
        if k == "w":
            ly = y - g.unit_origin(g.unit_of(x, y))[1]; w = g.rcfg_weight(x, y, ly - (ly > g.YC))
        elif k == "cx":
            w = g.rcfg_cx(x, y, g.unit_of(x, y) == 0)
        elif k == "cy":
            w = g.rcfg_cy(x, y)
        elif k == "head":
            w = g.rcfg_head(x, y)
        elif k == "hctl":
            w = g.rcfg_hctl(x, y, M["VR"])
        else:
            continue
        L.append(f"{{x}} {{y}} " + " ".join(str(int(v)) for v in w))
open({out!r}, "w").write("\\n".join(L) + "\\n")
"""
        if "GAPX" in M:
            return "skipped: tools/split/split_geo.py has no gap slot (Rust only)"
    elif fam in ("gemma", "g12"):
        mod, sub, bld = ("split_geo_gemma", "gemma", "build_gemma") if fam == "gemma" else ("split_geo_g12", "g12", "build_g12")
        tiny = M["geo"] == "tiny"
        code = f"""
import sys; sys.path.insert(0, {os.path.join(TOOLS, sub)!r}); import {mod} as S
import json; M = json.loads({json.dumps(M)!r})
g = S.GEOS[M["geo"]](G2=M.get("G2", 2 if {tiny} else 16), WL=1, G2W=M.get("G2W", 2 if {tiny} else 24))
g.{bld}()
S.dump_rcfg(g, M["VR"], {out + ".k"!r})
L = sorted((int(l.split()[1]), int(l.split()[0]), l) for l in open({out + ".k"!r}))
open({out!r}, "w").write("".join(" ".join(l.split()[:2] + l.split()[3:]) + "\\n" for _, _, l in L))
"""
    elif fam == "gptoss":
        r = sh([PY, os.path.join(TOOLS, "gptoss", "rcfg_hash.py"), M["geo"]], cwd=os.path.join(TOOLS, "gptoss"))
        open(out, "w").write(r)
        return "hashes by kind only (rcfg_hash.py)"
    if code:
        sh([PY, "-c", code])
        if os.path.exists(out + ".k"):
            os.remove(out + ".k")
    return None


def capture(c, top, B):
    d = os.path.join(top, c["name"])
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    notes = {}
    if c["kind"] == "full":
        prep, prefix = os.path.join(JOB, c["prep"]), c["prefix"]
    else:
        prep, prefix = os.path.join(d, "prep"), c["prefix"]
        ck = os.path.join(TINY, c["ckpt"])
        sh([PY] + fill([os.path.join(TOOLS, c["prep"][0])] + c["prep"][1:], ckpt=ck, out=prep), cwd=os.path.dirname(os.path.join(TOOLS, c["prep"][0])),
           log=os.path.join(d, "prep.log"))
    for f in ("meta.json", "params.txt"):
        if os.path.exists(os.path.join(prep, prefix + f)) and c["kind"] == "full":
            shutil.copy(os.path.join(prep, prefix + f), os.path.join(d, f))
    params = os.path.join(prep, prefix + "params.txt")
    # the Rust host
    h = os.path.join(d, "host")
    sh([B, prep, prefix, c["split"], str(c["vocab"]), h], log=os.path.join(d, "host.log"))
    # the build
    b = os.path.join(d, "build")
    mk = c["mkfull"]
    sh([PY] + fill([os.path.join(TOOLS, mk[0])] + mk[1:], out=b, params=params, prep=prep, prefix=prefix), cwd=os.path.dirname(os.path.join(TOOLS, mk[0])))
    base = open(params).read() if os.path.exists(params) else ""
    cs = c["cslc"]
    used = merge(base, cs.replace("@params", "")) if cs.startswith("@params") else merge(cs)
    open(os.path.join(b, "params.used"), "w").write(used + "\n")
    # the Python geometry
    try:
        n = py_rcfg(c, prep, prefix, os.path.join(d, "py_rcfg.txt"))
        if n:
            notes["py_rcfg"] = n
    except Exception as e:  # recorded, not fatal: the Python tools go away in the refactor
        notes["py_rcfg"] = f"error: {e}"
    if os.path.exists(os.path.join(d, "py_rcfg.txt")) and c["family"] != "gptoss":
        same = open(os.path.join(d, "py_rcfg.txt")).read() == open(os.path.join(h, "rcfg.txt")).read()
        notes["py_rcfg_equals_rust"] = same
    if notes:
        json.dump(notes, open(os.path.join(d, "notes.json"), "w"), indent=1)
    return notes


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
    B = binary()
    for c in cfgs:
        try:
            n = capture(c, top, B)
            print(c["name"], "ok", json.dumps(n) if n else "", flush=True)
        except Exception as e:
            print(c["name"], "FAILED", e, flush=True)
            open(os.path.join(top, c["name"], "FAILED"), "w").write(str(e) + "\n")
    m = manifest(top)
    json.dump(m, open(os.path.join(top, "manifest.json"), "w"), indent=0, sort_keys=True)
    print("manifest", len(m), "files")


if __name__ == "__main__":
    main()
