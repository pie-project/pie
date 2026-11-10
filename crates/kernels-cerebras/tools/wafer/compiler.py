"""compiler.py: the wafer program for the Python tools, from the Rust `wafer-compile` compiler. Rust is the one plan and
geometry: this module only runs the binary, reads what it writes, and executes its placement (`bindings.json`) on a runner.

  plan(out, model=None, shape=None, preset=None, set="")   wafer-compile plan: <out>/plan.json, <out>/tiling.json
  tiling(plan_dir)                                         <plan_dir>/tiling.json, as attributes (the preps)
  build(plan, out, emit=None, levers="", kernels=None)     wafer-compile build: everything the plan builds, into <out>
  load(out)                                                the files of <out>, as a Compiled
  place(put, npy, compiled)                                     every copy of bindings.json, in order

The binary: $WAFER_COMPILE, else `wafer-compile` on PATH, else the crate's cargo target (`cargo run --bin wafer-compile`).
Runners on the appliance only `load` / `place` a directory written on the user node (build.py does)."""
import json, os, shutil, subprocess
import numpy as np

CRATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def _cmd():
    if os.environ.get("WAFER_COMPILE"):
        return [os.environ["WAFER_COMPILE"]]
    if shutil.which("wafer-compile"):
        return ["wafer-compile"]
    return ["cargo", "run", "-q", "--release", "--manifest-path", os.path.join(CRATE, "Cargo.toml"),
            "--bin", "wafer-compile", "--"]


def plan(out, model=None, shape=None, preset=None, set=""):
    """wafer-compile plan (--model <deployment> | --shape <shape.json>) [--preset ..] [--set ..] --out <out>"""
    a = _cmd() + ["plan", "--out", os.path.abspath(out)]
    a += ["--model", model] if model else ["--shape", os.path.abspath(shape)]
    if preset:
        a += ["--preset", preset]
    if set:
        a += ["--set", set]
    subprocess.run(a, check=True)
    return out


class Named:
    """A JSON object's fields as attributes."""

    def __init__(self, d):
        self.__dict__.update(d)


def tiling(plan_dir):
    """What a prep tiles the checkpoint by (wafer-compile plan's tiling.json): the plan's cut in the prep's terms (the
    unit shape, the head, Gemma's PLE storage columns, Gemma 12B's row groups and every row's dims) and the preset."""
    return Named(json.load(open(os.path.join(plan_dir, "tiling.json"))))


def build(plan, out, emit=None, levers="", kernels=None):
    """wafer-compile build --plan <plan.json> --out <out> [--emit ..] [--levers ..] [--kernels ..]"""
    a = _cmd() + ["build", "--plan", os.path.abspath(plan), "--out", os.path.abspath(out)]
    if emit:
        a += ["--emit", emit if isinstance(emit, str) else ",".join(emit)]
    if levers:
        a += ["--levers", levers]
    if kernels:
        a += ["--kernels", os.path.abspath(kernels)]
    subprocess.run(a, check=True)
    return out


class Compiled:
    """What wafer-compile wrote: floor, host, kernels, plan (dicts), rcfg ([H][W][n] u32), bindings (list).
    The attributes the runners read (PX, PY, XC, YC, UW, UH, origins, HX0, HXC,
    HC, HR, unit_origin, wl, hx, hc) are derived from the floor and the host layout."""

    def __init__(self, out):
        def j(n):
            p = os.path.join(out, n)
            return json.load(open(p)) if os.path.exists(p) else None
        self.dir = out
        self.floor, self.host, self.kernels, self.plan = j("floor.json"), j("host.json"), j("kernels.json"), j("plan.json")
        self.bindings = j("bindings.json") or []
        p = os.path.join(out, "rcfg.npy")
        self.rcfg = np.load(p, mmap_mode="r") if os.path.exists(p) else None
        p = os.path.join(out, "rcfg_n.npy")
        self.rcfg_n = np.load(p) if os.path.exists(p) else None
        f, h = self.floor, self.host
        self.W, self.H = f["w"], f["h"]
        us = f["units"]
        self.origins = [(u["rect"]["x"], u["rect"]["y"]) for u in us]
        self.UW, self.UH = us[0]["rect"]["w"], us[0]["rect"]["h"]
        self.XC, self.YC = us[0]["cx"], us[0]["cy"]
        self.PY = h["rows"] if h else self.UH - 1
        self.PX = self.UW - 1
        hb = f["head"]
        self.HX0, self.HR = hb["rect"]["x"], hb["rect"]["h"]
        ctl = hb["ctl"]
        if "Column" in ctl:
            self.HXC, self.HC = ctl["Column"] - self.HX0, hb["rect"]["w"] - 1
        else:
            self.HXC, self.HC = ctl["Row"] - hb["rect"]["y"], hb["rect"]["w"]
        self.root = (hb["root"]["x"], hb["root"]["y"])
        # rows a unit keeps KV on (Gemma 12B's attention rows, RA)
        part = (self.plan or {}).get("partition") or {}
        self.RA = part.get("kv_rows", self.PY)
        # gpt-oss: the band's top row and columns, the units' attention / MoE columns
        self.HB0, self.HVC = hb["rect"]["y"], hb["rect"]["w"]
        self.AX = part.get("attn_cols") or 0
        self.MXc = self.PX - self.AX
        self._dims = [0] * self.PY
        for (r0, r1, d, _) in (h["x_runs"] if h else []):
            for r in range(r0, r1):
                self._dims[r] = d

    def units(self):
        return len(self.origins)

    def unit_origin(self, u):
        return self.origins[u]

    @staticmethod
    def wl(i, c):
        """Local index i of an axis with its control line at c, skipping it."""
        return i + (1 if i >= c else 0)

    def hx(self, c):
        return self.HX0 + c + (1 if c >= self.HXC else 0)

    def hc(self, r):
        """x's dims a CX row (row r's run)."""
        return self._dims[r]

    def rcfg_at(self, x, y):
        return np.asarray(self.rcfg[y, x, :int(self.rcfg_n[y, x])])


def load(out):
    return Compiled(out)


def _slice(v, npy):
    """A Slice's words (Slice in src/spatial/weights.rs: lead, dims, pad_to, across, repeat)."""
    a = npy(v["name"])
    dims = [tuple(d) for d in v["dims"]]
    if v["lead"] is not None and a.ndim > len(dims) + 1:
        a = a[v["lead"]]
    a = np.asarray(a[tuple(slice(lo, hi) for lo, hi in dims)])
    per = int(np.prod(a.shape[len(dims):])) if a.ndim > len(dims) else 1
    pes = a.reshape(-1, per)
    to = v["pad_to"] or per
    if to > per:
        p = np.zeros((pes.shape[0], to), pes.dtype)
        p[:, :per] = pes
        pes = p
    if v["across"] > 1:
        pes = np.repeat(pes, v["across"], axis=0)
    return np.tile(pes.reshape(-1), max(v["repeat"], 1))


def _words(src, rect, npy, compiled, has):
    """A binding source's words, row-major over the rectangle (Source in src/spatial/weights.rs);
    None for an optional slice whose file is not staged."""
    if src == "Rcfg":
        x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
        n = compiled.rcfg_n[y:y + h, x:x + w]
        if n.min() != n.max() or n.min() == 0:
            raise ValueError(f"rcfg words over ({x}, {y}, {w}, {h}): {n.min()}..{n.max()} a PE")
        return np.ascontiguousarray(compiled.rcfg[y:y + h, x:x + w, :int(n.min())]).reshape(-1)
    (k, v), = src.items()
    if k == "Inline":
        return np.asarray(v, np.uint32)
    if k == "Staged":
        return np.asarray(npy(v["name"])[v["index"][0]:v["index"][1]]).reshape(-1)
    if k == "Slice":
        if v["optional"] and not has(v["name"]):
            return None
        return _slice(v, npy)
    raise ValueError(f"binding source {k}")


def place(put, npy, compiled, has=lambda name: True):
    """Every copy of compiled.bindings in order: put(sym, x, y, w, h, words, b16) (the runners' put); `has(name)`:
    whether a staged file exists (the optional CY tables)."""
    for b in compiled.bindings:
        r = b["rect"]
        if r["w"] * r["h"] == 0:
            continue
        v = _words(b["source"], r, npy, compiled, has)
        if v is None:
            continue
        put(b["sym"], r["x"], r["y"], r["w"], r["h"], v, b["b16"])
