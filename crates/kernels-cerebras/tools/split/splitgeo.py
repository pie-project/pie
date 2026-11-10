"""splitgeo.py: the role-split geometry for the Python tools, from the Rust `split-geo` binary. Rust is the one geometry: this module only runs
the binary, reads what it writes, and executes its placement (`bindings.json`) on a runner.

  build(prep, prefix, overrides, out, what=None, levers="")   run split-geo into <out>
  load(out)                                                    the files of <out>, as a Geo
  place(put, npy, geo)                                         every copy of bindings.json, in order
  geometry("gemma:e2b")                                        a named geometry, before any meta (the preps)

The binary: $SPLIT_GEO, else `split-geo` on PATH, else the crate's cargo target (`cargo run --bin split-geo`).
Runners on the appliance only `load` / `place` a geo directory written on the user node (mkfull.py does);
a geometry override (G2, G2W, ...) needs that directory rebuilt with the same override."""
import json, os, shutil, subprocess
import numpy as np

CRATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def _cmd():
    if os.environ.get("SPLIT_GEO"):
        return [os.environ["SPLIT_GEO"]]
    if shutil.which("split-geo"):
        return ["split-geo"]
    return ["cargo", "run", "-q", "--release", "--manifest-path", os.path.join(CRATE, "Cargo.toml"),
            "--bin", "split-geo", "--"]


def build(prep, prefix, overrides, out, what=None, levers="", kernels=None):
    """split-geo <prep> <prefix> <overrides> --out <out> [--what ..] [--levers ..] [--kernels ..]"""
    a = _cmd() + [os.path.abspath(prep), prefix, overrides or "", "--out", os.path.abspath(out)]
    if what:
        a += ["--what", what if isinstance(what, str) else ",".join(what)]
    if levers:
        a += ["--levers", levers]
    if kernels:
        a += ["--kernels", os.path.abspath(kernels)]
    subprocess.run(a, check=True)
    return out


class Named:
    """A named geometry's fields as attributes (`split-geo --geometry`)."""

    def __init__(self, d):
        self.__dict__.update(d)


def geometry(spec):
    """split-geo --geometry <family>:<name>: what a prep needs before its meta exists (the unit shape, the
    head, Gemma's PLE storage columns, Gemma 12B's row groups and every row's dims)."""
    r = subprocess.run(_cmd() + ["--geometry", spec], check=True, capture_output=True, text=True)
    return Named(json.loads(r.stdout))


class Geo:
    """What split-geo wrote: floor, host, kernels, meta (dicts), rcfg ([H][W][n] u32), bindings (list).
    The attributes the runners used from the old Python geometries (PX, PY, XC, YC, UW, UH, origins, HX0, HXC,
    HC, HR, unit_origin, wl, hx, hc) are derived from the floor and the host layout."""

    def __init__(self, out):
        def j(n):
            p = os.path.join(out, n)
            return json.load(open(p)) if os.path.exists(p) else None
        self.dir = out
        self.floor, self.host, self.kernels, self.meta = j("floor.json"), j("host.json"), j("kernels.json"), j("meta.json")
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
        self.RA = (self.meta or {}).get("PYW", self.PY)
        # gpt-oss: the band's top row and columns, the units' attention / MoE columns
        self.HB0, self.HVC = hb["rect"]["y"], hb["rect"]["w"]
        part = ((self.meta or {}).get("plan") or {}).get("partition") or {}
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
    return Geo(out)


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


def _words(src, rect, npy, geo, has):
    """A binding source's words, row-major over the rectangle (Source in src/spatial/weights.rs);
    None for an optional slice whose file is not staged."""
    if src == "Rcfg":
        x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
        n = geo.rcfg_n[y:y + h, x:x + w]
        if n.min() != n.max() or n.min() == 0:
            raise ValueError(f"rcfg words over ({x}, {y}, {w}, {h}): {n.min()}..{n.max()} a PE")
        return np.ascontiguousarray(geo.rcfg[y:y + h, x:x + w, :int(n.min())]).reshape(-1)
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


def place(put, npy, geo, has=lambda name: True):
    """Every copy of geo.bindings in order: put(sym, x, y, w, h, words, b16) (the runners' put); `has(name)`:
    whether a staged file exists (the optional CY tables)."""
    for b in geo.bindings:
        r = b["rect"]
        if r["w"] * r["h"] == 0:
            continue
        v = _words(b["source"], r, npy, geo, has)
        if v is None:
            continue
        put(b["sym"], r["x"], r["y"], r["w"], r["h"], v, b["b16"])
