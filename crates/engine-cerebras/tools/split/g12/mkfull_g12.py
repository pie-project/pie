#!/usr/bin/env python3
"""mkfull_g12.py <outdir> <geo: tiny | g12> [G2=..,G2W=..]: the Gemma 4 12B role-split payload (layout.csl + one source
a PE kind + the host files). Geometry from split_geo_g12.GEOS[geo]: attention rows (local y < RA: weight ATT 1, Hc A;
their CX Hc A) and plain rows (weight ATT 2, Hc W). Kernel sources: $SPLIT_SRC, else this branch's src/spatial/csl/split."""
import os, shutil, sys
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import split_geo_g12 as S
SRC_DIR = os.environ.get("SPLIT_SRC", os.path.join(H, "..", "..", "..", "src", "spatial", "csl", "split"))
out, geo = sys.argv[1], sys.argv[2]
kw = dict(kv.split("=") for kv in (sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else []))
os.makedirs(out, exist_ok=True)
g = S.GEOS[geo](G2=int(kw.get("G2", 16)), WL=1, G2W=int(kw.get("G2W", 24)))
g.build_g12()
NAMES = [("w", "u16"), ("rcfg", "u32"), ("tdbg", "u32"), ("x", "f32"), ("xdbg", "f32"), ("xred", "f32"), ("gam", "f32"),
         ("qkw", "f32"), ("rot", "f32"), ("tab", "u32"), ("hw", "u16"), ("hg", "f32"), ("hemb", "u16"), ("res", "f32"),
         ("ring", "u32"), ("lgt", "f32"), ("force", "u32"), ("nforce", "u32")]
OWN = {"weight": ["w", "rcfg", "tdbg"], "cx": ["x", "gam", "rcfg", "xdbg", "tdbg", "xred"],
       "cy": ["qkw", "rot", "tab", "rcfg", "tdbg"], "head": ["hw", "hg", "rcfg", "hemb", "tdbg"],
       "hctl": ["res", "rcfg", "tdbg", "ring", "lgt", "force", "nforce"], "filler": []}
SRC = {"weight": "weight_g12.csl", "cx": "ctl_cx_g12.csl", "cy": "ctl_cy_g12.csl", "head": "head_g12.csl", "hctl": "head_ctl.csl"}
FILLER = '''param memcpy_params;
const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);
fn init() void { sys_mod.unblock_cmd_stream(); }
fn compute(steps: u32, start_pos: u32) void { sys_mod.unblock_cmd_stream(); }
comptime { @export_symbol(init); @export_symbol(compute); }
'''
for k in OWN:
    body = FILLER if k == "filler" else open(os.path.join(SRC_DIR, SRC[k])).read()
    d = ["// the other programs' exported names (one-word dummies)"]
    for n, t in NAMES:
        if n in OWN[k]: continue
        d.append(f"var dd_{n} = @zeros([1]{t});\nvar dd_{n}_ptr: [*]{t} = &dd_{n};")
    d.append("comptime {")
    for n, t in NAMES:
        if n not in OWN[k]: d.append(f'  @export_symbol(dd_{n}_ptr, "{n}");')
    d.append("}")
    open(os.path.join(out, f"{k}.csl"), "w").write(body + "\n" + "\n".join(d) + "\n")

P = ["K", "KM", "HcA", "HcW", "Qh", "DcS", "DcG", "NES", "NEG", "HDS", "HDG", "HR_", "Ic", "VR", "HDc", "NPX"]
PD = {"WIN": 0, "WFIFO": 32, "WSPEC": 0, "DIAG": 0, "TSTAMP": 0, "XKEEP": 0, "ATTL": 2, "HACC16": 0, "WFORCE": 0, "FIFOW": 32}
L = [f"param {p}: u16;" for p in P] + [f"param {p}: u16 = {v};" for p, v in PD.items()]
L += ["param kernel_rows: u16 = 0;", "param kernel_cols: u16 = 0;", "param iters: u16 = 0;"]
L.append(f"const W: u16 = {g.W};\nconst HH: u16 = {g.H};\nconst PX: u16 = {g.PX};\nconst PY: u16 = {g.PY};\nconst XC: u16 = {g.XC};\nconst YC: u16 = {g.YC};\nconst RA: u16 = {g.RA};")
L.append(f"const HX0: u16 = {g.HX0};\nconst HC: u16 = {g.HC};\nconst HXC: u16 = {g.HXC};\nconst HR: u16 = {g.HR};")
L.append('const memcpy = @import_module("<memcpy/get_params>", .{ .width = W, .height = HH });')
L.append("layout {\n  @set_rectangle(W, HH);\n  for (@range(u16, W)) |x| {\n    const mp = memcpy.get_params(x);\n    for (@range(u16, HH)) |y| {")
L.append("      var done = false;")
L.append("      if (x >= HX0) {\n        done = true;\n        if (x - HX0 > HC or y >= HR) { @set_tile_code(x, y, \"filler.csl\", .{ .memcpy_params = mp }); }\n"
         "        else if (x - HX0 == HXC) { @set_tile_code(x, y, \"hctl.csl\", .{ .memcpy_params = mp, .VR = VR, .FIFOW = FIFOW, .WFORCE = @as(u32, WFORCE), .WFN = @as(u32, HR) }); }\n"
         "        else { @set_tile_code(x, y, \"head.csl\", .{ .memcpy_params = mp, .VR = VR, .HDc = HDc, .NPX = NPX, .HACC16 = HACC16 }); }\n      }")
for u, (X0, Y0) in enumerate(g.origins):
    L.append(f"      if (!done and x >= {X0} and x < {X0 + g.UW} and y >= {Y0} and y < {Y0 + g.UH}) {{\n        done = true;\n"
             f"        const lx = x - {X0};\n        const ly = y - {Y0};\n"
             f"        const hc = if (ly < RA) HcA else HcW;\n"
             f"        const att = if (ly < RA) 1 else 2;\n"
             f"        if (lx == XC and ly == YC) {{ @set_tile_code(x, y, \"filler.csl\", .{{ .memcpy_params = mp }}); }}\n"
             f"        else if (lx == XC) {{ @set_tile_code(x, y, \"cx.csl\", .{{ .memcpy_params = mp, .K = K, .Hc = hc, .DIAG = DIAG, .FIFOW = FIFOW, .XKEEP = XKEEP }}); }}\n"
             f"        else if (ly == YC) {{ @set_tile_code(x, y, \"cy.csl\", .{{ .memcpy_params = mp, .K = K, .KM = KM, .Qh = Qh, .DcS = DcS, .DcG = DcG, .Ic = Ic, .HR = HR_, .DIAG = DIAG, .FIFOW = FIFOW, .ATTL = ATTL, .TSTAMP = TSTAMP, .HDS = HDS, .HDG = HDG }}); }}\n"
             f"        else {{ @set_tile_code(x, y, \"weight.csl\", .{{ .memcpy_params = mp, .K = K, .Hc = hc, .Ic = Ic, .MV = 1, .KM = KM, .ATT = att, .Qh = Qh, .DcS = DcS, .DcG = DcG, .NES = NES, .NEG = NEG, .PYW = RA, .HDS = HDS, .HDG = HDG, .WIN = WIN, .WSPEC = WSPEC, .DIAG = DIAG, .FIFOW = WFIFO, .TSTAMP = TSTAMP }}); }}\n      }}")
L.append("      if (!done) { @set_tile_code(x, y, \"filler.csl\", .{ .memcpy_params = mp }); }")
L.append("    }\n  }")
for n, t in NAMES:
    L.append(f'  @export_name("{n}", [*]{t}, true);')
L.append('  @export_name("init", fn()void);\n  @export_name("compute", fn(u32, u32)void);\n}')
open(os.path.join(out, "layout.csl"), "w").write("\n".join(L) + "\n")
for f in ("../split_geo.py", "split_geo_g12.py", "run_g12.py"):
    shutil.copy(os.path.join(H, f), os.path.join(out, os.path.basename(f)))
print("ok", out, "W", g.W, "H", g.H)
