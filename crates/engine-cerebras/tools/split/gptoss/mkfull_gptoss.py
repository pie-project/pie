#!/usr/bin/env python3
"""mkfull_gptoss.py <outdir> <geo: tiny | full> <params.txt> [k=v,...]: the gpt-oss role-split payload: layout.csl
(the geometry of split_geo_gptoss.py) and one source a PE kind (attention / MoE weight, CX, attention / MoE CY,
head, head control, filler) with every program's exported names (one-word dummies for the names a kind does not
own). Shape params from the prep's params.txt; diagnostics (DIAG, TSTAMP, XKEEP) default off. Kernel sources:
$SPLIT_SRC, else this branch's src/spatial/csl/split."""
import os, shutil, sys
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import split_geo_gptoss as S
SRC_DIR = os.environ.get("SPLIT_SRC", os.path.join(H, "..", "..", "..", "src", "spatial", "csl", "split"))
out, geo, ptxt = sys.argv[1], sys.argv[2], sys.argv[3]
kw = dict(kv.split("=") for kv in (sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] else []))
os.makedirs(out, exist_ok=True)
g = S.GEOS[geo]()
NAMES = [("w", "u16"), ("wx", "u16"), ("wr", "f32"), ("rcfg", "u32"), ("tdbg", "u32"), ("x", "f32"), ("xdbg", "f32"), ("xred", "f32"),
         ("gam", "f32"), ("bo", "f32"), ("bd", "f32"), ("qb", "f32"), ("snk", "f32"), ("rot", "f32"), ("rb", "f32"), ("gb", "f32"),
         ("ps", "f32"), ("hw", "u16"), ("hg", "f32"), ("hemb", "u16"), ("res", "f32"), ("ring", "u32"), ("lgt", "f32"),
         ("force", "u32"), ("nforce", "u32")]
OWN = {"wa": ["w", "rcfg", "tdbg"], "wm": ["wx", "wr", "rcfg", "tdbg"], "cx": ["x", "gam", "bo", "bd", "rcfg", "xdbg", "tdbg", "xred"],
       "cya": ["qb", "snk", "rot", "rcfg", "tdbg"], "cym": ["rb", "gb", "ps", "rcfg", "tdbg"], "head": ["hw", "hg", "rcfg", "hemb", "tdbg"],
       "hctl": ["res", "rcfg", "tdbg", "ring", "lgt", "force", "nforce"], "filler": []}
SRC = {"wa": "weight_gptoss_att.csl", "wm": "weight_gptoss_moe.csl", "cx": "ctl_cx_gptoss.csl", "cya": "ctl_cy_gptoss_att.csl",
       "cym": "ctl_cy_gptoss_moe.csl", "head": "head_pe_gptoss.csl", "hctl": "head_ctl.csl"}
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
P = ["K", "KM", "Hc", "NQA", "QO", "Qh", "Dc", "NROT", "Ic", "NEX", "TK", "HR", "NLV", "VR"]
PD = {"DIAG": 0, "TSTAMP": 0, "XKEEP": 0, "FIFOW": 32, "WFORCE": 0, "NE": int(kw.get("NE", 128)), "WIN": int(kw.get("WIN", 128))}
L = [f"param {p}: u16;" for p in P] + [f"param {p}: u16 = {v};" for p, v in PD.items()]
import json as _j, math as _mt
_m = _j.load(open(os.path.join(os.path.dirname(os.path.abspath(ptxt)), os.path.basename(ptxt).replace("params.txt", "meta.json"))))
L.append(f"const QSC: f32 = {repr(float(1.0 / _mt.sqrt(_m['HD'])))};")
L += ["param kernel_rows: u16 = 0;", "param kernel_cols: u16 = 0;", "param iters: u16 = 0;"]
L.append(f"const W: u16 = {g.W};\nconst HH: u16 = {g.H};\nconst UW: u16 = {g.UW};\nconst UH: u16 = {g.UH};\nconst XC: u16 = {g.XC};\nconst YC: u16 = {g.YC};")
L.append(f"const NUX: u16 = {g.UNX};\nconst NUY: u16 = {g.UNY};\nconst HB0: u16 = {g.HB0};\nconst HVC: u16 = {g.HVC};\nconst HXC: u16 = {g.HXC};\nconst PY: u16 = {g.PY};")
L.append('const memcpy = @import_module("<memcpy/get_params>", .{ .width = W, .height = HH });')
L.append("layout {\n  @set_rectangle(W, HH);\n  for (@range(u16, W)) |x| {\n    const mp = memcpy.get_params(x);\n    for (@range(u16, HH)) |y| {")
L.append("      if (y >= HB0) {\n"
         "        if (x >= HVC) { @set_tile_code(x, y, \"filler.csl\", .{ .memcpy_params = mp }); }\n"
         "        else if (y - HB0 == HXC) { @set_tile_code(x, y, \"hctl.csl\", .{ .memcpy_params = mp, .VR = VR, .FIFOW = FIFOW, .WFORCE = @as(u32, WFORCE), .WFN = @as(u32, HVC) }); }\n"
         "        else { @set_tile_code(x, y, \"head.csl\", .{ .memcpy_params = mp, .VR = VR, .HDc = Hc }); }\n"
         "      } else if (x / UW >= NUX or y / UH >= NUY) { @set_tile_code(x, y, \"filler.csl\", .{ .memcpy_params = mp }); }\n"
         "      else {\n        const lx = x % UW;\n        const ly = y % UH;\n"
         "        if (lx == XC and ly == YC) { @set_tile_code(x, y, \"filler.csl\", .{ .memcpy_params = mp }); }\n"
         "        else if (lx == XC) { @set_tile_code(x, y, \"cx.csl\", .{ .memcpy_params = mp, .K = K, .Hc = Hc, .NEX = NEX, .TK = TK, .DIAG = DIAG, .XKEEP = XKEEP }); }\n"
         "        else if (ly == YC and lx < XC) { @set_tile_code(x, y, \"cya.csl\", .{ .memcpy_params = mp, .K = K, .NQA = NQA, .Qh = Qh, .Dc = Dc, .NROT = NROT, .HR = HR, .NLV = NLV, .QSC = QSC, .DIAG = DIAG, .TSTAMP = TSTAMP, .FIFOW = FIFOW }); }\n"
         "        else if (ly == YC) { @set_tile_code(x, y, \"cym.csl\", .{ .memcpy_params = mp, .K = K, .Ic = Ic, .NEX = NEX, .TK = TK, .HR = HR, .DIAG = DIAG, .TSTAMP = TSTAMP, .FIFOW = FIFOW }); }\n"
         "        else if (lx < XC) { @set_tile_code(x, y, \"wa.csl\", .{ .memcpy_params = mp, .K = K, .KM = KM, .Hc = Hc, .NQA = NQA, .QO = QO, .Qh = Qh, .Dc = Dc, .NE = NE, .PYW = PY, .WIN = WIN, .DIAG = DIAG, .TSTAMP = TSTAMP, .FIFOW = FIFOW }); }\n"
         "        else { @set_tile_code(x, y, \"wm.csl\", .{ .memcpy_params = mp, .K = K, .Hc = Hc, .Ic = Ic, .NEX = NEX, .TK = TK, .DIAG = DIAG, .TSTAMP = TSTAMP, .FIFOW = FIFOW }); }\n"
         "      }")
L.append("    }\n  }")
for n, t in NAMES:
    L.append(f'  @export_name("{n}", [*]{t}, true);')
L.append('  @export_name("init", fn()void);\n  @export_name("compute", fn(u32, u32)void);\n}')
open(os.path.join(out, "layout.csl"), "w").write("\n".join(L) + "\n")
params = open(ptxt).read().strip()
extra = ",".join(f"{k}:{v}" for k, v in kw.items() if k not in ("NE", "WIN"))
open(os.path.join(out, "params.txt"), "w").write(params + (("," + extra) if extra else "") + "\n")
for f in ("split_geo.py", "split_geo_gptoss.py", "run_gptoss.py"):
    if os.path.exists(os.path.join(H, f)):
        shutil.copy(os.path.join(H, f), os.path.join(out, f))
print("ok", out, "W", g.W, "H", g.H, "params", open(os.path.join(out, "params.txt")).read().strip())
