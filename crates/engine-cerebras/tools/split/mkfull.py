#!/usr/bin/env python3
"""mkfull.py <outdir>: the role-split whole-program build directory (layout.csl and one source a PE
kind: weight, cx, cy, head, hctl, filler), from the kernels in src/spatial/csl/split (or $SPLIT_SRC).
Units UNX x UNY of PX x PY weight PEs (+ the CX column and CY row), the head east of them (HC + 1
columns, HR rows). Every unit has the same K layers (KM: attention bits). A gap slot (GAPX, GAPY: a unit count
one short of the grid) holds relay PEs. The shape params come
from the prep (prep_q35.py's params.txt); the lever and diagnostic params default to the shipped
build and can be overridden on the cslc command line (link.sh). See spatial/split/README.md."""
import os, sys
S = os.environ.get("SPLIT_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "spatial", "csl", "split"))
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
NAMES = [("w", "u16"), ("rcfg", "u32"), ("x", "f32"), ("gam", "@fp16()"), ("cw", "f32"), ("gpar", "f32"),
         ("gnw", "f32"), ("qkw", "f32"), ("rot", "f32"), ("tab", "u32"), ("hw", "u16"), ("hg", "f32"), ("res", "f32"), ("xdbg", "f32"), ("tdbg", "u32"), ("xred", "f32"), ("ring", "u32"), ("hemb", "u16"), ("lgt", "f32"), ("force", "u32"), ("nforce", "u32")]
OWN = {"weight": ["w", "rcfg", "tdbg"], "cx": ["x", "gam", "rcfg", "xdbg", "tdbg", "xred"], "cy": ["cw", "gpar", "gnw", "qkw", "rot", "tab", "rcfg", "tdbg"],
       "head": ["hw", "hg", "rcfg", "hemb", "tdbg"], "hctl": ["res", "rcfg", "tdbg", "ring", "lgt", "force", "nforce"], "filler": [],
       "relay": ["rcfg"]}
SRC = {"weight": "weight_q35.csl", "cx": "ctl_cx.csl", "cy": "ctl_cy_q35.csl", "head": "head_pe.csl", "hctl": "head_ctl.csl",
       "relay": "relay.csl"}
FILLER = '''param memcpy_params;
const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);
fn init() void { sys_mod.unblock_cmd_stream(); }
fn compute(steps: u32, start_pos: u32) void { sys_mod.unblock_cmd_stream(); }
comptime { @export_symbol(init); @export_symbol(compute); }
'''
for k in OWN:
    body = FILLER if k == "filler" else open(os.path.join(S, SRC[k])).read()
    d = ["// the other programs' exported names (one-word dummies)"]
    for n, t in NAMES:
        if n in OWN[k]: continue
        d.append(f"var dd_{n} = @zeros([1]{t});\nvar dd_{n}_ptr: [*]{t} = &dd_{n};")
    d.append("comptime {")
    for n, t in NAMES:
        if n not in OWN[k]: d.append(f'  @export_symbol(dd_{n}_ptr, "{n}");')
    d.append("}")
    open(os.path.join(out, f"{k}.csl"), "w").write(body + "\n" + "\n".join(d) + "\n")
# shape (the prep's params.txt)
P = ["PX", "PY", "UNX", "UNY", "HC", "HR", "VR", "K", "KM", "Hc", "Ic", "NQA", "NQG", "Qh", "Dc", "DK", "DV", "NE",
     "HR_", "HD", "NROT", "HK", "HV"]
# levers and diagnostics, default = the shipped build (the kernels' headers describe each)
DEF = [("NPX", 1),                                          # head: x in one f16 part
       ("SILU16", 1), ("CONVQK", 1), ("ATTV", 1), ("GDEC", 1), ("ATTL", 2),   # CY / weight levers
       ("FIFOW", 32),
       ("GATE", 1), ("SLW", 0),                                 # CY: attention's output gate (Qwen3: 0), silu16's tail
       ("GAPX", 255), ("GAPY", 255),
       ("RLV", 4), ("RBITS", 3),                                # CY RoPE tables: positions < 2^(RBITS RLV) (the prep's)                            # the unit slot left out of the snake (relay PEs)
       ("DIAG", 0), ("TSTAMP", 0), ("XKEEP", 0), ("HACC16", 0), ("WFORCE", 0)]  # diagnostics / measurement
L = [f"param {p}: u16;" for p in P] + [f"param {p}: u16 = {v};" for p, v in DEF]
L += ["param kernel_rows: u16 = 0;", "param kernel_cols: u16 = 0;", "param iters: u16 = 0;"]
L.append('''const GDNF: u16 = if (KM == (1 << K) - 1) 0 else 1;
const ATTF: u16 = if (KM == 0) 0 else 1;
const XC: u16 = PX / 2;
const YC: u16 = PY / 2;
const UW: u16 = PX + 1;
const UH: u16 = PY + 1;
const HX0: u16 = UNX * UW;
const HXC: u16 = HC / 2;
const W: u16 = HX0 + HC + 1;
const HH: u16 = if (UNY * UH > HR) UNY * UH else HR;
const QO: u16 = Qh * Dc;
const memcpy = @import_module("<memcpy/get_params>", .{ .width = W, .height = HH });''')
L.append('''layout {
  @set_rectangle(W, HH);
  for (@range(u16, W)) |x| {
    const mp = memcpy.get_params(x);
    for (@range(u16, HH)) |y| {
      const lx = x % UW;
      const ly = y % UH;
      if (x >= HX0) {
        if (x - HX0 > HC or y >= HR) {
          @set_tile_code(x, y, "filler.csl", .{ .memcpy_params = mp });
        } else if (x - HX0 == HXC) {
          @set_tile_code(x, y, "hctl.csl", .{ .memcpy_params = mp, .VR = VR, .FIFOW = FIFOW, .WFORCE = @as(u32, WFORCE), .WFN = @as(u32, HR) });
        } else {
          @set_tile_code(x, y, "head.csl", .{ .memcpy_params = mp, .VR = VR, .HDc = 2 * Hc, .NPX = NPX, .HACC16 = HACC16 });
        }
      } else if (x / UW == GAPX and y / UH == GAPY) {
        @set_tile_code(x, y, "relay.csl", .{ .memcpy_params = mp });
      } else if (y >= UNY * UH or (lx == XC and ly == YC)) {
        @set_tile_code(x, y, "filler.csl", .{ .memcpy_params = mp });
      } else if (lx == XC) {
        @set_tile_code(x, y, "cx.csl", .{ .memcpy_params = mp, .K = K, .Hc = Hc, .HOSTX = 0, .XG16 = 0, .DIAG = DIAG, .FIFOW = FIFOW, .XKEEP = XKEEP });
      } else if (ly == YC) {
        @set_tile_code(x, y, "cy.csl", .{ .memcpy_params = mp, .K = K, .KM = KM, .NQA = NQA, .NQG = NQG, .Qh = Qh, .Dc = Dc,
          .DK = DK, .DV = DV, .Ic = Ic, .HR = HR_, .CONV = 4, .NROT = NROT, .HK = HK, .HV = HV, .DIAG = DIAG, .FIFOW = FIFOW, .SILU16 = SILU16, .CONVQK = CONVQK, .ATTV = ATTV, .GDEC = GDEC, .ATTL = ATTL, .TSTAMP = TSTAMP, .GATE = GATE, .SLW = SLW, .RLV = RLV, .RBITS = RBITS });
      } else {
        @set_tile_code(x, y, "weight.csl", .{ .memcpy_params = mp, .K = K, .Hc = Hc, .QO = QO, .Ic = Ic, .MV = 1, .KM = KM,
          .NQA = NQA, .NQG = NQG, .GDN = GDNF, .ATT = ATTF, .DV = DV, .DK = DK, .Qh = Qh, .Dc = Dc, .NE = NE, .PYW = PY, .HD = HD, .DIAG = DIAG, .FIFOW = FIFOW, .GDEC = GDEC, .ATTL = ATTL, .TSTAMP = TSTAMP });
      }
    }
  }''')
for n, t in NAMES:
    L.append(f'  @export_name("{n}", [*]{t}, true);')
L.append('  @export_name("init", fn()void);\n  @export_name("compute", fn(u32, u32)void);\n}')
open(os.path.join(out, "layout.csl"), "w").write("\n".join(L) + "\n")
print("ok", out)
