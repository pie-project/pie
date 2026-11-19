//! The GEMM regime's emitter: a [`GemmProgram`]'s kernel, `layout.csl` and host arrays.
//!
//! - **Every PE** (`gpe.csl`, one file; its place, class and weight segment config words) runs
//!   the program's ops in order, an interpreter over a table of 16-bit words.
//! - **`gseg.csl`**: the all-reduces over a row or a head's band, the printed regions' `seg`.
//! - **The tapes** (`gtape.csl`, rows below the array): a column's weight stream, each PE
//!   sending its words, then forwarding those of the PEs past it up the column's tree.
//! - Colours: the row line, the K / V line, the weight tree, the tape and the all-reduce chain
//!   (two each, by parity), the row's and the band's broadcasts.

use std::fmt::Write as _;

use super::placed::{At, Placed, Span};

use poem_ir::ValueId;

use crate::compile::emit::grid::{GridArray, GridFill};
use crate::compile::emit::stream::COMMON;
use crate::compile::floor::Coord;
use crate::compile::stream::gemm::{GOp, GVal, GemmProgram, RArg, local, piece, tape_spans};
use crate::compile::stream::host::{CLAMP, NONE, ONE, Word};

type R<T> = Result<T, String>;

const C_XF: u32 = 0;
const C_XB: u32 = 1;
const C_WT: u32 = 2;
const C_TA: [u32; 2] = [3, 10];
const C_KF: u32 = 4;
const C_KB: u32 = 5;
const C_A: [u32; 2] = [6, 7];
const C_RB: u32 = 8;
const C_BB: u32 = 9;

const OP_IN: u32 = 1;
const OP_OUT: u32 = 2;
const OP_GEMM: u32 = 3;
const OP_REG: u32 = 5;
const OP_ATTN: u32 = 8;

/// The emitted program.
#[derive(Clone, Debug)]
pub struct Emitted {
    pub files: Vec<(String, String)>,
    /// Its placement (`layout.csl`'s): what the static check reads.
    pub placed: super::placed::Placed,
    pub arrays: Vec<GridArray>,
    pub w: u32,
    pub h: u32,
}

/// The program's table and the host arrays' layouts (words a PE).
pub struct Table {
    /// Each op's words (code, length, its words), streamed in before its weights.
    pub ops: Vec<Vec<u32>>,
    pub consts: Vec<f32>,
    /// (op, value, offset, words a row) of each entry value in `hin`.
    pub hin: Vec<(usize, ValueId, u32, u32)>,
    pub hin_words: u32,
    /// (value, column, offset, classes) of each per-row scalar in `hsc` (`tb` words each).
    pub hsc: Vec<(ValueId, u32, u32, u16)>,
    pub hsc_words: u32,
    /// Each class's words of `hin` and `hsc` (a PE's arrays hold its class's entries alone).
    pub hin_cls: Vec<u32>,
    pub hsc_cls: Vec<u32>,
    /// (value, from, width, offset) of each value read back into `hout` (`tb` rows each).
    pub hout: Vec<(ValueId, u32, u32, u32)>,
    pub hout_words: u32,
    /// Each class's columns a row of `hout` (its values out side by side).
    pub hout_cols: Vec<u32>,
    /// The attention's scratch: scores and the reductions' row (f32 offsets in the arena).
    pub sc: u32,
    pub rt: u32,
    pub xt: u32,
    pub aw: u32,
}

fn slot(p: &GemmProgram, v: ValueId) -> R<u32> {
    p.slots
        .get(&v)
        .copied()
        .ok_or_else(|| format!("value %{}: no slot", v.0))
}

/// `v`'s local width (an even value's: every PE alike).
fn lw(p: &GemmProgram, v: &GVal) -> u32 {
    local(v, p.c, 0).iter().map(|x| x.4).sum()
}

/// The local offset of `v`'s segment from `from`.
fn off(p: &GemmProgram, v: &GVal, from: u32) -> u32 {
    local(v, p.c, 0)
        .iter()
        .find(|x| x.0 == from)
        .map_or(0, |x| x.2)
}

/// A value whose pieces differ by column (a host value read by GEMMs alone).
fn uneven(p: &GemmProgram, v: &GVal) -> bool {
    v.segs().iter().any(|(_, w)| !w.is_multiple_of(p.c))
}

#[allow(clippy::too_many_lines)]
pub fn table(p: &GemmProgram) -> R<Table> {
    // every row's scalar the regions read, once, at the classes of every region reading it
    let mut each: Vec<(ValueId, u32, u16)> = Vec::new();
    for g in &p.l.regs {
        let body = g
            .printed
            .as_ref()
            .and_then(|pr| {
                pr.csl
                    .find("fn row() void {")
                    .map(|i| pr.csl[i..].to_string())
            })
            .unwrap_or_default();
        for (k, a) in g.args.iter().enumerate() {
            if let RArg::Each { v, col } = a
                && body.contains(&format!("sc{k}"))
            {
                match each.iter_mut().find(|e| e.0 == *v && e.1 == *col) {
                    Some(e) => e.2 |= g.mask,
                    None => each.push((*v, *col, g.mask)),
                }
            }
        }
    }
    let mut per_op: Vec<Vec<u32>> = Vec::new();
    let mut hout_cols = vec![0u32; p.l.class_rows.len()];
    let mut consts: Vec<f32> = Vec::new();
    let (mut hin, mut hout) = (Vec::new(), Vec::new());
    let ncls = p.l.class_rows.len();
    let (mut hin_cls, mut hsc_cls) = (vec![0u32; ncls], vec![0u32; ncls]);
    let mut hsc: Vec<(ValueId, u32, u32, u16)> = Vec::new();
    // the first offset free in every class of `mask`, then taken there for `n` words
    let take = |at: &mut Vec<u32>, mask: u16, n: u32| -> u32 {
        let o = (0..at.len())
            .filter(|k| (mask >> k) & 1 != 0)
            .map(|k| at[k])
            .max()
            .unwrap_or(0);
        for (k, x) in at.iter_mut().enumerate() {
            if (mask >> k) & 1 != 0 {
                *x = o + n;
            }
        }
        o
    };
    for &(v, col, mask) in &each {
        let o = take(&mut hsc_cls, mask, p.tb);
        hsc.push((v, col, o, mask));
    }
    let tb = p.tb;
    let all: u32 = (1u32 << p.l.class_rows.len()) - 1;
    for (oi, op) in p.l.ops.iter().enumerate() {
        let mut a: Vec<u32> = Vec::new();
        let code = match op {
            GOp::In { v, mask, lane } => {
                let val = &p.l.vals[v];
                let un = uneven(p, val);
                if un && val.segs().len() > 1 {
                    return Err(format!("value %{}: uneven over segments", v.0));
                }
                let rows = if *lane { 1 } else { tb };
                let most = (0..p.c)
                    .map(|ci| crate::compile::stream::gemm::local_width(val, p.c, ci))
                    .max()
                    .unwrap_or(0);
                let o = take(&mut hin_cls, *mask, rows * most);
                hin.push((oi, *v, o, most));
                a.extend([
                    u32::from(*mask),
                    u32::from(*lane),
                    slot(p, *v)?,
                    o,
                    if un { val.width } else { lw(p, val) },
                    u32::from(un),
                ]);
                OP_IN
            }
            GOp::Out {
                v,
                from,
                width,
                mask,
            } => {
                let val = &p.l.vals[v];
                let un = uneven(p, val);
                if un && val.segs().len() > 1 {
                    return Err(format!("value %{} read back unevenly", v.0));
                }
                // a class's values out side by side a row: this one's column there, each class's
                let n = width.div_ceil(p.c);
                hout.push((*v, *from, *width, 0));
                a.extend([
                    u32::from(*mask),
                    slot(p, *v)? + if un { 0 } else { off(p, val, *from) },
                    if un { 0 } else { lw(p, val) },
                    if un { *width } else { width / p.c },
                    u32::from(un),
                ]);
                for (k, col) in hout_cols.iter_mut().enumerate() {
                    if (mask >> k) & 1 != 0 {
                        if un && *col > 0 {
                            return Err("an uneven value read back beside another".into());
                        }
                        a.push(*col);
                        *col += n;
                    } else {
                        a.push(0);
                    }
                }
                OP_OUT
            }
            GOp::Gemm {
                x,
                from,
                k,
                y,
                x32,
                round,
                mask,
                lane,
                ..
            } => {
                let xv = &p.l.vals[x];
                let yv = &p.l.vals[y];
                let un = uneven(p, xv);
                let uy = uneven(p, yv);
                let fl = u32::from(*round)
                    | (u32::from(*x32) << 1)
                    | (u32::from(un) << 2)
                    | (u32::from(uy) << 3);
                let segs = local(yv, p.c, 0);
                a.extend([
                    u32::from(*mask),
                    u32::from(*lane),
                    slot(p, *x)? + if un { 0 } else { off(p, xv, *from) },
                    if un { 0 } else { lw(p, xv) },
                    *k,
                    slot(p, *y)?,
                    if uy { 0 } else { lw(p, yv) },
                    segs.len() as u32,
                    fl,
                ]);
                for s in &segs {
                    // (an uneven output's piece: the PE's own of its one segment)
                    a.extend([s.2, if uy { s.1 } else { s.4 }]);
                }
                OP_GEMM
            }
            GOp::Region(ri) => {
                let g = &p.l.regs[*ri];
                let pr = g.printed.as_ref().ok_or("an unprinted region")?;
                let ch = g.chunk;
                a.extend([
                    u32::from(g.mask),
                    u32::from(g.lane),
                    g.module as u32,
                    g.args.len() as u32,
                    g.outs.len() as u32,
                    0,
                ]);
                let mut nstream = 0u32;
                // (a row's scalar the region's row() never reads, the lane id that picked a lane
                // vector, is not written)
                let body = pr.csl.find("fn row() void {").map_or("", |i| &pr.csl[i..]);
                for (k, arg) in g.args.iter().enumerate() {
                    match arg {
                        RArg::Row { v, from } => {
                            let val = &p.l.vals[v];
                            a.extend([0, slot(p, *v)? + off(p, val, *from), lw(p, val)]);
                        }
                        RArg::Lane { v, from } => {
                            let val = &p.l.vals[v];
                            a.extend([1, slot(p, *v)? + off(p, val, *from), 0]);
                        }
                        // a static resident comes down the column's stream, f32
                        RArg::Static { .. } => {
                            a.extend([2, 0, 0]);
                            nstream += ch;
                        }
                        // (a row's scalar read again is the one already held)
                        RArg::Each { .. } if !body.contains(&format!("sc{k}")) => {
                            a.extend([5, 0, 0]);
                        }
                        RArg::Each { v, col } => {
                            let at = hsc
                                .iter()
                                .find(|h| h.0 == *v && h.1 == *col)
                                .map_or(0, |h| h.2);
                            a.extend([3, at, 0]);
                        }
                        RArg::One { .. } => {
                            a.extend([4, 0, 0]);
                            nstream += 1;
                        }
                    }
                }
                for o in &g.outs {
                    let val = &p.l.vals[o];
                    a.extend([slot(p, *o)?, lw(p, val)]);
                }
                let _ = pr;
                // (its stream's words, for the rows of other classes to drain)
                a[5] = nstream;
                OP_REG
            }
            GOp::Attend {
                sides,
                scale,
                causal,
                round,
                ..
            } => {
                a.push(sides.len() as u32);
                for s in sides {
                    let q = &p.l.vals[&s.q.0];
                    let k = &p.l.vals[&s.k.0];
                    let v = &p.l.vals[&s.v.0];
                    let o = &p.l.vals[&s.o];
                    a.extend([
                        u32::from(s.mask),
                        slot(p, s.q.0)? + off(p, q, s.q.1),
                        lw(p, q),
                        slot(p, s.k.0)? + off(p, k, s.k.1),
                        lw(p, k),
                        slot(p, s.v.0)? + off(p, v, s.v.1),
                        lw(p, v),
                        slot(p, s.o)?,
                        lw(p, o),
                    ]);
                }
                let kc = consts.len() as u32;
                consts.push(*scale);
                a.extend([
                    p.qchunk,
                    p.band,
                    p.kbm,
                    kc,
                    u32::from(*causal),
                    u32::from(*round),
                ]);
                OP_ATTN
            }
        };
        let mut w = vec![code, a.len() as u32 + 2];
        w.extend(a);
        per_op.push(w);
    }
    let _ = all;
    for w in &per_op {
        if let Some(x) = w.iter().find(|&&x| x > 0xffff) {
            return Err(format!("a program word {x} past 16 bits"));
        }
    }
    let hout_w = tb * hout_cols.iter().copied().max().unwrap_or(0);
    let kb = p.kbm * tb;
    let sc = p.arena;
    let rt = sc + tb * kb;
    let xt = rt
        + kb.max(
            p.l.vals
                .values()
                .map(|v| crate::compile::stream::gemm::local_width(v, p.c, 0) + 1)
                .max()
                .unwrap_or(1),
        );
    let most_x = p.xbn;
    let aw = xt + most_x.max(2);
    Ok(Table {
        ops: per_op,
        consts,
        hin,
        hin_words: hin_cls.iter().copied().max().unwrap_or(0),
        hsc,
        hsc_words: hsc_cls.iter().copied().max().unwrap_or(0),
        hin_cls,
        hsc_cls,
        hout,
        hout_words: hout_w,
        hout_cols,
        sc,
        rt,
        xt,
        aw,
    })
}

/// Each op's words as the stream brings them ([`Table::ops`]).
pub fn op_words(p: &GemmProgram) -> R<Vec<Vec<u32>>> {
    Ok(table(p)?.ops)
}

fn list<T: std::fmt::Display>(v: &[T]) -> String {
    let mut o = String::new();
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            o.push_str(", ");
        }
        let _ = write!(o, "{x}");
    }
    o
}

/// The region dispatch: module `m`'s parameters off the table (a kind and two words each:
/// 0 a row's slot and width, 1 a lane vector's slot, 2 a static resident's offset, 3 a per-row
/// scalar's, 4 a static scalar's), its outputs (slot, width), its rows a call each.
fn reg_fn(p: &GemmProgram) -> String {
    let mut o = String::from(
        "fn region_op(a: u16) void {\n  const b = a + 2;\n  if (((prog[b] >> CLS) & 1) == 0) { wdrain(@as(u32, prog[b + 5])); return; }\n  const rows: u16 = if (prog[b + 1] != 0) 1 else TB;\n  const m = prog[b + 2]; const na = prog[b + 3];\n  const d = b + 6; const od = d + 3 * na;\n",
    );
    for (mi, _) in p.modules.iter().enumerate() {
        let Some(g) = p.l.regs.iter().find(|r| r.module == mi) else {
            continue;
        };
        let ch = g.chunk;
        let _ = writeln!(o, "  if (m == {mi}) {{");
        // (the residents and static scalars in the parameters' order: the stream's)
        for (k, at) in g.params.iter().enumerate() {
            match at {
                crate::compile::emit::region::ParamAt::Resident => {
                    let _ = writeln!(
                        o,
                        "    if (prog[d + {0}] == 1) {{ @fmovs(rm{mi}.IN{k}, md(prog[d + {0} + 1], {ch})); }} else {{ @fmovs(rm{mi}.IN{k}, fab.fin(q_wi, {ch})); }}",
                        3 * k
                    );
                }
                crate::compile::emit::region::ParamAt::Scalar => {
                    let _ = writeln!(
                        o,
                        "    if (prog[d + {0}] == 4) {{ @fmovs(md(XT, 1), fab.fin(q_wi, 1)); rm{mi}.sc{k} = mem[XT]; }}",
                        3 * k
                    );
                }
                crate::compile::emit::region::ParamAt::Row => {}
            }
        }
        o.push_str("    var t: u16 = 0;\n    while (t < rows) : (t += 1) {\n");
        for (k, at) in g.params.iter().enumerate() {
            match at {
                crate::compile::emit::region::ParamAt::Row => {
                    let _ = writeln!(
                        o,
                        "      @fmovs(rm{mi}.IN{k}, md(prog[d + {0} + 1] + t * prog[d + {0} + 2], {ch}));",
                        3 * k
                    );
                }
                crate::compile::emit::region::ParamAt::Scalar => {
                    let _ = writeln!(
                        o,
                        "      if (prog[d + {0}] == 3) {{ rm{mi}.sc{k} = hsc[prog[d + {0} + 1] + t]; }}",
                        3 * k
                    );
                }
                crate::compile::emit::region::ParamAt::Resident => {}
            }
        }
        let _ = writeln!(o, "      rm{mi}.row();");
        for q in 0..g.outs.len() {
            let _ = writeln!(
                o,
                "      @fmovs(md(prog[od + {0}] + t * prog[od + {0} + 1], {ch}), rm{mi}.OUT{q});",
                2 * q
            );
        }
        o.push_str("    }\n  }\n");
    }
    o.push_str("}\n");
    o
}

/// The GEMM regime's PE ops (`gemm_pe.csl`): every PE's code but its config and its modules.
pub const GPE_OPS: &str = include_str!("../../../kernels/runtime/gemm_pe.csl");

/// `gseg.csl`: the all-reduces of the array's rows and heads' bands.
fn gseg_file() -> String {
    String::from(
        "// The GEMM regime's all-reduces (kernels-cerebras emit::gemm): over a row's\n\
         // PEs (a row's sums: `all_reduce`) or a head's band of them (`head_reduce`, the attention's\n\
         // scores): a chain east by parity colours (fadds in and out), the last PE's sum back west\n\
         // over the row's or the band's broadcast colour. The printed regions' `seg`.\n\
         param fab: imported_module;\n\
         param C_A0: u16;\nparam C_A1: u16;\nparam C_RB: u16;\nparam C_BB: u16;\n\
         const q_ai = @get_input_queue(4);\nconst q_bi = @get_input_queue(5);\nconst q_ao = @get_output_queue(4);\nconst q_bo = @get_output_queue(5);\n\
         const tile_config = @import_module(\"<tile_config>\");\n\
         var X: u16 = 0;\nvar GX: u16 = 1;\nvar BAND: u16 = 1;\nvar mode: u16 = 2;\n\
         fn setmode(m: u16) void {\n  if (m == mode) { return; }\n  while ((tile_config.output_queue_status.get_raw() & 0x2000) == 0) {}\n  const c = if (m == 0) C_RB else C_BB;\n  fab.iqc(5, c); fab.oqc(5, c);\n  mode = m;\n}\n\
         fn setup(x: u16, gx: u16, band: u16) void {\n  X = x; GX = gx; BAND = band;\n  fab.oqc(4, if ((x & 1) == 0) C_A0 else C_A1);\n  fab.iqc(4, if ((x & 1) == 0) C_A1 else C_A0);\n  mode = 2;\n  setmode(0);\n}\n\
         fn allred(d: mem1d_dsd, n: u16, band: bool) void {\n  const lo = if (band) (X / BAND) * BAND else 0;\n  const hi = if (band) lo + BAND - 1 else GX - 1;\n  if (lo == hi) { return; }\n  setmode(if (band) 1 else 0);\n  const d2 = @set_dsd_length(d, n);\n  if (X == lo) { @fmovs(fab.fout(q_ao, n), d2); }\n  else if (X < hi) { @fadds(fab.fout(q_ao, n), fab.fin(q_ai, n), d2); }\n  else { @fadds(d2, d2, fab.fin(q_ai, n)); }\n  if (X == hi) { @fmovs(fab.fout(q_bo, n), d2); } else { @fmovs(d2, fab.fin(q_bi, n)); }\n}\n\
         fn all_reduce(d: mem1d_dsd, n: u16) void { allred(d, n, false); }\n\
         fn head_reduce(d: mem1d_dsd, n: u16) void { allred(d, n, true); }\n\
         comptime {\n  @initialize_queue(q_ai, .{ .color = @get_color(C_A1) });\n  @initialize_queue(q_bi, .{ .color = @get_color(C_RB) });\n  @initialize_queue(q_ao, .{ .color = @get_color(C_A0) });\n  @initialize_queue(q_bo, .{ .color = @get_color(C_RB) });\n}\n",
    )
}

#[allow(clippy::too_many_lines)]
fn pe_file(p: &GemmProgram, tb: &Table) -> String {
    let mut t = format!(
        "// {}: a PE of the GEMM regime (kernels-cerebras emit::gemm): the program's\n// ops over its rows and columns, in step with the array.\nparam memcpy_params;\n",
        p.l.reading
    );
    t.push_str(COMMON);
    let kb = p.kbm * p.tb;
    let nc = p.qchunk;
    let xbh = 2 * p.xbn.div_ceil(2).max(1) + 2;
    let _ = writeln!(
        t,
        "const q_xi = @get_input_queue(2);\nconst q_xb = @get_input_queue(7);\nconst q_wi = @get_input_queue(3);\nconst q_ki = @get_input_queue(6);\nconst q_xo = @get_output_queue(2);\nconst q_co = @get_output_queue(3);\n\
         const matvec = @import_module(\"ops/mv.csl\");\n\
         const gseg = @import_module(\"gseg.csl\", .{{ .fab = fab, .C_A0 = {}, .C_A1 = {}, .C_RB = {C_RB}, .C_BB = {C_BB} }});\n\
         const C_XF: u16 = {C_XF};\nconst C_XB: u16 = {C_XB};\nconst C_WT: u16 = {C_WT};\nconst C_KF: u16 = {C_KF};\nconst C_KB: u16 = {C_KB};",
        C_A[0], C_A[1]
    );
    for (mi, (name, text)) in p.modules.iter().enumerate() {
        let _ = writeln!(
            t,
            "const rm{mi} = @import_module(\"{name}\", .{{ .math = math, .seg = gseg }});"
        );
        let _ = text;
    }
    let _ = writeln!(
        t,
        "const GX: u16 = {};\nconst GY: u16 = {};\nconst RQ: u16 = {};\nvar R0: u16 = 0;\nconst TB: u16 = {};\nconst BAND: u16 = {};\nconst HROW = [{}]u16 {{ {} }};\nconst NOPS: u16 = {};\nconst OPN: u16 = {};\nvar prog = @zeros([OPN]u16);\nconst D_prog = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{OPN}} -> prog[i] }});\nvar D_progw = D_prog;\nconst fk = [{}]f32 {{ {} }};",
        p.c,
        p.r,
        p.r / p.l.batch.max(1),
        p.tb,
        p.band,
        tb.hout_cols.len(),
        list(&tb.hout_cols),
        tb.ops.len(),
        tb.ops
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(2)
            .next_multiple_of(2)
            .max(2),
        tb.consts.len().max(1),
        if tb.consts.is_empty() {
            "0.0".to_string()
        } else {
            tb.consts
                .iter()
                .map(|c| format!("{c:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        }
    );
    let wbn = p.wbn.max(2);
    let (hin, hsc, hout) = (
        tb.hin_words.max(1),
        tb.hsc_words.max(1),
        tb.hout_words.max(1),
    );
    let _ = writeln!(
        t,
        "const AW: u16 = {aw};\nvar mem: [AW]f32 align(16) = @zeros([AW]f32);\nconst D_mem = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{AW}} -> mem[i] }});\n\
         const SC: u16 = {sc};\nconst RT: u16 = {rt};\nconst XT: u16 = {xt};\n\
         const XBN: u16 = {xbh};\nvar xh: [XBN]@fp16() align(16) = @zeros([XBN]@fp16());\nconst D_xh = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{XBN}} -> xh[i] }});\n\
         const WBN: u16 = {wbn};\nvar wb: [WBN]u32 align(16) = @zeros([WBN]u32);\nconst D_wb = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{WBN}} -> wb[i] }});\n\
         const D_wa0 = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{XBN}} -> xh[i] }});\nvar D_xhw = D_wb;\n\
         var kt: [{ktn}]@fp16() align(16) = @zeros([{ktn}]@fp16());\nconst D_kt = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{{ktn}}} -> kt[i] }});\n\
         var vv: [{ktn}]@fp16() align(16) = @zeros([{ktn}]@fp16());\nconst D_vv = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{{ktn}}} -> vv[i] }});\n\
         var qh: [{qhn}]@fp16() align(16) = @zeros([{qhn}]@fp16());\nconst D_qh = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{{qhn}}} -> qh[i] }});\n\
         var kvh: [{kvn}]@fp16() align(16) = @zeros([{kvn}]@fp16());\nconst D_kvh = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{{kvn}}} -> kvh[i] }});\nvar D_kvw = D_wb;\n\
         var ph: [{phn}]@fp16() align(16) = @zeros([{phn}]@fp16());\nconst D_ph = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{{phn}}} -> ph[i] }});\n\
         var mm = @zeros([TB]f32);\nvar ll = @zeros([TB]f32);\n\
         var hin: [{hin}]f32 align(16) = @zeros([{hin}]f32);\nvar hin_p: [*]f32 = &hin;\nconst D_hin = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{{hin}}} -> hin[i] }});\n\
         var hsc: [{hsc}]f32 = @zeros([{hsc}]f32);\nvar hsc_p: [*]f32 = &hsc;\n\
         var hout: [{hout}]f32 align(16) = @zeros([{hout}]f32);\nvar hout_p: [*]f32 = &hout;\nconst D_hout = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{{hout}}} -> hout[i] }});\n\
         var pcfg = @zeros([4]u32);\nvar pcfg_p: [*]u32 = &pcfg;",
        aw = tb.aw,
        sc = tb.sc,
        rt = tb.rt,
        xt = tb.xt,
        ktn = (nc * kb).max(2),
        qhn = (p.tb * nc).max(2),
        kvn = (2 * p.tb * nc).max(2),
        phn = kb.max(2),
    );
    // (a weight block of an odd count of 16-bit weights takes whole words, the last half a pad)
    if p.odd {
        t.push_str(
            &GPE_OPS
                .replace(
                    "tot += @as(u32, kc) * @as(u32, if (uy) ylw else prog[b + 10 + 2 * s]) / 2;",
                    "tot += (@as(u32, kc) * @as(u32, if (uy) ylw else prog[b + 10 + 2 * s]) + 1) / 2;",
                )
                .replace("wget(0, kc * p / 2);", "wget(0, (kc * p + 1) / 2);"),
        );
    } else {
        t.push_str(GPE_OPS);
    }
    t.push_str(&reg_fn(p));
    // the regions' tables, the host's to write
    let mut exports = Vec::new();
    for (mi, _) in p.modules.iter().enumerate() {
        let Some(g) = p.l.regs.iter().find(|r| r.module == mi) else {
            continue;
        };
        for tbl in &g
            .printed
            .as_ref()
            .map(|x| x.tables.clone())
            .unwrap_or_default()
        {
            let ty = if tbl.u16 { "u16" } else { "f32" };
            let sym = format!("t{mi}_{}", tbl.name);
            let _ = writeln!(t, "var {sym}_p: [*]{ty} = &rm{mi}.T_{}_a;", tbl.name);
            exports.push(sym);
        }
    }
    let mut init = String::new();
    for (mi, (_, text)) in p.modules.iter().enumerate() {
        if text.contains("fn init(") {
            let _ = writeln!(init, "  rm{mi}.init();");
        }
    }
    let _ = write!(
        t,
        "task go() void {{\n  stamp(0);\n  X = @as(u16, pcfg[0]);\n  Y = @as(u16, pcfg[1]);\n  CLS = @as(u16, pcfg[2]);\n  R0 = @as(u16, pcfg[3]) * RQ;\n  D_xhw = @set_dsd_base_addr(D_wb, @ptrcast([*]u32, &xh[0]));\n  D_progw = @set_dsd_base_addr(D_wb, @ptrcast([*]u32, &prog[0]));\n  D_kvw = @set_dsd_base_addr(D_wb, @ptrcast([*]u32, &kvh[0]));\n  gseg.setup(X, GX, BAND);\n  fab.iqc(2, C_XF);\n  fab.iqc(6, C_KF);\n{init}  run_all();\n  stamp(1);\n  // (a line's words left queued: the next launch would read them first)\n  if ((tile_config.input_queue_status.get_raw() & LINE_QUEUES) != LINE_QUEUES) {{ tsc[2] |= LINE_LEFT; }}\n  sys_mod.unblock_cmd_stream();\n}}\n"
    );
    t.push_str("comptime {\n  @bind_local_task(go, T_GO);\n  @export_symbol(run);\n  @export_symbol(tsc_p, \"tsc\");\n  @export_symbol(pcfg_p, \"pcfg\");\n  @export_symbol(hin_p, \"hin\");\n  @export_symbol(hsc_p, \"hsc\");\n  @export_symbol(hout_p, \"hout\");\n");
    for s in &exports {
        let _ = writeln!(t, "  @export_symbol({s}_p, \"{s}\");");
    }
    t.push_str("  @initialize_queue(q_xi, .{ .color = @get_color(C_XF) });\n  @initialize_queue(q_xb, .{ .color = @get_color(C_XB) });\n  @initialize_queue(q_wi, .{ .color = @get_color(C_WT) });\n  @initialize_queue(q_ki, .{ .color = @get_color(C_KF) });\n  @initialize_queue(q_xo, .{ .color = @get_color(C_XF) });\n  @initialize_queue(q_co, .{ .color = @get_color(C_KF) });\n}\n");
    t
}

/// A tape PE's image past its tape's words, as a link measures it: the check's estimate of it
/// misses by ~500 B, past the stack's room. Re-measure on a change to `gtape.csl`'s code.
pub const TAPE_IMAGE_BYTES: i64 = 4544;

/// Words a tape PE holds: the room `gtape.csl`'s image leaves past its own bytes, the stack
/// and the margin.
#[must_use]
pub fn tape_words() -> u32 {
    crate::compile::check::tape_words(TAPE_IMAGE_BYTES)
}

fn tape_file(tw: u32) -> String {
    let mut t = String::from(
        "// The GEMM regime's weight tape (kernels-cerebras emit::gemm): a PE's words of its column's\n// stream, then the words of the PEs past it forwarded.\nparam memcpy_params;\nparam C_O: u16;\nparam C_I: u16;\n",
    );
    t.push_str(COMMON);
    let _ = writeln!(
        t,
        "const q_i = @get_input_queue(2);\nconst q_o = @get_output_queue(2);\nconst TW: u16 = {tw};\nvar tape = @zeros([TW]u32);\nvar tape_p: [*]u32 = &tape;\nconst D_tape = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{TW}} -> tape[i] }});\nvar pcfg = @zeros([2]u32);\nvar pcfg_p: [*]u32 = &pcfg;"
    );
    t.push_str(
        "task go() void {\n  stamp(0);\n  const own = @as(u16, pcfg[0]);\n  if (own > 0) { @mov32(fab.fout(q_o, own), @set_dsd_length(D_tape, own)); }\n  stamp(2);\n  var left: u32 = pcfg[1];\n  while (left > 0) {\n    const n: u16 = if (left > 16384) 16384 else @as(u16, left);\n    @mov32(fab.fout(q_o, n), fab.fin(q_i, n));\n    left -= @as(u32, n);\n  }\n  stamp(1);\n  sys_mod.unblock_cmd_stream();\n}\n",
    );
    t.push_str("comptime {\n  @bind_local_task(go, T_GO);\n  @export_symbol(run);\n  @export_symbol(tsc_p, \"tsc\");\n  @export_symbol(pcfg_p, \"pcfg\");\n  @export_symbol(tape_p, \"tape\");\n  @initialize_queue(q_i, .{ .color = @get_color(C_I) });\n  @initialize_queue(q_o, .{ .color = @get_color(C_O) });\n}\n");
    t
}

/// The table symbols' types, for `layout.csl`'s exports.
fn table_syms(p: &GemmProgram) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    for (mi, _) in p.modules.iter().enumerate() {
        let Some(g) = p.l.regs.iter().find(|r| r.module == mi) else {
            continue;
        };
        for tbl in g
            .printed
            .as_ref()
            .map(|x| x.tables.clone())
            .unwrap_or_default()
        {
            out.push((format!("t{mi}_{}", tbl.name), tbl.u16));
        }
    }
    out
}

/// The program's placement: the array's PEs and the tapes' below them, every line's and tree's
/// routes, the host's symbols.
#[must_use]
pub fn placed(p: &GemmProgram) -> Placed {
    let (w, h) = (p.c, p.r);
    let fh = h + p.tape;
    let mut l = Placed::new(
        format!(
            "// A GEMM-regime reading's wafer program (kernels-cerebras emit::gemm): an array of {w} x {h} PEs, {} rows of tapes below it.",
            p.tape
        ),
        w,
        fh,
    );
    let ax = (0, w, 1);
    let ay = (0, h, 1);
    let at = |xs: Span, ys: Span| At::Region(xs, ys);
    l.tile(at(ax, ay), "gpe.csl", &[]);
    for k in 0..p.tape {
        let co = if k == 0 { C_WT } else { C_TA[(k % 2) as usize] };
        let ci = C_TA[((k + 1) % 2) as usize];
        l.tile(
            at(ax, (h + k, h + k + 1, 1)),
            "gtape.csl",
            &[("C_O", co.to_string()), ("C_I", ci.to_string())],
        );
    }
    let mut rule = |c: u32, xs: Span, ys: Span, rx: &'static str, tx: &[&'static str]| {
        l.route(c, at(xs, ys), rx, tx);
    };
    // the row line: east (pass on to the ramp; the last to the ramp), west
    rule(C_XF, (0, w - 1, 1), ay, "WEST", &["EAST", "RAMP"]);
    rule(C_XF, (w - 1, w, 1), ay, "WEST", &["RAMP"]);
    rule(C_XB, (0, 1, 1), ay, "EAST", &["RAMP"]);
    rule(C_XB, (1, w, 1), ay, "EAST", &["WEST", "RAMP"]);
    // the column line, a request's rows of PEs each: south, north
    let rq = h / p.l.batch.max(1);
    for b in 0..p.l.batch.max(1) {
        let (y0, y1) = (b * rq, (b + 1) * rq);
        rule(C_KF, ax, (y0, y1 - 1, 1), "NORTH", &["SOUTH", "RAMP"]);
        rule(C_KF, ax, (y1 - 1, y1, 1), "NORTH", &["RAMP"]);
        rule(C_KB, ax, (y0, y0 + 1, 1), "SOUTH", &["RAMP"]);
        rule(C_KB, ax, (y0 + 1, y1, 1), "SOUTH", &["NORTH", "RAMP"]);
    }
    // the weight tree: the tape's head north, up the column to every PE's ramp
    rule(C_WT, ax, (h, h + 1, 1), "RAMP", &["NORTH"]);
    rule(C_WT, ax, (1, h, 1), "SOUTH", &["NORTH", "RAMP"]);
    rule(C_WT, ax, (0, 1, 1), "SOUTH", &["RAMP"]);
    // the tape: PE k sends north to k - 1 on C_TA[k % 2]
    for k in 1..p.tape {
        let c = C_TA[(k % 2) as usize];
        rule(c, ax, (h + k, h + k + 1, 1), "RAMP", &["NORTH"]);
        rule(c, ax, (h + k - 1, h + k, 1), "SOUTH", &["RAMP"]);
    }
    // the all-reduce chain: x sends east on A[x % 2], receives from the west on the other
    for q in 0..2u32 {
        let c = C_A[q as usize];
        rule(c, (q, w - 1, 2), ay, "RAMP", &["EAST"]);
        rule(c, (1 + q, w, 2), ay, "WEST", &["RAMP"]);
    }
    // the broadcasts: from a segment's last PE west
    let mut bcast = |c: u32, lo: u32, hi: u32| {
        if hi <= lo {
            return;
        }
        rule(c, (hi, hi + 1, 1), ay, "RAMP", &["WEST"]);
        rule(c, (lo + 1, hi, 1), ay, "EAST", &["WEST", "RAMP"]);
        rule(c, (lo, lo + 1, 1), ay, "EAST", &["RAMP"]);
    };
    bcast(C_RB, 0, w - 1);
    let b = p.band.max(1);
    for s in 0..w / b {
        bcast(C_BB, s * b, s * b + b - 1);
    }
    for (n, ty) in [
        ("tsc", "u32"),
        ("pcfg", "u32"),
        ("tape", "u32"),
        ("hin", "f32"),
        ("hsc", "f32"),
        ("hout", "f32"),
    ] {
        l.export(n, ty);
    }
    for (s, u16) in table_syms(p) {
        l.export(&s, if u16 { "u16" } else { "f32" });
    }
    l
}

/// The array PE's files (`gpe.csl` and what it imports: the printed regions, `gseg.csl`, the
/// ops), as [`emit`] writes them: what a check of its image reads before anything is placed.
pub fn pe_files(p: &GemmProgram) -> R<std::collections::BTreeMap<String, String>> {
    let tb = table(p)?;
    let mut files: std::collections::BTreeMap<String, String> = p.modules.iter().cloned().collect();
    files.insert("gseg.csl".into(), gseg_file());
    files.insert("gpe.csl".into(), pe_file(p, &tb));
    for (n, s) in crate::sources::OPS
        .iter()
        .chain(crate::sources::stream_ops())
    {
        files.insert(format!("ops/{n}"), (*s).to_string());
    }
    Ok(files)
}

/// The emitted program.
#[allow(clippy::too_many_lines)]
pub fn emit(p: &GemmProgram) -> R<Emitted> {
    let tb = table(p)?;
    let mut files: Vec<(String, String)> = p.modules.clone();
    files.push(("gseg.csl".into(), gseg_file()));
    files.push(("gpe.csl".into(), pe_file(p, &tb)));
    files.push(("gtape.csl".into(), tape_file(tape_words())));
    let pl = placed(p);
    files.push(("layout.csl".into(), pl.csl()));
    let mut arrays = Vec::new();
    let ieee = !p.l.round16;
    for x in 0..p.c {
        let total = p.wcol[x as usize];
        let mut sent = 0u32;
        for k in 0..p.tape {
            let own = tape_words().min(total - sent);
            sent += own;
            let at = Coord::new(x, p.r + k);
            arrays.push(GridArray {
                at,
                sym: "tape".into(),
                u16: false,
                len: tape_words(),
                fill: GridFill::Spans(tape_spans(p, x, k)),
                group: None,
            });
            arrays.push(GridArray {
                at,
                sym: "pcfg".into(),
                u16: false,
                len: 2,
                fill: GridFill::Words(vec![own, total - sent]),
                group: None,
            });
        }
    }
    for y in 0..p.r {
        let cls = p.row_class[y as usize];
        let req = p.row_req[y as usize];
        let step_row = |t: u32| y * p.tb + t;
        for x in 0..p.c {
            let at = Coord::new(x, y);
            arrays.push(GridArray {
                at,
                sym: "pcfg".into(),
                u16: false,
                len: 4,
                fill: GridFill::Words(vec![x, y, cls as u32, req]),
                group: None,
            });
            // the step's entries: the host's values on the PE's rows, the rows' scalars
            if tb.hin_cls[cls] > 0 {
                let mut ent: Vec<Word> = vec![[NONE, 0, 0, 0]; tb.hin_cls[cls] as usize];
                for &(oi, v, o, most) in &tb.hin {
                    let GOp::In { mask, lane, .. } = &p.l.ops[oi] else {
                        continue;
                    };
                    if (mask >> cls) & 1 == 0 {
                        continue;
                    }
                    let val = &p.l.vals[&v];
                    let rows = if *lane { 1 } else { p.tb };
                    let cl = if ieee { CLAMP } else { 0 };
                    let lw: u32 = local(val, p.c, x).iter().map(|s| s.4).sum();
                    for tt in 0..rows {
                        // (a lane vector: a row a lane, a request's lanes in turn: its first's)
                        let lanes = p.l.class_rows.len() as u32;
                        let (row, one) = if *lane {
                            (req * lanes, 0)
                        } else {
                            (step_row(tt), ONE)
                        };
                        let mut e = 0;
                        for s in local(val, p.c, x) {
                            for i in 0..s.4 {
                                ent[(o + tt * lw + e) as usize] =
                                    [v.0, row, s.0 + s.3 + i, one | cl];
                                e += 1;
                            }
                        }
                    }
                    let _ = most;
                }
                arrays.push(GridArray {
                    at,
                    sym: "hin".into(),
                    u16: false,
                    len: tb.hin_cls[cls],
                    fill: GridFill::Entry(ent),
                    group: None,
                });
            }
            if tb.hsc_cls[cls] > 0 {
                let mut ent: Vec<Word> = vec![[NONE, 0, 0, 0]; tb.hsc_cls[cls] as usize];
                for &(v, col, o, mask) in &tb.hsc {
                    if (mask >> cls) & 1 == 0 {
                        continue;
                    }
                    for tt in 0..p.tb {
                        ent[(o + tt) as usize] = [v.0, step_row(tt), col, ONE];
                    }
                }
                arrays.push(GridArray {
                    at,
                    sym: "hsc".into(),
                    u16: false,
                    len: tb.hsc_cls[cls],
                    fill: GridFill::Entry(ent),
                    group: None,
                });
            }
            // the regions' tables: the PE's chunk's
            for (mi, _) in p.modules.iter().enumerate() {
                let Some(g) = p.l.regs.iter().find(|r| r.module == mi) else {
                    continue;
                };
                for tbl in g
                    .printed
                    .as_ref()
                    .map(|x| x.tables.clone())
                    .unwrap_or_default()
                {
                    let n = tbl.of_chunk.iter().map(Vec::len).max().unwrap_or(1).max(1) as u32;
                    // (a head region's chunk its place in the head's band)
                    let row = &tbl.of_chunk[x as usize % tbl.of_chunk.len().max(1)];
                    let mut ws: Vec<u32> = row
                        .iter()
                        .map(|&v| if tbl.u16 { v as u32 } else { v.to_bits() })
                        .collect();
                    ws.resize(n as usize, 0);
                    arrays.push(GridArray {
                        at,
                        sym: format!("t{mi}_{}", tbl.name),
                        u16: tbl.u16,
                        len: n,
                        fill: GridFill::Words(ws),
                        group: None,
                    });
                }
            }
            // what the host reads back: the PE's rows of each value out
            let mut parts: Vec<(u32, u32, u32)> = Vec::new();
            for op in &p.l.ops {
                let GOp::Out {
                    v,
                    from,
                    width,
                    mask,
                } = op
                else {
                    continue;
                };
                if (mask >> cls) & 1 == 0 {
                    continue;
                }
                let (lo, hi) = piece(x, p.c, *width);
                if hi > lo {
                    parts.push((v.0, from + lo, hi - lo));
                }
            }
            if !parts.is_empty() {
                let rows: Vec<usize> = (0..p.tb).map(|tt| step_row(tt) as usize).collect();
                let n: u32 = parts.iter().map(|x| x.2).sum();
                arrays.push(GridArray {
                    at,
                    sym: "hout".into(),
                    u16: false,
                    len: p.tb * n,
                    fill: GridFill::Back { rows, parts },
                    group: None,
                });
            }
        }
    }
    let (w, h) = p.fabric();
    Ok(Emitted {
        files,
        placed: pl,
        arrays,
        w,
        h,
    })
}
