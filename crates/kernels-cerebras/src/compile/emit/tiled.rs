//! The tiled regime's emitter: a [`TileProgram`]'s kernels, `layout.csl` and host arrays.
//!
//! - **The image PEs** (`tpe.csl`, one file) run the program's ops in order, an interpreter over
//!   a table of 16-bit words.
//! - **The tape** (`ttape.csl`): each PE sends its words, then forwards those of the PEs past it.
//! - **The hub** (`thub.csl`, east of the array's last row): the all-reduces' sums and the
//!   attention's keys up the last column, back out along the broadcast tree.
//! - The spine and the rest: routes only (`filler.csl`).
//!
//! Colours: the shifts E, W, S, N two each (by parity), the weight tree, the broadcast tree, the
//! tape two.

use std::fmt::Write as _;

use super::placed::{At, Placed, Span};

use poem_ir::ValueId;

use crate::compile::emit::grid::{GridArray, GridFill};
use crate::compile::emit::stream::COMMON;
use crate::compile::floor::Coord;
use crate::compile::stream::grid::{RegIn, Res};
use crate::compile::stream::host::{NONE, Word};
use crate::compile::stream::tiled::{Key, Lay, TOp, TileProgram, tape_spans};

type R<T> = Result<T, String>;

const E: [u32; 2] = [0, 1];
const W: [u32; 2] = [2, 3];
const S: [u32; 2] = [4, 5];
const N: [u32; 2] = [6, 7];
const T: u32 = 8;
const B: u32 = 9;
const A: [u32; 2] = [10, 11];

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

/// Op codes (the image PE's interpreter).
const OP_IN: u32 = 1;
const OP_OUT: u32 = 2;
const OP_CONV: u32 = 3;
const OP_NORM: u32 = 4;
const OP_REG: u32 = 5;
const OP_UP: u32 = 6;
const OP_ATTN: u32 = 8;

fn lay_words(l: Lay) -> [u32; 6] {
    [l.th, l.tw, l.lsy, l.lsx, l.nrb, l.ncb]
}

/// The program's layouts (an op names one by its index in the table).
#[derive(Default)]
struct Lays(Vec<Lay>);

impl Lays {
    fn of(&mut self, l: Lay) -> u32 {
        if let Some(i) = self.0.iter().position(|x| *x == l) {
            return i as u32;
        }
        self.0.push(l);
        self.0.len() as u32 - 1
    }
}

/// The program's words and f32 constants, and each host value's and output's place in `hin` /
/// `hout` (words a busy PE).
struct Table {
    words: Vec<u32>,
    lays: Vec<u32>,
    consts: Vec<f32>,
    hin: Vec<(ValueId, u32)>,
    hout: Vec<(ValueId, u32)>,
    hin_words: u32,
    hout_words: u32,
}

/// A region call's descriptor: its inputs (offset, a row's or not), its outputs.
fn desc(p: &TileProgram, r: usize, res_at: &dyn Fn(usize) -> u32) -> R<Vec<u32>> {
    let g = &p.regs[r];
    let mut d = vec![
        g.ins.len() as u32,
        g.outs.len() as u32,
        g.region.width,
        g.chunk,
    ];
    let mut k = 0;
    for i in &g.ins {
        match i {
            RegIn::Row(v) => d.extend([slot(p, *v)?, 1]),
            RegIn::Res(Res::Value { .. }) => {
                d.extend([res_at(k), 0]);
                k += 1;
            }
            RegIn::Res(Res::GnScale(_)) => d.extend([res_at(0), 0]),
            RegIn::Res(Res::GnShift(_)) => d.extend([res_at(1), 0]),
        }
    }
    for o in &g.outs {
        d.push(slot(p, *o)?);
    }
    Ok(d)
}

fn slot(p: &TileProgram, v: ValueId) -> R<u32> {
    p.slots
        .get(&Key::Val(v))
        .copied()
        .ok_or_else(|| format!("value %{}: no slot", v.0))
}

fn temp(p: &TileProgram, i: usize) -> u32 {
    p.slots.get(&Key::Temp(i)).copied().unwrap_or(0)
}

/// Each op's words (its code, its length, its arguments), the op table cut at its ops.
pub fn op_words(p: &TileProgram) -> R<Vec<Vec<u32>>> {
    let words = table(p)?.words;
    let mut out = Vec::with_capacity(p.ops.len());
    let mut a = 0usize;
    while a + 1 < words.len() {
        let n = words[a + 1] as usize;
        out.push(words[a..a + n].to_vec());
        a += n;
    }
    Ok(out)
}

#[allow(clippy::too_many_lines)]
fn table(p: &TileProgram) -> R<Table> {
    let mut words = Vec::new();
    let mut consts: Vec<f32> = Vec::new();
    let mut hin = Vec::new();
    let mut hout = Vec::new();
    let (mut hin_words, mut hout_words) = (0, 0);
    let mut ly = Lays::default();
    for (i, op) in p.ops.iter().enumerate() {
        let mut a: Vec<u32> = Vec::new();
        let code = match op {
            TOp::In(v) => {
                hin.push((*v, hin_words));
                a.extend([slot(p, *v)?, hin_words, p.vals[v].width]);
                a.push(ly.of(p.lay_of(*v)));
                hin_words += p.words(*v);
                OP_IN
            }
            TOp::Out(v) => {
                hout.push((*v, hout_words));
                a.extend([slot(p, *v)?, hout_words, p.vals[v].width]);
                a.push(ly.of(p.lay_of(*v)));
                hout_words += p.words(*v);
                OP_OUT
            }
            TOp::Conv {
                x,
                y,
                bias,
                k,
                cin,
                cout,
                ..
            } => {
                let (kb, kc) = (p.kb[i], p.kc[i]);
                let fl = u32::from(bias.is_some()) | (u32::from(p.vals[y].round) << 1);
                a.extend([
                    slot(p, *x)?,
                    slot(p, *y)?,
                    *cin,
                    *cout,
                    *k,
                    kb,
                    kc / kb,
                    fl,
                    temp(p, i),
                    kc,
                ]);
                a.push(ly.of(p.lay_of(*x)));
                OP_CONV
            }
            TOp::Norm {
                x,
                y,
                groups,
                eps,
                reg,
                ..
            } => {
                let c = p.vals[x].width;
                let [_, h, w] = p.vals[x].grid;
                let kc = consts.len() as u32;
                consts.push(1.0 / (f64::from(h * w) * f64::from(c / groups)) as f32);
                consts.push(*eps);
                let tp = temp(p, i);
                a.extend([
                    slot(p, *x)?,
                    slot(p, *y)?,
                    c,
                    *groups,
                    c / groups,
                    kc,
                    p.regs[*reg].module as u32,
                    tp,
                ]);
                a.push(ly.of(p.lay_of(*x)));
                a.extend(desc(p, *reg, &|k| tp + k as u32 * c)?);
                OP_NORM
            }
            TOp::Region(r) => {
                let g = &p.regs[*r];
                let nres = g.ins.iter().filter(|i| matches!(i, RegIn::Res(_))).count() as u32;
                let c = g.region.width;
                let tp = temp(p, i);
                a.extend([g.module as u32, nres, tp, c]);
                a.push(ly.of(p.lay_of(g.outs[0])));
                a.extend(desc(p, *r, &|k| tp + k as u32 * c)?);
                OP_REG
            }
            TOp::Up { x, y, axis, sel } => {
                let rd = u32::from(p.vals[y].round && !p.vals[x].round);
                a.extend([
                    slot(p, *x)?,
                    slot(p, *y)?,
                    p.vals[x].width,
                    p.vals[y].width,
                    *axis as u32,
                    u32::from(*sel),
                    rd,
                ]);
                a.push(ly.of(p.lay_of(*x)));
                a.push(ly.of(p.lay_of(*y)));
                OP_UP
            }
            TOp::Attend { q, k, v, o, scale } => {
                let kc = consts.len() as u32;
                consts.push(*scale);
                a.extend([
                    slot(p, *q)?,
                    slot(p, *k)?,
                    slot(p, *v)?,
                    slot(p, *o)?,
                    p.vals[q].width,
                    kc,
                    u32::from(p.vals[o].round),
                    temp(p, i),
                ]);
                a.push(ly.of(p.lay_of(*q)));
                OP_ATTN
            }
        };
        words.push(code);
        words.push(a.len() as u32 + 2);
        words.extend(a);
    }
    if words.iter().any(|&w| w > 0xffff) {
        return Err("a program word past 16 bits".into());
    }
    Ok(Table {
        words,
        lays: ly.0.iter().flat_map(|l| lay_words(*l)).collect(),
        consts,
        hin,
        hout,
        hin_words,
        hout_words,
    })
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

/// The region dispatch: module `m` over a pixel `p`'s row, a chunk at a time, its descriptor at
/// `prog[a..]`.
fn reg_fn(p: &TileProgram) -> String {
    let most = p
        .regs
        .iter()
        .map(|r| r.ins.len() + r.outs.len())
        .max()
        .unwrap_or(1)
        .max(1);
    let mut o = format!(
        "var ia = @zeros([{most}]u16);\nfn reg(m: u16, a: u16, p: u16) void {{\n  const nin = prog[a]; const nout = prog[a + 1]; const c = prog[a + 2]; const ch = prog[a + 3];\n  var k: u16 = 0;\n  while (k < nin + nout) : (k += 1) {{\n    if (k < nin) {{ ia[k] = prog[a + 4 + 2 * k] + (if (prog[a + 5 + 2 * k] != 0) p * c else 0); }}\n    else {{ ia[k] = prog[a + 4 + nin + k] + p * c; }}\n  }}\n  var j: u16 = 0;\n  while (j < c) : (j += ch) {{\n"
    );
    for (mi, _) in p.modules.iter().enumerate() {
        let Some(r) = p.regs.iter().find(|r| r.module == mi) else {
            continue;
        };
        let (nin, nout) = (r.ins.len(), r.outs.len());
        let _ = write!(o, "    if (m == {mi}) {{");
        for k in 0..nin {
            let _ = write!(o, " @fmovs(rm{mi}.IN{k}, md(ia[{k}] + j, ch));");
        }
        let _ = write!(o, " rm{mi}.row();");
        for q in 0..nout {
            let _ = write!(o, " @fmovs(md(ia[{}] + j, ch), rm{mi}.OUT{q});", nin + q);
        }
        o.push_str(" }\n");
    }
    o.push_str("  }\n}\n");
    o
}

/// The image PE's ops (its interpreter's).
pub const TPE_OPS: &str = include_str!("../../../kernels/runtime/tiled_pe.csl");

/// `run_all` where the op table streams in: each op's words off the stream into `prog` (its
/// first word its code and length), then the op at 0.
const RUN_STREAMED: &str = "fn run_all() void {\n  var i: u16 = 0;\n  while (i < NOPS) : (i += 1) {\n    @mov32(@set_dsd_length(D_progw, 1), fab.fin(q_t, 1));\n    const n = (prog[1] + 1) >> 1;\n    if (n > 1) { @mov32(@set_dsd_length(@increment_dsd_offset(D_progw, 1, u32), n - 1), fab.fin(q_t, n - 1)); }\n    const op = prog[0];\n    if (op == 1) { opin(0); }\n    else if (op == 2) { opout(0); }\n    else if (op == 3) { conv(0); }\n    else if (op == 4) { norm(0); }\n    else if (op == 5) { region_op(0); }\n    else if (op == 6) { up(0); }\n    else if (op == 8) { attend(0); }\n  }\n}\n";

fn pe_file(p: &TileProgram, tb: &Table) -> String {
    let mut t = format!(
        "// {}: an image PE of the tiled regime (kernels-cerebras emit::tiled): the\n// program's ops over its pixels, in step with the array.\n",
        p.reading
    );
    t.push_str("param memcpy_params;\n");
    for c in [
        "C_EO", "C_EI", "C_WO", "C_WI", "C_SO", "C_SI", "C_NO", "C_NI", "C_T", "C_B",
    ] {
        let _ = writeln!(t, "param {c}: u16;");
    }
    t.push_str(COMMON);
    t.push_str("const q_t = @get_input_queue(2);\nconst q_ei = @get_input_queue(3);\nconst q_wi = @get_input_queue(4);\nconst q_si = @get_input_queue(5);\nconst q_ni = @get_input_queue(6);\nconst q_bi = @get_input_queue(7);\nconst q_eo = @get_output_queue(2);\nconst q_wo = @get_output_queue(3);\nconst q_so = @get_output_queue(4);\nconst q_no = @get_output_queue(5);\n");
    t.push_str("const matvec = @import_module(\"ops/mv.csl\");\n");
    for (mi, (name, _)) in p.modules.iter().enumerate() {
        let _ = writeln!(
            t,
            "const rm{mi} = @import_module(\"{name}\", .{{ .math = math, .seg = math }});"
        );
    }
    // the op table: held whole, or (streamed) an op's words at a time off the stream
    let streamed = !p.op_words.is_empty();
    let table_decl = if streamed {
        let opn = p.op_words.iter().copied().max().unwrap_or(2).max(2);
        format!(
            "const NOPS: u16 = {};\nconst OPN: u16 = {opn};\nvar prog = @zeros([OPN]u16);\nconst D_prog = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{OPN}} -> prog[i] }});\nvar D_progw = D_prog;",
            p.op_words.len()
        )
    } else {
        format!(
            "const NPROG: u16 = {};\nconst prog = [{}]u16 {{ {} }};",
            tb.words.len(),
            tb.words.len().max(1),
            list(&tb.words)
        )
    };
    let _ = writeln!(
        t,
        "const GX: u16 = {};\nconst GY: u16 = {};\n{table_decl}\nconst lays = [{}]u16 {{ {} }};\nconst fk = [{}]f32 {{ {} }};",
        p.gx,
        p.gy,
        tb.lays.len(),
        list(&tb.lays),
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
    let (aw, hxn, wbn) = (p.arena, p.hx, p.wb);
    let (hin, hout) = (tb.hin_words.max(1), tb.hout_words.max(1));
    let _ = writeln!(
        t,
        "const AW: u16 = {aw};\nvar mem: [AW]f32 align(16) = @zeros([AW]f32);\nconst D_mem = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{AW}} -> mem[i] }});\n\
         const HXN: u16 = {hxn};\nvar hx: [HXN]@fp16() align(16) = @zeros([HXN]@fp16());\nconst D_hx = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{HXN}} -> hx[i] }});\n\
         const WBN: u16 = {wbn};\nvar wb: [WBN]u32 align(16) = @zeros([WBN]u32);\nconst D_wb = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{WBN}} -> wb[i] }});\n\
         const D_wa0 = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{HXN}} -> hx[i] }});\nvar D_wa = D_wa0;\n\
         var hin: [{hin}]f32 align(16) = @zeros([{hin}]f32);\nvar hin_p: [*]f32 = &hin;\nconst D_hin = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{{hin}}} -> hin[i] }});\n\
         var hout: [{hout}]f32 align(16) = @zeros([{hout}]f32);\nvar hout_p: [*]f32 = &hout;\nconst D_hout = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{{hout}}} -> hout[i] }});\n\
         var pcfg = @zeros([2]u32);\nvar pcfg_p: [*]u32 = &pcfg;"
    );
    // (a tight program's rounding staged a chunk at a time: its staging holds no whole value)
    let ops = if p.tight {
        TPE_OPS.replace(
            "fn rnd(o: u16, n: u16) void { if (n == 0) { return; } @fs2h(hd(0, n), md(o, n)); @fh2s(md(o, n), hd(0, n)); }",
            "fn rnd(o: u16, n: u16) void {\n  var d: u16 = 0;\n  while (d < n) {\n    const m = if (n - d > HXN) HXN else n - d;\n    @fs2h(hd(0, m), md(o + d, m)); @fh2s(md(o + d, m), hd(0, m));\n    d += m;\n  }\n}",
        )
    } else {
        TPE_OPS.to_string()
    };
    let ops = ops.as_str();
    if streamed {
        // (each op's words off the stream into `prog`, its first word its code and length)
        // (tiled_pe.csl's `run_all`, the table's walk, in place: the template holds it once)
        let all = ops.find("fn run_all() void {").unwrap_or(ops.len());
        let end = ops[all..].find("\n}\n").map_or(ops.len(), |e| all + e + 3);
        t.push_str(&ops[..all]);
        t.push_str(RUN_STREAMED);
        t.push_str(&ops[end..]);
    } else {
        t.push_str(ops);
    }
    t.push_str(&reg_fn(p));
    let mut init = String::new();
    for mi in 0..p.modules.len() {
        let _ = writeln!(init, "  rm{mi}.init();");
    }
    let progw = if streamed {
        "  D_progw = @set_dsd_base_addr(D_wb, @ptrcast([*]u32, &prog[0]));\n"
    } else {
        ""
    };
    let _ = write!(
        t,
        "task go() void {{\n  stamp(0);\n  X = @as(u16, pcfg[0]);\n  Y = @as(u16, pcfg[1]);\n  D_wa = @set_dsd_base_addr(D_wa0, @ptrcast([*]@fp16(), &wb[0]));\n  D_hx32 = @set_dsd_base_addr(D_wb, @ptrcast([*]u32, &hx[0]));\n{progw}{init}  run_all();\n  stamp(1);\n  sys_mod.unblock_cmd_stream();\n}}\n"
    );
    t.push_str("comptime {\n  @bind_local_task(go, T_GO);\n  @export_symbol(run);\n  @export_symbol(tsc_p, \"tsc\");\n  @export_symbol(pcfg_p, \"pcfg\");\n  @export_symbol(hin_p, \"hin\");\n  @export_symbol(hout_p, \"hout\");\n");
    for (q, c) in [
        ("q_t", "C_T"),
        ("q_ei", "C_EI"),
        ("q_wi", "C_WI"),
        ("q_si", "C_SI"),
        ("q_ni", "C_NI"),
        ("q_bi", "C_B"),
        ("q_eo", "C_EO"),
        ("q_wo", "C_WO"),
        ("q_so", "C_SO"),
        ("q_no", "C_NO"),
    ] {
        let _ = writeln!(
            t,
            "  @initialize_queue({q}, .{{ .color = @get_color({c}) }});"
        );
    }
    t.push_str("}\n");
    t
}

/// A tape PE's image past its tape's words, as a link measures it: the check's estimate of it
/// misses by ~500 B, past the stack's room. Re-measure on a change to `ttape.csl`'s code.
pub const TAPE_IMAGE_BYTES: i64 = 4556;

/// Words a tape PE holds: the room `ttape.csl`'s image leaves past its own bytes, the stack
/// and the margin.
#[must_use]
pub fn tape_words() -> u32 {
    crate::compile::check::tape_words(TAPE_IMAGE_BYTES)
}

fn tape_file(tw: u32) -> String {
    let mut t = String::from(
        "// The tiled regime's weight tape (kernels-cerebras emit::tiled): a PE's words of the\n// reading's stream, then the words of the PEs past it forwarded.\nparam memcpy_params;\nparam C_O: u16;\nparam C_I: u16;\n",
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

fn hub_file(p: &TileProgram) -> String {
    let hub = p.hub();
    let mut t = String::from(
        "// The tiled regime's hub (kernels-cerebras emit::tiled): what comes up the array's last\n// column (an all-reduce's sums, the attention's keys) out along the broadcast tree.\nparam memcpy_params;\nparam C_I: u16;\nparam C_B: u16;\n",
    );
    t.push_str(COMMON);
    let _ = writeln!(
        t,
        "const q_i = @get_input_queue(2);\nconst q_o = @get_output_queue(2);\nconst NH: u16 = {};\nconst hub = [{}]u32 {{ {} }};",
        hub.len(),
        hub.len().max(1),
        if hub.is_empty() {
            "0".to_string()
        } else {
            list(&hub)
        }
    );
    t.push_str(
        "task go() void {\n  stamp(0);\n  var k: u16 = 0;\n  while (k < NH) : (k += 1) {\n    var left: u32 = hub[k];\n    while (left > 0) {\n      const n: u16 = if (left > 16384) 16384 else @as(u16, left);\n      @mov32(fab.fout(q_o, n), fab.fin(q_i, n));\n      left -= @as(u32, n);\n    }\n  }\n  stamp(1);\n  sys_mod.unblock_cmd_stream();\n}\n",
    );
    t.push_str("comptime {\n  @bind_local_task(go, T_GO);\n  @export_symbol(run);\n  @export_symbol(tsc_p, \"tsc\");\n  @initialize_queue(q_i, .{ .color = @get_color(C_I) });\n  @initialize_queue(q_o, .{ .color = @get_color(C_B) });\n}\n");
    t
}

fn filler_file() -> String {
    let mut t = String::from(
        "// An idle PE of the tiled regime (kernels-cerebras emit::tiled): routes only.\nparam memcpy_params;\n",
    );
    t.push_str(COMMON);
    t.push_str("task go() void { sys_mod.unblock_cmd_stream(); }\ncomptime {\n  @bind_local_task(go, T_GO);\n  @export_symbol(run);\n  @export_symbol(tsc_p, \"tsc\");\n}\n");
    t
}

/// The program's placement: the image array's PEs (each its shifts' parities), the hub, the
/// tape, the fillers; the shifts', the weight tree's, the broadcast's and the tape's routes.
#[must_use]
pub fn placed(p: &TileProgram) -> Placed {
    let (fw, fh) = p.fabric();
    let (gx, gy) = (p.gx, p.gy);
    let mut l = Placed::new(
        format!(
            "// A tiled grid reading's wafer program (kernels-cerebras emit::tiled): {fw} x {fh} PEs, an image array of {gx} x {gy}."
        ),
        fw,
        fh,
    );
    // a rule: colour `c` on fabric x in [x0, x1) step sx, y in [y0, y1) step sy
    let mut routes: Vec<(u32, At, &'static str, Vec<&'static str>)> = Vec::new();
    let mut rule = |c: u32, xs: Span, ys: Span, rx: &'static str, tx: &[&'static str]| {
        routes.push((c, At::Region(xs, ys), rx, tx.to_vec()));
    };
    let all_y = (0, gy, 1);
    // the shifts (image X at fabric x = X + 1): a PE of parity q sends on colour q and
    // receives its neighbour's on 1 - q
    for q in 0..2u32 {
        let qi = q as usize;
        let lo = if q == 0 { 2 } else { 1 }; // the first coordinate of parity q past 0
        let cols = (1, gx + 1, 1);
        // X < gx - 1 sends E and receives W-going; X >= 1 sends W and receives E-going
        rule(E[qi], (1 + q, gx, 2), all_y, "RAMP", &["EAST"]);
        rule(E[1 - qi], (1 + lo, gx + 1, 2), all_y, "WEST", &["RAMP"]);
        rule(W[qi], (1 + lo, gx + 1, 2), all_y, "RAMP", &["WEST"]);
        rule(W[1 - qi], (1 + q, gx, 2), all_y, "EAST", &["RAMP"]);
        let last = gy.saturating_sub(1);
        rule(S[qi], cols, (q, last, 2), "RAMP", &["SOUTH"]);
        rule(S[1 - qi], cols, (lo, gy, 2), "NORTH", &["RAMP"]);
        rule(N[qi], cols, (lo, gy, 2), "RAMP", &["NORTH"]);
        rule(N[1 - qi], cols, (q, last, 2), "SOUTH", &["RAMP"]);
    }
    // the corner's E colour to the hub, the hub's in
    let cpar = ((gx - 1) % 2) as usize;
    rule(E[cpar], (gx, gx + 1, 1), (gy - 1, gy, 1), "RAMP", &["EAST"]);
    rule(
        E[cpar],
        (gx + 1, gx + 2, 1),
        (gy - 1, gy, 1),
        "WEST",
        &["RAMP"],
    );
    // the weight tree: tape 0 north into the spine, the spine north and east, every row east
    rule(T, (0, 1, 1), (gy, gy + 1, 1), "RAMP", &["NORTH"]);
    rule(T, (0, 1, 1), (0, 1, 1), "SOUTH", &["EAST"]);
    rule(T, (0, 1, 1), (1, gy, 1), "SOUTH", &["EAST", "NORTH"]);
    rule(T, (1, gx, 1), all_y, "WEST", &["RAMP", "EAST"]);
    rule(T, (gx, gx + 1, 1), all_y, "WEST", &["RAMP"]);
    // the broadcast tree: the hub west into the corner, up the last column, west along the rows
    rule(B, (gx + 1, gx + 2, 1), (gy - 1, gy, 1), "RAMP", &["WEST"]);
    let rw: &[&'static str] = if gx > 1 { &["RAMP", "WEST"] } else { &["RAMP"] };
    let rwn: Vec<&'static str> = rw.iter().copied().chain(["NORTH"]).collect();
    rule(
        B,
        (gx, gx + 1, 1),
        (gy - 1, gy, 1),
        "EAST",
        if gy > 1 { &rwn } else { rw },
    );
    if gy > 1 {
        rule(B, (gx, gx + 1, 1), (0, 1, 1), "SOUTH", rw);
        rule(B, (gx, gx + 1, 1), (1, gy - 1, 1), "SOUTH", &rwn);
    }
    rule(B, (2, gx, 1), all_y, "EAST", &["RAMP", "WEST"]);
    rule(B, (1, 2.min(gx), 1), all_y, "EAST", &["RAMP"]);
    // the tape: PE k sends towards k - 1 on A[k % 2]
    let dir = |a: (u32, u32), b: (u32, u32)| -> (&'static str, &'static str) {
        // (towards b from a, from a as b sees it)
        if b.0 < a.0 {
            ("WEST", "EAST")
        } else if b.0 > a.0 {
            ("EAST", "WEST")
        } else if b.1 < a.1 {
            ("NORTH", "SOUTH")
        } else {
            ("SOUTH", "NORTH")
        }
    };
    for k in 1..p.tape {
        let (a, b) = (p.tape_at(k), p.tape_at(k - 1));
        let (to, from) = dir(a, b);
        let c = A[(k % 2) as usize];
        routes.push((c, At::Pe(a.0, a.1), "RAMP", vec![to]));
        routes.push((c, At::Pe(b.0, b.1), from, vec!["RAMP"]));
    }
    // the code
    for px in 0..2u32 {
        for py in 0..2u32 {
            let (x0, y0) = (1 + px, py);
            if x0 > gx || y0 >= gy {
                continue;
            }
            let c = |v: u32| v.to_string();
            l.tile(
                At::Region((x0, gx + 1, 2), (y0, gy, 2)),
                "tpe.csl",
                &[
                    ("C_EO", c(E[px as usize])),
                    ("C_EI", c(E[1 - px as usize])),
                    ("C_WO", c(W[px as usize])),
                    ("C_WI", c(W[1 - px as usize])),
                    ("C_SO", c(S[py as usize])),
                    ("C_SI", c(S[1 - py as usize])),
                    ("C_NO", c(N[py as usize])),
                    ("C_NI", c(N[1 - py as usize])),
                    ("C_T", c(T)),
                    ("C_B", c(B)),
                ],
            );
        }
    }
    l.tile(
        At::Pe(gx + 1, gy - 1),
        "thub.csl",
        &[("C_I", E[cpar].to_string()), ("C_B", B.to_string())],
    );
    let mut used = std::collections::BTreeSet::new();
    for k in 0..p.tape {
        let (x, y) = p.tape_at(k);
        used.insert((x, y));
        let co = if k == 0 { T } else { A[(k % 2) as usize] };
        let ci = A[((k + 1) % 2) as usize];
        l.tile(
            At::Pe(x, y),
            "ttape.csl",
            &[("C_O", co.to_string()), ("C_I", ci.to_string())],
        );
    }
    // the rest: the spine, the hub's column but its row, the tape rows' remainder
    for y in 0..gy {
        l.tile(At::Pe(0, y), "filler.csl", &[]);
    }
    for y in 0..gy.saturating_sub(1) {
        l.tile(At::Pe(gx + 1, y), "filler.csl", &[]);
    }
    for y in gy..fh {
        for x in 0..fw {
            if !used.contains(&(x, y)) {
                l.tile(At::Pe(x, y), "filler.csl", &[]);
            }
        }
    }
    for (c, at, rx, tx) in routes {
        l.route(c, at, rx, &tx);
    }
    for (n, ty) in [
        ("tsc", "u32"),
        ("pcfg", "u32"),
        ("hin", "f32"),
        ("hout", "f32"),
        ("tape", "u32"),
    ] {
        l.export(n, ty);
    }
    l
}

/// The image PEs' place in fabric coordinates.
fn at(x: u32, y: u32) -> Coord {
    Coord::new(x + 1, y)
}

/// The emitted program.
#[allow(clippy::too_many_lines)]
pub fn emit(p: &TileProgram) -> R<Emitted> {
    // (a hub loop a PE's blocking send leaves open hangs the program: refused)
    if let Some(l) = crate::compile::check::hazard::hub_loops(p, TPE_OPS).first() {
        return Err(format!("{}: {l}", p.reading));
    }
    let tb = table(p)?;
    let mut files: Vec<(String, String)> = p.modules.clone();
    files.push(("tpe.csl".into(), pe_file(p, &tb)));
    files.push(("ttape.csl".into(), tape_file(tape_words())));
    files.push(("thub.csl".into(), hub_file(p)));
    files.push(("filler.csl".into(), filler_file()));
    let pl = placed(p);
    files.push(("layout.csl".into(), pl.csl()));
    let mut arrays = Vec::new();
    let (gx, gy) = (p.gx, p.gy);
    for y in 0..gy {
        for x in 0..gx {
            let c = at(x, y);
            arrays.push(GridArray {
                at: c,
                sym: "pcfg".into(),
                u16: false,
                len: 2,
                fill: GridFill::Words(vec![x, y]),
                group: None,
            });
        }
    }
    // the host's values: on the PEs of their layouts' busy rows
    if tb.hin_words > 0 {
        for y in 0..gy {
            let rows_busy = tb.hin.iter().any(|(v, _)| {
                let l = p.lay_of(*v);
                y & ((1 << l.lsy) - 1) == 0
            });
            if !rows_busy {
                continue;
            }
            for x in 0..gx {
                let mut ent: Vec<Word> = vec![[NONE, 0, 0, 0]; tb.hin_words as usize];
                for &(v, off) in &tb.hin {
                    let l = p.lay_of(v);
                    let wd = p.vals[&v].width;
                    let [_, _, w] = p.vals[&v].grid;
                    if y & ((1 << l.lsy) - 1) != 0 || x & ((1 << l.lsx) - 1) != 0 {
                        continue;
                    }
                    let (pr, pc) = (y >> l.lsy, x >> l.lsx);
                    for i in 0..l.th {
                        for j in 0..l.tw {
                            let row = (pr * l.th + i) * w + pc * l.tw + j;
                            for e in 0..wd {
                                ent[(off + (i * l.tw + j) * wd + e) as usize] = [v.0, row, e, 0];
                            }
                        }
                    }
                }
                arrays.push(GridArray {
                    at: at(x, y),
                    sym: "hin".into(),
                    u16: false,
                    len: tb.hin_words,
                    fill: GridFill::Entry(ent),
                    group: None,
                });
            }
        }
    }
    // the outputs, read back from every busy PE
    if tb.hout.len() != 1 {
        return Err(format!(
            "{} outputs: the tiled regime reads one back",
            tb.hout.len()
        ));
    }
    let (ov, _) = tb.hout[0];
    let l = p.lay_of(ov);
    let [_, _, ow] = p.vals[&ov].grid;
    let wd = p.vals[&ov].width;
    for y in 0..gy {
        for x in 0..gx {
            if y & ((1 << l.lsy) - 1) != 0 || x & ((1 << l.lsx) - 1) != 0 {
                continue;
            }
            let (pr, pc) = (y >> l.lsy, x >> l.lsx);
            let mut rows = Vec::new();
            for i in 0..l.th {
                for j in 0..l.tw {
                    rows.push(((pr * l.th + i) * ow + pc * l.tw + j) as usize);
                }
            }
            arrays.push(GridArray {
                at: at(x, y),
                sym: "hout".into(),
                u16: false,
                len: l.px() * wd,
                fill: GridFill::Back {
                    rows,
                    parts: vec![(ov.0, 0, wd)],
                },
                group: None,
            });
        }
    }
    // the tape
    let total = p.stream_words;
    let mut sent = 0u32;
    for k in 0..p.tape {
        let (x, y) = p.tape_at(k);
        let own = tape_words().min(total - sent);
        sent += own;
        arrays.push(GridArray {
            at: Coord::new(x, y),
            sym: "tape".into(),
            u16: false,
            len: tape_words(),
            fill: GridFill::Spans(tape_spans(p, k)),
            group: None,
        });
        arrays.push(GridArray {
            at: Coord::new(x, y),
            sym: "pcfg".into(),
            u16: false,
            len: 2,
            fill: GridFill::Words(vec![own, total - sent]),
            group: None,
        });
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
