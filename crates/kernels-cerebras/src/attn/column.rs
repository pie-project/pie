//! Attention on a unit's K / V column (`ops/satt.csl`, `ops/cy_satt.csl`): every row's q, k and
//! v stored down the column, each query attending what is stored (a step's rows, or a decode
//! step's cache in pages across launches).

use std::fmt::Write as _;

use crate::compile::emit::stream::{
    COMMON, Fill, HostArray, buf, cfg_decl, kv_consts, kv_joint, paging, rowp_array, rows_of,
};
use crate::compile::floor::Coord;
use crate::compile::floorplan::stream::KvColumn;
use crate::compile::stream::build::Program;
use crate::compile::stream::place::KvSched;
use crate::compile::stream::unit::{Joint, Lowering, Side, Step, side_of};
use crate::kernel::{Column, ColumnAsk, GemmForm, Kernel, LineAlong, R, Source, UnitKernel};
use crate::tuning;
use poem_ir::{Attn, Def, Operation};

/// Attention on a K / V column.
pub struct Attention;

/// The registry's attention column.
pub static ATTENTION: Attention = Attention;

impl Kernel for Attention {
    fn name(&self) -> &'static str {
        "attn.column"
    }

    fn ops(&self) -> &'static [&'static str] {
        &[
            "attn.ragged",
            "attn.prefill",
            "attn.decode",
            "attn.kv_append",
            "attn.sink",
        ]
    }

    fn sources(&self) -> &'static [Source] {
        &[
            (
                "cy_satt.csl",
                include_str!("../../kernels/attn/cy_satt.csl"),
            ),
            ("satt.csl", include_str!("../../kernels/attn/satt.csl")),
        ]
    }

    fn unit(&self) -> Option<&dyn UnitKernel> {
        Some(self)
    }

    fn gemm(&self) -> Option<&'static GemmForm> {
        Some(&GEMM)
    }

    fn column(&self) -> Option<&dyn Column> {
        Some(self)
    }
}

impl UnitKernel for Attention {
    /// The attention ops: each a row's q / k / v stored, then its query's output back.
    fn lower(&self, cx: &mut Lowering, at: u32) -> R<bool> {
        let name = cx.name;
        let refuse = |at: u32, what: String| format!("{name}: node {at}: {what}");
        let Lowering {
            r,
            nodes,
            steps,
            sides,
            joint,
            skip,
            appended,
            ..
        } = cx;
        let r = *r;
        let nodes: &[u32] = nodes;
        let trace = r.trace;
        match &trace.nodes[at as usize].op {
            Operation::Attn(Attn::Ragged {
                q,
                k,
                v,
                head_dim,
                kv_heads,
                sm_scale,
                o,
                ..
            }) => {
                let (q, k, v) = (r.cols(*q), r.cols(*k), r.cols(*v));
                for c in [q, k, v] {
                    if side_of(sides, c) != Some(Side::Col) {
                        return Err(refuse(
                            at,
                            "attn.ragged reads a q / k / v not at the column roots".into(),
                        ));
                    }
                }
                **joint = Some(Joint {
                    q_heads: q.width / head_dim,
                    kv_heads: *kv_heads,
                    head_dim: *head_dim,
                    scale: *sm_scale,
                    causal: false,
                    window: None,
                    state: None,
                    sink: None,
                    kv_rep: 1,
                });
                steps.push(Step::Store { q, k, v });
                sides.insert(*o, Side::Col);
                steps.push(Step::Attend {
                    out: *o,
                    round: r.round(*o),
                });
            }
            // the step's keys and values into the pages its prefill reads: stored with the
            // row's query when the prefill comes
            Operation::Attn(Attn::KvAppend {
                k,
                v: Some(v),
                cache,
                ..
            }) => {
                appended.push((*cache, r.cols(*k), r.cols(*v)));
            }
            Operation::Attn(Attn::Prefill {
                q,
                cache,
                head_dim,
                kv_heads,
                sm_scale,
                causal,
                o,
                ..
            }) => {
                let Some(&(_, k, v)) = appended.iter().find(|(c, _, _)| c == cache) else {
                    return Err(refuse(
                        at,
                        "attn.prefill reads pages no step of its unit appends".into(),
                    ));
                };
                let q = r.cols(*q);
                for c in [q, k, v] {
                    if side_of(sides, c) != Some(Side::Col) {
                        return Err(refuse(
                            at,
                            "attn.prefill reads a q / k / v not at the column roots".into(),
                        ));
                    }
                }
                **joint = Some(Joint {
                    q_heads: q.width / head_dim,
                    kv_heads: *kv_heads,
                    head_dim: *head_dim,
                    scale: *sm_scale,
                    causal: *causal,
                    window: None,
                    state: None,
                    sink: None,
                    kv_rep: 1,
                });
                steps.push(Step::Store { q, k, v });
                sides.insert(*o, Side::Col);
                steps.push(Step::Attend {
                    out: *o,
                    round: r.round(*o),
                });
            }
            // a decode step's attention over its layer's cache: the row stored with its query
            // (at its position), then attending every row stored at or before it
            Operation::Attn(Attn::Decode {
                q,
                plan,
                cache,
                head_dim,
                sm_scale,
                window,
                o,
                lse,
                ..
            }) => {
                // its log-sum-exp read by a sink alone: the sink folded in (the Attend's)
                let sink = match lse {
                    None => None,
                    Some(l) => {
                        let sn = nodes
                            .iter()
                            .copied()
                            .find(|&n| {
                                matches!(&trace.nodes[n as usize].op,
                                Operation::Attn(Attn::Sink { o: so, lse: sl, .. })
                                    if so == o && sl == l)
                            })
                            .ok_or_else(|| {
                                refuse(at, "attn.decode's log-sum-exp is read by no sink".into())
                            })?;
                        let Operation::Attn(Attn::Sink { sink, .. }) = &trace.nodes[sn as usize].op
                        else {
                            unreachable!()
                        };
                        skip.insert(sn);
                        Some(*sink)
                    }
                };
                let Some(&(_, k, v)) = appended.iter().find(|(c, _, _)| c == cache) else {
                    return Err(refuse(
                        at,
                        "attn.decode reads pages no step of its unit appends".into(),
                    ));
                };
                let kv_heads = match r.def(*plan) {
                    Some(Def::Op(n)) => match &trace.nodes[*n as usize].op {
                        Operation::Attn(Attn::PlanDecode { kv_heads, .. }) => *kv_heads,
                        _ => return Err(refuse(at, "attn.decode's plan is no decode plan".into())),
                    },
                    _ => return Err(refuse(at, "attn.decode's plan is no decode plan".into())),
                };
                let q = r.cols(*q);
                for c in [q, k, v] {
                    if side_of(sides, c) != Some(Side::Col) {
                        return Err(refuse(
                            at,
                            format!(
                                "attn.decode reads a q / k / v not at the column roots (value \
                             %{} at {:?})",
                                c.base.0,
                                side_of(sides, c)
                            ),
                        ));
                    }
                }
                **joint = Some(Joint {
                    q_heads: q.width / head_dim,
                    kv_heads,
                    head_dim: *head_dim,
                    scale: *sm_scale,
                    causal: true,
                    window: *window,
                    state: None,
                    sink,
                    kv_rep: 1,
                });
                steps.push(Step::Store { q, k, v });
                sides.insert(*o, Side::Col);
                steps.push(Step::Attend {
                    out: *o,
                    round: r.round(*o),
                });
            }
            _ => return Ok(false),
        }
        Ok(true)
    }
}

/// Its GEMM form (`gemm_pe.csl`'s `attend`): its key blocks over a band's reductions, its
/// members' order down the column line; it orders the row where its code ends in a row-wide
/// all-reduce.
static GEMM: GemmForm = GemmForm {
    func: "attend",
    what: "an attention",
    line: LineAlong::Column,
    masked: false,
    orders_row: crate::compile::check::hazard::attention_orders_row,
    what_price: "attention",
    price: gemm_price,
    drift: "the attention: its key blocks over a band's reductions, a drift growing with its key \
            blocks",
    drift_after: "its key blocks' band reductions let the row's bands drift apart",
};

/// Its GEMM op's cycles: per key block, its turn, the scores' product and all-reduces, the
/// softmax and the values' product, plus a block's fitted overhead (`gemm_key_block`).
fn gemm_price(p: &crate::compile::stream::gemm::GemmProgram, i: usize, _: &mut [f64; 5]) -> f64 {
    use crate::compile::cost::gemm::{allred, mac};
    let crate::compile::stream::gemm::GOp::Attend { causal, .. } = &p.l.ops[i] else {
        return 0.0;
    };
    let tb = p.tb;
    let kb = p.kbm * tb;
    let nc = p.qchunk;
    let half = tb * nc;
    // (a request's key blocks: its rows of PEs')
    let blocks = p.r / p.l.batch.max(1) / p.kbm;
    let sl = kb / p.band.max(1);
    let blk = tuning::get().gemm_key_block
        + f64::from(p.kbm)
            * (tuning::get().gemm_turn
                + 3.0 * f64::from(half)
                + f64::from(tb) * (tuning::get().gemm_call + f64::from(nc))
                + tuning::get().gemm_call
                + f64::from(half))
        + f64::from(tb) * mac(kb * nc)
        + 2.0 * allred(tb * kb, p.band)
        + f64::from(tb * kb) * 1.0
        + f64::from(tb)
            * (2.0 * (f64::from(kb) + 8.0 * tuning::get().gemm_call)
                + f64::from(sl) * tuning::get().gemm_exp
                + 3.0 * tuning::get().gemm_exp
                + 4.0 * tuning::get().gemm_call
                + f64::from(kb)
                + mac(kb * nc))
        + if *causal {
            f64::from(tb * kb) * 4.0
        } else {
            0.0
        };
    f64::from(blocks) * blk
}

/// A column with no attention of its own (its block's first unit has none): the reading's,
/// whose sizes the program's shape states.
pub const UNSTATED: Joint = Joint {
    q_heads: 0,
    kv_heads: 0,
    head_dim: 0,
    scale: 0.0,
    causal: false,
    window: None,
    state: None,
    sink: None,
    kv_rep: 1,
};

impl Column for Attention {
    fn claims(&self, j: &Joint) -> bool {
        j.state.is_none()
    }

    fn advances(&self) -> bool {
        true
    }

    fn region(&self, _j: &Joint) -> &'static str {
        "kv"
    }

    fn paged(&self) -> bool {
        true
    }

    fn replicates(&self) -> bool {
        true
    }

    fn pya(&self, a: &ColumnAsk) -> u32 {
        crate::compile::stream::build::kv_pya_split(a.positions, a.q_per_kv, a.head_dim, a.split)
    }

    fn root_file(&self, p: &Program, k: usize, ks: &KvSched) -> R<(String, String)> {
        kv_root_file(p, k, ks)
    }

    fn pe_file(&self, p: &Program, k: usize, ks: &KvSched) -> R<(String, String)> {
        kv_pe_file(p, k, ks)
    }

    fn root_arrays(&self, p: &Program, kc: &KvColumn, ks: &KvSched, root: Coord) -> Vec<HostArray> {
        let Some(sk) = kv_joint(p, ks).sink else {
            return Vec::new();
        };
        // the root's heads' sink logits, its step's first position
        let (qh, ..) = kv_consts(p, ks);
        vec![
            HostArray {
                at: root,
                sym: "snk".into(),
                u16: false,
                len: qh,
                fill: Fill::Slices(vec![(sk.0, kc.head * qh, qh)]),
            },
            HostArray {
                at: root,
                sym: "pos".into(),
                u16: false,
                len: 1,
                fill: Fill::Words(vec![0]),
            },
        ]
    }

    fn pe_arrays(&self, p: &Program, c: Coord) -> Vec<HostArray> {
        let mut out = Vec::new();
        if let Some(n) = rows_of(p) {
            // a decode step's rows' (lane slot, state slot, position)
            out.push(rowp_array(c, n));
            if let Some(g) = paging(p) {
                // the lane slots' pages
                out.push(HostArray {
                    at: c,
                    sym: "ptab".into(),
                    u16: false,
                    len: g.lanes * g.pages,
                    fill: Fill::Words(vec![0; (g.lanes * g.pages) as usize]),
                });
            }
        }
        out
    }
}

/// A K / V column's root: each unit's rows' [q | k | v] down the column in the store order,
/// then each query's output back to the unit that asked it.
fn kv_root_file(p: &Program, k: usize, ks: &KvSched) -> R<(String, String)> {
    let (qh, dc, qd, _) = kv_consts(p, ks);
    let sink = kv_joint(p, ks).sink.is_some();
    let (lo, hi) = if p.round16 {
        ("1.0e-30", "1.0e30")
    } else {
        ("0.00390625", "32768.0")
    };
    let name = format!("kr_{k}.csl");
    let mut t = String::new();
    let _ = writeln!(
        t,
        "// K / V column {k}'s root (kernels-cerebras emit::stream; ops/cy_satt.csl)."
    );
    t.push_str("param memcpy_params;\n");
    for u in 0..ks.units.len() {
        let _ = writeln!(t, "param C_I{u}: u16;\nparam C_O{u}: u16;");
    }
    t.push_str("param C_AR: u16;\nparam C_AM: u16;\n");
    t.push_str(COMMON);
    let n = ks.units.len();
    for u in 0..n {
        let _ = writeln!(
            t,
            "const q_i{u} = @get_input_queue({});\nconst q_o{u} = @get_output_queue({});",
            2 + u,
            3 + u
        );
    }
    let _ = writeln!(
        t,
        "const q_ar = @get_input_queue({});\nconst q_am = @get_output_queue(2);",
        2 + n
    );
    let _ = writeln!(
        t,
        "const cy = @import_module(\"ops/cy_satt.csl\", .{{ .fab = fab, .math = math, .YO = q_am, .IN = q_ar, .Qh = {qh}, .Dc = {dc}, .LO = {lo}, .HI = {hi}{} }});",
        if sink { ", .SINK = true" } else { "" }
    );
    if sink {
        // its heads' sink logits (the checkpoint's), and the step's first position (the
        // reference the PEs start from at 0)
        t.push_str(
            "var snk_p: [*]f32 = &cy.sk;\nvar pos = @zeros([1]u32);\nvar pos_p: [*]u32 = &pos;\n",
        );
    }
    let sw = qd + 2 * dc;
    t.push_str(&buf("sx", sw));
    t.push_str(&buf("ob", qd));
    let mut body = String::from("  stamp(0);\n  cy.passes = 0;\n");
    if sink {
        body.push_str("  if (pos[0] == 0) { cy.init(); }\n");
    }
    for &(u, rows, _) in &ks.stores {
        let _ = writeln!(
            body,
            "  {{ var t: u16 = 0; while (t < {rows}) : (t += 1) {{ @fmovs(D_sx, fab.fin(q_i{u}, {sw})); @fmovs(fab.fout(q_am, {sw}), D_sx); }} }}"
        );
    }
    for &(u, rows) in &ks.queries {
        let _ = writeln!(
            body,
            "  {{ var t: u16 = 0; while (t < {rows}) : (t += 1) {{ cy.query_q(D_ob); @fmovs(fab.fout(q_o{u}, {qd}), D_ob); }} }}"
        );
    }
    body.push_str("  stamp(1);\n  tsc[2] = cy.passes;\n  sys_mod.unblock_cmd_stream();\n");
    let _ = write!(t, "task go() void {{\n{body}}}\n");
    t.push_str("comptime {\n  @bind_local_task(go, T_GO);\n  @export_symbol(run);\n  @export_symbol(tsc_p, \"tsc\");\n");
    for u in 0..n {
        let _ = writeln!(
            t,
            "  @initialize_queue(q_i{u}, .{{ .color = @get_color(C_I{u}) }});\n  @initialize_queue(q_o{u}, .{{ .color = @get_color(C_O{u}) }});"
        );
    }
    if sink {
        t.push_str("  @export_symbol(snk_p, \"snk\");\n  @export_symbol(pos_p, \"pos\");\n");
    }
    t.push_str("  @initialize_queue(q_ar, .{ .color = @get_color(C_AR) });\n  @initialize_queue(q_am, .{ .color = @get_color(C_AM) });\n}\n");
    Ok((name, t))
}

/// A K / V column's attention PE: the rows it keeps (q too), then every query.
fn kv_pe_file(p: &Program, k: usize, ks: &KvSched) -> R<(String, String)> {
    let (qh, dc, _, sc) = kv_consts(p, ks);
    let total: u32 = ks.stores.iter().map(|s| s.1).sum();
    let pya = p.stream.kv[k].pya;
    // a decode step's cache keeps its positions across steps (in pages); a stream step's rows
    // its own
    let paged = paging(p);
    let ne = match paged {
        Some(g) => g.pages * (g.size / pya),
        None => total.div_ceil(pya),
    };
    let pmax = if p.round16 { "1.0e30" } else { "65504.0" };
    let causal = kv_joint(p, ks).causal;
    if causal && (ks.stores.len() != 1 || ks.stores[0].2 != 0) {
        return Err(format!(
            "K / V column {k}: a causal attention over {} stores: its rows' order is one unit's",
            ks.stores.len()
        ));
    }
    let name = format!("kp_{k}.csl");
    let mut t = String::new();
    let _ = writeln!(
        t,
        "// K / V column {k}'s attention PEs (kernels-cerebras emit::stream; ops/satt.csl)."
    );
    t.push_str("param memcpy_params;\nparam C_AM: u16;\nparam C_S: u16;\nparam C_R: u16;\n");
    t.push_str(COMMON);
    // pcfg: the PE's index in the column, its chain role (the column's PEs one image)
    t.push_str(&cfg_decl(&[0, 0]));
    t.push_str("const q_am = @get_input_queue(2);\nconst q_r = @get_input_queue(3);\nconst q_s = @get_output_queue(2);\n");
    t.push_str("const red = @import_module(\"ops/schain.csl\", .{ .fab = fab, .IN1 = q_r, .OUT = q_s });\n");
    let _ = writeln!(
        t,
        "const att = @import_module(\"ops/satt.csl\", .{{ .fab = fab, .math = math, .red = red, .YM = q_am, .Qh = {qh}, .Dc = {dc}, .NE = {ne}, .PYA = {}, .SC = {sc:?}, .PMAX = {pmax}, .QH = true{}{} }});",
        pya,
        match paged {
            Some(g) => format!(
                ", .PAGED = true, .EP = {}, .LANES = {}, .MAXP = {}, .ROWS = {total}, .ADVANCE = true",
                g.size / pya,
                g.lanes,
                g.pages
            ),
            None => String::new(),
        },
        if causal {
            match kv_joint(p, ks).window {
                // a sliding window's: of the rows stored before a query, its last alone
                Some(w) => format!(", .CAUSAL = true, .WIN = {w}"),
                None => ", .CAUSAL = true".to_string(),
            }
        } else {
            String::new()
        }
    );
    let mut body = String::from("  stamp(0);\n");
    if paged.is_some() {
        // a decode step's rows each at its lane's slot (`rowp`: lane slot, position; `ptab`:
        // each lane slot's pages, the host's)
        t.push_str("var rowp_p: [*]u32 = &att.rowp;\nvar ptab_p: [*]u32 = &att.ptab;\n");
        body.push_str("  att.bind(@as(u16, pcfg[0]));\n  red.role = @as(u16, pcfg[1]);\n");
        let _ = writeln!(
            body,
            "  {{ var t: u16 = 0; while (t < {total}) : (t += 1) {{ att.store_t(t); }} }}\n  {{ var t: u16 = 0; while (t < {total}) : (t += 1) {{ att.query_t(t); }} }}"
        );
    } else {
        body.push_str("  att.init(@as(u16, pcfg[0]));\n  red.role = @as(u16, pcfg[1]);\n");
        for &(_, rows, p0) in &ks.stores {
            let _ = writeln!(
                body,
                "  {{ var t: u16 = 0; while (t < {rows}) : (t += 1) {{ att.store_q({p0} + t); }} }}"
            );
        }
        let _ = writeln!(
            body,
            "  {{ var t: u16 = 0; while (t < {total}) : (t += 1) {{ att.query_q(t); }} }}"
        );
    }
    body.push_str("  stamp(1);\n  sys_mod.unblock_cmd_stream();\n");
    let _ = write!(t, "task go() void {{\n{body}}}\n");
    t.push_str("comptime {\n  @bind_local_task(go, T_GO);\n  @export_symbol(run);\n  @export_symbol(tsc_p, \"tsc\");\n  @export_symbol(pcfg_p, \"pcfg\");\n  @initialize_queue(q_am, .{ .color = @get_color(C_AM) });\n  @initialize_queue(q_r, .{ .color = @get_color(C_R) });\n  @initialize_queue(q_s, .{ .color = @get_color(C_S) });\n");
    if paged.is_some() {
        t.push_str("  @export_symbol(rowp_p, \"rowp\");\n  @export_symbol(ptab_p, \"ptab\");\n");
    }
    t.push_str("}\n");
    Ok((name, t))
}
