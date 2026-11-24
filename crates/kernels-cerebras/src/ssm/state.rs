//! A recurrence on a unit's state column (`ops/sst.csl`, `ops/cy_sst.csl`): each head's state
//! held over the column's PEs, its root running the head's conv, gates and step. Gated
//! DeltaNet, Mamba-2 and Kimi's delta attention are its variants.

use std::fmt::Write as _;

use crate::compile::emit::stream::{
    COMMON, Fill, HostArray, buf, cfg_decl, kv_joint, rowp_array, rows_of,
};
use crate::compile::floor::Coord;
use crate::compile::floorplan::stream::KvColumn;
use crate::compile::stream::build::PYA;
use crate::compile::stream::build::Program;
use crate::compile::stream::place::KvSched;
use crate::compile::stream::unit::{
    Cols, Joint, Kda, Lowering, Side, State as StateCol, Step, View, side_of,
};
use crate::kernel::{Column, ColumnAsk, Kernel, R, Source, UnitKernel};
use poem_ir::{Dtype, Operation, Ssm, ValueId};

/// A recurrence on a state column.
pub struct State;

/// The registry's state column.
pub static STATE: State = State;

impl Kernel for State {
    fn name(&self) -> &'static str {
        "ssm.state"
    }

    fn ops(&self) -> &'static [&'static str] {
        &[
            "ssm.causal_conv1d",
            "ssm.gdn_gates",
            "ssm.gated_delta",
            "ssm.mamba2",
            "ssm.kda",
        ]
    }

    fn sources(&self) -> &'static [Source] {
        &[
            ("sst.csl", include_str!("../../kernels/ssm/sst.csl")),
            ("cy_sst.csl", include_str!("../../kernels/ssm/cy_sst.csl")),
        ]
    }

    fn unit(&self) -> Option<&dyn UnitKernel> {
        Some(self)
    }

    fn column(&self) -> Option<&dyn Column> {
        Some(self)
    }
}

impl UnitKernel for State {
    /// A recurrence from its conv: the conv, gates and step (and Mamba-2's gated norm) of a row.
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
            ..
        } = cx;
        let r = *r;
        let nodes: &[u32] = nodes;
        let trace = r.trace;
        match &trace.nodes[at as usize].op {
            // a gated delta rule (its conv, its gates, its recurrence): a step's row through its
            // heads' state columns, the conv's history and the gates at their roots. Mamba-2 (its
            // conv, its scan, its gated group norm): through its heads' state columns, the norm as
            // root regions and constant matvecs (crate::compile::stream::mamba)
            Operation::Ssm(Ssm::CausalConv1d {
                x,
                weight,
                bias: Some(cb),
                conv_width,
                dilation: 1,
                y: conv_y,
                ..
            }) if nodes.iter().any(|&n| {
                matches!(&trace.nodes[n as usize].op,
                Operation::Ssm(Ssm::Mamba2 { xbc, .. }) if xbc == conv_y)
            }) =>
            {
                use crate::compile::stream::mamba::{Kind as MK, synth as msynth};
                let refuse_at = |what: &str| refuse(at, format!("ssm.mamba2 {what}"));
                let mn = nodes
                    .iter()
                    .copied()
                    .find(|&n| {
                        matches!(&trace.nodes[n as usize].op,
                        Operation::Ssm(Ssm::Mamba2 { xbc, .. }) if xbc == conv_y)
                    })
                    .unwrap_or_else(|| unreachable!());
                let Operation::Ssm(Ssm::Mamba2 {
                    dt,
                    z,
                    a_log,
                    dt_bias,
                    d,
                    norm,
                    eps,
                    y,
                    ..
                }) = &trace.nodes[mn as usize].op
                else {
                    unreachable!()
                };
                let (h, g, n, p, inner) = crate::compile::stream::mamba::dims(trace, mn)
                    .ok_or_else(|| refuse_at("of no inner width"))?;
                if g == 0 || !h.is_multiple_of(g) || !inner.is_multiple_of(g) {
                    return Err(refuse_at(&format!("of {h} heads over {g} groups")));
                }
                skip.insert(mn);
                let xc = r.cols(*x);
                let dtc = r.cols(*dt);
                for c in [xc, dtc] {
                    if side_of(sides, c) != Some(Side::Col) {
                        return Err(refuse_at("reads a value not at the column roots"));
                    }
                }
                // xbc: [x (inner) | B (a group's n each) | C]: q C, k B, v x
                let q = Cols {
                    from: xc.from + inner + g * n,
                    width: g * n,
                    ..xc
                };
                let k = Cols {
                    from: xc.from + inner,
                    width: g * n,
                    ..xc
                };
                let v = Cols { width: inner, ..xc };
                let dth = Cols { width: h, ..dtc };
                **joint = Some(Joint {
                    q_heads: h,
                    kv_heads: h,
                    head_dim: n,
                    scale: 1.0,
                    causal: true,
                    window: None,
                    state: Some(StateCol {
                        b: dth,
                        a: dth,
                        conv: *weight,
                        taps: *conv_width,
                        a_log: *a_log,
                        dt: *dt_bias,
                        k_from: inner,
                        v_from: 0,
                        ratio: h / g,
                        q_from: inner + g * n,
                        v_dim: p,
                        mamba: Some(crate::compile::stream::unit::Mamba { d: *d, bias: *cb }),
                        kda: None,
                    }),
                    sink: None,
                    kv_rep: 1,
                });
                steps.push(Step::Store { q, k, v });
                let raw = msynth(mn, MK::Raw);
                sides.insert(raw, Side::Col);
                steps.push(Step::Attend {
                    out: raw,
                    round: false,
                });
                // the gate, each group's sum of squares and its spread, the norm
                let zd = r.dtype(*z).unwrap_or(Dtype::Bf16);
                steps.push(Step::Region {
                    side: Side::Col,
                    chain: crate::compile::stream::mamba::gate(mn, *z, zd, inner)
                        .map_err(|e| refuse(at, e))?,
                });
                for kind in [MK::Gated, MK::Squares] {
                    sides.insert(msynth(mn, kind), Side::Col);
                }
                let gw = inner / g;
                steps.push(Step::Matvec {
                    node: mn,
                    from: Side::Col,
                    input: Cols {
                        base: msynth(mn, MK::Squares),
                        from: 0,
                        width: inner,
                    },
                    weight: ValueId(crate::compile::stream::host::UNIT),
                    view: View::Sum { rep: gw },
                    output: msynth(mn, MK::Sums),
                    k: inner,
                    n: g,
                    x32: true,
                    round: false,
                });
                sides.insert(msynth(mn, MK::Sums), Side::Row);
                steps.push(Step::Matvec {
                    node: mn,
                    from: Side::Row,
                    input: Cols {
                        base: msynth(mn, MK::Sums),
                        from: 0,
                        width: g,
                    },
                    weight: ValueId(crate::compile::stream::host::UNIT),
                    view: View::Spread { rep: gw },
                    output: msynth(mn, MK::Spread),
                    k: g,
                    n: inner,
                    x32: true,
                    round: false,
                });
                sides.insert(msynth(mn, MK::Spread), Side::Col);
                let wd = r.dtype(*norm).unwrap_or(Dtype::Bf16);
                let yd = r.dtype(*y).unwrap_or(Dtype::Bf16);
                steps.push(Step::Region {
                    side: Side::Col,
                    chain: crate::compile::stream::mamba::norm(
                        mn, gw, *eps, *norm, wd, *y, yd, inner,
                    )
                    .map_err(|e| refuse(at, e))?,
                });
                sides.insert(*y, Side::Col);
            }
            Operation::Ssm(Ssm::CausalConv1d {
                x,
                weight,
                bias: None,
                conv_width,
                dilation: 1,
                y: conv_y,
                ..
            }) => {
                let refuse_at = |what: &str| refuse(at, format!("ssm.causal_conv1d {what}"));
                let find = |pick: &dyn Fn(&Operation) -> bool| {
                    nodes
                        .iter()
                        .copied()
                        .find(|&n| n > at && pick(&trace.nodes[n as usize].op))
                };
                // Kimi's delta attention: a gated delta rule of a decay a key channel
                if let Some(kn) = find(
                    &|op| matches!(op, Operation::Ssm(Ssm::Kda { mixed, .. }) if mixed == conv_y),
                ) {
                    let Operation::Ssm(Ssm::Kda {
                        f,
                        b,
                        dt_bias,
                        a_log,
                        heads,
                        head_dim,
                        norm_eps,
                        gate_floor,
                        y,
                        ..
                    }) = &trace.nodes[kn as usize].op
                    else {
                        unreachable!()
                    };
                    skip.insert(kn);
                    let (h, d) = (*heads, *head_dim);
                    let xc = r.cols(*x);
                    let bc = r.cols(*b);
                    for c in [xc, bc] {
                        if side_of(sides, c) != Some(Side::Col) {
                            return Err(refuse_at("reads a value not at the column roots"));
                        }
                    }
                    // the forget logits (a low-rank projection's, at the row roots) to the
                    // column roots: a constant matvec (one where its output is its input)
                    let mut fc = r.cols(*f);
                    match side_of(sides, fc) {
                        Some(Side::Col) => {}
                        Some(Side::Row) => {
                            let moved = crate::compile::stream::mamba::synth(
                                kn,
                                crate::compile::stream::mamba::Kind::Moved,
                            );
                            steps.push(Step::Matvec {
                                node: kn,
                                from: Side::Row,
                                input: fc,
                                weight: ValueId(crate::compile::stream::host::UNIT),
                                view: View::Spread { rep: 1 },
                                output: moved,
                                k: h * d,
                                n: h * d,
                                x32: true,
                                round: false,
                            });
                            sides.insert(moved, Side::Col);
                            fc = Cols {
                                base: moved,
                                from: 0,
                                width: h * d,
                            };
                        }
                        None => return Err(refuse_at("reads forget logits on no roots")),
                    }
                    let q = Cols { width: h * d, ..xc };
                    let k = Cols {
                        from: xc.from + h * d,
                        width: h * d,
                        ..xc
                    };
                    let v = Cols {
                        from: xc.from + 2 * h * d,
                        width: h * d,
                        ..xc
                    };
                    **joint = Some(Joint {
                        q_heads: h,
                        kv_heads: h,
                        head_dim: d,
                        scale: 1.0,
                        causal: true,
                        window: None,
                        state: Some(StateCol {
                            b: Cols { width: h, ..bc },
                            a: fc,
                            conv: *weight,
                            taps: *conv_width,
                            a_log: *a_log,
                            dt: *dt_bias,
                            k_from: h * d,
                            v_from: 2 * h * d,
                            ratio: 1,
                            q_from: 0,
                            v_dim: 0,
                            mamba: None,
                            kda: Some(Kda {
                                head_dim: d,
                                eps: *norm_eps,
                                floor: *gate_floor,
                            }),
                        }),
                        sink: None,
                        kv_rep: 1,
                    });
                    steps.push(Step::Store { q, k, v });
                    sides.insert(*y, Side::Col);
                    steps.push(Step::Attend {
                        out: *y,
                        round: r.round(*y),
                    });
                    return Ok(true);
                }
                let gd = find(&|op| {
                matches!(op, Operation::Ssm(Ssm::GatedDelta { qkv, .. }) if qkv == conv_y)
            })
            .ok_or_else(|| refuse_at("feeds no gated delta rule"))?;
                let Operation::Ssm(Ssm::GatedDelta {
                    gates,
                    k_heads,
                    v_heads,
                    k_dim,
                    v_dim,
                    y,
                    ..
                }) = &trace.nodes[gd as usize].op
                else {
                    unreachable!()
                };
                // (a state column a value head; several value heads a key head share its q / k)
                if k_dim != v_dim || *k_heads == 0 || !v_heads.is_multiple_of(*k_heads) {
                    return Err(refuse_at(&format!(
                        "into {k_heads} key heads of {k_dim} beside {v_heads} value heads of \
                     {v_dim}: a state column a value head, its key head's q / k"
                    )));
                }
                let gn = find(&|op| {
                matches!(op, Operation::Ssm(Ssm::GdnGates { gates: g, .. }) if g == gates)
            })
            .or_else(|| {
                // (the gates made before the conv)
                nodes.iter().copied().find(|&n| {
                    matches!(&trace.nodes[n as usize].op,
                        Operation::Ssm(Ssm::GdnGates { gates: g, .. }) if g == gates)
                })
            })
            .ok_or_else(|| refuse_at("gated by no gdn_gates"))?;
                let Operation::Ssm(Ssm::GdnGates {
                    ba, dt_bias, a_log, ..
                }) = &trace.nodes[gn as usize].op
                else {
                    unreachable!()
                };
                skip.extend([gn, gd]);
                let (h, hk, d) = (*v_heads, *k_heads, *k_dim);
                let xc = r.cols(*x);
                let bac = r.cols(*ba);
                for c in [xc, bac] {
                    if side_of(sides, c) != Some(Side::Col) {
                        return Err(refuse_at("reads a value not at the column roots"));
                    }
                }
                let q = Cols {
                    width: hk * d,
                    ..xc
                };
                let k = Cols {
                    from: xc.from + hk * d,
                    width: hk * d,
                    ..xc
                };
                let v = Cols {
                    from: xc.from + 2 * hk * d,
                    width: h * d,
                    ..xc
                };
                **joint = Some(Joint {
                    q_heads: h,
                    kv_heads: h,
                    head_dim: d,
                    scale: 1.0,
                    causal: true,
                    window: None,
                    state: Some(StateCol {
                        b: Cols { width: h, ..bac },
                        a: Cols {
                            from: bac.from + h,
                            width: h,
                            ..bac
                        },
                        conv: *weight,
                        taps: *conv_width,
                        a_log: *a_log,
                        dt: *dt_bias,
                        k_from: hk * d,
                        v_from: 2 * hk * d,
                        ratio: h / hk,
                        q_from: 0,
                        v_dim: 0,
                        mamba: None,
                        kda: None,
                    }),
                    sink: None,
                    kv_rep: 1,
                });
                steps.push(Step::Store { q, k, v });
                sides.insert(*y, Side::Col);
                steps.push(Step::Attend {
                    out: *y,
                    round: r.round(*y),
                });
            }
            _ => return Ok(false),
        }
        Ok(true)
    }
}

impl Column for State {
    fn claims(&self, j: &Joint) -> bool {
        j.state.is_some()
    }

    fn region(&self, j: &Joint) -> &'static str {
        match j.state {
            Some(st) if st.mamba.is_some() => "mamba2",
            Some(st) if st.kda.is_some() => "kda",
            _ => "gdn",
        }
    }

    fn paged(&self) -> bool {
        false
    }

    fn replicates(&self) -> bool {
        false
    }

    /// A decode step's root and PEs add one to each row's position once every row has read it.
    fn advances(&self) -> bool {
        true
    }

    /// Its state's rows over the fewest PEs (a decode step's a state a lane slot, a row a lane).
    fn pya(&self, a: &ColumnAsk) -> u32 {
        PYA * a.lanes.max(1)
    }

    /// Its head's beta and decay logits.
    fn head_parts(&self, j: &Joint, h: u32) -> Vec<(Cols, u32, u32)> {
        let Some(st) = j.state else {
            return Vec::new();
        };
        let aw = st.a_width();
        vec![(st.b, st.b.from + h, 1), (st.a, st.a.from + h * aw, aw)]
    }

    fn root_file(&self, p: &Program, k: usize, ks: &KvSched) -> R<(String, String)> {
        st_root_file(p, k, ks)
    }

    fn pe_file(&self, p: &Program, k: usize, ks: &KvSched) -> R<(String, String)> {
        st_pe_file(p, k, ks)
    }

    fn root_arrays(&self, p: &Program, kc: &KvColumn, ks: &KvSched, root: Coord) -> Vec<HostArray> {
        let j = kv_joint(p, ks);
        let Some(st) = j.state else {
            return Vec::new();
        };
        // the root's conv taps for its head's channels, its decay's parameters
        let (d, dv, t, h) = (j.head_dim, j.v_dim(), st.taps, kc.head);
        let (qc, kc_, vc) = (
            st.q_from + j.q_head(h) * d,
            st.k_from + j.k_head(h) * d,
            st.v_from + h * dv,
        );
        let mut slices = vec![
            (st.conv.0, qc * t, d * t),
            (st.conv.0, kc_ * t, d * t),
            (st.conv.0, vc * t, dv * t),
            (st.a_log.0, h, 1),
            (st.dt.0, h * st.a_width(), st.a_width()),
        ];
        // Mamba-2's: its head's skip, its channels' conv bias
        if let Some(m) = st.mamba {
            slices.extend([
                (m.d.0, h, 1),
                (m.bias.0, qc, d),
                (m.bias.0, kc_, d),
                (m.bias.0, vc, dv),
            ]);
        }
        let len = slices.iter().map(|s| s.2).sum();
        let mut out = vec![HostArray {
            at: root,
            sym: "stw".into(),
            u16: false,
            len,
            fill: Fill::Slices(slices),
        }];
        if let Some(n) = rows_of(p) {
            out.push(rowp_array(root, n));
        }
        out
    }

    fn pe_arrays(&self, p: &Program, c: Coord) -> Vec<HostArray> {
        // a decode step's rows' (lane slot, state slot, position)
        rows_of(p).map(|n| rowp_array(c, n)).into_iter().collect()
    }
}

/// A state column's root (ops/cy_sst.csl): each row of its unit's, its head's [q | k | v | b |
/// a] in, the step, its output back.
fn st_root_file(p: &Program, k: usize, ks: &KvSched) -> R<(String, String)> {
    let j = kv_joint(p, ks);
    let st = j.state.ok_or("a state column of no recurrence")?;
    let (d, dv) = (j.head_dim, j.v_dim());
    let n_rows = rows_of(p).unwrap_or(1);
    let name = format!("sr_{k}.csl");
    let mut t = String::new();
    let _ = writeln!(
        t,
        "// State column {k}'s root (kernels-cerebras emit::stream; ops/cy_sst.csl)."
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
        "const cy = @import_module(\"ops/cy_sst.csl\", .{{ .fab = fab, .math = math, .YO = q_am, .IN = q_ar, .DK = {d}, .DV = {dv}, .CONV = {}, .SLOTS = {n_rows}, .ROWS = {n_rows}{} }});",
        st.taps,
        match (st.mamba, st.kda) {
            (Some(_), _) => ", .MAMBA = true".to_string(),
            (None, Some(k)) => format!(
                ", .KDA = true, .FLOOR = {:?}, .L2EPS = {:?}",
                k.floor, k.eps
            ),
            (None, None) => String::new(),
        }
    );
    let sw = 2 * d + dv + 1 + st.a_width();
    t.push_str(&buf("sx", sw));
    t.push_str(&buf("ob", dv));
    // a decode step's rows each a lane's (its state slot's history; `rowp`, the host's)
    t.push_str("var stw_p: [*]f32 = &cy.cw;\nvar rowp_p: [*]u32 = &cy.rowp;\n");
    let decode = rows_of(p).is_some();
    let mut body = String::from(if decode {
        "  stamp(0);\n"
    } else {
        // (a stream step's rows one sequence's, from a cleared history)
        "  stamp(0);\n  cy.init();\n"
    });
    let mut at = 0;
    for &(u, rows, _) in &ks.stores {
        let step = if decode {
            format!("cy.row({at} + t, &sx, &ob);")
        } else {
            "cy.step(&sx, &ob);".to_string()
        };
        let _ = writeln!(
            body,
            "  {{ var t: u16 = 0; while (t < {rows}) : (t += 1) {{ @fmovs(D_sx, fab.fin(q_i{u}, {sw})); {step} @fmovs(fab.fout(q_o{u}, {dv}), D_ob); }} }}"
        );
        at += rows;
    }
    if decode {
        body.push_str("  cy.advance();\n");
    }
    body.push_str("  stamp(1);\n  sys_mod.unblock_cmd_stream();\n");
    let _ = write!(t, "task go() void {{\n{body}}}\n");
    t.push_str("comptime {\n  @bind_local_task(go, T_GO);\n  @export_symbol(run);\n  @export_symbol(tsc_p, \"tsc\");\n  @export_symbol(stw_p, \"stw\");\n");
    if decode {
        t.push_str("  @export_symbol(rowp_p, \"rowp\");\n");
    }
    for u in 0..n {
        let _ = writeln!(
            t,
            "  @initialize_queue(q_i{u}, .{{ .color = @get_color(C_I{u}) }});\n  @initialize_queue(q_o{u}, .{{ .color = @get_color(C_O{u}) }});"
        );
    }
    t.push_str("  @initialize_queue(q_ar, .{ .color = @get_color(C_AR) });\n  @initialize_queue(q_am, .{ .color = @get_color(C_AM) });\n}\n");
    Ok((name, t))
}

/// A state column's PE (ops/sst.csl): its rows of each head's state, a step a row.
fn st_pe_file(p: &Program, k: usize, ks: &KvSched) -> R<(String, String)> {
    let j = kv_joint(p, ks);
    let (d, dv) = (j.head_dim, j.v_dim());
    let pya = p.stream.kv[k].pya;
    let rows: u32 = ks.stores.iter().map(|s| s.1).sum();
    let name = format!("sp_{k}.csl");
    let mut t = String::new();
    let _ = writeln!(
        t,
        "// State column {k}'s PEs (kernels-cerebras emit::stream; ops/sst.csl)."
    );
    t.push_str("param memcpy_params;\nparam C_AM: u16;\nparam C_S: u16;\nparam C_R: u16;\n");
    t.push_str(COMMON);
    t.push_str(&cfg_decl(&[0, 0]));
    t.push_str("const q_am = @get_input_queue(2);\nconst q_r = @get_input_queue(3);\nconst q_s = @get_output_queue(2);\n");
    t.push_str("const red = @import_module(\"ops/schain.csl\", .{ .fab = fab, .IN1 = q_r, .OUT = q_s });\n");
    let _ = writeln!(
        t,
        "const st = @import_module(\"ops/sst.csl\", .{{ .fab = fab, .red = red, .YM = q_am, .DK = {d}, .DV = {dv}, .PYA = {pya}, .SLOTS = {rows}, .ROWS = {rows} }});"
    );
    // a decode step's rows each a lane's (its state slot; `rowp`, the host's)
    t.push_str("var rowp_p: [*]u32 = &st.rowp;\n");
    // (a stream step's rows one sequence's, from a cleared state)
    let (init, step, end) = if rows_of(p).is_some() {
        (
            "st.myrow = @as(u16, pcfg[0]);",
            "st.row(t);",
            "\n  st.advance();",
        )
    } else {
        ("st.init(@as(u16, pcfg[0]));", "st.step();", "")
    };
    let body = format!(
        "  stamp(0);\n  {init}\n  red.role = @as(u16, pcfg[1]);\n  {{ var t: u16 = 0; while (t < {rows}) : (t += 1) {{ {step} }} }}{end}\n  stamp(1);\n  sys_mod.unblock_cmd_stream();\n"
    );
    let _ = write!(t, "task go() void {{\n{body}}}\n");
    t.push_str("comptime {\n  @bind_local_task(go, T_GO);\n  @export_symbol(run);\n  @export_symbol(tsc_p, \"tsc\");\n  @export_symbol(pcfg_p, \"pcfg\");\n  @initialize_queue(q_am, .{ .color = @get_color(C_AM) });\n  @initialize_queue(q_r, .{ .color = @get_color(C_R) });\n  @initialize_queue(q_s, .{ .color = @get_color(C_S) });\n");
    if rows_of(p).is_some() {
        t.push_str("  @export_symbol(rowp_p, \"rowp\");\n");
    }
    t.push_str("}\n");
    Ok((name, t))
}
