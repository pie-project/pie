//! A recurrence on a unit's state column (P8, `ops/sst.csl`, `ops/cy_sst.csl`): each head's state
//! held over the column's PEs, its root running the head's conv, gates and step. Gated
//! DeltaNet, Mamba-2 and Kimi's delta attention are its variants.

use std::fmt::Write as _;

use super::super::{Column, ColumnAsk, Kernel, R, Source};
use crate::emit::stream::{COMMON, Fill, HostArray, buf, cfg_decl, kv_joint, rowp_array, rows_of};
use crate::floor::Coord;
use crate::floorplan::stream::KvColumn;
use crate::stream::build::PYA;
use crate::stream::build::Program;
use crate::stream::place::KvSched;
use crate::stream::unit::{Cols, Joint};

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
            (
                "sst.csl",
                include_str!("../../../programs/wafer/ops/sst.csl"),
            ),
            (
                "cy_sst.csl",
                include_str!("../../../programs/wafer/ops/cy_sst.csl"),
            ),
        ]
    }

    fn column(&self) -> Option<&dyn Column> {
        Some(self)
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

/// A state column's root (P8, ops/cy_sst.csl): each row of its unit's, its head's [q | k | v |
/// b | a] in, the step, its output back.
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

/// A state column's PE (P8, ops/sst.csl): its rows of each head's state, a step a row.
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
    let (init, step) = if rows_of(p).is_some() {
        ("st.myrow = @as(u16, pcfg[0]);", "st.row(t);")
    } else {
        ("st.init(@as(u16, pcfg[0]));", "st.step();")
    };
    let body = format!(
        "  stamp(0);\n  {init}\n  red.role = @as(u16, pcfg[1]);\n  {{ var t: u16 = 0; while (t < {rows}) : (t += 1) {{ {step} }} }}\n  stamp(1);\n  sys_mod.unblock_cmd_stream();\n"
    );
    let _ = write!(t, "task go() void {{\n{body}}}\n");
    t.push_str("comptime {\n  @bind_local_task(go, T_GO);\n  @export_symbol(run);\n  @export_symbol(tsc_p, \"tsc\");\n  @export_symbol(pcfg_p, \"pcfg\");\n  @initialize_queue(q_am, .{ .color = @get_color(C_AM) });\n  @initialize_queue(q_r, .{ .color = @get_color(C_R) });\n  @initialize_queue(q_s, .{ .color = @get_color(C_S) });\n");
    if rows_of(p).is_some() {
        t.push_str("  @export_symbol(rowp_p, \"rowp\");\n");
    }
    t.push_str("}\n");
    Ok((name, t))
}
