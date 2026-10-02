//! Gated delta net: the short causal conv, the gate preparation, and the
//! delta rule recurrence, on one PE.
//!
//! Reference: kernels-xla `attn::ssm` and the host references in
//! `engine-xla/tests/kernels_ssm.rs`. Conv state holds `hist = (w-1)·dil+1`
//! rows per slot, oldest first; delta state holds `[hv, dv, dk]` per slot;
//! both banks are f32. Slot `i32::MAX` (or any slot outside the bank) means
//! the write does not land.

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::csl::Arg;
use crate::cx::{Ctx, expect};
use crate::error::{Error, refuse};
use crate::program::{BankSplit, BlockPlan, ColPlane, LaneKind, LanePlan, Segment};
use crate::tensor::{RaggedTensor, RecurrentPool, Tensor};

const MAX_OFFSET: u64 = i16::MAX as u64;

/// Where a row's slot comes from.
enum Lanes {
    /// One row per lane: slot `slots[row]`.
    PerRow,
    /// A CSR over rows: lane `l` spans `indptr[l]..indptr[l+1]`, slot `slots[indptr[l]]`.
    Ragged(Tensor),
}

fn conv_shapes(
    op: &'static str,
    x: Tensor,
    weight: Tensor,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: Tensor,
) -> Result<(u32, u32, u32), Error> {
    expect(op, x, &[Dtype::Bf16, Dtype::F32])?;
    expect(op, y, &[Dtype::Bf16, Dtype::F32])?;
    expect(op, weight, &[Dtype::Bf16, Dtype::F32])?;
    expect(op, state.conv_state, &[Dtype::F32])?;
    expect(op, state.slots, &[Dtype::I32])?;
    if state.conv_state.buf != state.new_conv_state.buf {
        return Err(refuse(
            op,
            "the conv state rolls in place; separate new_conv_state planes are not placed",
        ));
    }
    if conv_width == 0 || dilation == 0 {
        return Err(refuse(
            op,
            format!("conv width {conv_width}, dilation {dilation}"),
        ));
    }
    let c = x.width;
    let hist = (conv_width - 1) * dilation + 1;
    if weight.elements() != u64::from(c) * u64::from(conv_width) {
        return Err(refuse(
            op,
            format!(
                "weight holds {} elements for {c} channels x {conv_width} taps",
                weight.elements()
            ),
        ));
    }
    if state.conv_state.width != hist * c {
        return Err(refuse(
            op,
            format!(
                "conv state rows are {} wide, {hist} x {c} wanted",
                state.conv_state.width
            ),
        ));
    }
    if y.rows < x.rows || y.width != c {
        return Err(refuse(
            op,
            format!("y is {}x{}, x is {}x{}", y.rows, y.width, x.rows, x.width),
        ));
    }
    Ok((c, hist, conv_width))
}

fn conv(
    ctx: &Ctx<'_>,
    op: &'static str,
    x: Tensor,
    lanes: Lanes,
    weight: Tensor,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: Tensor,
) -> Result<(), Error> {
    let (c, hist, width) = conv_shapes(op, x, weight, state, conv_width, dilation, y)?;
    let slots_rows = state.conv_state.rows;
    let stride = hist * c;
    let lanes_n = match &lanes {
        Lanes::PerRow => x.rows,
        Lanes::Ragged(t) => t.elements().saturating_sub(1) as u32,
    };
    // A slot's state is `[hist][c]`: split along the channels, far enough
    // that a block, the window scratch, the rows of `x` and `y` (sharded by
    // the same channels) and the weight's `[c][taps]` (its rows) fit a PE.
    let extra = |ag: u32, bg: u32| {
        let cb = u64::from(c / bg);
        u64::from(hist / ag) * cb
            + u64::from(x.rows + y.rows) * cb
            + cb * u64::from(width)
            + u64::from(x.rows) * 2
    };
    // The window needs every history row: only the channels split.
    let split = lane_split(
        op,
        state.conv_state.elements(),
        lanes_n,
        hist,
        c,
        1,
        false,
        extra,
    )?;
    // Activations and weights travel by channel block when the channels split.
    let shard_cols = split.block.is_some_and(|b| b.b_groups > 1);
    let win_words = split.block.map_or(u64::from(stride), |b| b.words());
    ctx.emit(&mut |cx| {
        let xb = cx.read(x)?;
        let wb = cx.read(weight)?;
        let sb = cx.read(state.slots)?;
        cx.read(state.conv_state)?;
        let bank = cx.write(state.conv_state)?;
        let yb = cx.write(y)?;
        let win = cx.scratch("win", win_words);
        let (indptr, per_row) = match &lanes {
            Lanes::PerRow => (sb.ptr(), true),
            Lanes::Ragged(t) => (cx.read(*t)?.ptr(), false),
        };
        let plane = |b: &crate::program::Buf, by_rows: bool| ColPlane {
            name: b.name.clone(),
            rows: b.rows,
            width: b.width,
            by_rows,
            segments: vec![Segment::along_b()],
            row_block: false,
        };
        let (rows_out, cols) = if shard_cols {
            (None, vec![plane(&xb, false), plane(&yb, false), plane(&wb, true)])
        } else {
            (Some((&yb, 0, 1)), Vec::new())
        };
        let (range, slots_rows) =
            lane_range(cx, &split, &lanes, lanes_n, &sb, &bank, stride, (hist, c), rows_out, cols, slots_rows)?;
        // The conv takes `l0, l1, ch0, ch1`: the block's `b` range.
        let args: Vec<&str> = range.split(", ").collect();
        let range = format!("{}, {}, {}, {}", args[0], args[1], args[4], args[5]);
        cx.library("k_conv1d");
        let mut args = vec![
            Arg::Ptr(xb),
            Arg::Ptr(wb),
            Arg::Ptr(bank.clone()),
            Arg::Ptr(sb.clone()),
            Arg::Expr(indptr.clone()),
            Arg::Ptr(yb),
            Arg::Scratch(win.clone(), "f32"),
        ];
        args.extend(Arg::exprs(&range));
        args.extend([
            Arg::Bool(per_row),
            Arg::Int(i64::from(c)),
            Arg::Int(i64::from(hist)),
            Arg::Int(i64::from(width)),
            Arg::Int(i64::from(dilation)),
            Arg::Int(slots_rows as i64),
            Arg::Bool(shard_cols),
        ]);
        cx.call("k_conv1d", args);
        Ok(())
    })
}

/// A lane split: the PE count, the lanes per PE, and the block (if split).
struct Split {
    pes: u32,
    lanes_per_pe: u32,
    block: Option<BlockPlan>,
}

/// How a recurrent bank of `elements` over `lanes` lanes (slot state
/// `[a][b][w]`, `stride = a·b·w`) spreads over PEs: whole slots per PE when
/// the bank and the phase's other words fit [`pe_words`], else each slot
/// split along `b` and then (when `a_splits`) `a` until one block plus
/// `extra(a_groups, b_groups)` (the words the phase holds per PE besides
/// its bank rows: scratch, its planes, whole buffers) fits, as many lanes
/// per PE as the rest allows. Returns the PE count, the lanes per PE, and
/// the block (if split).
fn lane_split(
    op: &'static str,
    elements: u64,
    lanes: u32,
    a: u32,
    b: u32,
    w: u32,
    a_splits: bool,
    extra: impl Fn(u32, u32) -> u64,
) -> Result<Split, Error> {
    use crate::linear::gemm::{ARRAY_WORDS, pe_words};
    if lanes == 0 || (elements + extra(1, 1) <= pe_words() && elements <= ARRAY_WORDS) {
        return Ok(Split {
            pes: 1,
            lanes_per_pe: lanes.max(1),
            block: None,
        });
    }
    let divisors = |n: u32| (1..=n).filter(move |d| n.is_multiple_of(*d));
    let block_words = |ag: u32, bg: u32| u64::from(a / ag) * u64::from(b / bg) * u64::from(w);
    let mut found = None;
    'search: for bg in divisors(b) {
        for ag in divisors(if a_splits { a } else { 1 }) {
            if block_words(ag, bg) + extra(ag, bg) <= pe_words() {
                found = Some((ag, bg));
                break 'search;
            }
        }
    }
    let Some((a_groups, b_groups)) = found else {
        let budget = pe_words();
        return Err(refuse(
            op,
            format!(
                "one row of a slot's state ({w} words) and the phase's other words ({}) exceed a PE's {budget}",
                extra(a, b)
            ),
        ));
    };
    let plan = BlockPlan {
        a,
        b,
        w,
        a_groups,
        b_groups,
    };
    let words = plan.words().max(1);
    // The bank is one array: its slot rows within ARRAY_WORDS too.
    let room = pe_words()
        .saturating_sub(extra(a_groups, b_groups))
        .min(ARRAY_WORDS);
    let lanes_per_pe = (room / words).clamp(1, u64::from(lanes)) as u32;
    let lane_groups = lanes.div_ceil(lanes_per_pe);
    Ok(Split {
        pes: lane_groups * a_groups * b_groups,
        lanes_per_pe,
        block: Some(plan),
    })
}

/// The call arguments naming a PE's lane range and block (`l0, l1, a0, a1,
/// b0, b1`) and the local slot rows it holds: the whole range on one PE,
/// else read from a lane plan's header.
#[allow(clippy::too_many_arguments)]
fn lane_range(
    cx: &mut crate::cx::Cx<'_>,
    split: &Split,
    lanes: &Lanes,
    lanes_n: u32,
    slots: &crate::program::Buf,
    bank: &crate::program::Buf,
    stride: u32,
    (a, b): (u32, u32),
    rows_out: Option<(&crate::program::Buf, u32, u32)>,
    cols: Vec<ColPlane>,
    slots_rows: u32,
) -> Result<(String, u32), Error> {
    if split.pes <= 1 && split.block.is_none() {
        return Ok((format!("0, {lanes_n}, 0, {a}, 0, {b}"), slots_rows));
    }
    let header = cx.unique("lanes");
    let plan = LanePlan {
        pes: split.pes,
        kind: match lanes {
            Lanes::PerRow => LaneKind::PerRow { lanes: lanes_n },
            Lanes::Ragged(t) => LaneKind::Ragged {
                indptr: cx.read(*t)?.name,
                lanes: lanes_n,
            },
        },
        header: header.clone(),
        slots: slots.name.clone(),
        banks: vec![(bank.name.clone(), stride, BankSplit::Block)],
        block: split.block,
        lanes_per_pe: split.lanes_per_pe,
        a_first: false,
        row_outputs: rows_out
            .into_iter()
            .map(|(b, sa, sb)| (b.name.clone(), b.width, sa, sb))
            .collect(),
        cols,
        reduce: Vec::new(),
        window: None,
        pages: None,
    };
    let hdr = cx.lanes(plan)?;
    Ok((
        format!("{0}[0], {0}[1], {0}[3], {0}[4], {0}[5], {0}[6]", hdr.name),
        split.lanes_per_pe,
    ))
}

/// One row per lane: `y = silu(conv(state ++ x))`, the window shifts by one.
pub fn causal_conv1d(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: Tensor,
) -> Result<(), Error> {
    conv(
        ctx,
        "attention.ssm_causal_conv1d",
        x,
        Lanes::PerRow,
        weight,
        state,
        conv_width,
        dilation,
        y,
    )
}

/// The conv over a query CSR; `state.slots` is per row (a lane's slot is its
/// first row's).
pub fn causal_conv1d_chunked(
    ctx: &Ctx<'_>,
    x: RaggedTensor,
    weight: Tensor,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: Tensor,
) -> Result<(), Error> {
    expect(
        "attention.ssm_causal_conv1d_chunked",
        x.indptr,
        &[Dtype::I32],
    )?;
    conv(
        ctx,
        "attention.ssm_causal_conv1d_chunked",
        x.data,
        Lanes::Ragged(x.indptr),
        weight,
        state,
        conv_width,
        dilation,
        y,
    )
}

/// `[b | a]` → `[g_log | beta]`: `g = -exp(a_log) · softplus(a + dt_bias)`,
/// `beta = σ(b)`, per value head.
pub fn gdn_prep(
    ctx: &Ctx<'_>,
    ba: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    gates: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_gdn_prep";
    expect(OP, ba, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, dt_bias, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, a_log, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, gates, &[Dtype::F32])?;
    if !ba.width.is_multiple_of(2) {
        return Err(refuse(OP, format!("ba is {} wide", ba.width)));
    }
    let hv = ba.width / 2;
    if dt_bias.elements() != u64::from(hv) || a_log.elements() != u64::from(hv) {
        return Err(refuse(
            OP,
            format!(
                "{} biases and {} a_log for {hv} heads",
                dt_bias.elements(),
                a_log.elements()
            ),
        ));
    }
    if gates.rows < ba.rows || gates.width != ba.width {
        return Err(refuse(
            OP,
            format!(
                "gates is {}x{}, ba is {}x{}",
                gates.rows, gates.width, ba.rows, ba.width
            ),
        ));
    }
    ctx.emit(&mut |cx| {
        let bab = cx.read(ba)?;
        let dtb = cx.read(dt_bias)?;
        let alb = cx.read(a_log)?;
        let gb = cx.write(gates)?;
        cx.library("k_gdn_prep");
        cx.call(
            "k_gdn_prep",
            vec![
                Arg::Ptr(bab),
                Arg::Ptr(dtb),
                Arg::Ptr(alb),
                Arg::Ptr(gb),
                Arg::Int(i64::from(ba.rows)),
                Arg::Int(i64::from(hv)),
            ],
        );
        Ok(())
    })
}

fn delta_shapes(
    op: &'static str,
    qkv: Tensor,
    gates: Tensor,
    state: &RecurrentPool,
    hk: u32,
    hv: u32,
    dk: u32,
    dv: u32,
    y: Tensor,
) -> Result<(), Error> {
    expect(op, qkv, &[Dtype::Bf16, Dtype::F32])?;
    expect(op, gates, &[Dtype::F32])?;
    expect(op, y, &[Dtype::F32])?;
    expect(op, state.state, &[Dtype::F32])?;
    expect(op, state.slots, &[Dtype::I32])?;
    if hk == 0 || hv == 0 || !hv.is_multiple_of(hk) || dk == 0 || dv == 0 {
        return Err(refuse(
            op,
            format!("{hv} value heads over {hk} key heads, dk {dk}, dv {dv}"),
        ));
    }
    if qkv.width != 2 * hk * dk + hv * dv {
        return Err(refuse(
            op,
            format!(
                "qkv is {} wide, {} wanted",
                qkv.width,
                2 * hk * dk + hv * dv
            ),
        ));
    }
    if gates.rows < qkv.rows || gates.width != 2 * hv {
        return Err(refuse(
            op,
            format!(
                "gates is {}x{}, {}x{} wanted",
                gates.rows,
                gates.width,
                qkv.rows,
                2 * hv
            ),
        ));
    }
    if y.rows < qkv.rows || y.width != hv * dv {
        return Err(refuse(
            op,
            format!(
                "y is {}x{}, {}x{} wanted",
                y.rows,
                y.width,
                qkv.rows,
                hv * dv
            ),
        ));
    }
    if state.state.width != hv * dv * dk {
        return Err(refuse(
            op,
            format!(
                "state rows are {} wide, {} wanted",
                state.state.width,
                hv * dv * dk
            ),
        ));
    }
    if u64::from(dk) > MAX_OFFSET {
        return Err(refuse(
            op,
            "a state row longer than one PE's DSD offset range",
        ));
    }
    Ok(())
}

fn delta(
    ctx: &Ctx<'_>,
    op: &'static str,
    qkv: Tensor,
    lanes: Lanes,
    gates: Tensor,
    state: &RecurrentPool,
    hk: u32,
    hv: u32,
    dk: u32,
    dv: u32,
    y: Tensor,
) -> Result<(), Error> {
    delta_shapes(op, qkv, gates, state, hk, hv, dk, dv, y)?;
    let width = qkv.width;
    let stride = hv * dv * dk;
    let slots_rows = state.state.rows;
    let lanes_n = match &lanes {
        Lanes::PerRow => qkv.rows,
        Lanes::Ragged(t) => t.elements().saturating_sub(1) as u32,
    };
    // A slot's state is `[hv][dv][dk]`: split along the state rows, then
    // heads. Split, a PE holds of qkv only its heads' q and k (the k heads
    // its v heads share) and its block's v cells, of y its block's cells;
    // the gates stay whole.
    let group = (hv / hk).max(1);
    let extra = |ag: u32, bg: u32| {
        let heads = hv / ag;
        let k_heads = heads.div_ceil(group).max(1);
        let cells = u64::from(heads) * u64::from(dv / bg);
        let per_row = 2 * u64::from(k_heads) * u64::from(dk) + cells;
        u64::from(qkv.rows) * per_row
            + gates.elements()
            + u64::from(y.rows) * cells
            + 2 * u64::from(dk)
    };
    let split = lane_split(op, state.state.elements(), lanes_n, hv, dv, dk, true, extra)?;
    let sharded = split.block.is_some();
    ctx.emit(&mut |cx| {
        let qb = cx.read(qkv)?;
        let gb = cx.read(gates)?;
        let sb = cx.read(state.slots)?;
        cx.read(state.state)?;
        let bank = cx.write(state.state)?;
        let yb = cx.write(y)?;
        let qn = cx.scratch("qn", u64::from(dk));
        let kn = cx.scratch("kn", u64::from(dk));
        let (indptr, per_row) = match &lanes {
            Lanes::PerRow => (sb.ptr(), true),
            Lanes::Ragged(t) => (cx.read(*t)?.ptr(), false),
        };
        let (rows_out, cols) = if sharded {
            let heads = |base: u32| Segment {
                base,
                a_group: group,
                a_stride: dk,
                b_stride: 0,
                span: dk,
            };
            let v_cells = Segment {
                base: 2 * hk * dk,
                a_group: 1,
                a_stride: dv,
                b_stride: 1,
                span: 1,
            };
            let y_cells = Segment {
                base: 0,
                a_group: 1,
                a_stride: dv,
                b_stride: 1,
                span: 1,
            };
            (
                None,
                vec![
                    ColPlane {
                        name: qb.name.clone(),
                        rows: qb.rows,
                        width: qb.width,
                        by_rows: false,
                        segments: vec![heads(0), heads(hk * dk), v_cells],
                        row_block: false,
                    },
                    ColPlane {
                        name: yb.name.clone(),
                        rows: yb.rows,
                        width: yb.width,
                        by_rows: false,
                        segments: vec![y_cells],
                        row_block: false,
                    },
                ],
            )
        } else {
            (Some((&yb, dv, 1)), Vec::new())
        };
        let (range, slots_rows) =
            lane_range(cx, &split, &lanes, lanes_n, &sb, &bank, stride, (hv, dv), rows_out, cols, slots_rows)?;
        cx.library("k_delta");
        let mut args = vec![
            Arg::Ptr(qb),
            Arg::Ptr(gb),
            Arg::Ptr(sb.clone()),
            Arg::Expr(indptr.clone()),
            Arg::Ptr(bank.clone()),
            Arg::Ptr(yb),
            Arg::Scratch(qn.clone(), "f32"),
            Arg::Scratch(kn.clone(), "f32"),
        ];
        args.extend(Arg::exprs(&range));
        args.extend([
            Arg::Bool(per_row),
            Arg::Int(i64::from(hk)),
            Arg::Int(i64::from(hv)),
            Arg::Int(i64::from(dk)),
            Arg::Int(i64::from(dv)),
            Arg::Int(i64::from(width)),
            Arg::Int(slots_rows as i64),
            Arg::Float(1.0 / (dk as f32).sqrt()),
            Arg::Bool(sharded),
        ]);
        cx.call("k_delta", args);
        Ok(())
    })
}

/// One token per lane of the gated delta rule; `y` is the f32 accumulator
/// (`z` gates it later, in the gated rmsnorm).
pub fn gated_delta(
    ctx: &Ctx<'_>,
    qkv: Tensor,
    z: Tensor,
    gates: Tensor,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: Tensor,
) -> Result<(), Error> {
    let _ = z;
    delta(
        ctx,
        "attention.ssm_gated_delta",
        qkv,
        Lanes::PerRow,
        gates,
        state,
        k_heads,
        v_heads,
        k_dim,
        v_dim,
        y,
    )
}

/// The gated delta rule over a query CSR; `state.slots` is per row.
pub fn gated_delta_chunked(
    ctx: &Ctx<'_>,
    qkv: RaggedTensor,
    z: Tensor,
    gates: Tensor,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: Tensor,
) -> Result<(), Error> {
    let _ = z;
    expect(
        "attention.ssm_gated_delta_chunked",
        qkv.indptr,
        &[Dtype::I32],
    )?;
    delta(
        ctx,
        "attention.ssm_gated_delta_chunked",
        qkv.data,
        Lanes::Ragged(qkv.indptr),
        gates,
        state,
        k_heads,
        v_heads,
        k_dim,
        v_dim,
        y,
    )
}
