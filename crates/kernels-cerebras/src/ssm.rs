//! State-space mixers. Gated delta net: the short causal conv, the gate
//! preparation, and the delta rule recurrence, on one PE.
//!
//! Reference: kernels-xla `ssm` and the host references in
//! `engine-xla/tests/kernels_ssm.rs`. Conv state holds `hist = (w-1)·dil+1`
//! rows per slot, oldest first; delta state holds `[hv, dv, dk]` per slot;
//! both banks are f32. Slot `i32::MAX` (or any slot outside the bank) means
//! the write does not land.

#![allow(clippy::too_many_arguments)]

pub mod dyn_conv;

use dtype::Dtype;

use crate::csl::Arg;
use crate::cx::{Ctx, expect};
use crate::error::{Error, refuse};
use crate::library::Kernel;
use crate::phase::{BankSplit, Block, ColPlane, LaneKind, Lanes, Segment};
use crate::plan::{LaneSplit, Slots};
use crate::tensor::{RaggedTensor, RecurrentPool, Tensor};

const MAX_OFFSET: u64 = i16::MAX as u64;

/// Where a row's slot comes from.
enum LaneRows {
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
    lanes: LaneRows,
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
        LaneRows::PerRow => x.rows,
        LaneRows::Ragged(t) => t.elements().saturating_sub(1) as u32,
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
    let split = ctx.plan().slots(
        op,
        &Slots {
            elements: state.conv_state.elements(),
            lanes: lanes_n,
            a: hist,
            b: c,
            w: 1,
            a_splits: false,
            beside: &extra,
        },
    )?;
    // Activations and weights travel by channel block when the channels split.
    let shard_cols = split.block.is_some_and(|(_, bg)| bg > 1);
    let win_words = split.block.map_or(u64::from(stride), |(ag, bg)| {
        u64::from(hist / ag) * u64::from(c / bg)
    });
    ctx.emit(&mut |cx| {
        let xb = cx.read(x)?;
        let wb = cx.read(weight)?;
        let sb = cx.read(state.slots)?;
        cx.read(state.conv_state)?;
        let bank = cx.write(state.conv_state)?;
        let yb = cx.write(y)?;
        let win = cx.scratch("win", win_words);
        let (indptr, per_row) = match &lanes {
            LaneRows::PerRow => (sb.ptr(), true),
            LaneRows::Ragged(t) => (cx.read(*t)?.ptr(), false),
        };
        let plane = |b: &crate::phase::Buf, by_rows: bool| ColPlane {
            name: b.name.clone(),
            rows: b.rows,
            width: b.width,
            by_rows,
            segments: vec![Segment::along_b()],
            row_block: false,
        };
        let (rows_out, cols) = if shard_cols {
            (
                None,
                vec![plane(&xb, false), plane(&yb, false), plane(&wb, true)],
            )
        } else {
            (Some((&yb, 0, 1)), Vec::new())
        };
        let (range, slots_rows) = lane_range(
            cx,
            &split,
            &lanes,
            lanes_n,
            &sb,
            &bank,
            stride,
            Block {
                a: hist,
                b: c,
                w: 1,
            },
            rows_out,
            cols,
            slots_rows,
        )?;
        // The conv takes `l0, l1, ch0, ch1`: the block's `b` range.
        let args: Vec<&str> = range.split(", ").collect();
        let range = format!("{}, {}, {}, {}", args[0], args[1], args[4], args[5]);
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
        cx.call(Kernel::Conv1d, args);
        Ok(())
    })
}

/// The call arguments naming a PE's lane range and block (`l0, l1, a0, a1,
/// b0, b1`) and the local slot rows it holds: the whole range on one PE,
/// else read from a lane plan's header.
#[allow(clippy::too_many_arguments)]
fn lane_range(
    cx: &mut crate::cx::Cx<'_>,
    split: &LaneSplit,
    rows: &LaneRows,
    lanes_n: u32,
    slots: &crate::phase::Buf,
    bank: &crate::phase::Buf,
    stride: u32,
    block: Block,
    rows_out: Option<(&crate::phase::Buf, u32, u32)>,
    cols: Vec<ColPlane>,
    slots_rows: u32,
) -> Result<(String, u32), Error> {
    if split.pes <= 1 && split.block.is_none() {
        return Ok((
            format!("0, {lanes_n}, 0, {}, 0, {}", block.a, block.b),
            slots_rows,
        ));
    }
    let lanes = Lanes {
        kind: match rows {
            LaneRows::PerRow => LaneKind::PerRow { lanes: lanes_n },
            LaneRows::Ragged(t) => LaneKind::Ragged {
                indptr: cx.read(*t)?.name,
                lanes: lanes_n,
            },
        },
        slots: slots.name.clone(),
        banks: vec![(bank.name.clone(), stride, BankSplit::Block)],
        block: Some(block),
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
    let hdr = cx.lanes(lanes, *split)?.header;
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
        "ssm.causal_conv1d",
        x,
        LaneRows::PerRow,
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
    expect("ssm.causal_conv1d_chunked", x.indptr, &[Dtype::I32])?;
    conv(
        ctx,
        "ssm.causal_conv1d_chunked",
        x.data,
        LaneRows::Ragged(x.indptr),
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
    const OP: &str = "ssm.gdn_gates";
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
        cx.call(
            Kernel::GdnPrep,
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
    lanes: LaneRows,
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
        LaneRows::PerRow => qkv.rows,
        LaneRows::Ragged(t) => t.elements().saturating_sub(1) as u32,
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
    let split = ctx.plan().slots(
        op,
        &Slots {
            elements: state.state.elements(),
            lanes: lanes_n,
            a: hv,
            b: dv,
            w: dk,
            a_splits: true,
            beside: &extra,
        },
    )?;
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
            LaneRows::PerRow => (sb.ptr(), true),
            LaneRows::Ragged(t) => (cx.read(*t)?.ptr(), false),
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
        let (range, slots_rows) = lane_range(
            cx,
            &split,
            &lanes,
            lanes_n,
            &sb,
            &bank,
            stride,
            Block {
                a: hv,
                b: dv,
                w: dk,
            },
            rows_out,
            cols,
            slots_rows,
        )?;
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
        cx.call(Kernel::Delta, args);
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
        "ssm.gated_delta",
        qkv,
        LaneRows::PerRow,
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
    expect("ssm.gated_delta_chunked", qkv.indptr, &[Dtype::I32])?;
    delta(
        ctx,
        "ssm.gated_delta_chunked",
        qkv.data,
        LaneRows::Ragged(qkv.indptr),
        gates,
        state,
        k_heads,
        v_heads,
        k_dim,
        v_dim,
        y,
    )
}
