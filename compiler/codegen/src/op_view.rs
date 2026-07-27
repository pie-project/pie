//! The `pie_native::ptir::container::COp` projection of a [`pie_ir::op::Op`].
//!
//! The C++ emitter switches on a *decoded wire op*: a flat record of tag,
//! positional args, result count, channel index and a handful of immediates.
//! The Rust plan carries the typed [`Op`] enum instead, so the port derives
//! the same record here once and the emitters read it exactly as the C++ ones
//! read `COp`. Keeping the projection in one place is what makes the two
//! emitters diffable line by line.
//!
//! The field set is deliberately the subset the emitter touches — `tag`,
//! `args`, `results`, `chan`, `name_idx`, `intr`, `imm`, `pred_tag`,
//! `pred_payload`. The rest of `COp` (literals, shapes, rng kind) never
//! reaches emitted text.

use pie_ir::op::Op;
use pie_ir::types::{Predicate, ValueId};

/// The wire view of one normalized op, matching `container::decode_op`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OpView {
    pub tag: u8,
    /// `chan_take` / `chan_read` / `chan_put` target, else `-1`.
    pub chan: i64,
    /// `kernel_call` / `sink_call` name-table index.
    pub name_idx: u16,
    /// Positional operands **as the container encodes them**: this is not
    /// [`Op::operands`]. `pivot_threshold` encodes its predicate payload in a
    /// separate field, so `args` holds the input alone.
    pub args: Vec<ValueId>,
    /// SSA ids defined.
    pub results: u32,
    /// `intrinsic_val` id.
    pub intr: u16,
    /// `top_k` k / `iota` len / `rng` stream / mask window immediates.
    pub imm: u32,
    /// `pivot_threshold` predicate tag.
    pub pred_tag: u8,
    /// `pivot_threshold` predicate payload (a value id for every tag).
    pub pred_payload: u32,
}

impl OpView {
    /// Project one op. Mirrors `decode_op`'s per-tag operand layout.
    pub fn of(op: &Op) -> Self {
        let mut view = OpView {
            tag: op.tag(),
            chan: -1,
            results: op.result_count(),
            ..OpView::default()
        };
        match *op {
            // `pivot_threshold` encodes `input` then a 5-byte predicate, so
            // the predicate operand is NOT an arg — unlike `Op::operands`.
            Op::PivotThreshold { input, predicate } => {
                view.args = vec![input];
                let (tag, payload) = match predicate {
                    Predicate::RankLe(value) => (0u8, value),
                    Predicate::CummassLe(value) => (1, value),
                    Predicate::ProbGe(value) => (2, value),
                };
                view.pred_tag = tag;
                view.pred_payload = payload;
            }
            Op::TopK { input, k } => {
                view.args = vec![input];
                view.imm = k;
            }
            Op::Iota { len } => view.imm = len,
            Op::Rng { stream, .. } => view.imm = stream,
            Op::CausalMask { positions, len } => {
                view.args = vec![positions];
                view.imm = len;
            }
            Op::SlidingWindowMask { positions, len, .. } => {
                view.args = vec![positions];
                view.imm = len;
            }
            Op::SinkWindowMask { positions, len, .. } => {
                view.args = vec![positions];
                view.imm = len;
            }
            Op::ChanTake(chan) | Op::ChanRead(chan) => view.chan = chan as i64,
            Op::ChanPut { chan, value } => {
                view.chan = chan as i64;
                view.args = vec![value];
            }
            Op::IntrinsicVal { intr, .. } => view.intr = intr as u16,
            Op::KernelCall { name, ref args, .. } | Op::SinkCall { name, ref args } => {
                view.name_idx = name;
                view.args = args.clone();
            }
            _ => view.args = op.operands(),
        }
        view
    }

    /// Project a whole stage body.
    pub fn of_all(ops: &[Op]) -> Vec<OpView> {
        ops.iter().map(OpView::of).collect()
    }
}

/// `bases[node]` — the SSA id each op's first result defines, the running sum
/// of `results` the C++ emitters recompute at the top of every entry point.
pub fn result_bases(ops: &[OpView]) -> Vec<u32> {
    let mut bases = Vec::with_capacity(ops.len());
    let mut next_value = 0u32;
    for op in ops {
        bases.push(next_value);
        next_value = next_value.wrapping_add(op.results);
    }
    bases
}
