//! The pinned numeric contract: the parts of tier-0 that a backend has to
//! reproduce bit-for-bit, kept together so they can be read together.
//!
//! Reduction is a canonical width-32 tree regardless of launch geometry,
//! argmax breaks ties toward the lower index and never selects NaN, and every
//! lane conversion is exact in the operands' dtype rather than in f32. Those
//! are the rules the MSL and CUDA emitters are diffed against.

use alloc::vec::Vec;

use pie_ir::types::{DType, Shape, ValueType};

use super::Value;

// ── value helpers (dtype-exact, unlike the PSIR f32 evaluator) ─────────────

pub(super) fn lanes_f32(v: &Value) -> Vec<f32> {
    match v {
        Value::F32(x) => x.clone(),
        Value::I32(x) => x.iter().map(|&a| a as f32).collect(),
        Value::U32(x) => x.iter().map(|&a| a as f32).collect(),
        Value::Bool(x) => x.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect(),
    }
}

pub(super) fn lanes_i64(v: &Value) -> Vec<i64> {
    match v {
        Value::F32(x) => x.iter().map(|&a| a as i64).collect(),
        Value::I32(x) => x.iter().map(|&a| a as i64).collect(),
        Value::U32(x) => x.iter().map(|&a| a as i64).collect(),
        Value::Bool(x) => x.iter().map(|&b| b as i64).collect(),
    }
}

pub(super) fn from_i64(dtype: DType, x: Vec<i64>) -> Value {
    match dtype {
        DType::I32 => Value::I32(x.iter().map(|&a| a as i32).collect()),
        DType::U32 => Value::U32(x.iter().map(|&a| a as u32).collect()),
        DType::F32 => Value::F32(x.iter().map(|&a| a as f32).collect()),
        DType::Bool => Value::Bool(x.iter().map(|&a| a != 0).collect()),
    }
}

pub(super) fn pick(len: usize, i: usize) -> usize {
    if len == 1 { 0 } else { i }
}

/// Elementwise binary, exact in the operands' common dtype.
pub(super) fn bin_arith(
    a: &Value,
    b: &Value,
    dtype: DType,
    f_f: impl Fn(f32, f32) -> f32,
    f_i: impl Fn(i64, i64) -> i64,
) -> Value {
    if dtype == DType::F32 {
        let (av, bv) = (lanes_f32(a), lanes_f32(b));
        let n = av.len().max(bv.len());
        Value::F32(
            (0..n)
                .map(|i| f_f(av[pick(av.len(), i)], bv[pick(bv.len(), i)]))
                .collect(),
        )
    } else {
        let (av, bv) = (lanes_i64(a), lanes_i64(b));
        let n = av.len().max(bv.len());
        from_i64(
            dtype,
            (0..n)
                .map(|i| f_i(av[pick(av.len(), i)], bv[pick(bv.len(), i)]))
                .collect(),
        )
    }
}

pub(super) fn cmp_op(
    a: &Value,
    b: &Value,
    in_dtype: DType,
    f_f: impl Fn(f32, f32) -> bool,
    f_i: impl Fn(i64, i64) -> bool,
) -> Value {
    if in_dtype == DType::F32 {
        let (av, bv) = (lanes_f32(a), lanes_f32(b));
        let n = av.len().max(bv.len());
        Value::Bool(
            (0..n)
                .map(|i| f_f(av[pick(av.len(), i)], bv[pick(bv.len(), i)]))
                .collect(),
        )
    } else {
        let (av, bv) = (lanes_i64(a), lanes_i64(b));
        let n = av.len().max(bv.len());
        Value::Bool(
            (0..n)
                .map(|i| f_i(av[pick(av.len(), i)], bv[pick(bv.len(), i)]))
                .collect(),
        )
    }
}

pub(super) fn map_f32(v: &Value, f: impl Fn(f32) -> f32) -> Value {
    Value::F32(lanes_f32(v).into_iter().map(f).collect())
}

/// Canonical width-32 tree. Physical launch dimensions never affect this
/// logical order.
pub(super) fn canonical_reduce<T: Copy>(
    row: &[T],
    identity: T,
    combine: impl Fn(T, T) -> T + Copy,
) -> T {
    if row.is_empty() {
        return identity;
    }
    let mut level = row.to_vec();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(32));
        for chunk in level.chunks(32) {
            let mut lanes = [identity; 32];
            lanes[..chunk.len()].copy_from_slice(chunk);
            for offset in [16usize, 8, 4, 2, 1] {
                for lane in 0..offset {
                    lanes[lane] = combine(lanes[lane], lanes[lane + offset]);
                }
            }
            next.push(lanes[0]);
        }
        level = next;
    }
    level[0]
}

#[derive(Clone, Copy)]
pub(super) struct ArgmaxCandidate {
    value: f32,
    index: u32,
    have: bool,
}

pub(super) fn combine_argmax(left: ArgmaxCandidate, right: ArgmaxCandidate) -> ArgmaxCandidate {
    match (left.have, right.have) {
        (false, false) => left,
        (true, false) => left,
        (false, true) => right,
        (true, true) => {
            if right.value > left.value || (right.value == left.value && right.index < left.index) {
                right
            } else {
                left
            }
        }
    }
}

/// Inclusive prefix scan of each row, combining with `combine`.
///
/// Sequential and left-to-right, unlike [`reduce_tree`]'s width-32 tree: a
/// scan's every prefix is an output, so there is no associativity freedom for
/// a launch geometry to spend and no canonical tree to pin. The caller passes
/// the combiner rather than an `Add` bound so that integer lanes can say
/// `wrapping_add` — a scan whose sum leaves the dtype must wrap the way the
/// device wraps, not panic in a debug build and wrap in a release one.
pub(super) fn scan_rows<T: Copy>(
    lanes: &[T],
    rows: usize,
    identity: T,
    combine: impl Fn(T, T) -> T,
) -> Vec<T> {
    let len = lanes.len().checked_div(rows).unwrap_or(0);
    let mut out = Vec::with_capacity(lanes.len());
    for row in 0..rows {
        let mut acc = identity;
        for &lane in &lanes[row * len..(row + 1) * len] {
            acc = combine(acc, lane);
            out.push(acc);
        }
    }
    out
}

/// Argmax with the pinned contract: lower index wins ties; NaN never selected
/// (all-NaN row -> 0), evaluated through the canonical tree.
pub(super) fn argmax_row(row: &[f32]) -> i32 {
    let candidates: Vec<_> = row
        .iter()
        .enumerate()
        .map(|(index, &value)| ArgmaxCandidate {
            value,
            index: index as u32,
            have: !value.is_nan(),
        })
        .collect();
    canonical_reduce(
        &candidates,
        ArgmaxCandidate {
            value: f32::NEG_INFINITY,
            index: 0,
            have: false,
        },
        combine_argmax,
    )
    .index as i32
}

pub(super) fn argmax_ordered<T: Ord>(row: &[T]) -> i32 {
    let Some((mut best_index, mut best)) = row.first().map(|value| (0usize, value)) else {
        return 0;
    };
    for (index, value) in row.iter().enumerate().skip(1) {
        if value > best {
            best = value;
            best_index = index;
        }
    }
    best_index as i32
}

/// Which end of the float order an extremum walks toward.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Extremum {
    Max,
    Min,
}

/// What a NaN-against-NaN pair produces — the only axis on which the reduction
/// and elementwise forms differ.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum NanPair {
    /// A reduction folds NaN∧NaN to its identity, so an all-NaN row reduces to
    /// ∓inf rather than to NaN. (A single NaN is already dropped by the
    /// asymmetric arms, so this is what makes the fold associative.)
    Identity,
    /// Elementwise `max`/`min` propagate: NaN∧NaN yields the left operand.
    Left,
}

/// The one extremum rule. IEEE leaves two cases to the caller — NaN pairs and
/// signed zeros — and both matter here: this is the tier-0 oracle, so whatever
/// it decides is the contract every backend is compared against.
pub(super) fn extremum(left: f32, right: f32, end: Extremum, pair: NanPair) -> f32 {
    match (left.is_nan(), right.is_nan()) {
        (true, true) => match pair {
            NanPair::Identity => match end {
                Extremum::Max => f32::NEG_INFINITY,
                Extremum::Min => f32::INFINITY,
            },
            NanPair::Left => left,
        },
        (true, false) => right,
        (false, true) => left,
        // `-0.0 == 0.0`, so `f32::max`/`min` are free to return either operand.
        // Pin the sign instead: a max is negative only when both inputs are, a
        // min whenever either is. Getting these backwards is invisible in every
        // comparison and visible in `to_bits`.
        (false, false) if left == 0.0 && right == 0.0 => {
            let negative = match end {
                Extremum::Max => left.is_sign_negative() && right.is_sign_negative(),
                Extremum::Min => left.is_sign_negative() || right.is_sign_negative(),
            };
            if negative { -0.0 } else { 0.0 }
        }
        (false, false) => match end {
            Extremum::Max => left.max(right),
            Extremum::Min => left.min(right),
        },
    }
}

/// `reduce_max`'s combiner — see [`NanPair::Identity`].
pub(super) fn canonical_max(left: f32, right: f32) -> f32 {
    extremum(left, right, Extremum::Max, NanPair::Identity)
}

/// `reduce_min`'s combiner — see [`NanPair::Identity`].
pub(super) fn canonical_min(left: f32, right: f32) -> f32 {
    extremum(left, right, Extremum::Min, NanPair::Identity)
}

/// `max_elem`'s combiner — see [`NanPair::Left`].
pub(super) fn element_max(left: f32, right: f32) -> f32 {
    extremum(left, right, Extremum::Max, NanPair::Left)
}

/// `min_elem`'s combiner — see [`NanPair::Left`].
pub(super) fn element_min(left: f32, right: f32) -> f32 {
    extremum(left, right, Extremum::Min, NanPair::Left)
}

/// sort_desc order with the pinned contract: descending; ties → lower
/// original index first; NaN below −inf (last).
pub(super) fn sort_desc_order(row: &[f32]) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..row.len() as u32).collect();
    idx.sort_by(|&a, &b| {
        let (x, y) = (row[a as usize], row[b as usize]);
        match (x.is_nan(), y.is_nan()) {
            (true, true) => a.cmp(&b),
            (true, false) => core::cmp::Ordering::Greater, // NaN last
            (false, true) => core::cmp::Ordering::Less,
            (false, false) => y.partial_cmp(&x).unwrap().then(a.cmp(&b)),
        }
    });
    idx
}

/// [`Shape::rows`] as an index type, saturating rather than truncating.
///
/// A row count only has to fit `u64`, so narrowing it is a real conversion
/// even on a 64-bit host. Saturating keeps the answer conservative: every
/// caller divides a materialized lane count by this, and no buffer the
/// interpreter holds is `usize::MAX` long, so an unrepresentable row count
/// yields an empty row rather than a wrapped-small one that would slice the
/// data at the wrong stride.
pub(super) fn rows_of(shape: Shape) -> usize {
    usize::try_from(shape.rows()).unwrap_or(usize::MAX)
}

/// Which of the three row reductions [`eval_op`] is evaluating.
///
/// This exists because the dispatch it replaces re-matched `op` *inside* a
/// combined `ReduceSum | ReduceMax | ReduceMin` arm and used `_` for the third
/// case. That arm answered "minimum" for anything it did not recognise, so a
/// fourth reduce op routed through it would have made the tier-0 oracle every
/// backend diffs against return a silently wrong number -- not an error, a
/// plausible one. Naming the three makes the compiler ask the question.
#[derive(Clone, Copy)]
pub(super) enum ReduceKind {
    Sum,
    Max,
    Min,
}

pub(super) fn reduce_rows(kind: ReduceKind, ty: ValueType, data: &Value) -> Value {
    let rows = rows_of(ty.shape);
    let len = data.len().checked_div(rows).unwrap_or(0);
    if ty.dtype == DType::F32 {
        let x = lanes_f32(data);
        let f: fn(&[f32]) -> f32 = match kind {
            ReduceKind::Sum => |row| canonical_reduce(row, 0.0, |a, b| a + b),
            ReduceKind::Max => |row| canonical_reduce(row, f32::NEG_INFINITY, canonical_max),
            ReduceKind::Min => |row| canonical_reduce(row, f32::INFINITY, canonical_min),
        };
        Value::F32((0..rows).map(|r| f(&x[r * len..(r + 1) * len])).collect())
    } else {
        let x = lanes_i64(data);
        let f: fn(&[i64]) -> i64 = match kind {
            ReduceKind::Sum => |row| canonical_reduce(row, 0, i64::wrapping_add),
            ReduceKind::Max => |row| canonical_reduce(row, i64::MIN, i64::max),
            ReduceKind::Min => |row| canonical_reduce(row, i64::MAX, i64::min),
        };
        from_i64(
            ty.dtype,
            (0..rows).map(|r| f(&x[r * len..(r + 1) * len])).collect(),
        )
    }
}
