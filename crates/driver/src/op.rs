//! Op evaluation: the numeric helpers and the switch over the op set.
//!
//! [`eval_op`] is the interpreter's arithmetic core — one stage-body op in, the
//! result cell(s) written into the SSA value array. Everything above it in the
//! module is the small set of numerically delicate primitives the reference
//! semantics pin exactly.
//!
//! # The canonical helpers are the whole point
//!
//! [`canonical_reduce`], [`canonical_max`], [`canonical_min`], [`combine_argmax`]
//! and [`sort_desc_order`] exist so a reduction produces the **same bits** here
//! as it does in the CUDA path and the compiler's reference evaluator. That is
//! not a nicety: a reduction that is merely *close* yields a plausible, wrong
//! token, and the divergence is invisible until a replay is compared against its
//! original. So the order is fixed (a width-32 pairwise tree, not a left fold),
//! `NaN` propagation is fixed (a max over an all-`NaN` row is `-inf`, and `NaN`
//! never wins an argmax), and signed zero is fixed (`max(-0, +0) = +0`). Each of
//! these has a test that fails if the rule is relaxed.

use driver_api::plan::{LaunchOp, LaunchPackage};
use tensor_ir::op::tags;
use tensor_ir::{DType, RngKind, rng};

use super::shape_numel;
use super::value::{Value, concrete_dtype, pick};
use crate::{Error, Result};

/// The number of rows a shape reduces over: the product of every axis but the
/// last (one for a scalar or vector). The per-row length is `numel / rows`.
///
/// This is the *logical* row count and must never depend on physical launch
/// geometry — a reduction's shape decides its order, and the order decides its
/// bits.
#[must_use]
pub fn canonical_rows(dims: &[u32]) -> usize {
    if dims.len() < 2 {
        return 1;
    }
    dims[..dims.len() - 1].iter().map(|&d| d as usize).product()
}

/// A width-32 pairwise tree reduction — the canonical order shared with the
/// reference evaluator.
///
/// Not a left fold: floating-point addition is not associative, so the shape of
/// the reduction tree is part of the answer. This groups lanes into blocks of
/// 32, folds each block by halving offsets (`16, 8, 4, 2, 1`), then reduces the
/// per-block results the same way, matching a warp-width GPU reduction. An empty
/// row returns `identity`.
pub fn canonical_reduce<T: Copy>(row: &[T], identity: T, combine: impl Fn(T, T) -> T) -> T {
    if row.is_empty() {
        return identity;
    }
    let mut level = row.to_vec();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(32));
        let mut base = 0;
        while base < level.len() {
            let mut lanes = [identity; 32];
            let count = 32.min(level.len() - base);
            lanes[..count].copy_from_slice(&level[base..base + count]);
            for offset in [16usize, 8, 4, 2, 1] {
                for lane in 0..offset {
                    lanes[lane] = combine(lanes[lane], lanes[lane + offset]);
                }
            }
            next.push(lanes[0]);
            base += 32;
        }
        level = next;
    }
    level[0]
}

/// Canonical float maximum: `NaN`-aware and signed-zero-aware.
///
/// `fmax` alone gets two cases wrong for reproducibility. First, a max whose
/// operands are both `NaN` must be `-inf` (the reduction identity), so an all-
/// `NaN` row reduces to `-inf` rather than a `NaN` that would then poison an
/// argmax. Second, `max(-0.0, +0.0)` must be `+0.0` (and only `max(-0, -0)` is
/// `-0`), because the sign of a zero is observable through `1/x` and must not
/// depend on operand order.
#[must_use]
pub fn canonical_max(left: f32, right: f32) -> f32 {
    match (left.is_nan(), right.is_nan()) {
        (true, true) => f32::NEG_INFINITY,
        (true, false) => right,
        (false, true) => left,
        (false, false) => {
            if left == 0.0 && right == 0.0 {
                if left.is_sign_negative() && right.is_sign_negative() {
                    -0.0
                } else {
                    0.0
                }
            } else {
                left.max(right)
            }
        }
    }
}

/// Canonical float minimum, the mirror of [`canonical_max`]: an all-`NaN` row
/// reduces to `+inf`, and `min(-0.0, +0.0)` is `-0.0` (only `min(+0, +0)` is
/// `+0`).
#[must_use]
pub fn canonical_min(left: f32, right: f32) -> f32 {
    match (left.is_nan(), right.is_nan()) {
        (true, true) => f32::INFINITY,
        (true, false) => right,
        (false, true) => left,
        (false, false) => {
            if left == 0.0 && right == 0.0 {
                if left.is_sign_negative() || right.is_sign_negative() {
                    -0.0
                } else {
                    0.0
                }
            } else {
                left.min(right)
            }
        }
    }
}

/// One candidate in a float argmax reduction.
#[derive(Clone, Copy, Debug)]
pub struct ArgmaxCandidate {
    /// The lane's value.
    pub value: f32,
    /// The lane's index.
    pub index: u32,
    /// Whether this candidate is eligible (`false` for a `NaN` lane, which must
    /// never be selected).
    pub have: bool,
}

impl Default for ArgmaxCandidate {
    /// The reduction identity: ineligible, so it loses to any real candidate.
    /// `value` is `-inf` for parity with the C++ default, though `have == false`
    /// already keeps it from ever being chosen.
    fn default() -> Self {
        ArgmaxCandidate {
            value: f32::NEG_INFINITY,
            index: 0,
            have: false,
        }
    }
}

/// One candidate in an integer argmax reduction.
#[derive(Clone, Copy, Debug, Default)]
pub struct IntArgmaxCandidate {
    /// The lane's value.
    pub value: i64,
    /// The lane's index.
    pub index: u32,
    /// Whether this candidate is eligible.
    pub have: bool,
}

/// Combine two float argmax candidates: higher value wins, ties break to the
/// **lower index**.
///
/// The tie-break is a pinned contract, not an accident of iteration order: the
/// same row must argmax to the same lane on every backend, so a tie always
/// resolves to the smallest index regardless of how the tree groups the lanes.
#[must_use]
pub fn combine_argmax(left: ArgmaxCandidate, right: ArgmaxCandidate) -> ArgmaxCandidate {
    if !right.have {
        return left;
    }
    if !left.have
        || right.value > left.value
        || (right.value == left.value && right.index < left.index)
    {
        return right;
    }
    left
}

/// Combine two integer argmax candidates, with the same lower-index tie-break as
/// [`combine_argmax`].
#[must_use]
pub fn combine_int_argmax(
    left: IntArgmaxCandidate,
    right: IntArgmaxCandidate,
) -> IntArgmaxCandidate {
    if !right.have {
        return left;
    }
    if !left.have
        || right.value > left.value
        || (right.value == left.value && right.index < left.index)
    {
        return right;
    }
    left
}

/// Argmax of a float row through the canonical tree: lower index wins ties,
/// `NaN` is never selected, and an all-`NaN` row answers `0`.
#[must_use]
pub fn argmax_row(row: &[f32]) -> i32 {
    let candidates: Vec<ArgmaxCandidate> = row
        .iter()
        .enumerate()
        .map(|(j, &value)| ArgmaxCandidate {
            value,
            index: j as u32,
            have: !value.is_nan(),
        })
        .collect();
    canonical_reduce(&candidates, ArgmaxCandidate::default(), combine_argmax).index as i32
}

/// Argmax of an integer row through the canonical tree, with the same tie-break.
#[must_use]
pub fn argmax_row_i64(row: &[i64]) -> i32 {
    let candidates: Vec<IntArgmaxCandidate> = row
        .iter()
        .enumerate()
        .map(|(index, &value)| IntArgmaxCandidate {
            value,
            index: index as u32,
            have: true,
        })
        .collect();
    canonical_reduce(
        &candidates,
        IntArgmaxCandidate::default(),
        combine_int_argmax,
    )
    .index as i32
}

/// The descending sort order of a float row: ties resolve to the lower original
/// index, and `NaN` sorts below `-inf` (last).
///
/// A **stable** sort over indices, so the tie-break is exact rather than
/// implementation-defined. Returns the permutation, not the sorted values, so
/// callers (`sort_desc`, `top_k`, the top-p pivot) can reorder both a value and
/// its index array consistently.
#[must_use]
pub fn sort_desc_order(row: &[f32]) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..row.len() as u32).collect();
    idx.sort_by(|&a, &b| {
        let x = row[a as usize];
        let y = row[b as usize];
        let nx = x.is_nan();
        let ny = y.is_nan();
        match (nx, ny) {
            // Exactly one is NaN: the NaN sorts last.
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            // Both NaN, or a value tie: lower original index first.
            (true, true) => a.cmp(&b),
            (false, false) => {
                if x == y {
                    a.cmp(&b)
                } else {
                    // Descending: the larger value comes first.
                    y.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)
                }
            }
        }
    });
    idx
}

/// Draw `n` RNG lanes from `seed_eff`, uniform or Gumbel-transformed.
///
/// Reuses the canonical RNG contract in [`tensor_ir::rng`] — the Rust source of
/// truth the C++ `rng_contract.generated.h` is projected from — rather than
/// transcribing the hash. Gumbel noise is `-ln(-ln(u))`; because
/// [`tensor_ir::rng::hash_uniform`] is bounded strictly below `1.0`, the inner
/// `-ln(u)` is finite and the transform never produces `+inf`.
#[must_use]
pub fn rng_lanes(seed_eff: u64, n: usize, gumbel: bool) -> Vec<f32> {
    (0..n)
        .map(|j| {
            let u = rng::hash_uniform(seed_eff, j as u32);
            if gumbel { -(-u.ln()).ln() } else { u }
        })
        .collect()
}

/// Gather lanes by flat index, dtype-preserving. A `None` index fills zero — the
/// out-of-range rule the index ops share.
///
/// `None` rather than a `usize::MAX` sentinel: "no source lane" is the absence
/// of an index, and modeling it as a real-looking maximum index is exactly the
/// kind of sentinel this port removes.
#[must_use]
pub fn gather_flat(v: &Value, idx: &[Option<usize>]) -> Value {
    match v {
        Value::I32(src) => Value::I32(idx.iter().map(|&i| i.map_or(0, |i| src[i])).collect()),
        Value::U32(src) => Value::U32(idx.iter().map(|&i| i.map_or(0, |i| src[i])).collect()),
        Value::Bool(src) => Value::Bool(idx.iter().map(|&i| i.map_or(0, |i| src[i])).collect()),
        Value::F32(src) => Value::F32(idx.iter().map(|&i| i.map_or(0.0, |i| src[i])).collect()),
    }
}

/// Left-aligned broadcast replicate, dtype-preserving.
///
/// Source dims align to the **leading** target dims (not the trailing ones, as
/// NumPy would); a source extent of 1 (or a missing leading axis) replicates
/// across the corresponding target axis. Implemented as an index map fed to
/// [`gather_flat`], so it inherits the dtype dispatch for free.
#[must_use]
pub fn broadcast_value(v: &Value, src: &[u32], target: &[u32]) -> Value {
    let r = target.len();
    let sdim = |i: usize| -> u64 { if i < src.len() { u64::from(src[i]) } else { 1 } };
    let mut sstride = vec![1u64; r.max(1)];
    for i in (0..r.saturating_sub(1)).rev() {
        sstride[i] = sstride[i + 1] * sdim(i + 1);
    }
    let n = shape_numel(target) as usize;
    let mut idx = Vec::with_capacity(n);
    for lin in 0..n as u64 {
        let mut rem = lin;
        let mut sidx = 0u64;
        for i in 0..r {
            let mut stride = 1u64;
            for &d in &target[i + 1..r] {
                stride *= u64::from(d);
            }
            let stride = stride.max(1);
            let coord = rem / stride;
            rem %= stride;
            if sdim(i) != 1 {
                sidx += coord * sstride[i];
            }
        }
        idx.push(Some(sidx as usize));
    }
    gather_flat(v, &idx)
}

/// Elementwise binary arithmetic on the float or integer path, per the operand
/// dtype, with scalar broadcast on either side.
fn bin_arith(
    a: &Value,
    b: &Value,
    dtype: DType,
    f_f: impl Fn(f32, f32) -> f32,
    f_i: impl Fn(i64, i64) -> i64,
) -> Value {
    if dtype == DType::F32 {
        let av = a.lanes_f32();
        let bv = b.lanes_f32();
        let n = av.len().max(bv.len());
        Value::F32(
            (0..n)
                .map(|i| f_f(av[pick(av.len(), i)], bv[pick(bv.len(), i)]))
                .collect(),
        )
    } else {
        let av = a.lanes_i64();
        let bv = b.lanes_i64();
        let n = av.len().max(bv.len());
        let o: Vec<i64> = (0..n)
            .map(|i| f_i(av[pick(av.len(), i)], bv[pick(bv.len(), i)]))
            .collect();
        Value::from_i64(dtype, &o)
    }
}

/// Elementwise comparison to a `Bool` result, on the float or integer path.
fn cmp_op(
    a: &Value,
    b: &Value,
    in_dtype: DType,
    f_f: impl Fn(f32, f32) -> bool,
    f_i: impl Fn(i64, i64) -> bool,
) -> Value {
    if in_dtype == DType::F32 {
        let av = a.lanes_f32();
        let bv = b.lanes_f32();
        let n = av.len().max(bv.len());
        Value::Bool(
            (0..n)
                .map(|i| u8::from(f_f(av[pick(av.len(), i)], bv[pick(bv.len(), i)])))
                .collect(),
        )
    } else {
        let av = a.lanes_i64();
        let bv = b.lanes_i64();
        let n = av.len().max(bv.len());
        Value::Bool(
            (0..n)
                .map(|i| u8::from(f_i(av[pick(av.len(), i)], bv[pick(bv.len(), i)])))
                .collect(),
        )
    }
}

/// Apply a scalar float function to every lane, converting the input to `f32`.
fn map_f32(v: &Value, f: impl Fn(f32) -> f32) -> Value {
    Value::F32(v.lanes_f32().into_iter().map(f).collect())
}

/// The dtype of value `id` in the trace, folded through [`concrete_dtype`].
fn ty_dtype(package: &LaunchPackage, id: usize) -> DType {
    concrete_dtype(package.values[id].dtype)
}

/// The shape of value `id` in the trace.
fn ty_shape(package: &LaunchPackage, id: usize) -> &[u32] {
    &package.values[id].shape
}

/// Evaluate one compute op, its SSA operands already resolved in `vals`.
///
/// Writes the result cell(s) into `vals` at the op's `result_id` (two cells for
/// `sort_desc`/`top_k`) and returns `Ok(())`. A semantic fault — a bool op on a
/// non-bool operand, a rank mismatch, an op this interpreter cannot run —
/// returns `Err(reason)` instead of the C++ `false` + out-param, so a fault
/// cannot be mistaken for a computed value.
///
/// # Errors
///
/// Returns the fault reason if the op's operands violate its typing, or if the
/// op is one the Metal host interpreter does not execute.
pub fn eval_op(op: &LaunchOp, package: &LaunchPackage, vals: &mut [Value]) -> Result<()> {
    let a0 = op.args.first().copied().unwrap_or(0) as usize;
    let a1 = op.args.get(1).copied().unwrap_or(0) as usize;
    let a2 = op.args.get(2).copied().unwrap_or(0) as usize;
    let result = op.result_id as usize;
    let code = op.code as u8;

    // Set one result cell.
    macro_rules! out {
        ($x:expr) => {{
            vals[result] = $x;
            return Ok(());
        }};
    }
    macro_rules! fault {
        ($m:expr) => {
            return Err(Error::Program {
                message: $m.to_string(),
            })
        };
    }

    match code {
        tags::EXP => out!(map_f32(&vals[a0], f32::exp)),
        tags::LOG => out!(map_f32(&vals[a0], f32::ln)),
        tags::RECIP => out!(map_f32(&vals[a0], |x| 1.0 / x)),
        tags::NEG => match &vals[a0] {
            Value::F32(v) => out!(Value::F32(v.iter().map(|&e| -e).collect())),
            Value::I32(v) => out!(Value::I32(v.iter().map(|&e| e.wrapping_neg()).collect())),
            Value::U32(v) => out!(Value::U32(v.iter().map(|&e| e.wrapping_neg()).collect())),
            Value::Bool(_) => fault!("neg on bool"),
        },
        tags::ABS => match &vals[a0] {
            Value::F32(v) => out!(Value::F32(v.iter().map(|&e| e.abs()).collect())),
            Value::I32(v) => out!(Value::I32(
                v.iter()
                    .map(|&e| if e == i32::MIN { e } else { e.abs() })
                    .collect()
            )),
            other => out!(other.clone()),
        },
        tags::SIGN => match &vals[a0] {
            Value::F32(v) => out!(Value::F32(
                v.iter()
                    .map(|&e| if e > 0.0 {
                        1.0
                    } else if e < 0.0 {
                        -1.0
                    } else {
                        0.0
                    })
                    .collect()
            )),
            Value::I32(v) => out!(Value::I32(v.iter().map(|&e| e.signum()).collect())),
            Value::U32(v) => out!(Value::U32(v.iter().map(|&e| u32::from(e != 0)).collect())),
            Value::Bool(_) => fault!("sign on bool"),
        },
        tags::CAST => {
            let want = concrete_dtype(op.dtype);
            let x = &vals[a0];
            match want {
                DType::F32 => out!(Value::F32(x.lanes_f32())),
                DType::I32 => {
                    if let Value::F32(f) = x {
                        out!(Value::I32(f.iter().map(|&e| e as i32).collect()));
                    }
                    out!(Value::from_i64(DType::I32, &x.lanes_i64()));
                }
                DType::U32 => {
                    if let Value::F32(f) = x {
                        out!(Value::U32(f.iter().map(|&e| e as u32).collect()));
                    }
                    out!(Value::from_i64(DType::U32, &x.lanes_i64()));
                }
                DType::Bool => {
                    out!(Value::Bool(
                        x.lanes_f32().iter().map(|&e| u8::from(e != 0.0)).collect()
                    ));
                }
            }
        }

        tags::ADD => out!(bin_arith(
            &vals[a0],
            &vals[a1],
            ty_dtype(package, a0),
            |x, y| x + y,
            i64::wrapping_add
        )),
        tags::SUB => out!(bin_arith(
            &vals[a0],
            &vals[a1],
            ty_dtype(package, a0),
            |x, y| x - y,
            i64::wrapping_sub
        )),
        tags::MUL => out!(bin_arith(
            &vals[a0],
            &vals[a1],
            ty_dtype(package, a0),
            |x, y| x * y,
            i64::wrapping_mul
        )),
        tags::DIV => out!(bin_arith(
            &vals[a0],
            &vals[a1],
            ty_dtype(package, a0),
            |x, y| x / y,
            |x, y| if y == 0 { 0 } else { x.wrapping_div(y) }
        )),
        tags::REM => out!(bin_arith(
            &vals[a0],
            &vals[a1],
            ty_dtype(package, a0),
            |x, y| x % y,
            |x, y| if y == 0 { 0 } else { x.wrapping_rem(y) }
        )),
        tags::MAX_ELEM => out!(bin_arith(
            &vals[a0],
            &vals[a1],
            ty_dtype(package, a0),
            |x, y| x.max(y),
            |x, y| x.max(y)
        )),
        tags::MIN_ELEM => out!(bin_arith(
            &vals[a0],
            &vals[a1],
            ty_dtype(package, a0),
            |x, y| x.min(y),
            |x, y| x.min(y)
        )),

        tags::GT => out!(cmp_op(
            &vals[a0],
            &vals[a1],
            ty_dtype(package, a0),
            |x, y| x > y,
            |x, y| x > y
        )),
        tags::GE => out!(cmp_op(
            &vals[a0],
            &vals[a1],
            ty_dtype(package, a0),
            |x, y| x >= y,
            |x, y| x >= y
        )),
        tags::EQ => out!(cmp_op(
            &vals[a0],
            &vals[a1],
            ty_dtype(package, a0),
            |x, y| x == y,
            |x, y| x == y
        )),
        tags::NE => out!(cmp_op(
            &vals[a0],
            &vals[a1],
            ty_dtype(package, a0),
            |x, y| x != y,
            |x, y| x != y
        )),
        tags::LT => out!(cmp_op(
            &vals[a0],
            &vals[a1],
            ty_dtype(package, a0),
            |x, y| x < y,
            |x, y| x < y
        )),
        tags::LE => out!(cmp_op(
            &vals[a0],
            &vals[a1],
            ty_dtype(package, a0),
            |x, y| x <= y,
            |x, y| x <= y
        )),
        tags::AND | tags::OR => {
            let (Value::Bool(x), Value::Bool(y)) = (&vals[a0], &vals[a1]) else {
                fault!("and/or on non-bool");
            };
            let is_and = code == tags::AND;
            let n = x.len().max(y.len());
            let o: Vec<u8> = (0..n)
                .map(|i| {
                    let p = x[pick(x.len(), i)] != 0;
                    let q = y[pick(y.len(), i)] != 0;
                    u8::from(if is_and { p && q } else { p || q })
                })
                .collect();
            out!(Value::Bool(o));
        }
        tags::NOT => {
            let Value::Bool(x) = &vals[a0] else {
                fault!("not on non-bool");
            };
            out!(Value::Bool(x.iter().map(|&e| u8::from(e == 0)).collect()));
        }

        tags::SELECT => {
            let Value::Bool(c) = &vals[a0] else {
                fault!("select cond");
            };
            let cb = c.clone();
            let x = &vals[a1];
            let y = &vals[a2];
            let n = cb.len().max(x.len()).max(y.len());
            let sel = |i: usize| cb[pick(cb.len(), i)] != 0;
            let d = ty_dtype(package, a1);
            match d {
                DType::F32 => {
                    let xf = x.lanes_f32();
                    let yf = y.lanes_f32();
                    out!(Value::F32(
                        (0..n)
                            .map(|i| if sel(i) {
                                xf[pick(xf.len(), i)]
                            } else {
                                yf[pick(yf.len(), i)]
                            })
                            .collect()
                    ));
                }
                DType::Bool => {
                    let (Value::Bool(xb), Value::Bool(yb)) = (x, y) else {
                        fault!("select bool arms");
                    };
                    out!(Value::Bool(
                        (0..n)
                            .map(|i| if sel(i) {
                                xb[pick(xb.len(), i)]
                            } else {
                                yb[pick(yb.len(), i)]
                            })
                            .collect()
                    ));
                }
                _ => {
                    let xi = x.lanes_i64();
                    let yi = y.lanes_i64();
                    let o: Vec<i64> = (0..n)
                        .map(|i| {
                            if sel(i) {
                                xi[pick(xi.len(), i)]
                            } else {
                                yi[pick(yi.len(), i)]
                            }
                        })
                        .collect();
                    out!(Value::from_i64(d, &o));
                }
            }
        }

        tags::REDUCE_SUM | tags::REDUCE_MAX | tags::REDUCE_MIN => {
            let dtype = ty_dtype(package, a0);
            let rows = canonical_rows(ty_shape(package, a0));
            let data = &vals[a0];
            let len = data.len().checked_div(rows).unwrap_or(0);
            if dtype == DType::F32 {
                let x = data.lanes_f32();
                let o: Vec<f32> = (0..rows)
                    .map(|r| {
                        let row = &x[r * len..r * len + len];
                        match code {
                            tags::REDUCE_SUM => canonical_reduce(row, 0.0, |l, r| l + r),
                            tags::REDUCE_MAX => {
                                canonical_reduce(row, f32::NEG_INFINITY, canonical_max)
                            }
                            _ => canonical_reduce(row, f32::INFINITY, canonical_min),
                        }
                    })
                    .collect();
                out!(Value::F32(o));
            }
            let x = data.lanes_i64();
            let o: Vec<i64> = (0..rows)
                .map(|r| {
                    let row = &x[r * len..r * len + len];
                    match code {
                        tags::REDUCE_SUM => canonical_reduce(row, 0i64, i64::wrapping_add),
                        tags::REDUCE_MAX => canonical_reduce(row, i64::MIN, |l, r| l.max(r)),
                        _ => canonical_reduce(row, i64::MAX, |l, r| l.min(r)),
                    }
                })
                .collect();
            out!(Value::from_i64(dtype, &o));
        }
        tags::REDUCE_ARGMAX => {
            let dtype = ty_dtype(package, a0);
            let rows = canonical_rows(ty_shape(package, a0));
            let data = &vals[a0];
            let len = data.len().checked_div(rows).unwrap_or(0);
            let o: Vec<i32> = if dtype == DType::F32 {
                let x = data.lanes_f32();
                (0..rows)
                    .map(|r| argmax_row(&x[r * len..r * len + len]))
                    .collect()
            } else {
                let x = data.lanes_i64();
                (0..rows)
                    .map(|r| argmax_row_i64(&x[r * len..r * len + len]))
                    .collect()
            };
            out!(Value::I32(o));
        }
        tags::CUMSUM | tags::CUMPROD => {
            let rows = canonical_rows(ty_shape(package, a0));
            let x = vals[a0].lanes_f32();
            let len = x.len().checked_div(rows).unwrap_or(0);
            let is_sum = code == tags::CUMSUM;
            let mut o = Vec::with_capacity(x.len());
            for r in 0..rows {
                let mut acc = if is_sum { 0.0 } else { 1.0 };
                for j in 0..len {
                    acc = if is_sum {
                        acc + x[r * len + j]
                    } else {
                        acc * x[r * len + j]
                    };
                    o.push(acc);
                }
            }
            out!(Value::F32(o));
        }

        tags::BROADCAST => {
            let src = ty_shape(package, a0).to_vec();
            out!(broadcast_value(&vals[a0], &src, &op.shape));
        }
        tags::RESHAPE => out!(vals[a0].clone()),
        tags::TRANSPOSE => {
            let dims = ty_shape(package, a0);
            if dims.len() != 2 {
                fault!("transpose rank");
            }
            let m = dims[0] as usize;
            let n = dims[1] as usize;
            let idx: Vec<Option<usize>> = (0..m * n).map(|o| Some((o % m) * n + o / m)).collect();
            out!(gather_flat(&vals[a0], &idx));
        }

        tags::SORT_DESC => {
            let x = vals[a0].lanes_f32();
            let order = sort_desc_order(&x);
            let sorted: Vec<f32> = order.iter().map(|&k| x[k as usize]).collect();
            vals[result] = Value::F32(sorted);
            vals[result + 1] = Value::U32(order);
            Ok(())
        }
        tags::TOP_K => {
            let rows = canonical_rows(ty_shape(package, a0));
            let x = vals[a0].lanes_f32();
            let len = x.len().checked_div(rows).unwrap_or(0);
            let k = op.imm as usize;
            let mut vs = Vec::with_capacity(rows * k);
            let mut is = Vec::with_capacity(rows * k);
            for r in 0..rows {
                let order = sort_desc_order(&x[r * len..r * len + len]);
                for &p in order.iter().take(k) {
                    vs.push(x[r * len + p as usize]);
                    is.push(p);
                }
            }
            vals[result] = Value::F32(vs);
            vals[result + 1] = Value::U32(is);
            Ok(())
        }
        tags::MATMUL => {
            let ta = ty_shape(package, a0);
            let tb = ty_shape(package, a1);
            if ta.len() != 2 || tb.len() != 2 {
                fault!("matmul rank");
            }
            let m = ta[0] as usize;
            let kk = ta[1] as usize;
            let n = tb[1] as usize;
            let x = vals[a0].lanes_f32();
            let y = vals[a1].lanes_f32();
            let mut o = vec![0.0f32; m * n];
            for i in 0..m {
                for l in 0..kk {
                    let xv = x[i * kk + l];
                    if xv == 0.0 {
                        continue;
                    }
                    for j in 0..n {
                        o[i * n + j] += xv * y[l * n + j];
                    }
                }
            }
            out!(Value::F32(o));
        }
        tags::PIVOT_THRESHOLD => {
            let rows = canonical_rows(ty_shape(package, a0));
            let x = vals[a0].lanes_f32();
            let len = x.len().checked_div(rows).unwrap_or(0);
            let payload = op.pred_payload as usize;
            let mut keep = vec![0u8; x.len()];
            for r in 0..rows {
                let row = &x[r * len..r * len + len];
                let k = &mut keep[r * len..r * len + len];
                match op.pred_tag {
                    // RankLe: keep the lanes with fewer than `k` strictly-larger
                    // finite lanes.
                    0 => {
                        let kv = vals[payload].lanes_i64();
                        let kk = kv[pick(kv.len(), r)].clamp(0, len as i64);
                        for i in 0..len {
                            if row[i].is_nan() {
                                continue;
                            }
                            let greater =
                                row.iter().filter(|&&y| !y.is_nan() && y > row[i]).count() as i64;
                            k[i] = u8::from(greater < kk);
                        }
                    }
                    // CummassLe: keep the descending prefix whose exclusive mass
                    // is below `p`.
                    1 => {
                        let pv = vals[payload].lanes_f32();
                        let p = pv[pick(pv.len(), r)];
                        let order = sort_desc_order(row);
                        let mut excl = 0.0f32;
                        for i in order {
                            k[i as usize] = u8::from(excl < p);
                            excl += row[i as usize];
                        }
                    }
                    // ProbGe: keep lanes at or above the threshold.
                    _ => {
                        let tv = vals[payload].lanes_f32();
                        let thr = tv[pick(tv.len(), r)];
                        for i in 0..len {
                            k[i] = u8::from(row[i] >= thr);
                        }
                    }
                }
            }
            out!(Value::Bool(keep));
        }

        tags::GATHER => {
            let ts = ty_shape(package, a0);
            let rest: usize = ts
                .iter()
                .skip(1)
                .map(|&d| d as usize)
                .product::<usize>()
                .max(1);
            let n0 = ts.first().copied().unwrap_or(1) as usize;
            let ix = vals[a1].lanes_i64();
            let mut flat = Vec::with_capacity(ix.len() * rest);
            for i in ix {
                if i >= 0 && (i as usize) < n0 {
                    for r in 0..rest {
                        flat.push(Some(i as usize * rest + r));
                    }
                } else {
                    flat.extend(std::iter::repeat_n(None, rest));
                }
            }
            out!(gather_flat(&vals[a0], &flat));
        }
        tags::GATHER_ROW => {
            let ts = ty_shape(package, a0);
            if ts.len() != 2 {
                fault!("gather_row");
            }
            let m = ts[0] as usize;
            let n = ts[1] as usize;
            let ix = vals[a1].lanes_i64();
            let flat: Vec<Option<usize>> = (0..m)
                .map(|i| {
                    let c = ix[i];
                    if c >= 0 && (c as usize) < n {
                        Some(i * n + c as usize)
                    } else {
                        None
                    }
                })
                .collect();
            out!(gather_flat(&vals[a0], &flat));
        }
        tags::SCATTER_ADD | tags::SCATTER_SET => {
            let dtype = ty_dtype(package, a0);
            let ts = ty_shape(package, a0);
            let rest: usize = ts
                .iter()
                .skip(1)
                .map(|&d| d as usize)
                .product::<usize>()
                .max(1);
            let n0 = ts.first().copied().unwrap_or(1) as usize;
            let ix = vals[a1].lanes_i64();
            let is_add = code == tags::SCATTER_ADD;
            let scalar_val = vals[a2].len() == 1 && ix.len() * rest != 1;
            if dtype == DType::F32 || (is_add && dtype != DType::I32 && dtype != DType::U32) {
                let mut outv = vals[a0].lanes_f32();
                let vals_f = vals[a2].lanes_f32();
                for (k, &i) in ix.iter().enumerate() {
                    if i < 0 || i as usize >= n0 {
                        continue;
                    }
                    for r in 0..rest {
                        let src = if scalar_val {
                            vals_f[0]
                        } else {
                            vals_f[k * rest + r]
                        };
                        let dst = &mut outv[i as usize * rest + r];
                        if is_add { *dst += src } else { *dst = src }
                    }
                }
                out!(Value::F32(outv));
            }
            let mut outv = vals[a0].lanes_i64();
            let vals_i = vals[a2].lanes_i64();
            for (k, &i) in ix.iter().enumerate() {
                if i < 0 || i as usize >= n0 {
                    continue;
                }
                for r in 0..rest {
                    let src = if scalar_val {
                        vals_i[0]
                    } else {
                        vals_i[k * rest + r]
                    };
                    let dst = &mut outv[i as usize * rest + r];
                    if is_add { *dst += src } else { *dst = src }
                }
            }
            out!(Value::from_i64(dtype, &outv));
        }
        tags::IOTA => out!(Value::U32((0..op.imm).collect())),
        tags::MASK_APPLY_PACKED => {
            let ls = ty_shape(package, a0);
            let n = ls.last().copied().unwrap_or(1) as usize;
            let x = vals[a0].lanes_f32();
            let Value::U32(mask) = &vals[a1] else {
                fault!("mask_apply mask");
            };
            let o: Vec<f32> = (0..x.len())
                .map(|j| {
                    let c = j % n;
                    let w = c >> 5;
                    let word = mask.get(w).copied().unwrap_or(0);
                    if (word >> (c & 31)) & 1 != 0 {
                        x[j]
                    } else {
                        f32::NEG_INFINITY
                    }
                })
                .collect();
            out!(Value::F32(o));
        }
        tags::CAUSAL_MASK | tags::SLIDING_WINDOW_MASK | tags::SINK_WINDOW_MASK => {
            let Value::U32(positions) = &vals[a0] else {
                fault!("structured mask positions");
            };
            let positions = positions.clone();
            let key_count = op.imm;
            let window = if code == tags::SLIDING_WINDOW_MASK {
                op.imm2
            } else {
                op.imm3
            };
            let mut mask = Vec::with_capacity(positions.len() * key_count as usize);
            for position in positions {
                for key in 0..key_count {
                    let mut allowed = key <= position;
                    if allowed && code != tags::CAUSAL_MASK {
                        let recent = key.saturating_add(window) > position;
                        allowed = if code == tags::SLIDING_WINDOW_MASK {
                            recent
                        } else {
                            key < op.imm2 || recent
                        };
                    }
                    mask.push(u8::from(allowed));
                }
            }
            out!(Value::Bool(mask));
        }
        tags::RNG => {
            let seed_eff = rng::seed_eff_stream(0, op.imm);
            let n = shape_numel(&op.shape) as usize;
            out!(Value::F32(rng_lanes(
                seed_eff,
                n,
                op.rng_kind == RngKind::Gumbel as u8
            )));
        }
        tags::RNG_KEYED => {
            let st = vals[a0].lanes_i64();
            let key = (st[0] as u64 & 0xFFFF_FFFF) as u32;
            let ctr = if st.len() > 1 {
                (st[1] as u64 & 0xFFFF_FFFF) as u32
            } else {
                0
            };
            let seed64 = rng::keyed_seed(key, ctr);
            let n = shape_numel(&op.shape) as usize;
            out!(Value::F32(rng_lanes(
                seed64,
                n,
                op.rng_kind == RngKind::Gumbel as u8
            )));
        }
        tags::KERNEL_CALL => {
            if op.args.len() != 1 {
                fault!("Metal identity boundary arity");
            }
            out!(vals[a0].clone());
        }
        tags::SINK_CALL => Ok(()),

        _ => Err(Error::Program {
            message: format!(
                "op not executable on the Metal host interpreter: {}",
                tensor_ir::op::spec(code).map_or("?", |row| row.name)
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_reduce_uses_a_pairwise_tree_not_a_left_fold() {
        // A big term followed by 33 ones. A left fold adds each one into the
        // running 1e8 and loses every one to rounding (ulp at 1e8 is 8). The
        // pairwise tree sums the ones among themselves first, so it recovers 24
        // of them before the big term absorbs the rest — a different, and
        // reproducible, answer.
        let mut row = vec![1.0e8f32];
        row.extend(std::iter::repeat_n(1.0f32, 33));
        let tree = canonical_reduce(&row, 0.0, |l, r| l + r);
        let left_fold = row.iter().fold(0.0f32, |l, &r| l + r);
        assert_eq!(
            tree, 100_000_024.0,
            "the width-32 pairwise tree must land on exactly 1e8 + 24 here; a \
             different value means the reduction tree shape changed"
        );
        assert_eq!(
            left_fold, 100_000_000.0,
            "a left fold loses every one to rounding — this is the order the \
             canonical tree must NOT be"
        );
        assert_ne!(
            tree, left_fold,
            "if the pairwise tree ever equals the left fold here, the reduction \
             has silently become a left fold and reproducibility is lost"
        );
    }

    #[test]
    fn canonical_max_makes_an_all_nan_row_reduce_to_neg_inf() {
        let row = [f32::NAN, f32::NAN, f32::NAN];
        let reduced = canonical_reduce(&row, f32::NEG_INFINITY, canonical_max);
        assert_eq!(
            reduced,
            f32::NEG_INFINITY,
            "a max over only NaNs must be -inf, not NaN, or it would poison a later argmax"
        );
    }

    #[test]
    fn canonical_max_and_min_pin_the_sign_of_a_zero() {
        assert!(
            canonical_max(-0.0, 0.0).is_sign_positive(),
            "max(-0, +0) must be +0 regardless of order; the zero's sign is observable via 1/x"
        );
        assert!(
            canonical_max(-0.0, -0.0).is_sign_negative(),
            "max(-0, -0) is -0"
        );
        assert!(
            canonical_min(-0.0, 0.0).is_sign_negative(),
            "min(-0, +0) must be -0"
        );
        assert!(
            canonical_min(0.0, 0.0).is_sign_positive(),
            "min(+0, +0) is +0"
        );
    }

    #[test]
    fn canonical_max_never_selects_a_nan_over_a_number() {
        assert_eq!(
            canonical_max(f32::NAN, 2.0),
            2.0,
            "a NaN must lose to a real value"
        );
        assert_eq!(
            canonical_max(2.0, f32::NAN),
            2.0,
            "regardless of operand order"
        );
    }

    #[test]
    fn combine_argmax_breaks_ties_toward_the_lower_index() {
        let low = ArgmaxCandidate {
            value: 5.0,
            index: 1,
            have: true,
        };
        let high = ArgmaxCandidate {
            value: 5.0,
            index: 7,
            have: true,
        };
        assert_eq!(
            combine_argmax(low, high).index,
            1,
            "equal values: the lower index wins"
        );
        assert_eq!(
            combine_argmax(high, low).index,
            1,
            "and the result is order-independent"
        );
    }

    #[test]
    fn argmax_row_ignores_nan_and_answers_zero_for_an_all_nan_row() {
        assert_eq!(
            argmax_row(&[1.0, f32::NAN, 3.0, 2.0]),
            2,
            "the max is lane 2; the NaN is skipped"
        );
        assert_eq!(
            argmax_row(&[f32::NAN, f32::NAN]),
            0,
            "an all-NaN row argmaxes to 0"
        );
        assert_eq!(
            argmax_row(&[4.0, 4.0, 4.0]),
            0,
            "a tie argmaxes to the lowest index"
        );
    }

    #[test]
    fn sort_desc_order_is_descending_lower_index_ties_nan_last() {
        let order = sort_desc_order(&[1.0, 3.0, 3.0, f32::NAN, 2.0]);
        assert_eq!(
            order,
            vec![1, 2, 4, 0, 3],
            "3.0@1 then 3.0@2 (tie -> lower index), 2.0, 1.0, then the NaN last"
        );
    }

    #[test]
    fn broadcast_replicates_along_leading_aligned_axes() {
        // Source [2] -> target [2, 3]: source axis aligns to the LEADING target
        // axis, so each source lane spreads across the 3 trailing lanes.
        let src = Value::F32(vec![10.0, 20.0]);
        let out = broadcast_value(&src, &[2], &[2, 3]);
        assert_eq!(
            out,
            Value::F32(vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0]),
            "left-aligned broadcast spreads lane i across the trailing axis, not NumPy trailing-align"
        );
    }

    #[test]
    fn gather_flat_fills_zero_for_a_none_index() {
        let src = Value::I32(vec![5, 6, 7]);
        let out = gather_flat(&src, &[Some(2), None, Some(0)]);
        assert_eq!(
            out,
            Value::I32(vec![7, 0, 5]),
            "a None index must fill zero, not read out of range"
        );
    }
}

#[cfg(test)]
mod coverage_tests {
    /// **Which ops this interpreter executes, and which it refuses BY NAME.**
    ///
    /// "Is the PTIR port complete" was asked by hand three times in this arc
    /// and got a different answer each time, because it was answered by
    /// grepping for `tags::NAME` and GUESSING the constant's spelling —
    /// `CUM_SUM` where the table says `CUMSUM`. Three ops looked missing and
    /// all three were already implemented.
    ///
    /// `OP_TABLE` is the whole vocabulary and this file is the whole executor,
    /// so the question has an exact answer and no spelling in it: read the
    /// source, take the tags it matches, and subtract.
    ///
    /// Read rather than run because running wants a `LaunchPackage` with a
    /// type table, and building fifty-five of those to ask which ARM a tag
    /// reaches is a lot of scaffolding for a question about a match.
    ///
    /// What it pins is not a NUMBER — a refusal can be legitimate, and the
    /// ones here are: `Const` is a value ROOT rather than an op, and the
    /// channel ops belong to `meta`. What it pins is that every refusal names
    /// its op, which the catch-all does by reading `OP_TABLE`.
    #[test]
    fn every_op_is_executed_here_or_refused_somewhere_that_names_it() {
        let src = include_str!("op.rs");
        // The tags this file's match arms name, from the arms themselves.
        let matched: std::collections::BTreeSet<&str> = src
            .match_indices("tags::")
            .map(|(at, _)| {
                let rest = &src[at + 6..];
                let end = rest
                    .find(|c: char| !c.is_ascii_uppercase() && !c.is_ascii_digit() && c != '_')
                    .unwrap_or(rest.len());
                &rest[..end]
            })
            .collect();

        // The names `OP_TABLE` states, in the spelling the tag constant takes.
        let mut elsewhere = Vec::new();
        for row in tensor_ir::op::OP_TABLE {
            let konst = row.name.to_ascii_uppercase();
            if !matched.contains(konst.as_str()) {
                elsewhere.push(row.name);
            }
        }

        eprintln!(
            "{} of {} ops are executed here; {} live elsewhere: {:?}",
            tensor_ir::op::OP_TABLE.len() - elsewhere.len(),
            tensor_ir::op::OP_TABLE.len(),
            elsewhere.len(),
            elsewhere
        );

        // The catch-all must NAME the op it refuses, which is the property
        // that makes an unservable program diagnosable rather than a failure.
        assert!(
            src.contains("op not executable on the Metal host interpreter")
                && src.contains("tensor_ir::op::spec(code)"),
            "the catch-all must read the refused op's name out of OP_TABLE"
        );
        assert!(
            elsewhere.len() * 4 < tensor_ir::op::OP_TABLE.len(),
            "{} of {} ops are not executed here, which is more than the \
             handful that legitimately live in `meta`, `params` and `step`: \
             {elsewhere:?}",
            elsewhere.len(),
            tensor_ir::op::OP_TABLE.len()
        );
    }
}
