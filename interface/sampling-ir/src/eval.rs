//! Reference CPU interpreter for the shape-typed Sampling IR — **feature `eval`**.
//!
//! GPU-free executable semantics: walk a [`SamplingProgram`]'s flat op list over
//! positionally-bound input values + an ambient RNG seed, returning the declared
//! outputs. The parity oracle the CUDA codegen is checked against.
//!
//! **Scope (v0):** scalar / vector values (M=1 decode). Matrix (rank ≥ 2)
//! per-row evaluation is the L7 (hotel) shape-aware follow-up and returns
//! [`EvalError::Unsupported`]. Inputs are supplied **positionally** (one [`Value`]
//! per [`SamplingProgram::inputs`] slot — binding is attach-time, resolved by the
//! caller).

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::types::*;

/// A runtime value: a flat buffer (length 1 == scalar) tagged by dtype.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    F32(Vec<f32>),
    I32(Vec<i32>),
    U32(Vec<u32>),
    Bool(Vec<bool>),
}

impl Value {
    pub fn len(&self) -> usize {
        match self {
            Value::F32(v) => v.len(),
            Value::I32(v) => v.len(),
            Value::U32(v) => v.len(),
            Value::Bool(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn dtype(&self) -> DType {
        match self {
            Value::F32(_) => DType::F32,
            Value::I32(_) => DType::I32,
            Value::U32(_) => DType::U32,
            Value::Bool(_) => DType::Bool,
        }
    }
    fn to_f32(&self) -> Vec<f32> {
        match self {
            Value::F32(v) => v.clone(),
            Value::I32(v) => v.iter().map(|&x| x as f32).collect(),
            Value::U32(v) => v.iter().map(|&x| x as f32).collect(),
            Value::Bool(v) => v.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect(),
        }
    }
    fn as_bool(&self) -> Result<Vec<bool>, EvalError> {
        match self {
            Value::Bool(v) => Ok(v.clone()),
            _ => Err(EvalError::Type("expected Bool".into())),
        }
    }
    fn int_lanes(&self) -> Result<Vec<i64>, EvalError> {
        match self {
            Value::I32(v) => Ok(v.iter().map(|&x| x as i64).collect()),
            Value::U32(v) => Ok(v.iter().map(|&x| x as i64).collect()),
            _ => Err(EvalError::Type("expected integer indices".into())),
        }
    }
}

/// Errors from [`eval`].
#[derive(Clone, Debug, PartialEq)]
pub enum EvalError {
    BadValueId(ValueId),
    /// `Op::Input(index)` referenced a slot with no supplied value.
    MissingInput(InputIndex),
    Type(String),
    Shape(String),
    /// A feature outside the v0 scope (matrix per-row, etc.).
    Unsupported(&'static str),
}

/// Inputs bound at evaluation time: one [`Value`] per program input slot, plus
/// the ambient RNG seed `S` (the runtime's per-row `sample_seed`).
pub struct InputBindings<'a> {
    /// `inputs[i]` is the value bound to slot `i` (logits / tensor data;
    /// binding is attach-time, resolved by the caller).
    pub inputs: &'a [Value],
    pub seed: u32,
}

impl<'a> InputBindings<'a> {
    pub fn new(inputs: &'a [Value], seed: u32) -> Self {
        Self { inputs, seed }
    }
}

/// Evaluate a program's flat op list, returning one [`Value`] per declared
/// output (in `outputs` order).
pub fn eval(prog: &SamplingProgram, inputs: &InputBindings) -> Result<Vec<Value>, EvalError> {
    // Per-value [`ValueType`]s drive the shape-aware (matrix / per-row) arms:
    // reductions/scans/argmax run per-row over the last axis for rank ≥ 2,
    // `Broadcast` replicates to a target shape, `GatherRow`/per-row
    // `PivotThreshold` index by row. (Scalar/vector ⇒ `rows == 1` ⇒ the original
    // semantics.)
    let types = crate::value_types(prog)
        .map_err(|_| EvalError::Shape("program failed shape validation".into()))?;
    let mut vals: Vec<Value> = Vec::with_capacity(prog.ops.len());
    for op in &prog.ops {
        match eval_op(op, &vals, &types, inputs)? {
            OpResult::One(v) => vals.push(v),
            OpResult::Two(a, b) => {
                vals.push(a);
                vals.push(b);
            }
        }
    }
    prog.outputs.iter().map(|o| get(&vals, o.value).cloned()).collect()
}

/// The [`Shape`] of value `id` from the validated type table.
fn shape_of(types: &[ValueType], id: ValueId) -> Result<Shape, EvalError> {
    types.get(id as usize).map(|t| t.shape).ok_or(EvalError::BadValueId(id))
}

enum OpResult {
    One(Value),
    Two(Value, Value),
}

fn get(vals: &[Value], id: ValueId) -> Result<&Value, EvalError> {
    vals.get(id as usize).ok_or(EvalError::BadValueId(id))
}

fn literal_value(lit: Literal) -> Value {
    match lit {
        Literal::F32(x) => Value::F32(vec![x]),
        Literal::I32(x) => Value::I32(vec![x]),
        Literal::U32(x) => Value::U32(vec![x]),
        Literal::Bool(x) => Value::Bool(vec![x]),
    }
}

fn eval_op(
    op: &Op,
    vals: &[Value],
    types: &[ValueType],
    inputs: &InputBindings,
) -> Result<OpResult, EvalError> {
    use Op::*;
    let one = |v: Value| Ok(OpResult::One(v));
    match *op {
        // ── leaves ─────────────────────────────────────────────────────────
        Input(index) => inputs
            .inputs
            .get(index as usize)
            .cloned()
            .map(OpResult::One)
            .ok_or(EvalError::MissingInput(index)),
        Const(lit) => one(literal_value(lit)),

        // ── unary ──────────────────────────────────────────────────────────
        Exp(a) => one(map_f32(get(vals, a)?, |x| x.exp())),
        Log(a) => one(map_f32(get(vals, a)?, |x| x.ln())),
        Neg(a) => one(map_f32(get(vals, a)?, |x| -x)),
        Recip(a) => one(map_f32(get(vals, a)?, |x| 1.0 / x)),
        Abs(a) => one(map_f32(get(vals, a)?, |x| x.abs())),
        Sign(a) => one(map_f32(get(vals, a)?, |x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        })),

        // ── binary ─────────────────────────────────────────────────────────
        Add(a, b) => one(zip_f32(get(vals, a)?, get(vals, b)?, |x, y| x + y)?),
        Sub(a, b) => one(zip_f32(get(vals, a)?, get(vals, b)?, |x, y| x - y)?),
        Mul(a, b) => one(zip_f32(get(vals, a)?, get(vals, b)?, |x, y| x * y)?),
        Div(a, b) => one(zip_f32(get(vals, a)?, get(vals, b)?, |x, y| x / y)?),
        MaxElem(a, b) => one(zip_f32(get(vals, a)?, get(vals, b)?, f32::max)?),
        MinElem(a, b) => one(zip_f32(get(vals, a)?, get(vals, b)?, f32::min)?),
        Gt(a, b) => one(cmp(get(vals, a)?, get(vals, b)?, |x, y| x > y)?),
        Ge(a, b) => one(cmp(get(vals, a)?, get(vals, b)?, |x, y| x >= y)?),
        Eq(a, b) => one(cmp(get(vals, a)?, get(vals, b)?, |x, y| x == y)?),
        Select { cond, a, b } => one(select(get(vals, cond)?, get(vals, a)?, get(vals, b)?)?),

        // ── reductions (per-row over the last axis for rank ≥ 2) ───────────
        ReduceSum(a) => one(reduce_rows(get(vals, a)?, shape_of(types, a)?, 0.0, |x, y| x + y)),
        ReduceMax(a) => {
            one(reduce_rows(get(vals, a)?, shape_of(types, a)?, f32::NEG_INFINITY, f32::max))
        }
        ReduceMin(a) => one(reduce_rows(get(vals, a)?, shape_of(types, a)?, f32::INFINITY, f32::min)),
        ReduceArgmax(a) => one(argmax_rows(get(vals, a)?, shape_of(types, a)?)),
        Broadcast { value, shape } => {
            one(broadcast_to(get(vals, value)?, shape_of(types, value)?, shape)?)
        }

        // ── scans (per-row over the last axis for rank ≥ 2) ────────────────
        CumSum(a) => one(scan_rows(get(vals, a)?, shape_of(types, a)?, |acc, x| acc + x, 0.0)),
        CumProd(a) => one(scan_rows(get(vals, a)?, shape_of(types, a)?, |acc, x| acc * x, 1.0)),

        // ── sort / threshold ───────────────────────────────────────────────
        SortDesc(a) => {
            let (sorted, idx) = sort_desc(&get(vals, a)?.to_f32());
            Ok(OpResult::Two(Value::F32(sorted), Value::U32(idx)))
        }
        PivotThreshold { input, predicate } => one(Value::Bool(pivot_threshold(
            get(vals, input)?,
            shape_of(types, input)?,
            predicate,
            vals,
        )?)),

        // ── indexing ───────────────────────────────────────────────────────
        Gather { src, idx } => one(gather(get(vals, src)?, get(vals, idx)?)?),
        GatherRow { src, idx } => {
            one(gather_row(get(vals, src)?, shape_of(types, src)?, get(vals, idx)?)?)
        }
        MaskApply { logits, mask } => one(mask_apply(get(vals, logits)?, get(vals, mask)?)?),
        ScatterAdd { base, idx, vals: v } => {
            one(scatter(get(vals, base)?, get(vals, idx)?, get(vals, v)?, true)?)
        }
        ScatterSet { base, idx, vals: v } => {
            one(scatter(get(vals, base)?, get(vals, idx)?, get(vals, v)?, false)?)
        }

        // ── rng ────────────────────────────────────────────────────────────
        Rng { stream, shape, kind } => {
            // Matrix RNG flattens to `row*ncols + col` = the column index `j`
            // over `0..numel` (= charlie's matrix codegen flattening).
            let len = shape.numel() as usize;
            one(Value::F32(rng(inputs.seed, stream, kind, len)))
        }
    }
}

// ── elementwise helpers ────────────────────────────────────────────────────

fn map_f32(v: &Value, f: impl Fn(f32) -> f32) -> Value {
    Value::F32(v.to_f32().iter().map(|&x| f(x)).collect())
}

fn out_len(la: usize, lb: usize) -> Result<usize, EvalError> {
    if la == lb {
        Ok(la)
    } else if la == 1 {
        Ok(lb)
    } else if lb == 1 {
        Ok(la)
    } else {
        Err(EvalError::Shape(format!("non-broadcastable lengths {la} vs {lb}")))
    }
}

fn zip_f32(a: &Value, b: &Value, f: impl Fn(f32, f32) -> f32) -> Result<Value, EvalError> {
    let (av, bv) = (a.to_f32(), b.to_f32());
    let n = out_len(av.len(), bv.len())?;
    let pick = |l: usize, i: usize| if l == 1 { 0 } else { i };
    Ok(Value::F32((0..n).map(|i| f(av[pick(av.len(), i)], bv[pick(bv.len(), i)])).collect()))
}

fn cmp(a: &Value, b: &Value, f: impl Fn(f32, f32) -> bool) -> Result<Value, EvalError> {
    let (av, bv) = (a.to_f32(), b.to_f32());
    let n = out_len(av.len(), bv.len())?;
    let pick = |l: usize, i: usize| if l == 1 { 0 } else { i };
    Ok(Value::Bool((0..n).map(|i| f(av[pick(av.len(), i)], bv[pick(bv.len(), i)])).collect()))
}

fn select(cond: &Value, a: &Value, b: &Value) -> Result<Value, EvalError> {
    let c = cond.as_bool()?;
    let n = c.len().max(a.len()).max(b.len());
    for (name, l) in [("cond", c.len()), ("a", a.len()), ("b", b.len())] {
        if l != 1 && l != n {
            return Err(EvalError::Shape(format!("select {name} len {l} vs {n}")));
        }
    }
    let pick = |l: usize, i: usize| if l == 1 { 0 } else { i };
    let sel = |i: usize| c[pick(c.len(), i)];
    match (a, b) {
        (Value::Bool(av), Value::Bool(bv)) => Ok(Value::Bool(
            (0..n).map(|i| if sel(i) { av[pick(av.len(), i)] } else { bv[pick(bv.len(), i)] }).collect(),
        )),
        (Value::I32(_), Value::I32(_)) | (Value::U32(_), Value::U32(_)) => {
            let (av, bv) = (a.int_lanes()?, b.int_lanes()?);
            let out: Vec<i64> =
                (0..n).map(|i| if sel(i) { av[pick(av.len(), i)] } else { bv[pick(bv.len(), i)] }).collect();
            Ok(if matches!(a, Value::U32(_)) {
                Value::U32(out.iter().map(|&x| x as u32).collect())
            } else {
                Value::I32(out.iter().map(|&x| x as i32).collect())
            })
        }
        _ => {
            let (av, bv) = (a.to_f32(), b.to_f32());
            Ok(Value::F32(
                (0..n).map(|i| if sel(i) { av[pick(av.len(), i)] } else { bv[pick(bv.len(), i)] }).collect(),
            ))
        }
    }
}

fn argmax(v: &[f32]) -> i32 {
    let mut best = f32::NEG_INFINITY;
    let mut bi = 0i32;
    for (j, &x) in v.iter().enumerate() {
        if x > best {
            best = x;
            bi = j as i32;
        }
    }
    bi
}

/// Shape-directed broadcast (the [`Op::Broadcast`] rule): `src`'s dims are
/// **left-aligned** against `target` (trailing axes padded with `1`); each axis
/// must equal the target or be `1` (replicated). Folds scalar-broadcast
/// (`[] → [..]`) and per-row broadcast (`[m] → [m, n]`, i.e. `[m]` viewed as
/// `[m, 1]`). Matches `validate::can_broadcast_to`. **Preserves `value`'s dtype**
/// (replicates F32/I32/U32/Bool — e.g. mirostat's broadcast token id).
fn broadcast_to(value: &Value, src_shape: Shape, target: Shape) -> Result<Value, EvalError> {
    let r = target.rank();
    let td = target.dims();
    let sd = src_shape.dims();
    let sdim = |i: usize| if i < sd.len() { sd[i] } else { 1u32 };
    for i in 0..r {
        let s = sdim(i);
        if s != td[i] && s != 1 {
            return Err(EvalError::Shape(format!(
                "broadcast {sd:?} → {td:?}: axis {i} {s} vs {}",
                td[i]
            )));
        }
    }
    // Row-major strides over the (left-aligned) source extents.
    let mut sstride = vec![1u64; r];
    for i in (0..r.saturating_sub(1)).rev() {
        sstride[i] = sstride[i + 1] * sdim(i + 1) as u64;
    }
    let n = target.numel() as usize;
    // For each output element, the source flat index (size-1 axes index 0).
    let src_idx: Vec<usize> = (0..n as u64)
        .map(|lin| {
            let mut rem = lin;
            let mut sidx = 0u64;
            for i in 0..r {
                let stride: u64 = td[i + 1..].iter().map(|&d| d as u64).product();
                let coord = rem / stride.max(1);
                rem %= stride.max(1);
                if sdim(i) != 1 {
                    sidx += coord * sstride[i];
                }
            }
            sidx as usize
        })
        .collect();
    Ok(match value {
        Value::F32(s) => Value::F32(src_idx.iter().map(|&k| s[k]).collect()),
        Value::I32(s) => Value::I32(src_idx.iter().map(|&k| s[k]).collect()),
        Value::U32(s) => Value::U32(src_idx.iter().map(|&k| s[k]).collect()),
        Value::Bool(s) => Value::Bool(src_idx.iter().map(|&k| s[k]).collect()),
    })
}

/// Per-row reduction over the last axis (`rows = shape.rows()`); rank ≤ 1 ⇒ one
/// row (whole-vector reduce → scalar), preserving the original semantics.
fn reduce_rows(v: &Value, shape: Shape, init: f32, f: impl Fn(f32, f32) -> f32) -> Value {
    let data = v.to_f32();
    let rows = shape.rows() as usize;
    let len = if rows == 0 { 0 } else { data.len() / rows };
    Value::F32(
        (0..rows).map(|r| data[r * len..(r + 1) * len].iter().fold(init, |a, &x| f(a, x))).collect(),
    )
}

/// Per-row argmax over the last axis → `I32` token per row (rank ≤ 1 ⇒ one).
fn argmax_rows(v: &Value, shape: Shape) -> Value {
    let data = v.to_f32();
    let rows = shape.rows() as usize;
    let len = if rows == 0 { 0 } else { data.len() / rows };
    Value::I32((0..rows).map(|r| argmax(&data[r * len..(r + 1) * len])).collect())
}

/// Per-row scan over the last axis (each row scanned independently; rank ≤ 1 ⇒
/// one whole-vector scan).
fn scan_rows(v: &Value, shape: Shape, f: impl Fn(f32, f32) -> f32, init: f32) -> Value {
    let data = v.to_f32();
    let rows = shape.rows() as usize;
    let len = if rows == 0 { 0 } else { data.len() / rows };
    let mut out = Vec::with_capacity(data.len());
    for r in 0..rows {
        let mut acc = init;
        for &x in &data[r * len..(r + 1) * len] {
            acc = f(acc, x);
            out.push(acc);
        }
    }
    Value::F32(out)
}

/// Per-row column pick `out[i] = src[i, idx[i]]` ([`Op::GatherRow`]). `src` is
/// `[m, n]`, `idx` is `[m]`; result `[m]` of `src`'s dtype. Invalid column
/// (`< 0` or `>= n`) → `0` (fill-0). The lossless accept-ratio `p[i, draft[i]]`.
fn gather_row(src: &Value, src_shape: Shape, idx: &Value) -> Result<Value, EvalError> {
    let n = src_shape.last_len().unwrap_or(1) as usize;
    let rows = src_shape.rows() as usize;
    let ix = idx.int_lanes()?;
    if ix.len() != rows {
        return Err(EvalError::Shape(format!("gather_row idx len {} vs rows {rows}", ix.len())));
    }
    let pick = |i: usize| -> Option<usize> {
        let c = ix[i];
        (c >= 0 && (c as usize) < n).then_some(i * n + c as usize)
    };
    Ok(match src {
        Value::I32(s) => Value::I32((0..rows).map(|i| pick(i).map_or(0, |k| s[k])).collect()),
        Value::U32(s) => Value::U32((0..rows).map(|i| pick(i).map_or(0, |k| s[k])).collect()),
        _ => {
            let s = src.to_f32();
            Value::F32((0..rows).map(|i| pick(i).map_or(0.0, |k| s[k])).collect())
        }
    })
}

fn sort_desc(v: &[f32]) -> (Vec<f32>, Vec<u32>) {
    let mut idx: Vec<u32> = (0..v.len() as u32).collect();
    idx.sort_by(|&a, &b| {
        v[b as usize].partial_cmp(&v[a as usize]).unwrap_or(core::cmp::Ordering::Equal).then(a.cmp(&b))
    });
    let sorted = idx.iter().map(|&i| v[i as usize]).collect();
    (sorted, idx)
}

/// Sort-free top-k / top-p / min-p mask in original index order, **per row** for
/// rank ≥ 2. The `CummassLe`/`ProbGe` threshold operand may be a scalar (shared)
/// or a per-row `[rows]` vector (alpha §5a.1 — the per-row pivot, now accepted by
/// the validator). Rank ≤ 1 ⇒ one row (the vector case).
fn pivot_threshold(
    input: &Value,
    shape: Shape,
    predicate: Predicate,
    vals: &[Value],
) -> Result<Vec<bool>, EvalError> {
    let x = input.to_f32();
    let rows = shape.rows() as usize;
    let len = if rows == 0 { 0 } else { x.len() / rows };
    let row_scalar = |id: ValueId, r: usize| -> Result<f32, EvalError> {
        let v = get(vals, id)?.to_f32();
        if v.is_empty() {
            return Err(EvalError::Shape("pivot threshold operand empty".into()));
        }
        Ok(if v.len() == rows { v[r] } else { v[0] })
    };
    let row_int = |id: ValueId, r: usize| -> Result<i64, EvalError> {
        let v = get(vals, id)?.int_lanes()?;
        if v.is_empty() {
            return Err(EvalError::Shape("pivot threshold k operand empty".into()));
        }
        Ok(if v.len() == rows { v[r] } else { v[0] })
    };
    let mut keep = vec![false; x.len()];
    for r in 0..rows {
        let row = &x[r * len..(r + 1) * len];
        let k = &mut keep[r * len..(r + 1) * len];
        match predicate {
            Predicate::RankLe(k_id) => {
                let kk = row_int(k_id, r)?.clamp(0, len as i64) as usize;
                let (_, order) = sort_desc(row);
                for &i in order.iter().take(kk) {
                    k[i as usize] = true;
                }
            }
            Predicate::CummassLe(p_id) => {
                let p = row_scalar(p_id, r)?;
                let (sorted, order) = sort_desc(row);
                let mut excl = 0.0f32;
                for (rank, &i) in order.iter().enumerate() {
                    k[i as usize] = excl < p; // inclusive nucleus
                    excl += sorted[rank];
                }
            }
            Predicate::ProbGe(thr_id) => {
                let thr = row_scalar(thr_id, r)?;
                for (i, &xi) in row.iter().enumerate() {
                    k[i] = xi >= thr;
                }
            }
        }
    }
    Ok(keep)
}

fn gather(src: &Value, idx: &Value) -> Result<Value, EvalError> {
    let s = src.to_f32();
    let ix = idx.int_lanes()?;
    Ok(Value::F32(
        ix.iter().map(|&i| if i >= 0 && (i as usize) < s.len() { s[i as usize] } else { 0.0 }).collect(),
    ))
}

/// Apply a packed allowed-token bitmask ([`Op::MaskApply`]): `out[j] =
/// bit_j(mask) ? logits[j] : −∞`, `bit_j = (mask[j>>5] >> (j&31)) & 1` (bit 1 =
/// allowed → pass-through, bit 0 = disallowed → `−∞`). `mask` is a packed
/// `[ceil(n/32)]` U32 vector; tail bits ≥ `n` are not read.
fn mask_apply(logits: &Value, mask: &Value) -> Result<Value, EvalError> {
    let l = logits.to_f32();
    let words = match mask {
        Value::U32(w) => w,
        _ => return Err(EvalError::Type("mask-apply: mask must be U32".into())),
    };
    Ok(Value::F32(
        l.iter()
            .enumerate()
            .map(|(j, &x)| {
                let bit = words.get(j >> 5).map_or(0, |&w| (w >> (j & 31)) & 1);
                if bit == 1 { x } else { f32::NEG_INFINITY }
            })
            .collect(),
    ))
}

fn scatter(base: &Value, idx: &Value, vals: &Value, add: bool) -> Result<Value, EvalError> {
    let mut out = base.to_f32();
    let ix = idx.int_lanes()?;
    let vv = vals.to_f32();
    if vv.len() != ix.len() && vv.len() != 1 {
        return Err(EvalError::Shape(format!("scatter vals {} vs idx {}", vv.len(), ix.len())));
    }
    for (k, &i) in ix.iter().enumerate() {
        if i >= 0 && (i as usize) < out.len() {
            let val = vv[if vv.len() == 1 { 0 } else { k }];
            if add {
                out[i as usize] += val;
            } else {
                out[i as usize] = val;
            }
        }
    }
    Ok(Value::F32(out))
}

// ── rng: ambient seed + static stream (matches charlie's codegen) ───────────

fn splitmix64(mut x: u64) -> u64 {
    x ^= x >> 27;
    x = x.wrapping_mul(0x3C79_AC49_2BA7_B653);
    x ^= x >> 33;
    x = x.wrapping_mul(0x1C69_B3F7_4AC4_AE35);
    x ^= x >> 27;
    x
}

/// `seed_eff(S, stream) = (S ^ 0xA5A5A5A5) ^ splitmix64(stream * golden)`.
/// `stream = 0` ⇒ `S ^ 0xA5A5A5A5` exactly (today's `seed_eff`, bit-parity).
fn seed_eff_stream(seed: u32, stream: u32) -> u64 {
    let salt = splitmix64((stream as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    ((seed as u64) ^ 0xA5A5_A5A5u64) ^ salt
}

fn hash_uniform(seed_eff: u64, j: u32) -> f32 {
    let x = seed_eff.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul((j as u64) + 1));
    let bits = (splitmix64(x) >> 40) as u32;
    (bits as f32 + 0.5) * (1.0 / 16_777_216.0)
}

fn rng(seed: u32, stream: u32, kind: RngKind, len: usize) -> Vec<f32> {
    let seed_eff = seed_eff_stream(seed, stream);
    (0..len as u32)
        .map(|j| {
            let u = hash_uniform(seed_eff, j);
            match kind {
                RngKind::Uniform => u,
                RngKind::Gumbel => -((-(u.ln())).ln()),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bind(inputs: &[Value]) -> InputBindings<'_> {
        InputBindings::new(inputs, 12345)
    }

    #[test]
    fn argmax_program() {
        let p = SamplingProgram {
            inputs: vec![InputDecl::new(Shape::vector(4), DType::F32)],
            ops: vec![Op::Input(0), Op::ReduceArgmax(0)],
            outputs: vec![OutputDecl::new(1, OutputKind::Token)],
        };
        let out = eval(&p, &bind(&[Value::F32(vec![0.1, 0.9, 0.3, 0.2])])).unwrap();
        assert_eq!(out, vec![Value::I32(vec![1])]);
    }

    #[test]
    fn broadcast_preserves_dtype() {
        // i32 scalar broadcast to [3] stays I32 (the bug foxtrot caught).
        let p = SamplingProgram {
            inputs: vec![],
            ops: vec![Op::Const(Literal::I32(5)), Op::Broadcast { value: 0, shape: Shape::vector(3) }],
            outputs: vec![OutputDecl::new(1, OutputKind::Token)],
        };
        let out = eval(&p, &bind(&[])).unwrap();
        assert_eq!(out, vec![Value::I32(vec![5, 5, 5])]);
    }

    #[test]
    fn scatter_skips_sentinel() {
        // base[1] += 5 (token 1), the -1 lane skipped.
        let p = SamplingProgram {
            inputs: vec![
                InputDecl::new(Shape::vector(4), DType::F32),
                InputDecl::new(Shape::vector(2), DType::I32),
                InputDecl::new(Shape::vector(2), DType::F32),
            ],
            ops: vec![Op::Input(0), Op::Input(1), Op::Input(2), Op::ScatterAdd { base: 0, idx: 1, vals: 2 }, Op::ReduceArgmax(3)],
            outputs: vec![OutputDecl::new(4, OutputKind::Token)],
        };
        let out = eval(
            &p,
            &bind(&[
                Value::F32(vec![0.0, 0.0, 0.0, 0.0]),
                Value::I32(vec![1, -1]),
                Value::F32(vec![5.0, 9.0]),
            ]),
        )
        .unwrap();
        assert_eq!(out, vec![Value::I32(vec![1])]);
    }

    #[test]
    fn rng_stream0_matches_legacy_seed_eff() {
        assert_eq!(seed_eff_stream(0, 0), 0xA5A5_A5A5);
        assert_eq!(seed_eff_stream(12345, 0), (12345u64) ^ 0xA5A5_A5A5);
        assert_ne!(seed_eff_stream(12345, 0), seed_eff_stream(12345, 1));
    }

    #[test]
    fn gumbel_argmax_program() {
        let p = SamplingProgram {
            inputs: vec![InputDecl::new(Shape::vector(5), DType::F32)],
            ops: vec![
                Op::Input(0),
                Op::Rng { stream: 0, shape: Shape::vector(5), kind: RngKind::Gumbel },
                Op::Add(0, 1),
                Op::ReduceArgmax(2),
            ],
            outputs: vec![OutputDecl::new(3, OutputKind::Token)],
        };
        let out = eval(&p, &bind(&[Value::F32(vec![0.0; 5])])).unwrap();
        match &out[0] {
            Value::I32(v) => assert!((0..5).contains(&v[0])),
            _ => panic!("expected token"),
        }
    }

    #[test]
    fn matrix_reduce_then_broadcast_per_row() {
        // Per-row max over [2,4] → [2], broadcast back to [2,4] (each row filled
        // with its own max). Exercises the rank≥2 reduce + matrix Broadcast arms.
        let p = SamplingProgram {
            inputs: vec![InputDecl::new(Shape::matrix(2, 4), DType::F32)],
            ops: vec![Op::Input(0), Op::ReduceMax(0), Op::Broadcast { value: 1, shape: Shape::matrix(2, 4) }],
            outputs: vec![OutputDecl::new(2, OutputKind::Logits)],
        };
        let logits = [1.0, 9.0, 2.0, 3.0, /* row0 max 9 */ 0.5, 0.1, 0.4, 0.2 /* row1 max 0.5 */];
        let out = eval(&p, &bind(&[Value::F32(logits.to_vec())])).expect("matrix eval");
        assert_eq!(out, vec![Value::F32(vec![9.0, 9.0, 9.0, 9.0, 0.5, 0.5, 0.5, 0.5])]);
    }
}
