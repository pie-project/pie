use eta_compiler::codegen::launch::{LaunchOp, LaunchPackage};
use eta_ir::op::tags;
use eta_ir::{Dtype, RngKind, rng};

use super::value::{Value, pick};
use crate::{Error, Result, shape_numel};

#[must_use]
pub fn canonical_rows(dims: &[u32]) -> usize {
    if dims.len() < 2 {
        return 1;
    }
    dims[..dims.len() - 1].iter().map(|&d| d as usize).product()
}

pub fn canonical_reduce<T: Copy>(row: &[T], identity: T, combine: impl Fn(T, T) -> T) -> T {
    canonical_reduce_by(row.len(), identity, |j| row[j], combine)
}

pub fn canonical_reduce_by<T: Copy>(
    len: usize,
    identity: T,
    at: impl Fn(usize) -> T,
    combine: impl Fn(T, T) -> T,
) -> T {
    if len == 0 {
        return identity;
    }
    if len == 1 {
        return at(0);
    }
    let fold = |lanes: &mut [T; 32]| {
        for offset in [16usize, 8, 4, 2, 1] {
            for lane in 0..offset {
                lanes[lane] = combine(lanes[lane], lanes[lane + offset]);
            }
        }
    };
    let mut level: Vec<T> = Vec::with_capacity(len.div_ceil(32));
    let mut base = 0;
    while base < len {
        let mut lanes = [identity; 32];
        let count = 32.min(len - base);
        for (lane, slot) in lanes[..count].iter_mut().enumerate() {
            *slot = at(base + lane);
        }
        fold(&mut lanes);
        level.push(lanes[0]);
        base += 32;
    }
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(32));
        let mut base = 0;
        while base < level.len() {
            let mut lanes = [identity; 32];
            let count = 32.min(level.len() - base);
            lanes[..count].copy_from_slice(&level[base..base + count]);
            fold(&mut lanes);
            next.push(lanes[0]);
            base += 32;
        }
        level = next;
    }
    level[0]
}

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

#[derive(Clone, Copy, Debug, Default)]
pub struct IntArgmaxCandidate {
    pub value: i64,

    pub index: u32,

    pub have: bool,
}

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

#[must_use]
pub fn argmax_row(row: &[f32]) -> i32 {
    let mut best = f32::NEG_INFINITY;
    let mut at = 0u32;
    let mut have = false;
    for (j, &value) in row.iter().enumerate() {
        if value.is_nan() {
            continue;
        }
        if !have || value > best {
            best = value;
            at = j as u32;
            have = true;
        }
    }
    at as i32
}

#[must_use]
pub fn argmax_row_i64(row: &[i64]) -> i32 {
    canonical_reduce_by(
        row.len(),
        IntArgmaxCandidate::default(),
        |j| IntArgmaxCandidate {
            value: row[j],
            index: j as u32,
            have: true,
        },
        combine_int_argmax,
    )
    .index as i32
}

#[must_use]
pub fn sort_desc_order(row: &[f32]) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..row.len() as u32).collect();
    idx.sort_by(|&a, &b| {
        let x = row[a as usize];
        let y = row[b as usize];
        let nx = x.is_nan();
        let ny = y.is_nan();
        match (nx, ny) {
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,

            (true, true) => a.cmp(&b),
            (false, false) => {
                if x == y {
                    a.cmp(&b)
                } else {
                    y.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)
                }
            }
        }
    });
    idx
}

fn nucleus_prefix(row: &[f32], p: f32, keep: &mut [u8]) -> bool {
    if p.is_nan() {
        return false;
    }
    if 0.0 >= p {
        return true;
    }
    const WIDE: usize = 64;
    const GIVE_UP: usize = 32 * WIDE;
    if WIDE >= row.len() {
        return false;
    }
    let mut best: Vec<u32> = Vec::with_capacity(WIDE + 1);
    let mut inserts = 0usize;
    for (i, &v) in row.iter().enumerate() {
        if v.is_nan() || v < 0.0 {
            return false;
        }
        let i = i as u32;
        if best.len() == WIDE && desc_by_value(row, i, best[WIDE - 1]) != std::cmp::Ordering::Less {
            continue;
        }
        let at = best.partition_point(|&b| desc_by_value(row, b, i) == std::cmp::Ordering::Less);
        best.insert(at, i);
        best.truncate(WIDE);
        inserts += 1;
        if inserts > GIVE_UP {
            return false;
        }
    }
    let mut excl = 0.0f32;
    for &i in &best {
        if excl >= p {
            return true;
        }
        keep[i as usize] = 1;
        excl += row[i as usize];
    }
    if excl >= p {
        return true;
    }
    for &i in &best {
        keep[i as usize] = 0;
    }
    false
}

fn desc_by_value(row: &[f32], a: u32, b: u32) -> std::cmp::Ordering {
    let (x, y) = (row[a as usize], row[b as usize]);
    match (x.is_nan(), y.is_nan()) {
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (true, true) => a.cmp(&b),
        (false, false) => {
            if x == y {
                a.cmp(&b)
            } else {
                y.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)
            }
        }
    }
}

#[must_use]
pub fn rng_lanes(seed_eff: u64, n: usize, kind: RngKind) -> Vec<f32> {
    const SPLIT_ABOVE: usize = 1 << 15;
    let lane = |j: usize| match kind {
        RngKind::Uniform => rng::hash_uniform(seed_eff, j as u32),
        RngKind::Gumbel => -(-rng::hash_uniform(seed_eff, j as u32).ln()).ln(),
        RngKind::Normal => rng::hash_normal(seed_eff, j as u32),
    };
    if n <= SPLIT_ABOVE {
        return (0..n).map(lane).collect();
    }
    let threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .clamp(1, 8);
    if threads == 1 {
        return (0..n).map(lane).collect();
    }
    let mut out = vec![0.0f32; n];
    let chunk = n.div_ceil(threads);
    std::thread::scope(|scope| {
        for (c, slice) in out.chunks_mut(chunk).enumerate() {
            let lane = &lane;
            scope.spawn(move || {
                let base = c * chunk;
                for (k, slot) in slice.iter_mut().enumerate() {
                    *slot = lane(base + k);
                }
            });
        }
    });
    out
}

#[must_use]
pub fn gather_flat(v: &Value, idx: &[Option<usize>]) -> Value {
    match v {
        Value::I32(src) => Value::I32(idx.iter().map(|&i| i.map_or(0, |i| src[i])).collect()),
        Value::U32(src) => Value::U32(idx.iter().map(|&i| i.map_or(0, |i| src[i])).collect()),
        Value::Bool(src) => Value::Bool(idx.iter().map(|&i| i.map_or(0, |i| src[i])).collect()),
        Value::F32(src) => Value::F32(idx.iter().map(|&i| i.map_or(0.0, |i| src[i])).collect()),
    }
}

#[must_use]
pub fn broadcast_value(v: &Value, src: &[u32], target: &[u32]) -> Value {
    let r = target.len();
    let sdim = |i: usize| -> u64 { if i < src.len() { u64::from(src[i]) } else { 1 } };
    let n = shape_numel(target) as usize;

    if src.iter().all(|&d| d == 1) {
        return match v {
            Value::I32(s) => Value::I32(vec![s.first().copied().unwrap_or(0); n]),
            Value::U32(s) => Value::U32(vec![s.first().copied().unwrap_or(0); n]),
            Value::Bool(s) => Value::Bool(vec![s.first().copied().unwrap_or(0); n]),
            Value::F32(s) => Value::F32(vec![s.first().copied().unwrap_or(0.0); n]),
        };
    }

    let mut sstride = vec![1u64; r.max(1)];
    for i in (0..r.saturating_sub(1)).rev() {
        sstride[i] = sstride[i + 1] * sdim(i + 1);
    }
    let mut tstride = vec![1u64; r.max(1)];
    for i in (0..r.saturating_sub(1)).rev() {
        tstride[i] = tstride[i + 1] * u64::from(target[i + 1]);
    }
    let mut idx = Vec::with_capacity(n);
    for lin in 0..n as u64 {
        let mut rem = lin;
        let mut sidx = 0u64;
        for i in 0..r {
            let stride = tstride[i].max(1);
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

fn bin_arith(
    a: &Value,
    b: &Value,
    dtype: Dtype,
    f_f: impl Fn(f32, f32) -> f32,
    f_i: impl Fn(i64, i64) -> i64,
) -> Value {
    if dtype == Dtype::F32 {
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

fn cmp_op(
    a: &Value,
    b: &Value,
    in_dtype: Dtype,
    f_f: impl Fn(f32, f32) -> bool,
    f_i: impl Fn(i64, i64) -> bool,
) -> Value {
    if in_dtype == Dtype::F32 {
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

fn map_f32(v: &Value, f: impl Fn(f32) -> f32) -> Value {
    Value::F32(v.lanes_f32().into_iter().map(f).collect())
}

fn ty_dtype(package: &LaunchPackage, id: usize) -> Dtype {
    package.values[id].dtype
}

fn ty_shape(package: &LaunchPackage, id: usize) -> &[u32] {
    &package.values[id].shape
}

pub fn eval_op(op: &LaunchOp, package: &LaunchPackage, vals: &mut [Value]) -> Result<()> {
    let a0 = op.args.first().copied().unwrap_or(0) as usize;
    let a1 = op.args.get(1).copied().unwrap_or(0) as usize;
    let a2 = op.args.get(2).copied().unwrap_or(0) as usize;
    let result = op.result_id as usize;
    let code = op.tag;

    macro_rules! out {
        ($x:expr) => {{
            vals[result] = $x;
            return Ok(());
        }};
    }
    macro_rules! fault {
        ($m:expr) => {
            return Err(Error {
                message: $m.to_string(),
            })
        };
    }

    match code {
        tags::EXP => out!(map_f32(&vals[a0], f32::exp)),
        tags::LOG => out!(map_f32(&vals[a0], f32::ln)),
        tags::RECIP => out!(map_f32(&vals[a0], |x| 1.0 / x)),
        tags::SIN => out!(map_f32(&vals[a0], f32::sin)),
        tags::COS => out!(map_f32(&vals[a0], f32::cos)),
        tags::SQRT => out!(map_f32(&vals[a0], f32::sqrt)),
        tags::RSQRT => out!(map_f32(&vals[a0], |x| 1.0 / x.sqrt())),
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
            let want = op.dtype;
            let x = &vals[a0];
            match want {
                Dtype::F32 => out!(Value::F32(x.lanes_f32())),
                Dtype::I32 => {
                    if let Value::F32(f) = x {
                        out!(Value::I32(f.iter().map(|&e| e as i32).collect()));
                    }
                    out!(Value::from_i64(Dtype::I32, &x.lanes_i64()));
                }
                Dtype::U32 => {
                    if let Value::F32(f) = x {
                        out!(Value::U32(f.iter().map(|&e| e as u32).collect()));
                    }
                    out!(Value::from_i64(Dtype::U32, &x.lanes_i64()));
                }
                Dtype::Bool => {
                    out!(Value::Bool(
                        x.lanes_f32().iter().map(|&e| u8::from(e != 0.0)).collect()
                    ));
                }
                other => crate::value::no_lane(other),
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
                Dtype::F32 => {
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
                Dtype::Bool => {
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
            if dtype == Dtype::F32 {
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
            let o: Vec<i32> = if dtype == Dtype::F32 {
                match data {
                    Value::F32(x) => (0..rows)
                        .map(|r| argmax_row(&x[r * len..r * len + len]))
                        .collect(),
                    _ => {
                        let x = data.lanes_f32();
                        (0..rows)
                            .map(|r| argmax_row(&x[r * len..r * len + len]))
                            .collect()
                    }
                }
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

            if vals.get(payload).is_none_or(Value::is_empty) {
                return Err(Error {
                    message: format!(
                        "pivot_threshold reads value {payload} as its predicate payload \
                         and that value has no lanes"
                    ),
                });
            }
            let mut keep = vec![0u8; x.len()];
            for r in 0..rows {
                let row = &x[r * len..r * len + len];
                let k = &mut keep[r * len..r * len + len];
                match op.pred_tag {
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

                    1 => {
                        let pv = vals[payload].lanes_f32();
                        let p = pv[pick(pv.len(), r)];
                        if !nucleus_prefix(row, p, k) {
                            let order = sort_desc_order(row);
                            let mut excl = 0.0f32;
                            for i in order {
                                k[i as usize] = u8::from(excl < p);
                                excl += row[i as usize];
                            }
                        }
                    }

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
            if dtype == Dtype::F32 || (is_add && dtype != Dtype::I32 && dtype != Dtype::U32) {
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
            out!(Value::F32(rng_lanes(seed_eff, n, op.rng_kind)));
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
            out!(Value::F32(rng_lanes(seed64, n, op.rng_kind)));
        }
        tags::KERNEL_CALL => {
            if op.args.len() != 1 {
                fault!("Metal identity boundary arity");
            }
            out!(vals[a0].clone());
        }
        tags::SINK_CALL => Ok(()),

        _ => Err(Error {
            message: format!(
                "op not executable on the Metal host interpreter: {}",
                eta_ir::op::spec(code).map_or("?", |row| row.name)
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::{argmax_row, canonical_reduce_by};

    #[derive(Clone, Copy, Debug)]
    struct ArgmaxCandidate {
        value: f32,

        index: u32,

        have: bool,
    }

    impl Default for ArgmaxCandidate {
        fn default() -> Self {
            ArgmaxCandidate {
                value: f32::NEG_INFINITY,
                index: 0,
                have: false,
            }
        }
    }

    #[must_use]
    fn combine_argmax(left: ArgmaxCandidate, right: ArgmaxCandidate) -> ArgmaxCandidate {
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

    fn argmax_row_tree(row: &[f32]) -> i32 {
        canonical_reduce_by(
            row.len(),
            ArgmaxCandidate::default(),
            |j| ArgmaxCandidate {
                value: row[j],
                index: j as u32,
                have: !row[j].is_nan(),
            },
            combine_argmax,
        )
        .index as i32
    }

    #[test]
    fn the_scan_answers_what_the_canonical_tree_does() {
        let mut rows: Vec<Vec<f32>> = vec![
            vec![],
            vec![1.0],
            vec![f32::NAN],
            vec![f32::NAN, f32::NAN, f32::NAN],
            vec![1.0, 3.0, 3.0, 2.0],
            vec![-0.0, 0.0, -0.0],
            vec![f32::NAN, 5.0, f32::NAN, 5.0],
            vec![f32::NEG_INFINITY, f32::NEG_INFINITY],
            vec![f32::INFINITY, f32::INFINITY, 1.0],
        ];
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for len in [31usize, 32, 33, 64, 1000] {
            for _ in 0..12 {
                rows.push(
                    (0..len)
                        .map(|_| match next() % 8 {
                            0 => f32::NAN,
                            1 => 0.0,
                            2 => -0.0,
                            3 => 7.5,
                            _ => (next() % 64) as f32 - 32.0,
                        })
                        .collect(),
                );
            }
        }
        for row in &rows {
            assert_eq!(
                argmax_row(row),
                argmax_row_tree(row),
                "argmax disagreed on {row:?}"
            );
        }
    }
}

#[cfg(test)]
mod rng_tests {
    use super::rng_lanes;
    use eta_ir::RngKind;

    #[test]
    fn op_every_case() {
        a_split_draw_is_the_serial_draw();
        a_partial_nucleus_is_the_ordered_nucleus();
        a_broadcast_is_its_coordinate_walk();
    }

    fn a_split_draw_is_the_serial_draw() {
        let serial = |seed: u64, n: usize, kind: RngKind| -> Vec<f32> {
            (0..n)
                .map(|j| match kind {
                    RngKind::Uniform => eta_ir::rng::hash_uniform(seed, j as u32),
                    RngKind::Gumbel => -(-eta_ir::rng::hash_uniform(seed, j as u32).ln()).ln(),
                    RngKind::Normal => eta_ir::rng::hash_normal(seed, j as u32),
                })
                .collect()
        };
        for &n in &[1usize, 1 << 15, (1 << 15) + 1, 100_003, 262_144] {
            for kind in [RngKind::Uniform, RngKind::Gumbel, RngKind::Normal] {
                let want = serial(0x9E37_79B9_7F4A_7C15, n, kind);
                let got = rng_lanes(0x9E37_79B9_7F4A_7C15, n, kind);
                assert_eq!(got.len(), want.len(), "length at n={n}");
                for (j, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(g.to_bits(), w.to_bits(), "lane {j} of {n} ({kind:?})");
                }
            }
        }
    }
    fn a_partial_nucleus_is_the_ordered_nucleus() {
        fn ordered(row: &[f32], p: f32) -> Vec<u8> {
            let mut k = vec![0u8; row.len()];
            let mut excl = 0.0f32;
            for i in super::sort_desc_order(row) {
                k[i as usize] = u8::from(excl < p);
                excl += row[i as usize];
            }
            k
        }
        fn partial(row: &[f32], p: f32) -> Vec<u8> {
            let mut k = vec![0u8; row.len()];
            if super::nucleus_prefix(row, p, &mut k) {
                k
            } else {
                ordered(row, p)
            }
        }

        let mut row: Vec<f32> = (0..5000)
            .map(|i| 1.0f32 / ((i + 1) as f32 * (i + 1) as f32))
            .collect();
        let total: f32 = row.iter().sum();
        for v in &mut row {
            *v /= total;
        }
        for p in [0.0, 1e-9, 0.5, 0.9, 0.95, 0.999, 1.0, 2.0] {
            assert_eq!(partial(&row, p), ordered(&row, p), "softmax row at p={p}");
        }

        let flat = vec![0.001f32; 4096];
        for p in [0.0, 0.25, 1.0, 5.0] {
            assert_eq!(partial(&flat, p), ordered(&flat, p), "flat row at p={p}");
        }

        let mut sparse = vec![0.0f32; 3000];
        sparse[7] = 0.4;
        sparse[1500] = 0.6;
        for p in [0.3, 0.5, 0.95, 1.0] {
            assert_eq!(partial(&sparse, p), ordered(&sparse, p), "sparse at p={p}");
        }

        let mut negative = row.clone();
        negative[9] = -0.5;
        let mut nan = row.clone();
        nan[11] = f32::NAN;
        for bad in [&negative, &nan] {
            let mut k = vec![0u8; bad.len()];
            assert!(!super::nucleus_prefix(bad, 0.9, &mut k), "must fall back");
            assert!(k.iter().all(|&b| b == 0), "the fallback's `keep` is clean");
            assert_eq!(partial(bad, 0.9), ordered(bad, 0.9));
        }
        let mut k = vec![0u8; row.len()];
        assert!(!super::nucleus_prefix(&row, f32::NAN, &mut k));
    }

    #[test]
    #[ignore = "a measurement, not a gate"]
    #[allow(clippy::print_stdout)]
    fn a_nucleus_costs_less_than_an_order() {
        let vocab = 262_144usize;
        let build = |peak: f32, spread: f32| -> Vec<f32> {
            let raw: Vec<f32> = (0..vocab)
                .map(|i| {
                    let x = ((i * 2_654_435_761usize) % 100_003) as f32 / 100_003.0;
                    (x * spread - spread).exp()
                })
                .collect();
            let tail: f32 = raw.iter().sum();
            let mut row: Vec<f32> = raw.into_iter().map(|v| v / tail * (1.0 - peak)).collect();
            row[12345] += peak;
            row
        };
        let named: Vec<(&str, Vec<f32>)> = vec![
            ("one lane", build(0.97, 20.0)),
            ("a hundred", build(0.30, 20.0)),
            ("heavy tail", build(0.0, 12.0)),
        ];
        for (what, row) in &named {
            let mut k = vec![0u8; vocab];
            let mut excl = 0.0f32;
            for i in super::sort_desc_order(row) {
                k[i as usize] = u8::from(excl < 0.95);
                excl += row[i as usize];
            }
            let mut k2 = vec![0u8; vocab];
            let answered = super::nucleus_prefix(row, 0.95, &mut k2);
            if answered {
                assert_eq!(k, k2, "{what}");
            }
            println!(
                "{what}: nucleus {} of {vocab}, partial {}",
                k.iter().filter(|&&b| b == 1).count(),
                if answered { "answers" } else { "defers" }
            );
        }
        let rows: Vec<Vec<f32>> = named.into_iter().map(|(_, r)| r).collect();
        let p = 0.95f32;
        for (at, row) in rows.iter().enumerate() {
            let (mut ord_ns, mut part_ns) = (0u128, 0u128);
            for _ in 0..3 {
                let mut k = vec![0u8; vocab];
                let t = std::time::Instant::now();
                let mut excl = 0.0f32;
                for i in super::sort_desc_order(row) {
                    k[i as usize] = u8::from(excl < p);
                    excl += row[i as usize];
                }
                ord_ns += t.elapsed().as_nanos();

                let mut k2 = vec![0u8; vocab];
                let t = std::time::Instant::now();
                let answered = super::nucleus_prefix(row, p, &mut k2);
                part_ns += t.elapsed().as_nanos();
                if answered {
                    assert_eq!(k, k2);
                }
            }
            let answers = {
                let mut k = vec![0u8; vocab];
                super::nucleus_prefix(row, p, &mut k)
            };
            let paid = if answers {
                part_ns as f64 / 3e6
            } else {
                (part_ns + ord_ns) as f64 / 3e6
            };
            println!(
                "row {at} ({}): ordered {:.2} ms, now {:.2} ms, {:.2}x",
                if answers { "answers" } else { "defers" },
                ord_ns as f64 / 3e6,
                paid,
                ord_ns as f64 / 3e6 / paid
            );
        }
    }
    fn a_broadcast_is_its_coordinate_walk() {
        use crate::value::Value;
        fn walked(v: &Value, src: &[u32], target: &[u32]) -> Value {
            let r = target.len();
            let sdim = |i: usize| -> u64 { if i < src.len() { u64::from(src[i]) } else { 1 } };
            let mut sstride = vec![1u64; r.max(1)];
            for i in (0..r.saturating_sub(1)).rev() {
                sstride[i] = sstride[i + 1] * sdim(i + 1);
            }
            let n: usize = target.iter().map(|&d| d as usize).product::<usize>().max(1);
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
            super::gather_flat(v, &idx)
        }

        let cases: Vec<(Vec<f32>, Vec<u32>, Vec<u32>)> = vec![
            (vec![0.25], vec![1], vec![1, 64]),
            (vec![0.25], vec![1, 1], vec![3, 64]),
            ((0..8).map(|i| i as f32).collect(), vec![1, 8], vec![5, 8]),
            ((0..5).map(|i| i as f32).collect(), vec![5, 1], vec![5, 8]),
            (
                (0..6).map(|i| i as f32).collect(),
                vec![1, 3, 2],
                vec![4, 3, 2],
            ),
            (
                (0..4).map(|i| i as f32).collect(),
                vec![4, 1, 1],
                vec![4, 3, 2],
            ),
        ];
        for (data, src, target) in cases {
            let v = Value::F32(data);
            assert_eq!(
                super::broadcast_value(&v, &src, &target),
                walked(&v, &src, &target),
                "src {src:?} -> {target:?}"
            );
        }
    }
}
