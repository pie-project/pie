use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;

use eta_ir::op::Op;
use eta_ir::rng;
use eta_ir::types::{Dtype, Literal, Predicate, RngKind, Shape, ValueId, ValueType};

use super::numeric::*;
use super::{ChanEffect, Evaled, PassInputs, StepError, Value, value_matches};

/// One op over `vals` (indexed by the op's value ids), `ty_of` its
/// operands' types: its result(s), or the effect it asks of the caller (a
/// channel, a kernel, a sink). The one evaluator of an op on the host: the
/// tier-0 interpreter here and eta-exec's both run it.
///
/// A `bf16` operand is widened to f32 exactly, the op computes in f32, and
/// a `bf16` result is rounded once, to nearest even (`eta_ir::types`' bf16
/// rules): a reduction's or a scan's running value is f32 throughout.
pub fn eval_op(
    op: &Op,
    vals: &[Value],
    ty_of: &dyn Fn(ValueId) -> ValueType,
    inputs: &PassInputs,
    names: &[String],
) -> Result<Evaled, StepError> {
    let operands = op.operands();
    let effect = op.is_effectful() || matches!(op, Op::Output { .. });
    let bf16 = |id: &ValueId| vals[*id as usize].dtype() == Dtype::Bf16;
    let casts_to_bf16 = matches!(
        op,
        Op::Cast {
            dtype: Dtype::Bf16,
            ..
        }
    );
    if effect || !(casts_to_bf16 || operands.iter().any(bf16)) {
        return eval_wide(op, vals, ty_of, inputs, names);
    }
    // The op over its operands widened, numbered 0.. in its operand order.
    let local: Vec<Value> = operands
        .iter()
        .map(|&id| widen(&vals[id as usize]))
        .collect();
    let mut wide = op.clone();
    let mut next = 0;
    wide.map_operands(|_| {
        next += 1;
        next - 1
    });
    if let Op::Cast { dtype, .. } = &mut wide
        && *dtype == Dtype::Bf16
    {
        *dtype = Dtype::F32;
    }
    let wide_ty = |id: ValueId| {
        let ty = ty_of(operands[id as usize]);
        if ty.dtype == Dtype::Bf16 {
            ValueType::new(ty.shape, Dtype::F32)
        } else {
            ty
        }
    };
    let narrow = narrows(op, |id| bf16(&id));
    let round = |value: Value| match value {
        Value::F32(x) if narrow => Value::bf16_of(&x),
        other => other,
    };
    Ok(match eval_wide(&wide, &local, &wide_ty, inputs, names)? {
        Evaled::One(value) => Evaled::One(round(value)),
        Evaled::Two(values, indices) => Evaled::Two(round(values), indices),
        other => other,
    })
}

/// `value` with a `bf16` value widened to its exact f32s.
fn widen(value: &Value) -> Value {
    match value {
        Value::Bf16(x) => Value::F32(x.iter().map(|&b| eta_ir::types::bf16_to_f32(b)).collect()),
        other => other.clone(),
    }
}

/// Whether `op`'s (first) result is `bf16`, given which of its operands are
/// (`eta_ir::infer`'s rules): a cast to it; the result of a reduction, a
/// product, a comparison, a pivot or an argmax never; a gather's, a
/// scatter's or a mask's where its source, base or logits is; an
/// elementwise op's, a scan's, an order op's values and a shape op's where
/// any operand is.
pub(crate) fn narrows(op: &Op, bf16: impl Fn(ValueId) -> bool) -> bool {
    match *op {
        Op::Cast { dtype, .. } => dtype == Dtype::Bf16,
        Op::ReduceSum(_)
        | Op::ReduceMax(_)
        | Op::ReduceMin(_)
        | Op::ReduceArgmax(_)
        | Op::GumbelArgmax { .. }
        | Op::MatMul(..)
        | Op::Gt(..)
        | Op::Ge(..)
        | Op::Eq(..)
        | Op::Ne(..)
        | Op::Lt(..)
        | Op::Le(..)
        | Op::And(..)
        | Op::Or(..)
        | Op::Not(_)
        | Op::PivotThreshold { .. } => false,
        Op::Gather { src, .. } | Op::GatherRow { src, .. } => bf16(src),
        Op::ScatterAdd { base, .. } | Op::ScatterSet { base, .. } => bf16(base),
        Op::MaskApply { logits, .. } => bf16(logits),
        Op::Select { a, b, .. } => bf16(a) || bf16(b),
        _ => op.operands().into_iter().any(bf16),
    }
}

fn eval_wide(
    op: &Op,
    vals: &[Value],
    ty_of: &dyn Fn(ValueId) -> ValueType,
    inputs: &PassInputs,
    names: &[String],
) -> Result<Evaled, StepError> {
    use Evaled::One;
    let v = |id: ValueId| &vals[id as usize];
    let fault = |m: String| StepError::Fault(m);

    Ok(match *op {
        Op::Const(lit) => One(match lit {
            Literal::F32(x) => Value::F32(vec![x]),
            Literal::I32(x) => Value::I32(vec![x]),
            Literal::U32(x) => Value::U32(vec![x]),
            Literal::Bool(x) => Value::Bool(vec![x]),
        }),

        Op::Exp(a) => One(map_f32(v(a), |x| x.exp())),
        Op::Log(a) => One(map_f32(v(a), |x| x.ln())),
        Op::Recip(a) => One(map_f32(v(a), |x| 1.0 / x)),
        Op::Sin(a) => One(map_f32(v(a), f32::sin)),
        Op::Cos(a) => One(map_f32(v(a), f32::cos)),
        Op::Sqrt(a) => One(map_f32(v(a), f32::sqrt)),
        Op::Rsqrt(a) => One(map_f32(v(a), |x| 1.0 / x.sqrt())),
        Op::Neg(a) => One(match v(a) {
            Value::F32(x) => Value::F32(x.iter().map(|&a| -a).collect()),
            Value::I32(x) => Value::I32(x.iter().map(|&a| a.wrapping_neg()).collect()),
            Value::U32(x) => Value::U32(x.iter().map(|&a| a.wrapping_neg()).collect()),
            Value::Bool(_) => return Err(fault("neg on bool".into())),
            Value::Bf16(_) => return Err(unwidened()),
        }),
        Op::Abs(a) => One(match v(a) {
            Value::F32(x) => Value::F32(x.iter().map(|&a| a.abs()).collect()),
            Value::I32(x) => Value::I32(x.iter().map(|&a| a.wrapping_abs()).collect()),
            Value::U32(x) => Value::U32(x.clone()),
            Value::Bool(x) => Value::Bool(x.clone()),
            Value::Bf16(_) => return Err(unwidened()),
        }),
        Op::Sign(a) => One(match v(a) {
            Value::F32(x) => Value::F32(
                x.iter()
                    .map(|&a| {
                        if a > 0.0 {
                            1.0
                        } else if a < 0.0 {
                            -1.0
                        } else {
                            0.0
                        }
                    })
                    .collect(),
            ),
            Value::I32(x) => Value::I32(x.iter().map(|&a| a.signum()).collect()),
            Value::U32(x) => Value::U32(x.iter().map(|&a| (a != 0) as u32).collect()),
            Value::Bool(_) => return Err(fault("sign on bool".into())),
            Value::Bf16(_) => return Err(unwidened()),
        }),
        Op::Cast { value, dtype } => One(match dtype {
            Dtype::F32 => Value::F32(lanes_f32(v(value))),
            Dtype::I32 => {
                if v(value).dtype() == Dtype::F32 {
                    Value::I32(lanes_f32(v(value)).iter().map(|&x| x as i32).collect())
                } else {
                    from_i64(Dtype::I32, lanes_i64(v(value)))
                }
            }
            Dtype::U32 => {
                if v(value).dtype() == Dtype::F32 {
                    Value::U32(lanes_f32(v(value)).iter().map(|&x| x as u32).collect())
                } else {
                    from_i64(Dtype::U32, lanes_i64(v(value)))
                }
            }
            Dtype::Bool => Value::Bool(lanes_f32(v(value)).iter().map(|&x| x != 0.0).collect()),
            _ => super::no_interpreter_lane(dtype),
        }),

        Op::Add(a, b) => One(bin_arith(
            v(a),
            v(b),
            ty_of(a).dtype,
            |x, y| x + y,
            |x, y| x.wrapping_add(y),
        )),
        Op::Sub(a, b) => One(bin_arith(
            v(a),
            v(b),
            ty_of(a).dtype,
            |x, y| x - y,
            |x, y| x.wrapping_sub(y),
        )),
        Op::Mul(a, b) => One(bin_arith(
            v(a),
            v(b),
            ty_of(a).dtype,
            |x, y| x * y,
            |x, y| x.wrapping_mul(y),
        )),
        Op::Div(a, b) => One(bin_arith(
            v(a),
            v(b),
            ty_of(a).dtype,
            |x, y| x / y,
            |x, y| if y == 0 { 0 } else { x.wrapping_div(y) },
        )),
        Op::MaxElem(a, b) => One(bin_arith(
            v(a),
            v(b),
            ty_of(a).dtype,
            element_max,
            |x, y| x.max(y),
        )),
        Op::MinElem(a, b) => One(bin_arith(
            v(a),
            v(b),
            ty_of(a).dtype,
            element_min,
            |x, y| x.min(y),
        )),
        Op::Rem(a, b) => One(bin_arith(
            v(a),
            v(b),
            ty_of(a).dtype,
            |x, y| x % y,
            |x, y| if y == 0 { 0 } else { x.wrapping_rem(y) },
        )),

        Op::Gt(a, b) => One(cmp_op(
            v(a),
            v(b),
            ty_of(a).dtype,
            |x, y| x > y,
            |x, y| x > y,
        )),
        Op::Ge(a, b) => One(cmp_op(
            v(a),
            v(b),
            ty_of(a).dtype,
            |x, y| x >= y,
            |x, y| x >= y,
        )),
        Op::Eq(a, b) => One(cmp_op(
            v(a),
            v(b),
            ty_of(a).dtype,
            |x, y| x == y,
            |x, y| x == y,
        )),
        Op::Ne(a, b) => One(cmp_op(
            v(a),
            v(b),
            ty_of(a).dtype,
            |x, y| x != y,
            |x, y| x != y,
        )),
        Op::Lt(a, b) => One(cmp_op(
            v(a),
            v(b),
            ty_of(a).dtype,
            |x, y| x < y,
            |x, y| x < y,
        )),
        Op::Le(a, b) => One(cmp_op(
            v(a),
            v(b),
            ty_of(a).dtype,
            |x, y| x <= y,
            |x, y| x <= y,
        )),
        Op::And(a, b) | Op::Or(a, b) => {
            let (Value::Bool(x), Value::Bool(y)) = (v(a), v(b)) else {
                return Err(fault("and/or on non-bool".into()));
            };
            let n = x.len().max(y.len());
            let is_and = matches!(op, Op::And(..));
            One(Value::Bool(
                (0..n)
                    .map(|i| {
                        let (p, q) = (x[pick(x.len(), i)], y[pick(y.len(), i)]);
                        if is_and { p && q } else { p || q }
                    })
                    .collect(),
            ))
        }
        Op::Not(a) => {
            let Value::Bool(x) = v(a) else {
                return Err(fault("not on non-bool".into()));
            };
            One(Value::Bool(x.iter().map(|&b| !b).collect()))
        }

        Op::Select { cond, a, b } => {
            let Value::Bool(c) = v(cond) else {
                return Err(fault("select cond".into()));
            };
            let (av, bv) = (v(a), v(b));
            let n = c.len().max(av.len()).max(bv.len());
            let sel = |i: usize| c[pick(c.len(), i)];
            One(match ty_of(a).dtype {
                Dtype::F32 => {
                    let (x, y) = (lanes_f32(av), lanes_f32(bv));
                    Value::F32(
                        (0..n)
                            .map(|i| {
                                if sel(i) {
                                    x[pick(x.len(), i)]
                                } else {
                                    y[pick(y.len(), i)]
                                }
                            })
                            .collect(),
                    )
                }
                Dtype::Bool => {
                    let (Value::Bool(x), Value::Bool(y)) = (av, bv) else {
                        return Err(fault("select bool arms".into()));
                    };
                    Value::Bool(
                        (0..n)
                            .map(|i| {
                                if sel(i) {
                                    x[pick(x.len(), i)]
                                } else {
                                    y[pick(y.len(), i)]
                                }
                            })
                            .collect(),
                    )
                }
                d => {
                    let (x, y) = (lanes_i64(av), lanes_i64(bv));
                    from_i64(
                        d,
                        (0..n)
                            .map(|i| {
                                if sel(i) {
                                    x[pick(x.len(), i)]
                                } else {
                                    y[pick(y.len(), i)]
                                }
                            })
                            .collect(),
                    )
                }
            })
        }

        Op::ReduceSum(a) => One(reduce_rows(ReduceKind::Sum, ty_of(a), v(a))),
        Op::ReduceMax(a) => One(reduce_rows(ReduceKind::Max, ty_of(a), v(a))),
        Op::ReduceMin(a) => One(reduce_rows(ReduceKind::Min, ty_of(a), v(a))),
        Op::ReduceArgmax(a) => {
            let t = ty_of(a);
            let rows = rows_of(t.shape);
            let result = match v(a) {
                Value::F32(values) => {
                    let len = values.len().checked_div(rows).unwrap_or(0);
                    (0..rows)
                        .map(|row| argmax_row(&values[row * len..(row + 1) * len]))
                        .collect()
                }
                Value::I32(values) => {
                    let len = values.len().checked_div(rows).unwrap_or(0);
                    (0..rows)
                        .map(|row| argmax_ordered(&values[row * len..(row + 1) * len]))
                        .collect()
                }
                Value::U32(values) => {
                    let len = values.len().checked_div(rows).unwrap_or(0);
                    (0..rows)
                        .map(|row| argmax_ordered(&values[row * len..(row + 1) * len]))
                        .collect()
                }
                Value::Bool(_) => return Err(fault("argmax on bool".into())),
                Value::Bf16(_) => return Err(unwidened()),
            };
            One(Value::I32(result))
        }

        Op::GumbelArgmax {
            logits,
            keep,
            scale,
            state,
        } => {
            // Each lane as `select(keep, logits * scale, -inf) +
            // rng_keyed(state)` holds it, in f32, then the argmax.
            let t = ty_of(logits);
            let x = lanes_f32(v(logits));
            let Value::Bool(k) = v(keep) else {
                return Err(fault("gumbel_argmax keep".into()));
            };
            let s = lanes_f32(v(scale))[0];
            let st = lanes_i64(v(state));
            let (key, ctr) = (st[0] as u64 & 0xFFFF_FFFF, st[1] as u64 & 0xFFFF_FFFF);
            let seed64 = rng::keyed_seed(key as u32, ctr as u32);
            let noise = rng_lanes(seed64, x.len(), RngKind::Gumbel);
            let perturbed: Vec<f32> = (0..x.len())
                .map(|i| {
                    if k[pick(k.len(), i)] {
                        x[i] * s + noise[i]
                    } else {
                        f32::NEG_INFINITY
                    }
                })
                .collect();
            let rows = rows_of(t.shape);
            let len = perturbed.len().checked_div(rows).unwrap_or(0);
            One(Value::I32(
                (0..rows)
                    .map(|row| argmax_row(&perturbed[row * len..(row + 1) * len]))
                    .collect(),
            ))
        }

        Op::Broadcast { value, shape } => {
            let src = ty_of(value).shape;
            One(broadcast_value(v(value), src, shape))
        }
        Op::Reshape { value, .. } => One(v(value).clone()),
        Op::Transpose(a) => {
            let t = ty_of(a);
            let [m, n] = *t.shape.dims() else {
                return Err(fault("transpose rank".into()));
            };
            let (m, n) = (m as usize, n as usize);
            let idx: Vec<usize> = (0..m * n).map(|o| (o % m) * n + o / m).collect();
            One(gather_flat(v(a), &idx))
        }

        Op::CumSum(a) | Op::CumProd(a) => {
            let rows = rows_of(ty_of(a).shape);
            let is_sum = matches!(op, Op::CumSum(_));

            match v(a) {
                Value::I32(x) if is_sum => {
                    One(Value::I32(scan_rows(x, rows, 0, i32::wrapping_add)))
                }
                Value::I32(x) => One(Value::I32(scan_rows(x, rows, 1, i32::wrapping_mul))),
                Value::U32(x) if is_sum => {
                    One(Value::U32(scan_rows(x, rows, 0, u32::wrapping_add)))
                }
                Value::U32(x) => One(Value::U32(scan_rows(x, rows, 1, u32::wrapping_mul))),
                other => {
                    let lanes = lanes_f32(other);
                    let (identity, combine): (f32, fn(f32, f32) -> f32) = if is_sum {
                        (0.0, |a, b| a + b)
                    } else {
                        (1.0, |a, b| a * b)
                    };
                    One(Value::F32(scan_rows(&lanes, rows, identity, combine)))
                }
            }
        }

        Op::SortDesc(a) => {
            let x = lanes_f32(v(a));
            let order = sort_desc_order(&x);
            let sorted: Vec<f32> = order.iter().map(|&i| x[i as usize]).collect();
            Evaled::Two(Value::F32(sorted), Value::U32(order))
        }
        Op::TopK { input, k } => {
            let t = ty_of(input);
            let rows = rows_of(t.shape);
            let x = lanes_f32(v(input));
            let len = x.len().checked_div(rows).unwrap_or(0);
            let k = k as usize;
            let mut vs = Vec::with_capacity(rows * k);
            let mut is = Vec::with_capacity(rows * k);
            for r in 0..rows {
                let row = &x[r * len..(r + 1) * len];
                let order = sort_desc_order(row);
                for &i in order.iter().take(k) {
                    vs.push(row[i as usize]);
                    is.push(i);
                }
            }
            Evaled::Two(Value::F32(vs), Value::U32(is))
        }
        Op::MatMul(a, b) => {
            let (ta, tb) = (ty_of(a), ty_of(b));
            let [m, kk] = *ta.shape.dims() else {
                return Err(fault("matmul a".into()));
            };
            let [_, n] = *tb.shape.dims() else {
                return Err(fault("matmul b".into()));
            };
            let (m, kk, n) = (m as usize, kk as usize, n as usize);
            let (x, y) = (lanes_f32(v(a)), lanes_f32(v(b)));
            let mut out = vec![0.0f32; m * n];
            for i in 0..m {
                for l in 0..kk {
                    let xv = x[i * kk + l];
                    if xv == 0.0 {
                        continue;
                    }
                    for j in 0..n {
                        out[i * n + j] += xv * y[l * n + j];
                    }
                }
            }
            One(Value::F32(out))
        }
        Op::PivotThreshold { input, predicate } => {
            let t = ty_of(input);
            let rows = rows_of(t.shape);
            let x = lanes_f32(v(input));
            let len = x.len().checked_div(rows).unwrap_or(0);
            let payload = match predicate {
                Predicate::RankLe(id) | Predicate::CummassLe(id) | Predicate::ProbGe(id) => id,
            };
            if v(payload).is_empty() {
                return Err(fault(format!(
                    "pivot_threshold reads value {payload} as its predicate payload and that \
                     value has no lanes"
                )));
            }
            let mut keep = vec![false; x.len()];
            for r in 0..rows {
                let row = &x[r * len..(r + 1) * len];
                let k = &mut keep[r * len..(r + 1) * len];
                match predicate {
                    Predicate::RankLe(kid) => {
                        let kv = lanes_i64(v(kid));
                        let kk = kv[pick(kv.len(), r)].clamp(0, len as i64);
                        eta_ir::rank::rank_cut(row, kk as usize, |i| k[i] = true);
                    }
                    Predicate::CummassLe(pid) => {
                        let pv = lanes_f32(v(pid));
                        let p = pv[pick(pv.len(), r)];
                        if !nucleus_prefix(row, p, k) {
                            let order = sort_desc_order(row);
                            let mut excl = 0.0f32;
                            for &i in &order {
                                k[i as usize] = excl < p;
                                excl += row[i as usize];
                            }
                        }
                    }
                    Predicate::ProbGe(tid) => {
                        let tv = lanes_f32(v(tid));
                        let thr = tv[pick(tv.len(), r)];
                        for (i, &xi) in row.iter().enumerate() {
                            k[i] = xi >= thr;
                        }
                    }
                }
            }
            One(Value::Bool(keep))
        }
        Op::Gather { src, idx } => {
            let ts = ty_of(src);
            let rest: usize = ts.shape.dims()[1..]
                .iter()
                .map(|&d| d as usize)
                .product::<usize>()
                .max(1);
            let n0 = ts.shape.dims()[0] as usize;
            let ix = lanes_i64(v(idx));
            let mut flat = Vec::with_capacity(ix.len() * rest);
            for &i in &ix {
                if i >= 0 && (i as usize) < n0 {
                    let base = i as usize * rest;
                    flat.extend(base..base + rest);
                } else {
                    flat.extend(std::iter::repeat_n(usize::MAX, rest));
                }
            }
            One(gather_flat_fill0(v(src), &flat))
        }
        Op::GatherRow { src, idx } => {
            let ts = ty_of(src);
            let [m, n] = *ts.shape.dims() else {
                return Err(fault("gather_row".into()));
            };
            let (m, n) = (m as usize, n as usize);
            let ix = lanes_i64(v(idx));
            let flat: Vec<usize> = (0..m)
                .map(|i| {
                    let c = ix[i];
                    if c >= 0 && (c as usize) < n {
                        i * n + c as usize
                    } else {
                        usize::MAX
                    }
                })
                .collect();
            One(gather_flat_fill0(v(src), &flat))
        }
        Op::ScatterAdd {
            base,
            idx,
            vals: vv,
        }
        | Op::ScatterSet {
            base,
            idx,
            vals: vv,
        } => {
            let tb = ty_of(base);
            let rest: usize = tb.shape.dims()[1..]
                .iter()
                .map(|&d| d as usize)
                .product::<usize>()
                .max(1);
            let n0 = tb.shape.dims()[0] as usize;
            let ix = lanes_i64(v(idx));
            let val = v(vv);
            let scalar_val = val.len() == 1 && ix.len() * rest != 1;
            let is_add = matches!(op, Op::ScatterAdd { .. });
            if tb.dtype == Dtype::F32 || is_add && tb.dtype != Dtype::I32 && tb.dtype != Dtype::U32
            {
                let mut out = lanes_f32(v(base));
                let vals_f = lanes_f32(val);
                for (k, &i) in ix.iter().enumerate() {
                    if i >= 0 && (i as usize) < n0 {
                        for r in 0..rest {
                            let src = if scalar_val {
                                vals_f[0]
                            } else {
                                vals_f[k * rest + r]
                            };
                            let dst = &mut out[i as usize * rest + r];
                            if is_add { *dst += src } else { *dst = src }
                        }
                    }
                }
                One(Value::F32(out))
            } else {
                let mut out = lanes_i64(v(base));
                let vals_i = lanes_i64(val);
                for (k, &i) in ix.iter().enumerate() {
                    if i >= 0 && (i as usize) < n0 {
                        for r in 0..rest {
                            let src = if scalar_val {
                                vals_i[0]
                            } else {
                                vals_i[k * rest + r]
                            };
                            let dst = &mut out[i as usize * rest + r];
                            if is_add {
                                *dst = dst.wrapping_add(src)
                            } else {
                                *dst = src
                            }
                        }
                    }
                }
                One(from_i64(tb.dtype, out))
            }
        }
        Op::Iota { len } => One(Value::U32((0..len).collect())),
        Op::MaskApply { logits, mask } => {
            let n = ty_of(logits).shape.last_len().unwrap_or(1) as usize;
            let x = lanes_f32(v(logits));
            let Value::U32(words) = v(mask) else {
                return Err(fault("mask_apply mask".into()));
            };
            One(Value::F32(
                x.iter()
                    .enumerate()
                    .map(|(j, &l)| {
                        let c = j % n;
                        let bit = words.get(c >> 5).map_or(0, |&w| (w >> (c & 31)) & 1);
                        if bit == 1 { l } else { f32::NEG_INFINITY }
                    })
                    .collect(),
            ))
        }
        Op::CausalMask { positions, len } => {
            let Value::U32(positions) = v(positions) else {
                return Err(fault("causal_mask positions".into()));
            };
            One(Value::Bool(
                positions
                    .iter()
                    .flat_map(|&position| (0..len).map(move |key| key <= position))
                    .collect(),
            ))
        }
        Op::SlidingWindowMask {
            positions,
            len,
            window,
        } => {
            let Value::U32(positions) = v(positions) else {
                return Err(fault("sliding_window_mask positions".into()));
            };
            One(Value::Bool(
                positions
                    .iter()
                    .flat_map(|&position| {
                        (0..len).map(move |key| {
                            key <= position && key.saturating_add(window) > position
                        })
                    })
                    .collect(),
            ))
        }
        Op::SinkWindowMask {
            positions,
            len,
            sink,
            window,
        } => {
            let Value::U32(positions) = v(positions) else {
                return Err(fault("sink_window_mask positions".into()));
            };
            One(Value::Bool(
                positions
                    .iter()
                    .flat_map(|&position| {
                        (0..len).map(move |key| {
                            key <= position && (key < sink || key.saturating_add(window) > position)
                        })
                    })
                    .collect(),
            ))
        }
        Op::Rng {
            stream,
            shape,
            kind,
        } => One(Value::F32(rng_lanes(
            rng::seed_eff_stream(0, stream),
            shape.numel() as usize,
            kind,
        ))),
        Op::RngKeyed { state, shape, kind } => {
            let st = lanes_i64(v(state));
            let (key, ctr) = (st[0] as u64 & 0xFFFF_FFFF, st[1] as u64 & 0xFFFF_FFFF);
            let seed64 = rng::keyed_seed(key as u32, ctr as u32);
            One(Value::F32(rng_lanes(seed64, shape.numel() as usize, kind)))
        }

        Op::ChanTake(c) => Evaled::Chan(ChanEffect::Take(c)),
        Op::ChanRead(c) => Evaled::Chan(ChanEffect::Read(c)),
        Op::ChanPut { chan, value } => Evaled::Chan(ChanEffect::Put(chan, value)),

        Op::Output {
            index,
            lane,
            shape,
            dtype,
        } => {
            let want = ValueType::new(shape, dtype);
            let name = names.get(index as usize).map_or("", String::as_str);
            match inputs.output(name, lane) {
                Some(val) if value_matches(val, want) => One(val.clone()),
                Some(_) => {
                    return Err(StepError::Fault(format!(
                        "output `{name}` of lane {lane} violates its declared type"
                    )));
                }
                None => {
                    return Err(StepError::MissingOutput {
                        name: name.into(),
                        lane,
                    });
                }
            }
        }
        Op::KernelCall {
            name,
            ref args,
            shape,
            dtype,
        } => Evaled::Kernel {
            name,
            args: args.clone(),
            result: ValueType::new(shape, dtype),
        },
        Op::SinkCall { name, ref args } => Evaled::Sink {
            name,
            args: args.clone(),
        },
    })
}

pub(super) fn gather_flat(v: &Value, idx: &[usize]) -> Value {
    match v {
        Value::F32(x) => Value::F32(idx.iter().map(|&i| x[i]).collect()),
        Value::I32(x) => Value::I32(idx.iter().map(|&i| x[i]).collect()),
        Value::U32(x) => Value::U32(idx.iter().map(|&i| x[i]).collect()),
        Value::Bool(x) => Value::Bool(idx.iter().map(|&i| x[i]).collect()),
        Value::Bf16(x) => Value::Bf16(idx.iter().map(|&i| x[i]).collect()),
    }
}

pub(super) fn gather_flat_fill0(v: &Value, idx: &[usize]) -> Value {
    match v {
        Value::F32(x) => Value::F32(
            idx.iter()
                .map(|&i| if i == usize::MAX { 0.0 } else { x[i] })
                .collect(),
        ),
        Value::I32(x) => Value::I32(
            idx.iter()
                .map(|&i| if i == usize::MAX { 0 } else { x[i] })
                .collect(),
        ),
        Value::U32(x) => Value::U32(
            idx.iter()
                .map(|&i| if i == usize::MAX { 0 } else { x[i] })
                .collect(),
        ),
        Value::Bool(x) => Value::Bool(idx.iter().map(|&i| i != usize::MAX && x[i]).collect()),
        Value::Bf16(x) => Value::Bf16(
            idx.iter()
                .map(|&i| if i == usize::MAX { 0 } else { x[i] })
                .collect(),
        ),
    }
}

/// A `bf16` value reached the f32 evaluator, which [`eval_op`] widens
/// every one before.
fn unwidened() -> StepError {
    StepError::Fault("a bf16 operand reached the interpreter's f32 evaluator unwidened".into())
}

pub(super) fn broadcast_value(value: &Value, src_shape: Shape, target: Shape) -> Value {
    let r = target.rank();
    let td = target.dims();
    let sd = src_shape.dims();
    let sdim = |i: usize| if i < sd.len() { sd[i] } else { 1u32 };
    let mut sstride = vec![1u64; r.max(1)];
    for i in (0..r.saturating_sub(1)).rev() {
        sstride[i] = sstride[i + 1] * sdim(i + 1) as u64;
    }
    let mut tstride = vec![1u64; r.max(1)];
    for i in (0..r.saturating_sub(1)).rev() {
        tstride[i] = tstride[i + 1] * td[i + 1] as u64;
    }
    let n = target.numel() as usize;
    let src_idx: Vec<usize> = (0..n as u64)
        .map(|lin| {
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
            sidx as usize
        })
        .collect();
    gather_flat(value, &src_idx)
}

/// `n` draws of `kind` from the stream `seed_eff`, lane `j` the stream's
/// `j`th: the same lanes however they are split, a long run's chunks on
/// threads of their own.
#[cfg_attr(
    target_arch = "wasm32",
    allow(
        clippy::disallowed_methods,
        reason = "available_parallelism is Err on wasm32, so one worker: the serial branch returns before the scope"
    )
)]
pub(super) fn rng_lanes(seed_eff: u64, n: usize, kind: RngKind) -> Vec<f32> {
    const SPLIT_ABOVE: usize = 1 << 15;
    let lane = |j: usize| {
        let j = j as u32;
        match kind {
            RngKind::Uniform => rng::hash_uniform(seed_eff, j),
            RngKind::Gumbel => -((-(rng::hash_uniform(seed_eff, j).ln())).ln()),
            RngKind::Normal => rng::hash_normal(seed_eff, j),
        }
    };
    let threads = std::thread::available_parallelism()
        .map_or(1, core::num::NonZeroUsize::get)
        .clamp(1, 8);
    if n <= SPLIT_ABOVE || threads == 1 {
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

/// The nucleus cut of a row whose mass `p` lies within its 64 largest
/// values: those values kept in `keep` as the full sort's walk keeps them
/// (the same order, NaN last, ties by index; the same running sum), and
/// `true`. `false`, `keep` untouched, where the row is no wider than that,
/// holds a NaN or a negative value (its running sum then not rising), `p`
/// is NaN, the 64 hold less than `p`, or finding them took too many
/// inserts: the caller sorts the whole row.
fn nucleus_prefix(row: &[f32], p: f32, keep: &mut [bool]) -> bool {
    const WIDE: usize = 64;
    const GIVE_UP: usize = 32 * WIDE;
    if p.is_nan() || WIDE >= row.len() {
        return false;
    }
    let mut best: Vec<u32> = Vec::with_capacity(WIDE + 1);
    let mut inserts = 0usize;
    for (i, &value) in row.iter().enumerate() {
        if value.is_nan() || value < 0.0 {
            return false;
        }
        let i = i as u32;
        if best.len() == WIDE && desc_by_value(row, i, best[WIDE - 1]) != Ordering::Less {
            continue;
        }
        let at = best.partition_point(|&b| desc_by_value(row, b, i) == Ordering::Less);
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
        keep[i as usize] = true;
        excl += row[i as usize];
    }
    if excl >= p {
        return true;
    }
    for &i in &best {
        keep[i as usize] = false;
    }
    false
}

/// [`sort_desc_order`]'s order of two of `row`'s indices.
fn desc_by_value(row: &[f32], a: u32, b: u32) -> Ordering {
    let (x, y) = (row[a as usize], row[b as usize]);
    match (x.is_nan(), y.is_nan()) {
        (true, true) => a.cmp(&b),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => y.partial_cmp(&x).unwrap_or(Ordering::Equal).then(a.cmp(&b)),
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::{nucleus_prefix, rng_lanes, sort_desc_order};
    use eta_ir::RngKind;

    /// The full sort's nucleus walk, the fast prefix's reference.
    fn walk(row: &[f32], p: f32) -> Vec<bool> {
        let mut keep = vec![false; row.len()];
        let mut excl = 0.0f32;
        for i in sort_desc_order(row) {
            keep[i as usize] = excl < p;
            excl += row[i as usize];
        }
        keep
    }

    /// Where the 64-wide prefix answers, it answers what the full sort's
    /// walk does: softmax-like rows with ties, peaked and flat, at masses
    /// from none to all of the row; where it does not, it leaves `keep`
    /// untouched.
    #[test]
    fn the_nucleus_prefix_keeps_what_the_full_walk_keeps() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut answered = 0;
        for width in [65usize, 128, 1000, 4096] {
            for peak in [1.0f32, 8.0, 64.0] {
                let mut row: Vec<f32> = (0..width)
                    .map(|_| ((next() % 1000) as f32 / 1000.0 * peak).exp())
                    .collect();
                row[(next() as usize) % width] = row[0];
                let total: f32 = row.iter().sum();
                for value in &mut row {
                    *value /= total;
                }
                for p in [-1.0f32, 0.0, 0.001, 0.1, 0.5, 0.9, 0.999, 1.5, f32::NAN] {
                    let mut keep = vec![false; width];
                    if nucleus_prefix(&row, p, &mut keep) {
                        answered += 1;
                        assert_eq!(keep, walk(&row, p), "width {width}, peak {peak}, p {p}");
                    } else {
                        assert!(keep.iter().all(|&k| !k), "a refused prefix keeps nothing");
                    }
                }
            }
        }
        assert!(answered > 0, "the prefix answers some rows");
        let mut negative = vec![0.01f32; 100];
        negative[3] = -0.5;
        let mut keep = vec![false; 100];
        assert!(!nucleus_prefix(&negative, 0.0, &mut keep));
    }

    /// A long draw's threads make the lanes one thread would.
    #[test]
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
}
