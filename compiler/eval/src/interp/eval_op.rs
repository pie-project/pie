//! One op, evaluated against already-evaluated operands.
//!
//! No channel state, no readiness, no instance: everything `eval_op` needs
//! arrives as an argument. That is what lets `pareval` fold the same function
//! over host-known values without a second evaluator — the property the
//! module doc of `pareval.rs` rests on.

use alloc::vec;
use alloc::vec::Vec;

use pie_ir::op::{IntrinsicId, Op};
use pie_ir::rng;
use pie_ir::types::{DType, Literal, Predicate, RngKind, Shape, ValueId, ValueType};

use super::numeric::*;
use super::{ChanEffect, Evaled, PassInputs, StepError, Value, value_matches};

pub(crate) fn eval_op(
    op: &Op,
    vals: &[Value],
    ty_of: &dyn Fn(ValueId) -> ValueType,
    inputs: &PassInputs,
    layer: u32,
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
        Op::Neg(a) => One(match v(a) {
            Value::F32(x) => Value::F32(x.iter().map(|&a| -a).collect()),
            Value::I32(x) => Value::I32(x.iter().map(|&a| a.wrapping_neg()).collect()),
            Value::U32(x) => Value::U32(x.iter().map(|&a| a.wrapping_neg()).collect()),
            Value::Bool(_) => return Err(fault("neg on bool".into())),
        }),
        Op::Abs(a) => One(match v(a) {
            Value::F32(x) => Value::F32(x.iter().map(|&a| a.abs()).collect()),
            Value::I32(x) => Value::I32(x.iter().map(|&a| a.wrapping_abs()).collect()),
            // Already non-negative. Spelled out because the `_ => clone()`
            // this replaces would answer "abs is the identity" for a signed
            // dtype added later.
            Value::U32(x) => Value::U32(x.clone()),
            Value::Bool(x) => Value::Bool(x.clone()),
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
        }),
        Op::Cast { value, dtype } => One(match dtype {
            DType::F32 => Value::F32(lanes_f32(v(value))),
            DType::I32 => {
                if v(value).dtype() == DType::F32 {
                    Value::I32(lanes_f32(v(value)).iter().map(|&x| x as i32).collect())
                } else {
                    from_i64(DType::I32, lanes_i64(v(value)))
                }
            }
            DType::U32 => {
                if v(value).dtype() == DType::F32 {
                    Value::U32(lanes_f32(v(value)).iter().map(|&x| x as u32).collect())
                } else {
                    from_i64(DType::U32, lanes_i64(v(value)))
                }
            }
            DType::Bool => Value::Bool(lanes_f32(v(value)).iter().map(|&x| x != 0.0).collect()),
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
                DType::F32 => {
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
                DType::Bool => {
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
            };
            One(Value::I32(result))
        }

        Op::Broadcast { value, shape } => {
            let src = ty_of(value).shape;
            One(broadcast_value(v(value), src, shape))
        }
        Op::Reshape { value, .. } => One(v(value).clone()), // metadata only (row-major)
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

            // Scanned in the input's own dtype rather than through f32, which
            // is the whole reason the op is not F32-only: a u32 offset scan
            // past 2^24 is not representable in f32 and must not be rounded
            // on its way through the interpreter either.
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
            let mut keep = vec![false; x.len()];
            for r in 0..rows {
                let row = &x[r * len..(r + 1) * len];
                let k = &mut keep[r * len..(r + 1) * len];
                match predicate {
                    Predicate::RankLe(kid) => {
                        let kv = lanes_i64(v(kid));
                        let kk = kv[pick(kv.len(), r)].clamp(0, len as i64);
                        // pinned: #strictly-greater < k (ties may admit > k)
                        for (i, &xi) in row.iter().enumerate() {
                            if xi.is_nan() {
                                continue;
                            }
                            let greater =
                                row.iter().filter(|&&y| !y.is_nan() && y > xi).count() as i64;
                            k[i] = greater < kk;
                        }
                    }
                    Predicate::CummassLe(pid) => {
                        let pv = lanes_f32(v(pid));
                        let p = pv[pick(pv.len(), r)];
                        let order = sort_desc_order(row);
                        let mut excl = 0.0f32;
                        for &i in &order {
                            k[i as usize] = excl < p;
                            excl += row[i as usize];
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
                    flat.extend(std::iter::repeat_n(usize::MAX, rest)); // fill-0
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
            if tb.dtype == DType::F32 || is_add && tb.dtype != DType::I32 && tb.dtype != DType::U32
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
            // Per-row over the LAST axis: the single packed mask (one word
            // row, [ceil(n/32)] — the validator's shape rule) broadcasts
            // across rows; the bit index is the COLUMN `j % n`, never the
            // flat element index. Per-row *distinct* masks use the composed
            // bool-mask form (select), not this packed op.
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
        } => {
            // Ambient-seed form: the per-fire seed is 0 in the reference
            // interpreter unless the harness overrides via a keyed op —
            // PTIR programs use rng_keyed; this stays for PSIR parity work.
            One(Value::F32(rng_ambient(
                0,
                stream,
                kind,
                shape.numel() as usize,
            )))
        }
        Op::RngKeyed { state, shape, kind } => {
            let st = lanes_i64(v(state));
            let (key, ctr) = (st[0] as u64 & 0xFFFF_FFFF, st[1] as u64 & 0xFFFF_FFFF);
            let seed64 = rng::keyed_seed(key as u32, ctr as u32);
            let n = shape.numel() as usize;
            One(Value::F32(
                (0..n as u32)
                    .map(|j| {
                        let u = rng::hash_uniform(seed64, j);
                        match kind {
                            RngKind::Uniform => u,
                            RngKind::Gumbel => -((-(u.ln())).ln()),
                        }
                    })
                    .collect(),
            ))
        }

        Op::ChanTake(c) => Evaled::Chan(ChanEffect::Take(c)),
        Op::ChanRead(c) => Evaled::Chan(ChanEffect::Read(c)),
        Op::ChanPut { chan, value } => Evaled::Chan(ChanEffect::Put(chan, value)),

        Op::IntrinsicVal { intr, shape, dtype } => {
            let want = ValueType::new(shape, dtype);
            let got = match intr {
                IntrinsicId::Logits => inputs.logits.clone(),
                IntrinsicId::MtpLogits => inputs.mtp_logits.clone(),
                IntrinsicId::Hidden => inputs.hidden.clone(),
                IntrinsicId::ValueHead => inputs.value_head.clone(),
                IntrinsicId::Query => inputs.query.get(layer as usize).cloned(),
                IntrinsicId::Layer => Some(Value::U32(vec![layer])),
                IntrinsicId::MtpDrafts => inputs.mtp_drafts.clone(),
                IntrinsicId::AttnScore => inputs.attn_score.get(layer as usize).cloned(),
            };
            match got {
                Some(val) if value_matches(&val, want) => One(val),
                Some(_) => {
                    return Err(StepError::Fault(format!(
                        "intrinsic {} input violates its declared type",
                        intr.name()
                    )));
                }
                None => return Err(StepError::MissingIntrinsic(intr)),
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
    }
}

/// Flat gather where `usize::MAX` means fill-0.
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
    }
}

/// Left-aligned broadcast replicate (v4-exact), dtype-preserving.
pub(super) fn broadcast_value(value: &Value, src_shape: Shape, target: Shape) -> Value {
    let r = target.rank();
    let td = target.dims();
    let sd = src_shape.dims();
    let sdim = |i: usize| if i < sd.len() { sd[i] } else { 1u32 };
    let mut sstride = vec![1u64; r.max(1)];
    for i in (0..r.saturating_sub(1)).rev() {
        sstride[i] = sstride[i + 1] * sdim(i + 1) as u64;
    }
    let n = target.numel() as usize;
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
    gather_flat(value, &src_idx)
}

pub(super) fn rng_ambient(seed: u32, stream: u32, kind: RngKind, len: usize) -> Vec<f32> {
    let seed_eff = rng::seed_eff_stream(seed, stream);
    (0..len as u32)
        .map(|j| {
            let u = rng::hash_uniform(seed_eff, j);
            match kind {
                RngKind::Uniform => u,
                RngKind::Gumbel => -((-(u.ln())).ln()),
            }
        })
        .collect()
}

// Re-export for parity harnesses.
