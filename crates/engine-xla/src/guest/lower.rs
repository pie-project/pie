//! One eta stage (`LaunchPackage::stages[at]`) as one StableHLO function
//! over a batch of lanes.
//!
//! Every eta value of shape `S` becomes a tensor `[B, S...]`: lane `b` of the
//! batch is one instance's pass. The semantics are the host interpreter's
//! (`eta_exec::eval_op`), op by op; where the interpreter is exact (integer
//! arithmetic, comparisons, argmax, orderings and their ties, masks, the
//! uniform draw's bits) so is this, and float math agrees within rounding.
//!
//! The function's parameters are one packed `u32 [B, in_words]` plane (every
//! host-fed root, bit-exact: f32/i32/u32 one word per element, bools 32 per
//! word; plus, per device-read intrinsic, the lane's first readout row), then
//! the fire's logits readout `f32 [N, W]` and its draft readout when the
//! stage reads them. The one result is `u32 [B, out_words]`: every value the
//! stage puts, packed the same way.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use eta_compiler::codegen::launch::{LaunchOp, LaunchPackage, ValueOrigin};
use eta_ir::op::{IntrinsicId, tags};
use eta_ir::{Dtype, RngKind, rng};
use kernels_xla::hlo::{Cmp, Elem, Fold, Func, Malformed, ScatterDims, Ty, Val};

/// Why a stage has no StableHLO form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refused(pub String);

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<Malformed> for Refused {
    fn from(m: Malformed) -> Refused {
        Refused(format!("the builder refused {m}"))
    }
}

type Lowering<T> = Result<T, Refused>;

fn refuse<T>(why: impl Into<String>) -> Lowering<T> {
    Err(Refused(why.into()))
}

/// The shape of one lowering: how many lanes, and which readouts the
/// device binds (`[rows, width]` of the fire's readout planes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Batch {
    pub lanes: u32,
    pub logits: Option<(u32, u32)>,
    pub mtp: Option<(u32, u32)>,
}

/// What a packed slot carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feed {
    /// The value itself: `dtype` over `numel` elements.
    Value(Dtype, usize),
    /// The lane's first row in the readout `intrinsic` reads (one i32 word);
    /// the stage reads `rows` rows from there.
    Row { intrinsic: IntrinsicId, rows: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    /// The package-global value id.
    pub value: u32,
    pub feed: Feed,
    /// First word in the lane's packed row.
    pub at: usize,
    pub words: usize,
}

/// How a lowered stage's one input and one output are laid out.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Layout {
    pub inputs: Vec<Slot>,
    pub in_words: usize,
    pub outputs: Vec<Slot>,
    pub out_words: usize,
    pub logits: bool,
    pub mtp: bool,
    /// Carried cells: channels whose cell stays on the device between
    /// passes. Each is its own `u32 [B, words]` parameter (after the pack
    /// and the readouts) and its own result (after the pack), per channel.
    pub carried_in: Vec<(u32, Slot)>,
    pub carried_out: Vec<(u32, Slot)>,
}

impl Layout {
    /// Whether the stage returns anything (else there is nothing to run).
    #[must_use]
    pub fn runs(&self) -> bool {
        self.out_words > 0 || !self.carried_out.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct Lowered {
    pub text: String,
    pub layout: Layout,
}

/// Words a value of `dtype` over `numel` elements packs into.
#[must_use]
pub fn words(dtype: Dtype, numel: usize) -> usize {
    if dtype == Dtype::Bool {
        numel.div_ceil(32)
    } else {
        numel
    }
}

fn elem(dtype: Dtype) -> Lowering<Elem> {
    Ok(match dtype {
        Dtype::F32 => Elem::F32,
        Dtype::I32 => Elem::I32,
        Dtype::U32 => Elem::U32,
        Dtype::Bool => Elem::Pred,
        other => return refuse(format!("{other:?} is not a dtype eta computes in")),
    })
}

fn numel(shape: &[u32]) -> usize {
    shape.iter().map(|&d| d as usize).product::<usize>().max(1)
}

fn op_name(tag: u8) -> &'static str {
    eta_ir::op::spec(tag).map_or("?", |row| row.name)
}

/// The values stage `at` reads without computing them, in order of first
/// use (channel cells, constants, intrinsics).
#[must_use]
pub fn roots(package: &LaunchPackage, at: usize) -> Vec<u32> {
    let Some(stage) = package.stages.get(at) else {
        return Vec::new();
    };
    let mut produced: HashSet<u32> = HashSet::new();
    for op in &stage.ops {
        for r in 0..u32::from(op.result_count) {
            produced.insert(op.result_id + r);
        }
    }
    // Roots in order of first use.
    let mut roots: Vec<u32> = Vec::new();
    let mut seen: HashSet<u32> = HashSet::new();
    let mut note = |id: u32, roots: &mut Vec<u32>| {
        if !produced.contains(&id) && seen.insert(id) {
            roots.push(id);
        }
    };
    for op in &stage.ops {
        for &arg in &op.args {
            note(arg, &mut roots);
        }
        if op.tag == tags::PIVOT_THRESHOLD {
            note(op.pred_payload, &mut roots);
        }
    }
    for put in &stage.puts {
        note(put.value, &mut roots);
    }
    roots
}

/// Which readouts stage `at` reads: the logits, the draft head's.
#[must_use]
pub fn reads(package: &LaunchPackage, at: usize) -> (bool, bool) {
    let mut logits = false;
    let mut mtp = false;
    for id in roots(package, at) {
        match package.values.get(id as usize).and_then(|v| v.intrinsic) {
            Some(IntrinsicId::Logits) => logits = true,
            Some(IntrinsicId::MtpLogits | IntrinsicId::MtpDrafts) => mtp = true,
            _ => {}
        }
    }
    (logits, mtp)
}

/// Lowers stage `at` of `package` for `batch`, with the cells of the
/// `carried` channels on the device (see `guest::carried_channels`).
pub fn lower(
    package: &LaunchPackage,
    at: usize,
    batch: Batch,
    carried: &[u32],
) -> Lowering<Lowered> {
    if at >= package.stages.len() {
        return refuse(format!("the package has no stage {at}"));
    }
    let stage = &package.stages[at];
    let roots = roots(package, at);

    let mut layout = Layout::default();
    for &id in &roots {
        let value = package
            .values
            .get(id as usize)
            .ok_or_else(|| Refused(format!("value {id} is past the package's values")))?;
        elem(value.dtype)?;
        let n = numel(&value.shape);
        let feed = match value.source {
            ValueOrigin::Const => continue,
            ValueOrigin::ChannelTake | ValueOrigin::ChannelRead
                if carried.contains(&value.channel) =>
            {
                layout.carried_in.push((
                    value.channel,
                    Slot {
                        value: id,
                        feed: Feed::Value(value.dtype, n),
                        at: 0,
                        words: words(value.dtype, n),
                    },
                ));
                continue;
            }
            ValueOrigin::ChannelTake | ValueOrigin::ChannelRead => Feed::Value(value.dtype, n),
            ValueOrigin::Intrinsic => {
                let intrinsic = value
                    .intrinsic
                    .ok_or_else(|| Refused(format!("intrinsic root {id} names no intrinsic")))?;
                let read = match intrinsic {
                    IntrinsicId::Logits => batch.logits,
                    IntrinsicId::MtpLogits | IntrinsicId::MtpDrafts => batch.mtp,
                    _ => None,
                };
                match read {
                    None => Feed::Value(value.dtype, n),
                    Some((_, width)) => {
                        let rows = if intrinsic == IntrinsicId::MtpDrafts {
                            n as u32
                        } else {
                            if width == 0 || !n.is_multiple_of(width as usize) {
                                return refuse(format!(
                                    "the `{}` root holds {n} values, not whole {width}-wide rows",
                                    intrinsic.name()
                                ));
                            }
                            (n / width as usize) as u32
                        };
                        if intrinsic == IntrinsicId::Logits {
                            layout.logits = true;
                        } else {
                            layout.mtp = true;
                        }
                        Feed::Row { intrinsic, rows }
                    }
                }
            }
            ValueOrigin::OpResult => {
                return refuse(format!(
                    "value {id} is another stage's result, and a stage reads only roots"
                ));
            }
        };
        let words = match feed {
            Feed::Value(dtype, n) => words(dtype, n),
            Feed::Row { .. } => 1,
        };
        layout.inputs.push(Slot {
            value: id,
            feed,
            at: layout.in_words,
            words,
        });
        layout.in_words += words;
    }
    let mut outs_seen = HashSet::new();
    for put in &stage.puts {
        let value = &package.values[put.value as usize];
        let n = numel(&value.shape);
        let w = words(value.dtype, n);
        if carried.contains(&put.channel) {
            layout.carried_out.push((
                put.channel,
                Slot {
                    value: put.value,
                    feed: Feed::Value(value.dtype, n),
                    at: 0,
                    words: w,
                },
            ));
            continue;
        }
        if !outs_seen.insert(put.value) {
            continue;
        }
        layout.outputs.push(Slot {
            value: put.value,
            feed: Feed::Value(value.dtype, n),
            at: layout.out_words,
            words: w,
        });
        layout.out_words += w;
    }

    let mut l = Lower {
        f: Func::new("main"),
        package,
        b: i64::from(batch.lanes.max(1)),
        env: HashMap::new(),
    };
    let pack = (layout.in_words > 0).then(|| {
        l.f.param(Ty::new(Elem::U32, &[l.b, layout.in_words as i64]), None)
    });
    let logits = if layout.logits {
        let (rows, width) = batch.logits.unwrap_or((1, 1));
        Some(l.f.param(
            Ty::new(Elem::F32, &[i64::from(rows), i64::from(width)]),
            None,
        ))
    } else {
        None
    };
    let mtp = if layout.mtp {
        let (rows, width) = batch.mtp.unwrap_or((1, 1));
        Some(l.f.param(
            Ty::new(Elem::F32, &[i64::from(rows), i64::from(width)]),
            None,
        ))
    } else {
        None
    };
    let carried_params: Vec<Val> = layout
        .carried_in
        .iter()
        .map(|(_, slot)| {
            l.f.param(Ty::new(Elem::U32, &[l.b, slot.words as i64]), None)
        })
        .collect();
    for ((_, slot), &param) in layout.carried_in.iter().zip(&carried_params) {
        let Feed::Value(dtype, n) = slot.feed else {
            unreachable!("carried cells are values");
        };
        let v = l.unpack(param, 0, dtype, n)?;
        let v = l.reshaped(v, &package.values[slot.value as usize].shape)?;
        l.env.insert(slot.value, v);
    }

    // Roots.
    for &id in &roots {
        if l.env.contains_key(&id) {
            continue;
        }
        let value = &package.values[id as usize];
        let v = if value.source == ValueOrigin::Const {
            l.constant(value.dtype, value.literal_bits)?
        } else {
            let slot = *layout
                .inputs
                .iter()
                .find(|s| s.value == id)
                .expect("every non-const root has a slot");
            let pack = pack.expect("a slot implies a packed input");
            match slot.feed {
                Feed::Value(dtype, n) => {
                    let v = l.unpack(pack, slot.at, dtype, n)?;
                    l.reshaped(v, &value.shape)?
                }
                Feed::Row { intrinsic, rows } => {
                    let plane = if intrinsic == IntrinsicId::Logits {
                        logits
                    } else {
                        mtp
                    }
                    .expect("a row feed implies its readout parameter");
                    l.read_rows(pack, slot.at, plane, intrinsic, rows, &value.shape)?
                }
            }
        };
        l.env.insert(id, v);
    }

    for op in &stage.ops {
        l.op(op).map_err(|Refused(why)| {
            Refused(format!(
                "stage {at}, `{}` ({:#04x}): {why}",
                op_name(op.tag),
                op.tag
            ))
        })?;
    }

    let mut packed = Vec::with_capacity(layout.outputs.len());
    for slot in &layout.outputs {
        let v = l.get(slot.value)?;
        let Feed::Value(dtype, n) = slot.feed else {
            unreachable!("outputs are values");
        };
        packed.push(l.pack(v, dtype, n)?);
    }
    let mut results = Vec::with_capacity(1 + layout.carried_out.len());
    if !packed.is_empty() {
        results.push(l.f.concat(&packed, 1)?);
    }
    for (_, slot) in &layout.carried_out {
        let v = l.get(slot.value)?;
        let Feed::Value(dtype, n) = slot.feed else {
            unreachable!("carried cells are values");
        };
        results.push(l.pack(v, dtype, n)?);
    }
    let text = if results.is_empty() {
        String::new()
    } else {
        l.f.module("eta_stage", &results)
    };
    Ok(Lowered { text, layout })
}

struct Lower<'p> {
    f: Func,
    package: &'p LaunchPackage,
    b: i64,
    env: HashMap<u32, Val>,
}

impl Lower<'_> {
    fn shape(&self, id: u32) -> Vec<u32> {
        self.package.values[id as usize].shape.clone()
    }

    fn dtype(&self, id: u32) -> Dtype {
        self.package.values[id as usize].dtype
    }

    fn full(&self, shape: &[u32]) -> Vec<i64> {
        let mut dims = Vec::with_capacity(shape.len() + 1);
        dims.push(self.b);
        dims.extend(shape.iter().map(|&d| i64::from(d)));
        dims
    }

    fn get(&self, id: u32) -> Lowering<Val> {
        self.env
            .get(&id)
            .copied()
            .ok_or_else(|| Refused(format!("value {id} is read before anything defines it")))
    }

    fn set(&mut self, id: u32, v: Val) -> Lowering<()> {
        let want = self.full(&self.shape(id));
        let got = self.f.dims(v).to_vec();
        if got != want {
            return refuse(format!("value {id} lowered to {got:?}, declared {want:?}"));
        }
        let e = elem(self.dtype(id))?;
        if self.f.elem(v) != e {
            return refuse(format!(
                "value {id} lowered to {:?}, declared {e:?}",
                self.f.elem(v)
            ));
        }
        self.env.insert(id, v);
        Ok(())
    }

    /// `v` (one lane's value is `[B, ..]` of any shape with the same count)
    /// reshaped to `shape`.
    fn reshaped(&mut self, v: Val, shape: &[u32]) -> Lowering<Val> {
        let dims = self.full(shape);
        Ok(self.f.reshape(v, &dims)?)
    }

    /// The interpreter's `pick`: an operand of one element stands for every
    /// lane of the result; otherwise the operand is the result's shape.
    fn fit(&mut self, v: Val, shape: &[u32]) -> Lowering<Val> {
        let want = self.full(shape);
        let have = self.f.dims(v).to_vec();
        if have == want {
            return Ok(v);
        }
        let per_lane: i64 = have[1..].iter().product();
        let wanted: i64 = want[1..].iter().product();
        if per_lane == 1 {
            let flat = self.f.reshape(v, &[self.b])?;
            return Ok(self.f.broadcast(flat, &want, &[0])?);
        }
        if per_lane == wanted {
            return Ok(self.f.reshape(v, &want)?);
        }
        refuse(format!("an operand of {have:?} does not fit {want:?}"))
    }

    fn arg(&mut self, op: &LaunchOp, i: usize, shape: &[u32]) -> Lowering<Val> {
        let id = *op
            .args
            .get(i)
            .ok_or_else(|| Refused(format!("operand {i} is missing")))?;
        let v = self.get(id)?;
        self.fit(v, shape)
    }

    fn raw_arg(&self, op: &LaunchOp, i: usize) -> Lowering<(u32, Val)> {
        let id = *op
            .args
            .get(i)
            .ok_or_else(|| Refused(format!("operand {i} is missing")))?;
        Ok((id, self.get(id)?))
    }

    // ------------------------------------------------------------- constants

    fn constant(&mut self, dtype: Dtype, bits: u32) -> Lowering<Val> {
        let b = self.b;
        Ok(match dtype {
            Dtype::F32 => self
                .f
                .const_f(Elem::F32, f64::from(f32::from_bits(bits)), &[b]),
            Dtype::I32 => self.f.const_i(Elem::I32, i64::from(bits as i32), &[b]),
            Dtype::U32 => self.f.const_i(Elem::U32, i64::from(bits), &[b]),
            Dtype::Bool => self.f.const_i(Elem::Pred, i64::from(bits != 0), &[b]),
            other => return refuse(format!("a {other:?} constant")),
        })
    }

    fn splat_f(&mut self, like: Val, x: f32) -> Val {
        self.f.like_f(like, f64::from(x))
    }

    fn u64s(&mut self, x: u64, dims: &[i64]) -> Val {
        let ty = Ty::new(Elem::U64, dims);
        let mut attr = String::new();
        let _ = write!(attr, "value = dense<{x}> : {ty}");
        self.f.op_n("constant", &[], &attr, vec![ty])[0]
    }

    // ---------------------------------------------------------------- packing

    /// Lane words `[at, at + words)` of the pack as `dtype` `[B, numel]`.
    fn unpack(&mut self, pack: Val, at: usize, dtype: Dtype, n: usize) -> Lowering<Val> {
        let w = words(dtype, n);
        let b = self.b;
        let cut = self
            .f
            .slice(pack, &[0, at as i64], &[b, (at + w) as i64], &[1, 1])?;
        Ok(match dtype {
            Dtype::U32 => cut,
            Dtype::F32 => self.f.bitcast(cut, Elem::F32)?,
            Dtype::I32 => self.f.bitcast(cut, Elem::I32)?,
            Dtype::Bool => {
                let dims = [b, w as i64, 32];
                let spread = self.f.broadcast(cut, &dims, &[0, 1])?;
                let shift = self.f.iota(Elem::U32, &dims, 2);
                let moved = self.f.shr(spread, shift)?;
                let one = self.f.const_i(Elem::U32, 1, &dims);
                let bit = self.f.and(moved, one)?;
                let flat = self.f.reshape(bit, &[b, (w * 32) as i64])?;
                let flat = self.f.slice(flat, &[0, 0], &[b, n as i64], &[1, 1])?;
                let zero = self.f.const_i(Elem::U32, 0, &[b, n as i64]);
                self.f.compare(Cmp::Ne, flat, zero)?
            }
            other => return refuse(format!("a {other:?} slot")),
        })
    }

    /// `v` (`dtype`, `numel` per lane) as packed words `u32 [B, words]`.
    fn pack(&mut self, v: Val, dtype: Dtype, n: usize) -> Lowering<Val> {
        let b = self.b;
        let flat = self.f.reshape(v, &[b, n as i64])?;
        Ok(match dtype {
            Dtype::U32 => flat,
            Dtype::F32 | Dtype::I32 => self.f.bitcast(flat, Elem::U32)?,
            Dtype::Bool => {
                let w = n.div_ceil(32);
                let bits = self.f.convert(flat, Elem::U32);
                let zero = self.f.const_i(Elem::U32, 0, &[]);
                let padded = self
                    .f
                    .pad(bits, zero, &[0, 0], &[0, (w * 32 - n) as i64], &[0, 0])?;
                let dims = [b, w as i64, 32];
                let grid = self.f.reshape(padded, &dims)?;
                let shift = self.f.iota(Elem::U32, &dims, 2);
                let placed = self.f.shl(grid, shift)?;
                self.f.reduce(placed, &[2], Fold::Sum)?
            }
            other => return refuse(format!("a {other:?} put")),
        })
    }

    /// The lane's `rows` readout rows starting at the row its slot names, as
    /// the intrinsic's value.
    fn read_rows(
        &mut self,
        pack: Val,
        at: usize,
        plane: Val,
        intrinsic: IntrinsicId,
        rows: u32,
        shape: &[u32],
    ) -> Lowering<Val> {
        let b = self.b;
        let width = self.f.dims(plane)[1];
        let first = self.unpack(pack, at, Dtype::I32, 1)?; // [B, 1]
        let r = i64::from(rows);
        let first = self.f.broadcast(first, &[b, r], &[0, 1])?;
        let step = self.f.iota(Elem::I32, &[b, r], 1);
        let ids = self.f.add(first, step)?;
        let ids = self.f.reshape(ids, &[b * r])?;
        let picked = self.f.take_rows(plane, ids)?; // [B*r, width]
        if intrinsic == IntrinsicId::MtpDrafts {
            let grid = self.f.reshape(picked, &[b, r, width])?;
            let tokens = self.argmax_f32(grid)?;
            return self.reshaped(tokens, shape);
        }
        self.reshaped(picked, shape)
    }

    // ---------------------------------------------------------------- helpers

    fn is_nan(&mut self, x: Val) -> Lowering<Val> {
        Ok(self.f.compare(Cmp::Ne, x, x)?)
    }

    /// `cond ? a : b` with `a`, `b` float scalars.
    fn select_f(&mut self, cond: Val, a: f32, b: f32) -> Lowering<Val> {
        let dims = self.f.dims(cond).to_vec();
        let a = self.f.const_f(Elem::F32, f64::from(a), &dims);
        let b = self.f.const_f(Elem::F32, f64::from(b), &dims);
        Ok(self.f.select(cond, a, b)?)
    }

    /// Replaces NaN lanes of `x` with `with`.
    fn denan(&mut self, x: Val, with: f32) -> Lowering<Val> {
        let nan = self.is_nan(x)?;
        let w = self.splat_f(x, with);
        Ok(self.f.select(nan, w, x)?)
    }

    /// Argmax along the last axis: NaN lanes lose to every other lane, ties
    /// go to the lowest index, an all-NaN row answers 0 (`argmax_row`).
    fn argmax_f32(&mut self, x: Val) -> Lowering<Val> {
        let dims = self.f.dims(x).to_vec();
        let last = (dims.len() - 1) as i64;
        let valid = {
            let nan = self.is_nan(x)?;
            let not = self.f.not(nan);
            self.f.convert(not, Elem::I32)
        };
        let clean = self.denan(x, f32::NEG_INFINITY)?;
        let index = self.f.iota(Elem::I32, &dims, last);
        let lo = self.f.const_f(Elem::F32, f64::NEG_INFINITY, &[]);
        let far = self.f.const_i(Elem::I32, i64::from(i32::MAX), &[]);
        let none = self.f.const_i(Elem::I32, 0, &[]);
        let out = self.f.reduce_with(
            &[clean, index, valid],
            &[lo, far, none],
            &[last],
            |f, a, b| {
                let (av, ai, ak, bv, bi, bk) = (a[0], a[1], a[2], b[0], b[1], b[2]);
                let valid_gt = f.compare(Cmp::Gt, ak, bk)?;
                let valid_eq = f.compare(Cmp::Eq, ak, bk)?;
                let gt = f.compare(Cmp::Gt, av, bv)?;
                let eq = f.compare(Cmp::Eq, av, bv)?;
                let lt = f.compare(Cmp::Lt, ai, bi)?;
                let tie = f.and(eq, lt)?;
                let wins = f.or(gt, tie)?;
                let wins = f.and(valid_eq, wins)?;
                let take = f.or(valid_gt, wins)?;
                Ok(vec![
                    f.select(take, av, bv)?,
                    f.select(take, ai, bi)?,
                    f.select(take, ak, bk)?,
                ])
            },
        )?;
        Ok(out[1])
    }

    /// Integer argmax along the last axis: the maximum, lowest index on ties.
    fn argmax_int(&mut self, x: Val) -> Lowering<Val> {
        let dims = self.f.dims(x).to_vec();
        let last = (dims.len() - 1) as i64;
        let e = self.f.elem(x);
        let index = self.f.iota(Elem::I32, &dims, last);
        let lo = match e {
            Elem::U32 => self.f.const_i(e, 0, &[]),
            _ => self.f.const_i(e, i64::from(i32::MIN), &[]),
        };
        let far = self.f.const_i(Elem::I32, i64::from(i32::MAX), &[]);
        let out = self
            .f
            .reduce_with(&[x, index], &[lo, far], &[last], |f, a, b| {
                let gt = f.compare(Cmp::Gt, a[0], b[0])?;
                let eq = f.compare(Cmp::Eq, a[0], b[0])?;
                let lt = f.compare(Cmp::Lt, a[1], b[1])?;
                let tie = f.and(eq, lt)?;
                let take = f.or(gt, tie)?;
                Ok(vec![
                    f.select(take, a[0], b[0])?,
                    f.select(take, a[1], b[1])?,
                ])
            })?;
        Ok(out[1])
    }

    /// Inclusive scan along the last axis (Hillis–Steele: log₂ n shifted
    /// combines, which XLA fuses; a `reduce_window` scan is quadratic).
    fn scan(&mut self, x: Val, fold: Fold) -> Lowering<Val> {
        let dims = self.f.dims(x).to_vec();
        let r = dims.len();
        let n = dims[r - 1];
        let e = self.f.elem(x);
        let identity = match fold {
            Fold::Prod => self.f.const_f(e, 1.0, &[]),
            _ => self.f.const_f(e, 0.0, &[]),
        };
        let mut y = x;
        let mut d = 1i64;
        while d < n {
            let mut lo = vec![0; r];
            let mut hi = dims.clone();
            hi[r - 1] = n - d;
            let kept = self.f.slice(y, &lo, &hi, &vec![1; r])?;
            lo[r - 1] = d;
            let shifted = self.f.pad(kept, identity, &lo, &vec![0; r], &vec![0; r])?;
            y = match fold {
                Fold::Prod => self.f.mul(y, shifted)?,
                _ => self.f.add(y, shifted)?,
            };
            d *= 2;
        }
        Ok(y)
    }

    /// Exclusive prefix sum along the last axis.
    fn scan_exclusive(&mut self, x: Val) -> Lowering<Val> {
        let dims = self.f.dims(x).to_vec();
        let r = dims.len();
        let n = dims[r - 1];
        let mut hi = dims.clone();
        hi[r - 1] = n - 1;
        let kept = self.f.slice(x, &vec![0; r], &hi, &vec![1; r])?;
        let zero = self.f.const_f(Elem::F32, 0.0, &[]);
        let mut lo = vec![0; r];
        lo[r - 1] = 1;
        let shifted = self.f.pad(kept, zero, &lo, &vec![0; r], &vec![0; r])?;
        self.scan(shifted, Fold::Sum)
    }

    /// `x` (f32) sorted descending along the last axis in the interpreter's
    /// order (`sort_desc_order`): NaN last, -0 equal to +0, ties by the lower
    /// index. Returns the sorted values and their source indices (i32).
    fn sort_desc(&mut self, x: Val) -> Lowering<(Val, Val)> {
        let dims = self.f.dims(x).to_vec();
        let last = (dims.len() - 1) as i64;
        let key = self.order_key(x)?;
        let index = self.f.iota(Elem::I32, &dims, last);
        // A stable sort on the key alone: equal keys keep their index order.
        let sorted = self.f.sort(&[key, index, x], last, true, |f, a, b| {
            f.compare(Cmp::Gt, a[0], b[0])
        })?;
        Ok((sorted[2], sorted[1]))
    }

    /// The order keys of `x` sorted descending (stably) along the last
    /// axis, with their source indices when `indexed`.
    fn sort_keys(&mut self, key: Val, indexed: bool) -> Lowering<(Val, Option<Val>)> {
        let dims = self.f.dims(key).to_vec();
        let last = (dims.len() - 1) as i64;
        if !indexed {
            let sorted = self
                .f
                .sort(&[key], last, true, |f, a, b| f.compare(Cmp::Gt, a[0], b[0]))?;
            return Ok((sorted[0], None));
        }
        let index = self.f.iota(Elem::I32, &dims, last);
        let sorted = self.f.sort(&[key, index], last, true, |f, a, b| {
            f.compare(Cmp::Gt, a[0], b[0])
        })?;
        Ok((sorted[0], Some(sorted[1])))
    }

    /// The value an order key stands for (-0 read as +0, NaN as a NaN).
    fn key_value(&mut self, key: Val) -> Lowering<Val> {
        let flip = self.f.like_i(key, 0x7FFF_FFFF);
        let flipped = self.f.xor(key, flip)?;
        let zero = self.f.like_i(key, 0);
        let negative = self.f.compare(Cmp::Lt, key, zero)?;
        let bits = self.f.select(negative, flipped, key)?;
        Ok(self.f.bitcast(bits, Elem::F32)?)
    }

    /// Per row of the sorted `[B, R, n]` `plane`, the element at position
    /// `at` (`[B, R]` i32; out of range reads `fill`).
    fn at_position(&mut self, plane: Val, at: Val, fill: i64) -> Lowering<Val> {
        let grid = self.f.dims(plane).to_vec();
        let at = self.f.broadcast(at, &grid, &[0, 1])?;
        let col = self.f.iota(Elem::I32, &grid, 2);
        let here = self.f.compare(Cmp::Eq, col, at)?;
        let floor = self.f.like_i(plane, fill);
        let pick = self.f.select(here, plane, floor)?;
        Ok(self.f.reduce(pick, &[2], Fold::Max)?)
    }

    /// An i32 whose order is the interpreter's descending-sort order of the
    /// f32 `x`: monotone in the value, -0 and +0 one key, NaN below -inf.
    fn order_key(&mut self, x: Val) -> Lowering<Val> {
        let zero = self.splat_f(x, 0.0);
        let is_zero = self.f.compare(Cmp::Eq, x, zero)?;
        let x = self.f.select(is_zero, zero, x)?;
        let bits = self.f.bitcast(x, Elem::I32)?;
        let flip = self.f.like_i(bits, 0x7FFF_FFFF);
        let flipped = self.f.xor(bits, flip)?;
        let nought = self.f.like_i(bits, 0);
        let negative = self.f.compare(Cmp::Lt, bits, nought)?;
        let key = self.f.select(negative, flipped, bits)?;
        let nan = self.is_nan(x)?;
        let bottom = self.f.like_i(bits, i64::from(i32::MIN));
        Ok(self.f.select(nan, bottom, key)?)
    }

    /// `x` of `[B, lead.., n]` as `[B, rows, n]`, with `rows` the product of
    /// the lead axes (1 for a vector).
    fn rows_of(&mut self, x: Val) -> Lowering<(Val, i64, i64)> {
        let dims = self.f.dims(x).to_vec();
        let n = *dims.last().unwrap_or(&1);
        let rows: i64 = if dims.len() <= 2 {
            1
        } else {
            dims[1..dims.len() - 1].iter().product()
        };
        let v = self.f.reshape(x, &[self.b, rows, n])?;
        Ok((v, rows, n))
    }

    /// An index operand as i32 `[B, k]` plus whether each lane is in
    /// `0..bound`, and the index clamped into range.
    fn index(&mut self, idx: Val, bound: i64) -> Lowering<(Val, Val)> {
        let b = self.b;
        let dims = self.f.dims(idx).to_vec();
        let k: i64 = dims[1..].iter().product();
        let flat = self.f.reshape(idx, &[b, k])?;
        let valid = match self.f.elem(flat) {
            Elem::U32 => {
                let top = self.f.const_i(Elem::U32, bound, &[b, k]);
                self.f.compare(Cmp::Lt, flat, top)?
            }
            Elem::I32 => {
                let top = self.f.const_i(Elem::I32, bound, &[b, k]);
                let zero = self.f.const_i(Elem::I32, 0, &[b, k]);
                let under = self.f.compare(Cmp::Lt, flat, top)?;
                let over = self.f.compare(Cmp::Ge, flat, zero)?;
                self.f.and(under, over)?
            }
            other => return refuse(format!("an index of {other:?}")),
        };
        let as_i32 = match self.f.elem(flat) {
            Elem::U32 => self.f.bitcast(flat, Elem::I32)?,
            _ => flat,
        };
        let zero = self.f.const_i(Elem::I32, 0, &[b, k]);
        let safe = self.f.select(valid, as_i32, zero)?;
        Ok((safe, valid))
    }

    fn zero_like(&mut self, v: Val) -> Val {
        let ty = self.f.ty(v).clone();
        if ty.elem.is_float() {
            self.f.const_f(ty.elem, 0.0, &ty.dims)
        } else {
            self.f.const_i(ty.elem, 0, &ty.dims)
        }
    }

    /// splitmix64 over u64 lanes.
    fn splitmix(&mut self, mut x: Val) -> Lowering<Val> {
        let dims = self.f.dims(x).to_vec();
        for round in rng::RNG_FORMULA.splitmix64_rounds {
            let s = self.u64s(u64::from(round.xor_shift), &dims);
            let sh = self.f.shr(x, s)?;
            x = self.f.xor(x, sh)?;
            if let Some(m) = round.multiplier {
                let c = self.u64s(m, &dims);
                x = self.f.mul(x, c)?;
            }
        }
        Ok(x)
    }

    /// `hash_uniform(seed[b], lane[b, j])` for u64 `seed [B]` and u64
    /// `lane [B, n]`.
    fn uniform(&mut self, seed: Val, lane: Val) -> Lowering<Val> {
        let dims = self.f.dims(lane).to_vec();
        let seed = self.f.broadcast(seed, &dims, &[0])?;
        let bias = self.u64s(rng::RNG_FORMULA.lane_index_bias, &dims);
        let biased = self.f.add(lane, bias)?;
        let stride = self.u64s(rng::RNG_FORMULA.lane_stride, &dims);
        let step = self.f.mul(stride, biased)?;
        let x = self.f.add(seed, step)?;
        let h = self.splitmix(x)?;
        let shift = self.u64s(u64::from(rng::RNG_FORMULA.uniform_mantissa_shift), &dims);
        let top = self.f.shr(h, shift)?;
        let bits = self.f.convert(top, Elem::U32);
        let asf = self.f.convert(bits, Elem::F32);
        let mid = self.f.const_f(
            Elem::F32,
            f64::from(rng::RNG_FORMULA.uniform_midpoint),
            &dims,
        );
        let raw = self.f.add(asf, mid)?;
        let scale = 1.0f32 / (1u32 << rng::RNG_FORMULA.uniform_mantissa_bits) as f32;
        let unit = self.f.const_f(Elem::F32, f64::from(scale), &dims);
        let raw = self.f.mul(raw, unit)?;
        let cap = self
            .f
            .const_f(Elem::F32, f64::from(rng::UNIFORM_MAX), &dims);
        let under = self.f.compare(Cmp::Lt, raw, cap)?;
        Ok(self.f.select(under, raw, cap)?)
    }

    /// `rng_lanes(seed[b], n, kind)` as `[B, n]` f32.
    fn draw(&mut self, seed: Val, n: i64, kind: RngKind) -> Lowering<Val> {
        let b = self.b;
        let j = self.f.iota(Elem::U64, &[b, n], 1);
        Ok(match kind {
            RngKind::Uniform => self.uniform(seed, j)?,
            RngKind::Gumbel => {
                let u = self.uniform(seed, j)?;
                let l = self.f.log(u);
                let nl = self.f.neg(l);
                let ll = self.f.log(nl);
                self.f.neg(ll)
            }
            RngKind::Normal => {
                let two = self.u64s(u64::from(eta_ir::rng::NORMAL_PAIR_STRIDE), &[b, n]);
                let lane = self.f.mul(j, two)?;
                let one = self.u64s(1, &[b, n]);
                let next = self.f.add(lane, one)?;
                let u0 = self.uniform(seed, lane)?;
                let u1 = self.uniform(seed, next)?;
                let l = self.f.log(u0);
                let m2 = self.f.const_f(Elem::F32, -2.0, &[b, n]);
                let r = self.f.mul(m2, l)?;
                let radius = self.f.sqrt(r);
                let tau = self
                    .f
                    .const_f(Elem::F32, f64::from(eta_ir::rng::NORMAL_TWO_PI), &[b, n]);
                let a = self.f.mul(tau, u1)?;
                let c = self.f.cos(a);
                self.f.mul(radius, c)?
            }
        })
    }

    // -------------------------------------------------------------------- ops

    #[allow(clippy::too_many_lines)]
    fn op(&mut self, op: &LaunchOp) -> Lowering<()> {
        if op.tag == tags::SINK_CALL {
            // A library sink (`lora`, `metal.discard`): the interpreter does
            // nothing for it, and neither does the device.
            return Ok(());
        }
        let result = op.result_id;
        if result as usize >= self.package.values.len() {
            return refuse(format!("result {result} is past the package's values"));
        }
        let shape = self.shape(result);
        let dt =
            |l: &Self, i: usize| -> Dtype { op.args.get(i).map_or(Dtype::F32, |&a| l.dtype(a)) };
        match op.tag {
            tags::EXP
            | tags::LOG
            | tags::SIN
            | tags::COS
            | tags::SQRT
            | tags::RSQRT
            | tags::RECIP => {
                if dt(self, 0) != Dtype::F32 {
                    return refuse("a float map over a non-float operand");
                }
                let x = self.arg(op, 0, &shape)?;
                let y = match op.tag {
                    tags::EXP => self.f.exp(x),
                    tags::LOG => self.f.log(x),
                    tags::SIN => self.f.sin(x),
                    tags::COS => self.f.cos(x),
                    tags::SQRT => self.f.sqrt(x),
                    tags::RSQRT => {
                        let s = self.f.sqrt(x);
                        let one = self.splat_f(s, 1.0);
                        self.f.div(one, s)?
                    }
                    _ => {
                        let one = self.splat_f(x, 1.0);
                        self.f.div(one, x)?
                    }
                };
                self.set(result, y)
            }
            tags::NEG => {
                let x = self.arg(op, 0, &shape)?;
                let y = match dt(self, 0) {
                    Dtype::F32 => self.f.neg(x),
                    Dtype::I32 | Dtype::U32 => {
                        let z = self.zero_like(x);
                        self.f.sub(z, x)?
                    }
                    _ => return refuse("neg on bool"),
                };
                self.set(result, y)
            }
            tags::ABS => {
                let x = self.arg(op, 0, &shape)?;
                let y = match dt(self, 0) {
                    Dtype::F32 => self.f.abs(x),
                    Dtype::I32 => {
                        let z = self.zero_like(x);
                        let n = self.f.sub(z, x)?;
                        let neg = self.f.compare(Cmp::Lt, x, z)?;
                        self.f.select(neg, n, x)?
                    }
                    _ => x,
                };
                self.set(result, y)
            }
            tags::SIGN => {
                let x = self.arg(op, 0, &shape)?;
                let y = match dt(self, 0) {
                    Dtype::F32 => {
                        let z = self.zero_like(x);
                        let pos = self.f.compare(Cmp::Gt, x, z)?;
                        let neg = self.f.compare(Cmp::Lt, x, z)?;
                        let lo = self.select_f(neg, -1.0, 0.0)?;
                        let one = self.splat_f(x, 1.0);
                        self.f.select(pos, one, lo)?
                    }
                    Dtype::I32 => {
                        let z = self.zero_like(x);
                        let one = self.f.like_i(x, 1);
                        let m1 = self.f.like_i(x, -1);
                        let pos = self.f.compare(Cmp::Gt, x, z)?;
                        let neg = self.f.compare(Cmp::Lt, x, z)?;
                        let lo = self.f.select(neg, m1, z)?;
                        self.f.select(pos, one, lo)?
                    }
                    Dtype::U32 => {
                        let z = self.zero_like(x);
                        let nz = self.f.compare(Cmp::Ne, x, z)?;
                        self.f.convert(nz, Elem::U32)
                    }
                    _ => return refuse("sign on bool"),
                };
                self.set(result, y)
            }
            tags::CAST => {
                let x = self.arg(op, 0, &shape)?;
                let y = self.cast(x, dt(self, 0), op.dtype)?;
                self.set(result, y)
            }
            tags::ADD
            | tags::SUB
            | tags::MUL
            | tags::DIV
            | tags::REM
            | tags::MAX_ELEM
            | tags::MIN_ELEM => {
                let d = dt(self, 0);
                let a = self.arg(op, 0, &shape)?;
                let b = self.arg(op, 1, &shape)?;
                let y = self.arith(op.tag, d, a, b)?;
                self.set(result, y)
            }
            tags::GT | tags::GE | tags::EQ | tags::NE | tags::LT | tags::LE => {
                if dt(self, 0) == Dtype::Bool {
                    return refuse("an ordered comparison of bools");
                }
                let a = self.arg(op, 0, &shape)?;
                let b = self.arg(op, 1, &shape)?;
                let dir = match op.tag {
                    tags::GT => Cmp::Gt,
                    tags::GE => Cmp::Ge,
                    tags::EQ => Cmp::Eq,
                    tags::NE => Cmp::Ne,
                    tags::LT => Cmp::Lt,
                    _ => Cmp::Le,
                };
                let y = self.f.compare(dir, a, b)?;
                self.set(result, y)
            }
            tags::AND | tags::OR => {
                if dt(self, 0) != Dtype::Bool || dt(self, 1) != Dtype::Bool {
                    return refuse("and/or on non-bool");
                }
                let a = self.arg(op, 0, &shape)?;
                let b = self.arg(op, 1, &shape)?;
                let y = if op.tag == tags::AND {
                    self.f.and(a, b)?
                } else {
                    self.f.or(a, b)?
                };
                self.set(result, y)
            }
            tags::NOT => {
                if dt(self, 0) != Dtype::Bool {
                    return refuse("not on non-bool");
                }
                let a = self.arg(op, 0, &shape)?;
                let y = self.f.not(a);
                self.set(result, y)
            }
            tags::SELECT => {
                if dt(self, 0) != Dtype::Bool || dt(self, 1) != dt(self, 2) {
                    return refuse("select's condition or arms");
                }
                let c = self.arg(op, 0, &shape)?;
                let a = self.arg(op, 1, &shape)?;
                let b = self.arg(op, 2, &shape)?;
                let y = self.f.select(c, a, b)?;
                self.set(result, y)
            }
            tags::REDUCE_SUM | tags::REDUCE_MAX | tags::REDUCE_MIN => {
                let (_, x) = self.raw_arg(op, 0)?;
                let d = dt(self, 0);
                let (x, _, _) = self.rows_of(x)?;
                let y = match (op.tag, d) {
                    (tags::REDUCE_SUM, Dtype::F32 | Dtype::I32 | Dtype::U32) => {
                        self.f.reduce(x, &[2], Fold::Sum)?
                    }
                    (tags::REDUCE_MAX, Dtype::F32) => {
                        let c = self.denan(x, f32::NEG_INFINITY)?;
                        self.f.reduce(c, &[2], Fold::Max)?
                    }
                    (tags::REDUCE_MIN, Dtype::F32) => {
                        let c = self.denan(x, f32::INFINITY)?;
                        self.f.reduce(c, &[2], Fold::Min)?
                    }
                    (tags::REDUCE_MAX, Dtype::I32 | Dtype::U32) => {
                        self.f.reduce(x, &[2], Fold::Max)?
                    }
                    (tags::REDUCE_MIN, Dtype::I32 | Dtype::U32) => {
                        self.f.reduce(x, &[2], Fold::Min)?
                    }
                    _ => return refuse(format!("a reduction over {d:?}")),
                };
                let y = self.reshaped(y, &shape)?;
                self.set(result, y)
            }
            tags::REDUCE_ARGMAX => {
                let (_, x) = self.raw_arg(op, 0)?;
                let d = dt(self, 0);
                let (x, _, _) = self.rows_of(x)?;
                let y = match d {
                    Dtype::F32 => self.argmax_f32(x)?,
                    Dtype::I32 | Dtype::U32 => self.argmax_int(x)?,
                    _ => return refuse("argmax over bools"),
                };
                let y = self.reshaped(y, &shape)?;
                self.set(result, y)
            }
            tags::CUMSUM | tags::CUMPROD => {
                if dt(self, 0) != Dtype::F32 {
                    // The interpreter scans in f32 and hands back f32 lanes
                    // whatever the declared dtype: nothing to agree with.
                    return refuse("a scan over integers");
                }
                let (_, x) = self.raw_arg(op, 0)?;
                let (x, _, _) = self.rows_of(x)?;
                let fold = if op.tag == tags::CUMSUM {
                    Fold::Sum
                } else {
                    Fold::Prod
                };
                let y = self.scan(x, fold)?;
                let y = self.reshaped(y, &shape)?;
                self.set(result, y)
            }
            tags::BROADCAST => {
                let (id, x) = self.raw_arg(op, 0)?;
                let src = self.shape(id);
                if src.len() > shape.len() {
                    return refuse("a broadcast to a lower rank");
                }
                // The interpreter aligns the source's axes to the target's
                // leading axes (`broadcast_value`).
                let mut padded: Vec<u32> = src.clone();
                padded.resize(shape.len(), 1);
                for (s, t) in padded.iter().zip(&shape) {
                    if *s != 1 && s != t {
                        return refuse(format!("a broadcast of {src:?} to {shape:?}"));
                    }
                }
                let staged = self.reshaped(x, &padded)?;
                let dims = self.full(&shape);
                let map: Vec<i64> = (0..dims.len() as i64).collect();
                let y = self.f.broadcast(staged, &dims, &map)?;
                self.set(result, y)
            }
            tags::RESHAPE | tags::KERNEL_CALL => {
                if op.tag == tags::KERNEL_CALL && op.args.len() != 1 {
                    return refuse("a kernel call that is not the identity boundary");
                }
                let (id, x) = self.raw_arg(op, 0)?;
                if self.dtype(id) != self.dtype(result) || numel(&self.shape(id)) != numel(&shape) {
                    return refuse("a reshape that changes the count or the dtype");
                }
                let y = self.reshaped(x, &shape)?;
                self.set(result, y)
            }
            tags::TRANSPOSE => {
                let (id, x) = self.raw_arg(op, 0)?;
                if self.shape(id).len() != 2 {
                    return refuse("a transpose of other than a matrix");
                }
                let y = self.f.transpose(x, &[0, 2, 1])?;
                self.set(result, y)
            }
            tags::SORT_DESC | tags::TOP_K => {
                let (id, x) = self.raw_arg(op, 0)?;
                if self.dtype(id) != Dtype::F32 {
                    return refuse("an ordering of non-floats");
                }
                let src = self.shape(id);
                let (grid, _, n) = if op.tag == tags::SORT_DESC {
                    // The interpreter orders the whole value as one row.
                    let n = numel(&src) as i64;
                    let v = self.f.reshape(x, &[self.b, 1, n])?;
                    (v, 1, n)
                } else {
                    self.rows_of(x)?
                };
                let k = if op.tag == tags::TOP_K {
                    i64::from(op.imm)
                } else {
                    n
                };
                if k < 1 || k > n {
                    return refuse(format!("top-{k} of {n}"));
                }
                let (values, index) = self.sort_desc(grid)?;
                let d = self.f.dims(values).to_vec();
                let values = self
                    .f
                    .slice(values, &[0, 0, 0], &[d[0], d[1], k], &[1, 1, 1])?;
                let index = self
                    .f
                    .slice(index, &[0, 0, 0], &[d[0], d[1], k], &[1, 1, 1])?;
                let index = self.f.bitcast(index, Elem::U32)?;
                let values = self.reshaped(values, &shape)?;
                let second = self.shape(result + 1);
                let index = self.reshaped(index, &second)?;
                self.set(result, values)?;
                self.set(result + 1, index)
            }
            tags::MATMUL => {
                let (ia, a) = self.raw_arg(op, 0)?;
                let (ib, b) = self.raw_arg(op, 1)?;
                let (sa, sb) = (self.shape(ia), self.shape(ib));
                if sa.len() != 2 || sb.len() != 2 || sa[1] != sb[0] {
                    return refuse("matmul of other than conformant matrices");
                }
                if self.dtype(ia) != Dtype::F32 || self.dtype(ib) != Dtype::F32 {
                    return refuse("matmul of non-floats");
                }
                let y = self
                    .f
                    .dot_general(a, b, &[0], &[0], &[2], &[1], Elem::F32)?;
                self.set(result, y)
            }
            tags::PIVOT_THRESHOLD => self.pivot(op, &shape),
            tags::GATHER => {
                let (is, src) = self.raw_arg(op, 0)?;
                let (_, idx) = self.raw_arg(op, 1)?;
                let ss = self.shape(is);
                let n0 = i64::from(*ss.first().unwrap_or(&1));
                let rest: i64 = ss
                    .iter()
                    .skip(1)
                    .map(|&d| i64::from(d))
                    .product::<i64>()
                    .max(1);
                let b = self.b;
                let (safe, valid) = self.index(idx, n0)?;
                let k = self.f.dims(safe)[1];
                let table = self.f.reshape(src, &[b * n0, rest])?;
                let base = self.f.iota(Elem::I32, &[b, k], 0);
                let stride = self.f.const_i(Elem::I32, n0, &[b, k]);
                let base = self.f.mul(base, stride)?;
                let rows = self.f.add(base, safe)?;
                let rows = self.f.reshape(rows, &[b * k])?;
                let picked = self.f.take_rows(table, rows)?; // [B*k, rest]
                let picked = self.f.reshape(picked, &[b, k, rest])?;
                let keep = self.f.broadcast(valid, &[b, k, rest], &[0, 1])?;
                let zero = self.zero_like(picked);
                let y = self.f.select(keep, picked, zero)?;
                let y = self.reshaped(y, &shape)?;
                self.set(result, y)
            }
            tags::GATHER_ROW => {
                let (is, src) = self.raw_arg(op, 0)?;
                let (_, idx) = self.raw_arg(op, 1)?;
                let ss = self.shape(is);
                if ss.len() != 2 {
                    return refuse("gather_row of other than a matrix");
                }
                let (m, n) = (i64::from(ss[0]), i64::from(ss[1]));
                let b = self.b;
                let (safe, valid) = self.index(idx, n)?;
                if self.f.dims(safe)[1] != m {
                    return refuse("gather_row's index is not one per row");
                }
                let flat = self.f.reshape(src, &[b * m * n, 1])?;
                let row = self.f.iota(Elem::I32, &[b, m], 0);
                let bm = self.f.const_i(Elem::I32, m, &[b, m]);
                let row = self.f.mul(row, bm)?;
                let r = self.f.iota(Elem::I32, &[b, m], 1);
                let row = self.f.add(row, r)?;
                let nn = self.f.const_i(Elem::I32, n, &[b, m]);
                let at = self.f.mul(row, nn)?;
                let at = self.f.add(at, safe)?;
                let at = self.f.reshape(at, &[b * m])?;
                let picked = self.f.take_rows(flat, at)?;
                let picked = self.f.reshape(picked, &[b, m])?;
                let zero = self.zero_like(picked);
                let y = self.f.select(valid, picked, zero)?;
                self.set(result, y)
            }
            tags::SCATTER_ADD | tags::SCATTER_SET => self.scatter(op, &shape),
            tags::IOTA => {
                let y = self.f.iota(Elem::U32, &[self.b, i64::from(op.imm)], 1);
                self.set(result, y)
            }
            tags::MASK_APPLY_PACKED => {
                let (_, x) = self.raw_arg(op, 0)?;
                let (im, mask) = self.raw_arg(op, 1)?;
                if dt(self, 0) != Dtype::F32 || self.dtype(im) != Dtype::U32 {
                    return refuse("mask_apply of other than f32 logits and u32 words");
                }
                let n = i64::from(*shape.last().unwrap_or(&1));
                let w = numel(&self.shape(im)) as i64;
                let b = self.b;
                let mask = self.f.reshape(mask, &[b, w])?;
                let dims = [b, w, 32];
                let spread = self.f.broadcast(mask, &dims, &[0, 1])?;
                let shift = self.f.iota(Elem::U32, &dims, 2);
                let moved = self.f.shr(spread, shift)?;
                let one = self.f.const_i(Elem::U32, 1, &dims);
                let bit = self.f.and(moved, one)?;
                let flat = self.f.reshape(bit, &[b, w * 32])?;
                let cols = if w * 32 >= n {
                    self.f.slice(flat, &[0, 0], &[b, n], &[1, 1])?
                } else {
                    let zero = self.f.const_i(Elem::U32, 0, &[]);
                    self.f.pad(flat, zero, &[0, 0], &[0, n - w * 32], &[0, 0])?
                };
                let zero = self.f.const_i(Elem::U32, 0, &[b, n]);
                let on = self.f.compare(Cmp::Ne, cols, zero)?;
                let full = self.full(&shape);
                let last = (full.len() - 1) as i64;
                let on = self.f.broadcast(on, &full, &[0, last])?;
                let x = self.fit(x, &shape)?;
                let ninf = self.splat_f(x, f32::NEG_INFINITY);
                let y = self.f.select(on, x, ninf)?;
                self.set(result, y)
            }
            tags::CAUSAL_MASK | tags::SLIDING_WINDOW_MASK | tags::SINK_WINDOW_MASK => {
                let (ip, pos) = self.raw_arg(op, 0)?;
                if self.dtype(ip) != Dtype::U32 {
                    return refuse("structured mask positions that are not u32");
                }
                let full = self.full(&shape);
                let r = full.len();
                let map: Vec<i64> = (0..(r - 1) as i64).collect();
                let pos = self.f.broadcast(pos, &full, &map)?;
                let key = self.f.iota(Elem::U32, &full, (r - 1) as i64);
                let mut allowed = self.f.compare(Cmp::Le, key, pos)?;
                if op.tag != tags::CAUSAL_MASK {
                    let window = if op.tag == tags::SLIDING_WINDOW_MASK {
                        op.imm2
                    } else {
                        op.imm3
                    };
                    let k64 = self.f.convert(key, Elem::U64);
                    let p64 = self.f.convert(pos, Elem::U64);
                    let wv = self.u64s(u64::from(window), &full);
                    let reach = self.f.add(k64, wv)?;
                    let cap = self.u64s(u64::from(u32::MAX), &full);
                    let reach = self.f.min(reach, cap)?;
                    let recent = self.f.compare(Cmp::Gt, reach, p64)?;
                    let keep = if op.tag == tags::SLIDING_WINDOW_MASK {
                        recent
                    } else {
                        let sink = self.f.const_i(Elem::U32, i64::from(op.imm2), &full);
                        let early = self.f.compare(Cmp::Lt, key, sink)?;
                        self.f.or(early, recent)?
                    };
                    allowed = self.f.and(allowed, keep)?;
                }
                self.set(result, allowed)
            }
            tags::RNG | tags::RNG_KEYED => {
                let n = numel(&shape) as i64;
                let seed = if op.tag == tags::RNG {
                    let s = rng::seed_eff_stream(0, op.imm);
                    self.u64s(s, &[self.b])
                } else {
                    let (is, state) = self.raw_arg(op, 0)?;
                    if self.dtype(is) != Dtype::U32 || numel(&self.shape(is)) < 1 {
                        return refuse("an rng state that is not u32 words");
                    }
                    let per = numel(&self.shape(is)) as i64;
                    let b = self.b;
                    let words = self.f.reshape(state, &[b, per])?;
                    let key = self.f.slice(words, &[0, 0], &[b, 1], &[1, 1])?;
                    let key = self.f.reshape(key, &[b])?;
                    let key = self.f.convert(key, Elem::U64);
                    let ctr = if per > 1 {
                        let c = self.f.slice(words, &[0, 1], &[b, 2], &[1, 1])?;
                        let c = self.f.reshape(c, &[b])?;
                        self.f.convert(c, Elem::U64)
                    } else {
                        self.u64s(0, &[b])
                    };
                    let shift = self.u64s(u64::from(rng::RNG_FORMULA.keyed_word_bits), &[b]);
                    let hi = self.f.shl(key, shift)?;
                    let joined = self.f.or(hi, ctr)?;
                    self.splitmix(joined)?
                };
                let y = self.draw(seed, n, op.rng_kind)?;
                let y = self.reshaped(y, &shape)?;
                self.set(result, y)
            }
            other => refuse(format!("`{}` has no StableHLO form here", op_name(other))),
        }
    }

    fn cast(&mut self, x: Val, from: Dtype, to: Dtype) -> Lowering<Val> {
        if from == to {
            return Ok(x);
        }
        let dims = self.f.dims(x).to_vec();
        Ok(match (from, to) {
            (_, Dtype::F32) => self.f.convert(x, Elem::F32),
            (Dtype::F32, Dtype::I32) => {
                // `as i32`: saturating, NaN to 0, toward zero.
                let lo = self.f.const_f(Elem::F32, -2_147_483_648.0, &dims);
                let hi = self.f.const_f(Elem::F32, 2_147_483_648.0, &dims);
                let under = self.f.compare(Cmp::Le, x, lo)?;
                let over = self.f.compare(Cmp::Ge, x, hi)?;
                let nan = self.is_nan(x)?;
                let zero = self.f.const_f(Elem::F32, 0.0, &dims);
                let bad = self.f.or(under, over)?;
                let bad = self.f.or(bad, nan)?;
                let safe = self.f.select(bad, zero, x)?;
                let v = self.f.convert(safe, Elem::I32);
                let min = self.f.const_i(Elem::I32, i64::from(i32::MIN), &dims);
                let max = self.f.const_i(Elem::I32, i64::from(i32::MAX), &dims);
                let v = self.f.select(under, min, v)?;
                self.f.select(over, max, v)?
            }
            (Dtype::F32, Dtype::U32) => {
                // `as u32`: saturating at 0 and u32::MAX, NaN to 0.
                let zero = self.f.const_f(Elem::F32, 0.0, &dims);
                let top = self.f.const_f(Elem::F32, 4_294_967_296.0, &dims);
                let half = self.f.const_f(Elem::F32, 2_147_483_648.0, &dims);
                let nan = self.is_nan(x)?;
                let low = self.f.compare(Cmp::Le, x, zero)?;
                let low = self.f.or(low, nan)?;
                let over = self.f.compare(Cmp::Ge, x, top)?;
                let upper = self.f.compare(Cmp::Ge, x, half)?;
                let safe = self.f.select(low, zero, x)?;
                let safe = self.f.select(over, zero, safe)?;
                let shifted = self.f.sub(safe, half)?;
                let part = self.f.select(upper, shifted, safe)?;
                let part = self.f.convert(part, Elem::I32);
                let part = self.f.bitcast(part, Elem::U32)?;
                let bias = self.f.const_i(Elem::U32, 0x8000_0000, &dims);
                let biased = self.f.add(part, bias)?;
                let v = self.f.select(upper, biased, part)?;
                let max = self.f.const_i(Elem::U32, i64::from(u32::MAX), &dims);
                self.f.select(over, max, v)?
            }
            (Dtype::U32, Dtype::I32) => self.f.bitcast(x, Elem::I32)?,
            (Dtype::I32, Dtype::U32) => self.f.bitcast(x, Elem::U32)?,
            (Dtype::Bool, Dtype::I32 | Dtype::U32) => self.f.convert(x, elem(to)?),
            (Dtype::F32 | Dtype::I32 | Dtype::U32, Dtype::Bool) => {
                let zero = self.zero_like(x);
                self.f.compare(Cmp::Ne, x, zero)?
            }
            _ => return refuse(format!("a cast from {from:?} to {to:?}")),
        })
    }

    fn arith(&mut self, tag: u8, d: Dtype, a: Val, b: Val) -> Lowering<Val> {
        let float = d == Dtype::F32;
        if d == Dtype::Bool {
            return refuse("arithmetic on bools");
        }
        Ok(match tag {
            tags::ADD => self.f.add(a, b)?,
            tags::SUB => self.f.sub(a, b)?,
            tags::MUL => self.f.mul(a, b)?,
            tags::DIV if float => self.f.div(a, b)?,
            tags::REM if float => fmod(&mut self.f, a, b)?,
            tags::DIV | tags::REM => {
                // The interpreter divides in i64: by zero is 0, and
                // i32::MIN / -1 wraps (i32::MIN % -1 is 0).
                let zero = self.zero_like(b);
                let by_zero = self.f.compare(Cmp::Eq, b, zero)?;
                let minus = if d == Dtype::I32 {
                    let m1 = self.f.like_i(b, -1);
                    Some(self.f.compare(Cmp::Eq, b, m1)?)
                } else {
                    None
                };
                let odd = match minus {
                    Some(m) => self.f.or(by_zero, m)?,
                    None => by_zero,
                };
                let one = self.f.like_i(b, 1);
                let safe = self.f.select(odd, one, b)?;
                let q = if tag == tags::DIV {
                    self.f.div(a, safe)?
                } else {
                    self.f.rem(a, safe)?
                };
                let q = match (minus, tag == tags::DIV) {
                    (Some(m), true) => {
                        let za = self.zero_like(a);
                        let neg = self.f.sub(za, a)?;
                        self.f.select(m, neg, q)?
                    }
                    (Some(m), false) => {
                        let za = self.zero_like(a);
                        self.f.select(m, za, q)?
                    }
                    _ => q,
                };
                self.f.select(by_zero, zero, q)?
            }
            tags::MAX_ELEM | tags::MIN_ELEM if float => {
                // `f32::max`/`min`: a NaN operand yields the other one.
                let m = if tag == tags::MAX_ELEM {
                    self.f.max(a, b)?
                } else {
                    self.f.min(a, b)?
                };
                let an = self.is_nan(a)?;
                let bn = self.is_nan(b)?;
                let m = self.f.select(bn, a, m)?;
                self.f.select(an, b, m)?
            }
            tags::MAX_ELEM => self.f.max(a, b)?,
            tags::MIN_ELEM => self.f.min(a, b)?,
            _ => return refuse("an arithmetic tag"),
        })
    }

    fn pivot(&mut self, op: &LaunchOp, shape: &[u32]) -> Lowering<()> {
        let result = op.result_id;
        let (ix, x) = self.raw_arg(op, 0)?;
        if self.dtype(ix) != Dtype::F32 {
            return refuse("a pivot over non-floats");
        }
        let (x, rows, n) = self.rows_of(x)?;
        let b = self.b;
        let payload = op.pred_payload;
        let pv = self.get(payload)?;
        let pd = self.dtype(payload);
        let pn = numel(&self.shape(payload)) as i64;
        // The payload per row: one value for every row, or one per row.
        let per_row = if pn == 1 {
            let flat = self.f.reshape(pv, &[b])?;
            self.f.broadcast(flat, &[b, rows], &[0])?
        } else if pn == rows {
            self.f.reshape(pv, &[b, rows])?
        } else {
            return refuse(format!("a pivot payload of {pn} for {rows} rows"));
        };
        let grid = [b, rows, n];
        let keep = match op.pred_tag {
            0 => {
                // rank_le(k): a lane is kept when fewer than k non-NaN lanes
                // are greater, i.e. when it is >= the k-th largest.
                let k = match pd {
                    Dtype::I32 => {
                        let lo = self.f.const_i(Elem::I32, 0, &[b, rows]);
                        let hi = self.f.const_i(Elem::I32, n, &[b, rows]);
                        self.f.clamp(lo, per_row, hi)?
                    }
                    Dtype::U32 => {
                        let hi = self.f.const_i(Elem::U32, n, &[b, rows]);
                        let big = self.f.compare(Cmp::Gt, per_row, hi)?;
                        let k = self.f.select(big, hi, per_row)?;
                        self.f.bitcast(k, Elem::I32)?
                    }
                    _ => return refuse("rank_le over a non-integer k"),
                };
                let key = self.order_key(x)?;
                let (sorted, _) = self.sort_keys(key, false)?;
                let nan = self.is_nan(x)?;
                let real = self.f.not(nan);
                let count = self.f.convert(real, Elem::I32);
                let count = self.f.reduce(count, &[2], Fold::Sum)?;
                let t = self.f.min(k, count)?;
                let one = self.f.const_i(Elem::I32, 1, &[b, rows]);
                let at = self.f.sub(t, one)?;
                let pivot = self.at_position(sorted, at, i64::from(i32::MIN))?;
                let pivot = self.f.broadcast(pivot, &grid, &[0, 1])?;
                let zero = self.f.const_i(Elem::I32, 0, &[b, rows]);
                let some = self.f.compare(Cmp::Gt, t, zero)?;
                let some = self.f.broadcast(some, &grid, &[0, 1])?;
                let above = self.f.compare(Cmp::Ge, key, pivot)?;
                let keep = self.f.and(above, real)?;
                self.f.and(keep, some)?
            }
            1 => {
                // cummass_le(p): in descending order, a lane is kept while
                // the mass before it is below p.
                if pd != Dtype::F32 {
                    return refuse("cummass_le over a non-float p");
                }
                let key = self.order_key(x)?;
                let (sorted, index) = self.sort_keys(key, true)?;
                let index = index.expect("indexed");
                let values = self.key_value(sorted)?;
                let before = self.scan_exclusive(values)?;
                let p = self.f.broadcast(per_row, &grid, &[0, 1])?;
                let kept = self.f.compare(Cmp::Lt, before, p)?;
                let kept = self.f.convert(kept, Elem::I32);
                // With no negative lane the mass before a lane only grows
                // along the order, so the kept lanes are the first `L`: a
                // lane is kept when it orders at or before the `L`-th. With
                // a negative lane anywhere, sort the verdicts back by index.
                let zero_f = self.f.const_f(Elem::F32, 0.0, &grid);
                let negative = self.f.compare(Cmp::Lt, x, zero_f)?;
                let all: Vec<i64> = vec![0, 1, 2];
                let any = self.f.reduce(negative, &all, Fold::Or)?;
                let branch = self.f.convert(any, Elem::I32);
                let out_ty = Ty::new(Elem::I32, &grid);
                let kept_rows = self.f.reduce(kept, &[2], Fold::Sum)?;
                let one = self.f.const_i(Elem::I32, 1, &[b, rows]);
                let last = self.f.sub(kept_rows, one)?;
                let pivot_key = self.at_position(sorted, last, i64::from(i32::MIN))?;
                let pivot_at = self.at_position(index, last, -1)?;
                let prefix = Box::new(|f: &mut Func| -> kernels_xla::hlo::Built<Vec<Val>> {
                    let pk = f.broadcast(pivot_key, &grid, &[0, 1])?;
                    let pi = f.broadcast(pivot_at, &grid, &[0, 1])?;
                    let col = f.iota(Elem::I32, &grid, 2);
                    let above = f.compare(Cmp::Gt, key, pk)?;
                    let level = f.compare(Cmp::Eq, key, pk)?;
                    let early = f.compare(Cmp::Le, col, pi)?;
                    let tie = f.and(level, early)?;
                    let keep = f.or(above, tie)?;
                    let zero = f.like_i(kept_rows, 0);
                    let some = f.compare(Cmp::Gt, kept_rows, zero)?;
                    let some = f.broadcast(some, &grid, &[0, 1])?;
                    let keep = f.and(keep, some)?;
                    Ok(vec![f.convert(keep, Elem::I32)])
                })
                    as Box<dyn FnOnce(&mut Func) -> kernels_xla::hlo::Built<Vec<Val>>>;
                let back = Box::new(|f: &mut Func| -> kernels_xla::hlo::Built<Vec<Val>> {
                    let back = f.sort(&[index, kept], 2, false, |f, a, b| {
                        f.compare(Cmp::Lt, a[0], b[0])
                    })?;
                    Ok(vec![back[1]])
                })
                    as Box<dyn FnOnce(&mut Func) -> kernels_xla::hlo::Built<Vec<Val>>>;
                let verdict = self.f.case(branch, &[out_ty], vec![prefix, back])?;
                let zero = self.f.const_i(Elem::I32, 0, &grid);
                self.f.compare(Cmp::Ne, verdict[0], zero)?
            }
            _ => {
                if pd != Dtype::F32 {
                    return refuse("prob_ge over a non-float threshold");
                }
                let t = self.f.broadcast(per_row, &grid, &[0, 1])?;
                self.f.compare(Cmp::Ge, x, t)?
            }
        };
        let keep = self.reshaped(keep, shape)?;
        self.set(result, keep)
    }

    fn scatter(&mut self, op: &LaunchOp, shape: &[u32]) -> Lowering<()> {
        let result = op.result_id;
        let (ib, base) = self.raw_arg(op, 0)?;
        let (_, idx) = self.raw_arg(op, 1)?;
        let (iv, vals) = self.raw_arg(op, 2)?;
        let d = self.dtype(ib);
        let add = op.tag == tags::SCATTER_ADD;
        if add && d == Dtype::Bool {
            return refuse("scatter_add into bools");
        }
        if self.dtype(iv) != d {
            return refuse("scatter values of another dtype than the base");
        }
        let bs = self.shape(ib);
        let n0 = i64::from(*bs.first().unwrap_or(&1));
        let rest: i64 = bs
            .iter()
            .skip(1)
            .map(|&x| i64::from(x))
            .product::<i64>()
            .max(1);
        let b = self.b;
        let (safe, valid) = self.index(idx, n0)?;
        let k = self.f.dims(safe)[1];
        let vn = numel(&self.shape(iv)) as i64;
        let updates = if vn == 1 && k * rest != 1 {
            let flat = self.f.reshape(vals, &[b])?;
            self.f.broadcast(flat, &[b, k, rest], &[0])?
        } else if vn == k * rest {
            self.f.reshape(vals, &[b, k, rest])?
        } else {
            return refuse(format!("{vn} scatter values for {k} rows of {rest}"));
        };
        let mut valid = valid;
        if !add && k > 1 {
            // Sequential semantics: the last write to a row wins. Drop every
            // write a later one to the same row overrides.
            if k > 4096 {
                return refuse(format!(
                    "scatter_set of {k} rows (duplicates are resolved in k^2)"
                ));
            }
            let dims = [b, k, k];
            let me = self.f.broadcast(safe, &dims, &[0, 1])?;
            let other = self.f.broadcast(safe, &dims, &[0, 2])?;
            let same = self.f.compare(Cmp::Eq, me, other)?;
            let ov = self.f.broadcast(valid, &dims, &[0, 2])?;
            let same = self.f.and(same, ov)?;
            let i = self.f.iota(Elem::I32, &dims, 1);
            let j = self.f.iota(Elem::I32, &dims, 2);
            let later = self.f.compare(Cmp::Gt, j, i)?;
            let shadowed = self.f.and(same, later)?;
            let shadowed = self.f.convert(shadowed, Elem::I32);
            let shadowed = self.f.reduce(shadowed, &[2], Fold::Max)?;
            let zero = self.f.const_i(Elem::I32, 0, &[b, k]);
            let last = self.f.compare(Cmp::Eq, shadowed, zero)?;
            valid = self.f.and(valid, last)?;
        }
        let row = self.f.iota(Elem::I32, &[b, k], 0);
        let stride = self.f.const_i(Elem::I32, n0, &[b, k]);
        let row = self.f.mul(row, stride)?;
        let row = self.f.add(row, safe)?;
        let out = self.f.const_i(Elem::I32, b * n0, &[b, k]);
        let row = self.f.select(valid, row, out)?;
        let row = self.f.reshape(row, &[b * k, 1])?;
        let table = self.f.reshape(base, &[b * n0, rest])?;
        let updates = self.f.reshape(updates, &[b * k, rest])?;
        let combine = if add {
            kernels_xla::hlo::Combine::Add
        } else {
            kernels_xla::hlo::Combine::Set
        };
        let y = self.f.scatter(
            table,
            row,
            updates,
            &ScatterDims {
                update_window_dims: vec![1],
                inserted_window_dims: vec![0],
                scatter_dims_to_operand_dims: vec![0],
                index_vector_dim: 1,
                ..ScatterDims::default()
            },
            combine,
        )?;
        let y = self.reshaped(y, shape)?;
        self.set(result, y)
    }
}

/// The exponent of positive finite nonzero f32 lanes: `floor(log2 v)`,
/// subnormals included.
fn exponent(f: &mut Func, v: Val) -> kernels_xla::hlo::Built<Val> {
    let bits = f.bitcast(v, Elem::I32)?;
    let s23 = f.like_i(bits, 23);
    let field = f.shr(bits, s23)?;
    let bias = f.like_i(bits, 127);
    let normal = f.sub(field, bias)?;
    let lz = f.unary("count_leading_zeros", bits);
    let base = f.like_i(bits, -118);
    let sub = f.sub(base, lz)?;
    let zero = f.like_i(bits, 0);
    let is_normal = f.compare(Cmp::Gt, field, zero)?;
    f.select(is_normal, normal, sub)
}

/// `2^k` for i32 lanes `k` in `0..=127`.
fn pow2(f: &mut Func, k: Val) -> kernels_xla::hlo::Built<Val> {
    let bias = f.like_i(k, 127);
    let e = f.add(k, bias)?;
    let s23 = f.like_i(k, 23);
    let bits = f.shl(e, s23)?;
    f.bitcast(bits, Elem::F32)
}

/// `x % y` as Rust computes it for f32 (C `fmod`: exact, the sign of `x`).
/// XLA's float remainder rounds the quotient, so this subtracts `y`'s
/// largest power-of-two multiple below the remainder until it is below
/// `|y|`: every subtraction is exact (Sterbenz), and each halves the rest.
fn fmod(f: &mut Func, x: Val, y: Val) -> kernels_xla::hlo::Built<Val> {
    let dims = f.dims(x).to_vec();
    let ax = f.abs(x);
    let ay = f.abs(y);
    let zero = f.const_f(Elem::F32, 0.0, &dims);
    let one = f.const_f(Elem::F32, 1.0, &dims);
    let inf = f.const_f(Elem::F32, f64::INFINITY, &dims);
    let y0 = f.compare(Cmp::Eq, ay, zero)?;
    let xfin = f.is_finite(x);
    let xbad = f.not(xfin);
    let ynan = f.compare(Cmp::Ne, y, y)?;
    let bad = f.or(y0, xbad)?;
    let bad = f.or(bad, ynan)?;
    let yinf = f.compare(Cmp::Eq, ay, inf)?;
    let small = f.compare(Cmp::Lt, ax, ay)?;
    let pass = f.or(yinf, small)?;
    let quiet = f.or(bad, pass)?;
    let r0 = f.select(quiet, zero, ax)?;
    let d = f.select(quiet, one, ay)?;
    let ed = exponent(f, d)?;
    let axes: Vec<i64> = (0..dims.len() as i64).collect();
    let out = f.while_loop(
        &[r0],
        |f, a| {
            let ge = f.compare(Cmp::Ge, a[0], d)?;
            f.reduce(ge, &axes, Fold::Or)
        },
        |f, a| {
            let r = a[0];
            let act = f.compare(Cmp::Ge, r, d)?;
            let er = exponent(f, r)?;
            let e = f.sub(er, ed)?;
            let nought = f.like_i(e, 0);
            let e = f.select(act, e, nought)?;
            let cap = f.like_i(e, 100);
            let e1 = f.min(e, cap)?;
            let rest = f.sub(e, e1)?;
            let e2 = f.min(rest, cap)?;
            let e3 = f.sub(rest, e2)?;
            let mut t = d;
            for k in [e1, e2, e3] {
                let p = pow2(f, k)?;
                t = f.mul(t, p)?;
            }
            let over = f.compare(Cmp::Gt, t, r)?;
            let half = f.like_f(t, 0.5);
            let halved = f.mul(t, half)?;
            let t = f.select(over, halved, t)?;
            let less = f.sub(r, t)?;
            Ok(vec![f.select(act, less, r)?])
        },
    )?;
    let r = out[0];
    let rb = f.bitcast(r, Elem::I32)?;
    let xb = f.bitcast(x, Elem::I32)?;
    let mag = f.like_i(rb, 0x7FFF_FFFF);
    let sign = f.like_i(rb, i64::from(i32::MIN));
    let rb = f.and(rb, mag)?;
    let xs = f.and(xb, sign)?;
    let bits = f.or(rb, xs)?;
    let res = f.bitcast(bits, Elem::F32)?;
    let res = f.select(pass, x, res)?;
    let nan = f.const_f(Elem::F32, f64::NAN, &dims);
    f.select(bad, nan, res)
}
