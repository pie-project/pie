//! One eta stage (`LaunchPackage::stages[at]`) as one StableHLO function
//! over a batch of lanes.
//!
//! Every eta value of shape `S` becomes a tensor `[B, S...]`: lane `b` of the
//! batch is one instance's pass. The semantics are the host interpreter's
//! (`eta_compiler::eval::interp::eval_op`), op by op; where the interpreter is exact (integer
//! arithmetic, comparisons, argmax, orderings and their ties, masks, the
//! uniform draw's bits) so is this, and float math agrees within rounding.
//!
//! The function's parameters are one packed `u32 [B, in_words]` plane (every
//! host-fed root, bit-exact: f32/i32/u32 one word per element, bools 32 per
//! word; plus, per output read on the device, the lane's first row of it),
//! then each plane of the fire's outputs the stage reads. The one result is
//! `u32 [B, out_words]`: every value the stage puts, packed the same way.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use eta_compiler::codegen::launch::{LaunchOp, LaunchPackage, ValueOrigin};
use eta_ir::op::tags;
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

/// Output `name` of the pass's lane `lane`, read off a fire's plane of
/// `rows` rows a layer, `layers` deep (0: one), `width` of `dtype` wide.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Read {
    pub name: String,
    pub lane: u32,
    /// The fire's readout rows when `readout`, else its plane's.
    pub rows: u32,
    pub readout: bool,
    pub layers: u32,
    pub width: u32,
    pub dtype: Dtype,
    /// The plane holds no denormal (`readout::Plane::normal`).
    pub normal: bool,
}

/// The shape of one lowering: how many lanes, and the outputs the device
/// binds.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Batch {
    pub lanes: u32,
    pub reads: Vec<Read>,
}

/// What a packed slot carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feed {
    /// The value itself: `dtype` over `numel` elements.
    Value(Dtype, usize),
    /// The lane's first row of `Layout::reads[read]` (one i32 word); the
    /// stage reads `rows` rows of each layer from there.
    Row { read: usize, rows: u32 },
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
    /// The planes the stage reads, a parameter each after the pack.
    pub reads: Vec<Read>,
    /// Carried cells: channels whose cell stays on the device between
    /// passes, in each lane's `Channel` slots of the pool. The stage takes
    /// the pool (its last parameter, donated, and its last result) and, at
    /// `carried_rows_at` of each lane's packed row, one word a channel of
    /// `carried`: the pool row the lane's cell starts at.
    pub carried_in: Vec<(u32, Slot)>,
    pub carried_out: Vec<(u32, Slot)>,
    /// Puts to a channel the next fire reads off the device (`embed_tokens`):
    /// packed for the host as any output, and written to the lane's slot
    /// of the ring too (its row at `carried_rows_at`, as a carried cell's).
    pub fed_out: Vec<(u32, Slot)>,
    pub carried: Vec<(u32, usize)>,
    pub carried_rows_at: usize,
}

impl Layout {
    /// Whether the stage returns anything (else there is nothing to run).
    #[must_use]
    pub fn runs(&self) -> bool {
        self.out_words > 0 || !self.carried_out.is_empty()
    }

    /// Whether the stage reads or writes the pool (its carried cells).
    #[must_use]
    pub fn pooled(&self) -> bool {
        !self.carried.is_empty()
    }
}

/// The pool's rows `ids` (an i32 vector) as a gather and a scatter keep
/// them: two indices at least. A single row XLA turns into a `[1, 128]`
/// `dynamic-slice` (or `dynamic-update-slice`) of the pool, which it may lay
/// out column-major, and those take the pool in their result's layout: the
/// whole pool copied, and copied again where XLA rematerializes it, to touch
/// one lane's row, past the HBM a large pool leaves (gumbel_watermark's
/// stage asked 48 GB of a 31 GB chip). A gather's or a scatter's operand
/// keeps its own layout. One row is read, or written, twice.
fn pool_ids(f: &mut Func, ids: Val) -> Lowering<(Val, bool)> {
    if f.dims(ids)[0] != 1 {
        return Ok((ids, false));
    }
    Ok((f.concat(&[ids, ids], 0)?, true))
}

/// The pool's rows `ids` (an i32 vector), `[n, 128]` (`pool_ids`).
fn pool_rows_read(f: &mut Func, pool: Val, ids: Val) -> Lowering<Val> {
    let (ids, twice) = pool_ids(f, ids)?;
    let rows = f.take_rows(pool, ids)?;
    if !twice {
        return Ok(rows);
    }
    Ok(f.slice(rows, &[0, 0], &[1, 128], &[1, 1])?)
}

/// The pool with `rows` (`[n, 128]`) over its rows `ids` (`pool_ids`); an
/// id off the pool writes nothing.
fn pool_rows_write(f: &mut Func, pool: Val, ids: Val, rows: Val) -> Lowering<Val> {
    let (ids, twice) = pool_ids(f, ids)?;
    let rows = if twice {
        f.concat(&[rows, rows], 0)?
    } else {
        rows
    };
    Ok(f.put_rows(pool, ids, rows, kernels_xla::hlo::Combine::Set)?)
}

/// `a / b` and `a % b`, elementwise, truncating (Rust's `/` and `%`), for a
/// nonzero `b`: u32, or i32 by its magnitudes (`i32::MIN / -1` wraps to
/// `i32::MIN`, `% -1` is 0). Restoring long division over the 32 bits in
/// shifts, compares and subtracts, exact for every u32: the remainder's top
/// bit stands in for the 33rd bit its shift would carry.
fn int_divrem(f: &mut Func, a: Val, b: Val, signed: bool) -> Lowering<(Val, Val)> {
    let dims = f.dims(a).to_vec();
    let (ua, ub) = if signed {
        let zero = f.const_i(Elem::I32, 0, &dims);
        let magnitude = |f: &mut Func, x: Val| -> Lowering<Val> {
            let negative = f.compare(Cmp::Lt, x, zero)?;
            let flipped = f.sub(zero, x)?;
            let m = f.select(negative, flipped, x)?;
            Ok(f.bitcast(m, Elem::U32)?)
        };
        (magnitude(f, a)?, magnitude(f, b)?)
    } else {
        (a, b)
    };
    let zero = f.const_i(Elem::U32, 0, &dims);
    let one = f.const_i(Elem::U32, 1, &dims);
    let top = f.const_i(Elem::U32, 31, &dims);
    let mut q = zero;
    let mut r = zero;
    for bit in (0..32).rev() {
        let at = f.const_i(Elem::U32, bit, &dims);
        let next = f.shr(ua, at)?;
        let next = f.and(next, one)?;
        let carried = f.shr(r, top)?;
        let carried = f.compare(Cmp::Ne, carried, zero)?;
        let shifted = f.shl(r, one)?;
        let shifted = f.or(shifted, next)?;
        let fits = f.compare(Cmp::Ge, shifted, ub)?;
        let take = f.or(carried, fits)?;
        let less = f.sub(shifted, ub)?;
        r = f.select(take, less, shifted)?;
        let place = f.shl(one, at)?;
        let place = f.select(take, place, zero)?;
        q = f.or(q, place)?;
    }
    if !signed {
        return Ok((q, r));
    }
    let zero = f.const_i(Elem::I32, 0, &dims);
    let q = f.bitcast(q, Elem::I32)?;
    let r = f.bitcast(r, Elem::I32)?;
    let a_neg = f.compare(Cmp::Lt, a, zero)?;
    let b_neg = f.compare(Cmp::Lt, b, zero)?;
    let flip = f.xor(a_neg, b_neg)?;
    let nq = f.sub(zero, q)?;
    let q = f.select(flip, nq, q)?;
    let nr = f.sub(zero, r)?;
    let r = f.select(a_neg, nr, r)?;
    Ok((q, r))
}

/// Pool rows (u32 `[_, 128]`) a carried cell of `words` takes.
#[must_use]
pub fn cell_rows(words: usize) -> usize {
    words.div_ceil(128).max(1)
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
    } else if dtype == Dtype::Bf16 {
        // two to a word, element `2w` in the low half (`eta_ir::types`)
        numel.div_ceil(2)
    } else {
        numel
    }
}

fn elem(dtype: Dtype) -> Lowering<Elem> {
    Ok(match dtype {
        // A bf16 value is held as the f32 it stands for, rounded to bf16 on
        // its store (`Lower::set`): every op loads it exactly and computes
        // in f32 (`eta_ir::types`' bf16 rules).
        Dtype::F32 | Dtype::Bf16 => Elem::F32,
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

/// The most rows one of stage `at`'s guest lanes reads of a fire's plane
/// it is bound to read as `reads` states (its readout rows: a beam group of
/// `B` beams reads `B`), 1 where it reads none: what a batch of the stage
/// spans of a fire's rows, a guest lane at a time.
#[must_use]
pub fn lane_rows(package: &LaunchPackage, at: usize, reads: &[Read]) -> u32 {
    roots(package, at)
        .iter()
        .filter_map(|&id| package.values.get(id as usize))
        .filter(|value| value.source == ValueOrigin::Output)
        .filter_map(|value| {
            let name = value.output.as_deref().unwrap_or("");
            let read = reads
                .iter()
                .find(|r| r.name == name && r.lane == value.lane)?;
            let per = read.width as usize * read.layers.max(1) as usize;
            (per > 0).then(|| numel(&value.shape) / per)
        })
        .max()
        .map_or(1, |rows| u32::try_from(rows.max(1)).unwrap_or(u32::MAX))
}

/// The values stage `at` reads without computing them, in order of first
/// use (channel cells, constants, outputs).
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
        for arg in op.reads() {
            note(arg, &mut roots);
        }
    }
    for put in &stage.puts {
        note(put.value, &mut roots);
    }
    roots
}

/// The outputs stage `at` reads, by name and the pass's lane.
#[must_use]
pub fn outputs(package: &LaunchPackage, at: usize) -> Vec<(String, u32)> {
    roots(package, at)
        .iter()
        .filter_map(|&id| {
            let value = package.values.get(id as usize)?;
            Some((value.output.clone()?, value.lane))
        })
        .collect()
}

/// Lowers stage `at` of `package` for `batch`, with the cells of the
/// `carried` channels on the device (see `guest::carried_channels`).
pub fn lower(
    package: &LaunchPackage,
    at: usize,
    batch: Batch,
    carried: &[u32],
    pool_rows: i64,
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
            ValueOrigin::Output => {
                let name = value.output.as_deref().unwrap_or("");
                let found =
                    (batch.reads.iter()).find(|read| read.name == name && read.lane == value.lane);
                match found {
                    None => Feed::Value(value.dtype, n),
                    Some(read) => {
                        if read.dtype != value.dtype {
                            return refuse(format!(
                                "output `{name}` is read as {:?} and lands as {:?}",
                                value.dtype, read.dtype
                            ));
                        }
                        let per = read.width as usize * read.layers.max(1) as usize;
                        if per == 0 || !n.is_multiple_of(per) {
                            return refuse(format!(
                                "output `{name}` holds {n} values, not whole {}-wide rows of \
                                 {} layer(s)",
                                read.width,
                                read.layers.max(1)
                            ));
                        }
                        let at = match layout.reads.iter().position(|seen| seen == read) {
                            Some(at) => at,
                            None => {
                                layout.reads.push(read.clone());
                                layout.reads.len() - 1
                            }
                        };
                        Feed::Row {
                            read: at,
                            rows: (n / per) as u32,
                        }
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
        let fed = package
            .ports
            .iter()
            .any(|p| !p.is_const && p.port == eta_ir::registry::Port::EmbedTokens && p.channel == put.channel);
        if carried.contains(&put.channel) && fed {
            layout.fed_out.push((
                put.channel,
                Slot {
                    value: put.value,
                    feed: Feed::Value(value.dtype, n),
                    at: 0,
                    words: w,
                },
            ));
        } else if carried.contains(&put.channel) {
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

    // One pack word a carried channel: the lane's cell's first pool row.
    let mut chans: Vec<(u32, usize)> = Vec::new();
    for (c, slot) in layout
        .carried_in
        .iter()
        .chain(&layout.carried_out)
        .chain(&layout.fed_out)
    {
        if !chans.iter().any(|(k, _)| k == c) {
            chans.push((*c, slot.words));
        }
    }
    layout.carried_rows_at = layout.in_words;
    layout.in_words += chans.len();
    layout.carried = chans;

    let mut l = Lower {
        f: Func::new("main"),
        package,
        b: i64::from(batch.lanes.max(1)),
        env: HashMap::new(),
        splats: HashSet::new(),
        denormal: HashSet::new(),
        making_denormal: true,
    };
    let pack = (layout.in_words > 0).then(|| {
        l.f.param(Ty::new(Elem::U32, &[l.b, layout.in_words as i64]), None)
    });
    let planes: Vec<Val> = (layout.reads.iter())
        .map(|read| {
            let rows = i64::from(read.rows) * i64::from(read.layers.max(1));
            // Row-major, as a fire lands its planes: the layout XLA picks
            // for a narrow plane (`[128, 4]`, column-major) refuses the
            // fire's buffer at the stage's run.
            let p = l.f.param(
                Ty::new(elem(read.dtype)?, &[rows, i64::from(read.width)]),
                None,
            );
            l.f.pin_row_major(p);
            Ok(p)
        })
        .collect::<Lowering<_>>()?;
    // The pool, last: aliased to the last result.
    let pool = layout.pooled().then(|| {
        let result = u32::from(layout.out_words > 0);
        let p = l.f.param(Ty::new(Elem::U32, &[pool_rows.max(1), 128]), Some(result));
        l.f.pin_row_major(p);
        p
    });
    // Each carried channel's rows, `[B · rows]`: its cell's rows a lane.
    let mut cell_at: HashMap<u32, Val> = HashMap::new();
    for (k, &(c, w)) in layout.carried.iter().enumerate() {
        let pack = pack.expect("carried rows ride the pack");
        let first = l.f.slice(
            pack,
            &[0, (layout.carried_rows_at + k) as i64],
            &[l.b, (layout.carried_rows_at + k + 1) as i64],
            &[1, 1],
        )?;
        let first = l.f.bitcast(first, Elem::I32)?;
        let nr = cell_rows(w) as i64;
        let first = l.f.broadcast(first, &[l.b, nr], &[0, 1])?;
        let step = l.f.iota(Elem::I32, &[l.b, nr], 1);
        let rows = l.f.add(first, step)?;
        cell_at.insert(c, l.f.reshape(rows, &[l.b * nr])?);
    }
    for (c, slot) in &layout.carried_in {
        let Feed::Value(dtype, n) = slot.feed else {
            unreachable!("carried cells are values");
        };
        let nr = cell_rows(slot.words) as i64;
        let pool = pool.expect("carried cells ride the pool");
        let cells = pool_rows_read(&mut l.f, pool, cell_at[c])?;
        let cells = l.f.reshape(cells, &[l.b, nr * 128])?;
        let v = l.unpack(cells, 0, dtype, n)?;
        let v = l.reshaped(v, &package.values[slot.value as usize].shape)?;
        l.env.insert(slot.value, v);
        l.denormal.insert(slot.value);
    }

    // Roots.
    for &id in &roots {
        if l.env.contains_key(&id) {
            continue;
        }
        let value = &package.values[id as usize];
        let v = if value.source == ValueOrigin::Const {
            l.splats.insert(id);
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
                Feed::Row { read, rows } => {
                    let plane = planes[read];
                    let r = &layout.reads[read];
                    l.read_rows(pack, slot.at, plane, rows, r, &value.shape)?
                }
            }
        };
        // A loaded value may hold a denormal (`Lower::denormal`), but for a
        // row of a plane the engine states free of them, or a constant
        // that is not one.
        let normal = if value.source == ValueOrigin::Const {
            let x = f32::from_bits(value.literal_bits);
            value.dtype != Dtype::F32 || x == 0.0 || x.is_normal() || !x.is_finite()
        } else {
            layout.inputs.iter().find(|s| s.value == id).is_some_and(
                |s| matches!(s.feed, Feed::Row { read, .. } if layout.reads[read].normal),
            )
        };
        l.env.insert(id, v);
        if !normal {
            l.denormal.insert(id);
        }
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
    if let Some(mut pool) = pool {
        for (c, slot) in layout.carried_out.iter().chain(&layout.fed_out) {
            let v = l.get(slot.value)?;
            let Feed::Value(dtype, n) = slot.feed else {
                unreachable!("carried cells are values");
            };
            let cell = l.pack(v, dtype, n)?;
            let nr = cell_rows(slot.words) as i64;
            let w = slot.words as i64;
            let cell = if nr * 128 > w {
                let z = l.f.const_i(Elem::U32, 0, &[]);
                l.f.pad(cell, z, &[0, 0], &[0, nr * 128 - w], &[0, 0])?
            } else {
                cell
            };
            let cell = l.f.reshape(cell, &[l.b * nr, 128])?;
            pool = pool_rows_write(&mut l.f, pool, cell_at[c], cell)?;
        }
        results.push(pool);
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
    /// Values one constant fills: a constant, or a broadcast or reshape of
    /// one. A division by one keeps XLA's divide, which it turns into a
    /// multiply and shifts.
    splats: HashSet<u32>,
    /// Values that may hold an f32 denormal on the device: loaded ones,
    /// transcendental results, and what moves them unchanged. v6e's float
    /// arithmetic flushes denormal operands and results to zero, so a sum,
    /// product, quotient or matmul never holds one (`round_bf16`).
    denormal: HashSet<u32>,
    /// Whether the op being lowered may make a denormal (`Lower::op`).
    making_denormal: bool,
}

impl Lower<'_> {
    fn shape(&self, id: u32) -> Vec<u32> {
        self.package.values[id as usize].shape.clone()
    }

    /// The dtype value `id` computes as: a bf16 value is an f32 to every op
    /// (it loads exactly); [`Self::stored`] is its declared dtype.
    fn dtype(&self, id: u32) -> Dtype {
        match self.stored(id) {
            Dtype::Bf16 => Dtype::F32,
            d => d,
        }
    }

    fn stored(&self, id: u32) -> Dtype {
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
        // A bf16 value's store rounds its f32 to bf16, nearest-even, a NaN
        // quiet, and holds the f32 that bf16 stands for.
        let v = if self.stored(id) != Dtype::Bf16 {
            v
        } else if self.making_denormal {
            self.round_bf16(v)?
        } else {
            self.round_bf16_normal(v)?
        };
        if self.making_denormal {
            self.denormal.insert(id);
        }
        self.env.insert(id, v);
        Ok(())
    }

    /// `v` (f32) rounded to bf16, nearest-even, as the f32 it stands for
    /// (`eta_ir::types::bf16_bits`), in integer ops on its bits: XLA, free to
    /// keep excess precision, drops an f32 -> bf16 -> f32 convert pair (v6e:
    /// half of a row came back unrounded).
    fn round_bf16(&mut self, v: Val) -> Lowering<Val> {
        let bits = self.f.bitcast(v, Elem::U32)?;
        let dims = self.f.dims(bits).to_vec();
        let k = |f: &mut Func, x: i64| f.const_i(Elem::U32, x, &dims);
        let sixteen = k(&mut self.f, 16);
        let one = k(&mut self.f, 1);
        let lsb = self.f.shr(bits, sixteen)?;
        let lsb = self.f.and(lsb, one)?;
        let half = k(&mut self.f, 0x7FFF);
        let bias = self.f.add(half, lsb)?;
        let up = self.f.add(bits, bias)?;
        let top = k(&mut self.f, 0xFFFF_0000);
        let rounded = self.f.and(up, top)?;
        // a NaN stays a NaN, quiet: its sign and exponent, the quiet bit set
        let quiet = k(&mut self.f, 0x0040_0000);
        let nan = self.f.or(bits, quiet)?;
        let nan = self.f.and(nan, top)?;
        let is_nan = self.f.compare(Cmp::Ne, v, v)?;
        let out = self.f.select(is_nan, nan, rounded)?;
        Ok(self.f.bitcast(out, Elem::F32)?)
    }

    /// [`Self::round_bf16`] of a value that holds no denormal (a sum,
    /// product, quotient or matmul: v6e flushes them): `reduce_precision`
    /// rounds a normal or zero f32 to bf16 nearest-even as `bf16_bits` does
    /// (it flushes a denormal, which such a value never is), a NaN made
    /// quiet as `bf16_bits` makes it. One op where the bit form takes ten
    /// (v6e qwen3-0.6b sampled c64: the bit form's rounding ~1.5%).
    fn round_bf16_normal(&mut self, v: Val) -> Lowering<Val> {
        let ty = self.f.ty(v).clone();
        let rounded = self.f.op_named(
            "stablehlo.reduce_precision",
            &[v],
            "exponent_bits = 8 : i32, mantissa_bits = 7 : i32",
            vec![ty],
        )[0];
        let bits = self.f.bitcast(v, Elem::U32)?;
        let dims = self.f.dims(bits).to_vec();
        let quiet = self.f.const_i(Elem::U32, 0x0040_0000, &dims);
        let top = self.f.const_i(Elem::U32, 0xFFFF_0000, &dims);
        let nan = self.f.or(bits, quiet)?;
        let nan = self.f.and(nan, top)?;
        let nan = self.f.bitcast(nan, Elem::F32)?;
        let is_nan = self.f.compare(Cmp::Ne, v, v)?;
        Ok(self.f.select(is_nan, nan, rounded)?)
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
            Dtype::Bf16 => {
                // element `2w` the low half of word `w`, `2w + 1` the high;
                // each half shifted up 16 is the f32 it stands for
                let dims = [b, w as i64, 2];
                let spread = self.f.broadcast(cut, &dims, &[0, 1])?;
                let half = self.f.iota(Elem::U32, &dims, 2);
                let sixteen = self.f.const_i(Elem::U32, 16, &dims);
                let by = self.f.mul(half, sixteen)?;
                let moved = self.f.shr(spread, by)?;
                let low = self.f.const_i(Elem::U32, 0xFFFF, &dims);
                let bits = self.f.and(moved, low)?;
                let up = self.f.shl(bits, sixteen)?;
                let flat = self.f.reshape(up, &[b, (w * 2) as i64])?;
                let flat = self.f.slice(flat, &[0, 0], &[b, n as i64], &[1, 1])?;
                self.f.bitcast(flat, Elem::F32)?
            }
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
            Dtype::Bf16 => {
                // the held f32 is bf16-exact: its upper 16 bits, two a word
                let w = n.div_ceil(2);
                let bits = self.f.bitcast(flat, Elem::U32)?;
                let sixteen = self.f.const_i(Elem::U32, 16, &[b, n as i64]);
                let top = self.f.shr(bits, sixteen)?;
                let zero = self.f.const_i(Elem::U32, 0, &[]);
                let padded = self
                    .f
                    .pad(top, zero, &[0, 0], &[0, (w * 2 - n) as i64], &[0, 0])?;
                let dims = [b, w as i64, 2];
                let pairs = self.f.reshape(padded, &dims)?;
                let half = self.f.iota(Elem::U32, &dims, 2);
                let s16 = self.f.const_i(Elem::U32, 16, &dims);
                let by = self.f.mul(half, s16)?;
                let placed = self.f.shl(pairs, by)?;
                self.f.reduce(placed, &[2], Fold::Sum)?
            }
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

    /// The lane's `rows` rows of each layer of `read`'s plane, starting at
    /// the row its slot names, as the output's value.
    fn read_rows(
        &mut self,
        pack: Val,
        at: usize,
        plane: Val,
        rows: u32,
        read: &Read,
        shape: &[u32],
    ) -> Lowering<Val> {
        let b = self.b;
        let first = self.unpack(pack, at, Dtype::I32, 1)?; // [B, 1]
        let r = i64::from(rows) * i64::from(read.layers.max(1));
        let first = self.f.broadcast(first, &[b, r], &[0, 1])?;
        let step = if read.layers <= 1 {
            self.f.iota(Elem::I32, &[b, r], 1)
        } else {
            let offsets: Vec<i64> = (0..r)
                .map(|i| i % i64::from(rows) + i / i64::from(rows) * i64::from(read.rows))
                .collect();
            let offsets = self.f.const_ints(Elem::I32, &offsets, &[r])?;
            self.f.broadcast(offsets, &[b, r], &[1])?
        };
        let ids = self.f.add(first, step)?;
        let ids = self.f.reshape(ids, &[b * r])?;
        let picked = self.f.take_rows(plane, ids)?; // [B*r, width]
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
        let one = i64::from(fold == Fold::Prod);
        let identity = if e == Elem::F32 {
            self.f.const_f(e, one as f64, &[])
        } else {
            self.f.const_i(e, one, &[])
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

    /// The first `k` of `x` (`[B, R, n]` f32) in [`Self::sort_desc`]'s order
    /// and their source indices (i32), as `[B, R, k]`, by selection, not a
    /// sort: each row splits into chunks enough to fill an f32 tile's
    /// sublanes, each chunk gives its first `k` ([`Self::select_first`]),
    /// and the chunks' candidates give the row's. XLA lowers `chlo.top_k`
    /// to a sort (v6e, the beam inferlet's guest stage: its top-1 of 151936
    /// log-probs a whole-row sort, 793 us a step, its top-4 of 4 x 151936 a
    /// sort of 128 chunks, 154 us).
    fn top_desc(&mut self, x: Val, k: i64) -> Lowering<(Val, Val)> {
        let dims = self.f.dims(x).to_vec();
        let flat = flat_rows(&dims);
        let (rows, n) = (flat[0], flat[1]);
        let key = self.order_key(x)?;
        let key = self.f.reshape(key, &flat)?;
        // chunks of whole lane tiles, enough that the rows fill the sublanes,
        // each holding more than its `k` candidates
        let chunks = (F32_SUBLANES / rows).min(n / k.max(LANE_TILE)).max(1);
        let index = self.f.iota(Elem::I32, &flat, 1);
        let (key, index) = if chunks > 1 {
            let width = ((n + chunks - 1) / chunks + LANE_TILE - 1) / LANE_TILE * LANE_TILE;
            let pad = width * chunks - n;
            let lowest = self.f.const_i(Elem::I32, i64::from(i32::MIN), &[]);
            let far = self.f.const_i(Elem::I32, i64::from(i32::MAX), &[]);
            let key = self.f.pad(key, lowest, &[0, 0], &[0, pad], &[0, 0])?;
            let index = self.f.pad(index, far, &[0, 0], &[0, pad], &[0, 0])?;
            let key = self.f.reshape(key, &[rows * chunks, width])?;
            let index = self.f.reshape(index, &[rows * chunks, width])?;
            let (key, index) = self.select_first(key, index, k, Some(n))?;
            (
                self.f.reshape(key, &[rows, chunks * k])?,
                self.f.reshape(index, &[rows, chunks * k])?,
            )
        } else {
            (key, index)
        };
        let (_, index) = self.select_first(key, index, k, None)?;
        // `x[r, index[r, j]]`, through the flat row-major plane
        let plane = self.f.reshape(x, &[rows * n])?;
        let base = self.f.iota(Elem::I32, &[rows, k], 0);
        let stride = self.f.const_i(Elem::I32, n, &[rows, k]);
        let base = self.f.mul(base, stride)?;
        let at = self.f.add(base, index)?;
        let at = self.f.reshape(at, &[rows, k, 1])?;
        let values = self.f.gather(
            plane,
            at,
            &kernels_xla::hlo::GatherDims {
                offset_dims: vec![],
                collapsed_slice_dims: vec![0],
                start_index_map: vec![0],
                index_vector_dim: 2,
                ..kernels_xla::hlo::GatherDims::default()
            },
            &[1],
        )?;
        let mut out = dims;
        *out.last_mut().expect("a last axis") = k;
        Ok((self.f.reshape(values, &out)?, self.f.reshape(index, &out)?))
    }

    /// The first `k` of each row of the order keys `key` (`[M, m]` i32,
    /// beside their source indices `index`), in order: `k` rounds, each one
    /// reduce taking the row's first still-held lane (the greater key, the
    /// lower index on equal keys) and letting it go. A lane indexed at
    /// `past` or beyond (a chunk's padding) is held by none. Exact: what a
    /// stable descending sort's first `k` are.
    fn select_first(
        &mut self,
        key: Val,
        index: Val,
        k: i64,
        past: Option<i64>,
    ) -> Lowering<(Val, Val)> {
        let dims = self.f.dims(key).to_vec();
        let mut held = match past {
            Some(n) => {
                let bound = self.f.const_i(Elem::I32, n, &dims);
                let within = self.f.compare(Cmp::Lt, index, bound)?;
                self.f.convert(within, Elem::I32)
            }
            None => self.f.const_i(Elem::I32, 1, &dims),
        };
        let none = self.f.const_i(Elem::I32, 0, &[]);
        let lowest = self.f.const_i(Elem::I32, i64::from(i32::MIN), &[]);
        let far = self.f.const_i(Elem::I32, i64::from(i32::MAX), &[]);
        let (mut keys, mut picks) = (Vec::new(), Vec::new());
        for round in 0..k {
            let out = self.f.reduce_with(
                &[held, key, index],
                &[none, lowest, far],
                &[1],
                |f, a, b| {
                    let more_held = f.compare(Cmp::Gt, a[0], b[0])?;
                    let both = f.compare(Cmp::Eq, a[0], b[0])?;
                    let gt = f.compare(Cmp::Gt, a[1], b[1])?;
                    let eq = f.compare(Cmp::Eq, a[1], b[1])?;
                    let lt = f.compare(Cmp::Lt, a[2], b[2])?;
                    let tie = f.and(eq, lt)?;
                    let first = f.or(gt, tie)?;
                    let first = f.and(both, first)?;
                    let take = f.or(more_held, first)?;
                    Ok(vec![
                        f.select(take, a[0], b[0])?,
                        f.select(take, a[1], b[1])?,
                        f.select(take, a[2], b[2])?,
                    ])
                },
            )?;
            let (got_key, got) = (out[1], out[2]);
            keys.push(self.f.reshape(got_key, &[dims[0], 1])?);
            picks.push(self.f.reshape(got, &[dims[0], 1])?);
            if round + 1 < k {
                let at = self.f.broadcast(got, &dims, &[0])?;
                let taken = self.f.compare(Cmp::Eq, index, at)?;
                let zero = self.f.const_i(Elem::I32, 0, &dims);
                held = self.f.select(taken, zero, held)?;
            }
        }
        let keys = self.f.concat(&keys, 1)?;
        let picks = self.f.concat(&picks, 1)?;
        Ok((keys, picks))
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

    /// The keyed stream's seed from operand `i`, an rng state of u32 words
    /// (key, then counter): `rng_keyed`'s.
    fn keyed_seed(&mut self, op: &LaunchOp, i: usize) -> Lowering<Val> {
        let (is, state) = self.raw_arg(op, i)?;
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
        self.splitmix(joined)
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
            // `lora` applies in the forward, through the adapter the
            // attachment names (`adapter.rs`): the pass runs nothing for it.
            // Any other sink this shell runs nothing for anywhere, so a
            // program calling one is refused by name at its registration,
            // never answered as if the sink had run.
            let name = (self.package.names.get(op.name_index as usize))
                .map_or("<unnamed>", String::as_str);
            if name == crate::adapter::LORA {
                return Ok(());
            }
            return refuse(format!(
                "calls sink `{name}`, which this engine does not run: its program would \
                 answer as if the sink had acted"
            ));
        }
        let result = op.result_id;
        if result as usize >= self.package.values.len() {
            return refuse(format!("result {result} is past the package's values"));
        }
        let shape = self.shape(result);
        // Whether this op's results may hold a denormal: never a computed
        // one (the device flushes them), what moves a value unchanged where
        // an operand may, and everything else (a transcendental, a load).
        self.making_denormal = match op.tag {
            tags::ADD
            | tags::SUB
            | tags::MUL
            | tags::DIV
            | tags::REM
            | tags::MATMUL
            | tags::REDUCE_SUM
            | tags::CUMSUM
            | tags::CUMPROD => false,
            tags::ABS
            | tags::NEG
            | tags::SELECT
            | tags::MAX_ELEM
            | tags::MIN_ELEM
            | tags::RESHAPE
            | tags::BROADCAST
            | tags::TRANSPOSE
            | tags::GATHER
            | tags::GATHER_ROW
            | tags::SORT_DESC
            | tags::TOP_K
            | tags::REDUCE_MAX
            | tags::REDUCE_MIN
            | tags::SCATTER_SET
            | tags::SCATTER_ADD
            | tags::CAST
            | tags::MASK_APPLY_PACKED
            | tags::CAUSAL_MASK
            | tags::SLIDING_WINDOW_MASK
            | tags::SINK_WINDOW_MASK => op.args.iter().any(|a| self.denormal.contains(a)),
            _ => true,
        };
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
                let constant = self.splats.contains(&op.args[1]);
                let y = self.arith(op.tag, d, a, b, constant)?;
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
                // Integers scan in wrapping arithmetic, as the interpreter
                // does (`i32::wrapping_add`, `u32::wrapping_mul`): XLA's
                // integer add and multiply wrap, and both stay associative
                // wrapped, so the doubling scan lands the walk's answer.
                if dt(self, 0) == Dtype::Bool {
                    return refuse("a scan over bools");
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
                if self.splats.contains(&id) {
                    self.splats.insert(result);
                }
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
                if self.splats.contains(&id) {
                    self.splats.insert(result);
                }
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
                let (values, index) = if k < n {
                    self.top_desc(grid, k)?
                } else {
                    self.sort_desc(grid)?
                };
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
                    self.keyed_seed(op, 0)?
                };
                let y = self.draw(seed, n, op.rng_kind)?;
                let y = self.reshaped(y, &shape)?;
                self.set(result, y)
            }
            tags::GUMBEL_ARGMAX => {
                // Each lane `logits[i] * scale + gumbel(state, i)`, `-inf`
                // where `keep` is false, in f32 from its operands as loaded
                // and never stored (a bf16 store would collapse near ties),
                // then `reduce_argmax`: the unfused draw's operations in its
                // order, `rng_keyed`'s noise over the logits' shape. `scale`
                // is the host's rounded reciprocal of the temperature.
                let (il, x) = self.raw_arg(op, 0)?;
                if self.dtype(il) != Dtype::F32 {
                    return refuse("a gumbel draw over non-float logits");
                }
                let row = self.shape(il);
                let n = numel(&row) as i64;
                let keep = self.arg(op, 1, &row)?;
                let scale = self.arg(op, 2, &row)?;
                let scaled = self.f.mul(x, scale)?;
                let floor = self.splat_f(scaled, f32::NEG_INFINITY);
                let masked = self.f.select(keep, scaled, floor)?;
                let seed = self.keyed_seed(op, 3)?;
                let noise = self.draw(seed, n, RngKind::Gumbel)?;
                let noise = self.reshaped(noise, &row)?;
                let perturbed = self.f.add(masked, noise)?;
                let (rows, _, _) = self.rows_of(perturbed)?;
                let y = self.argmax_f32(rows)?;
                let y = self.reshaped(y, &shape)?;
                self.set(result, y)
            }
            other => refuse(format!("`{}` has no StableHLO form here", op_name(other))),
        }
    }

    fn cast(&mut self, x: Val, from: Dtype, to: Dtype) -> Lowering<Val> {
        // bf16 widens exactly (it is held as its f32); a cast to bf16 is the
        // f32's, rounded by the result's store
        let f32_of = |d: Dtype| if d == Dtype::Bf16 { Dtype::F32 } else { d };
        let (from, to) = (f32_of(from), f32_of(to));
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

    fn arith(&mut self, tag: u8, d: Dtype, a: Val, b: Val, constant: bool) -> Lowering<Val> {
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
                // TPU has no integer divide: XLA's expansion of one by a
                // varying divisor compiles ~36 s for a vocabulary-wide
                // vector on v6e, so that divides by long division in
                // shifts; by a constant, XLA's multiply and shifts.
                let q = match (constant, tag == tags::DIV) {
                    (true, true) => self.f.div(a, safe)?,
                    (true, false) => self.f.rem(a, safe)?,
                    (false, div) => {
                        let (quotient, remainder) =
                            int_divrem(&mut self.f, a, safe, d == Dtype::I32)?;
                        if div { quotient } else { remainder }
                    }
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
                let nan = self.is_nan(x)?;
                let real = self.f.not(nan);
                let count = self.f.convert(real, Elem::I32);
                let count = self.f.reduce(count, &[2], Fold::Sum)?;
                let t = self.f.min(k, count)?;
                // the t-th largest key by bisection, a masked count a round,
                // without a sort (v6e, 4 top-k sampled lanes: the sort was
                // 546 of the stage's 979 us)
                let (ukey, pivot) = kth_ukey(&mut self.f, key, real, t, &grid)?;
                let pivot = self.f.broadcast(pivot, &grid, &[0, 1])?;
                let zero = self.f.const_i(Elem::I32, 0, &[b, rows]);
                let some = self.f.compare(Cmp::Gt, t, zero)?;
                let some = self.f.broadcast(some, &grid, &[0, 1])?;
                let above = self.f.compare(Cmp::Ge, ukey, pivot)?;
                let keep = self.f.and(above, real)?;
                self.f.and(keep, some)?
            }
            1 => {
                // cummass_le(p): in descending order (order keys, ties by the
                // lower index), a lane is kept while the mass before it is
                // below p.
                if pd != Dtype::F32 {
                    return refuse("cummass_le over a non-float p");
                }
                let key = self.order_key(x)?;
                let p = self.f.broadcast(per_row, &grid, &[0, 1])?;
                let zero_f = self.f.const_f(Elem::F32, 0.0, &grid);
                let negative = self.f.compare(Cmp::Lt, x, zero_f)?;
                let all: Vec<i64> = vec![0, 1, 2];
                let any = self.f.reduce(negative, &all, Fold::Or)?;
                let branch = self.f.convert(any, Elem::I32);
                let out_ty = Ty::new(Elem::I32, &grid);
                // With no negative lane the mass before a lane only grows
                // along the order, so the kept lanes are a prefix of it: the
                // levels (keys) above a pivot level whole, and the pivot
                // level's lanes up to a pivot index. Both are found by
                // bisection, a masked reduce a round, without a sort
                // (v6e census, sampled c64: the sort was most of the step).
                let cut = Box::new(move |f: &mut Func| -> kernels_xla::hlo::Built<Vec<Val>> {
                    let keep = nucleus_prefix(f, x, key, p, per_row, &grid)?;
                    Ok(vec![f.convert(keep, Elem::I32)])
                })
                    as Box<dyn FnOnce(&mut Func) -> kernels_xla::hlo::Built<Vec<Val>>>;
                // With a negative lane the mass before a lane need not grow:
                // the lanes in order, their verdicts sorted back by index.
                let general = Box::new(move |f: &mut Func| -> kernels_xla::hlo::Built<Vec<Val>> {
                    let index = f.iota(Elem::I32, &grid, 2);
                    let sorted = f.sort(&[key, index], 2, true, |f, a, b| {
                        f.compare(Cmp::Gt, a[0], b[0])
                    })?;
                    let values = key_value_of(f, sorted[0])?;
                    let before = scan_sum_exclusive(f, values)?;
                    let kept = f.compare(Cmp::Lt, before, p)?;
                    let kept = f.convert(kept, Elem::I32);
                    let back = f.sort(&[sorted[1], kept], 2, false, |f, a, b| {
                        f.compare(Cmp::Lt, a[0], b[0])
                    })?;
                    Ok(vec![back[1]])
                })
                    as Box<dyn FnOnce(&mut Func) -> kernels_xla::hlo::Built<Vec<Val>>>;
                let verdict = self.f.case(branch, &[out_ty], vec![cut, general])?;
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

/// Rows an f32 tile holds on v6e's (and v5e's) vector unit, `T(8,128)`: a
/// reduce along the lanes of fewer rows leaves sublanes idle.
const F32_SUBLANES: i64 = 8;

/// Lanes a tile holds, `T(8,128)`.
const LANE_TILE: i64 = 128;

/// `dims` as `[rows, n]`: every axis but the last folded into the rows
/// (`chlo.top_k`'s operand).
fn flat_rows(dims: &[i64]) -> Vec<i64> {
    let (&n, lead) = dims.split_last().expect("a sorted value has an axis");
    vec![lead.iter().product(), n]
}

/// The value an order key stands for (-0 read as +0, NaN as a NaN).
fn key_value_of(f: &mut Func, key: Val) -> kernels_xla::hlo::Built<Val> {
    let flip = f.like_i(key, 0x7FFF_FFFF);
    let flipped = f.xor(key, flip)?;
    let zero = f.like_i(key, 0);
    let negative = f.compare(Cmp::Lt, key, zero)?;
    let bits = f.select(negative, flipped, key)?;
    f.bitcast(bits, Elem::F32)
}

/// The exclusive prefix sum of `x` (f32) along its last axis: log2(n)
/// shifted adds.
fn scan_sum_exclusive(f: &mut Func, x: Val) -> kernels_xla::hlo::Built<Val> {
    let dims = f.dims(x).to_vec();
    let r = dims.len();
    let n = dims[r - 1];
    let zero = f.const_f(Elem::F32, 0.0, &[]);
    let shift = |f: &mut Func, v: Val, k: i64| -> kernels_xla::hlo::Built<Val> {
        let mut hi = dims.clone();
        hi[r - 1] = n - k;
        let head = f.slice(v, &vec![0; r], &hi, &vec![1; r])?;
        let mut lo = vec![0; r];
        lo[r - 1] = k;
        f.pad(head, zero, &lo, &vec![0; r], &vec![0; r])
    };
    let mut v = shift(f, x, 1)?;
    let mut k = 1;
    while k < n {
        let moved = shift(f, v, k)?;
        v = f.add(v, moved)?;
        k *= 2;
    }
    Ok(v)
}

/// The kept lanes of `cummass_le(p)` over a row with no negative lane, as
/// `[B, R, n]` bools: in descending order (order keys `key`, ties by the
/// lower index), a lane is kept while the mass before it is below `p`
/// (`[B, R, n]`, its row's `p` broadcast; `p_row` `[B, R]`). The kept lanes
/// are a prefix of the order, so the pivot is found by bisection: the
/// smallest key `T` whose mass above is below `p` (32 rounds over the keys
/// as unsigned), the pivot level the least key at or above it (every level
/// above is kept whole: its mass before is bounded by a kept key's), and in
/// that level the last index whose mass before (the mass above the level,
/// and the level's lanes of lower index) is below `p` (one round a bit of
/// the row's length). A NaN lane (the least key) adds a NaN to any mass it
/// enters, as the order's running sum does.
/// The keys as unsigned in the same order (`ukey`), and each row's `t`-th
/// largest of the real lanes' (`[B, R]` u32): the largest `K` with at least
/// `t` real lanes at or above it, by bisection over the u32 range, a masked
/// count a round (32 rounds), the sorted order's `t`-th key without a sort.
/// A row with `t` 0 answers 0 (its lanes are cut by the caller).
fn kth_ukey(
    f: &mut Func,
    key: Val,
    real: Val,
    t: Val,
    grid: &[i64; 3],
) -> kernels_xla::hlo::Built<(Val, Val)> {
    let row = [grid[0], grid[1]];
    let sign = f.like_i(key, i64::from(i32::MIN));
    let ukey = f.xor(key, sign)?;
    let ukey = f.bitcast(ukey, Elem::U32)?;
    let count_ge = |f: &mut Func, at: Val| -> kernels_xla::hlo::Built<Val> {
        let at = f.broadcast(at, grid, &[0, 1])?;
        let ge = f.compare(Cmp::Ge, ukey, at)?;
        let ge = f.and(ge, real)?;
        let n = f.convert(ge, Elem::I32);
        f.reduce(n, &[2], Fold::Sum)
    };
    // count_ge(lo) >= t holds from lo = 0 (t is at most the real lanes);
    // the answer lies in [lo, hi]: take the upper half while it holds
    let lo0 = f.const_i(Elem::U32, 0, &row);
    let hi0 = f.const_i(Elem::U32, i64::from(u32::MAX), &row);
    let out = f.for_loop(32, &[lo0, hi0], |f, _, st| {
        let (lo, hi) = (st[0], st[1]);
        let span = f.sub(hi, lo)?;
        let one = f.like_i(span, 1);
        let half = f.shr(span, one)?;
        let odd = f.and(span, one)?;
        let half = f.add(half, odd)?;
        let mid = f.add(lo, half)?;
        let c = count_ge(f, mid)?;
        let holds = f.compare(Cmp::Ge, c, t)?;
        let below = f.sub(mid, one)?;
        let lo = f.select(holds, mid, lo)?;
        let hi = f.select(holds, hi, below)?;
        Ok(vec![lo, hi])
    })?;
    Ok((ukey, out[0]))
}

fn nucleus_prefix(
    f: &mut Func,
    x: Val,
    key: Val,
    p: Val,
    p_row: Val,
    grid: &[i64; 3],
) -> kernels_xla::hlo::Built<Val> {
    let [b, rows, n] = *grid;
    let row = [b, rows];
    // the keys as unsigned, in the same order
    let sign = f.like_i(key, i64::from(i32::MIN));
    let ukey = f.xor(key, sign)?;
    let ukey = f.bitcast(ukey, Elem::U32)?;
    let zero_x = f.const_f(Elem::F32, 0.0, grid);
    // the mass of the lanes whose key is above `at` (`[B, R]` u32)
    let mass_above = |f: &mut Func, at: Val| -> kernels_xla::hlo::Built<Val> {
        let at = f.broadcast(at, grid, &[0, 1])?;
        let above = f.compare(Cmp::Gt, ukey, at)?;
        let v = f.select(above, x, zero_x)?;
        f.reduce(v, &[2], Fold::Sum)
    };
    // T: the least u32 K with mass(key > K) < p. mass(key > max) is 0.
    let lo0 = f.const_i(Elem::U32, 0, &row);
    let hi0 = f.const_i(Elem::U32, i64::from(u32::MAX), &row);
    let m0 = mass_above(f, lo0)?;
    let below0 = f.compare(Cmp::Lt, m0, p_row)?;
    let out = f.for_loop(32, &[lo0, hi0], |f, _, st| {
        let (lo, hi) = (st[0], st[1]);
        let span = f.sub(hi, lo)?;
        let one = f.like_i(span, 1);
        let half = f.shr(span, one)?;
        let mid = f.add(lo, half)?;
        let m = mass_above(f, mid)?;
        let ok = f.compare(Cmp::Lt, m, p_row)?;
        let hi = f.select(ok, mid, hi)?;
        let lo = f.select(ok, lo, mid)?;
        Ok(vec![lo, hi])
    })?;
    let t = f.select(below0, lo0, out[1])?;
    // the pivot level: the least key at or above T
    let t_grid = f.broadcast(t, grid, &[0, 1])?;
    let at_or_above = f.compare(Cmp::Ge, ukey, t_grid)?;
    let top = f.like_i(ukey, i64::from(u32::MAX));
    let candidates = f.select(at_or_above, ukey, top)?;
    let level = f.reduce(candidates, &[2], Fold::Min)?;
    let above = mass_above(f, level)?;
    let level_grid = f.broadcast(level, grid, &[0, 1])?;
    let in_level = f.compare(Cmp::Eq, ukey, level_grid)?;
    let level_x = f.select(in_level, x, zero_x)?;
    let col = f.iota(Elem::I32, grid, 2);
    // J: the last index whose mass before is below p (index 0 always is:
    // the mass above the level is below p, the level's key at or above T)
    let mass_before = |f: &mut Func, j: Val| -> kernels_xla::hlo::Built<Val> {
        let j = f.broadcast(j, grid, &[0, 1])?;
        let earlier = f.compare(Cmp::Lt, col, j)?;
        let v = f.select(earlier, level_x, zero_x)?;
        let s = f.reduce(v, &[2], Fold::Sum)?;
        f.add(above, s)
    };
    let bits = 64 - (n as u64).leading_zeros() as i64;
    let jlo = f.const_i(Elem::I32, 0, &row);
    let jhi = f.const_i(Elem::I32, n, &row);
    let out = f.for_loop(bits, &[jlo, jhi], |f, _, st| {
        let (lo, hi) = (st[0], st[1]);
        let span = f.sub(hi, lo)?;
        let one = f.like_i(span, 1);
        let half = f.shr(span, one)?;
        let mid = f.add(lo, half)?;
        let next = f.add(lo, one)?;
        let mid = f.max(mid, next)?;
        let m = mass_before(f, mid)?;
        let ok = f.compare(Cmp::Lt, m, p_row)?;
        let lo = f.select(ok, mid, lo)?;
        let hi = f.select(ok, hi, mid)?;
        Ok(vec![lo, hi])
    })?;
    let last = out[0];
    // the kept lanes: above the level, or in it up to the last index; none
    // where p is not above zero (the first lane's mass before, 0, is not
    // below it)
    let level_up = f.compare(Cmp::Gt, ukey, level_grid)?;
    let last_grid = f.broadcast(last, grid, &[0, 1])?;
    let early = f.compare(Cmp::Le, col, last_grid)?;
    let tie = f.and(in_level, early)?;
    let keep = f.or(level_up, tie)?;
    let positive = f.compare(Cmp::Gt, p, zero_x)?;
    f.and(keep, positive)
}
