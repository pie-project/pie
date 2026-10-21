//! One eta stage (`LaunchPackage::stages[at]`) as one CSL program over a
//! rectangle of PEs: lane `y` of the batch runs on row `y`, spread over the
//! row's `cols` PEs along its wide axis (the vocabulary): every value whose
//! last dimension is the wide axis is a *wide* value, each PE holding one
//! block of it; every other value is *local*, replicated on the row's PEs.
//! With one column the lane's whole pass sits in one PE and the stage is a
//! straight-line function; with several, the stage is a chain of segments
//! and the reductions over the wide axis (sums, maxima, argmax, cumulative
//! sums, top-k, gathers) combine across the row through the SDK's 2-D
//! collectives: the PEs gather their partials to the row's first PE, which
//! merges and broadcasts the result, and the next segment runs when the
//! broadcast lands.
//!
//! The semantics are the host interpreter's (`eta_exec::eval_op`), op by op:
//! where the interpreter is exact (integer arithmetic, comparisons, argmax,
//! orderings and their ties, masks, the uniform draw's bits, the 32-lane
//! tree a float reduction folds in on one PE) so is this, and float math
//! agrees within rounding (`math.log_f32`/`cos_f32` differ from libm in the
//! last bit; a reduction split over PEs folds its partials in another
//! order).
//!
//! The program's parameters are one packed `u32 [B·cols, in_words]` plane
//! (one row per PE: every host-fed root the PE holds, bit-exact, f32/i32/u32
//! one word per element, bools 32 per word, a wide root as the PE's block,
//! a readout intrinsic's rows as the PE's block of each row) and its result
//! one `u32 [B·cols, out_words]` plane, packed the same way (a local output
//! is read from the row's first PE, a wide one block by block).
//!
//! What does not fit the PEs is refused: the stage then runs in the host
//! interpreter.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;

use eta_compiler::codegen::launch::{LaunchOp, LaunchPackage, LaunchStage, ValueOrigin};
use eta_ir::op::{IntrinsicId, tags};
use eta_ir::{Dtype, RngKind, rng};
use kernels_cerebras::csl::Block;
use kernels_cerebras::program::{Export, Manifest, Rendered, Shard, Symbol};

/// Why a stage has no CSL form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refused(pub String);

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

type Lowering<T> = Result<T, Refused>;

fn refuse<T>(why: impl Into<String>) -> Lowering<T> {
    Err(Refused(why.into()))
}

/// The shape of one lowering: how many lanes, how many PEs each lane
/// spreads over, and the width of the readout planes the host cuts rows
/// from (`None`: the intrinsic is fed as a value).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Batch {
    pub lanes: u32,
    pub cols: u32,
    pub logits: Option<u32>,
    pub mtp: Option<u32>,
}

/// What a packed slot carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feed {
    /// The value itself: `dtype` over `numel` elements (a wide value: the
    /// PE's block of them).
    Value(Dtype, usize),
    /// `rows` rows of `width` f32 words the host cuts from the readout
    /// `intrinsic` reads, starting at the lane's first row (a wide feed:
    /// the PE's block of each row).
    Rows {
        intrinsic: IntrinsicId,
        rows: u32,
        width: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    /// The package-global value id.
    pub value: u32,
    pub feed: Feed,
    /// Whether the value spreads over the row's PEs along its last axis.
    pub wide: bool,
    /// Whether the value travels in its own buffer (`in{k}` / `out{k}`
    /// for the k-th slot): the value's array on the PE is the buffer, so
    /// nothing is copied. Bools travel bit-packed in the `pack` / `out`
    /// buffers instead.
    pub direct: bool,
    /// First word in the PE's packed row (a direct slot: 0).
    pub at: usize,
    /// Words in the PE's packed row (a direct slot: its buffer's).
    pub words: usize,
    /// A wide slot's spread axis (the vocabulary, or the words of a packed
    /// mask over it) and the elements of it each PE holds.
    pub axis: usize,
    pub block: usize,
}

/// A buffer's name and its words a PE.
pub type Named = (String, usize);

/// How a lowered stage's buffers are laid out, per PE.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Layout {
    pub inputs: Vec<Slot>,
    /// Words of the `pack` buffer (the bit-packed bool inputs).
    pub in_words: usize,
    pub outputs: Vec<Slot>,
    /// Words of the `out` buffer (the bit-packed bool outputs).
    pub out_words: usize,
    /// PEs a lane spreads over.
    pub cols: u32,
    /// The wide axis (the vocabulary), when the stage has one.
    pub wide: Option<u32>,
    /// Elements of the wide axis each PE holds (`block_len`): PE `x` holds
    /// `[x · block, min(v, (x + 1) · block))`, the last PEs possibly fewer
    /// or none.
    pub block: usize,
    /// Debug dumps (`PIE_CEREBRAS_GUEST_DEBUG=array:words,...`): extra
    /// output buffers `dbg{i}` reading the named PE arrays after the run.
    pub debug: Vec<Named>,
}

impl Layout {
    /// Whether the stage returns anything (else there is nothing to run).
    #[must_use]
    pub fn runs(&self) -> bool {
        !self.outputs.is_empty()
    }

    /// The buffer name of input slot `k`.
    #[must_use]
    pub fn input_name(&self, k: usize) -> String {
        if self.inputs[k].direct {
            format!("in{k}")
        } else {
            "pack".to_string()
        }
    }

    /// The buffer name of output slot `k`.
    #[must_use]
    pub fn output_name(&self, k: usize) -> String {
        if self.outputs[k].direct {
            format!("out{k}")
        } else {
            "out".to_string()
        }
    }

    /// The stage's buffers in signature order: the direct inputs, `pack`
    /// when any bool input, the direct outputs, `out` when any bool output;
    /// each with its words a PE.
    #[must_use]
    pub fn buffers(&self) -> (Vec<Named>, Vec<Named>) {
        let mut ins: Vec<Named> = self
            .inputs
            .iter()
            .enumerate()
            .filter(|(_, s)| s.direct)
            .map(|(k, s)| (format!("in{k}"), s.words))
            .collect();
        if self.in_words > 0 {
            ins.push(("pack".to_string(), self.in_words));
        }
        let mut outs: Vec<Named> = self
            .outputs
            .iter()
            .enumerate()
            .filter(|(_, s)| s.direct)
            .map(|(k, s)| (format!("out{k}"), s.words))
            .collect();
        if self.out_words > 0 {
            outs.push(("out".to_string(), self.out_words));
        }
        outs.extend(self.debug.iter().cloned());
        (ins, outs)
    }
}

#[derive(Clone, Debug)]
pub struct Lowered {
    /// The rendered program (see `crate::trace::render`); empty when the
    /// stage puts nothing.
    pub text: String,
    pub layout: Layout,
    /// PEs the program runs on (`lanes × cols`).
    pub pes: u32,
    /// Data words one PE holds.
    pub words: u64,
    /// Every buffer of the program, for the device signature.
    pub buffers: Vec<Export>,
}

/// The data words one PE gives a guest stage: well under the 48 KB a PE
/// holds, since the stage's code (one function per segment), the
/// collectives library and the task table share it
/// (`PIE_CEREBRAS_GUEST_WORDS` overrides it).
#[must_use]
pub fn guest_words() -> u64 {
    std::env::var("PIE_CEREBRAS_GUEST_WORDS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(6200)
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

/// The wide axis of `package`: the last dimension of its logits intrinsic
/// (the vocabulary), when it reads one.
#[must_use]
pub fn wide_axis(package: &LaunchPackage) -> Option<u32> {
    package
        .values
        .iter()
        .filter(|v| {
            v.source == ValueOrigin::Intrinsic
                && matches!(
                    v.intrinsic,
                    Some(IntrinsicId::Logits | IntrinsicId::MtpLogits)
                )
        })
        .filter_map(|v| v.shape.last().copied())
        .max()
}

/// The stage's values that are packed masks over the `v`-wide axis: `u32`
/// values of `v.div_ceil(32)` words whose every use is `mask_apply_packed`'s
/// mask, or a reshape (an identity kernel call) into one. They spread with
/// the axis, a PE holding its block's words; any other use keeps a value
/// local (replicated), since its words would then index by position.
fn packed_masks(package: &LaunchPackage, stage: &LaunchStage, v: u32) -> HashSet<u32> {
    let words = v.div_ceil(32);
    let candidate = |id: u32| -> bool {
        let value = &package.values[id as usize];
        value.dtype == Dtype::U32 && value.shape.last() == Some(&words) && words != v
    };
    let mut packed: HashSet<u32> = HashSet::new();
    loop {
        let mut changed = false;
        'ids: for id in 0..package.values.len() as u32 {
            if packed.contains(&id) || !candidate(id) {
                continue;
            }
            let mut used = false;
            for op in &stage.ops {
                let arity = op.args.len();
                for (k, &arg) in op.args.iter().enumerate() {
                    if arg != id {
                        continue;
                    }
                    used = true;
                    let fine = (op.tag == tags::MASK_APPLY_PACKED && k == 1)
                        || ((op.tag == tags::RESHAPE || op.tag == tags::KERNEL_CALL)
                            && arity == 1
                            && packed.contains(&op.result_id));
                    if !fine {
                        continue 'ids;
                    }
                }
                if op.tag == tags::PIVOT_THRESHOLD && op.pred_payload == id {
                    continue 'ids;
                }
            }
            if used {
                packed.insert(id);
                changed = true;
            }
        }
        if !changed {
            return packed;
        }
    }
}

/// Columns a lane may spread over for a wide axis `v`, fewest first: up to
/// 1024 (the compiler and the fabric's width judge the rest), and no more
/// than leave each PE a block of 32.
#[must_use]
pub fn column_choices(v: u32) -> Vec<u32> {
    let most = v.div_ceil(32).clamp(1, 1024);
    (1..=most).collect()
}

/// Elements of a `v`-wide axis each of `cols` PEs holds: the ceiling share,
/// rounded up to whole 32-lane words (so packed bools split with the
/// blocks). PE `x` holds `min(block, v - x · block)` of them, at least 0.
#[must_use]
pub fn block_len(v: u32, cols: u32) -> usize {
    if cols <= 1 {
        return v as usize;
    }
    (v as usize).div_ceil(cols as usize).div_ceil(32) * 32
}

/// The element type a value computes in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ty {
    F32,
    I32,
    U32,
    /// Stored as a u32 lane of 0 or 1.
    Bool,
}

impl Ty {
    fn of(dtype: Dtype) -> Lowering<Ty> {
        Ok(match dtype {
            Dtype::F32 => Ty::F32,
            Dtype::I32 => Ty::I32,
            Dtype::U32 => Ty::U32,
            Dtype::Bool => Ty::Bool,
            other => return refuse(format!("{other:?} is not a dtype eta computes in")),
        })
    }

    fn elem(self) -> &'static str {
        match self {
            Ty::F32 => "f32",
            Ty::I32 => "i32",
            Ty::U32 => "u32",
            // A bool lane is a 16-bit 0 or 1: half the words of a u32.
            Ty::Bool => "u16",
        }
    }

    /// `expr` (a lane of this type) as a u32 word.
    fn to_word(self, expr: &str) -> String {
        match self {
            Ty::U32 => expr.to_string(),
            Ty::Bool => format!("@as(u32, {expr})"),
            _ => format!("@bitcast(u32, {expr})"),
        }
    }

    /// A u32 word `expr` as a lane of this type.
    fn of_word(self, expr: &str) -> String {
        match self {
            Ty::U32 => expr.to_string(),
            Ty::Bool => format!("@as(u16, {expr} & 1)"),
            Ty::F32 => format!("@bitcast(f32, {expr})"),
            Ty::I32 => format!("@bitcast(i32, {expr})"),
        }
    }
}

/// A value's array on the PE.
#[derive(Clone, Debug)]
struct Var {
    name: String,
    ty: Ty,
    /// Elements on this PE (a wide value: its block).
    n: usize,
    /// Whether the last axis spreads over the row's PEs.
    wide: bool,
    /// A wide value that is a packed mask: its block is `block / 32` words
    /// of the mask, not lanes of the axis.
    packed: bool,
}

/// A bit-exact f32 literal.
fn flit(x: f32) -> String {
    format!("@bitcast(f32, @as(u32, {:#x}))", x.to_bits())
}

fn ilit(x: i32) -> String {
    format!("@bitcast(i32, @as(u32, {:#x}))", x as u32)
}

fn ulit(x: u32) -> String {
    format!("@as(u32, {x:#x})")
}

fn u64lit(x: u64) -> String {
    format!("@as(u64, {x:#x})")
}

/// A collective a segment ends with.
#[derive(Clone, Debug)]
enum Collective {
    /// Every PE's `count` words of `send` land in the root's `recv`, PE by PE.
    Gather {
        send: String,
        recv: String,
        count: usize,
    },
    /// The root's `count` words of `buf` land in every PE's `buf`.
    Broadcast { buf: String, count: usize },
    /// The f32 sums over the PEs of `count` words of `send` land in the
    /// root's `recv` (the fabric adds; the SDK's `reduce_fadds`).
    Reduce {
        send: String,
        recv: String,
        count: usize,
    },
    /// No transfer: the next segment runs at once.
    Fall,
    /// Back to segment `to` while `counter` (a u32 global the segments
    /// advance) is below `times`, else on to the next segment.
    Jump {
        to: usize,
        counter: String,
        times: usize,
    },
}

/// Lowers stage `at` of `package` for `batch`.
#[allow(clippy::too_many_lines)]
pub fn lower(package: &LaunchPackage, at: usize, batch: Batch) -> Lowering<Lowered> {
    if at >= package.stages.len() {
        return refuse(format!("the package has no stage {at}"));
    }
    let stage = &package.stages[at];
    let roots = roots(package, at);
    let lanes = batch.lanes.max(1);
    let cols = batch.cols.max(1);
    let wide = if cols > 1 {
        let v = wide_axis(package)
            .ok_or_else(|| Refused("a lane over several PEs needs a wide axis".into()))?;
        if cols > v.div_ceil(32) {
            return refuse(format!(
                "a {v}-wide axis leaves {cols} PEs less than a word each"
            ));
        }
        Some(v)
    } else {
        None
    };
    let block = wide.map_or(1, |v| block_len(v, cols));
    // Packed masks (`u32 [v / 32]` words, read only by `mask_apply_packed`
    // or reshaped into such a read) spread with the axis: a PE holds its
    // block's words.
    let packed = wide.map_or_else(HashSet::new, |v| packed_masks(package, stage, v));
    // A value's spread: `(axis, elements of it this PE holds)`.
    let spread = |id: u32| -> Option<(usize, usize)> {
        let v = wide?;
        let shape = &package.values[id as usize].shape;
        if packed.contains(&id) {
            Some((v.div_ceil(32) as usize, block / 32))
        } else if shape.last() == Some(&v) {
            Some((v as usize, block))
        } else {
            None
        }
    };
    let is_wide = |id: u32| -> bool { spread(id).is_some() };
    // A wide value's PE share: its rows, a block each.
    let block_of = |dtype: Dtype, id: u32| -> usize {
        let n = numel(&package.values[id as usize].shape);
        match spread(id) {
            Some((axis, blk)) => words(dtype, (n / axis) * blk),
            None => words(dtype, n),
        }
    };

    let mut layout = Layout {
        cols,
        wide,
        block,
        ..Layout::default()
    };
    for &id in &roots {
        let value = package
            .values
            .get(id as usize)
            .ok_or_else(|| Refused(format!("value {id} is past the package's values")))?;
        Ty::of(value.dtype)?;
        let n = numel(&value.shape);
        let w = is_wide(id);
        let (axis, blk) = spread(id).unwrap_or((0, 0));
        let feed = match value.source {
            ValueOrigin::Const => continue,
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
                    Some(width) => {
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
                        Feed::Rows {
                            intrinsic,
                            rows,
                            width,
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
        let (per_pe, wide_slot) = match feed {
            Feed::Value(dtype, _) => (block_of(dtype, id), w),
            Feed::Rows { rows, width, .. } => {
                if cols > 1 && Some(width) != wide {
                    return refuse(format!(
                        "a {width}-wide readout beside a {}-wide axis",
                        wide.unwrap_or(0)
                    ));
                }
                (
                    rows as usize * if cols > 1 { block } else { width as usize },
                    cols > 1,
                )
            }
        };
        let direct = !matches!(feed, Feed::Value(Dtype::Bool, _));
        layout.inputs.push(Slot {
            value: id,
            feed,
            wide: wide_slot,
            direct,
            at: if direct { 0 } else { layout.in_words },
            words: per_pe,
            axis,
            block: blk,
        });
        if !direct {
            layout.in_words += per_pe;
        }
    }
    let mut outs_seen = HashSet::new();
    for put in &stage.puts {
        let value = &package.values[put.value as usize];
        let n = numel(&value.shape);
        if !outs_seen.insert(put.value) {
            continue;
        }
        let per_pe = block_of(value.dtype, put.value);
        let direct = value.dtype != Dtype::Bool;
        let (axis, blk) = spread(put.value).unwrap_or((0, 0));
        layout.outputs.push(Slot {
            value: put.value,
            feed: Feed::Value(value.dtype, n),
            wide: is_wide(put.value),
            direct,
            at: if direct { 0 } else { layout.out_words },
            words: per_pe,
            axis,
            block: blk,
        });
        if !direct {
            layout.out_words += per_pe;
        }
    }
    if !layout.runs() {
        return Ok(Lowered {
            text: String::new(),
            layout,
            pes: lanes * cols,
            words: 0,
            buffers: Vec::new(),
        });
    }

    // Where each value is read last (a put value lives to the end).
    let mut last_use: HashMap<u32, usize> = HashMap::new();
    for (i, op) in stage.ops.iter().enumerate() {
        for &arg in &op.args {
            last_use.insert(arg, i);
        }
        if op.tag == tags::PIVOT_THRESHOLD {
            last_use.insert(op.pred_payload, i);
        }
    }
    for put in &stage.puts {
        last_use.insert(put.value, usize::MAX);
    }

    let mut g = Gen {
        b: Block::new(1),
        segments: Vec::new(),
        package,
        env: HashMap::new(),
        scratch: Vec::new(),
        helpers: BTreeSet::new(),
        tmp: 0,
        words: (layout.in_words + layout.out_words) as u64,
        pack: (layout.in_words > 0).then(|| "pack".to_string()),
        out: (layout.out_words > 0).then(|| "out".to_string()),
        exports: Vec::new(),
        debug: Vec::new(),
        packed,
        cols: cols as usize,
        wide,
        last_use,
        arrays: HashMap::new(),
        live: HashMap::new(),
        free: Vec::new(),
        op_temps: Vec::new(),
    };

    // Roots: constants, then the pack's slots.
    for &id in &roots {
        let value = &package.values[id as usize];
        if value.source == ValueOrigin::Const {
            let ty = Ty::of(value.dtype)?;
            let v = g.fresh(ty, 1, false, id);
            let lit = match ty {
                Ty::F32 => flit(f32::from_bits(value.literal_bits)),
                Ty::I32 => ilit(value.literal_bits as i32),
                Ty::U32 => ulit(value.literal_bits),
                Ty::Bool => u32::from(value.literal_bits != 0).to_string(),
            };
            g.b.line(format!("{}[0] = {lit};", v.name));
            g.bind(id, v);
            continue;
        }
        let k = layout
            .inputs
            .iter()
            .position(|s| s.value == id)
            .expect("every non-const root has a slot");
        g.unpack(id, layout.inputs[k], k)?;
    }

    for (i, op) in stage.ops.iter().enumerate() {
        g.op(op).map_err(|Refused(why)| {
            Refused(format!(
                "stage {at}, `{}` ({:#04x}): {why}",
                op_name(op.tag),
                op.tag
            ))
        })?;
        g.release(i);
    }

    for (k, slot) in layout.outputs.iter().enumerate() {
        g.pack(slot, k)?;
    }
    if let Ok(spec) = std::env::var("PIE_CEREBRAS_GUEST_DEBUG") {
        for (i, item) in spec.split(',').filter(|s| !s.is_empty()).enumerate() {
            let (name, words) = item.split_once(':').unwrap_or((item, "8"));
            let words: usize = words.parse().unwrap_or(8);
            if let Some(&(len, elem)) = g.arrays.get(name) {
                let cap = if elem == "u16" { len.div_ceil(2) } else { len };
                let words = words.min(cap).max(1);
                g.debug.push((name.to_string(), words));
                g.exports.push((name.to_string(), elem, format!("dbg{i}")));
                layout.debug.push((format!("dbg{i}"), words));
            } else {
                eprintln!("PIE_CEREBRAS_GUEST_DEBUG: no array `{name}` in this stage");
            }
        }
    }

    // The code shares the PE with the data: a segment (a function and a
    // step) costs near 360 bytes, 90 words, past the 17 the budget assumes.
    let segments = g.segments.len() + 1;
    let budget = guest_words().saturating_sub(90 * segments.saturating_sub(17) as u64);
    if g.words > budget {
        return refuse(format!(
            "one PE's share of a pass holds {} words and a guest PE holds {budget} beside {segments} segments",
            g.words
        ));
    }

    let words = g.words;
    let (pe, layout_text) = g.render(&layout, lanes, cols);
    let pes = lanes * cols;
    let export = |name: &str, width: usize, role: Symbol| Export {
        buf: 0,
        name: name.to_string(),
        rows: pes,
        width: width as u32,
        dtype: dtype::Dtype::U32,
        elem: "u32",
        role,
        shard: Shard::Rows(pes),
        packed: false,
        keep: false,
    };
    // Every buffer is uploaded before the run, in this order, the outputs
    // (as zeros) first: an output's array may be an input's, reused once
    // the input died, and the input's words must land last.
    let (ins, outs) = layout.buffers();
    let exports: Vec<Export> = outs
        .iter()
        .map(|(name, width)| export(name, *width, Symbol::Output))
        .chain(
            ins.iter()
                .map(|(name, width)| export(name, *width, Symbol::Input)),
        )
        .collect();
    let rendered = Rendered {
        layout: layout_text,
        pe,
        manifest: Manifest {
            exports: exports.clone(),
            entry: "run".into(),
            rect: (cols, lanes),
            views: Vec::new(),
            lanes: Vec::new(),
            table: None,
            tables: Vec::new(),
            host: None,
        },
    };
    Ok(Lowered {
        text: crate::trace::render(&rendered),
        layout,
        pes,
        words,
        buffers: exports,
    })
}

struct Gen<'p> {
    /// The segment under construction.
    b: Block,
    /// Finished segments, each ending in the collective it waits on.
    segments: Vec<(Block, Collective)>,
    package: &'p LaunchPackage,
    env: HashMap<u32, Var>,
    /// Per-PE arrays: `(name, elements, element type)`.
    scratch: Vec<(String, usize, &'static str)>,
    helpers: BTreeSet<&'static str>,
    tmp: usize,
    words: u64,
    pack: Option<String>,
    out: Option<String>,
    /// Arrays that are a direct slot's buffer: `(array, element type,
    /// buffer name)`.
    exports: Vec<(String, &'static str, String)>,
    /// Debug dumps: `(array, words)` exported as `dbg{i}`.
    debug: Vec<(String, usize)>,
    /// Values that are packed masks spread with the axis (`packed_masks`).
    packed: HashSet<u32>,
    /// PEs a lane spreads over.
    cols: usize,
    /// The wide axis.
    wide: Option<u32>,
    /// The op index after which each value is dead (`usize::MAX`: put).
    last_use: HashMap<u32, usize>,
    /// Arrays by name: their length and element type.
    arrays: HashMap<String, (usize, &'static str)>,
    /// Values alive under each array (a reshape shares its operand's).
    live: HashMap<String, usize>,
    /// Arrays free for reuse.
    free: Vec<(String, usize, &'static str)>,
    /// Temporaries of the op under construction, freed when it ends.
    op_temps: Vec<String>,
}

impl Gen<'_> {
    /// An array of `n` elements of `elem`: the smallest free one of that
    /// type that holds `n`, else a new one (what counts toward the PE's
    /// words).
    fn array(&mut self, hint: &str, n: usize, elem: &'static str) -> String {
        let n = n.max(1);
        // A free array holds the request when it is big enough and not
        // more than twice it (a small, long-lived value would otherwise
        // sit on a big array a later temporary needs).
        let fit = self
            .free
            .iter()
            .enumerate()
            .filter(|(_, (_, l, e))| *l >= n && *l <= 2 * n && *e == elem)
            .min_by_key(|(_, (_, l, _))| *l)
            .map(|(at, _)| at);
        if let Some(at) = fit {
            let (name, _, _) = self.free.swap_remove(at);
            return name;
        }
        self.tmp += 1;
        let name = format!("{hint}_{}", self.tmp);
        self.scratch.push((name.clone(), n, elem));
        self.arrays.insert(name.clone(), (n, elem));
        // 16-bit lanes pack two a word.
        self.words += if elem == "u16" { n.div_ceil(2) } else { n } as u64;
        name
    }

    /// Binds value `id` to `v` (one more value alive under its array).
    fn bind(&mut self, id: u32, v: Var) {
        *self.live.entry(v.name.clone()).or_insert(0) += 1;
        self.env.insert(id, v);
    }

    /// Frees what dies with op `at`: the values whose last use it was, and
    /// the op's temporaries.
    fn release(&mut self, at: usize) {
        let dead: Vec<u32> = self
            .env
            .keys()
            .copied()
            .filter(|id| self.last_use.get(id).is_none_or(|&l| l <= at))
            .collect();
        for id in dead {
            if let Some(v) = self.env.remove(&id)
                && let Some(n) = self.live.get_mut(&v.name)
            {
                *n -= 1;
                if *n == 0
                    && let Some(&(len, elem)) = self.arrays.get(&v.name)
                {
                    self.free.push((v.name.clone(), len, elem));
                }
            }
        }
        for name in std::mem::take(&mut self.op_temps) {
            if let Some(&(len, elem)) = self.arrays.get(&name) {
                self.free.push((name, len, elem));
            }
        }
    }
    fn shape(&self, id: u32) -> Vec<u32> {
        self.package.values[id as usize].shape.clone()
    }

    fn dtype(&self, id: u32) -> Dtype {
        self.package.values[id as usize].dtype
    }

    /// Value `id`'s spread: `(axis, elements of it this PE holds)`.
    fn spread(&self, id: u32) -> Option<(usize, usize)> {
        let v = self.wide.filter(|_| self.cols > 1)?;
        if self.packed.contains(&id) {
            return Some((v.div_ceil(32) as usize, self.block() / 32));
        }
        (self.shape(id).last() == Some(&v)).then(|| (v as usize, self.block()))
    }

    fn is_wide(&self, id: u32) -> bool {
        self.spread(id).is_some()
    }

    /// The PE's block of the wide axis.
    /// Elements of the wide axis each PE holds (the last PEs may hold
    /// fewer: `bn` of them at run time).
    fn block(&self) -> usize {
        self.wide.map_or(1, |v| block_len(v, self.cols as u32))
    }

    /// The elements of a row of `x` this PE holds: its block of a wide
    /// value (`bn`, set at run time; a packed mask's words of it), `len` of
    /// a local one.
    fn held(x: &Var, len: usize) -> String {
        if x.packed {
            "((bn + 31) >> 5)".to_string()
        } else if x.wide {
            "bn".to_string()
        } else {
            len.to_string()
        }
    }

    /// Refuses `what` over a packed mask, whose block is words, not lanes.
    fn lanes_only(x: &Var, what: &str) -> Lowering<()> {
        if x.packed {
            return refuse(format!("{what} over a packed mask"));
        }
        Ok(())
    }

    fn fresh(&mut self, ty: Ty, n: usize, wide: bool, hint: u32) -> Var {
        let name = self.array(&format!("v{hint}"), n, ty.elem());
        Var {
            name,
            ty,
            n,
            wide,
            packed: wide && self.packed.contains(&hint),
        }
    }

    /// A private array for an op's intermediates, freed when the op ends.
    fn temp(&mut self, elem: &'static str, n: usize) -> String {
        let name = self.array("t", n, elem);
        self.op_temps.push(name.clone());
        name
    }

    fn get(&self, id: u32) -> Lowering<Var> {
        self.env
            .get(&id)
            .cloned()
            .ok_or_else(|| Refused(format!("value {id} is read before anything defines it")))
    }

    fn arg(&self, op: &LaunchOp, i: usize) -> Lowering<Var> {
        let id = *op
            .args
            .get(i)
            .ok_or_else(|| Refused(format!("operand {i} is missing")))?;
        self.get(id)
    }

    /// The result array of `op`'s result `which`, declared and bound.
    fn result(&mut self, op: &LaunchOp, which: u32) -> Lowering<Var> {
        let id = op.result_id + which;
        if id as usize >= self.package.values.len() {
            return refuse(format!("result {id} is past the package's values"));
        }
        let ty = Ty::of(self.dtype(id))?;
        let shape = self.shape(id);
        let n = match self.spread(id) {
            Some((axis, blk)) => (numel(&shape) / axis) * blk,
            None => numel(&shape),
        };
        let wide = self.is_wide(id);
        let v = self.fresh(ty, n, wide, id);
        self.bind(id, v.clone());
        Ok(v)
    }

    /// `v[i]`, or `v[0]` for a one-element operand standing for every lane
    /// (the interpreter's `pick`).
    fn at(v: &Var, i: &str) -> String {
        if v.n == 1 {
            format!("{}[0]", v.name)
        } else {
            format!("{}[{i}]", v.name)
        }
    }

    /// Operands of one elementwise op agree on their spread: every one is
    /// wide, or local of one element, or local of the result's count.
    fn agree(&self, r: &Var, args: &[&Var]) -> Lowering<()> {
        for a in args {
            if a.n == 1 && !a.wide {
                continue;
            }
            if a.wide != r.wide || a.n != r.n {
                return refuse(format!(
                    "an operand of {} element(s) ({}) does not fit a result of {} ({})",
                    a.n,
                    if a.wide { "wide" } else { "local" },
                    r.n,
                    if r.wide { "wide" } else { "local" }
                ));
            }
        }
        Ok(())
    }

    fn loop_var(&mut self) -> String {
        self.tmp += 1;
        format!("li{}", self.tmp)
    }

    /// `for i in 0..n { body(i) }`.
    fn each(&mut self, n: usize, body: impl FnOnce(&mut Block, &str)) {
        let i = self.loop_var();
        self.b.nest(
            &format!("{{ var {i}: i32 = 0; while ({i} < {n}) : ({i} += 1) {{"),
            "} }",
            |blk| body(blk, &i),
        );
    }

    /// `for i in (0..n).rev() { body(i) }`.
    fn each_rev(&mut self, n: usize, body: impl FnOnce(&mut Block, &str)) {
        let i = self.loop_var();
        self.b.nest(
            &format!("{{ var {i}: i32 = {n} - 1; while ({i} >= 0) : ({i} -= 1) {{"),
            "} }",
            |blk| body(blk, &i),
        );
    }

    /// Every row of `x` (`rows` of `len`, `held` lanes of each on this PE)
    /// sorted descending by order key into `skey`/`sidx` (`rows * len`
    /// each, row `rr` at `rr * len`): `key`/`idx` (`len` each) hold one
    /// row's keys while it sorts, and the sort's scratch is row 0's slot of
    /// `skey`/`sidx` itself, so the rows go last to first and row 0 lands
    /// last, over its own scratch.
    #[allow(clippy::too_many_arguments)]
    fn sort_rows(
        &mut self,
        xn: &str,
        rows: usize,
        len: usize,
        held: &str,
        key: &str,
        idx: &str,
        skey: &str,
        sidx: &str,
    ) {
        self.need("g_copy");
        self.need("g_sort");
        self.each_rev(rows, |blk, rr| {
            blk.line(format!(
                "g_keys(@ptrcast([*]f32, &{xn}), {rr} * {len}, {held}, {len}, @ptrcast([*]i32, &{key}), @ptrcast([*]u32, &{idx}));"
            ));
            blk.line(format!(
                "g_msort(@ptrcast([*]i32, &{key}), @ptrcast([*]u32, &{idx}), {len}, @ptrcast([*]i32, &{skey}), @ptrcast([*]u32, &{sidx}));"
            ));
            blk.line(format!(
                "g_copy32(@ptrcast([*]u32, &{skey}), {rr} * {len}, @ptrcast([*]u32, &{key}), 0, {len}); g_copy32(@ptrcast([*]u32, &{sidx}), {rr} * {len}, @ptrcast([*]u32, &{idx}), 0, {len});"
            ));
        });
    }

    fn need(&mut self, helper: &'static str) {
        self.helpers.insert(helper);
    }

    /// `dst[doff..doff+n] = src[soff..soff+n]`, word for word.
    fn copy(&mut self, dst: &str, doff: &str, src: &str, soff: &str, n: usize) {
        self.need("g_copy");
        self.b.line(format!(
            "g_copy32(@ptrcast([*]u32, &{dst}), {doff}, @ptrcast([*]u32, &{src}), {soff}, {n});"
        ));
    }

    /// `copy` for arrays of `ty` (bools are 16-bit lanes).
    fn copy_of(&mut self, ty: Ty, dst: &str, doff: &str, src: &str, soff: &str, n: usize) {
        if ty != Ty::Bool {
            return self.copy(dst, doff, src, soff, n);
        }
        self.need("g_copy");
        self.b.line(format!(
            "g_copy16(@ptrcast([*]u16, &{dst}), {doff}, @ptrcast([*]u16, &{src}), {soff}, {n});"
        ));
    }

    /// Ends the segment under construction with `c`; the next statements
    /// run when it lands.
    fn wait(&mut self, c: Collective) {
        let b = std::mem::replace(&mut self.b, Block::new(1));
        self.segments.push((b, c));
    }

    /// Opens a loop over the segments that follow: the segment under
    /// construction ends, and the segments from here to `close_loop` run
    /// `times` times (every PE walks the same iterations in step).
    fn open_loop(&mut self, times: usize) -> (usize, String) {
        let counter = self.temp("u32", 1);
        self.b.line(format!("{counter}[0] = 0;"));
        self.wait(Collective::Fall);
        let _ = times;
        (self.segments.len(), counter)
    }

    fn close_loop(&mut self, (to, counter): (usize, String), times: usize) {
        self.b.line(format!("{counter}[0] = {counter}[0] + 1;"));
        self.wait(Collective::Jump { to, counter, times });
    }

    /// A combine across the row: every PE's `count` words of `send` gather
    /// to the row's first PE, which runs `merge` (reading `recv`, `cols`
    /// blocks of `count` words, writing `RES`, `res_count` words), and the
    /// result is broadcast back. On return every PE holds the result.
    fn combine(
        &mut self,
        send: &str,
        count: usize,
        res_count: usize,
        merge: impl FnOnce(&mut Block, &str),
    ) -> String {
        let recv = self.temp("u32", count * self.cols);
        let res = self.temp("u32", res_count);
        self.wait(Collective::Gather {
            send: send.to_string(),
            recv: recv.clone(),
            count,
        });
        self.b.nest("if (px == 0) {", "}", |blk| merge(blk, &recv));
        self.b.replace("RES", &res);
        self.wait(Collective::Broadcast {
            buf: res.clone(),
            count: res_count,
        });
        // The gathered words are dead once merged: the next combine of this
        // op (or anything after) may take the array.
        self.free_now(&recv);
        res
    }

    /// The f32 sums across the row of `count` words of `send` (f32 bit
    /// patterns; integers under 2^24 travel as f32 exactly), on every PE:
    /// the fabric adds to the root (`count` words there, not `count ×
    /// cols`), which broadcasts.
    fn combine_sum(&mut self, send: &str, count: usize) -> String {
        let res = self.temp("u32", count);
        self.wait(Collective::Reduce {
            send: send.to_string(),
            recv: res.clone(),
            count,
        });
        self.wait(Collective::Broadcast {
            buf: res.clone(),
            count,
        });
        res
    }

    /// Returns an op temporary to the pool before the op ends.
    fn free_now(&mut self, name: &str) {
        if let Some(at) = self.op_temps.iter().position(|t| t == name)
            && let Some(&(len, elem)) = self.arrays.get(name)
        {
            self.op_temps.remove(at);
            self.free.push((name.to_string(), len, elem));
        }
    }

    // ---------------------------------------------------------------- packing

    fn unpack(&mut self, id: u32, slot: Slot, k: usize) -> Lowering<()> {
        let pack = self.pack.clone().unwrap_or_default();
        let value = &self.package.values[id as usize];
        let ty = Ty::of(value.dtype)?;
        let n_total = numel(&value.shape);
        let n = match self.spread(id) {
            Some((axis, blk)) if slot.wide => (n_total / axis) * blk,
            _ => n_total,
        };
        match slot.feed {
            Feed::Value(_, _) if slot.direct => {
                // The value's array is the buffer the host writes.
                let v = self.fresh(ty, n, slot.wide, id);
                self.exports
                    .push((v.name.clone(), ty.elem(), format!("in{k}")));
                self.bind(id, v);
            }
            Feed::Value(_, _) => {
                let v = self.fresh(ty, n, slot.wide, id);
                let at = slot.at;
                self.each(n, |blk, i| {
                    blk.line(format!(
                        "{}[{i}] = @as(u16, ({pack}[{at} + ({i} >> 5)] >> @as(u32, {i} & 31)) & 1);",
                        v.name
                    ));
                });
                self.bind(id, v);
            }
            Feed::Rows {
                intrinsic,
                rows,
                width,
            } => {
                let (rows, width) = (
                    rows as usize,
                    if slot.wide {
                        self.block()
                    } else {
                        width as usize
                    },
                );
                // The rows land in the value's array itself.
                if intrinsic == IntrinsicId::MtpDrafts {
                    if ty != Ty::I32 {
                        return refuse("the draft tokens are i32");
                    }
                    // Each row's argmax, as the interpreter binds the drafts:
                    // the rows as a (wide) f32 value, then the argmax.
                    let plane = self.fresh(Ty::F32, rows * width, slot.wide, id);
                    self.exports
                        .push((plane.name.clone(), "f32", format!("in{k}")));
                    let v = self.fresh(Ty::I32, n, false, id);
                    self.argmax_rows(&plane, rows, width, &v);
                    // The rows were this op's own.
                    self.op_temps.push(plane.name);
                    self.bind(id, v);
                } else {
                    if ty != Ty::F32 {
                        return refuse("a readout is f32");
                    }
                    let v = self.fresh(Ty::F32, n, slot.wide, id);
                    self.exports.push((v.name.clone(), "f32", format!("in{k}")));
                    self.bind(id, v);
                }
            }
        }
        Ok(())
    }

    fn pack(&mut self, slot: &Slot, k: usize) -> Lowering<()> {
        let v = self.get(slot.value)?;
        if v.wide != slot.wide {
            return refuse(format!(
                "value {} is put {} and computed {}",
                slot.value,
                if slot.wide { "wide" } else { "local" },
                if v.wide { "wide" } else { "local" }
            ));
        }
        let n = v.n;
        if slot.direct {
            // The value's array is the buffer the host reads.
            if v.n == 1 && n != 1 {
                return refuse("a one-element array put as a vector");
            }
            self.exports
                .push((v.name.clone(), v.ty.elem(), format!("out{k}")));
            return Ok(());
        }
        let out = self.out.clone().unwrap_or_default();
        let at = slot.at;
        match v.ty {
            Ty::F32 | Ty::I32 | Ty::U32 => return refuse("a non-bool value in the bit-packed out"),
            Ty::Bool => {
                let w = n.div_ceil(32);
                self.each(w, |blk, i| {
                    blk.line(format!("{out}[{at} + {i}] = 0;"));
                });
                self.each(n, |blk, i| {
                    blk.line(format!(
                        "if ({} != 0) {{ {out}[{at} + ({i} >> 5)] = {out}[{at} + ({i} >> 5)] | (@as(u32, 1) << @as(u32, {i} & 31)); }}",
                        Gen::at(&v, i)
                    ));
                });
            }
        }
        Ok(())
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
        match op.tag {
            tags::EXP
            | tags::LOG
            | tags::SIN
            | tags::COS
            | tags::SQRT
            | tags::RSQRT
            | tags::RECIP => {
                let x = self.arg(op, 0)?;
                if x.ty != Ty::F32 {
                    return refuse("a float map over a non-float operand");
                }
                let r = self.result(op, 0)?;
                self.agree(&r, &[&x])?;
                let one = flit(1.0);
                let f = |xi: String| match op.tag {
                    tags::EXP => format!("math.exp_f32({xi})"),
                    tags::LOG => format!("math.log_f32({xi})"),
                    tags::SIN => format!("math.sin_f32({xi})"),
                    tags::COS => format!("math.cos_f32({xi})"),
                    tags::SQRT => format!("math.sqrt_f32({xi})"),
                    tags::RSQRT => format!("{one} / math.sqrt_f32({xi})"),
                    _ => format!("{one} / {xi}"),
                };
                self.each(r.n, |blk, i| {
                    blk.line(format!("{}[{i}] = {};", r.name, f(Gen::at(&x, i))));
                });
                Ok(())
            }
            tags::NEG => {
                let x = self.arg(op, 0)?;
                let r = self.result(op, 0)?;
                self.agree(&r, &[&x])?;
                let e: fn(String) -> String = match x.ty {
                    Ty::F32 => {
                        |xi| format!("@bitcast(f32, @bitcast(u32, {xi}) ^ @as(u32, 0x80000000))")
                    }
                    Ty::I32 => |xi| format!("@as(i32, 0) - {xi}"),
                    Ty::U32 => |xi| format!("@as(u32, 0) - {xi}"),
                    Ty::Bool => return refuse("neg on bool"),
                };
                self.each(r.n, |blk, i| {
                    blk.line(format!("{}[{i}] = {};", r.name, e(Gen::at(&x, i))));
                });
                Ok(())
            }
            tags::ABS => {
                let x = self.arg(op, 0)?;
                let r = self.result(op, 0)?;
                self.agree(&r, &[&x])?;
                self.each(r.n, |blk, i| {
                    let xi = Gen::at(&x, i);
                    match x.ty {
                        Ty::F32 => blk.line(format!("{}[{i}] = math.abs_f32({xi});", r.name)),
                        Ty::I32 => blk.line(format!(
                            "if ({xi} < 0) {{ {0}[{i}] = @as(i32, 0) - {xi}; }} else {{ {0}[{i}] = {xi}; }}",
                            r.name
                        )),
                        _ => blk.line(format!("{}[{i}] = {xi};", r.name)),
                    };
                });
                Ok(())
            }
            tags::SIGN => {
                let x = self.arg(op, 0)?;
                let r = self.result(op, 0)?;
                self.agree(&r, &[&x])?;
                let (one, m1, zero) = match x.ty {
                    Ty::F32 => (flit(1.0), flit(-1.0), flit(0.0)),
                    Ty::I32 => (ilit(1), ilit(-1), ilit(0)),
                    Ty::U32 => (ulit(1), ulit(0), ulit(0)),
                    Ty::Bool => return refuse("sign on bool"),
                };
                self.each(r.n, |blk, i| {
                    let xi = Gen::at(&x, i);
                    if x.ty == Ty::U32 {
                        blk.line(format!(
                            "if ({xi} != 0) {{ {0}[{i}] = {one}; }} else {{ {0}[{i}] = {zero}; }}",
                            r.name
                        ));
                    } else {
                        blk.line(format!(
                            "if ({xi} > {zero}) {{ {0}[{i}] = {one}; }} else if ({xi} < {zero}) {{ {0}[{i}] = {m1}; }} else {{ {0}[{i}] = {zero}; }}",
                            r.name
                        ));
                    }
                });
                Ok(())
            }
            tags::CAST => {
                let x = self.arg(op, 0)?;
                let r = self.result(op, 0)?;
                self.agree(&r, &[&x])?;
                self.cast(&x, &r)
            }
            tags::ADD
            | tags::SUB
            | tags::MUL
            | tags::DIV
            | tags::REM
            | tags::MAX_ELEM
            | tags::MIN_ELEM => {
                let a = self.arg(op, 0)?;
                let b = self.arg(op, 1)?;
                let r = self.result(op, 0)?;
                self.agree(&r, &[&a, &b])?;
                self.arith(op.tag, &a, &b, &r)
            }
            tags::GT | tags::GE | tags::EQ | tags::NE | tags::LT | tags::LE => {
                let a = self.arg(op, 0)?;
                let b = self.arg(op, 1)?;
                if a.ty == Ty::Bool {
                    return refuse("an ordered comparison of bools");
                }
                let r = self.result(op, 0)?;
                self.agree(&r, &[&a, &b])?;
                let sym = match op.tag {
                    tags::GT => ">",
                    tags::GE => ">=",
                    tags::EQ => "==",
                    tags::NE => "!=",
                    tags::LT => "<",
                    _ => "<=",
                };
                // A NaN operand compares false (true for `!=`), stated
                // outright: the compiler's float math may not keep IEEE
                // comparisons on NaN.
                let float = a.ty == Ty::F32;
                self.each(r.n, |blk, i| {
                    let (x, y) = (Gen::at(&a, i), Gen::at(&b, i));
                    let cond = if !float {
                        format!("{x} {sym} {y}")
                    } else if sym == "!=" {
                        format!("math.isNaN_f32({x}) or math.isNaN_f32({y}) or {x} != {y}")
                    } else {
                        format!("!math.isNaN_f32({x}) and !math.isNaN_f32({y}) and {x} {sym} {y}")
                    };
                    blk.line(format!(
                        "if ({cond}) {{ {0}[{i}] = 1; }} else {{ {0}[{i}] = 0; }}",
                        r.name
                    ));
                });
                Ok(())
            }
            tags::AND | tags::OR => {
                let a = self.arg(op, 0)?;
                let b = self.arg(op, 1)?;
                if a.ty != Ty::Bool || b.ty != Ty::Bool {
                    return refuse("and/or on non-bool");
                }
                let r = self.result(op, 0)?;
                self.agree(&r, &[&a, &b])?;
                let sym = if op.tag == tags::AND { "and" } else { "or" };
                self.each(r.n, |blk, i| {
                    blk.line(format!(
                        "if ({1} != 0 {sym} {2} != 0) {{ {0}[{i}] = 1; }} else {{ {0}[{i}] = 0; }}",
                        r.name,
                        Gen::at(&a, i),
                        Gen::at(&b, i)
                    ));
                });
                Ok(())
            }
            tags::NOT => {
                let a = self.arg(op, 0)?;
                if a.ty != Ty::Bool {
                    return refuse("not on non-bool");
                }
                let r = self.result(op, 0)?;
                self.agree(&r, &[&a])?;
                self.each(r.n, |blk, i| {
                    blk.line(format!(
                        "if ({1} == 0) {{ {0}[{i}] = 1; }} else {{ {0}[{i}] = 0; }}",
                        r.name,
                        Gen::at(&a, i)
                    ));
                });
                Ok(())
            }
            tags::SELECT => {
                let c = self.arg(op, 0)?;
                let a = self.arg(op, 1)?;
                let b = self.arg(op, 2)?;
                if c.ty != Ty::Bool || a.ty != b.ty {
                    return refuse("select's condition or arms");
                }
                let r = self.result(op, 0)?;
                self.agree(&r, &[&c, &a, &b])?;
                self.each(r.n, |blk, i| {
                    blk.line(format!(
                        "if ({1} != 0) {{ {0}[{i}] = {2}; }} else {{ {0}[{i}] = {3}; }}",
                        r.name,
                        Gen::at(&c, i),
                        Gen::at(&a, i),
                        Gen::at(&b, i)
                    ));
                });
                Ok(())
            }
            tags::REDUCE_SUM | tags::REDUCE_MAX | tags::REDUCE_MIN => self.reduce(op),
            tags::REDUCE_ARGMAX => self.argmax(op),
            tags::CUMSUM | tags::CUMPROD => self.scan(op),
            tags::BROADCAST => self.broadcast(op),
            tags::RESHAPE | tags::KERNEL_CALL => {
                if op.tag == tags::KERNEL_CALL && op.args.len() != 1 {
                    return refuse("a kernel call that is not the identity boundary");
                }
                let (id, x) = (op.args[0], self.arg(op, 0)?);
                let shape = self.shape(result);
                if self.dtype(id) != self.dtype(result) || numel(&self.shape(id)) != numel(&shape) {
                    return refuse("a reshape that changes the count or the dtype");
                }
                if x.wide != self.is_wide(result) || x.packed != self.packed.contains(&result) {
                    return refuse("a reshape across the wide axis");
                }
                // The same array under the result's id.
                self.bind(result, x);
                Ok(())
            }
            tags::TRANSPOSE => {
                let (id, x) = (op.args[0], self.arg(op, 0)?);
                let dims = self.shape(id);
                if dims.len() != 2 {
                    return refuse("a transpose of other than a matrix");
                }
                if x.wide {
                    return refuse("a transpose of a wide value");
                }
                let (m, n) = (dims[0] as usize, dims[1] as usize);
                let r = self.result(op, 0)?;
                if r.wide {
                    return refuse("a transpose into a wide value");
                }
                let xn = x.name.clone();
                self.each(r.n, |blk, o| {
                    blk.line(format!(
                        "{}[{o}] = {xn}[({o} % {m}) * {n} + {o} / {m}];",
                        r.name
                    ));
                });
                Ok(())
            }
            tags::SORT_DESC | tags::TOP_K => self.order(op),
            tags::MATMUL => {
                let (ia, a) = (op.args[0], self.arg(op, 0)?);
                let (ib, b) = (op.args[1], self.arg(op, 1)?);
                let (sa, sb) = (self.shape(ia), self.shape(ib));
                if sa.len() != 2 || sb.len() != 2 || sa[1] != sb[0] {
                    return refuse("matmul of other than conformant matrices");
                }
                if a.ty != Ty::F32 || b.ty != Ty::F32 {
                    return refuse("matmul of non-floats");
                }
                if a.wide || b.wide {
                    return refuse("matmul over a wide value");
                }
                let (m, kk, n) = (sa[0] as usize, sa[1] as usize, sb[1] as usize);
                let r = self.result(op, 0)?;
                if r.wide {
                    return refuse("matmul into a wide value");
                }
                let zero = flit(0.0);
                self.each(r.n, |blk, i| {
                    blk.line(format!("{}[{i}] = {zero};", r.name));
                });
                let (an, bn, rn) = (a.name.clone(), b.name.clone(), r.name.clone());
                // The interpreter's order: rows, then the contraction, then
                // the columns, skipping a zero left operand.
                self.each(m, |blk, i| {
                    blk.nest(
                        &format!("{{ var l: i32 = 0; while (l < {kk}) : (l += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!("var xv: f32 = {an}[{i} * {kk} + l];"));
                            blk.line(format!("if (xv == {zero}) {{ continue; }}"));
                            blk.nest(
                                &format!("{{ var j: i32 = 0; while (j < {n}) : (j += 1) {{"),
                                "} }",
                                |blk| {
                                    blk.line(format!(
                                        "{rn}[{i} * {n} + j] = {rn}[{i} * {n} + j] + xv * {bn}[l * {n} + j];"
                                    ));
                                },
                            );
                        },
                    );
                });
                Ok(())
            }
            tags::PIVOT_THRESHOLD => self.pivot(op),
            tags::GATHER => self.gather(op),
            tags::GATHER_ROW => self.gather_row(op),
            tags::SCATTER_ADD | tags::SCATTER_SET => self.scatter(op),
            tags::IOTA => {
                let r = self.result(op, 0)?;
                if r.ty != Ty::U32 {
                    return refuse("an iota that is not u32");
                }
                let base = if r.wide {
                    format!("@as(u32, px * {})", self.block())
                } else {
                    "@as(u32, 0)".to_string()
                };
                self.each(r.n, |blk, i| {
                    blk.line(format!("{}[{i}] = {base} + @as(u32, {i});", r.name));
                });
                Ok(())
            }
            tags::MASK_APPLY_PACKED => {
                let (ix, x) = (op.args[0], self.arg(op, 0)?);
                let mask = self.arg(op, 1)?;
                if x.ty != Ty::F32 || mask.ty != Ty::U32 {
                    return refuse("mask_apply of other than f32 logits and u32 words");
                }
                let shape = self.shape(ix);
                let n = *shape.last().unwrap_or(&1) as usize;
                let r = self.result(op, 0)?;
                self.agree(&r, &[&x])?;
                let ninf = flit(f32::NEG_INFINITY);
                let (xn, mn) = (x.name.clone(), mask.name.clone());
                // Column `c` of the row reads bit `c` of the words; a wide
                // logits row is this PE's block of the columns, and a wide
                // mask this PE's block of the words.
                let (len, col0, w, w0) = if r.wide {
                    let block = self.block();
                    let w0 = if mask.wide {
                        format!("px * {}", block / 32)
                    } else {
                        "0".to_string()
                    };
                    (block, format!("px * {block}"), mask.n, w0)
                } else {
                    if mask.wide {
                        return refuse("a wide mask over a local row");
                    }
                    (n, "0".to_string(), mask.n, "0".to_string())
                };
                self.each(r.n, |blk, j| {
                    blk.line(format!(
                        "var c: i32 = {col0} + {j} % {len}; var wi: i32 = (c >> 5) - ({w0}); var word: u32 = 0;"
                    ));
                    blk.line(format!("if (wi >= 0 and wi < {w}) {{ word = {mn}[wi]; }}"));
                    blk.line(format!(
                        "if (((word >> @as(u32, c & 31)) & 1) != 0) {{ {0}[{j}] = {xn}[{j}]; }} else {{ {0}[{j}] = {ninf}; }}",
                        r.name
                    ));
                });
                Ok(())
            }
            tags::CAUSAL_MASK | tags::SLIDING_WINDOW_MASK | tags::SINK_WINDOW_MASK => {
                let pos = self.arg(op, 0)?;
                if pos.ty != Ty::U32 || pos.wide {
                    return refuse("structured mask positions that are not local u32");
                }
                let keys = op.imm as usize;
                let r = self.result(op, 0)?;
                let (local_keys, key0) = if r.wide {
                    (self.block(), format!("@as(u32, px * {})", self.block()))
                } else {
                    (keys, "@as(u32, 0)".to_string())
                };
                if r.n != pos.n * local_keys {
                    return refuse("a structured mask that is not positions x keys");
                }
                let window = if op.tag == tags::SLIDING_WINDOW_MASK {
                    op.imm2
                } else {
                    op.imm3
                };
                let sink = op.imm2;
                let tag = op.tag;
                let (pn, rn) = (pos.name.clone(), r.name.clone());
                self.each(pos.n, |blk, p| {
                    blk.line(format!("var position: u32 = {pn}[{p}];"));
                    blk.nest(
                        &format!("{{ var kl: i32 = 0; while (kl < {local_keys}) : (kl += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!("var key: u32 = {key0} + @as(u32, kl);"));
                            blk.line("var allowed: bool = key <= position;");
                            if tag != tags::CAUSAL_MASK {
                                // `key.saturating_add(window) > position`.
                                blk.line(format!(
                                    "var recent: bool = (key > @as(u32, 0xFFFFFFFF) - @as(u32, {window})) or (key + @as(u32, {window}) > position);"
                                ));
                                if tag == tags::SLIDING_WINDOW_MASK {
                                    blk.line("allowed = allowed and recent;");
                                } else {
                                    blk.line(format!(
                                        "allowed = allowed and (key < @as(u32, {sink}) or recent);"
                                    ));
                                }
                            }
                            blk.line(format!(
                                "if (allowed) {{ {rn}[{p} * {local_keys} + kl] = 1; }} else {{ {rn}[{p} * {local_keys} + kl] = 0; }}"
                            ));
                        },
                    );
                });
                Ok(())
            }
            tags::RNG | tags::RNG_KEYED => {
                let r = self.result(op, 0)?;
                if r.ty != Ty::F32 {
                    return refuse("a draw that is not f32");
                }
                self.need("g_rng");
                let seed = if op.tag == tags::RNG {
                    u64lit(rng::seed_eff_stream(0, op.imm))
                } else {
                    let (is, state) = (op.args[0], self.arg(op, 0)?);
                    if state.ty != Ty::U32 || numel(&self.shape(is)) < 1 || state.wide {
                        return refuse("an rng state that is not local u32 words");
                    }
                    let ctr = if state.n > 1 {
                        format!("@as(u64, {}[1])", state.name)
                    } else {
                        "@as(u64, 0)".to_string()
                    };
                    format!(
                        "g_splitmix((@as(u64, {}[0]) << {}) | {ctr})",
                        state.name,
                        rng::RNG_FORMULA.keyed_word_bits
                    )
                };
                self.tmp += 1;
                let sv = format!("seed{}", self.tmp);
                self.b.line(format!("var {sv}: u64 = {seed};"));
                let seed = sv;
                let rn = r.name.clone();
                // Lane `j` of the draw is global: a wide draw's PE starts at
                // its block in each row of the value.
                let lane0 = if r.wide {
                    let block = self.block();
                    let v = self.wide.unwrap_or(1) as usize;
                    format!("({{j}} / {block}) * {v} + px * {block} + {{j}} % {block}")
                } else {
                    "{j}".to_string()
                };
                let lane = |j: &str| lane0.replace("{j}", j);
                match op.rng_kind {
                    RngKind::Uniform => self.each(r.n, |blk, j| {
                        blk.line(format!(
                            "{rn}[{j}] = g_uniform({seed}, @as(u32, {}));",
                            lane(j)
                        ));
                    }),
                    RngKind::Gumbel => self.each(r.n, |blk, j| {
                        blk.line(format!(
                            "{rn}[{j}] = @bitcast(f32, @bitcast(u32, math.log_f32(@bitcast(f32, @bitcast(u32, math.log_f32(g_uniform({seed}, @as(u32, {})))) ^ @as(u32, 0x80000000)))) ^ @as(u32, 0x80000000));",
                            lane(j)
                        ));
                    }),
                    RngKind::Normal => {
                        let stride = rng::NORMAL_PAIR_STRIDE;
                        let tau = flit(rng::NORMAL_TWO_PI);
                        let m2 = flit(-2.0);
                        self.each(r.n, |blk, j| {
                            blk.line(format!(
                                "var lane: u32 = @as(u32, {}) * @as(u32, {stride});",
                                lane(j)
                            ));
                            blk.line(format!("var un0: f32 = g_uniform({seed}, lane); var un1: f32 = g_uniform({seed}, lane + 1);"));
                            blk.line(format!(
                                "{rn}[{j}] = math.sqrt_f32({m2} * math.log_f32(un0)) * math.cos_f32({tau} * un1);"
                            ));
                        });
                    }
                }
                Ok(())
            }
            other => refuse(format!("`{}` has no CSL form here", op_name(other))),
        }
    }

    fn zero(ty: Ty) -> String {
        match ty {
            Ty::F32 => flit(0.0),
            Ty::I32 => ilit(0),
            Ty::U32 => ulit(0),
            Ty::Bool => "0".to_string(),
        }
    }

    /// Whether the index lane `ix` (of `idx`'s type) is inside `0..bound`.
    fn index_valid(idx: &Var, ix: &str, bound: usize) -> String {
        match idx.ty {
            Ty::U32 => format!("{ix} < @as(u32, {bound})"),
            Ty::I32 => format!("{ix} >= 0 and {ix} < {bound}"),
            Ty::Bool => format!("@as(i32, {ix}) < {bound}"),
            Ty::F32 => format!(
                "({ix} == {ix}) and {ix} >= {} and {ix} < {}",
                flit(0.0),
                flit(bound as f32)
            ),
        }
    }

    /// The row count and per-PE row length of a reduction over `x`'s last
    /// axis.
    fn rows_len(&self, id: u32, x: &Var) -> (usize, usize) {
        let rows = canonical_rows(&self.shape(id)).max(1);
        (rows, x.n / rows)
    }

    fn cast(&mut self, x: &Var, r: &Var) -> Lowering<()> {
        let (from, to) = (x.ty, r.ty);
        // A test and its two arms, or one expression.
        let (test, yes, no): (Option<&str>, String, String) = match (from, to) {
            (a, b) if a == b => (None, "{x}".into(), String::new()),
            (Ty::I32 | Ty::U32, Ty::F32) => (None, "@as(f32, {x})".into(), String::new()),
            (Ty::Bool, Ty::F32) => (Some("{x} != 0"), flit(1.0), flit(0.0)),
            (Ty::F32, Ty::I32) => {
                self.need("g_cast");
                (None, "g_f2i({x})".into(), String::new())
            }
            (Ty::F32, Ty::U32) => {
                self.need("g_cast");
                (None, "g_f2u({x})".into(), String::new())
            }
            (Ty::U32, Ty::I32) => (None, "@bitcast(i32, {x})".into(), String::new()),
            (Ty::Bool, Ty::I32) => (None, "@as(i32, {x})".into(), String::new()),
            (Ty::I32, Ty::U32) => (None, "@bitcast(u32, {x})".into(), String::new()),
            (Ty::Bool, Ty::U32) => (None, "@as(u32, {x})".into(), String::new()),
            // NaN != 0 holds: a NaN lane is true, as `e != 0.0` says.
            (Ty::F32, Ty::Bool) => (
                Some("math.isNaN_f32({x}) or {x} != ZERO"),
                "1".into(),
                "0".into(),
            ),
            (Ty::I32 | Ty::U32, Ty::Bool) => (Some("{x} != 0"), "1".into(), "0".into()),
            _ => return refuse(format!("a cast from {from:?} to {to:?}")),
        };
        let zero = flit(0.0);
        self.each(r.n, |blk, i| {
            let xi = Gen::at(x, i);
            match test {
                None => blk.line(format!("{}[{i}] = {};", r.name, yes.replace("{x}", &xi))),
                Some(t) => blk.line(format!(
                    "if ({1}) {{ {0}[{i}] = {yes}; }} else {{ {0}[{i}] = {no}; }}",
                    r.name,
                    t.replace("{x}", &xi).replace("ZERO", &zero)
                )),
            };
        });
        Ok(())
    }

    fn arith(&mut self, tag: u8, a: &Var, b: &Var, r: &Var) -> Lowering<()> {
        let d = a.ty;
        if d == Ty::Bool {
            return refuse("arithmetic on bools");
        }
        let float = d == Ty::F32;
        let e: Box<dyn Fn(String, String) -> String> = match tag {
            tags::ADD => Box::new(|x, y| format!("{x} + {y}")),
            tags::SUB => Box::new(|x, y| format!("{x} - {y}")),
            tags::MUL => Box::new(|x, y| format!("{x} * {y}")),
            tags::DIV if float => Box::new(|x, y| format!("{x} / {y}")),
            tags::REM if float => {
                self.need("g_fmod");
                Box::new(|x, y| format!("g_fmod({x}, {y})"))
            }
            tags::DIV | tags::REM => {
                // The interpreter divides in i64: by zero is 0, and
                // i32::MIN / -1 wraps (i32::MIN % -1 is 0).
                self.need("g_int");
                let name = match (tag, d) {
                    (tags::DIV, Ty::I32) => "g_idiv",
                    (tags::REM, Ty::I32) => "g_irem",
                    (tags::DIV, _) => "g_udiv",
                    _ => "g_urem",
                };
                Box::new(move |x, y| format!("{name}({x}, {y})"))
            }
            tags::MAX_ELEM | tags::MIN_ELEM if float => {
                // `f32::max`/`min`: a NaN operand yields the other one.
                self.need("g_tree");
                let name = if tag == tags::MAX_ELEM {
                    "g_fmax"
                } else {
                    "g_fmin"
                };
                Box::new(move |x, y| format!("{name}({x}, {y})"))
            }
            tags::MAX_ELEM => {
                self.need("g_int");
                let name = if d == Ty::I32 { "g_imax" } else { "g_umax" };
                Box::new(move |x, y| format!("{name}({x}, {y})"))
            }
            tags::MIN_ELEM => {
                self.need("g_int");
                let name = if d == Ty::I32 { "g_imin" } else { "g_umin" };
                Box::new(move |x, y| format!("{name}({x}, {y})"))
            }
            _ => return refuse("an arithmetic tag"),
        };
        self.each(r.n, |blk, i| {
            blk.line(format!(
                "{}[{i}] = {};",
                r.name,
                e(Gen::at(a, i), Gen::at(b, i))
            ));
        });
        Ok(())
    }

    // ------------------------------------------------------------- reductions

    fn reduce(&mut self, op: &LaunchOp) -> Lowering<()> {
        let (id, x) = (op.args[0], self.arg(op, 0)?);
        let (rows, len) = self.rows_len(id, &x);
        let r = self.result(op, 0)?;
        if r.wide {
            return refuse("a reduction into a wide value");
        }
        if r.n != rows {
            return refuse("a reduction that is not one value per row");
        }
        if x.ty == Ty::Bool {
            return refuse("a reduction over bools");
        }
        let xn = x.name.clone();
        // Each PE's partial over its block (the whole row when local).
        let part = if x.wide {
            self.temp(x.ty.elem(), rows)
        } else {
            r.name.clone()
        };
        if x.ty == Ty::F32 {
            self.need("g_tree");
            let lvl = self.temp("f32", len.div_ceil(32).max(1));
            let f = match op.tag {
                tags::REDUCE_SUM => "g_tsum",
                tags::REDUCE_MAX => "g_tmax",
                _ => "g_tmin",
            };
            let pn = part.clone();
            let held = Gen::held(&x, len);
            self.each(rows, |blk, rr| {
                blk.line(format!(
                    "{pn}[{rr}] = {f}(@ptrcast([*]f32, &{xn}), {rr} * {len}, {held}, @ptrcast([*]f32, &{lvl}));"
                ));
            });
        } else {
            let (ty, init) = match (op.tag, x.ty) {
                (tags::REDUCE_SUM, _) => (x.ty, "0".to_string()),
                (tags::REDUCE_MAX, Ty::I32) => (Ty::I32, ilit(i32::MIN)),
                (tags::REDUCE_MIN, Ty::I32) => (Ty::I32, ilit(i32::MAX)),
                (tags::REDUCE_MAX, _) => (Ty::U32, ulit(0)),
                (tags::REDUCE_MIN, _) => (Ty::U32, ulit(u32::MAX)),
                _ => unreachable!(),
            };
            let elem = ty.elem();
            let tag = op.tag;
            let pn = part.clone();
            let held = Gen::held(&x, len);
            self.each(rows, |blk, rr| {
                blk.line(format!("var acc: {elem} = {init};"));
                blk.nest(
                    &format!("{{ var j: i32 = 0; while (j < {held}) : (j += 1) {{"),
                    "} }",
                    |blk| {
                        let e = format!("{xn}[{rr} * {len} + j]");
                        match tag {
                            tags::REDUCE_SUM => blk.line(format!("acc = acc + {e};")),
                            tags::REDUCE_MAX => {
                                blk.line(format!("if ({e} > acc) {{ acc = {e}; }}"))
                            }
                            _ => blk.line(format!("if ({e} < acc) {{ acc = {e}; }}")),
                        };
                    },
                );
                blk.line(format!("{pn}[{rr}] = acc;"));
            });
        }
        if !x.wide {
            return Ok(());
        }
        // The partials combine across the row (f32 sums on the fabric).
        let cols = self.cols;
        let elem = x.ty.elem();
        let tag = op.tag;
        let float = x.ty == Ty::F32;
        if float {
            self.need("g_tree");
        }
        if float && tag == tags::REDUCE_SUM {
            let res = self.combine_sum(&part, rows);
            let rn = r.name.clone();
            self.each(rows, |blk, rr| {
                blk.line(format!("{rn}[{rr}] = @bitcast(f32, {res}[{rr}]);"));
            });
            return Ok(());
        }
        let res = self.combine(&part, rows, rows, |blk, recv| {
            blk.nest(
                &format!("{{ var rr: i32 = 0; while (rr < {rows}) : (rr += 1) {{"),
                "} }",
                |blk| {
                    blk.line(format!("var acc: {elem} = @bitcast({elem}, {recv}[rr]);"));
                    blk.nest(
                        &format!("{{ var c: i32 = 1; while (c < {cols}) : (c += 1) {{"),
                        "} }",
                        |blk| {
                            let e = format!("@bitcast({elem}, {recv}[c * {rows} + rr])");
                            match (tag, float) {
                                (tags::REDUCE_SUM, _) => blk.line(format!("acc = acc + {e};")),
                                (tags::REDUCE_MAX, true) => {
                                    blk.line(format!("acc = g_cmax(acc, {e});"))
                                }
                                (tags::REDUCE_MIN, true) => {
                                    blk.line(format!("acc = g_cmin(acc, {e});"))
                                }
                                (tags::REDUCE_MAX, false) => {
                                    blk.line(format!("if ({e} > acc) {{ acc = {e}; }}"))
                                }
                                _ => blk.line(format!("if ({e} < acc) {{ acc = {e}; }}")),
                            };
                        },
                    );
                    blk.line("RES[rr] = @bitcast(u32, acc);");
                },
            );
        });
        let rn = r.name.clone();
        self.each(rows, |blk, rr| {
            blk.line(format!("{rn}[{rr}] = @bitcast({elem}, {res}[{rr}]);"));
        });
        Ok(())
    }

    fn argmax(&mut self, op: &LaunchOp) -> Lowering<()> {
        let (id, x) = (op.args[0], self.arg(op, 0)?);
        let (rows, len) = self.rows_len(id, &x);
        let r = self.result(op, 0)?;
        if x.ty == Ty::Bool {
            return refuse("argmax over bools");
        }
        Gen::lanes_only(&x, "argmax")?;
        if r.wide || r.n != rows {
            return refuse("an argmax that is not one index per row");
        }
        self.argmax_rows(&x, rows, len, &r);
        Ok(())
    }

    /// `r[rr]` = the argmax of row `rr` (`len` lanes on this PE) of `x`:
    /// NaN never wins, ties go to the lowest index; a wide `x` combines the
    /// PEs' candidates across the row.
    fn argmax_rows(&mut self, x: &Var, rows: usize, len: usize, r: &Var) {
        debug_assert!(!x.packed, "argmax over a packed mask is refused before");
        let xn = x.name.clone();
        let elem = x.ty.elem();
        let float = x.ty == Ty::F32;
        let zero = Gen::zero(x.ty);
        // Per row: the best, its index (global when wide), whether any.
        let (best, bat, have) = (
            self.temp(elem, rows),
            self.temp("i32", rows),
            self.temp("u32", rows),
        );
        let base = if x.wide {
            format!("px * {}", self.block())
        } else {
            "0".to_string()
        };
        let held = Gen::held(x, len);
        self.each(rows, |blk, rr| {
            blk.line(format!(
                "var best: {elem} = {zero}; var bat: i32 = 0; var have: bool = false;"
            ));
            blk.nest(
                &format!("{{ var j: i32 = 0; while (j < {held}) : (j += 1) {{"),
                "} }",
                |blk| {
                    blk.line(format!("var x: {elem} = {xn}[{rr} * {len} + j];"));
                    if float {
                        blk.line("if (math.isNaN_f32(x)) { continue; }");
                    }
                    blk.line("if (!have or x > best) { best = x; bat = j; have = true; }");
                },
            );
            blk.line(format!(
                "{best}[{rr}] = best; {bat}[{rr}] = {base} + bat; if (have) {{ {have}[{rr}] = 1; }} else {{ {have}[{rr}] = 0; }}"
            ));
        });
        if !x.wide {
            let rn = r.name.clone();
            self.each(rows, |blk, rr| {
                blk.line(format!("{rn}[{rr}] = {bat}[{rr}];"));
            });
            return;
        }
        // Two rounds of a word a row: every PE offers its best (its type's
        // bottom when it has none), the root takes the greatest; then the
        // PEs holding it offer their index (the axis's length for the
        // others), the root takes the lowest. 0 when no PE has any.
        let send = self.temp("u32", rows);
        let bottom_bits = match x.ty {
            Ty::F32 => flit(f32::NEG_INFINITY),
            Ty::I32 => ilit(i32::MIN),
            _ => ulit(0),
        };
        let bottom_word = x.ty.to_word(&bottom_bits);
        self.each(rows, |blk, rr| {
            blk.line(format!(
                "if ({have}[{rr}] != 0) {{ {send}[{rr}] = {}; }} else {{ {send}[{rr}] = {bottom_word}; }}",
                x.ty.to_word(&format!("{best}[{rr}]"))
            ));
        });
        let cols = self.cols;
        let v = self.wide.unwrap_or(1);
        let xty = x.ty;
        let top = self.combine(&send, rows, rows, |blk, recv| {
            blk.nest(
                &format!("{{ var rr: i32 = 0; while (rr < {rows}) : (rr += 1) {{"),
                "} }",
                |blk| {
                    blk.line(format!(
                        "var acc: {elem} = {}; var any: bool = false;",
                        xty.of_word(&format!("{recv}[rr]"))
                    ));
                    blk.nest(
                        &format!("{{ var c: i32 = 0; while (c < {cols}) : (c += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!(
                                "var x: {elem} = {};",
                                xty.of_word(&format!("{recv}[c * {rows} + rr]"))
                            ));
                            if float {
                                blk.line("if (math.isNaN_f32(x)) { continue; }");
                            }
                            blk.line("if (!any or x > acc) { acc = x; any = true; }");
                        },
                    );
                    blk.line(format!("RES[rr] = {};", xty.to_word("acc")));
                },
            );
        });
        // A PE holding the best (and any lane) offers its lowest index.
        self.each(rows, |blk, rr| {
            blk.line(format!(
                "var tb: {elem} = {}; if ({have}[{rr}] != 0 and {best}[{rr}] == tb) {{ {send}[{rr}] = @bitcast(u32, {bat}[{rr}]); }} else {{ {send}[{rr}] = {v}; }}",
                xty.of_word(&format!("{top}[{rr}]"))
            ));
        });
        let res = self.combine(&send, rows, rows, |blk, recv| {
            blk.nest(
                &format!("{{ var rr: i32 = 0; while (rr < {rows}) : (rr += 1) {{"),
                "} }",
                |blk| {
                    blk.line(format!("var at: i32 = {v};"));
                    blk.line(format!(
                        "{{ var c: i32 = 0; while (c < {cols}) : (c += 1) {{ var xi: i32 = @bitcast(i32, {recv}[c * {rows} + rr]); if (xi < at) {{ at = xi; }} }} }}"
                    ));
                    blk.line(format!("if (at >= {v}) {{ at = 0; }} RES[rr] = @bitcast(u32, at);"));
                },
            );
        });
        let rn = r.name.clone();
        self.each(rows, |blk, rr| {
            blk.line(format!("{rn}[{rr}] = @bitcast(i32, {res}[{rr}]);"));
        });
    }

    fn scan(&mut self, op: &LaunchOp) -> Lowering<()> {
        let (id, x) = (op.args[0], self.arg(op, 0)?);
        if x.ty != Ty::F32 {
            // The interpreter scans in f32 and hands back f32 lanes
            // whatever the declared dtype: nothing to agree with.
            return refuse("a scan over integers");
        }
        let (rows, len) = self.rows_len(id, &x);
        let r = self.result(op, 0)?;
        self.agree(&r, &[&x])?;
        let sum = op.tag == tags::CUMSUM;
        let (init, sym) = if sum {
            (flit(0.0), "+")
        } else {
            (flit(1.0), "*")
        };
        let xn = x.name.clone();
        let rn = r.name.clone();
        // The scan of this PE's block; a wide scan then carries each
        // earlier block's total in.
        let total = x.wide.then(|| self.temp("f32", rows));
        let tn = total.clone();
        let held = Gen::held(&x, len);
        self.each(rows, |blk, rr| {
            blk.line(format!("var acc: f32 = {init};"));
            blk.nest(
                &format!("{{ var j: i32 = 0; while (j < {held}) : (j += 1) {{"),
                "} }",
                |blk| {
                    blk.line(format!("acc = acc {sym} {xn}[{rr} * {len} + j];"));
                    blk.line(format!("{rn}[{rr} * {len} + j] = acc;"));
                },
            );
            if let Some(t) = &tn {
                blk.line(format!("{t}[{rr}] = acc;"));
            }
        });
        let Some(total) = total else {
            return Ok(());
        };
        // The root folds the blocks' totals into each PE's carry: the fold
        // of the blocks before it (the identity for the first).
        let cols = self.cols;
        let res = self.combine(&total, rows, rows * cols, |blk, recv| {
            blk.nest(
                &format!("{{ var rr: i32 = 0; while (rr < {rows}) : (rr += 1) {{"),
                "} }",
                |blk| {
                    blk.line(format!("var carry: f32 = {init};"));
                    blk.nest(
                        &format!("{{ var c: i32 = 0; while (c < {cols}) : (c += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!("RES[c * {rows} + rr] = @bitcast(u32, carry);"));
                            blk.line(format!(
                                "carry = carry {sym} @bitcast(f32, {recv}[c * {rows} + rr]);"
                            ));
                        },
                    );
                },
            );
        });
        self.each(rows, |blk, rr| {
            blk.line(format!(
                "var carry: f32 = @bitcast(f32, {res}[px * {rows} + {rr}]);"
            ));
            blk.nest(
                &format!("{{ var j: i32 = 0; while (j < {len}) : (j += 1) {{"),
                "} }",
                |blk| {
                    blk.line(format!(
                        "{rn}[{rr} * {len} + j] = carry {sym} {rn}[{rr} * {len} + j];"
                    ));
                },
            );
        });
        Ok(())
    }

    fn broadcast(&mut self, op: &LaunchOp) -> Lowering<()> {
        let (id, x) = (op.args[0], self.arg(op, 0)?);
        let src = self.shape(id);
        let target = self.shape(op.result_id);
        let r = self.result(op, 0)?;
        let rank = target.len();
        if src.len() > rank {
            return refuse("a broadcast to a lower rank");
        }
        if src.iter().all(|&d| d == 1) {
            self.each(r.n, |blk, i| {
                blk.line(format!("{}[{i}] = {}[0];", r.name, x.name));
            });
            return Ok(());
        }
        // The interpreter aligns the source's axes to the target's leading
        // axes (`broadcast_value`). A wide result walks its PE's block of
        // the last axis; a wide source is read at its block.
        let sdim = |i: usize| -> u64 { if i < src.len() { u64::from(src[i]) } else { 1 } };
        let mut sstride = vec![1u64; rank.max(1)];
        for i in (0..rank.saturating_sub(1)).rev() {
            sstride[i] = sstride[i + 1] * sdim(i + 1);
        }
        let mut tstride = vec![1u64; rank.max(1)];
        for i in (0..rank.saturating_sub(1)).rev() {
            tstride[i] = tstride[i + 1] * u64::from(target[i + 1]);
        }
        if x.wide && (src.len() != rank || sdim(rank - 1) != u64::from(target[rank - 1])) {
            return refuse("a wide source broadcast along its wide axis");
        }
        Gen::lanes_only(&x, "a broadcast")?;
        Gen::lanes_only(&r, "a broadcast")?;
        let block = self.block();
        let v = self.wide.unwrap_or(1) as usize;
        let xn = x.name.clone();
        let (r_wide, x_wide) = (r.wide, x.wide);
        self.each(r.n, |blk, lin| {
            if r_wide {
                // The global index of this PE's element `lin`.
                blk.line(format!(
                    "var rem: i32 = ({lin} / {block}) * {v} + px * {block} + {lin} % {block}; var sidx: i32 = 0; var coord: i32 = 0;"
                ));
            } else {
                blk.line(format!(
                    "var rem: i32 = {lin}; var sidx: i32 = 0; var coord: i32 = 0;"
                ));
            }
            for i in 0..rank {
                let ts = tstride[i].max(1);
                blk.line(format!("coord = rem / {ts}; rem = rem % {ts};"));
                if sdim(i) != 1 {
                    blk.line(format!("sidx = sidx + coord * {};", sstride[i]));
                }
            }
            if x_wide {
                // The source's global index, as this PE's block index.
                blk.line(format!(
                    "sidx = (sidx / {v}) * {block} + sidx % {v} - px * {block};"
                ));
            }
            blk.line(format!("{}[{lin}] = {xn}[sidx];", r.name));
        });
        Ok(())
    }

    fn order(&mut self, op: &LaunchOp) -> Lowering<()> {
        let (id, x) = (op.args[0], self.arg(op, 0)?);
        if x.ty != Ty::F32 {
            return refuse("an ordering of non-floats");
        }
        Gen::lanes_only(&x, "an ordering")?;
        let (rows, len) = if op.tag == tags::SORT_DESC {
            if x.wide {
                return refuse("a full ordering of a wide value");
            }
            (1, x.n)
        } else {
            self.rows_len(id, &x)
        };
        let k = if op.tag == tags::TOP_K {
            op.imm as usize
        } else {
            len
        };
        if k < 1 || k > len {
            return refuse(format!("top-{k} of {len}"));
        }
        let values = self.result(op, 0)?;
        let index = self.result(op, 1)?;
        if index.ty != Ty::U32 {
            return refuse("an order that is not u32");
        }
        if values.wide || index.wide {
            return refuse("an ordering into a wide value");
        }
        self.need("g_sort");
        let key = self.temp("i32", len);
        let idx = self.temp("u32", len);
        let tk = self.temp("i32", len);
        let ti = self.temp("u32", len);
        let xn = x.name.clone();
        let held = Gen::held(&x, len);
        if !x.wide {
            self.each(rows, |blk, rr| {
                blk.line(format!(
                    "g_keys(@ptrcast([*]f32, &{xn}), {rr} * {len}, {len}, {len}, @ptrcast([*]i32, &{key}), @ptrcast([*]u32, &{idx}));"
                ));
                blk.line(format!(
                    "g_msort(@ptrcast([*]i32, &{key}), @ptrcast([*]u32, &{idx}), {len}, @ptrcast([*]i32, &{tk}), @ptrcast([*]u32, &{ti}));"
                ));
                blk.nest(
                    &format!("{{ var j: i32 = 0; while (j < {k}) : (j += 1) {{"),
                    "} }",
                    |blk| {
                        blk.line(format!("var src: u32 = {idx}[j];"));
                        blk.line(format!("{}[{rr} * {k} + j] = src;", index.name));
                        blk.line(format!(
                            "{}[{rr} * {k} + j] = {xn}[{rr} * {len} + @as(i32, src)];",
                            values.name
                        ));
                    },
                );
            });
            return Ok(());
        }
        self.need("g_copy");
        // Each PE sorts its block (every row's sorted keys and indices kept
        // in `skey`/`sidx`); then k rounds, each the row's best remaining
        // candidate: every PE offers its next unused (key, global index,
        // value), the root takes the greatest key (lowest index on a tie)
        // and the owner advances.
        let block = self.block();
        // Every row sorted, kept in `skey`/`sidx` (`tk`/`ti` are a row's
        // scratch in the local path; here the storage, scratch included).
        self.free_now(&tk);
        self.free_now(&ti);
        let skey = self.temp("i32", rows * len);
        let sidx = self.temp("u32", rows * len);
        self.sort_rows(&xn, rows, len, &held, &key, &idx, &skey, &sidx);
        let next = self.temp("i32", rows);
        let round = self.temp("u32", 1);
        self.each(rows, |blk, rr| {
            blk.line(format!("{next}[{rr}] = 0;"));
        });
        self.b.line(format!("{round}[0] = 0;"));
        let send = self.temp("u32", 4 * rows);
        let cols = self.cols;
        let lp = self.open_loop(k);
        self.each(rows, |blk, rr| {
            blk.line(format!("var at: i32 = 4 * {rr}; var nx: i32 = {next}[{rr}];"));
            blk.line(format!(
                "if (nx < {held}) {{ var src: i32 = @as(i32, {sidx}[{rr} * {len} + nx]); {send}[at] = @bitcast(u32, {skey}[{rr} * {len} + nx]); {send}[at + 1] = @as(u32, px * {block} + src); {send}[at + 2] = @bitcast(u32, {xn}[{rr} * {len} + src]); {send}[at + 3] = 1; }} else {{ {send}[at] = 0; {send}[at + 1] = 0; {send}[at + 2] = 0; {send}[at + 3] = 0; }}"
            ));
        });
        let per_pe = 4 * rows;
        let res = self.combine(&send, per_pe, 3 * rows, |blk, recv| {
            blk.nest(
                &format!("{{ var rr: i32 = 0; while (rr < {rows}) : (rr += 1) {{"),
                "} }",
                |blk| {
                    blk.line("var bc: i32 = -1; var bkey: i32 = 0; var bidx: u32 = 0;");
                    blk.nest(
                        &format!("{{ var c: i32 = 0; while (c < {cols}) : (c += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!("var at: i32 = c * {per_pe} + 4 * rr;"));
                            blk.line(format!("if ({recv}[at + 3] == 0) {{ continue; }}"));
                            blk.line(format!(
                                "var ck: i32 = @bitcast(i32, {recv}[at]); var ci: u32 = {recv}[at + 1];"
                            ));
                            blk.line("if (bc < 0 or ck > bkey or (ck == bkey and ci < bidx)) { bc = c; bkey = ck; bidx = ci; }");
                        },
                    );
                    blk.line(format!(
                        "var src: i32 = bc * {per_pe} + 4 * rr; RES[3 * rr] = @as(u32, bc); RES[3 * rr + 1] = {recv}[src + 1]; RES[3 * rr + 2] = {recv}[src + 2];"
                    ));
                },
            );
        });
        let (vn, inm) = (values.name.clone(), index.name.clone());
        self.each(rows, |blk, rr| {
            blk.line(format!("var j: i32 = @bitcast(i32, {round}[0]);"));
            blk.line(format!(
                "{inm}[{rr} * {k} + j] = {res}[3 * {rr} + 1]; {vn}[{rr} * {k} + j] = @bitcast(f32, {res}[3 * {rr} + 2]);"
            ));
            blk.line(format!(
                "if (@bitcast(i32, {res}[3 * {rr}]) == px) {{ {next}[{rr}] = {next}[{rr}] + 1; }}"
            ));
        });
        self.b.line(format!("{round}[0] = {round}[0] + 1;"));
        self.close_loop(lp, k);
        Ok(())
    }

    fn gather(&mut self, op: &LaunchOp) -> Lowering<()> {
        let (is, src) = (op.args[0], self.arg(op, 0)?);
        let idx = self.arg(op, 1)?;
        let ss = self.shape(is);
        let n0 = *ss.first().unwrap_or(&1) as usize;
        let rest = ss
            .iter()
            .skip(1)
            .map(|&d| d as usize)
            .product::<usize>()
            .max(1);
        let r = self.result(op, 0)?;
        if idx.wide || r.wide {
            return refuse("a gather by wide indices");
        }
        let k = idx.n;
        let zero = Gen::zero(r.ty);
        let (sn, rn, idn) = (src.name.clone(), r.name.clone(), idx.name.clone());
        let ixe = idx.ty.elem();
        let valid = Gen::index_valid(&idx, "ix", n0);
        if !src.wide {
            self.each(k, |blk, kk| {
                blk.line(format!("var ix: {ixe} = {idn}[{kk}];"));
                blk.nest(
                    &format!("{{ var q: i32 = 0; while (q < {rest}) : (q += 1) {{"),
                    "} }",
                    |blk| {
                        blk.line(format!(
                            "if ({valid}) {{ {rn}[{kk} * {rest} + q] = {sn}[@as(i32, ix) * {rest} + q]; }} else {{ {rn}[{kk} * {rest} + q] = {zero}; }}"
                        ));
                    },
                );
            });
            return Ok(());
        }
        // A wide source: a vector, each PE holding a block of it. The PE
        // owning an index answers it; the root takes the owner's word (a
        // zero for an index outside the vector).
        if ss.len() != 1 {
            return refuse("a gather from a wide matrix");
        }
        Gen::lanes_only(&src, "a gather")?;
        let block = self.block();
        let send = self.temp("u32", 2 * k);
        self.each(k, |blk, kk| {
            blk.line(format!("var ix: {ixe} = {idn}[{kk}];"));
            blk.line(format!("{send}[2 * {kk}] = 0; {send}[2 * {kk} + 1] = 0;"));
            blk.line(format!(
                "if ({valid}) {{ var g: i32 = @as(i32, ix) - px * {block}; if (g >= 0 and g < bn) {{ {send}[2 * {kk}] = {}; {send}[2 * {kk} + 1] = 1; }} }}",
                src.ty.to_word(&format!("{sn}[g]"))
            ));
        });
        let cols = self.cols;
        let res = self.combine(&send, 2 * k, k, |blk, recv| {
            blk.nest(
                &format!("{{ var kk: i32 = 0; while (kk < {k}) : (kk += 1) {{"),
                "} }",
                |blk| {
                    blk.line("RES[kk] = 0;");
                    blk.nest(
                        &format!("{{ var c: i32 = 0; while (c < {cols}) : (c += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!(
                                "if ({recv}[c * {} + 2 * kk + 1] != 0) {{ RES[kk] = {recv}[c * {} + 2 * kk]; }}",
                                2 * k,
                                2 * k
                            ));
                        },
                    );
                },
            );
        });
        let rty = r.ty;
        self.each(k, |blk, kk| {
            blk.line(format!(
                "{rn}[{kk}] = {};",
                rty.of_word(&format!("{res}[{kk}]"))
            ));
        });
        Ok(())
    }

    fn gather_row(&mut self, op: &LaunchOp) -> Lowering<()> {
        let (is, src) = (op.args[0], self.arg(op, 0)?);
        let idx = self.arg(op, 1)?;
        let ss = self.shape(is);
        if ss.len() != 2 {
            return refuse("gather_row of other than a matrix");
        }
        let (m, n) = (ss[0] as usize, ss[1] as usize);
        if idx.n != m || idx.wide {
            return refuse("gather_row's index is not one local index per row");
        }
        Gen::lanes_only(&src, "gather_row")?;
        let r = self.result(op, 0)?;
        if r.wide {
            return refuse("gather_row into a wide value");
        }
        let zero = Gen::zero(r.ty);
        let valid = Gen::index_valid(&idx, "ix", n);
        let (sn, rn, idn) = (src.name.clone(), r.name.clone(), idx.name.clone());
        let ixe = idx.ty.elem();
        if !src.wide {
            self.each(m, |blk, i| {
                blk.line(format!("var ix: {ixe} = {idn}[{i}];"));
                blk.line(format!(
                    "if ({valid}) {{ {rn}[{i}] = {sn}[{i} * {n} + @as(i32, ix)]; }} else {{ {rn}[{i}] = {zero}; }}"
                ));
            });
            return Ok(());
        }
        let block = self.block();
        let send = self.temp("u32", 2 * m);
        self.each(m, |blk, i| {
            blk.line(format!("var ix: {ixe} = {idn}[{i}];"));
            blk.line(format!("{send}[2 * {i}] = 0; {send}[2 * {i} + 1] = 0;"));
            blk.line(format!(
                "if ({valid}) {{ var g: i32 = @as(i32, ix) - px * {block}; if (g >= 0 and g < bn) {{ {send}[2 * {i}] = {}; {send}[2 * {i} + 1] = 1; }} }}",
                src.ty.to_word(&format!("{sn}[{i} * {block} + g]"))
            ));
        });
        let cols = self.cols;
        let res = self.combine(&send, 2 * m, m, |blk, recv| {
            blk.nest(
                &format!("{{ var i: i32 = 0; while (i < {m}) : (i += 1) {{"),
                "} }",
                |blk| {
                    blk.line("RES[i] = 0;");
                    blk.nest(
                        &format!("{{ var c: i32 = 0; while (c < {cols}) : (c += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!(
                                "if ({recv}[c * {} + 2 * i + 1] != 0) {{ RES[i] = {recv}[c * {} + 2 * i]; }}",
                                2 * m,
                                2 * m
                            ));
                        },
                    );
                },
            );
        });
        let rty = r.ty;
        self.each(m, |blk, i| {
            blk.line(format!(
                "{rn}[{i}] = {};",
                rty.of_word(&format!("{res}[{i}]"))
            ));
        });
        Ok(())
    }

    fn pivot(&mut self, op: &LaunchOp) -> Lowering<()> {
        let (ix, x) = (op.args[0], self.arg(op, 0)?);
        if x.ty != Ty::F32 {
            return refuse("a pivot over non-floats");
        }
        Gen::lanes_only(&x, "a pivot")?;
        let (rows, len) = self.rows_len(ix, &x);
        let payload = self.get(op.pred_payload)?;
        let pn = payload.n;
        if payload.wide || (pn != 1 && pn != rows) {
            return refuse(format!("a pivot payload of {pn} for {rows} rows"));
        }
        let r = self.result(op, 0)?;
        self.agree(&r, &[&x])?;
        if x.wide {
            return self.pivot_wide(op, &x, &payload, &r, rows, len);
        }
        let xn = x.name.clone();
        let rn = r.name.clone();
        match op.pred_tag {
            0 => {
                // rank_le(k): a lane is kept when fewer than k non-NaN lanes
                // are greater, i.e. when its order key is at or above the
                // k-th largest key.
                let k_expr = match payload.ty {
                    Ty::I32 => format!("g_iclamp(PAYLOAD, 0, {len})"),
                    Ty::U32 => format!("g_uclamp(PAYLOAD, {len})"),
                    _ => return refuse("rank_le over a non-integer k"),
                };
                self.need("g_int");
                self.need("g_sort");
                let key = self.temp("i32", len);
                let idx = self.temp("u32", len);
                let tk = self.temp("i32", len);
                let ti = self.temp("u32", len);
                let bottom = ilit(i32::MIN);
                self.each(rows, |blk, rr| {
                    let k_expr = k_expr.replace("PAYLOAD", &Gen::at(&payload, rr));
                    blk.line(format!("var k: i32 = {k_expr};"));
                    blk.line(format!(
                        "g_keys(@ptrcast([*]f32, &{xn}), {rr} * {len}, {len}, {len}, @ptrcast([*]i32, &{key}), @ptrcast([*]u32, &{idx}));"
                    ));
                    blk.line("var count: i32 = 0;");
                    blk.nest(
                        &format!("{{ var j: i32 = 0; while (j < {len}) : (j += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!(
                                "if ({key}[j] != {bottom}) {{ count = count + 1; }}"
                            ));
                        },
                    );
                    blk.line("var t: i32 = k; if (count < t) { t = count; }");
                    blk.line(format!(
                        "g_msort(@ptrcast([*]i32, &{key}), @ptrcast([*]u32, &{idx}), {len}, @ptrcast([*]i32, &{tk}), @ptrcast([*]u32, &{ti}));"
                    ));
                    // `key` holds the keys descending now; a lane's own key
                    // is recomputed from its value.
                    blk.line(format!(
                        "var pivot: i32 = {bottom}; if (t > 0) {{ pivot = {key}[t - 1]; }}"
                    ));
                    blk.nest(
                        &format!("{{ var j: i32 = 0; while (j < {len}) : (j += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!("var kj: i32 = g_okey({xn}[{rr} * {len} + j]);"));
                            blk.line(format!(
                                "if (t > 0 and kj != {bottom} and kj >= pivot) {{ {rn}[{rr} * {len} + j] = 1; }} else {{ {rn}[{rr} * {len} + j] = 0; }}"
                            ));
                        },
                    );
                });
                Ok(())
            }
            1 => {
                // cummass_le(p): in descending order, a lane is kept while
                // the mass before it is below p.
                if payload.ty != Ty::F32 {
                    return refuse("cummass_le over a non-float p");
                }
                self.need("g_sort");
                let key = self.temp("i32", len);
                let idx = self.temp("u32", len);
                let tk = self.temp("i32", len);
                let ti = self.temp("u32", len);
                let zero = flit(0.0);
                self.each(rows, |blk, rr| {
                    blk.line(format!("var p: f32 = {};", Gen::at(&payload, rr)));
                    blk.line(format!(
                        "g_keys(@ptrcast([*]f32, &{xn}), {rr} * {len}, {len}, {len}, @ptrcast([*]i32, &{key}), @ptrcast([*]u32, &{idx}));"
                    ));
                    blk.line(format!(
                        "g_msort(@ptrcast([*]i32, &{key}), @ptrcast([*]u32, &{idx}), {len}, @ptrcast([*]i32, &{tk}), @ptrcast([*]u32, &{ti}));"
                    ));
                    blk.line(format!("var excl: f32 = {zero};"));
                    blk.nest(
                        &format!("{{ var j: i32 = 0; while (j < {len}) : (j += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!("var src: i32 = @as(i32, {idx}[j]);"));
                            blk.line(format!(
                                "if (!math.isNaN_f32(excl) and !math.isNaN_f32(p) and excl < p) {{ {rn}[{rr} * {len} + src] = 1; }} else {{ {rn}[{rr} * {len} + src] = 0; }}"
                            ));
                            blk.line(format!("excl = excl + {xn}[{rr} * {len} + src];"));
                        },
                    );
                });
                Ok(())
            }
            _ => {
                if payload.ty != Ty::F32 {
                    return refuse("prob_ge over a non-float threshold");
                }
                self.each(rows, |blk, rr| {
                    blk.line(format!("var thr: f32 = {};", Gen::at(&payload, rr)));
                    blk.nest(
                        &format!("{{ var j: i32 = 0; while (j < {len}) : (j += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!(
                                "if (!math.isNaN_f32({xn}[{rr} * {len} + j]) and !math.isNaN_f32(thr) and {xn}[{rr} * {len} + j] >= thr) {{ {rn}[{rr} * {len} + j] = 1; }} else {{ {rn}[{rr} * {len} + j] = 0; }}"
                            ));
                        },
                    );
                });
                Ok(())
            }
        }
    }

    /// A pivot over a wide value: the threshold is found by bisection over
    /// the order keys, every step one count (or mass) combined across the
    /// row; every PE walks the same bisection, so only the totals travel.
    #[allow(clippy::too_many_lines)]
    fn pivot_wide(
        &mut self,
        op: &LaunchOp,
        x: &Var,
        payload: &Var,
        r: &Var,
        rows: usize,
        len: usize,
    ) -> Lowering<()> {
        self.need("g_copy");
        self.need("g_sort");
        self.need("g_int");
        let xn = x.name.clone();
        let rn = r.name.clone();
        let total = rows * len;
        let key = self.temp("i32", len);
        let idx = self.temp("u32", len);
        let skey = self.temp("i32", total);
        let sidx = self.temp("u32", total);
        let bottom = ilit(i32::MIN);
        let top = ilit(i32::MAX);
        // Every row's keys sorted descending, with their source indices
        // (the block's lanes past this PE's share key as NaN does: last).
        self.sort_rows(&xn, rows, len, "bn", &key, &idx, &skey, &sidx);
        let cols = self.cols;
        let lo = self.temp("i32", rows);
        let hi = self.temp("i32", rows);
        let mid = self.temp("i32", rows);
        let send = self.temp("u32", rows);
        match op.pred_tag {
            0 => {
                // rank_le(k): the k-th largest key (k capped by the non-NaN
                // count); kept: a real lane at or above it.
                let k_expr = match payload.ty {
                    Ty::I32 => format!("g_iclamp(PAYLOAD, 0, {})", len * cols),
                    Ty::U32 => format!("g_uclamp(PAYLOAD, {})", len * cols),
                    _ => return refuse("rank_le over a non-integer k"),
                };
                // The non-NaN count across the row.
                // Counts travel as f32 (exact below 2^24) and add on the fabric.
                self.each(rows, |blk, rr| {
                    blk.line(format!(
                        "{send}[{rr}] = @bitcast(u32, @as(f32, g_count_ge(@ptrcast([*]i32, &{skey}), {rr} * {len}, {len}, {bottom} + 1)));"
                    ));
                });
                let totals = self.combine_sum(&send, rows);
                let t = self.temp("i32", rows);
                self.each(rows, |blk, rr| {
                    let k_expr = k_expr.replace("PAYLOAD", &Gen::at(payload, rr));
                    blk.line(format!("var k: i32 = {k_expr};"));
                    blk.line(format!(
                        "var count: i32 = @as(i32, @bitcast(f32, {totals}[{rr}])); if (count < k) {{ k = count; }} {t}[{rr}] = k;"
                    ));
                    blk.line(format!("{lo}[{rr}] = {bottom} + 1; {hi}[{rr}] = {top};"));
                });
                // Shortcut: with k within the candidates each PE can offer,
                // the k-th largest key is among the PEs' top keys, found by
                // the root in one round; the bisection then runs one round.
                let kmax = len.min((1024 / (cols * rows)).max(1));
                let cand = self.temp("u32", kmax * rows);
                self.each(rows, |blk, rr| {
                    blk.line(format!(
                        "{{ var q: i32 = 0; while (q < {kmax}) : (q += 1) {{ {cand}[{rr} * {kmax} + q] = @bitcast(u32, {skey}[{rr} * {len} + q]); }} }}"
                    ));
                });
                let per_pe = kmax * rows;
                let taken = self.temp("u16", cols);
                let pick = self.combine(&cand, per_pe, 2 * rows, |blk, recv| {
                    blk.nest(
                        &format!("{{ var rr: i32 = 0; while (rr < {rows}) : (rr += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!(
                                "var tt: i32 = {t}[rr]; RES[2 * rr] = 0; RES[2 * rr + 1] = 0;"
                            ));
                            blk.nest(&format!("if (tt > 0 and tt <= {kmax}) {{"), "}", |blk| {
                                blk.line(format!(
                                    "{{ var c: i32 = 0; while (c < {cols}) : (c += 1) {{ {taken}[c] = 0; }} }}"
                                ));
                                blk.line("var pivot: i32 = 0;");
                                blk.nest(
                                    "{ var j: i32 = 0; while (j < tt) : (j += 1) {",
                                    "} }",
                                    |blk| {
                                        blk.line("var bc: i32 = -1; var bkey: i32 = 0;");
                                        blk.line(format!(
                                            "{{ var c: i32 = 0; while (c < {cols}) : (c += 1) {{ if (@as(i32, {taken}[c]) >= {kmax}) {{ continue; }} var ck: i32 = @bitcast(i32, {recv}[c * {per_pe} + rr * {kmax} + @as(i32, {taken}[c])]); if (bc < 0 or ck > bkey) {{ bc = c; bkey = ck; }} }} }}"
                                        ));
                                        blk.line(format!("pivot = bkey; {taken}[bc] = {taken}[bc] + 1;"));
                                    },
                                );
                                blk.line("RES[2 * rr] = @bitcast(u32, pivot); RES[2 * rr + 1] = 1;");
                            });
                        },
                    );
                });
                let shortcut = std::env::var_os("PIE_CEREBRAS_GUEST_BISECT").is_none();
                let settled = self.temp("u32", 1);
                self.b.line(format!("{settled}[0] = 1;"));
                self.each(rows, |blk, rr| {
                    if shortcut {
                        blk.line(format!(
                            "if ({pick}[2 * {rr} + 1] != 0) {{ {lo}[{rr}] = @bitcast(i32, {pick}[2 * {rr}]); {hi}[{rr}] = {lo}[{rr}]; }} else {{ {settled}[0] = 0; }}"
                        ));
                    } else {
                        blk.line(format!("{settled}[0] = 0; var unused: u32 = {pick}[2 * {rr}]; unused = unused;"));
                    }
                });
                // Bisection: the largest T with count(key >= T) >= t (one
                // round when the shortcut settled every row).
                let lp = self.open_loop(32);
                let counter = lp.1.clone();
                self.b
                    .line(format!("if ({settled}[0] != 0) {{ {counter}[0] = 31; }}"));
                self.each(rows, |blk, rr| {
                    blk.line(format!(
                        "var m64: i64 = (@as(i64, {lo}[{rr}]) + @as(i64, {hi}[{rr}]) + 1) / 2; {mid}[{rr}] = @as(i32, m64);"
                    ));
                    blk.line(format!(
                        "{send}[{rr}] = @bitcast(u32, @as(f32, g_count_ge(@ptrcast([*]i32, &{skey}), {rr} * {len}, {len}, {mid}[{rr}])));"
                    ));
                });
                let counts = self.combine_sum(&send, rows);
                self.each(rows, |blk, rr| {
                    blk.line(format!(
                        "if ({lo}[{rr}] < {hi}[{rr}]) {{ if (@as(i32, @bitcast(f32, {counts}[{rr}])) >= {t}[{rr}]) {{ {lo}[{rr}] = {mid}[{rr}]; }} else {{ {hi}[{rr}] = {mid}[{rr}] - 1; }} }}"
                    ));
                });
                self.close_loop(lp, 32);
                self.each(rows, |blk, rr| {
                    blk.nest(
                        &format!("{{ var j: i32 = 0; while (j < {len}) : (j += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!("var kj: i32 = g_okey({xn}[{rr} * {len} + j]);"));
                            blk.line(format!(
                                "if ({t}[{rr}] > 0 and kj != {bottom} and kj >= {lo}[{rr}]) {{ {rn}[{rr} * {len} + j] = 1; }} else {{ {rn}[{rr} * {len} + j] = 0; }}"
                            ));
                        },
                    );
                });
                Ok(())
            }
            1 => {
                // cummass_le(p): in descending order, a lane is kept while
                // the mass before it is below p. The boundary key T* is the
                // lowest key whose mass above it is below p; keys above T*
                // are kept, ties at T* in index order while the mass stays
                // below p, lower keys not.
                if payload.ty != Ty::F32 {
                    return refuse("cummass_le over a non-float p");
                }
                // Prefix masses of every sorted row.
                let pre = self.temp("f32", rows * (len + 1));
                let zero = flit(0.0);
                let len1 = len + 1;
                self.each(rows, |blk, rr| {
                    blk.line(format!("var acc: f32 = {zero}; {pre}[{rr} * {len1}] = acc;"));
                    blk.nest(
                        &format!("{{ var j: i32 = 0; while (j < {len}) : (j += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!(
                                "if (j < bn) {{ acc = acc + {xn}[{rr} * {len} + @as(i32, {sidx}[{rr} * {len} + j])]; }} {pre}[{rr} * {len1} + j + 1] = acc;"
                            ));
                        },
                    );
                    blk.line(format!("{lo}[{rr}] = {bottom} + 1; {hi}[{rr}] = {top};"));
                });
                // Shortcut: each PE offers the values of its top candidates
                // (the key is the value's); the root walks the merged order
                // adding the mass as the interpreter does. The walk settles
                // when it stops at a key strictly below the last kept one
                // (`Kb`) with no PE run dry above it: every PE then keeps its
                // lanes keyed at or above `Kb`. A stop inside a run of equal
                // keys, or past a PE's candidates, leaves the row to the
                // bisection.
                let m = len.min((512 / (cols * rows)).max(1));
                let cand = self.temp("u32", m * rows);
                self.each(rows, |blk, rr| {
                    blk.line(format!(
                        "{{ var q: i32 = 0; while (q < {m}) : (q += 1) {{ {cand}[{rr} * {m} + q] = @bitcast(u32, {xn}[{rr} * {len} + @as(i32, {sidx}[{rr} * {len} + q])]); }} }}"
                    ));
                });
                let per_pe = m * rows;
                let taken = self.temp("u16", cols);
                let pv_root = Gen::at(payload, "rr");
                let walk = self.combine(&cand, per_pe, 2 * rows, |blk, recv| {
                    blk.nest(
                        &format!("{{ var rr: i32 = 0; while (rr < {rows}) : (rr += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!(
                                "var p: f32 = {pv_root}; var excl: f32 = {zero}; var ok: bool = true; var more: bool = true; var kb: i32 = {top}; var any: bool = false;"
                            ));
                            blk.line(format!(
                                "{{ var c: i32 = 0; while (c < {cols}) : (c += 1) {{ {taken}[c] = 0; }} }}"
                            ));
                            blk.nest("while (more) {", "}", |blk| {
                                blk.line("var bc: i32 = -1; var bkey: i32 = 0; var exhausted: bool = false;");
                                blk.line(format!(
                                    "{{ var c: i32 = 0; while (c < {cols}) : (c += 1) {{ if (@as(i32, {taken}[c]) >= {m}) {{ if ({m} < {len}) {{ exhausted = true; }} continue; }} var ck: i32 = g_okey(@bitcast(f32, {recv}[c * {per_pe} + rr * {m} + @as(i32, {taken}[c])])); if (bc < 0 or ck > bkey) {{ bc = c; bkey = ck; }} }} }}"
                                ));
                                // The next in order is kept while the mass before it is
                                // below p; the walk stops at the first not kept, and is
                                // void when a PE ran out of candidates before that, or
                                // when it stops among equal keys.
                                blk.line("if (bc < 0) { more = false; if (exhausted) { ok = false; } }");
                                blk.nest("else {", "}", |blk| {
                                    blk.line("var keep: bool = !math.isNaN_f32(excl) and !math.isNaN_f32(p) and excl < p;");
                                    blk.line("if (!keep) { more = false; if (any and bkey == kb) { ok = false; } }");
                                    blk.line(format!(
                                        "else {{ if (exhausted and bkey <= g_exhausted_value(@ptrcast([*]u32, &{recv}), {per_pe}, {m}, rr, @ptrcast([*]u16, &{taken}), {cols})) {{ ok = false; more = false; }} else {{ excl = excl + @bitcast(f32, {recv}[bc * {per_pe} + rr * {m} + @as(i32, {taken}[bc])]); {taken}[bc] = {taken}[bc] + 1; kb = bkey; any = true; }} }}"
                                    ));
                                });
                            });
                            blk.line(format!(
                                "if (ok) {{ RES[2 * rr] = 1; }} else {{ RES[2 * rr] = 0; }} if (any) {{ RES[2 * rr + 1] = @bitcast(u32, kb); }} else {{ RES[2 * rr + 1] = @bitcast(u32, {top}); }}"
                            ));
                        },
                    );
                });
                let shortcut = std::env::var_os("PIE_CEREBRAS_GUEST_BISECT").is_none();
                let settled = self.temp("u32", 1);
                let okrow = self.temp("u32", rows);
                self.b.line(format!("{settled}[0] = 1;"));
                self.each(rows, |blk, rr| {
                    if shortcut {
                        blk.line(format!(
                            "{okrow}[{rr}] = {walk}[2 * {rr}]; if ({okrow}[{rr}] == 0) {{ {settled}[0] = 0; }}"
                        ));
                    } else {
                        blk.line(format!("{okrow}[{rr}] = 0; {settled}[0] = 0;"));
                    }
                });
                let pv = |rr: &str| Gen::at(payload, rr);
                // Bisection: the smallest T with mass(key > T) < p (one round
                // when the walk settled every row).
                let lp = self.open_loop(32);
                let counter = lp.1.clone();
                self.b
                    .line(format!("if ({settled}[0] != 0) {{ {counter}[0] = 31; }}"));
                self.each(rows, |blk, rr| {
                    blk.line(format!(
                        "var m64: i64 = @as(i64, {lo}[{rr}]) + (@as(i64, {hi}[{rr}]) - @as(i64, {lo}[{rr}])) / 2; {mid}[{rr}] = @as(i32, m64);"
                    ));
                    blk.line(format!(
                        "var above: i32 = g_count_gt(@ptrcast([*]i32, &{skey}), {rr} * {len}, {len}, {mid}[{rr}]); {send}[{rr}] = @bitcast(u32, {pre}[{rr} * {len1} + above]);"
                    ));
                });
                let mass = self.combine_sum(&send, rows);
                self.each(rows, |blk, rr| {
                    let p = pv(rr);
                    blk.line(format!(
                        "if ({lo}[{rr}] < {hi}[{rr}]) {{ var mm: f32 = @bitcast(f32, {mass}[{rr}]); if (!math.isNaN_f32(mm) and !math.isNaN_f32({p}) and mm < {p}) {{ {hi}[{rr}] = {mid}[{rr}]; }} else {{ {lo}[{rr}] = {mid}[{rr}] + 1; }} }}"
                    ));
                });
                self.close_loop(lp, 32);
                // T*: the lowest key present at or above the bisection's T,
                // or none (nothing is kept, as when p is NaN or at most 0):
                // each PE offers its lowest such key (the top key for none),
                // the root takes the least.
                let send2 = self.temp("u32", rows);
                self.each(rows, |blk, rr| {
                    let p = pv(rr);
                    blk.line(format!(
                        "var ge: i32 = g_count_ge(@ptrcast([*]i32, &{skey}), {rr} * {len}, {len}, {lo}[{rr}]);"
                    ));
                    blk.line(format!(
                        "var ok: bool = ge > 0 and !math.isNaN_f32({p}) and {zero} < {p};"
                    ));
                    blk.line(format!(
                        "if (ok) {{ {send2}[{rr}] = @bitcast(u32, {skey}[{rr} * {len} + ge - 1]); }} else {{ {send2}[{rr}] = @bitcast(u32, {top}); }}"
                    ));
                });
                let tstar = self.combine(&send2, rows, rows, |blk, recv| {
                    blk.nest(
                        &format!("{{ var rr: i32 = 0; while (rr < {rows}) : (rr += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!("var best: i32 = {top};"));
                            blk.line(format!(
                                "{{ var c: i32 = 0; while (c < {cols}) : (c += 1) {{ var kk: i32 = @bitcast(i32, {recv}[c * {rows} + rr]); if (kk < best) {{ best = kk; }} }} }}"
                            ));
                            blk.line("RES[rr] = @bitcast(u32, best);");
                        },
                    );
                });
                // The mass above T* and the ties at T* (a count), summed on
                // the fabric; the tie value is T*'s own. Every PE then knows
                // how many ties are kept (`m`): all, none, or the first m in
                // index order, found by bisecting the index space (a round
                // per bit; one round when no row has a partial tie).
                let send3 = self.temp("u32", 2 * rows);
                self.each(rows, |blk, rr| {
                    blk.line(format!("var ts: i32 = @bitcast(i32, {tstar}[{rr}]);"));
                    blk.line(format!(
                        "var above: i32 = g_count_gt(@ptrcast([*]i32, &{skey}), {rr} * {len}, {len}, ts); var ge: i32 = g_count_ge(@ptrcast([*]i32, &{skey}), {rr} * {len}, {len}, ts);"
                    ));
                    blk.line(format!(
                        "{send3}[2 * {rr}] = @bitcast(u32, {pre}[{rr} * {len1} + above]); {send3}[2 * {rr} + 1] = @bitcast(u32, @as(f32, ge - above));"
                    ));
                });
                let sums = self.combine_sum(&send3, 2 * rows);
                let kept_ties = self.temp("i32", rows);
                let all_ties = self.temp("i32", rows);
                let ilo = self.temp("i32", rows);
                let ihi = self.temp("i32", rows);
                let partial = self.temp("u32", 1);
                let v = self.wide.unwrap_or(1) as usize;
                self.b.line(format!("{partial}[0] = 0;"));
                self.each(rows, |blk, rr| {
                    let p = pv(rr);
                    blk.line(format!(
                        "var ts: i32 = @bitcast(i32, {tstar}[{rr}]); var tv: f32 = g_unkey(ts); var mass: f32 = @bitcast(f32, {sums}[2 * {rr}]); var ties: i32 = @as(i32, @bitcast(f32, {sums}[2 * {rr} + 1]));"
                    ));
                    blk.line(format!(
                        "var m: i32 = 0; var excl: f32 = mass; if (ts != {top}) {{ while (m < ties) {{ if (math.isNaN_f32(excl) or math.isNaN_f32({p}) or !(excl < {p})) {{ break; }} excl = excl + tv; m = m + 1; }} }}"
                    ));
                    blk.line(format!(
                        "{kept_ties}[{rr}] = m; {all_ties}[{rr}] = ties; {ilo}[{rr}] = 0; {ihi}[{rr}] = {v}; if (m > 0 and m < ties and {okrow}[{rr}] == 0) {{ {partial}[0] = 1; }}"
                    ));
                });
                // The smallest index bound I with m ties below it: ties with
                // an index under I are kept.
                let ibits = (v + 1).next_power_of_two().trailing_zeros() as usize + 1;
                let isend = self.temp("u32", rows);
                let imid = self.temp("i32", rows);
                let lp2 = self.open_loop(ibits);
                let counter2 = lp2.1.clone();
                self.b.line(format!(
                    "if ({partial}[0] == 0) {{ {counter2}[0] = {}; }}",
                    ibits - 1
                ));
                let block = self.block();
                self.each(rows, |blk, rr| {
                    blk.line(format!(
                        "{imid}[{rr}] = {ilo}[{rr}] + ({ihi}[{rr}] - {ilo}[{rr}]) / 2; var ts: i32 = @bitcast(i32, {tstar}[{rr}]); var below: i32 = 0;"
                    ));
                    blk.line(format!(
                        "{{ var j: i32 = 0; while (j < bn) : (j += 1) {{ if (px * {block} + j < {imid}[{rr}] and g_okey({xn}[{rr} * {len} + j]) == ts) {{ below = below + 1; }} }} }}"
                    ));
                    blk.line(format!("{isend}[{rr}] = @bitcast(u32, @as(f32, below));"));
                });
                let belows = self.combine_sum(&isend, rows);
                self.each(rows, |blk, rr| {
                    blk.line(format!(
                        "if ({ilo}[{rr}] < {ihi}[{rr}]) {{ if (@as(i32, @bitcast(f32, {belows}[{rr}])) >= {kept_ties}[{rr}]) {{ {ihi}[{rr}] = {imid}[{rr}]; }} else {{ {ilo}[{rr}] = {imid}[{rr}] + 1; }} }}"
                    ));
                });
                self.close_loop(lp2, ibits);
                self.each(rows, |blk, rr| {
                    blk.line(format!(
                        "var ts: i32 = @bitcast(i32, {tstar}[{rr}]); var any: bool = ts != {top}; var m: i32 = {kept_ties}[{rr}]; var ties: i32 = {all_ties}[{rr}]; var ibound: i32 = {ilo}[{rr}];"
                    ));
                    blk.line(format!(
                        "var walked: bool = {okrow}[{rr}] != 0; var kb: i32 = @bitcast(i32, {walk}[2 * {rr} + 1]);"
                    ));
                    blk.nest(
                        &format!("{{ var j: i32 = 0; while (j < {len}) : (j += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!("var kj: i32 = g_okey({xn}[{rr} * {len} + j]); var keep: bool = false;"));
                            // The walk's verdict: lanes keyed at or above the last kept.
                            blk.line(format!("if (walked) {{ keep = kb != {top} and kj != {bottom} and kj >= kb; }}"));
                            blk.line(format!(
                                "else if (any) {{ if (kj > ts) {{ keep = true; }} else if (kj == ts) {{ if (m >= ties) {{ keep = true; }} else if (m > 0) {{ keep = px * {block} + j < ibound; }} }} }}"
                            ));
                            blk.line(format!(
                                "if (keep) {{ {rn}[{rr} * {len} + j] = 1; }} else {{ {rn}[{rr} * {len} + j] = 0; }}"
                            ));
                        },
                    );
                });
                Ok(())
            }
            _ => {
                if payload.ty != Ty::F32 {
                    return refuse("prob_ge over a non-float threshold");
                }
                self.each(rows, |blk, rr| {
                    blk.line(format!("var thr: f32 = {};", Gen::at(payload, rr)));
                    blk.nest(
                        &format!("{{ var j: i32 = 0; while (j < {len}) : (j += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!(
                                "if (!math.isNaN_f32({xn}[{rr} * {len} + j]) and !math.isNaN_f32(thr) and {xn}[{rr} * {len} + j] >= thr) {{ {rn}[{rr} * {len} + j] = 1; }} else {{ {rn}[{rr} * {len} + j] = 0; }}"
                            ));
                        },
                    );
                });
                Ok(())
            }
        }
    }

    fn scatter(&mut self, op: &LaunchOp) -> Lowering<()> {
        let (ib, base) = (op.args[0], self.arg(op, 0)?);
        let idx = self.arg(op, 1)?;
        let vals = self.arg(op, 2)?;
        let add = op.tag == tags::SCATTER_ADD;
        if add && base.ty == Ty::Bool {
            return refuse("scatter_add into bools");
        }
        if vals.ty != base.ty {
            return refuse("scatter values of another dtype than the base");
        }
        if idx.wide || vals.wide {
            return refuse("a scatter by wide indices or of wide values");
        }
        let bs = self.shape(ib);
        let n0 = *bs.first().unwrap_or(&1) as usize;
        let rest = bs
            .iter()
            .skip(1)
            .map(|&d| d as usize)
            .product::<usize>()
            .max(1);
        let k = idx.n;
        let scalar = vals.n == 1 && k * rest != 1;
        if !scalar && vals.n != k * rest {
            return refuse(format!("{} scatter values for {k} rows of {rest}", vals.n));
        }
        if base.wide && bs.len() != 1 {
            return refuse("a scatter into a wide matrix");
        }
        Gen::lanes_only(&base, "a scatter")?;
        let r = self.result(op, 0)?;
        self.agree(&r, &[&base])?;
        let (bn, rn, idn, vn) = (
            base.name.clone(),
            r.name.clone(),
            idx.name.clone(),
            vals.name.clone(),
        );
        self.copy_of(r.ty, &rn, "0", &bn, "0", r.n);
        let valid = Gen::index_valid(&idx, "ix", n0);
        let ixe = idx.ty.elem();
        let block = self.block();
        let wide = base.wide;
        // Sequential: a later write to a row wins. A wide base's PE applies
        // the writes landing in its block.
        self.each(k, |blk, kk| {
            blk.line(format!("var ix: {ixe} = {idn}[{kk}];"));
            blk.nest(&format!("if ({valid}) {{"), "}", |blk| {
                if wide {
                    blk.line(format!("var g: i32 = @as(i32, ix) - px * {block};"));
                    blk.line("if (g >= 0 and g < bn) {");
                } else {
                    blk.line("var g: i32 = @as(i32, ix); {");
                }
                blk.nest(
                    &format!("{{ var q: i32 = 0; while (q < {rest}) : (q += 1) {{"),
                    "} }",
                    |blk| {
                        let src = if scalar {
                            format!("{vn}[0]")
                        } else {
                            format!("{vn}[{kk} * {rest} + q]")
                        };
                        let dst = format!("{rn}[g * {rest} + q]");
                        if add {
                            blk.line(format!("{dst} = {dst} + {src};"));
                        } else {
                            blk.line(format!("{dst} = {src};"));
                        }
                    },
                );
                blk.line("}");
            });
        });
        Ok(())
    }

    // ------------------------------------------------------------- rendering

    /// The PE program and the layout.
    fn render(mut self, layout: &Layout, lanes: u32, cols: u32) -> (String, String) {
        let last = std::mem::replace(&mut self.b, Block::new(1));
        let staged = !self.segments.is_empty();
        let (v, block) = (self.wide.unwrap_or(1), self.block());
        let mut pe = String::new();
        pe.push_str("param memcpy_params;\n");
        if staged {
            pe.push_str("param c2d_params;\n");
        }
        pe.push_str("const sys_mod = @import_module(\"<memcpy/memcpy>\", memcpy_params);\n");
        pe.push_str("const math = @import_module(\"<math>\");\n");
        if staged {
            pe.push_str(
                "const mpi_x = @import_module(\"<collectives_2d/pe>\", .{ .dim_params = c2d_params.x, .queues = [2]u16{2, 4}, .dest_dsr_ids = [1]u16{1}, .src0_dsr_ids = [1]u16{1}, .src1_dsr_ids = [1]u16{1} });\n",
            );
            pe.push_str("const step_id: local_task_id = @get_local_task_id(15);\n");
        }
        pe.push('\n');
        if let Some(pack) = &self.pack {
            let _ = writeln!(pe, "var {pack} = @zeros([{}]u32);", layout.in_words.max(1));
            let _ = writeln!(pe, "var {pack}_ptr: [*]u32 = &{pack};");
        }
        if let Some(out) = &self.out {
            let _ = writeln!(pe, "var {out} = @zeros([{}]u32);", layout.out_words.max(1));
            let _ = writeln!(pe, "var {out}_ptr: [*]u32 = &{out};");
        }
        pe.push_str("var px: i32 = 0;\n");
        // This PE's share of the wide axis (its whole block but for the
        // last PEs).
        let _ = writeln!(pe, "var bn: i32 = {block};");
        for (name, len, elem) in &self.scratch {
            let _ = writeln!(pe, "var {name} = @zeros([{len}]{elem});");
        }
        // The direct slots' buffers: pointers to the values' arrays.
        for (array, elem, export) in &self.exports {
            let _ = writeln!(pe, "var {export}_ptr: [*]{elem} = &{array};");
        }
        pe.push('\n');
        for name in &self.helpers {
            pe.push_str(&helper_source(name));
            pe.push('\n');
        }
        // The segments as functions, the step over them.
        let mut segs: Vec<(Block, Option<Collective>)> = self
            .segments
            .into_iter()
            .map(|(b, c)| (b, Some(c)))
            .collect();
        segs.push((last, None));
        for (i, (b, _)) in segs.iter().enumerate() {
            let _ = writeln!(pe, "fn seg{i}() void {{");
            b.render(&mut pe);
            pe.push_str("}\n");
        }
        if staged {
            pe.push_str("var st: u16 = 0;\nfn step_fn() void {\n  switch (st) {\n");
            for (i, (_, c)) in segs.iter().enumerate() {
                let _ = write!(pe, "    {i} => {{ seg{i}(); st = {}; ", i + 1);
                match c {
                    Some(Collective::Gather { send, recv, count }) => {
                        let _ = write!(
                            pe,
                            "mpi_x.gather(0, @ptrcast([*]u32, &{send}), @ptrcast([*]u32, &{recv}), {count}, step_id);"
                        );
                    }
                    Some(Collective::Broadcast { buf, count }) => {
                        let _ = write!(
                            pe,
                            "mpi_x.broadcast(0, @ptrcast([*]u32, &{buf}), {count}, step_id);"
                        );
                    }
                    Some(Collective::Reduce { send, recv, count }) => {
                        let _ = write!(
                            pe,
                            "mpi_x.reduce_fadds(0, @ptrcast([*]f32, &{send}), @ptrcast([*]f32, &{recv}), {count}, step_id);"
                        );
                    }
                    Some(Collective::Fall) => pe.push_str("step_fn();"),
                    Some(Collective::Jump { to, counter, times }) => {
                        let _ = write!(
                            pe,
                            "if ({counter}[0] < {times}) {{ st = {to}; }} step_fn();"
                        );
                    }
                    None => pe.push_str("sys_mod.unblock_cmd_stream();"),
                }
                pe.push_str(" },\n");
            }
            pe.push_str("    else => {},\n  }\n}\n");
            pe.push_str("task step_task() void { step_fn(); }\n");
            let _ = write!(
                pe,
                "fn run() void {{\n  st = 0;\n  mpi_x.init();\n  px = @as(i32, mpi_x.pe_id);\n  bn = {v} - px * {block}; if (bn > {block}) {{ bn = {block}; }} if (bn < 0) {{ bn = 0; }}\n  step_fn();\n}}\n",
            );
        } else {
            pe.push_str("fn run() void {\n  seg0();\n  sys_mod.unblock_cmd_stream();\n}\n");
        }
        pe.push_str("comptime {\n");
        if staged {
            pe.push_str("  @bind_local_task(step_task, step_id);\n");
        }
        if let Some(pack) = &self.pack {
            let _ = writeln!(pe, "  @export_symbol({pack}_ptr, \"{pack}\");");
        }
        if let Some(out) = &self.out {
            let _ = writeln!(pe, "  @export_symbol({out}_ptr, \"{out}\");");
        }
        for (_, _, export) in &self.exports {
            let _ = writeln!(pe, "  @export_symbol({export}_ptr, \"{export}\");");
        }
        pe.push_str("  @export_symbol(run);\n}\n");

        let mut layout_text = String::new();
        let _ = writeln!(
            layout_text,
            "const memcpy = @import_module(\"<memcpy/get_params>\", .{{ .width = {cols}, .height = {lanes} }});"
        );
        if staged {
            layout_text.push_str("const c2d = @import_module(\"<collectives_2d/params>\");\n");
        }
        let _ = writeln!(
            layout_text,
            "\nlayout {{\n  @set_rectangle({cols}, {lanes});"
        );
        let _ = write!(
            layout_text,
            "  var Px: u16 = 0;\n  while (Px < {cols}) : (Px += 1) {{\n    var Py: u16 = 0;\n    while (Py < {lanes}) : (Py += 1) {{\n"
        );
        if staged {
            layout_text.push_str(
                "      const params = c2d.get_params(Px, Py, .{ .x_colors = .{ @get_color(0), @get_color(1) }, .x_entrypoints = .{ @get_local_task_id(10), @get_local_task_id(11) }, .y_colors = .{ @get_color(4), @get_color(5) }, .y_entrypoints = .{ @get_local_task_id(12), @get_local_task_id(13) } });\n",
            );
            layout_text.push_str("      @set_tile_code(Px, Py, \"pe.csl\", .{ .memcpy_params = memcpy.get_params(Px), .c2d_params = params });\n");
        } else {
            layout_text.push_str("      @set_tile_code(Px, Py, \"pe.csl\", .{ .memcpy_params = memcpy.get_params(Px) });\n");
        }
        layout_text.push_str("    }\n  }\n");
        if let Some(pack) = &self.pack {
            let _ = writeln!(layout_text, "  @export_name(\"{pack}\", [*]u32, true);");
        }
        if let Some(out) = &self.out {
            let _ = writeln!(layout_text, "  @export_name(\"{out}\", [*]u32, true);");
        }
        for (_, elem, export) in &self.exports {
            let _ = writeln!(
                layout_text,
                "  @export_name(\"{export}\", [*]{elem}, true);"
            );
        }
        layout_text.push_str("  @export_name(\"run\", fn()void);\n}\n");
        (pe, layout_text)
    }
}

/// The interpreter's row count of a reduction: every axis but the last.
fn canonical_rows(dims: &[u32]) -> usize {
    if dims.len() < 2 {
        return 1;
    }
    dims[..dims.len() - 1].iter().map(|&d| d as usize).product()
}

/// The CSL source of a helper block (several functions that call each
/// other travel together).
fn helper_source(name: &str) -> String {
    let mut s = String::new();
    match name {
        "g_copy" => {
            s.push_str(
                r"// A word copy through a DSD move: a copy loop would become a
// `memmove` call the PE has no library for.
var g_dummy_u32 = @zeros([1]u32);
const g_base_u32 = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> g_dummy_u32[i] });
var g_dummy_u16 = @zeros([1]u16);
const g_base_u16 = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> g_dummy_u16[i] });
fn g_copy16(dst: [*]u16, doff: i32, src: [*]u16, soff: i32, n: i32) void {
  if (n <= 0) { return; }
  var d = @set_dsd_base_addr(g_base_u16, dst);
  d = @set_dsd_length(d, @bitcast(u16, @as(i16, n)));
  d = @increment_dsd_offset(d, @as(i16, doff), u16);
  var s = @set_dsd_base_addr(g_base_u16, src);
  s = @set_dsd_length(s, @bitcast(u16, @as(i16, n)));
  s = @increment_dsd_offset(s, @as(i16, soff), u16);
  @mov16(d, s);
}
fn g_copy32(dst: [*]u32, doff: i32, src: [*]u32, soff: i32, n: i32) void {
  if (n <= 0) { return; }
  var d = @set_dsd_base_addr(g_base_u32, dst);
  d = @set_dsd_length(d, @bitcast(u16, @as(i16, n)));
  d = @increment_dsd_offset(d, @as(i16, doff), u32);
  var s = @set_dsd_base_addr(g_base_u32, src);
  s = @set_dsd_length(s, @bitcast(u16, @as(i16, n)));
  s = @increment_dsd_offset(s, @as(i16, soff), u32);
  @mov32(d, s);
}
",
            );
        }
        "g_cast" => {
            let _ = write!(
                s,
                r"// `as i32`: toward zero, saturating, NaN to 0.
fn g_f2i(x: f32) i32 {{
  if (math.isNaN_f32(x)) {{ return 0; }}
  if (x <= {lo}) {{ return {min}; }}
  if (x >= {hi}) {{ return {max}; }}
  return @as(i32, x);
}}
// `as u32`: toward zero, saturating at 0 and u32::MAX, NaN to 0 (the
// compiler's own conversion saturates at 2^31).
fn g_f2u(x: f32) u32 {{
  if (math.isNaN_f32(x)) {{ return 0; }}
  if (x <= {zero}) {{ return 0; }}
  if (x >= {top}) {{ return @as(u32, 0xFFFFFFFF); }}
  if (x >= {hi}) {{ return @as(u32, @as(i32, x - {hi})) + @as(u32, 0x80000000); }}
  return @as(u32, @as(i32, x));
}}
",
                lo = flit(-2_147_483_648.0),
                hi = flit(2_147_483_648.0),
                top = flit(4_294_967_296.0),
                zero = flit(0.0),
                min = ilit(i32::MIN),
                max = ilit(i32::MAX),
            );
        }
        "g_int" => {
            s.push_str(
                r"// The interpreter's i64 division, truncated back: by zero is 0,
// i32::MIN / -1 wraps and i32::MIN % -1 is 0.
fn g_idiv(x: i32, y: i32) i32 {
  if (y == 0) { return 0; }
  if (y == -1) { return @as(i32, 0) - x; }
  return x / y;
}
// The signed remainder goes through the unsigned one on the magnitudes
// (the compiler's own gave 4 % -5 = -4; Rust: 4); the sign is the
// dividend's, as C and Rust truncate.
fn g_irem(x: i32, y: i32) i32 {
  if (y == 0) { return 0; }
  if (y == -1) { return 0; }
  var ax: u32 = @bitcast(u32, x); if (x < 0) { ax = @as(u32, 0) - ax; }
  var ay: u32 = @bitcast(u32, y); if (y < 0) { ay = @as(u32, 0) - ay; }
  var r: u32 = ax % ay;
  if (x < 0) { return @as(i32, 0) - @bitcast(i32, r); }
  return @bitcast(i32, r);
}
fn g_udiv(x: u32, y: u32) u32 {
  if (y == 0) { return 0; }
  return x / y;
}
fn g_urem(x: u32, y: u32) u32 {
  if (y == 0) { return 0; }
  return x % y;
}
fn g_imax(x: i32, y: i32) i32 { if (x > y) { return x; } return y; }
fn g_imin(x: i32, y: i32) i32 { if (x < y) { return x; } return y; }
fn g_umax(x: u32, y: u32) u32 { if (x > y) { return x; } return y; }
fn g_umin(x: u32, y: u32) u32 { if (x < y) { return x; } return y; }
fn g_iclamp(x: i32, lo: i32, hi: i32) i32 {
  if (x < lo) { return lo; }
  if (x > hi) { return hi; }
  return x;
}
fn g_uclamp(x: u32, hi: i32) i32 {
  if (x > @as(u32, hi)) { return hi; }
  return @as(i32, x);
}
",
            );
        }
        "g_fmod" => {
            s.push_str(
                r"// `x % y` as Rust computes it for f32 (C `fmod`: exact, the sign of `x`).
fn g_fmod(x: f32, y: f32) f32 {
  if (math.isNaN_f32(y) or math.isNaN_f32(x)) { return x + y; }
  var ax: f32 = math.abs_f32(x);
  var ay: f32 = math.abs_f32(y);
  if (ay == 0.0 or math.isInf_f32(x)) { return @bitcast(f32, @as(u32, 0x7FC00000)); }
  if (math.isInf_f32(y) or ax < ay) { return x; }
  // Subtract y's largest power-of-two multiple at or below the rest.
  var r: f32 = ax;
  while (r >= ay) {
    var er: i32 = @as(i32, (@bitcast(u32, r) >> 23) & 0xFF);
    var ed: i32 = @as(i32, (@bitcast(u32, ay) >> 23) & 0xFF);
    var t: f32 = ay;
    var e: i32 = er - ed;
    while (e > 0) : (e -= 1) { t = t * 2.0; }
    if (t > r) { t = t * 0.5; }
    r = r - t;
  }
  return @bitcast(f32, (@bitcast(u32, r) & 0x7FFFFFFF) | (@bitcast(u32, x) & 0x80000000));
}
",
            );
        }
        "g_tree" => {
            s.push_str(
                r"// The interpreter's canonical float max/min: a NaN loses to the other,
// two NaNs give the identity, zeros keep the sign the fold states.
fn g_cmax(l: f32, r: f32) f32 {
  var ln: bool = math.isNaN_f32(l); var rn: bool = math.isNaN_f32(r);
  if (ln and rn) { return @bitcast(f32, @as(u32, 0xFF800000)); }
  if (ln) { return r; }
  if (rn) { return l; }
  if (l == 0.0 and r == 0.0) {
    if ((@bitcast(u32, l) & 0x80000000) != 0 and (@bitcast(u32, r) & 0x80000000) != 0) { return @bitcast(f32, @as(u32, 0x80000000)); }
    return 0.0;
  }
  if (l > r) { return l; }
  return r;
}
fn g_cmin(l: f32, r: f32) f32 {
  var ln: bool = math.isNaN_f32(l); var rn: bool = math.isNaN_f32(r);
  if (ln and rn) { return @bitcast(f32, @as(u32, 0x7F800000)); }
  if (ln) { return r; }
  if (rn) { return l; }
  if (l == 0.0 and r == 0.0) {
    if ((@bitcast(u32, l) & 0x80000000) != 0 or (@bitcast(u32, r) & 0x80000000) != 0) { return @bitcast(f32, @as(u32, 0x80000000)); }
    return 0.0;
  }
  if (l < r) { return l; }
  return r;
}
// `f32::max` / `f32::min`: a NaN operand yields the other one.
fn g_fmax(a: f32, b: f32) f32 { if (math.isNaN_f32(a)) { return b; } if (math.isNaN_f32(b)) { return a; } if (a > b) { return a; } return b; }
fn g_fmin(a: f32, b: f32) f32 { if (math.isNaN_f32(a)) { return b; } if (math.isNaN_f32(b)) { return a; } if (a < b) { return a; } return b; }
// The interpreter's 32-lane tree fold over row `off..off+n` of `src`:
// blocks of 32 lanes fold by halving (16, 8, 4, 2, 1), their results fold
// the same way level by level. `kind`: 0 sum, 1 max, 2 min.
var g_lanes = @zeros([32]f32);
fn g_fold32(kind: i32) void {
  var offset: i32 = 16;
  while (offset >= 1) : (offset = offset / 2) {
    var lane: i32 = 0;
    while (lane < offset) : (lane += 1) {
      var l: f32 = g_lanes[lane];
      var r: f32 = g_lanes[lane + offset];
      if (kind == 0) { g_lanes[lane] = l + r; }
      else if (kind == 1) { g_lanes[lane] = g_cmax(l, r); }
      else { g_lanes[lane] = g_cmin(l, r); }
    }
  }
}
fn g_tfold(src: [*]f32, off: i32, n: i32, lvl: [*]f32, kind: i32) f32 {
  var ident: f32 = 0.0;
  if (kind == 1) { ident = @bitcast(f32, @as(u32, 0xFF800000)); }
  if (kind == 2) { ident = @bitcast(f32, @as(u32, 0x7F800000)); }
  if (n == 0) { return ident; }
  if (n == 1) { return src[off]; }
  var count: i32 = 0;
  var base: i32 = 0;
  while (base < n) : (base += 32) {
    var c: i32 = 32; if (n - base < c) { c = n - base; }
    var q: i32 = 0;
    while (q < 32) : (q += 1) { if (q < c) { g_lanes[q] = src[off + base + q]; } else { g_lanes[q] = ident; } }
    g_fold32(kind);
    lvl[count] = g_lanes[0];
    count = count + 1;
  }
  while (count > 1) {
    var next: i32 = 0;
    var b2: i32 = 0;
    while (b2 < count) : (b2 += 32) {
      var c: i32 = 32; if (count - b2 < c) { c = count - b2; }
      var q: i32 = 0;
      while (q < 32) : (q += 1) { if (q < c) { g_lanes[q] = lvl[b2 + q]; } else { g_lanes[q] = ident; } }
      g_fold32(kind);
      lvl[next] = g_lanes[0];
      next = next + 1;
    }
    count = next;
  }
  return lvl[0];
}
fn g_tsum(src: [*]f32, off: i32, n: i32, lvl: [*]f32) f32 { return g_tfold(src, off, n, lvl, 0); }
fn g_tmax(src: [*]f32, off: i32, n: i32, lvl: [*]f32) f32 { return g_tfold(src, off, n, lvl, 1); }
fn g_tmin(src: [*]f32, off: i32, n: i32, lvl: [*]f32) f32 { return g_tfold(src, off, n, lvl, 2); }
",
            );
        }
        "g_sort" => {
            s.push_str(
                r"// An i32 whose order is the interpreter's descending-sort order of an
// f32: monotone in the value, -0 and +0 one key, NaN below -inf.
fn g_okey(x: f32) i32 {
  if (math.isNaN_f32(x)) { return @bitcast(i32, @as(u32, 0x80000000)); }
  var v: f32 = x;
  if (v == 0.0) { v = 0.0; }
  var bits: i32 = @bitcast(i32, v);
  if (bits < 0) { return bits ^ 0x7FFFFFFF; }
  return bits;
}
// Row `off..off+cap` of `src` as order keys with their source indices;
// lanes past `n` (a block's beyond this PE's share) key as NaN: last.
fn g_keys(src: [*]f32, off: i32, n: i32, cap: i32, key: [*]i32, idx: [*]u32) void {
  var j: i32 = 0;
  while (j < cap) : (j += 1) {
    if (j < n) { key[j] = g_okey(src[off + j]); } else { key[j] = @bitcast(i32, @as(u32, 0x80000000)); }
    idx[j] = @as(u32, j);
  }
}
// Keys at or above `t` (strictly above for `g_count_gt`) in the
// descending-sorted row `off..off+n`.
fn g_count_ge(key: [*]i32, off: i32, n: i32, t: i32) i32 {
  var lo: i32 = 0; var hi: i32 = n;
  while (lo < hi) { var mid: i32 = lo + (hi - lo) / 2; if (key[off + mid] >= t) { lo = mid + 1; } else { hi = mid; } }
  return lo;
}
fn g_count_gt(key: [*]i32, off: i32, n: i32, t: i32) i32 {
  var lo: i32 = 0; var hi: i32 = n;
  while (lo < hi) { var mid: i32 = lo + (hi - lo) / 2; if (key[off + mid] > t) { lo = mid + 1; } else { hi = mid; } }
  return lo;
}
// The value whose order key is `k` (`g_okey`'s inverse; NaN's key gives
// a NaN).
fn g_unkey(k: i32) f32 {
  if (k < 0) { return @bitcast(f32, k ^ 0x7FFFFFFF); }
  return @bitcast(f32, k);
}
// The greatest last-offered key among the PEs whose `m` candidates are
// all taken (row `rr` of `recv`, `per_pe` words a PE, candidate values):
// a pick at or below it might have been beaten by a key that PE did not
// offer.
fn g_exhausted_value(recv: [*]u32, per_pe: i32, m: i32, rr: i32, taken: [*]u16, cols: i32) i32 {
  var best: i32 = @bitcast(i32, @as(u32, 0x80000000)); var any: bool = false;
  var c: i32 = 0;
  while (c < cols) : (c += 1) {
    if (@as(i32, taken[c]) < m) { continue; }
    var kk: i32 = g_okey(@bitcast(f32, recv[c * per_pe + rr * m + m - 1]));
    if (!any or kk > best) { best = kk; any = true; }
  }
  return best;
}
// A stable merge sort of (key, idx) by key descending: equal keys keep
// their index order. `tk`/`ti` are scratch of n; the result lands in
// `key`/`idx`.
fn g_msort(key: [*]i32, idx: [*]u32, n: i32, tk: [*]i32, ti: [*]u32) void {
  var width: i32 = 1;
  var in_tmp: bool = false;
  while (width < n) : (width = width * 2) {
    var lo: i32 = 0;
    while (lo < n) : (lo += width * 2) {
      var mid: i32 = lo + width; if (mid > n) { mid = n; }
      var hi: i32 = lo + width * 2; if (hi > n) { hi = n; }
      var a: i32 = lo; var b: i32 = mid; var o: i32 = lo;
      while (o < hi) : (o += 1) {
        var take_left: bool = false;
        if (a < mid and b < hi) {
          if (in_tmp) { take_left = tk[a] >= tk[b]; } else { take_left = key[a] >= key[b]; }
        } else { take_left = a < mid; }
        if (take_left) {
          if (in_tmp) { key[o] = tk[a]; idx[o] = ti[a]; } else { tk[o] = key[a]; ti[o] = idx[a]; }
          a = a + 1;
        } else {
          if (in_tmp) { key[o] = tk[b]; idx[o] = ti[b]; } else { tk[o] = key[b]; ti[o] = idx[b]; }
          b = b + 1;
        }
      }
    }
    in_tmp = !in_tmp;
  }
  if (in_tmp) {
    var j: i32 = 0;
    while (j < n) : (j += 1) { key[j] = tk[j]; idx[j] = ti[j]; }
  }
}
",
            );
        }
        "g_rng" => {
            let f = rng::RNG_FORMULA;
            let mut rounds = String::new();
            for round in f.splitmix64_rounds {
                let _ = writeln!(rounds, "  x = x ^ (x >> {});", round.xor_shift);
                if let Some(m) = round.multiplier {
                    let _ = writeln!(rounds, "  x = x * {};", u64lit(m));
                }
            }
            let _ = write!(
                s,
                r"// splitmix64, and the uniform draw `hash_uniform(seed, j)`, bit for bit.
fn g_splitmix(v: u64) u64 {{
  var x: u64 = v;
{rounds}  return x;
}}
fn g_uniform(seed: u64, j: u32) f32 {{
  var x: u64 = seed + {stride} * (@as(u64, j) + {bias});
  var bits: u32 = @as(u32, g_splitmix(x) >> {shift});
  var raw: f32 = (@as(f32, bits) + {mid}) * {unit};
  if (raw < {cap}) {{ return raw; }}
  return {cap};
}}
",
                stride = u64lit(f.lane_stride),
                bias = u64lit(f.lane_index_bias),
                shift = f.uniform_mantissa_shift,
                mid = flit(f.uniform_midpoint),
                unit = flit(1.0 / (1u32 << f.uniform_mantissa_bits) as f32),
                cap = flit(rng::UNIFORM_MAX),
            );
        }
        other => panic!("no guest helper {other}"),
    }
    s
}
