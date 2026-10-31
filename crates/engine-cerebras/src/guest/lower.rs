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
use eta_ir::op::tags;
use eta_ir::{Dtype, RngKind, rng};
use kernels_cerebras::csl::Block;
use kernels_cerebras::phase::Shard;

use crate::fabric::{Export, Manifest, Rendered, Symbol};

mod helpers;
mod ops;

use helpers::helper_source;

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

/// Output `name` of the pass's lane `lane`, cut by the host from a fire's
/// plane `layers` deep (0: one), `width` of `dtype` wide.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Read {
    pub name: String,
    pub lane: u32,
    pub layers: u32,
    pub width: u32,
    pub dtype: Dtype,
}

/// The shape of one lowering: how many lanes, how many PEs each lane
/// spreads over, and the outputs the host cuts rows of (any other output
/// is fed as a value).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Batch {
    pub lanes: u32,
    pub cols: u32,
    pub reads: Vec<Read>,
}

/// What a packed slot carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feed {
    /// The value itself: `dtype` over `numel` elements (a wide value: the
    /// PE's block of them).
    Value(Dtype, usize),
    /// `rows` rows of `width` words the host cuts from `Layout::reads[read]`,
    /// each layer's from the lane's first row (a wide feed: the PE's block
    /// of each row).
    Rows { read: usize, rows: u32, width: u32 },
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
    /// The outputs the host cuts rows of.
    pub reads: Vec<Read>,
    /// The wide axis (the vocabulary), when the stage has one.
    pub wide: Option<u32>,
    /// Elements of the wide axis each PE holds (`block_len`): PE `x` holds
    /// `[x · block, min(v, (x + 1) · block))`, the last PEs possibly fewer
    /// or none.
    pub block: usize,
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
/// (the `guest_words` switch overrides it).
#[must_use]
pub fn guest_words() -> u64 {
    crate::switches::parse::<u64>("guest_words")
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

/// The wide axis of `package`: the last dimension of a float output it
/// reads (the vocabulary), when it reads one.
#[must_use]
pub fn wide_axis(package: &LaunchPackage) -> Option<u32> {
    package
        .values
        .iter()
        .filter(|v| v.source == ValueOrigin::Output && v.dtype == Dtype::F32)
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
                for (k, arg) in op.reads().enumerate() {
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
                        let width = read.width;
                        let per = width as usize * read.layers.max(1) as usize;
                        if per == 0 || !n.is_multiple_of(per) {
                            return refuse(format!(
                                "output `{name}` holds {n} values, not whole {width}-wide rows \
                                 of {} layer(s)",
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
                        Feed::Rows {
                            read: at,
                            rows: (n / width as usize) as u32,
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
                if w && Some(width) != wide {
                    return refuse(format!(
                        "a {width}-wide output beside a {}-wide axis",
                        wide.unwrap_or(0)
                    ));
                }
                (rows as usize * if w { block } else { width as usize }, w)
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
        for arg in op.reads() {
            last_use.insert(arg, i);
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
            Feed::Rows { .. } => {
                // The rows land in the value's array itself.
                let v = self.fresh(ty, n, slot.wide, id);
                self.exports
                    .push((v.name.clone(), ty.elem(), format!("in{k}")));
                self.bind(id, v);
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
