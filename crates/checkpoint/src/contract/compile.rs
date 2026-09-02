//! Lowering the affine fragment of [`Expr`] to byte movement: maximal
//! contiguous [`Run`]s, or [`Lowering::Gather`] where that is not affordable.

use std::collections::HashMap;

use super::Expr;
use super::infer::Checked;
use crate::error::{Error, OrOverflow};
use crate::extent::{Dim, Rect};
use crate::types::Encoding;

/// Maximum tensor rank the walker supports. GPT-OSS MXFP4 blocks are rank 4;
/// the headroom is free because coordinates live on the stack.
const MAX_RANK: usize = 8;

/// Where a run's bytes come from.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Leaf {
    /// A tensor read from the checkpoint.
    Checkpoint(String),
    /// A buffer produced by an earlier contract.
    Contract(String),
}

impl Leaf {
    pub fn name(&self) -> &str {
        match self {
            Leaf::Checkpoint(name) | Leaf::Contract(name) => name,
        }
    }
}

/// One maximal contiguous copy: `len` elements from `source`, landing at
/// `dst_elem` in the output.
///
/// Offsets are in logical elements, not bytes (element width depends on the
/// encoding). Use [`CopyList::byte_runs`] to convert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub source: RunSource,
    pub dst_elem: i64,
    pub len: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunSource {
    /// Index into [`CopyList::leaves`], plus an element offset into that leaf.
    Leaf { leaf: usize, src_elem: i64 },
    /// A hole: a coordinate no source reaches, left as the zero the
    /// destination was filled with. Introduced by [`Expr::Fill`].
    Zero,
}

/// What one expression compiles to: the two shapes an executor can be asked
/// for, not two encodings of one thing.
///
/// * A [`CopyList`] is stretches of addresses. It folds into rectangles and
///   prices as [`CopyList::cost`] strided copies.
/// * A [`GatherList`] is a table of indices. It never folds and always costs
///   one element-granular pass.
///
/// [`compile`] always tries a copy list first, falling back only when it
/// would exceed `max_runs`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Lowering {
    Copy(CopyList),
    Gather(GatherList),
}

impl Lowering {
    /// Leaves in first-use order, whichever form this is.
    pub fn leaves(&self) -> &[Leaf] {
        match self {
            Lowering::Copy(copies) => &copies.leaves,
            Lowering::Gather(gather) => &gather.leaves,
        }
    }

    /// Total elements in the output.
    pub fn elements(&self) -> i64 {
        match self {
            Lowering::Copy(copies) => copies.elements,
            Lowering::Gather(gather) => gather.elements,
        }
    }

    /// What this expression costs to execute, in the units of
    /// [`CopyList::cost`]: a gather is one pass over the destination, so it
    /// costs one, same as a whole-tensor copy.
    pub fn cost(&self) -> usize {
        match self {
            Lowering::Copy(copies) => copies.cost(),
            Lowering::Gather(_) => 1,
        }
    }

    /// The copy list, or `None` for a gather. For callers that only ever
    /// see the affine fragment.
    pub fn as_copy(&self) -> Option<&CopyList> {
        match self {
            Lowering::Copy(copies) => Some(copies),
            Lowering::Gather(_) => None,
        }
    }
}

/// One leaf, one index table: `out[r, i, j] = src[r, indices[i], j]`, with
/// `j` ranging over a [`block`](Self::block) of elements and `r` over
/// [`rows`](Self::rows).
///
/// The lowering the copy list cannot express affordably: a permutation whose
/// blocks are single elements folds to a copy list of thousands of tiny
/// rectangles, while the same permutation is one index per block.
///
/// The table is stated once and repeated over `rows`: the axes outside the
/// gathered one are untouched, so writing their product into the table
/// would just repeat the same numbers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GatherList {
    /// Leaves in first-use order, as [`CopyList::leaves`] is. A gather reads
    /// exactly one of them, but the vector is kept whole so both forms
    /// answer [`Lowering::leaves`] the same way.
    pub leaves: Vec<Leaf>,
    /// Which leaf, indexing [`leaves`](Self::leaves).
    pub leaf: usize,
    /// Destination block `i` reads source block `indices[i]`.
    pub indices: Vec<i64>,
    /// Elements in one block: the operand's extent below the gathered axis.
    /// One, when the gather is on the innermost axis.
    pub block: i64,
    /// How many times the table repeats: the operand's extent above the
    /// gathered axis.
    pub rows: i64,
    /// Elements between consecutive source rows.
    pub src_row: i64,
    /// Total elements in the output.
    pub elements: i64,
}

impl GatherList {
    /// Elements between consecutive destination rows. Derived, not stored: a
    /// destination row is exactly the table.
    pub fn dst_row(&self) -> i64 {
        self.indices.len() as i64 * self.block
    }

    /// Elements the gather reads. The whole operand, because a table may name
    /// any of its blocks and the executor reads once.
    pub fn source_elements(&self) -> i64 {
        self.rows * self.src_row
    }

    /// The table's geometry in bytes, under the encoding the tensor is
    /// stored in. Lives here because [`ByteScale`] does: a blocked payload
    /// carries its scale inside the block, and this gets the same answer
    /// [`CopyList::byte_pieces`] does.
    pub fn byte_geometry(&self, encoding: &Encoding) -> Result<GatherBytes, Error> {
        let scale = ByteScale::of(encoding);
        Ok(GatherBytes {
            block_bytes: scale.extent(self.block, "gather block")?,
            rows: u64::try_from(self.rows)
                .map_err(|_| Error::Internal("gather has a negative row count".to_string()))?,
            src_row_bytes: scale.extent(self.src_row, "gather source row")?,
        })
    }

    /// The same mapping written as one rectangle per index. Not what an
    /// executor runs (that is the table, walked once), but what the mapping
    /// means, so the reference oracle can check a gather with the same
    /// scatter it checks every copy list with.
    pub fn byte_rects(&self, encoding: &Encoding) -> Result<Vec<Rect>, Error> {
        let scale = ByteScale::of(encoding);
        let block = scale.extent(self.block, "gather block")?;
        let src_row = scale.stride(self.src_row, "gather source row")?;
        let dst_row = scale.stride(self.dst_row(), "gather destination row")?;
        self.indices
            .iter()
            .enumerate()
            .map(|(at, index)| {
                let mut dims = Vec::with_capacity(2);
                if self.rows > 1 {
                    dims.push(Dim {
                        count: self.rows,
                        src_stride: src_row,
                        dst_stride: dst_row,
                    });
                }
                dims.push(Dim {
                    count: block as i64,
                    src_stride: 1,
                    dst_stride: 1,
                });
                Ok(Rect {
                    leaf: self.leaf,
                    src_offset: scale.offset(
                        index
                            .checked_mul(self.block)
                            .or_overflow("gather source offset overflows")?,
                        "gather source",
                    )?,
                    dst_offset: scale.offset(
                        (at as i64)
                            .checked_mul(self.block)
                            .or_overflow("gather destination offset overflows")?,
                        "gather destination",
                    )?,
                    dims,
                })
            })
            .collect()
    }
}

/// A [`GatherList`]'s geometry in bytes. `indices` stays in blocks — a block
/// is the unit the table names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatherBytes {
    /// Bytes in one gathered block.
    pub block_bytes: u64,
    /// How many times the table repeats.
    pub rows: u64,
    /// Bytes between consecutive source rows.
    pub src_row_bytes: u64,
}

impl GatherBytes {
    /// Bytes the gather reads: the whole operand.
    pub fn source_bytes(&self) -> u64 {
        self.rows.saturating_mul(self.src_row_bytes)
    }
}

/// The copy-list form of a lowering: maximal contiguous stretches.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CopyList {
    /// Leaves in first-use order; [`RunSource::Leaf`] indexes this.
    pub leaves: Vec<Leaf>,
    pub runs: Vec<Run>,
    /// Total elements in the output. `runs` covers exactly this many.
    pub elements: i64,
}

impl CopyList {
    /// How many separate copies this expression costs, known before any I/O
    /// happens. Count [`CopyList::pieces`] instead when the executor can
    /// issue strided copies — a row shard is thousands of runs but one
    /// piece. Nothing on the load path reads this; it is test observability
    /// ([`CopyList::cost`] picks the lowering).
    pub fn run_count(&self) -> usize {
        self.runs.len()
    }

    /// Mean run length in elements, or 0 for an empty lowering. Test
    /// observability, as [`CopyList::run_count`] is.
    pub fn mean_run_elements(&self) -> i64 {
        if self.runs.is_empty() {
            0
        } else {
            self.elements / self.runs.len() as i64
        }
    }

    /// Fold the runs back into loop nests.
    ///
    /// Runs are the semantics; pieces are the cost. A column shard is one
    /// run per row, but every row is the same length with both offsets in
    /// arithmetic progression, so the whole thing folds to a single
    /// rectangular copy (the low IR's `Extent`).
    ///
    /// A repeated sweep: find maximal consecutive groups of same-source,
    /// same-size items whose offsets are both in arithmetic progression, and
    /// wrap each group in one more loop. Cannot change the meaning: it only
    /// rewrites a run list into a form enumerating the same `(src, dst)`
    /// pairs.
    pub fn pieces(&self) -> Vec<Piece> {
        fold(self.runs.iter().map(seed).collect())
    }

    /// What this expression costs to execute: the number of strided copies,
    /// plus one fill if the destination has holes. Known before any I/O
    /// happens. A whole tensor, a row shard, and a strided expert select all
    /// cost 1; a fusion of three sources costs 3.
    ///
    /// `plan/build.rs` reads it to decide a lowering: cost 1 means one
    /// rectangle covering the whole destination, so the tensor aliases the
    /// checkpoint bytes instead of copying. It does not decide whether to
    /// slab-scatter — that coalesces across tensors and lives in
    /// `plan/passes/rewrite.rs` after offsets are assigned.
    pub fn cost(&self) -> usize {
        self.copy_pieces().len() + usize::from(self.needs_zero_fill())
    }

    /// The pieces that actually move data.
    ///
    /// A hole is not copied as zeros; the destination is zeroed once and
    /// then not written there, so padding costs one fill rather than one
    /// copy per band. They do not fold further: the destination stride
    /// across a padded row is wider than the row, and `fold` refuses a
    /// destination that skips.
    pub fn copy_pieces(&self) -> Vec<Piece> {
        fold(
            self.runs
                .iter()
                .filter(|run| run.source != RunSource::Zero)
                .map(seed)
                .collect(),
        )
    }

    /// Whether the destination has holes, and so must be zeroed before the
    /// copies in [`CopyList::copy_pieces`] run.
    pub fn needs_zero_fill(&self) -> bool {
        self.runs.iter().any(|run| run.source == RunSource::Zero)
    }

    /// Convert to byte offsets under `encoding`.
    pub fn byte_runs(&self, encoding: &Encoding) -> Result<Vec<ByteRun>, Error> {
        let scale = ByteScale::of(encoding);
        self.runs
            .iter()
            .map(|run| {
                Ok(ByteRun {
                    source: match run.source {
                        RunSource::Leaf { leaf, src_elem } => ByteRunSource::Leaf {
                            leaf,
                            src_offset: scale.offset(src_elem, "run source")?,
                        },
                        RunSource::Zero => ByteRunSource::Zero,
                    },
                    dst_offset: scale.offset(run.dst_elem, "run destination")?,
                    len: scale.offset(run.len, "run length")?,
                })
            })
            .collect()
    }

    /// The copy pieces with every offset, stride and extent converted to
    /// bytes: the form the low IR wants (`load_plan::Extent` addresses
    /// bytes; a sub-byte encoding has no element addresses).
    pub fn byte_pieces(&self, encoding: &Encoding) -> Result<Vec<Rect>, Error> {
        let scale = ByteScale::of(encoding);
        self.copy_pieces()
            .into_iter()
            .map(|piece| {
                let RunSource::Leaf { leaf, src_elem } = piece.source else {
                    unreachable!("copy_pieces excludes holes");
                };
                let dims = piece
                    .dims
                    .iter()
                    .enumerate()
                    .map(|(level, dim)| {
                        // The innermost dimension is the contiguous block the
                        // walker found, so in bytes it counts bytes and steps
                        // one at a time. The outer ones count iterations and
                        // keep their counts; only their strides scale.
                        if level + 1 == piece.dims.len() {
                            debug_assert_eq!((dim.src_stride, dim.dst_stride), (1, 1));
                            return Ok(Dim {
                                count: scale.extent(dim.count, "piece length")? as i64,
                                src_stride: 1,
                                dst_stride: 1,
                            });
                        }
                        Ok(Dim {
                            count: dim.count,
                            src_stride: scale.stride(dim.src_stride, "source stride")?,
                            dst_stride: scale.stride(dim.dst_stride, "destination stride")?,
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                Ok(Rect {
                    leaf,
                    src_offset: scale.offset(src_elem, "piece source")?,
                    dst_offset: scale.offset(piece.dst_elem, "piece destination")?,
                    dims,
                })
            })
            .collect()
    }
}

/// Element index to byte offset, for one encoding.
///
/// Sub-byte encodings (MXFP4, AWQ/GPTQ int4) only have byte addresses on
/// group boundaries; a violation is reported rather than silently rounded.
///
/// A blocked scheme is not just a bit width: a GGUF block carries its own
/// scale inside the payload (Q4_0 spends 18 bytes on 32 elements, 2 for the
/// F16 scale and 16 for the codes), so reading it as "4 bits per element"
/// forgets the scale and drifts every row after the first.
enum ByteScale {
    /// Elements are `bits` wide and pay for nothing else.
    Bits(i64),
    /// Elements come in blocks that cost `bytes` per `elems`, scale
    /// included. Only whole blocks have addresses: half a block is codes
    /// with no scale to read them by.
    Blocked { elems: i64, bytes: i64 },
}

impl ByteScale {
    fn of(encoding: &Encoding) -> Self {
        if let Encoding::Quant(spec) = encoding
            && let Some((elems, bytes)) = spec.scheme.block_layout()
        {
            return Self::Blocked {
                elems: i64::try_from(elems).unwrap_or(0),
                bytes: i64::try_from(bytes).unwrap_or(0),
            };
        }
        Self::Bits(i64::from(bits_per_element(encoding)))
    }

    fn scaled(&self, elems: i64, what: &str) -> Result<i64, Error> {
        let (bits, block) = match *self {
            Self::Bits(bits) => (bits, None),
            Self::Blocked {
                elems: per,
                bytes: cost,
            } => (0, Some((per, cost))),
        };
        if let Some((per, cost)) = block {
            if per == 0 {
                return Err(Error::Internal(
                    "blocked encoding with no block".to_string(),
                ));
            }
            if elems % per != 0 {
                return Err(Error::Contract(format!(
                    "{what} of {elems} elements does not land on a {per}-element \
                     block boundary; a blocked payload carries its scale inside \
                     the block, so a partial block has no byte address"
                )));
            }
            return (elems / per)
                .checked_mul(cost)
                .or_overflow("byte offset overflows");
        }
        let total = elems
            .checked_mul(bits)
            .or_overflow("byte offset overflows")?;
        if total % 8 != 0 {
            return Err(Error::Contract(format!(
                "{what} of {elems} elements is not byte-aligned under a {bits} -bit encoding"
            )));
        }
        Ok(total / 8)
    }

    fn offset(&self, elems: i64, what: &str) -> Result<u64, Error> {
        u64::try_from(self.scaled(elems, what)?).map_err(|_| {
            Error::Internal(format!(
                "{what} lowered to a negative byte offset from {elems} elements"
            ))
        })
    }

    fn extent(&self, elems: i64, what: &str) -> Result<u64, Error> {
        self.offset(elems, what)
    }

    fn stride(&self, elems: i64, what: &str) -> Result<i64, Error> {
        self.scaled(elems, what)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteRun {
    pub source: ByteRunSource,
    pub dst_offset: u64,
    pub len: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteRunSource {
    Leaf { leaf: usize, src_offset: u64 },
    Zero,
}

/// One rectangular copy: a loop nest, outermost dimension first.
///
/// Reading `dims` from the outside in, the element at loop counters
/// `(i0, .., in)` moves from `src_elem + sum(ik*src_stride_k)` to
/// `dst_elem + sum(ik*dst_stride_k)`. The innermost dimension always has
/// unit strides, so every piece ends in a contiguous stretch — the shape
/// `load_plan::Extent` already carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Piece {
    pub source: RunSource,
    pub dst_elem: i64,
    pub dims: Vec<Dim>,
}

impl Piece {
    /// Elements moved by this piece.
    pub fn elements(&self) -> i64 {
        self.dims.iter().map(|dim| dim.count).product()
    }
}

/// One run as a depth-1 nest. A hole has no source address, so its source
/// stride is zero rather than one.
fn seed(run: &Run) -> Piece {
    Piece {
        source: run.source,
        dst_elem: run.dst_elem,
        dims: vec![Dim {
            count: run.len,
            src_stride: i64::from(run.source != RunSource::Zero),
            dst_stride: 1,
        }],
    }
}

fn fold(mut items: Vec<Piece>) -> Vec<Piece> {
    while let Some(folded) = fold_once(&items) {
        items = folded;
    }
    items
}

/// One sweep of the fold. Returns `None` when nothing more can be merged.
fn fold_once(items: &[Piece]) -> Option<Vec<Piece>> {
    let mut out: Vec<Piece> = Vec::new();
    let mut changed = false;
    let mut at = 0;
    while at < items.len() {
        let head = &items[at];
        // The first pair fixes the strides; the rest must match exactly.
        let mut end = at + 1;
        let mut src_stride = 0;
        let mut dst_stride = 0;
        if let Some(next) = items.get(at + 1)
            && let Some(strides) = step_between(head, next)
            // The destination must stay dense: a loop that skips in the
            // destination is a scatter, which nothing below can execute.
            && strides.1 == head.elements()
            // The source must advance forward from `file_offset`; a
            // descending progression (e.g. Concat re-joining `[up | gate]`)
            // is left unfolded instead.
            && strides.0 >= 0
        {
            (src_stride, dst_stride) = strides;
            end = at + 2;
            while let Some(further) = items.get(end) {
                if step_between(&items[end - 1], further) != Some((src_stride, dst_stride)) {
                    break;
                }
                end += 1;
            }
        }
        if end - at >= 2 {
            let mut dims = Vec::with_capacity(head.dims.len() + 1);
            dims.push(Dim {
                count: (end - at) as i64,
                src_stride,
                dst_stride,
            });
            dims.extend_from_slice(&head.dims);
            out.push(Piece {
                source: head.source,
                dst_elem: head.dst_elem,
                dims,
            });
            changed = true;
        } else {
            out.push(head.clone());
        }
        at = end;
    }
    changed.then_some(out)
}

/// The `(src, dst)` step from `a` to `b`, if `b` is a translate of `a`. Two
/// pieces only compose when they read the same leaf (or are both padding)
/// and have identical extents.
fn step_between(a: &Piece, b: &Piece) -> Option<(i64, i64)> {
    if a.dims != b.dims {
        return None;
    }
    let src_stride = match (a.source, b.source) {
        (
            RunSource::Leaf { leaf, src_elem },
            RunSource::Leaf {
                leaf: next_leaf,
                src_elem: next_elem,
            },
        ) if leaf == next_leaf => next_elem - src_elem,
        (RunSource::Zero, RunSource::Zero) => 0,
        _ => return None,
    };
    Some((src_stride, b.dst_elem - a.dst_elem))
}

/// Storage width of one logical element, in bits.
pub fn bits_per_element(encoding: &Encoding) -> u32 {
    match encoding {
        Encoding::Raw(dtype) => u32::try_from(dtype.bytes_ceil()).unwrap_or(0) * 8,
        Encoding::Quant(spec) => u32::from(spec.normalized_bits()),
    }
}

/// Compile `expr` into the lowering that satisfies it, using the types the
/// resolver already resolved for it.
///
/// `max_runs` bounds the walker's own output so a pathological expression
/// cannot make the compiler allocate without limit. It is not the cost model
/// — ask [`CopyList::cost`] what the expression actually costs. A copy list
/// is tried first, always; past the cap, an [`Expr::Gather`] over a whole
/// tensor falls back to an index table, and anything else refuses.
pub fn compile(expr: &Expr, checked: &Checked, max_runs: usize) -> Result<Lowering, Error> {
    let mut builder = Builder {
        checked,
        nodes: Vec::new(),
        leaves: Vec::new(),
        leaf_index: HashMap::new(),
    };
    let root = builder.build(expr)?;
    builder.lower(root, max_runs)
}

/// A node of the flattened, shape-annotated expression.
struct Node {
    kind: Kind,
    shape: Vec<i64>,
    /// Row-major strides of `shape`, in elements.
    strides: Vec<i64>,
    elements: i64,
}

enum Kind {
    Leaf(usize),
    Slice {
        src: usize,
        axis: usize,
        start: i64,
        /// Whether the band covers the operand's whole axis, in which case it
        /// introduces no discontinuity at all.
        whole: bool,
    },
    Stride {
        src: usize,
        axis: usize,
        start: i64,
        step: i64,
    },
    Gather {
        src: usize,
        axis: usize,
        indices: Vec<i64>,
    },
    Concat {
        axis: usize,
        /// `(offset along axis, node)`, in increasing order.
        parts: Vec<(i64, usize)>,
    },
    Transmute {
        src: usize,
    },
    /// Backed by nothing. Every coordinate under it is a hole.
    Fill,
}

struct Builder<'a> {
    checked: &'a Checked,
    nodes: Vec<Node>,
    leaves: Vec<Leaf>,
    leaf_index: HashMap<Leaf, usize>,
}

impl Builder<'_> {
    fn push(&mut self, kind: Kind, shape: Vec<i64>) -> Result<usize, Error> {
        if shape.len() > MAX_RANK {
            return Err(Error::Contract(format!(
                "rank {} exceeds the supported maximum of {MAX_RANK}",
                shape.len()
            )));
        }
        let mut strides = vec![1_i64; shape.len()];
        for axis in (0..shape.len().saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1]
                .checked_mul(shape[axis + 1])
                .or_overflow("shape overflows i64")?;
        }
        let elements = match shape.first() {
            Some(first) => first
                .checked_mul(strides[0])
                .or_overflow("element count overflows i64")?,
            None => 1,
        };
        self.nodes.push(Node {
            kind,
            shape,
            strides,
            elements,
        });
        Ok(self.nodes.len() - 1)
    }

    fn intern(&mut self, leaf: Leaf) -> usize {
        if let Some(found) = self.leaf_index.get(&leaf) {
            return *found;
        }
        let index = self.leaves.len();
        self.leaves.push(leaf.clone());
        self.leaf_index.insert(leaf, index);
        index
    }

    /// Flatten `expr` into `self.nodes`, returning the root index.
    fn build(&mut self, expr: &Expr) -> Result<usize, Error> {
        match expr {
            Expr::Src(name) => {
                let shape = self
                    .checked
                    .source(name)
                    .ok_or_else(|| {
                        Error::Internal(format!(
                            "checkpoint tensor '{name}' was not resolved by the type checker"
                        ))
                    })?
                    .shape
                    .clone();
                let leaf = self.intern(Leaf::Checkpoint(name.clone()));
                self.push(Kind::Leaf(leaf), shape)
            }
            Expr::Out(name) => {
                let shape = self
                    .checked
                    .output(name)
                    .ok_or_else(|| {
                        Error::Internal(format!(
                            "contract '{name}' was not resolved by the type checker"
                        ))
                    })?
                    .shape
                    .clone();
                let leaf = self.intern(Leaf::Contract(name.clone()));
                self.push(Kind::Leaf(leaf), shape)
            }
            Expr::Slice {
                src,
                axis,
                start,
                len,
            } => {
                let inner = self.build(src)?;
                let axis = usize::from(axis.0);
                let mut shape = self.nodes[inner].shape.clone();
                let operand_extent = axis_extent(&shape, axis, "Slice")?;
                shape[axis] = *len;
                let whole = *start == 0 && *len == operand_extent;
                self.push(
                    Kind::Slice {
                        src: inner,
                        axis,
                        start: *start,
                        whole,
                    },
                    shape,
                )
            }
            Expr::Stride {
                src,
                axis,
                start,
                len,
                step,
            } => {
                let inner = self.build(src)?;
                let axis = usize::from(axis.0);
                let mut shape = self.nodes[inner].shape.clone();
                axis_extent(&shape, axis, "Stride")?;
                shape[axis] = *len;
                self.push(
                    Kind::Stride {
                        src: inner,
                        axis,
                        start: *start,
                        step: *step,
                    },
                    shape,
                )
            }
            Expr::Gather { src, axis, indices } => {
                let inner = self.build(src)?;
                let axis = usize::from(axis.0);
                let mut shape = self.nodes[inner].shape.clone();
                axis_extent(&shape, axis, "Gather")?;
                shape[axis] = indices.len() as i64;
                self.push(
                    Kind::Gather {
                        src: inner,
                        axis,
                        indices: indices.clone(),
                    },
                    shape,
                )
            }
            Expr::Concat { axis, parts } => {
                let axis = usize::from(axis.0);
                let mut placed = Vec::with_capacity(parts.len());
                let mut offset = 0_i64;
                let mut shape = Vec::new();
                for part in parts {
                    let inner = self.build(part)?;
                    if placed.is_empty() {
                        shape = self.nodes[inner].shape.clone();
                    }
                    placed.push((offset, inner));
                    offset += axis_extent(&self.nodes[inner].shape, axis, "Concat")?;
                }
                if placed.is_empty() {
                    return Err(Error::Internal(
                        "empty Concat escaped the type checker".to_string(),
                    ));
                }
                shape[axis] = offset;
                self.push(
                    Kind::Concat {
                        axis,
                        parts: placed,
                    },
                    shape,
                )
            }
            // Nothing moves. When the element width changes, `infer_transmute`
            // has already proved the operand is a whole tensor, so the leaf
            // is simply read under its new type over the same bytes.
            Expr::Transmute { src, to } => match src.as_ref() {
                Expr::Src(name) => {
                    let leaf = self.intern(Leaf::Checkpoint(name.clone()));
                    self.push(Kind::Leaf(leaf), to.shape.clone())
                }
                Expr::Out(name) => {
                    let leaf = self.intern(Leaf::Contract(name.clone()));
                    self.push(Kind::Leaf(leaf), to.shape.clone())
                }
                _ => {
                    let inner = self.build(src)?;
                    self.push(Kind::Transmute { src: inner }, to.shape.clone())
                }
            },
            Expr::Fill { ty, .. } => self.push(Kind::Fill, ty.shape.clone()),
            // Each of these needs a kernel; lowering them is `plan::build`'s
            // job, and reaching here means one was nested where only the
            // affine fragment fits.
            Expr::Repack { .. } | Expr::Cast { .. } | Expr::Scale { .. } | Expr::Bias { .. } => {
                Err(Error::Contract(format!(
                    "{} needs a kernel and cannot be lowered to byte runs",
                    expr.node_name()
                )))
            }
            Expr::Shard { .. } => Err(Error::Internal(
                "Shard reached lowering; Resolver::specialize rewrites it into \
                 this rank's Slice, and byte offsets cannot be symbolic"
                    .to_string(),
            )),
            Expr::SrcIndexed(template) => Err(Error::Internal(format!(
                "SrcIndexed('{template}') reached lowering; Resolver::specialize \
                 substitutes this instance's index, and a tensor name cannot be \
                 symbolic"
            ))),
            Expr::Select { .. } => Err(Error::Internal(
                "Select reached lowering; Resolver::specialize rewrites it into \
                 this instance's Slice, and byte offsets cannot be symbolic"
                    .to_string(),
            )),
        }
    }

    /// Pick a lowering: the copy list if it fits, the index table if the
    /// expression has one, and the refusal if it has neither.
    fn lower(&mut self, root: usize, max_runs: usize) -> Result<Lowering, Error> {
        if let Some(runs) = self.walk(root, max_runs)? {
            return Ok(Lowering::Copy(CopyList {
                leaves: std::mem::take(&mut self.leaves),
                runs,
                elements: self.nodes[root].elements,
            }));
        }
        let Some(gather) = self.gather(root)? else {
            return Err(Error::Contract(format!(
                "expression breaks into more than {max_runs} contiguous \
                 stretches; only a Gather over a whole tensor has a lowering \
                 that is not a copy list, and this is not one"
            )));
        };
        Ok(gather)
    }

    /// The index-table lowering, when the root is a [`Expr::Gather`] whose
    /// operand is a whole tensor. Only then: the table addresses the leaf
    /// directly, so nothing may sit between them, and a gather over a
    /// computed operand would need a second instruction to materialize it.
    fn gather(&mut self, root: usize) -> Result<Option<Lowering>, Error> {
        let Kind::Gather { src, axis, indices } = &self.nodes[root].kind else {
            return Ok(None);
        };
        let (src, axis, indices) = (*src, *axis, indices.clone());
        let Kind::Leaf(leaf) = self.nodes[src].kind else {
            return Ok(None);
        };
        let node = &self.nodes[root];
        let block = node.strides[axis];
        let dst_row = node
            .shape[axis]
            .checked_mul(block)
            .or_overflow("gather destination row overflows i64")?;
        if dst_row <= 0 {
            return Ok(None);
        }
        let src_row = self.nodes[src].shape[axis]
            .checked_mul(block)
            .or_overflow("gather source row overflows i64")?;
        Ok(Some(Lowering::Gather(GatherList {
            leaves: std::mem::take(&mut self.leaves),
            leaf,
            indices,
            block,
            rows: node.elements / dst_row,
            src_row,
            elements: node.elements,
        })))
    }

    /// Walk the output in flat order, emitting one run per maximal contiguous
    /// stretch. `None` when the list would exceed `max_runs`: the walker
    /// declines rather than fails, and [`Builder::lower`] turns that into an
    /// answer or an error.
    fn walk(&mut self, root: usize, max_runs: usize) -> Result<Option<Vec<Run>>, Error> {
        let total = self.nodes[root].elements;
        let mut runs: Vec<Run> = Vec::new();
        let mut flat = 0_i64;
        while flat < total {
            let mut coord = Coord::default();
            unflatten(flat, &self.nodes[root].shape, &mut coord);
            let (found, span) = self.step(root, &coord, flat)?;
            let span = span.clamp(1, total - flat);
            let source = match found {
                Some((leaf, src_elem)) => RunSource::Leaf { leaf, src_elem },
                None => RunSource::Zero,
            };
            match runs.last_mut() {
                // Each node's bound is conservative, so genuinely adjacent
                // stretches can arrive as two runs; merge them.
                Some(last) if adjacent(last, source, flat) => last.len += span,
                _ => {
                    if runs.len() >= max_runs {
                        return Ok(None);
                    }
                    runs.push(Run {
                        source,
                        dst_elem: flat,
                        len: span,
                    });
                }
            }
            flat += span;
        }
        Ok(Some(runs))
    }

    /// Resolve one output position and report how far the mapping stays
    /// contiguous from there. `(None, n)` means `n` padded elements.
    fn step(
        &self,
        index: usize,
        coord: &Coord,
        flat: i64,
    ) -> Result<(Option<(usize, i64)>, i64), Error> {
        let node = &self.nodes[index];
        let remaining = node.elements - flat;
        match &node.kind {
            Kind::Leaf(leaf) => Ok((Some((*leaf, flat)), remaining)),
            Kind::Slice {
                src,
                axis,
                start,
                whole,
            } => {
                let mut inner = *coord;
                inner.dims[*axis] = start + coord.dims[*axis];
                let inner_flat = flatten(&inner, &self.nodes[*src].shape);
                let (found, span) = self.step(*src, &inner, inner_flat)?;
                // A band leaves every inner extent alone, so the source
                // advances in lockstep except across a wrap of `axis`.
                let limit = if *whole || *axis == 0 {
                    i64::MAX
                } else {
                    distance_to(node, coord, flat, *axis - 1, coord.dims[*axis - 1] + 1)
                };
                Ok((found, span.min(limit).min(remaining)))
            }
            // Where a band advances in lockstep, a stride does not: the
            // source jumps by `step` every time `axis` increments.
            Kind::Stride {
                src,
                axis,
                start,
                step,
            } => {
                let mut inner = *coord;
                inner.dims[*axis] = start + coord.dims[*axis] * step;
                let inner_flat = flatten(&inner, &self.nodes[*src].shape);
                let (found, span) = self.step(*src, &inner, inner_flat)?;
                let limit = distance_to(node, coord, flat, *axis, coord.dims[*axis] + 1);
                Ok((found, span.min(limit).min(remaining)))
            }
            // A gather's run ends at the first index not one past the last,
            // so a permutation of contiguous blocks costs one run per block.
            Kind::Gather { src, axis, indices } => {
                let at = coord.dims[*axis] as usize;
                let mut inner = *coord;
                inner.dims[*axis] = *indices.get(at).ok_or_else(|| {
                    Error::Internal(format!("Gather has no index for position {at}"))
                })?;
                let inner_flat = flatten(&inner, &self.nodes[*src].shape);
                let (found, span) = self.step(*src, &inner, inner_flat)?;
                let mut end = at + 1;
                while end < indices.len() && indices[end] == indices[end - 1] + 1 {
                    end += 1;
                }
                let limit = distance_to(node, coord, flat, *axis, end as i64);
                Ok((found, span.min(limit).min(remaining)))
            }
            Kind::Concat { axis, parts } => {
                let at = coord.dims[*axis];
                let (offset, part) = parts
                    .iter()
                    .rev()
                    .find(|(offset, _)| *offset <= at)
                    .copied()
                    .ok_or_else(|| {
                        Error::Internal(format!("Concat has no part covering index {at}"))
                    })?;
                let mut inner = *coord;
                inner.dims[*axis] = at - offset;
                let inner_flat = flatten(&inner, &self.nodes[part].shape);
                let (found, span) = self.step(part, &inner, inner_flat)?;
                let part_end = offset + self.nodes[part].shape[*axis];
                let limit = distance_to(node, coord, flat, *axis, part_end);
                Ok((found, span.min(limit).min(remaining)))
            }
            Kind::Transmute { src } => {
                // A rename preserves flat order, so it is the identity here.
                let mut inner = Coord::default();
                unflatten(flat, &self.nodes[*src].shape, &mut inner);
                let (found, span) = self.step(*src, &inner, flat)?;
                Ok((found, span.min(remaining)))
            }
            // The whole node is a hole, so the walk can jump to its end.
            Kind::Fill => Ok((None, remaining)),
        }
    }
}

/// The operand's extent along `axis`, or an internal error if there is none
/// (the type checker bounds-checks every axis first, so this means a
/// compiler bug). Reported rather than panicked: this crate is reached
/// across an FFI boundary, where an unwind is not recoverable.
fn axis_extent(shape: &[i64], axis: usize, node: &str) -> Result<i64, Error> {
    shape.get(axis).copied().ok_or_else(|| {
        Error::Internal(format!(
            "{node} axis {axis} escaped the type checker (operand rank {})",
            shape.len()
        ))
    })
}

fn adjacent(last: &Run, source: RunSource, flat: i64) -> bool {
    if last.dst_elem + last.len != flat {
        return false;
    }
    match (last.source, source) {
        (RunSource::Zero, RunSource::Zero) => true,
        (
            RunSource::Leaf { leaf, src_elem },
            RunSource::Leaf {
                leaf: next,
                src_elem: next_elem,
            },
        ) => leaf == next && src_elem + last.len == next_elem,
        _ => false,
    }
}

/// Flat-index distance from `flat` until `coord[axis]` reaches `target`.
fn distance_to(node: &Node, coord: &Coord, flat: i64, axis: usize, target: i64) -> i64 {
    let stride = node.strides[axis];
    (target - coord.dims[axis]) * stride - flat.rem_euclid(stride)
}

#[derive(Clone, Copy)]
struct Coord {
    dims: [i64; MAX_RANK],
}

impl Default for Coord {
    fn default() -> Self {
        Self {
            dims: [0; MAX_RANK],
        }
    }
}

fn unflatten(flat: i64, shape: &[i64], out: &mut Coord) {
    let mut rest = flat;
    for axis in (0..shape.len()).rev() {
        let extent = shape[axis].max(1);
        out.dims[axis] = rest % extent;
        rest /= extent;
    }
}

fn flatten(coord: &Coord, shape: &[i64]) -> i64 {
    let mut flat = 0_i64;
    for (axis, extent) in shape.iter().enumerate() {
        flat = flat * extent + coord.dims[axis];
    }
    flat
}

