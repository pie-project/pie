//! Lowering the affine fragment of [`Expr`] to byte movement.
//!
//! Every expression built from `Src`, `Out`, `Fill`, `Slice`, `Concat` and
//! `Transmute` denotes a piecewise-affine partial map from output coordinates to
//! source coordinates. This module evaluates that map into a list of maximal
//! contiguous [`Run`]s — the only form the rest of the loader needs — without
//! materializing any intermediate buffer, however deeply the expression nests.
//!
//! The two escape hatches ([`Expr::Repack`], [`Expr::Cast`]) are rejected
//! here by design: they need a kernel, so they are the planner's problem.
//!
//! See `loader/spec.md` §3.1 and §3.3.

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
/// Offsets are in *logical elements*, not bytes, because the element width
/// depends on the encoding. Use [`Lowering::byte_runs`] to convert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub source: RunSource,
    pub dst_elem: i64,
    pub len: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunSource {
    /// Index into [`Lowering::leaves`], plus an element offset into that leaf.
    Leaf { leaf: usize, src_elem: i64 },
    /// A hole: a coordinate no source reaches, left as the zero the
    /// destination was filled with. Introduced by [`Expr::Fill`].
    Zero,
}

/// The compiled form of one affine expression.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Lowering {
    /// Leaves in first-use order; [`RunSource::Leaf`] indexes this.
    pub leaves: Vec<Leaf>,
    pub runs: Vec<Run>,
    /// Total elements in the output. `runs` covers exactly this many.
    pub elements: i64,
}

impl Lowering {
    /// How many separate copies this expression costs.
    ///
    /// This is the cost model of `spec.md` §3.3: few long runs lower to DMA,
    /// many short ones to a gather kernel. It is known before any I/O happens.
    ///
    /// Count [`Lowering::pieces`] instead when the executor can issue strided
    /// copies — a row shard is thousands of runs but one piece.
    ///
    /// Nothing on the load path reads this: [`Lowering::cost`] is what picks a
    /// lowering. It is here so this module's own tests can state the shape they
    /// expect — "a row shard of a fused bank is 2048 runs averaging 3072
    /// elements" is a claim about the algebra, and asserting it directly is how
    /// a regression in `compile` shows up as a failing number rather than as a
    /// slower plan nobody times.
    pub fn run_count(&self) -> usize {
        self.runs.len()
    }

    /// Mean run length in elements, or 0 for an empty lowering.
    ///
    /// Test observability, as [`Lowering::run_count`] is.
    pub fn mean_run_elements(&self) -> i64 {
        if self.runs.is_empty() {
            0
        } else {
            self.elements / self.runs.len() as i64
        }
    }

    /// Fold the runs back into loop nests.
    ///
    /// Runs are the semantics; pieces are the cost. A column shard is one run
    /// per row — thousands of them — but every row is the same length and the
    /// source and destination both advance by a fixed stride, so the whole
    /// thing is a single rectangular copy. That is exactly what the low IR's
    /// `Extent` expresses, and emitting the unfolded run list instead
    /// would inflate the plan by three orders of magnitude for every sharded
    /// tensor in the model.
    ///
    /// The pass is a repeated sweep: find maximal consecutive groups of
    /// same-source, same-size items whose source and destination offsets are
    /// both in arithmetic progression, and wrap each group in one more loop.
    /// Because it only ever rewrites a verified run list into a form that
    /// enumerates the same `(src, dst)` pairs, it cannot change the meaning.
    pub fn pieces(&self) -> Vec<Piece> {
        fold(self.runs.iter().map(seed).collect())
    }

    /// What this expression costs to execute: the number of strided copies,
    /// plus one fill if the destination has holes.
    ///
    /// This is the cost model of `spec.md` §3.3, and it is known before any I/O
    /// happens. A whole tensor, a row shard and a strided expert select all
    /// cost 1; a fusion of three sources costs 3.
    ///
    /// `plan/build.rs` reads it to decide a lowering: cost 1 is one rectangle
    /// covering the whole destination, so the tensor aliases the checkpoint
    /// bytes or views a buffer instead of copying. That is the cheapest thing
    /// the loader does and it is this number that selects it.
    ///
    /// It does *not* decide whether to slab-scatter. That choice weighs one
    /// over-read against many small reads, so its inputs are the gaps between
    /// source ranges and the ratio of span to payload — and it coalesces
    /// *across tensors*, sorted by file offset. A `Lowering` is one tensor's
    /// expression and has neither, which is why the decision lives in
    /// `plan/passes/rewrite.rs` after the offsets are assigned, and why
    /// `spec.md` §3.3 calls its thresholds a loader policy rather than a
    /// function of the algebra.
    pub fn cost(&self) -> usize {
        self.copy_pieces().len() + usize::from(self.needs_zero_fill())
    }

    /// The pieces that actually move data.
    ///
    /// A hole is not produced by copying zeros into the destination; it is
    /// produced by zeroing the destination once and then not writing there, so
    /// padding costs one fill rather than one copy per band. For a head-dim
    /// pad that halves the piece count: `2·n_heads` alternating data and zero
    /// runs become `n_heads` data runs and one fill.
    ///
    /// They do not fold further. The destination stride across a padded row is
    /// wider than the row, and `fold` refuses a destination that skips — see
    /// `spec.md` §3.3, whose cost table prices exactly this case (a `[4, 4]`
    /// padded by one column) at 5.
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
    /// copies in [`Lowering::copy_pieces`] run.
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

    /// The copy pieces with every offset, stride and extent converted to bytes.
    ///
    /// This is the form the low IR wants: `load_plan::Extent` addresses
    /// bytes, and a sub-byte encoding has no element addresses to speak of.
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
/// Sub-byte encodings (MXFP4, AWQ/GPTQ int4) only have byte addresses on group
/// boundaries. The checker's block-alignment rules (`spec.md` §3.2) are what
/// make this conversion total in practice, and a violation is reported rather
/// than silently rounded.
struct ByteScale {
    bits: i64,
}

impl ByteScale {
    fn of(encoding: &Encoding) -> Self {
        Self {
            bits: i64::from(bits_per_element(encoding)),
        }
    }

    fn scaled(&self, elems: i64, what: &str) -> Result<i64, Error> {
        let total = elems
            .checked_mul(self.bits)
            .or_overflow("byte offset overflows")?;
        if total % 8 != 0 {
            return Err(Error::Contract(format!(
                "{what} of {elems} elements is not byte-aligned under a {} -bit encoding",
                self.bits
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
/// `(i0, .., in)` moves from `src_elem + Σ iₖ·src_strideₖ` to
/// `dst_elem + Σ iₖ·dst_strideₖ`. The innermost dimension always has unit
/// strides, so every piece ends in a contiguous stretch.
///
/// This is the shape `load_plan::Extent` already carries, which is why
/// the fold exists: it is the difference between one instruction and one
/// instruction per row.
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
        // How far does the arithmetic progression starting here extend? The
        // first pair fixes the strides; the rest must match it exactly.
        let mut end = at + 1;
        let mut src_stride = 0;
        let mut dst_stride = 0;
        if let Some(next) = items.get(at + 1)
            && let Some(strides) = step_between(head, next)
            // The destination must stay dense. A copy engine addresses a
            // contiguous output range; a loop that skips in the destination is
            // a scatter, and nothing below this layer can execute one. See
            // `spec.md` §3.3.
            && strides.1 == head.elements()
            // The source must advance forward. A plan addresses its source as
            // a span starting at `file_offset`, so a descending progression
            // would read below that anchor. Such a run is a perfectly good
            // rectangle — it just is not one this ABI can name, and the pieces
            // are individually copyable, so leave them unfolded. `Concat` in
            // descending source order (e.g. re-joining a fused gate/up weight
            // as `[up | gate]`) is the case that reaches this.
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

/// The `(src, dst)` step from `a` to `b`, if `b` is a translate of `a`.
///
/// Two pieces only compose into a loop when they read the same leaf — or are
/// both padding — and have identical extents. A zero fill has no source
/// address, so its source stride is fixed at zero and only the destination has
/// to progress.
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
        Encoding::Raw(dtype) => u32::try_from(dtype.bytes()).unwrap_or(0) * 8,
        Encoding::Quant(spec) => u32::from(spec.normalized_bits()),
    }
}

/// Compile `expr` into runs, using the types the resolver
/// already resolved for it.
///
/// `max_runs` bounds the walker's own output so a pathological expression
/// cannot make the compiler allocate without limit. It is *not* the cost
/// model — a row shard is thousands of runs and one copy. Ask
/// [`Lowering::cost`] what the expression actually costs.
pub fn compile(expr: &Expr, checked: &Checked, max_runs: usize) -> Result<Lowering, Error> {
    let mut builder = Builder {
        checked,
        nodes: Vec::new(),
        leaves: Vec::new(),
        leaf_index: HashMap::new(),
    };
    let root = builder.build(expr)?;
    builder.walk(root, max_runs)
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
            // Nothing moves, so the only question is whether the operand's
            // element grid survives. It does when the element width is
            // unchanged, and flat order is then the identity; when it is not,
            // `infer_transmute` has already proved the operand is a whole tensor,
            // so the leaf is simply read under its new type over the same
            // bytes.
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
            // The kernel column of the algebra, as a set. Each of these costs
            // something no arrangement of byte runs can express, so lowering
            // them is `plan::build`'s job and reaching here means a kernel node
            // was nested where only the affine fragment fits.
            Expr::Repack { .. } | Expr::Cast { .. } | Expr::Scale { .. } => {
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

    /// Walk the output in flat order, emitting one run per maximal contiguous
    /// stretch.
    fn walk(&mut self, root: usize, max_runs: usize) -> Result<Lowering, Error> {
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
                // stretches can arrive as two runs. Merging keeps the cost model
                // honest.
                Some(last) if adjacent(last, source, flat) => last.len += span,
                _ => {
                    if runs.len() >= max_runs {
                        return Err(Error::Contract(format!(
                            "expression breaks into more than {max_runs} contiguous \
                             stretches; it needs a gather lowering, not a copy list"
                        )));
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
        Ok(Lowering {
            leaves: std::mem::take(&mut self.leaves),
            runs,
            elements: total,
        })
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
                // A band leaves every inner extent alone, so the source advances
                // in lockstep with the output even across increments of `axis`.
                // Only the wrap of `axis` breaks it — and not even that when the
                // band covers the whole axis.
                let limit = if *whole || *axis == 0 {
                    i64::MAX
                } else {
                    distance_to(node, coord, flat, *axis - 1, coord.dims[*axis - 1] + 1)
                };
                Ok((found, span.min(limit).min(remaining)))
            }
            // Where a band advances in lockstep, a stride does not: the source
            // jumps by `step` every time `axis` increments, so the run ends
            // there. This is the whole cost difference between the two nodes.
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
            // A gather's run ends at the first index that is not one past the
            // last, which is what makes a list with consecutive stretches
            // cheaper than one without: the byte runs coalesce over them, and a
            // permutation of contiguous blocks costs one run per block rather
            // than one per element.
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
            // The whole node is a hole, so the walk can jump to its end in one
            // step. Where the hole sits inside the output is the enclosing
            // `Concat`'s business, which is the point of the node being a leaf.
            Kind::Fill => Ok((None, remaining)),
        }
    }
}

/// The operand's extent along `axis`, or an internal error if there is none.
///
/// The type checker bounds-checks every axis before lowering ever runs, so a
/// failure here means the two disagree — a compiler bug. It is reported rather
/// than panicked because this crate is reached across an FFI boundary, where an
/// unwind is not recoverable.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::infer::{Checked, CheckpointTypes, Resolver, infer_type};
    use crate::contract::{ModelContract, Partition, TensorContract, TensorType};
    use crate::types::{Axis, DType, QuantScheme, QuantSpec};

    struct Fake(HashMap<String, TensorType>);

    impl Fake {
        fn new(entries: &[(&str, &[i64])]) -> Self {
            Self(
                entries
                    .iter()
                    .map(|(name, shape)| {
                        (
                            (*name).to_string(),
                            TensorType::raw(shape.to_vec(), DType::BF16),
                        )
                    })
                    .collect(),
            )
        }
    }

    impl CheckpointTypes for Fake {
        fn tensor_type(&self, name: &str) -> Option<TensorType> {
            self.0.get(name).cloned()
        }
    }

    /// What the retired `Pad` node meant, written as the composition that
    /// replaced it. `shape` is the operand's, which the old node left implicit
    /// and which the type checker now sees.
    fn pad(src: Expr, shape: &[i64], axis: u8, before: i64, after: i64) -> Expr {
        let leg = |extent: i64| {
            let mut shape = shape.to_vec();
            shape[usize::from(axis)] = extent;
            Expr::fill(0.0, TensorType::raw(shape, DType::BF16))
        };
        let mut parts = Vec::new();
        if before > 0 {
            parts.push(leg(before));
        }
        parts.push(src);
        if after > 0 {
            parts.push(leg(after));
        }
        Expr::concat(axis, parts)
    }

    /// Resolve a whole contract the way the frontend does, so that a view can
    /// be compiled against the bank it reads.
    fn resolve(contract: &ModelContract, checkpoint: &dyn CheckpointTypes) -> Checked {
        let mut resolver = Resolver::new(checkpoint, Partition::WHOLE);
        for tensor in &contract.tensors {
            let ty = resolver
                .infer(&tensor.expr, &tensor.name)
                .expect("type check failed");
            resolver.publish(&tensor.name, ty);
        }
        resolver.into_checked()
    }

    fn lower(expr: Expr, checkpoint: &dyn CheckpointTypes) -> Lowering {
        let (_, checked) = infer_type(&expr, checkpoint).expect("type check failed");
        compile(&expr, &checked, 1_000_000).expect("lowering failed")
    }

    fn qwen3() -> Fake {
        Fake::new(&[
            ("q_proj", &[2048, 2048]),
            ("k_proj", &[1024, 2048]),
            ("v_proj", &[1024, 2048]),
        ])
    }

    fn srcs(lowering: &Lowering) -> Vec<i64> {
        lowering
            .runs
            .iter()
            .map(|run| match run.source {
                RunSource::Leaf { src_elem, .. } => src_elem,
                RunSource::Zero => -1,
            })
            .collect()
    }

    fn mxfp4() -> QuantSpec {
        QuantSpec {
            scheme: QuantScheme::Mxfp4E2M1E8M0,
            logical_dtype: DType::BF16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: Some(Axis(1)),
        }
    }

    #[test]
    fn a_bare_source_is_one_run() {
        let out = lower(Expr::src("q_proj"), &qwen3());
        assert_eq!(out.elements, 2048 * 2048);
        assert_eq!(
            out.runs,
            vec![Run {
                source: RunSource::Leaf {
                    leaf: 0,
                    src_elem: 0
                },
                dst_elem: 0,
                len: 2048 * 2048,
            }]
        );
    }

    #[test]
    fn qkv_fusion_is_three_runs() {
        let out = lower(
            Expr::concat(
                0,
                vec![
                    Expr::src("q_proj"),
                    Expr::src("k_proj"),
                    Expr::src("v_proj"),
                ],
            ),
            &qwen3(),
        );
        assert_eq!(out.run_count(), 3);
        assert_eq!(out.leaves.len(), 3);
        assert_eq!(srcs(&out), vec![0, 0, 0]);
        let lens: Vec<i64> = out.runs.iter().map(|run| run.len).collect();
        assert_eq!(lens, vec![2048 * 2048, 1024 * 2048, 1024 * 2048]);
        let dsts: Vec<i64> = out.runs.iter().map(|run| run.dst_elem).collect();
        assert_eq!(dsts, vec![0, 2048 * 2048, 3072 * 2048]);
    }

    #[test]
    fn tp_shard_then_fuse_is_still_three_runs() {
        // spec.md §5. Rows are contiguous, so sharding costs nothing extra.
        let out = lower(
            Expr::concat(
                0,
                vec![
                    Expr::src("q_proj").slice(0, 1024, 1024),
                    Expr::src("k_proj").slice(0, 512, 512),
                    Expr::src("v_proj").slice(0, 512, 512),
                ],
            ),
            &qwen3(),
        );
        assert_eq!(out.run_count(), 3);
        assert_eq!(out.elements, 2048 * 2048);
        assert_eq!(srcs(&out), vec![1024 * 2048, 512 * 2048, 512 * 2048]);
    }

    #[test]
    fn column_shard_costs_one_run_per_row() {
        // Slicing the innermost axis is inherently fragmented; the cost model
        // must say so rather than hide it.
        let checkpoint = Fake::new(&[("down_proj", &[2048, 6144])]);
        let out = lower(Expr::src("down_proj").slice(1, 0, 3072), &checkpoint);
        assert_eq!(out.run_count(), 2048);
        assert_eq!(out.mean_run_elements(), 3072);
        assert_eq!(srcs(&out)[1], 6144);
        assert_eq!(out.runs[1].dst_elem, 3072);
    }

    #[test]
    fn strided_select_picks_alternating_rows() {
        let checkpoint = Fake::new(&[("gate_up", &[8, 4])]);
        let gate = lower(Expr::src("gate_up").stride(0, 0, 4, 2), &checkpoint);
        let up = lower(Expr::src("gate_up").stride(0, 1, 4, 2), &checkpoint);
        assert_eq!(srcs(&gate), vec![0, 8, 16, 24]);
        assert_eq!(srcs(&up), vec![4, 12, 20, 28]);
        assert!(gate.runs.iter().all(|run| run.len == 4));
    }

    #[test]
    fn expert_stack_is_one_run_per_expert() {
        let checkpoint = Fake::new(&[("e0", &[6144, 2048]), ("e1", &[6144, 2048])]);
        let out = lower(
            Expr::concat(
                0,
                vec![
                    Expr::src("e0").transmute(TensorType::raw(vec![1, 6144, 2048], DType::BF16)),
                    Expr::src("e1").transmute(TensorType::raw(vec![1, 6144, 2048], DType::BF16)),
                ],
            ),
            &checkpoint,
        );
        assert_eq!(out.run_count(), 2);
        assert_eq!(out.elements, 2 * 6144 * 2048);
    }

    #[test]
    fn interleaved_gate_up_shard_matches_the_qwen_moe_pass() {
        // The hand-written MoE pass this replaced emitted two byte spans per
        // expert: the local half of gate, then the local half of up. The
        // algebra derives the same thing from a declaration that mentions
        // neither experts nor MoE.
        let checkpoint = Fake::new(&[("gate_up", &[2, 8, 4])]);
        let out = lower(
            Expr::concat(
                1,
                vec![
                    Expr::src("gate_up").slice(1, 0, 2),
                    Expr::src("gate_up").slice(1, 4, 2),
                ],
            ),
            &checkpoint,
        );
        assert_eq!(out.run_count(), 4);
        assert_eq!(out.elements, 2 * 4 * 4);
        assert_eq!(srcs(&out), vec![0, 16, 32, 48]);
        let dsts: Vec<i64> = out.runs.iter().map(|run| run.dst_elem).collect();
        assert_eq!(dsts, vec![0, 8, 16, 24]);
        // One leaf, read twice.
        assert_eq!(out.leaves.len(), 1);
    }

    #[test]
    fn pad_on_the_outer_axis_brackets_the_data() {
        let checkpoint = Fake::new(&[("w", &[3, 4])]);
        let out = lower(pad(Expr::src("w"), &[3, 4], 0, 1, 2), &checkpoint);
        assert_eq!(out.elements, 6 * 4);
        assert_eq!(out.run_count(), 3);
        assert_eq!(out.runs[0].source, RunSource::Zero);
        assert_eq!(out.runs[0].len, 4);
        assert_eq!(out.runs[1].len, 12);
        assert_eq!(out.runs[2].source, RunSource::Zero);
        assert_eq!(out.runs[2].len, 8);
    }

    #[test]
    fn pad_on_an_inner_axis_interleaves() {
        let checkpoint = Fake::new(&[("w", &[3, 4])]);
        let out = lower(pad(Expr::src("w"), &[3, 4], 1, 0, 2), &checkpoint);
        assert_eq!(out.elements, 3 * 6);
        assert_eq!(out.run_count(), 6);
        assert_eq!(out.runs[0].len, 4);
        assert_eq!(out.runs[1].source, RunSource::Zero);
        assert_eq!(out.runs[1].len, 2);
    }

    #[test]
    fn a_view_reads_from_the_contract_that_backs_it() {
        let contract = ModelContract {
            alignment: 256,
            tensors: vec![
                TensorContract::new(
                    "qkv",
                    Expr::concat(0, vec![Expr::src("q_proj"), Expr::src("k_proj")]),
                    vec![3072, 2048],
                    Encoding::Raw(DType::BF16),
                ),
                TensorContract::new(
                    "k_view",
                    Expr::out("qkv").slice(0, 2048, 1024),
                    vec![1024, 2048],
                    Encoding::Raw(DType::BF16),
                ),
            ],
            groups: Vec::new(),
        };
        let checked = resolve(&contract, &qwen3());
        let view = compile(&contract.tensors[1].expr, &checked, 1024).unwrap();
        assert_eq!(view.leaves, vec![Leaf::Contract("qkv".to_string())]);
        assert_eq!(
            view.runs,
            vec![Run {
                source: RunSource::Leaf {
                    leaf: 0,
                    src_elem: 2048 * 2048
                },
                dst_elem: 0,
                len: 1024 * 2048,
            }]
        );
    }

    #[test]
    fn every_output_element_is_covered_exactly_once() {
        let checkpoint = Fake::new(&[("a", &[4, 6]), ("b", &[2, 6])]);
        let out = lower(
            Expr::concat(
                0,
                vec![
                    Expr::src("a").stride(0, 0, 2, 2),
                    pad(Expr::src("b"), &[2, 6], 1, 1, 0).slice(1, 0, 6),
                ],
            ),
            &checkpoint,
        );
        let mut next = 0;
        for run in &out.runs {
            assert_eq!(run.dst_elem, next, "gap or overlap at element {next}");
            next += run.len;
        }
        assert_eq!(next, out.elements);
    }

    #[test]
    fn deep_nesting_still_reaches_the_leaves() {
        // Slice(Concat(Slice(...))) — the composition the hand-written per-model
        // fusion pass refused.
        let checkpoint = Fake::new(&[("a", &[8, 4]), ("b", &[8, 4])]);
        let out = lower(
            Expr::concat(
                0,
                vec![Expr::src("a").slice(0, 2, 4), Expr::src("b").slice(0, 0, 4)],
            )
            .slice(0, 2, 4),
            &checkpoint,
        );
        assert_eq!(out.elements, 16);
        // Rows 2,3 of the concat are a's rows 4,5; rows 4,5 are b's rows 0,1.
        assert_eq!(srcs(&out), vec![16, 0]);
        assert_eq!(out.runs.iter().map(|run| run.len).sum::<i64>(), 16);
    }

    #[test]
    fn the_cost_model_charges_for_copies_not_for_runs() {
        // These are the four shapes the plan is made of. Each costs what it
        // should cost, and none of them costs what its run count says.
        let ck = Fake::new(&[("w", &[2048, 6144]), ("e", &[8, 4]), ("p", &[4, 4])]);
        // A whole tensor, a row shard and a strided select are each one copy.
        assert_eq!(lower(Expr::src("w"), &ck).cost(), 1);
        let shard = lower(Expr::src("w").slice(1, 0, 3072), &ck);
        assert_eq!(shard.run_count(), 2048);
        assert_eq!(shard.cost(), 1);
        assert_eq!(lower(Expr::src("e").stride(0, 0, 4, 2), &ck).cost(), 1);
        // Fusing three tensors is three copies: one per source.
        assert_eq!(
            lower(
                Expr::concat(
                    0,
                    vec![
                        Expr::src("q_proj"),
                        Expr::src("k_proj"),
                        Expr::src("v_proj")
                    ],
                ),
                &qwen3(),
            )
            .cost(),
            3
        );
        // A pad is one fill plus one copy per surviving band: the bands sit
        // in a destination that skips, and a copy engine writes forwards.
        assert_eq!(lower(pad(Expr::src("p"), &[4, 4], 1, 1, 1), &ck).cost(), 5);
    }

    #[test]
    fn the_budget_rejects_a_fragmented_expression() {
        let checkpoint = Fake::new(&[("w", &[64, 64])]);
        let expr = Expr::src("w").slice(1, 0, 32);
        let (_, checked) = infer_type(&expr, &checkpoint).unwrap();
        assert_eq!(compile(&expr, &checked, 64).unwrap().run_count(), 64);
        let err = compile(&expr, &checked, 16).unwrap_err();
        assert!(err.to_string().contains("gather lowering"));
    }

    #[test]
    fn escape_hatches_are_refused() {
        let checkpoint = Fake::new(&[("w", &[4, 32])]);
        let expr = Expr::src("w").cast(Encoding::Quant(mxfp4()));
        let (_, checked) = infer_type(&expr, &checkpoint).unwrap();
        let err = compile(&expr, &checked, 1024).unwrap_err();
        assert!(err.to_string().contains("needs a kernel"));
    }

    #[test]
    fn byte_runs_scale_by_the_encoding() {
        let out = lower(Expr::src("k_proj"), &qwen3());
        let bf16 = out.byte_runs(&Encoding::Raw(DType::BF16)).unwrap();
        assert_eq!(bf16[0].len, 1024 * 2048 * 2);
        let fp8 = out.byte_runs(&Encoding::Raw(DType::F8E4M3)).unwrap();
        assert_eq!(fp8[0].len, 1024 * 2048);
    }

    #[test]
    fn sub_byte_encodings_need_byte_aligned_runs() {
        let even = Fake::new(&[("w", &[4, 32])]);
        let out = lower(Expr::src("w"), &even);
        // 128 elements at 4 bits = 64 bytes.
        assert_eq!(out.byte_runs(&Encoding::Quant(mxfp4())).unwrap()[0].len, 64);

        let odd = Fake::new(&[("w", &[1, 5])]);
        let ragged = lower(Expr::src("w"), &odd);
        // 5 elements at 4 bits is 2.5 bytes: refused rather than rounded.
        assert!(ragged.byte_runs(&Encoding::Quant(mxfp4())).is_err());
    }

    #[test]
    fn bit_widths_match_the_encodings() {
        assert_eq!(bits_per_element(&Encoding::Raw(DType::F32)), 32);
        assert_eq!(bits_per_element(&Encoding::Raw(DType::BF16)), 16);
        assert_eq!(bits_per_element(&Encoding::Raw(DType::F8E4M3)), 8);
        assert_eq!(bits_per_element(&Encoding::Quant(mxfp4())), 4);
    }

    // ---- differential test against an independent reference ----------------
    //
    // The run walker's job is two things at once: map each output position to a
    // source, and prove how far that mapping stays contiguous. The mapping is
    // easy to check by inspection; the contiguity bounds are not. So resolve
    // every output element the slow, obvious way — one coordinate at a time,
    // straight off the `Expr`, with no arena and no span reasoning — and require
    // the compiled runs to reproduce it exactly.

    fn shape_of(expr: &Expr, checkpoint: &dyn CheckpointTypes) -> Vec<i64> {
        infer_type(expr, checkpoint)
            .expect("type check failed")
            .0
            .shape
    }

    fn flat_of(coord: &[i64], shape: &[i64]) -> i64 {
        coord
            .iter()
            .zip(shape)
            .fold(0, |flat, (at, extent)| flat * extent + at)
    }

    fn unflat_of(mut flat: i64, shape: &[i64]) -> Vec<i64> {
        let mut coord = vec![0; shape.len()];
        for axis in (0..shape.len()).rev() {
            coord[axis] = flat % shape[axis];
            flat /= shape[axis];
        }
        coord
    }

    /// Resolve one output coordinate to `(leaf name, flat element)`, or `None`
    /// for padding. Deliberately naive.
    fn oracle(expr: &Expr, coord: &[i64], ck: &dyn CheckpointTypes) -> Option<(String, i64)> {
        match expr {
            Expr::Src(name) => {
                let shape = ck.tensor_type(name).unwrap().shape;
                Some((name.clone(), flat_of(coord, &shape)))
            }
            Expr::Slice {
                src, axis, start, ..
            } => {
                let axis = usize::from(axis.0);
                let mut inner = coord.to_vec();
                inner[axis] = start + coord[axis];
                oracle(src, &inner, ck)
            }
            Expr::Stride {
                src,
                axis,
                start,
                step,
                ..
            } => {
                let axis = usize::from(axis.0);
                let mut inner = coord.to_vec();
                inner[axis] = start + coord[axis] * step;
                oracle(src, &inner, ck)
            }
            Expr::Gather { src, axis, indices } => {
                let axis = usize::from(axis.0);
                let mut inner = coord.to_vec();
                inner[axis] = indices[coord[axis] as usize];
                oracle(src, &inner, ck)
            }
            Expr::Concat { axis, parts } => {
                let axis = usize::from(axis.0);
                let mut offset = 0;
                for part in parts {
                    let extent = shape_of(part, ck)[axis];
                    if coord[axis] < offset + extent {
                        let mut inner = coord.to_vec();
                        inner[axis] = coord[axis] - offset;
                        return oracle(part, &inner, ck);
                    }
                    offset += extent;
                }
                panic!("Concat coordinate out of range");
            }
            // Same elements in the same flat order, so the coordinate is
            // simply re-indexed. The oracle's corpus is dense, and a rename
            // that also changed the element width would have no coordinate
            // correspondence to check.
            Expr::Transmute { src, .. } => {
                let mine = shape_of(expr, ck);
                let theirs = shape_of(src, ck);
                oracle(src, &unflat_of(flat_of(coord, &mine), &theirs), ck)
            }
            Expr::Fill { .. } => None,
            Expr::Out(_)
            | Expr::Repack { .. }
            | Expr::Cast { .. }
            | Expr::Scale { .. }
            | Expr::Shard { .. }
            | Expr::SrcIndexed(_)
            | Expr::Select { .. } => {
                unreachable!("the oracle covers only Src-rooted affine expressions")
            }
        }
    }

    fn assert_matches_oracle(expr: Expr, checkpoint: &dyn CheckpointTypes) {
        let out = lower(expr.clone(), checkpoint);
        let shape = shape_of(&expr, checkpoint);

        // Runs must tile the output exactly: no gaps, no overlaps, no slack.
        let mut next = 0;
        for run in &out.runs {
            assert_eq!(run.dst_elem, next, "gap or overlap at element {next}");
            assert!(run.len > 0, "empty run at element {next}");
            next += run.len;
        }
        assert_eq!(next, out.elements, "runs do not cover the output");

        // And they must be maximal: no neighbouring pair could have been one
        // run. Correct-but-fragmented would pass every other assertion here
        // while quietly wrecking the cost model.
        for pair in out.runs.windows(2) {
            assert!(
                !adjacent(&pair[0], pair[1].source, pair[1].dst_elem),
                "runs {:?} and {:?} should have merged",
                pair[0],
                pair[1]
            );
        }

        for run in &out.runs {
            for step in 0..run.len {
                let flat = run.dst_elem + step;
                let want = oracle(&expr, &unflat_of(flat, &shape), checkpoint);
                let got = match run.source {
                    RunSource::Leaf { leaf, src_elem } => {
                        Some((out.leaves[leaf].name().to_string(), src_elem + step))
                    }
                    RunSource::Zero => None,
                };
                assert_eq!(got, want, "element {flat} of {shape:?}");
            }
        }
    }

    #[test]
    fn runs_agree_with_the_oracle() {
        let ck = Fake::new(&[
            ("a", &[6, 4]),
            ("b", &[4, 4]),
            ("c", &[2, 3, 4]),
            ("d", &[24]),
        ]);
        let cases = vec![
            // Leaves and plain slices.
            Expr::src("a"),
            Expr::src("a").slice(0, 2, 3),
            Expr::src("a").slice(1, 1, 2),
            Expr::src("a").stride(0, 1, 3, 2),
            Expr::src("a").stride(1, 0, 2, 2),
            // Concatenation on every axis.
            Expr::concat(0, vec![Expr::src("a"), Expr::src("b")]),
            Expr::concat(1, vec![Expr::src("b"), Expr::src("b")]),
            // Shard-then-fuse, the spec.md §5 shape.
            Expr::concat(
                0,
                vec![Expr::src("a").slice(0, 0, 3), Expr::src("b").slice(0, 2, 2)],
            ),
            // Slice above a Concat: a composition the hand-written per-model
            // fusion pass could not express.
            Expr::concat(0, vec![Expr::src("a"), Expr::src("b")]).slice(0, 3, 5),
            Expr::concat(1, vec![Expr::src("b"), Expr::src("b")]).slice(1, 2, 4),
            // Renaming in both directions, and above a Concat.
            Expr::src("c").transmute(TensorType::raw(vec![6, 4], DType::BF16)),
            Expr::src("d").transmute(TensorType::raw(vec![2, 3, 4], DType::BF16)),
            Expr::concat(0, vec![Expr::src("b"), Expr::src("b")])
                .transmute(TensorType::raw(vec![4, 8], DType::BF16)),
            Expr::concat(
                0,
                vec![
                    Expr::src("b").transmute(TensorType::raw(vec![1, 4, 4], DType::BF16)),
                    Expr::src("b").transmute(TensorType::raw(vec![1, 4, 4], DType::BF16)),
                ],
            ),
            // Padding on each axis, and under a Concat.
            pad(Expr::src("b"), &[4, 4], 0, 1, 2),
            pad(Expr::src("b"), &[4, 4], 1, 2, 1),
            pad(Expr::src("c"), &[2, 3, 4], 1, 1, 1),
            Expr::concat(
                0,
                vec![
                    pad(Expr::src("b"), &[4, 4], 1, 0, 2),
                    pad(Expr::src("a"), &[6, 4], 1, 1, 1),
                ],
            ),
            // Three levels deep, mixing every constructor.
            Expr::concat(
                0,
                vec![
                    pad(Expr::src("a").stride(0, 0, 3, 2), &[3, 4], 1, 1, 0),
                    pad(
                        Expr::src("c")
                            .transmute(TensorType::raw(vec![6, 4], DType::BF16))
                            .slice(0, 1, 4),
                        &[4, 4],
                        1,
                        1,
                        0,
                    ),
                ],
            )
            .slice(0, 1, 5),
        ];
        for expr in cases {
            assert_matches_oracle(expr, &ck);
        }
    }

    // ---- folding runs back into loop nests ---------------------------------

    /// Expand a piece list back into `(source leaf, src element, dst element)`
    /// triples, in loop order. The fold is only allowed to change the *shape*
    /// of the description, never which pairs it names.
    fn expand(pieces: &[Piece]) -> Vec<(Option<usize>, i64, i64)> {
        let mut out = Vec::new();
        for piece in pieces {
            let total: i64 = piece.elements();
            for flat in 0..total {
                let (mut src, mut dst) = (0_i64, 0_i64);
                let mut rest = flat;
                for dim in piece.dims.iter().rev() {
                    let at = rest % dim.count;
                    rest /= dim.count;
                    src += at * dim.src_stride;
                    dst += at * dim.dst_stride;
                }
                match piece.source {
                    RunSource::Leaf { leaf, src_elem } => {
                        out.push((Some(leaf), src_elem + src, piece.dst_elem + dst));
                    }
                    RunSource::Zero => out.push((None, 0, piece.dst_elem + dst)),
                }
            }
        }
        out
    }

    fn expand_runs(lowering: &Lowering) -> Vec<(Option<usize>, i64, i64)> {
        let mut out = Vec::new();
        for run in &lowering.runs {
            for step in 0..run.len {
                match run.source {
                    RunSource::Leaf { leaf, src_elem } => {
                        out.push((Some(leaf), src_elem + step, run.dst_elem + step))
                    }
                    RunSource::Zero => out.push((None, 0, run.dst_elem + step)),
                }
            }
        }
        out
    }

    fn assert_fold_preserves(lowering: &Lowering) -> Vec<Piece> {
        let pieces = lowering.pieces();
        assert_eq!(
            pieces.iter().map(Piece::elements).sum::<i64>(),
            lowering.elements,
            "folded pieces do not cover the output"
        );
        let mut flat = expand_runs(lowering);
        let mut folded = expand(&pieces);
        folded.sort_unstable();
        flat.sort_unstable();
        assert_eq!(folded, flat, "the fold changed which elements move where");

        // And the copy-only view must name exactly the non-hole pairs.
        let mut copies = expand(&lowering.copy_pieces());
        let mut want: Vec<_> = flat
            .into_iter()
            .filter(|(leaf, ..)| leaf.is_some())
            .collect();
        copies.sort_unstable();
        want.sort_unstable();
        assert_eq!(copies, want, "copy_pieces dropped or invented an element");
        pieces
    }

    #[test]
    fn a_row_shard_folds_to_a_single_strided_copy() {
        // 2048 runs of 3072 elements, each advancing 6144 in the source and
        // 3072 in the destination. That is one `Extent`, and emitting
        // 2048 of anything instead would be the whole point missed.
        let ck = Fake::new(&[("w", &[2048, 6144])]);
        let out = lower(Expr::src("w").slice(1, 0, 3072), &ck);
        assert_eq!(out.run_count(), 2048);
        let pieces = assert_fold_preserves(&out);
        assert_eq!(
            pieces,
            vec![Piece {
                source: RunSource::Leaf {
                    leaf: 0,
                    src_elem: 0
                },
                dst_elem: 0,
                dims: vec![
                    Dim {
                        count: 2048,
                        src_stride: 6144,
                        dst_stride: 3072
                    },
                    Dim {
                        count: 3072,
                        src_stride: 1,
                        dst_stride: 1
                    },
                ],
            }]
        );
    }

    #[test]
    fn a_strided_select_folds_too() {
        let ck = Fake::new(&[("w", &[8, 4])]);
        let out = lower(Expr::src("w").stride(0, 0, 4, 2), &ck);
        assert_eq!(out.run_count(), 4);
        let pieces = assert_fold_preserves(&out);
        assert_eq!(pieces.len(), 1);
        assert_eq!(
            pieces[0].dims,
            vec![
                Dim {
                    count: 4,
                    src_stride: 8,
                    dst_stride: 4
                },
                Dim {
                    count: 4,
                    src_stride: 1,
                    dst_stride: 1
                },
            ]
        );
    }

    #[test]
    fn separate_leaves_stay_separate_pieces() {
        // Fusion reads three different tensors, so there is no progression to
        // find and the piece count equals the run count.
        let out = lower(
            Expr::concat(
                0,
                vec![
                    Expr::src("q_proj"),
                    Expr::src("k_proj"),
                    Expr::src("v_proj"),
                ],
            ),
            &qwen3(),
        );
        let pieces = assert_fold_preserves(&out);
        assert_eq!(pieces.len(), 3);
        assert!(pieces.iter().all(|piece| piece.dims.len() == 1));
    }

    #[test]
    fn a_sharded_fusion_folds_each_part_but_not_across_them() {
        // Three leaves, each contributing a strided half. Three pieces, each a
        // two-deep nest — a shape the hand-written per-model fusion pass could
        // not express.
        let ck = Fake::new(&[("q", &[512, 256]), ("k", &[256, 256]), ("v", &[256, 256])]);
        let out = lower(
            Expr::concat(
                0,
                vec![
                    Expr::src("q").slice(1, 0, 128),
                    Expr::src("k").slice(1, 0, 128),
                    Expr::src("v").slice(1, 0, 128),
                ],
            ),
            &ck,
        );
        assert_eq!(out.run_count(), 512 + 256 + 256);
        let pieces = assert_fold_preserves(&out);
        assert_eq!(pieces.len(), 3);
        assert_eq!(
            pieces[0].dims,
            vec![
                Dim {
                    count: 512,
                    src_stride: 256,
                    dst_stride: 128
                },
                Dim {
                    count: 128,
                    src_stride: 1,
                    dst_stride: 1
                },
            ]
        );
        assert_eq!(pieces[1].dst_elem, 512 * 128);
    }

    #[test]
    fn padding_costs_a_fill_and_a_copy_per_band() {
        // Zeros and data alternate row by row, so the faithful description is
        // one piece per band. Setting the holes aside leaves four data rows in
        // arithmetic progression — but they are 4 elements apart in a
        // destination whose rows are 6 wide, so widening them into one loop
        // would need a copy that skips in the destination, and nothing below
        // this layer can execute one. The honest answer is one copy per row.
        let ck = Fake::new(&[("w", &[4, 4])]);
        let out = lower(pad(Expr::src("w"), &[4, 4], 1, 1, 1), &ck);
        assert!(out.needs_zero_fill());
        let pieces = assert_fold_preserves(&out);
        assert!(pieces.len() > 1);
        for piece in &pieces {
            if piece.source == RunSource::Zero {
                assert!(piece.dims.iter().all(|dim| dim.src_stride == 0));
            }
        }
        let copies = out.copy_pieces();
        assert_eq!(copies.len(), 4);
        for (row, piece) in copies.iter().enumerate() {
            assert_eq!(
                *piece,
                Piece {
                    source: RunSource::Leaf {
                        leaf: 0,
                        src_elem: row as i64 * 4,
                    },
                    dst_elem: row as i64 * 6 + 1,
                    dims: vec![Dim {
                        count: 4,
                        src_stride: 1,
                        dst_stride: 1,
                    }],
                }
            );
        }
        assert_eq!(out.cost(), 5);
    }

    #[test]
    fn an_unpadded_output_needs_no_fill() {
        let out = lower(Expr::src("q_proj"), &qwen3());
        assert!(!out.needs_zero_fill());
        assert_eq!(out.copy_pieces(), out.pieces());
    }

    #[test]
    fn folding_agrees_with_the_runs_on_every_oracle_case() {
        let ck = Fake::new(&[("a", &[6, 4]), ("b", &[4, 4]), ("c", &[2, 3, 4])]);
        let cases = vec![
            Expr::src("a"),
            Expr::src("a").slice(1, 1, 2),
            Expr::src("a").stride(0, 1, 3, 2),
            Expr::concat(0, vec![Expr::src("a"), Expr::src("b")]),
            Expr::concat(1, vec![Expr::src("b"), Expr::src("b")]),
            Expr::src("c").transmute(TensorType::raw(vec![6, 4], DType::BF16)),
            pad(Expr::src("b"), &[4, 4], 0, 1, 2),
            pad(Expr::src("b"), &[4, 4], 1, 2, 1),
            pad(Expr::src("c"), &[2, 3, 4], 1, 1, 1),
            Expr::concat(0, vec![Expr::src("a"), Expr::src("b")]).slice(0, 3, 5),
            Expr::concat(
                0,
                vec![
                    pad(Expr::src("a").stride(0, 0, 3, 2), &[3, 4], 1, 1, 0),
                    pad(
                        Expr::src("c")
                            .transmute(TensorType::raw(vec![6, 4], DType::BF16))
                            .slice(0, 1, 4),
                        &[4, 4],
                        1,
                        1,
                        0,
                    ),
                ],
            )
            .slice(0, 1, 5),
        ];
        for expr in cases {
            assert_fold_preserves(&lower(expr, &ck));
        }
    }

    #[test]
    fn oracle_agreement_holds_for_views_over_contracts() {
        // `Out` cannot go through the oracle (it needs a contract scope), so
        // check the one property that matters: a view of a bank reads exactly
        // the bank rows it names.
        let contract = ModelContract {
            alignment: 256,
            tensors: vec![
                TensorContract::new(
                    "bank",
                    Expr::concat(
                        0,
                        vec![
                            Expr::src("q_proj"),
                            Expr::src("k_proj"),
                            Expr::src("v_proj"),
                        ],
                    ),
                    vec![4096, 2048],
                    Encoding::Raw(DType::BF16),
                ),
                TensorContract::new(
                    "q",
                    Expr::out("bank").slice(0, 0, 2048),
                    vec![2048, 2048],
                    Encoding::Raw(DType::BF16),
                ),
                TensorContract::new(
                    "k",
                    Expr::out("bank").slice(0, 2048, 1024),
                    vec![1024, 2048],
                    Encoding::Raw(DType::BF16),
                ),
                TensorContract::new(
                    "v",
                    Expr::out("bank").slice(0, 3072, 1024),
                    vec![1024, 2048],
                    Encoding::Raw(DType::BF16),
                ),
            ],
            groups: Vec::new(),
        };
        let checked = resolve(&contract, &qwen3());
        let starts: Vec<i64> = contract.tensors[1..]
            .iter()
            .map(|entry| {
                let out = compile(&entry.expr, &checked, 16).unwrap();
                assert_eq!(out.run_count(), 1);
                assert_eq!(out.leaves, vec![Leaf::Contract("bank".to_string())]);
                match out.runs[0].source {
                    RunSource::Leaf { src_elem, .. } => src_elem,
                    RunSource::Zero => panic!("view resolved to padding"),
                }
            })
            .collect();
        assert_eq!(starts, vec![0, 2048 * 2048, 3072 * 2048]);
    }
}
