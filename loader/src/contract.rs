//! The model contract: a declaration of what the model needs, as expressions
//! over the checkpoint's byte space.
//!
//! See `loader/spec.md` for the design rationale. In short: the contract
//! declares every tensor the driver will bind and where its bytes come from;
//! the compiler decides how to move them, and no part of it needs to know the
//! model family. The declarer is `pie_model::contract` — an author turns a
//! driver's facts-and-policy request into one of these on this side of the
//! ABI (`plan/model-in-rust.md` §2), so the contract is internal IR now, not
//! an input a caller hands over.
//!
//! This module owns the *declaration* — the grammar, the builders, and the
//! types a contract is written in. Everything that computes over one lives in a
//! child: [`infer`] says what an expression denotes and what it denotes at one
//! rank, [`compile`] solves it into byte rectangles, and [`rewrite`] edits a
//! checked contract in place.

use serde::{Deserialize, Serialize};

use crate::error::{Error, OrOverflow};
pub use crate::types::Visibility;
use crate::types::{Axis, DType, Encoding, QuantGranularity, RepackLayout, ScaleForm};

pub mod compile;
pub mod infer;
pub mod materialize;
pub mod rewrite;

/// A tensor-valued expression.
///
/// [`Expr::Src`] through [`Expr::Shard`] form the **affine fragment**: each
/// denotes a piecewise-affine partial map from output coordinates to source
/// coordinates, and the fragment is closed under composition. Any expression
/// built from them alone compiles to a set of byte spans without materializing
/// intermediates.
///
/// [`Expr::Repack`], [`Expr::Cast`] and [`Expr::Scale`] are the escape
/// hatches. They need a kernel and are deliberately marked as such.
///
/// The organizing question is what each node *preserves*, not what it is called.
/// Two axes classify the nine that have an operand: what is preserved, and
/// whether it is free. The three leaves — [`Expr::Src`], [`Expr::Out`] and
/// [`Expr::Fill`] — sit outside the table, because with no operand to preserve
/// anything *from*, neither axis says anything about them.
///
/// |          | free                                       | kernel  |
/// |----------|--------------------------------------------|---------|
/// | layout   | `Slice` `Stride` `Gather` `Concat` `Shard` | `Repack`|
/// | type     | `Transmute`                                | `Cast`  |
/// | value    | —                                          | `Scale` |
///
/// A row says what a node is allowed to change; everything below its row it
/// preserves. So the affine fragment preserves bytes, values and type together
/// and only moves them; [`Expr::Transmute`] preserves the bytes and renames
/// them; [`Expr::Cast`] preserves the values and rewrites their representation;
/// [`Expr::Scale`] preserves the type and rewrites the values. [`Expr::Repack`]
/// is not "what is left" -- it is placement priced as a kernel, and it is held
/// to its row: it may pad, and it may not reinterpret an element.
///
/// The empty cell is principled rather than missing. A value cannot be changed
/// for free, so there is nothing to put there.
///
/// Within a row, placement is a strict cost ladder that `infer` enforces in
/// both directions: a `Stride` expressible as a `Slice` is refused, a `Gather`
/// expressible as either is refused, and no node may denote exactly its
/// operand. The cheap node must not be spellable as the expensive one, or the
/// distinction buys nothing.
///
/// [`Expr::Gather`] has no user today — every selection a driver has needed so
/// far is a band or an arithmetic progression, and the one place that once
/// wanted an index list turned out to be a `Stride`. It is kept because it is
/// the ladder's top rung and the placement fragment's closure: it is what makes
/// "these elements, in this order" expressible at all, and without it the two
/// rungs below it are two special cases rather than the cheap ends of one
/// operation. That is recorded here because nothing else would tell a reader
/// why a fully implemented node has no callers.
///
/// **What the placement fragment cannot say.** Every node maps output axis `k`
/// from operand axis `k`, so as index maps they are the per-axis-diagonal
/// subgroup: no node permutes axes. A permutation that fixes the innermost
/// axis is expressible as a `Concat` of `Slice`s, verbosely; a transpose that
/// moves the innermost axis is not expressible at all, because it is the one
/// rearrangement that is not a reordering of runs. Nothing needs it today —
/// every tile swizzle that would want it is inside a [`Expr::Repack`] — and it
/// is recorded here so that the next node added is measured against it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Expr {
    /// A tensor read from the checkpoint, by its on-disk name.
    Src(String),
    /// A tensor produced by an earlier [`TensorContract`] in the same
    /// [`ModelContract`], by its declared name.
    ///
    /// Split from [`Expr::Src`] on purpose: contracts routinely re-publish
    /// checkpoint names (a bank plus views under the original names), so a
    /// single namespace with a resolution order would make the difference
    /// between "the file's q_proj" and "my q_proj" depend on declaration order.
    Out(String),
    /// A constant tensor: every element is `value`, read as [`f32::to_bits`]
    /// for the reason [`ScaleFactor::Uniform`] gives.
    ///
    /// The third leaf, and the one with no source at all. It exists so that
    /// zero-extension is expressible as what it is — `Concat([x, Fill], axis)` —
    /// rather than as a `Pad` node that was a [`Expr::Concat`] with a leg it could
    /// not name. Naming the leg is what lets the type checker see it: the
    /// padding is now a tensor with a declared type, and `Concat` checks it
    /// against the data the way it checks every other leg.
    ///
    /// `infer` admits exactly one constant today, and says so rather than
    /// implying it. A fill is realized by zeroing the destination and never
    /// writing there ([`StorageInstr::Fill`](crate::plan::StorageInstr::Fill)),
    /// so the value must be one a run of zero bytes denotes. That rules out
    /// [`DType::E8M0`](crate::types::DType::E8M0), whose zero byte means
    /// `2^-127`, and it rules out every [`Encoding::Quant`] — a code word means
    /// nothing without its block scale, and the code that reads as zero is
    /// scheme-specific (`Int4B8` stores zero as 8, and its code 0 is -8).
    ///
    /// The value is carried rather than assumed so that widening the set is a
    /// change to [`StorageInstr::Fill`](crate::plan::StorageInstr::Fill), not to
    /// the algebra. The node and the instruction share a name because the
    /// instruction is what the node lowers to -- though not one for one, since
    /// one zeroing covers every hole in a buffer.
    Fill { value: u32, ty: TensorType },
    /// A contiguous band: `out[.., i, ..] = src[.., start + i, ..]` for `i` in
    /// `0..len`.
    ///
    /// Covers tensor-parallel sharding and fused-tensor splitting. A band is
    /// the placement that costs nothing: it leaves every inner extent alone, so
    /// it never breaks a run below one row of the operand, and on the leading
    /// axis it is a byte range.
    ///
    /// That is why the strided case is [`Expr::Stride`] and not a `step` field
    /// here. The two have different costs, different rules on a quantized axis,
    /// and only one of them may ever need a kernel; a field cannot say which
    /// of those an expression is without every reader inspecting it.
    Slice {
        src: Box<Expr>,
        axis: Axis,
        start: i64,
        len: i64,
    },
    /// A strided selection: `out[.., i, ..] = src[.., start + i * step, ..]`
    /// for `i` in `0..len`, with `step >= 2`.
    ///
    /// The even/odd row interleave of a GPT-OSS gate/up pair is this, and so is
    /// every other "take one row in `n`". Separate from [`Expr::Slice`]
    /// because a stride is the one placement that fragments: it breaks the
    /// operand's contiguity at the innermost level it touches, so selecting
    /// single elements is thousands of tiny runs where a band is one.
    ///
    /// It also may not touch a quantized axis at all. A band can, provided it
    /// lands on group boundaries; a stride cannot, because taking every other
    /// element of a block leaves a block that no scale describes.
    Stride {
        src: Box<Expr>,
        axis: Axis,
        start: i64,
        len: i64,
        step: i64,
    },
    /// An explicit selection: `out[.., i, ..] = src[.., indices[i], ..]`.
    ///
    /// The third and last placement node, and the general one: [`Expr::Slice`]
    /// is a run, [`Expr::Stride`] is an arithmetic progression, and this is a
    /// list nobody can compute. They form a cost hierarchy the author picks a
    /// rung of, and the type checker holds them to it — a list that *is* a run
    /// or *is* a progression is rejected with the node to say it instead, so
    /// the cheap operation is never spelled as the expensive one.
    ///
    /// The indices are constants rather than a nested [`Expr`], which is what
    /// keeps the whole algebra `Eq` and hashable and keeps lowering a matter of
    /// arithmetic. That is not a limitation in practice: a permutation a kernel
    /// needs is a property of the kernel, known when the contract is written.
    ///
    /// Duplicates are allowed — reading one source row twice is a well-defined
    /// thing to want, and a broadcast is exactly that.
    ///
    /// Like a stride, it may not touch a quantized axis: a permutation of a
    /// block leaves a block no scale describes. A permutation *of whole blocks*
    /// is a placement its groups survive, and that is spelled `Concat` of `Slice`s.
    Gather {
        src: Box<Expr>,
        axis: Axis,
        indices: Vec<i64>,
    },
    /// Concatenation along `axis`. Parts must agree on every other extent.
    Concat { axis: Axis, parts: Vec<Expr> },
    /// Renaming: the same bytes, a different type.
    ///
    /// The one node that changes nothing at all. A checkpoint stores sub-byte
    /// weights packed into a wider word (MLX ships 4-bit values eight to a
    /// `u32`), and a stack of per-expert slabs is a concatenation of tensors
    /// that must first gain the axis they are stacked along. Both say the same
    /// thing — *these bytes are that type* — and used to be two nodes whose
    /// rules contradicted each other on a quantized operand: one refused it
    /// outright, the other required a whole tensor and allowed anything else.
    ///
    /// Three conditions, each with its own reason:
    ///
    /// * total byte size is preserved, because nothing moves;
    /// * if the element width changes, the operand must be a whole tensor
    ///   ([`Expr::Src`] or [`Expr::Out`]), because an element offset into a
    ///   partial view stops meaning anything once the element size does;
    /// * if both sides are quantized, the shape from the blocked axis onward
    ///   must be unchanged. Only leading axes may regroup, which admits the
    ///   rank lift a stack needs and rejects a genuine reblocking — whose byte
    ///   layout is a function of the shape it was packed for, so it is a
    ///   [`Expr::Repack`], not a rename.
    ///
    /// At most one extent may be `-1`, resolved to whatever makes the byte
    /// size work.
    Transmute { src: Box<Expr>, to: TensorType },
    /// This rank's `1/world` partition of `src` along `axis`.
    ///
    /// The one node whose meaning depends on the target, which is why a
    /// [`Resolver`](crate::contract::infer::Resolver) is built for a
    /// [`Partition`]. Given one, this types like anything else: its extent is
    /// [`local_range`], and the band it denotes is checked by everything a
    /// [`Expr::Slice`] is checked by, quantization-group alignment included.
    ///
    /// Lowering is where the split stops being symbolic. A byte offset cannot
    /// be, so
    /// [`Resolver::specialize`](crate::contract::infer::Resolver::specialize)
    /// rewrites each `Shard` into the concrete `Slice` this rank reads before
    /// anything below the frontend sees it — a rewrite the lowering requires,
    /// not a precondition the type checker has.
    ///
    /// Stated as a node rather than a field beside the expression so that a
    /// contract stays rank-independent — the author writes the partition, not
    /// the arithmetic — and so that it composes: a shard of one leg of a
    /// [`Expr::Concat`] is expressible, which a whole-expression flag cannot say.
    Shard { src: Box<Expr>, axis: Axis },
    /// A checkpoint tensor whose name carries this group instance's index:
    /// `Src(template.replace("{}", index))`.
    ///
    /// The second and last node whose meaning depends on something outside the
    /// expression, and it is the [`Expr::Shard`] of a [`GroupContract`]: a
    /// group is written once for the whole grid, and
    /// [`Resolver::specialize`](crate::contract::infer::Resolver::specialize)
    /// substitutes the instance before lowering, because a tensor name cannot
    /// be symbolic any more than a byte offset can.
    ///
    /// Exactly one `{}`, substituted with the index in decimal, and no other
    /// placeholder syntax. That is deliberate: the loader must never *parse* a
    /// checkpoint name — the whole point of a contract is that naming is the
    /// author's business — and a template with formatting options is a small
    /// language whose evaluator would be exactly that parser.
    ///
    /// Only for grids whose members are separate checkpoint tensors, the way
    /// DeepSeek-V4 ships one tensor per expert. A fused `[E, ..]` bank is a
    /// single tensor and its member is a band, which is [`Expr::Select`].
    SrcIndexed(String),
    /// This group instance's band: [`Expr::Slice`] at `start = index * stride`.
    ///
    /// What [`Expr::SrcIndexed`] is for a grid of separate tensors, this is for
    /// a fused one: the expert axis of a GPT-OSS `[E, rows, cols]` bank is
    /// `stride = len = 1`, and a bank that flattened the same grid into
    /// `[E * rows, cols]` is `stride = len = rows`.
    ///
    /// Affine in the index rather than an arbitrary function of it, because
    /// that is what makes every instance the *same* extent at a different
    /// offset — which is the property a cache slot rests on. A general
    /// index expression could denote members of differing size, and then
    /// "interchangeable" would be a claim nothing checks.
    ///
    /// Its type does not mention the index at all: `len` along `axis`, whatever
    /// the instance. Uniformity is therefore structural here, and only
    /// [`Expr::SrcIndexed`] — which can name tensors that genuinely differ —
    /// needs the per-instance check.
    Select {
        src: Box<Expr>,
        axis: Axis,
        stride: i64,
        len: i64,
    },
    /// Escape hatch: a backend-specific layout swizzle. Opaque to the type
    /// checker, so it must declare its own output type.
    ///
    /// Opaque in one direction only. *What* the swizzle does to a byte is the
    /// kernel's business -- a sub-word tile interleave is a fact about a GEMM,
    /// not about tensors -- but *how many* bytes there are on each side is not
    /// opaque at all: the operand's type says one side and `to` says the other.
    ///
    /// So a repack names a kernel and nothing else. Which rows it reads is
    /// `src`'s business, and `src` is an ordinary expression: a band is
    /// [`Expr::Slice`], this rank's share is [`Expr::Shard`], an interleaved
    /// half is [`Expr::Stride`]. That is what keeps the escape hatch from
    /// re-implementing the algebra beside it, and what lets a repack be
    /// sharded like everything else instead of by a rank resolved into an
    /// integer before the contract is written.
    ///
    /// The one geometric fact left to the kernel is padding: a layout with a
    /// tile quantum takes a `to` with more rows or columns than the operand
    /// has and zero-fills the tail. Stating it as `Concat[src, Fill]` would be
    /// truthful but would materialize the padded operand before the swizzle
    /// that is about to rewrite it anyway.
    ///
    /// Padding is the *only* thing it may add. An element is the same width on
    /// both sides, because this is the kernel-priced member of the placement
    /// family: it changes where a byte sits and never what a byte means. A
    /// repack that named a different element width sized its destination
    /// buffer from a lie, which is [`Expr::Transmute`]'s invariant reached
    /// from the other side.
    Repack {
        src: Box<Expr>,
        layout: RepackLayout,
        to: TensorType,
    },
    /// Escape hatch: the same values in a different representation.
    ///
    /// The exact dual of [`Expr::Transmute`], and the pair is worth reading
    /// together: a transmute preserves the *bytes* and renames them, a cast
    /// preserves the *values* and rewrites them. One is free; this one is a
    /// kernel. Rust spells the same distinction `transmute` and `as`.
    ///
    /// Covers all three directions at once, because they are one question --
    /// what representation, not what operation:
    ///
    /// * raw to raw is a numeric cast (BF16 to F32);
    /// * raw to quantized is load-time encoding, and it publishes a second
    ///   tensor holding the scales the encoder computes;
    /// * quantized to raw is decoding.
    ///
    /// Shape is preserved in every direction: a cast has an opinion about
    /// element representation and none about placement. Re-encoding one
    /// quantized scheme directly as another is refused -- no kernel does it,
    /// and the destination's scales are not a function of the source's -- so
    /// the two-step is what a contract says instead: declare the decoded
    /// tensor [`Visibility::Internal`] and cast *that*. A kernel node reads any
    /// expression, [`Expr::Out`] included, which is what makes step two
    /// spellable; what it may not read is another kernel directly, and naming
    /// the intermediate is how a contract sequences two. Which node decodes
    /// depends on the scheme -- a cast to the logical dtype where a backend
    /// implements one, a per-group [`Expr::Scale`] for a block-scaled scheme.
    ///
    /// Stated as a node rather than left implicit in the gap between what an
    /// expression yields and what its declaration says, because that gap hid a
    /// kernel: two declarations that differ only in `encoding` looked alike at
    /// the author's fingertips while one of them ran a converter and invented a
    /// scale tensor below the type checker.
    Cast { src: Box<Expr>, to: Encoding },
    /// Escape hatch: elementwise multiply. `out[i] = src[i] * factor[i]`.
    ///
    /// One of two operators that change a *value*, and the only one that changes
    /// it by arithmetic somebody asked for: a [`Expr::Cast`] moves a value only
    /// as far as its new representation forces. Everything else moves bytes or
    /// renames their type, which is why an expression's output can be checked by
    /// arithmetic on extents alone. That property is worth the price of
    /// admission being explicit: a family reaches for this only when a factor
    /// has to be folded into a weight, and folding it at load time is what stops
    /// the alternative — a driver copying the tensor to the host, scaling it
    /// there and uploading the result during bind, outside the plan entirely.
    ///
    /// With a [`ScaleFactor::PerBlock`] factor this is dequantization, and it
    /// overlaps [`Expr::Cast`] into a raw encoding on purpose: a cast decodes
    /// with the scales the *scheme* says are there, while this decodes with
    /// scales an author names. That is the whole difference, and it is why this
    /// is the only variant in the algebra with two children — outbound, a
    /// quantized tensor's scales are its outputs; inbound they are inputs.
    Scale { src: Box<Expr>, factor: ScaleFactor },
}

/// What [`Expr::Scale`] multiplies by.
///
/// The two cases are the same operation at different ranks, which is why they
/// are one node: a uniform factor is the rank-0 case of a factor read from a
/// tensor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleFactor {
    /// One compile-time constant for every element, as [`f32::to_bits`].
    ///
    /// Bits rather than `f32` so the IR stays integer-only. [`Expr`] and
    /// [`LoadPlan`](crate::plan::LoadPlan) derive `Eq` and are hashed into the
    /// cache key, and `f32` can do neither honestly: `NaN != NaN` makes
    /// equality partial, and `0.0 == -0.0` with different bits makes any hash
    /// that agrees with equality lossy. Two plans are the same plan when they
    /// name the same constant, so the constant is compared as what it is
    /// written as.
    ///
    /// `infer` rejects a factor that is not finite, and rejects zero — zero is
    /// what an all-zero FFI node carries, so accepting it would turn a
    /// forgotten field into a tensor of zeros that loads and runs.
    Uniform(u32),
    /// One factor per block, read from a companion expression whose shape says
    /// how big a block is: `out[i0, .., ik] = src[i0, .., ik] * by[i0 / g0, .., ik / gk]`
    /// where `gj = src.shape[j] / by.shape[j]`.
    ///
    /// **The grouping is the shape ratio, and nothing else states it.** A
    /// `[256, 512]` weight with `[2, 4]` factors is blocked 128x128; with
    /// `[256, 4]` it is blocked 1x128, which is the one-dimensional case; with
    /// `[256, 1]` it is one factor per row. This used to carry a `group` and an
    /// `axis` beside the operand, which said the same thing a second time and
    /// therefore could disagree with it — and being one number, could only ever
    /// say it about one axis, so a two-dimensional block scale was not
    /// expressible at all even though both executors' kernels index one.
    ///
    /// This is how a block-scaled checkpoint is dequantized, and stating it as
    /// an expression is the point. The scales are a tensor like any other, so
    /// they take the same [`Expr::Shard`], [`Expr::Slice`] and [`Expr::Concat`]
    /// the weight takes, written beside it and checked against the weight by
    /// `infer`. The alternative — pairing a weight with its scales by appending
    /// a suffix to its name, somewhere below the contract — cannot be checked
    /// at all: a partition applied to one and not the other is silent, and a
    /// pairing that fails to match simply does nothing.
    PerBlock { by: Box<Expr> },
}

/// The type of a tensor-valued expression: logical shape plus how its elements
/// are encoded.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorType {
    pub shape: Vec<i64>,
    pub encoding: Encoding,
}

impl TensorType {
    pub fn new(shape: Vec<i64>, encoding: Encoding) -> Self {
        Self { shape, encoding }
    }

    pub fn raw(shape: Vec<i64>, dtype: DType) -> Self {
        Self {
            shape,
            encoding: Encoding::Raw(dtype),
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Number of logical elements, or an error on overflow.
    pub fn element_count(&self) -> Result<i64, Error> {
        self.shape.iter().try_fold(1_i64, |acc, dim| {
            acc.checked_mul(*dim)
                .or_overflow(format!("shape {:?} overflows i64", self.shape))
        })
    }

    /// Size in bytes. Errors when a sub-byte encoding does not fill whole bytes.
    pub fn byte_size(&self) -> Result<u64, Error> {
        crate::types::encoding_nbytes(&self.shape, &self.encoding).ok_or_else(|| {
            Error::Contract(format!(
                "shape {:?} of {:?} has no whole-byte size",
                self.shape, self.encoding
            ))
        })
    }
}

/// One declared tensor.
///
/// `encoding` is what the driver wants the tensor to *be*, and the loader
/// inserts whatever cast, decode or encode reaches it. `shape` is different: it
/// is a *prediction*, checked against what the expression actually yields, so
/// that a driver whose model of the checkpoint is wrong fails to compile instead
/// of silently binding a plausible-looking buffer.
///
/// A prediction may be declined. `shape: None` says "I do not claim to know",
/// which is the honest answer for a packed quantized weight whose on-disk
/// extents are a property of the quantizer that produced the file rather than of
/// the model. Forcing a claim there is what produces a `LogicalShape`-style
/// helper: something whose only job is to erase a shape the driver was made to
/// state and could not stand behind.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorContract {
    pub name: String,
    pub expr: Expr,
    pub shape: Option<Vec<i64>>,
    pub encoding: Encoding,
    /// Set when this entry holds scales for another entry. See [`Scales`].
    pub scales: Option<Scales>,
    /// Whether the driver binds this. See [`Visibility`].
    ///
    /// Defaulted on the way in, so a contract written before this existed --
    /// or one whose author simply had nothing to hide -- reads as `Public`,
    /// which is what every such declaration meant.
    #[serde(default)]
    pub visibility: Visibility,
}

/// What a scale tensor scales, said by the entry that declares the scales.
///
/// A quantized weight and its scales are two runtime tensors, and the driver's
/// kernels need to know they belong together. The loader used to work that out
/// by matching name suffixes — `{name}_scale_inv`, then `{base}.scale` — with
/// the group size hardcoded to 128 beside them.
///
/// That guess had an author. `deepseek_v4_contract.hpp::dsv4_block_scales_to_fp32`
/// finds the pairing properly: it takes a `.scale` tensor, looks up
/// `<base>.weight`, and publishes the scale only if that companion is really
/// FP8-E4M3 — "guessing is how a scale tensor gets silently reinterpreted", as
/// its own comment puts it. Then it dropped the pair on the floor and the loader
/// re-derived it from strings. This field is the driver keeping what it found.
///
/// Only for scales the *checkpoint* shipped. When the loader quantizes a tensor
/// it creates the scale tensor itself and states the pairing from there, with no
/// name involved at all.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scales {
    /// Declared name of the tensor these scales belong to.
    ///
    /// Unlike [`Expr::Out`] this may name a *later* entry: it pairs two
    /// published tensors rather than feeding one into the other, and the one
    /// authoring site in the tree declares the scales first.
    pub of: String,
    pub granularity: QuantGranularity,
    /// Elements of `of` per scale entry, for [`QuantGranularity::PerGroup`].
    pub group_size: u32,
    pub channel_axis: u32,
    pub form: ScaleForm,
}

impl TensorContract {
    pub fn new(name: impl Into<String>, expr: Expr, shape: Vec<i64>, encoding: Encoding) -> Self {
        Self {
            name: name.into(),
            expr,
            shape: Some(shape),
            encoding,
            scales: None,
            visibility: Visibility::Public,
        }
    }

    /// A declaration that states the encoding it wants and declines to predict
    /// the shape.
    pub fn inferred(name: impl Into<String>, expr: Expr, encoding: Encoding) -> Self {
        Self {
            name: name.into(),
            expr,
            shape: None,
            encoding,
            scales: None,
            visibility: Visibility::Public,
        }
    }

    /// Keep this declaration out of the driver's namespace. See
    /// [`Visibility::Internal`].
    pub fn internal(mut self) -> Self {
        self.visibility = Visibility::Internal;
        self
    }

    /// Declare that this entry holds the scales for `of`.
    pub fn scaling(mut self, scales: Scales) -> Self {
        self.scales = Some(scales);
        self
    }
}

/// Everything one driver rank needs, as a name-resolved DAG.
///
/// `tensors` is in declaration order; [`Expr::Out`] may only name an earlier
/// entry, which makes the DAG acyclic by construction and lets the checker run
/// in one pass.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelContract {
    /// Byte alignment every materialized buffer must satisfy. A target
    /// property; there is no reason to repeat it per tensor.
    pub alignment: u32,
    pub tensors: Vec<TensorContract>,
    /// Grids of interchangeable tensors, declared once and instantiated
    /// `arity` times. See [`GroupContract`].
    ///
    /// Defaulted on the way in, so every contract written before groups
    /// existed reads as having none — which is what each of them meant.
    #[serde(default)]
    pub groups: Vec<GroupContract>,
}

/// A grid of interchangeable tensor sets, written once.
///
/// A mixture-of-experts checkpoint holds thousands of expert weights that
/// differ only in *which* expert they are: same shape, same encoding, same
/// expression, one index apart. Declared one by one they are thousands of
/// [`TensorContract`]s; declared as a group they are one, plus an `arity`.
///
/// The compression is the smaller half of the point. The larger half is that a
/// group *states* what a list cannot: that its members are the same size and
/// therefore interchangeable. That is exactly the claim a bounded cache of
/// slots rests on — page one member out, page another in, the slot fits either
/// way — and stating it lets the type checker prove it instead of leaving a
/// driver to assume it (see [`Expr::SrcIndexed`]).
///
/// What a group is *not* is a residency decision. It says these tensors form
/// `arity` interchangeable instances and how to build one; it does not say
/// where they live or when. A driver may materialize all `arity` of them and
/// keep them resident — which is the ordinary load, one member at a time
/// instead of all at once, and so at a fraction of the peak — or it may keep a
/// few slots and page. The contract reads the same either way, because where
/// bytes live at run time is the driver's business and not the checkpoint's.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupContract {
    /// Names this grid for diagnostics and for the driver's own bookkeeping.
    pub name: String,
    /// How many instances the grid has. The index runs `0..arity`.
    pub arity: u32,
    /// The tensors one instance is made of, as expressions that may mention
    /// the index through [`Expr::SrcIndexed`] and [`Expr::Select`].
    pub tensors: Vec<TensorContract>,
}

/// Which slice of a tensor-parallel world an expression is being read for.
///
/// The whole of a target's rank-dependence, as one value. [`Expr::Shard`] is
/// the only node that consults it, and it is carried rather than threaded so
/// that the type checker and the specializer cannot be given different answers:
/// they share one [`Resolver`](crate::contract::infer::Resolver), so they share
/// this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Partition {
    pub rank: u32,
    pub world: u32,
}

impl Partition {
    /// The unsplit tensor: one rank owning everything.
    ///
    /// What a contract denotes before a target is chosen, and what every
    /// single-GPU load uses.
    pub const WHOLE: Self = Self { rank: 0, world: 1 };

    pub fn new(rank: u32, world: u32) -> Self {
        Self { rank, world }
    }
}

impl Default for Partition {
    fn default() -> Self {
        Self::WHOLE
    }
}

/// The `[start, len)` of a `full`-long axis that `rank` of `world` owns.
///
/// The arithmetic [`Expr::Shard`] denotes, in one place, because three callers
/// need it: the type checker states a shard's extent, the specializer rewrites
/// it into the concrete [`Expr::Slice`] one rank reads, and the rewriter decides
/// whether a row-sharded read can be coalesced. Split between them, the answers
/// were free to disagree — and did, on which [`Error`] an indivisible axis
/// produces.
///
/// Both ways a shard can fail are refused here, and both as
/// [`Error::Shard`]: an axis `world` does not
/// divide, and a `rank` outside the group. The second used to fall through and
/// produce an over-wide band, which the slice bounds check then reported as an
/// `Error::Contract` — telling the driver to fix a contract that was fine.
///
/// `what` names the thing being split, because this is the message a user gets
/// for "tp_size does not divide this model". The driver used to pre-empt it with
/// its own per-family table of divisibility rules read off `config.json` — the
/// same fact checked twice, and only for the families someone had listed.
pub fn local_range(full: i64, world: u32, rank: u32, what: &str) -> Result<(i64, i64), Error> {
    let world = world.max(1);
    if rank >= world {
        return Err(Error::Shard(format!(
            "tp_rank {rank} is out of range for tp_size {world}; ranks are 0..{world}"
        )));
    }
    let world = i64::from(world);
    if full % world != 0 {
        return Err(Error::Shard(format!(
            "{what} is {full}, which tp_size {world} does not divide; use a \
             tp_size that divides it or run single-GPU"
        )));
    }
    let local = full / world;
    Ok((i64::from(rank) * local, local))
}

/// Resolve a [`Expr::Transmute`] shape against an operand of `total` elements *at
/// the requested type*, replacing a single `-1` with the extent that fits.
///
/// The only resolver. The type checker needs it to state the output shape and
/// the byte-run compiler needs it to place the operand, and a second spelling
/// of "what does `-1` mean here" is a plan that disagrees with the type it was
/// checked against — silently, because both answers are plausible integers.
///
/// Counted in elements of the *output* type rather than the operand's, so that
/// the wildcard means the same thing when the element width changes: a `[-1]`
/// over 64 bytes is 32 elements as `BF16` and 128 as a 4-bit code.
pub fn resolve_extents(requested: &[i64], total: i64) -> Result<Vec<i64>, Error> {
    if requested.is_empty() {
        return Err(Error::Contract(
            "Transmute needs at least one extent".to_string(),
        ));
    }
    let mut wildcard = None;
    let mut known = 1_i64;
    for (index, extent) in requested.iter().enumerate() {
        match *extent {
            -1 if wildcard.is_some() => {
                return Err(Error::Contract(
                    "Transmute allows at most one -1 extent".to_string(),
                ));
            }
            -1 => wildcard = Some(index),
            extent if extent < 1 => {
                return Err(Error::Contract(format!(
                    "Transmute extent {extent} must be >= 1 or -1"
                )));
            }
            extent => {
                known = known
                    .checked_mul(extent)
                    .or_overflow("Transmute extent overflows i64")?;
            }
        }
    }
    let mut shape = requested.to_vec();
    match wildcard {
        Some(index) if known > 0 && total % known == 0 => shape[index] = total / known,
        Some(_) => {
            return Err(Error::Contract(format!(
                "Transmute to {requested:?} does not divide {total} elements evenly"
            )));
        }
        None if known == total => {}
        None => {
            return Err(Error::Contract(format!(
                "Transmute to {requested:?} is {known} elements, not the {total} the operand's bytes hold"
            )));
        }
    }
    Ok(shape)
}

impl Expr {
    /// The node's name in the algebra, for diagnostics.
    ///
    /// This is the constructor's own name, not a description: it is what the
    /// table in this module's documentation calls the node, so an error can
    /// say which cell a rejected expression came from.
    pub fn node_name(&self) -> &'static str {
        match self {
            Expr::Src(_) => "Src",
            Expr::Out(_) => "Out",
            Expr::Fill { .. } => "Fill",
            Expr::Slice { .. } => "Slice",
            Expr::Stride { .. } => "Stride",
            Expr::Gather { .. } => "Gather",
            Expr::Concat { .. } => "Concat",
            Expr::Transmute { .. } => "Transmute",
            Expr::Shard { .. } => "Shard",
            Expr::Repack { .. } => "Repack",
            Expr::Cast { .. } => "Cast",
            Expr::Scale { .. } => "Scale",
            Expr::SrcIndexed(_) => "SrcIndexed",
            Expr::Select { .. } => "Select",
        }
    }

    pub fn src(name: impl Into<String>) -> Self {
        Expr::Src(name.into())
    }

    pub fn out(name: impl Into<String>) -> Self {
        Expr::Out(name.into())
    }

    /// A checkpoint tensor named by substituting the group index into
    /// `template`. See [`Expr::SrcIndexed`].
    pub fn src_indexed(template: impl Into<String>) -> Self {
        Expr::SrcIndexed(template.into())
    }

    /// This instance's band of a fused grid. See [`Expr::Select`].
    pub fn select(self, axis: u8, stride: i64, len: i64) -> Self {
        Expr::Select {
            src: Box::new(self),
            axis: Axis(axis),
            stride,
            len,
        }
    }

    pub fn slice(self, axis: u8, start: i64, len: i64) -> Self {
        Expr::Slice {
            src: Box::new(self),
            axis: Axis(axis),
            start,
            len,
        }
    }

    pub fn stride(self, axis: u8, start: i64, len: i64, step: i64) -> Self {
        Expr::Stride {
            src: Box::new(self),
            axis: Axis(axis),
            start,
            len,
            step,
        }
    }

    /// Select `indices` along `axis`, in the order given. See [`Expr::Gather`].
    pub fn gather(self, axis: u8, indices: Vec<i64>) -> Self {
        Expr::Gather {
            src: Box::new(self),
            axis: Axis(axis),
            indices,
        }
    }

    pub fn concat(axis: u8, parts: Vec<Expr>) -> Self {
        Expr::Concat {
            axis: Axis(axis),
            parts,
        }
    }

    pub fn fill(value: f32, ty: TensorType) -> Self {
        Expr::Fill {
            value: value.to_bits(),
            ty,
        }
    }

    pub fn transmute(self, to: TensorType) -> Self {
        Expr::Transmute {
            src: Box::new(self),
            to,
        }
    }

    pub fn repack(self, layout: RepackLayout, to: TensorType) -> Self {
        Expr::Repack {
            src: Box::new(self),
            layout,
            to,
        }
    }

    /// The same values in `to`. See [`Expr::Cast`].
    pub fn cast(self, to: Encoding) -> Self {
        Expr::Cast {
            src: Box::new(self),
            to,
        }
    }

    /// Multiply every element by `factor`.
    pub fn scale(self, factor: f32) -> Self {
        Expr::Scale {
            src: Box::new(self),
            factor: ScaleFactor::Uniform(factor.to_bits()),
        }
    }

    /// Multiply by `by`, one factor per block, blocked by the shape ratio.
    ///
    /// Over a quantized `self` this is dequantization, and the result is the
    /// scheme's logical dtype.
    pub fn scale_per_block(self, by: Expr) -> Self {
        Expr::Scale {
            src: Box::new(self),
            factor: ScaleFactor::PerBlock { by: Box::new(by) },
        }
    }

    pub fn shard(self, axis: u8) -> Self {
        Expr::Shard {
            src: Box::new(self),
            axis: Axis(axis),
        }
    }

    /// Whether this expression lies entirely in the affine fragment, and so
    /// compiles to byte spans with no kernel and no intermediate buffer.
    pub fn is_affine(&self) -> bool {
        match self {
            Expr::Src(_) | Expr::Out(_) | Expr::Fill { .. } | Expr::SrcIndexed(_) => true,
            Expr::Slice { src, .. }
            | Expr::Stride { src, .. }
            | Expr::Gather { src, .. }
            | Expr::Transmute { src, .. } => src.is_affine(),
            Expr::Concat { parts, .. } => parts.iter().all(Expr::is_affine),
            Expr::Shard { src, .. } | Expr::Select { src, .. } => src.is_affine(),
            Expr::Repack { .. } | Expr::Cast { .. } | Expr::Scale { .. } => false,
        }
    }

    /// Names of the checkpoint tensors this expression reads, in traversal
    /// order. Duplicates are preserved; the caller decides whether it cares.
    pub fn sources(&self) -> Vec<&str> {
        let mut found = Vec::new();
        self.visit_sources(&mut found);
        found
    }

    fn visit_sources<'a>(&'a self, found: &mut Vec<&'a str>) {
        self.visit(&mut |expr| {
            if let Expr::Src(name) = expr {
                found.push(name.as_str());
            }
        });
    }

    /// Names of the *earlier contracts* this expression reads, in traversal
    /// order. These are the edges of the contract DAG.
    pub fn outputs(&self) -> Vec<&str> {
        let mut found = Vec::new();
        self.visit(&mut |expr| {
            if let Expr::Out(name) = expr {
                found.push(name.as_str());
            }
        });
        found
    }

    fn visit<'a>(&'a self, seen: &mut impl FnMut(&'a Expr)) {
        seen(self);
        match self {
            Expr::Src(_) | Expr::Out(_) | Expr::Fill { .. } | Expr::SrcIndexed(_) => {}
            Expr::Slice { src, .. }
            | Expr::Stride { src, .. }
            | Expr::Gather { src, .. }
            | Expr::Transmute { src, .. }
            | Expr::Repack { src, .. }
            | Expr::Shard { src, .. }
            | Expr::Select { src, .. }
            | Expr::Cast { src, .. } => src.visit(seen),
            Expr::Scale { src, factor } => {
                src.visit(seen);
                if let ScaleFactor::PerBlock { by } = factor {
                    by.visit(seen);
                }
            }
            Expr::Concat { parts, .. } => {
                for part in parts {
                    part.visit(seen);
                }
            }
        }
    }

    /// Rebuild this node with each immediate operand replaced by `f` of it.
    ///
    /// The one place the shape of the grammar is written out for a *rewrite*.
    /// Every pass over an expression otherwise repeats the same ten arms to
    /// say "recurse and put it back", which is eight arms of ceremony around
    /// the one that does something — and nine chances to forget an operand when
    /// a variant is added. A rewrite is then `f` plus the arms it actually
    /// cares about, with `other => other.map_children(f)` for the rest.
    ///
    /// Only the *immediate* operands: `f` decides whether to recurse, which is
    /// what lets a caller stop at a node ([`Expr::Shard`] specialization needs
    /// the operand's type before it can rewrite itself, so it recurses first
    /// and then does its own work).
    pub fn map_children(
        self,
        mut f: impl FnMut(Expr) -> Result<Expr, Error>,
    ) -> Result<Expr, Error> {
        let mut boxed = |src: Box<Expr>| -> Result<Box<Expr>, Error> { Ok(Box::new(f(*src)?)) };
        Ok(match self {
            Expr::Src(_) | Expr::Out(_) | Expr::Fill { .. } | Expr::SrcIndexed(_) => self,
            Expr::Slice {
                src,
                axis,
                start,
                len,
            } => Expr::Slice {
                src: boxed(src)?,
                axis,
                start,
                len,
            },
            Expr::Stride {
                src,
                axis,
                start,
                len,
                step,
            } => Expr::Stride {
                src: boxed(src)?,
                axis,
                start,
                len,
                step,
            },
            Expr::Gather { src, axis, indices } => Expr::Gather {
                src: boxed(src)?,
                axis,
                indices,
            },
            Expr::Select {
                src,
                axis,
                stride,
                len,
            } => Expr::Select {
                src: boxed(src)?,
                axis,
                stride,
                len,
            },
            Expr::Transmute { src, to } => Expr::Transmute {
                src: boxed(src)?,
                to,
            },
            Expr::Repack { src, layout, to } => Expr::Repack {
                src: boxed(src)?,
                layout,
                to,
            },
            Expr::Cast { src, to } => Expr::Cast {
                src: boxed(src)?,
                to,
            },
            Expr::Scale { src, factor } => Expr::Scale {
                src: boxed(src)?,
                factor: match factor {
                    ScaleFactor::PerBlock { by } => ScaleFactor::PerBlock { by: boxed(by)? },
                    uniform => uniform,
                },
            },
            Expr::Shard { src, axis } => Expr::Shard {
                src: boxed(src)?,
                axis,
            },
            Expr::Concat { axis, parts } => Expr::Concat {
                axis,
                parts: parts.into_iter().map(f).collect::<Result<_, _>>()?,
            },
        })
    }
}
