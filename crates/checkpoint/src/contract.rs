//! The model contract: a declaration of what the model needs, as expressions
//! over the checkpoint's byte space.
//!
//! The contract declares every tensor the engine will bind and where its bytes
//! come from; the compiler decides how to move them, and no part of it needs
//! to know the model family. The declarer is `checkpoint_dsl`, on this side
//! of the ABI, so the contract is internal IR now, not an input a caller hands
//! over. See `loader/spec.md` for the design rationale.
//!
//! This module owns the *declaration* — the grammar, the builders, and the
//! types a contract is written in. Everything that computes over one lives in
//! a child: [`infer`] says what an expression denotes, [`compile`] solves it
//! into byte rectangles, and [`rewrite`] edits a checked contract in place.

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
/// intermediates. [`Expr::Repack`], [`Expr::Cast`] and [`Expr::Scale`] are the
/// escape hatches: they need a kernel, and are deliberately marked as such.
///
/// Two axes classify the nine nodes with an operand by what they preserve; the
/// three leaves — [`Expr::Src`], [`Expr::Out`] and [`Expr::Fill`] — sit outside
/// the table, having no operand to preserve anything from.
///
/// |          | free                                       | kernel  |
/// |----------|--------------------------------------------|---------|
/// | layout   | `Slice` `Stride` `Gather` `Concat` `Shard` | `Repack`|
/// | type     | `Transmute`                                | `Cast`  |
/// | value    | —                                          | `Scale` |
///
/// A row says what a node may change; everything below its row it preserves.
/// [`Expr::Repack`] is placement priced as a kernel, and is held to its row: it
/// may pad, and it may not reinterpret an element. The empty cell is
/// principled rather than missing — a value cannot be changed for free.
///
/// Within a row, placement is a strict cost ladder that `infer` enforces in
/// both directions: a `Stride` expressible as a `Slice` is refused, a `Gather`
/// expressible as either is refused, and no node may denote exactly its
/// operand. [`Expr::Gather`] has no user today; it is kept as the ladder's top
/// rung and the placement fragment's closure.
///
/// Every node maps output axis `k` from operand axis `k`, so no node permutes
/// axes: a permutation that fixes the innermost axis is a verbose
/// [`Expr::Concat`] of `Slice`s, and a transpose that moves the innermost axis
/// is not expressible at all.
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
    /// rather than as a `Pad` node that was a [`Expr::Concat`] with a leg it
    /// could not name. Naming the leg is what lets the type checker see it.
    ///
    /// `infer` admits exactly one constant today. A fill is realized by zeroing
    /// the destination and never writing there
    /// ([`StorageInstr::Fill`](crate::plan::StorageInstr::Fill)), so the value
    /// must be one a run of zero bytes denotes. That rules out
    /// [`DType::E8m0`](crate::types::DType::E8m0), whose zero byte means
    /// `2^-127`, and it rules out every [`Encoding::Quant`] — a code word means
    /// nothing without its block scale, and the code that reads as zero is
    /// scheme-specific (`Int4B8` stores zero as 8, and its code 0 is -8).
    ///
    /// The value is carried rather than assumed so that widening the set is a
    /// change to the instruction, not to the algebra.
    Fill { value: u32, ty: TensorType },
    /// A contiguous band: `out[.., i, ..] = src[.., start + i, ..]` for `i` in
    /// `0..len`.
    ///
    /// Covers tensor-parallel sharding and fused-tensor splitting. A band is
    /// the placement that costs nothing: it leaves every inner extent alone, so
    /// it never breaks a run below one row of the operand, and on the leading
    /// axis it is a byte range. That is why the strided case is [`Expr::Stride`]
    /// and not a `step` field here: the two have different costs and different
    /// rules on a quantized axis, and a field cannot say which one an
    /// expression is without every reader inspecting it.
    Slice {
        src: Box<Expr>,
        axis: Axis,
        start: i64,
        len: i64,
    },
    /// A strided selection: `out[.., i, ..] = src[.., start + i * step, ..]`
    /// for `i` in `0..len`, with `step >= 2`.
    ///
    /// The even/odd row interleave of a GPT-OSS gate/up pair is this. Separate
    /// from [`Expr::Slice`] because a stride is the one placement that
    /// fragments: it breaks the operand's contiguity at the innermost level it
    /// touches, so selecting single elements is thousands of tiny runs where a
    /// band is one. It also may not touch a quantized axis at all — taking
    /// every other element of a block leaves a block no scale describes.
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
    /// list nobody can compute. A list that *is* a run or *is* a progression is
    /// rejected with the node to say it instead.
    ///
    /// The indices are constants rather than a nested [`Expr`], which keeps the
    /// algebra `Eq` and hashable and keeps lowering a matter of arithmetic. A
    /// permutation a kernel needs is a property of the kernel, known when the
    /// contract is written. Duplicates are allowed — a broadcast is exactly
    /// reading one source row twice.
    ///
    /// Like a stride, it may not touch a quantized axis. A permutation *of
    /// whole blocks* is a placement its groups survive, and that is spelled
    /// `Concat` of `Slice`s.
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
    /// that must first gain the axis they are stacked along. Both say *these
    /// bytes are that type*.
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
    /// At [`Partition::WHOLE`] the band is the whole axis, and the node types
    /// as its operand rather than as a slice denoting it — which is what lets
    /// a contract be *typed* without a rank, and is how the shape it declares
    /// is checked.
    ///
    /// A byte offset cannot be symbolic, so
    /// [`Resolver::specialize`](crate::contract::infer::Resolver::specialize)
    /// rewrites each `Shard` into the concrete `Slice` this rank reads before
    /// anything below the frontend sees it — a rewrite the lowering requires,
    /// not a precondition the type checker has.
    ///
    /// A node rather than a field beside the expression so that a contract
    /// stays rank-independent and so that it composes: a shard of one leg of a
    /// [`Expr::Concat`] is expressible, which a whole-expression flag cannot say.
    Shard { src: Box<Expr>, axis: Axis },
    /// A checkpoint tensor whose name carries this group instance's index:
    /// `Src(template.replace("{}", index))`.
    ///
    /// The second and last node whose meaning depends on something outside the
    /// expression, and it is the [`Expr::Shard`] of a [`GroupContract`]:
    /// [`Resolver::specialize`](crate::contract::infer::Resolver::specialize)
    /// substitutes the instance before lowering, because a tensor name cannot
    /// be symbolic any more than a byte offset can.
    ///
    /// Exactly one `{}`, substituted with the index in decimal, and no other
    /// placeholder syntax. The loader must never *parse* a checkpoint name, and
    /// a template with formatting options is a small language whose evaluator
    /// would be exactly that parser.
    ///
    /// Only for grids whose members are separate checkpoint tensors. A fused
    /// `[E, ..]` bank is a single tensor and its member is a band, which is
    /// [`Expr::Select`].
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
    /// [`Expr::SrcIndexed`] needs the per-instance check.
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
    /// kernel's business; *how many* bytes there are on each side is not opaque
    /// at all, since the operand's type says one side and `to` says the other.
    ///
    /// So a repack names a kernel and nothing else. Which rows it reads is
    /// `src`'s business, and `src` is an ordinary expression. That keeps the
    /// escape hatch from re-implementing the algebra beside it, and lets a
    /// repack be sharded like everything else.
    ///
    /// The one geometric fact left to the kernel is padding: a layout with a
    /// tile quantum takes a `to` with more rows or columns than the operand has
    /// and zero-fills the tail. `Concat[src, Fill]` would be truthful but would
    /// materialize the padded operand before the swizzle rewrites it anyway.
    ///
    /// Padding is the *only* thing it may add. An element is the same width on
    /// both sides, because this is the kernel-priced member of the placement
    /// family: it changes where a byte sits and never what a byte means. A
    /// repack that named a different element width sized its destination buffer
    /// from a lie.
    Repack {
        src: Box<Expr>,
        layout: RepackLayout,
        to: TensorType,
    },
    /// Escape hatch: the same values in a different representation.
    ///
    /// The exact dual of [`Expr::Transmute`]: a transmute preserves the *bytes*
    /// and renames them, a cast preserves the *values* and rewrites them. One
    /// is free; this one is a kernel. Rust spells the same distinction
    /// `transmute` and `as`.
    ///
    /// Covers all three directions at once, because they are one question —
    /// what representation, not what operation:
    ///
    /// * raw to raw is a numeric cast (BF16 to F32);
    /// * raw to quantized is load-time encoding, and it publishes a second
    ///   tensor holding the scales the encoder computes;
    /// * quantized to raw is decoding.
    ///
    /// Shape is preserved in every direction. Re-encoding one quantized scheme
    /// directly as another is refused — no kernel does it, and the
    /// destination's scales are not a function of the source's — so a contract
    /// declares the decoded tensor [`Visibility::Internal`] and casts *that*. A
    /// kernel node reads any expression, [`Expr::Out`] included, but may not
    /// read another kernel directly; naming the intermediate is how a contract
    /// sequences two.
    ///
    /// Stated as a node rather than left implicit in the gap between what an
    /// expression yields and what its declaration says: that gap hid a kernel,
    /// with two declarations differing only in `encoding` looking alike while
    /// one of them ran a converter and invented a scale tensor.
    Cast { src: Box<Expr>, to: Encoding },
    /// Escape hatch: elementwise multiply. `out[i] = src[i] * factor[i]`.
    ///
    /// The only operator that changes a value by arithmetic somebody asked for
    /// — a [`Expr::Cast`] moves a value only as far as its new representation
    /// forces. Everything else moves bytes or renames their type, which is why
    /// an expression's output can be checked by arithmetic on extents alone. A
    /// family reaches for this only when a factor has to be folded into a
    /// weight, and folding it at load time is what stops an engine copying the
    /// tensor to the host, scaling it there and uploading the result during
    /// bind, outside the plan entirely.
    ///
    /// With a [`ScaleFactor::PerBlock`] factor this is dequantization, and it
    /// overlaps [`Expr::Cast`] into a raw encoding on purpose: a cast decodes
    /// with the scales the *scheme* says are there, this decodes with scales an
    /// author names. It is the only variant in the algebra with two children —
    /// outbound, a quantized tensor's scales are its outputs; inbound they are
    /// inputs.
    Scale { src: Box<Expr>, factor: ScaleFactor },
    /// Escape hatch: elementwise add of one constant. `out[i] = src[i] + by`,
    /// as [`f32::to_bits`] for the reason [`ScaleFactor::Uniform`] gives.
    ///
    /// The second of the two operators that change a value, and it exists
    /// because a checkpoint format can disagree with pie about where a
    /// constant lives rather than about what it is. Gemma is the case: its
    /// rmsnorm is `x * (1 + w)`, HuggingFace publishes `w`, and llama.cpp
    /// folds the one in and publishes `w + 1`. Both files describe the same
    /// model; only one of them matches the kernel pie runs. Nothing in a
    /// placement algebra can undo that, and the alternative to saying it here
    /// is an importer that copies the tensor to the host, subtracts, and
    /// writes the result outside the plan -- which is the thing this whole
    /// module exists to stop.
    ///
    /// Deliberately NOT a second field on [`Expr::Scale`]. An affine node
    /// whose two constants are both optional has four states, two of which
    /// are the identity written two ways, and every reader would have to
    /// check both to know what it is holding. Composing the two nodes says
    /// the same thing and each one still denotes exactly one operation.
    ///
    /// The per-block form arrived with its caller. This node's doc used to
    /// close "a per-block bias is not a thing any format publishes" — and
    /// then MLX's affine schemes were read on a plane whose kernels want
    /// bf16: their dequantization is `code · scale + zero` with BOTH factors
    /// per group, `.scales` and `.biases` beside every projection. So `Bias`
    /// now ranks the way [`Expr::Scale`] does, one node at two ranks, and a
    /// per-block `Scale` followed by a per-block `Bias` is how a contract
    /// states an affine decode whose result LANDS dense.
    Bias { src: Box<Expr>, by: BiasBy },
}

/// What [`Expr::Bias`] adds — [`ScaleFactor`]'s shape, for its reason: the
/// two cases are one operation at two ranks.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BiasBy {
    /// One compile-time constant for every element, as [`f32::to_bits`].
    Uniform(u32),
    /// One addend per block, read from a declared tensor — the zero-point
    /// half of an affine decode, as the per-block [`ScaleFactor`] is the
    /// scale half. The same declaration rule holds: the addends must be a
    /// tensor that exists ([`Expr::Out`]).
    PerBlock { by: Box<Expr> },
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
    /// `[256, 4]` it is blocked 1x128; with `[256, 1]` it is one factor per
    /// row. A `group`/`axis` pair beside the operand said the same thing twice
    /// and could disagree with it, and being one number could only say it about
    /// one axis, so a two-dimensional block scale was not expressible at all
    /// even though both executors' kernels index one.
    ///
    /// The scales are a tensor like any other, so they take the same
    /// [`Expr::Shard`], [`Expr::Slice`] and [`Expr::Concat`] the weight takes,
    /// written beside it and checked against the weight by `infer`. Pairing a
    /// weight with its scales by name suffix below the contract cannot be
    /// checked at all: a partition applied to one and not the other is silent,
    /// and a pairing that fails to match simply does nothing.
    PerBlock { by: Box<Expr> },
}

/// The type of a tensor-valued expression: logical shape plus how its elements
/// are encoded.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorType {
    /// Logical shape, in elements.
    pub shape: Vec<i64>,
    /// How those elements are encoded.
    pub encoding: Encoding,
}

impl TensorType {
    /// A type with the given shape and encoding.
    pub fn new(shape: Vec<i64>, encoding: Encoding) -> Self {
        Self { shape, encoding }
    }

    /// A raw (unquantized) type of `dtype`.
    pub fn raw(shape: Vec<i64>, dtype: DType) -> Self {
        Self {
            shape,
            encoding: Encoding::Raw(dtype),
        }
    }

    /// Number of axes.
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
/// `encoding` is what the engine wants the tensor to *be*, and the loader
/// inserts whatever cast, decode or encode reaches it. `shape` is different: it
/// is a *prediction*, checked against what the expression actually yields, so
/// that an engine whose model of the checkpoint is wrong fails to compile instead
/// of silently binding a plausible-looking buffer.
///
/// **The prediction is the WHOLE tensor's**, not this rank's band. A contract
/// is authored once per model from `tp = 1` shapes and the cut enters as an
/// [`Expr::Shard`] resolved against a [`Partition`] at compile time, so a
/// declaration is written before there is a rank to write one for. It is
/// checked against the type the expression has at [`Partition::WHOLE`] —
/// `plan::build`'s `check_declared_shape` — while the plan the same pass emits
/// declares this rank's band, which is what an engine binds. The two differ by
/// exactly the shards the expression names.
///
/// A prediction may be declined. `shape: None` says "I do not claim to know",
/// which is the honest answer for a packed quantized weight whose on-disk
/// extents are a property of the quantizer that produced the file rather than of
/// the model. Forcing a claim there is what produces a `LogicalShape`-style
/// helper: something whose only job is to erase a shape the engine was made to
/// state and could not stand behind.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorContract {
    /// Declared name of the tensor.
    pub name: String,
    /// How to build it from the checkpoint. See [`Expr`].
    pub expr: Expr,
    /// Declared logical shape of the whole tensor, or `None` where the
    /// contract makes no claim. Rank-independent: see this type's docs.
    pub shape: Option<Vec<i64>>,
    /// Declared encoding of its elements.
    pub encoding: Encoding,
    /// Set when this entry holds scales for another entry. See [`Scales`].
    pub scales: Option<Scales>,
    /// Set when this entry holds the ZERO POINTS of another entry — the
    /// declared name of the weight it offsets.
    ///
    /// The affine half of the pairing [`Scales`] states the scaling half of.
    /// A scheme whose elements are `code * scale + zero` publishes three
    /// tensors, and an engine handed two of them does not read a coarser
    /// weight: it reads the right spread around the wrong centre, with no
    /// NaN to notice it by. So the third plane is named here, by the entry
    /// that IS it, for the same reason and by the same rule the scales are —
    /// a suffix match is how a plane gets paired with a weight it never
    /// belonged to.
    ///
    /// A field of its own rather than a member of [`Scales`] because the two
    /// planes are two declarations: the scales entry says what it scales, and
    /// the zero-point entry says what it offsets, each at the moment it is
    /// written. `plan::build` resolves both onto the one
    /// [`QuantAttachment`](crate::plan::QuantAttachment) of the weight they
    /// name.
    ///
    /// Only for zero points the *checkpoint* shipped, exactly as [`Scales`]
    /// is: an encode the loader performs generates its own and records the id
    /// with no name involved.
    ///
    /// Defaulted on the way in, so a contract written before this existed
    /// reads as one that declares no zero points, which is what every such
    /// contract meant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zero_points: Option<String>,
    /// Whether the engine binds this. See [`Visibility`].
    ///
    /// Defaulted on the way in, so a contract written before this existed --
    /// or one whose author simply had nothing to hide -- reads as `Public`,
    /// which is what every such declaration meant.
    #[serde(default)]
    pub visibility: Visibility,
}

/// What a scale tensor scales, said by the entry that declares the scales.
///
/// A quantized weight and its scales are two runtime tensors, and the engine's
/// kernels need to know they belong together. The pairing is stated here rather
/// than guessed from name suffixes: a suffix match is how a scale tensor gets
/// silently reinterpreted as one it never belonged to.
///
/// Only for scales the *checkpoint* shipped. When the loader quantizes a tensor
/// it creates the scale tensor itself and states the PAIRING from there with no
/// name involved at all — a [`QuantAttachment`](crate::plan::QuantAttachment)
/// names two tensor ids, which is the whole point of stating it rather than
/// matching suffixes.
///
/// **The pairing has no name; the tensor does, and it is not free either.** An
/// encoded scales plane is bound by an engine out of the same table as a
/// shipped one, by name, so `plan::build`'s `ScaleLayout` publishes it under
/// the spelling the model plane binds — `<w>.scales` for MXFP4. That is the
/// accord recorded as open against kimi's runtime-quantized expert banks, and
/// it is settled in `ScaleLayout::for_encode`, where the comment on the MXFP4
/// arm carries the ruling. A contract does not declare that entry and must
/// not: the encode instruction is its one producer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scales {
    /// Declared name of the tensor these scales belong to.
    ///
    /// Unlike [`Expr::Out`] this may name a *later* entry: it pairs two
    /// published tensors rather than feeding one into the other, and the one
    /// authoring site in the tree declares the scales first.
    pub of: String,
    /// How finely the scales divide `of`. See [`QuantGranularity`].
    pub granularity: QuantGranularity,
    /// Elements of `of` per scale entry, for [`QuantGranularity::PerGroup`].
    pub group_size: u32,
    /// The axis of `of` that per-channel scales index.
    pub channel_axis: u32,
    /// How the scale values themselves are stored. See [`ScaleForm`].
    pub form: ScaleForm,
}

impl TensorContract {
    /// A public entry named `name` and built by `expr`, declaring no scales.
    pub fn new(name: impl Into<String>, expr: Expr, shape: Vec<i64>, encoding: Encoding) -> Self {
        Self {
            name: name.into(),
            expr,
            shape: Some(shape),
            encoding,
            scales: None,
            zero_points: None,
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
            zero_points: None,
            visibility: Visibility::Public,
        }
    }

    /// Keep this declaration out of the engine's namespace. See
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

    /// Declare that this entry holds the ZERO POINTS for `of` — the affine
    /// half of the pairing [`TensorContract::scaling`] states the other half
    /// of. See [`TensorContract::zero_points`].
    pub fn offsetting(mut self, of: impl Into<String>) -> Self {
        self.zero_points = Some(of.into());
        self
    }
}

/// Everything one engine rank needs, as a name-resolved DAG.
///
/// `tensors` is in declaration order; [`Expr::Out`] may only name an earlier
/// entry, which makes the DAG acyclic by construction and lets the checker run
/// in one pass.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelContract {
    /// Byte alignment every materialized buffer must satisfy. A target
    /// property; there is no reason to repeat it per tensor.
    pub alignment: u32,
    /// Every tensor the contract declares.
    pub tensors: Vec<TensorContract>,
    /// Grids of interchangeable tensors, declared once and instantiated
    /// `arity` times. See [`GroupContract`].
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
/// engine to assume it (see [`Expr::SrcIndexed`]).
///
/// What a group is *not* is a residency decision. It says these tensors form
/// `arity` interchangeable instances and how to build one; it does not say
/// where they live or when. An engine may materialize all `arity` of them and
/// keep them resident — which is the ordinary load, one member at a time
/// instead of all at once, and so at a fraction of the peak — or it may keep a
/// few slots and page. The contract reads the same either way, because where
/// bytes live at run time is the engine's business and not the checkpoint's.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupContract {
    /// Names this grid for diagnostics and for the engine's own bookkeeping.
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
    /// Which rank of `world` this partition is.
    pub rank: u32,
    /// How many ranks share the tensor.
    pub world: u32,
}

impl Partition {
    /// The unsplit tensor: one rank owning everything.
    ///
    /// What a contract denotes before a target is chosen, and what every
    /// single-GPU load uses.
    pub const WHOLE: Self = Self { rank: 0, world: 1 };

    /// The partition `rank` of `world` denotes.
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
/// `Error::Contract` — telling the engine to fix a contract that was fine.
///
/// `what` names the thing being split, because this is the message a user gets
/// for "tp_size does not divide this model". The engine used to pre-empt it with
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
            Expr::Bias { .. } => "Bias",
            Expr::SrcIndexed(_) => "SrcIndexed",
            Expr::Select { .. } => "Select",
        }
    }

    /// Whether an [`Expr::Shard`] appears anywhere in this expression.
    ///
    /// The one question that decides whether a declaration's shape and the
    /// plan's are the same number. A shard is the only node whose meaning
    /// depends on the target, so an expression without one denotes the same
    /// tensor at every rank and an expression with one denotes a band of what
    /// the contract declares.
    ///
    /// A name read through [`Expr::Out`] is not followed: it is a published
    /// tensor, whose own shards were resolved when it was built.
    #[must_use]
    pub fn is_sharded(&self) -> bool {
        match self {
            Expr::Shard { .. } => true,
            Expr::Src(_) | Expr::Out(_) | Expr::Fill { .. } | Expr::SrcIndexed(_) => false,
            Expr::Slice { src, .. }
            | Expr::Stride { src, .. }
            | Expr::Gather { src, .. }
            | Expr::Select { src, .. }
            | Expr::Transmute { src, .. }
            | Expr::Repack { src, .. }
            | Expr::Cast { src, .. } => src.is_sharded(),
            Expr::Bias { src, by } => {
                src.is_sharded()
                    || match by {
                        BiasBy::Uniform(_) => false,
                        BiasBy::PerBlock { by } => by.is_sharded(),
                    }
            }
            Expr::Scale { src, factor } => {
                src.is_sharded()
                    || match factor {
                        ScaleFactor::Uniform(_) => false,
                        ScaleFactor::PerBlock { by } => by.is_sharded(),
                    }
            }
            Expr::Concat { parts, .. } => parts.iter().any(Expr::is_sharded),
        }
    }

    /// A checkpoint tensor named `name`. See [`Expr::Src`].
    pub fn src(name: impl Into<String>) -> Self {
        Expr::Src(name.into())
    }

    /// A published tensor named `name`. See [`Expr::Out`].
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

    /// The `[start, start + len)` run of `axis`. See [`Expr::Slice`].
    pub fn slice(self, axis: u8, start: i64, len: i64) -> Self {
        Expr::Slice {
            src: Box::new(self),
            axis: Axis(axis),
            start,
            len,
        }
    }

    /// Every `step`-th element of `axis` from `start`. See [`Expr::Stride`].
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

    /// `parts` joined along `axis`. See [`Expr::Concat`].
    pub fn concat(axis: u8, parts: Vec<Expr>) -> Self {
        Expr::Concat {
            axis: Axis(axis),
            parts,
        }
    }

    /// A constant tensor of `value` with type `ty`. See [`Expr::Fill`].
    pub fn fill(value: f32, ty: TensorType) -> Self {
        Expr::Fill {
            value: value.to_bits(),
            ty,
        }
    }

    /// The same bytes read as `to`. See [`Expr::Transmute`].
    pub fn transmute(self, to: TensorType) -> Self {
        Expr::Transmute {
            src: Box::new(self),
            to,
        }
    }

    /// The same values in `layout`, typed `to`. See [`Expr::Repack`].
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

    /// Add `by` to every element.
    pub fn bias(self, by: f32) -> Self {
        Expr::Bias {
            src: Box::new(self),
            by: BiasBy::Uniform(by.to_bits()),
        }
    }

    /// Add `by`, one addend per block, blocked by the shape ratio — the
    /// zero-point half of an affine decode.
    pub fn bias_per_block(self, by: Expr) -> Self {
        Expr::Bias {
            src: Box::new(self),
            by: BiasBy::PerBlock { by: Box::new(by) },
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

    /// This rank's share of `axis`. See [`Expr::Shard`].
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
            Expr::Repack { .. } | Expr::Cast { .. } | Expr::Scale { .. } | Expr::Bias { .. } => {
                false
            }
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

    /// Hand every node of this expression, self first and then each operand in
    /// traversal order, to `seen` — the one place the shape of the grammar is
    /// written out for a *read*.
    pub fn visit<'a>(&'a self, seen: &mut impl FnMut(&'a Expr)) {
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
            Expr::Bias { src, by } => {
                src.visit(seen);
                if let BiasBy::PerBlock { by } = by {
                    by.visit(seen);
                }
            }
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
            Expr::Bias { src, by } => Expr::Bias {
                src: boxed(src)?,
                by: match by {
                    BiasBy::PerBlock { by } => BiasBy::PerBlock { by: boxed(by)? },
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
