//! The model contract, as a POD graph the driver builds and the loader reads.
//!
//! This is the direction the plan boundary does not go. `arena.rs` marshals a
//! *result* — Rust allocates, C++ borrows until release — whereas a contract is
//! an *argument*: C++ owns it, the loader borrows it for the length of one call
//! and copies whatever it keeps. That makes it strictly the easier half, and it
//! is the half that matters, because it is what lets the driver stop asking for
//! a model by name and start declaring what it wants (architecture.md §12 row
//! 12).
//!
//! # Why a flat array
//!
//! An [`crate::contract::Expr`] is a tree with `Box` and `Vec` in it, neither of
//! which has a stable layout, so it cannot cross as itself. The standard
//! encoding is a topologically sorted node array where every edge is an index —
//! the same shape as an SSA function body, and for the same reason. Two
//! properties fall out of requiring a child index to be *strictly less* than its
//! parent's:
//!
//! * the graph is acyclic by construction, so the reader needs no visited set;
//! * a shared subexpression is one node with two parents, which is what a
//!   contract that publishes views into a bank actually is.
//!
//! # Why one flat node struct rather than a union
//!
//! A tagged union in C++ needs either a discriminated-union library or manual
//! placement, and a designated initializer cannot span one. A flat struct with
//! per-kind fields costs about 150 bytes a node — a few hundred kilobytes for
//! the largest contract, borrowed for one call — and buys the property that a
//! field the caller forgot is a zero, not garbage from another variant.

use crate::contract::{
    Expr, GroupContract, ModelContract, ScaleFactor, Scales, TensorContract, TensorType, Visibility,
};
use crate::ffi::types::{
    PieLoaderBytes, PieLoaderDType, PieLoaderEncodingKind, PieLoaderI64Slice,
    PieLoaderQuantGranularity, PieLoaderQuantScheme, PieLoaderRepackLayout, PieLoaderScaleForm,
    PieLoaderSlice, PieLoaderU32Slice, PieLoaderVisibility,
};
use crate::types::{
    Axis, DType, Encoding, QuantGranularity, QuantScheme, QuantSpec, RepackLayout, ScaleForm,
};

/// `PieLoaderExprNode::src` when the node has no single operand.
pub const PIE_LOADER_NO_NODE: u32 = u32::MAX;

/// Which constructor a node is. Mirrors [`crate::contract::Expr`] exactly; a
/// variant added there without a variant here is a compile error in
/// `read_expr`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderExprKind {
    Src = 0,
    Out = 1,
    Fill = 2,
    Slice = 3,
    Stride = 4,
    Gather = 5,
    Concat = 6,
    Transmute = 7,
    Shard = 8,
    Repack = 9,
    Cast = 10,
    Scale = 11,
    /// The two group nodes. Both stand for an index the contract does not
    /// name, so both are an error outside a
    /// [`GroupContract`](crate::contract::GroupContract) -- which the C ABI
    /// cannot declare yet, so a C++ author has no use for these today. They are
    /// numbered here because the numbering is the ABI, and appending later
    /// would be a second decision to get right.
    SrcIndexed = 12,
    Select = 13,
}

impl TryFrom<u32> for PieLoaderExprKind {
    type Error = u32;
    fn try_from(value: u32) -> Result<Self, u32> {
        Ok(match value {
            0 => Self::Src,
            1 => Self::Out,
            2 => Self::Fill,
            3 => Self::Slice,
            4 => Self::Stride,
            5 => Self::Gather,
            6 => Self::Concat,
            7 => Self::Transmute,
            8 => Self::Shard,
            9 => Self::Repack,
            10 => Self::Cast,
            11 => Self::Scale,
            12 => Self::SrcIndexed,
            13 => Self::Select,
            other => return Err(other),
        })
    }
}

/// `QuantSpec::channel_axis`, as a signed integer so that `-1` is "unset".
///
/// A sentinel rather than a second `bool` field keeps the struct a plain list
/// of numbers, which is what a designated initializer is good at.
pub const PIE_LOADER_NO_AXIS: i32 = -1;

/// [`crate::types::QuantSpec`], flattened.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderQuantSpecView {
    /// A `PieLoaderQuantScheme` value, as `uint32_t`.
    pub scheme: u32,
    /// A `PieLoaderDType` value, as `uint32_t`.
    pub logical_dtype: u32,
    /// `0` asks for the scheme's default, exactly as the Rust side does.
    pub bits_per_element: u8,
    /// `0` asks for the scheme's default.
    pub group_size: u32,
    pub channel_axis: i32,
}

impl Default for PieLoaderQuantSpecView {
    fn default() -> Self {
        Self {
            scheme: PieLoaderQuantScheme::None as u32,
            logical_dtype: PieLoaderDType::BF16 as u32,
            bits_per_element: 0,
            group_size: 0,
            channel_axis: PIE_LOADER_NO_AXIS,
        }
    }
}

/// [`crate::types::Encoding`], flattened. `kind` selects which half is read.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderEncodingSpec {
    /// A `PieLoaderEncodingKind` value, as `uint32_t`.
    pub kind: u32,
    /// A `PieLoaderDType` value, as `uint32_t`. Read when `kind == Raw`.
    pub dtype: u32,
    /// Read when `kind == Quant`.
    pub quant: PieLoaderQuantSpecView,
}

impl Default for PieLoaderEncodingSpec {
    fn default() -> Self {
        Self {
            kind: PieLoaderEncodingKind::Raw as u32,
            dtype: PieLoaderDType::BF16 as u32,
            quant: PieLoaderQuantSpecView::default(),
        }
    }
}

/// One node of the expression graph.
///
/// Every field is read by exactly the kinds that need it and ignored by the
/// rest; see `read_expr` for the mapping, which is the only place it is
/// written down.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderExprNode {
    /// A `PieLoaderExprKind` value, as `uint32_t`. Not the enum type, so that a
    /// value the loader does not recognize is a diagnosable request rather than
    /// undefined behaviour on the Rust side.
    pub kind: u32,
    /// `Src` / `Out`: the name. Borrowed for the call.
    pub name: PieLoaderBytes,
    /// The single operand, for every kind that has one. Must be strictly less
    /// than this node's own index. [`PIE_LOADER_NO_NODE`] for `Src`, `Out` and
    /// `Concat`.
    pub src: u32,
    /// `Concat`: the operands, in order. Same index rule.
    pub parts: PieLoaderU32Slice,
    /// `Slice`, `Stride`, `Concat`, `Shard`, and a per-group `Scale`.
    pub axis: u8,
    /// `Slice` and `Stride`.
    pub start: i64,
    pub len: i64,
    /// `Stride`. A contiguous run is a `Slice`, so this must be at least 2.
    pub step: i64,
    /// `Gather`: the indices to read along `axis`, in output order.
    ///
    /// Constants rather than a nested node, which is what keeps a contract
    /// comparable and hashable. Borrowed for the call, like every other slice
    /// in this struct.
    pub indices: PieLoaderI64Slice,
    /// `Fill`: the constant, as the bit pattern of an IEEE-754 binary32, for
    /// the reason `scale_factor_bits` gives.
    ///
    /// An unset field reads as `+0.0`, which is the one value a fill may
    /// carry -- so unlike `Scale`, a `Fill` is not told apart from a forgotten
    /// field by this. Its shape is: the loader rejects a rank-0 fill, and a
    /// zeroed node has no shape at all.
    pub fill_bits: u32,
    /// `Transmute`: the type to read the same bytes under. `Repack`: the declared
    /// output type, which the checker takes on trust because it cannot see
    /// through the swizzle. `Fill`: the type of the constant tensor. At most
    /// one extent may be `-1`, and never for a `Fill`.
    pub out_shape: PieLoaderI64Slice,
    pub out_encoding: PieLoaderEncodingSpec,
    /// `Repack`: a `PieLoaderRepackLayout` value, as `uint32_t`. The whole of
    /// what a repack says -- every count a kernel needs is the operand's type
    /// or `out_shape`'s, so the loader derives it.
    pub repack_layout: u32,
    /// `Scale`: the multiplier, as the bit pattern of an IEEE-754 binary32.
    ///
    /// A `float` field would make this struct's layout depend on the C++
    /// compiler agreeing about float ABI. `pie_loader::ModelContract::scale`
    /// does the conversion, so no author writes a bit pattern.
    ///
    /// Leaving it unset means zero, which reads as `+0.0`; the loader rejects
    /// that rather than scaling a tensor to nothing.
    pub scale_factor_bits: u32,
    /// `Scale`: the operand holding the factors, one per block. Same index
    /// rule as `src`.
    ///
    /// [`PIE_LOADER_NO_NODE`] for a uniform factor, and that sentinel is what
    /// tells the two forms apart. There used to be a separate `scale_group`
    /// discriminant, on the grounds that a struct zeroed by hand would then
    /// always read as the uniform case; but `src` has always carried the same
    /// exposure — node zero is a legal index — so the sentinel is the rule
    /// everywhere rather than in all places but one, and
    /// [`PieLoaderExprNode::default`] sets it.
    ///
    /// The blocking itself is not stated here. It is the ratio of this
    /// operand's shape to `src`'s, which is the only place it can be stated
    /// once.
    pub scale_by: u32,
}

impl Default for PieLoaderExprNode {
    fn default() -> Self {
        Self {
            kind: PieLoaderExprKind::Src as u32,
            name: PieLoaderBytes::default(),
            src: PIE_LOADER_NO_NODE,
            parts: PieLoaderU32Slice::default(),
            axis: 0,
            start: 0,
            len: 0,
            step: 1,
            indices: PieLoaderI64Slice::default(),
            fill_bits: 0,
            out_shape: PieLoaderI64Slice::default(),
            out_encoding: PieLoaderEncodingSpec::default(),
            repack_layout: PieLoaderRepackLayout::None as u32,
            scale_factor_bits: 0,
            scale_by: PIE_LOADER_NO_NODE,
        }
    }
}

pub type PieLoaderExprNodeSlice = PieLoaderSlice<PieLoaderExprNode>;

/// One declared tensor: a name, the expression that produces it, and what the
/// driver believes the result is.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PieLoaderTensorContractView {
    pub name: PieLoaderBytes,
    /// Index of the root node in [`PieLoaderModelContractView::nodes`].
    pub root: u32,
    /// The shape *this rank* expects, or empty for "do not check".
    ///
    /// Empty is not a weaker contract by accident; it is the honest declaration
    /// for a packed quantized weight, whose on-disk extents are a property of
    /// the quantizer that produced the checkpoint and not of the model. The
    /// alternative — the driver guessing, and the loader checking the guess —
    /// is what a `LogicalShape`-style helper exists to work around, and the
    /// reason there is no such helper here.
    pub shape: PieLoaderI64Slice,
    /// What the driver wants the tensor to *be*. Unlike the shape this is never
    /// optional, because it is not a prediction: the loader inserts whatever
    /// cast, decode or encode is needed to reach it.
    pub encoding: PieLoaderEncodingSpec,
    /// Set when this entry holds the scales for another entry.
    pub scales: PieLoaderScalesView,
    /// Whether the driver binds this tensor, or the contract just needed a
    /// name for it. Zero -- the default -- is `Public`, so an author who does
    /// not set it gets the tensor bound, which is what every existing
    /// declaration means.
    pub visibility: PieLoaderVisibility,
}

impl TryFrom<u32> for PieLoaderVisibility {
    type Error = u32;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Public),
            1 => Ok(Self::Internal),
            other => Err(other),
        }
    }
}

impl From<PieLoaderVisibility> for Visibility {
    fn from(value: PieLoaderVisibility) -> Self {
        match value {
            PieLoaderVisibility::Public => Self::Public,
            PieLoaderVisibility::Internal => Self::Internal,
        }
    }
}

/// What a scale tensor scales, said by the entry that declares the scales.
///
/// The loader used to work this pairing out by matching name suffixes —
/// `{name}_scale_inv`, then `{base}.scale`, with the group size hardcoded to 128
/// beside them. The driver had already found it properly and thrown it away:
/// `dsv4_block_scales_to_fp32` takes a `.scale` tensor, looks up `<base>.weight`
/// and publishes only if that companion is really FP8-E4M3. This is where that
/// finding is kept.
///
/// Only for scales the *checkpoint* shipped. Scales the loader creates while
/// quantizing are paired at creation, with no name involved.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PieLoaderScalesView {
    /// Name of the tensor these scales belong to. Empty means this entry is
    /// not a scale tensor and the rest of this struct is ignored.
    ///
    /// Unlike `Expr::Out` this may name a *later* declaration: it pairs two
    /// published tensors rather than feeding one into the other.
    pub of: PieLoaderBytes,
    /// A [`PieLoaderQuantGranularity`], as a `u32`.
    ///
    /// Spelled as the underlying integer for the reason
    /// [`PieLoaderTargetSpec::encode_scratch_dtype`] is: this struct is an
    /// *input*, so the value is whatever C++ wrote, and a Rust enum holding a
    /// value outside its variants is undefined behaviour the moment the
    /// containing slice is borrowed — before `read_scales` could reject it.
    /// `read_scales` converts it.
    ///
    /// [`PieLoaderTargetSpec::encode_scratch_dtype`]: crate::ffi::entry::PieLoaderTargetSpec::encode_scratch_dtype
    pub granularity: u32,
    /// Elements of `of` per scale entry, for `PerGroup`.
    pub group_size: u32,
    pub channel_axis: u32,
    /// A [`PieLoaderScaleForm`], as a `u32`. See [`Self::granularity`].
    pub form: u32,
}

pub type PieLoaderTensorContractSlice = PieLoaderSlice<PieLoaderTensorContractView>;

/// A set of interchangeable declarations: the same tensors, `arity` times.
///
/// The expressions live in the same node pool as everything else and may use
/// the two index-parametric nodes, [`PieLoaderExprKind::SrcIndexed`] and
/// [`PieLoaderExprKind::Select`], which stand for the instance and are an error
/// anywhere but here.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PieLoaderGroupContractView {
    pub name: PieLoaderBytes,
    /// How many instances. The loader compiles every one of them and requires
    /// them to agree, so this is a claim being checked and not a hint.
    pub arity: u32,
    /// The declarations one instance publishes. `Expr::Out` inside a group
    /// names an earlier declaration *of the same group*: an instance is a
    /// self-contained load, so it cannot read the resident contract's outputs.
    pub tensors: PieLoaderTensorContractSlice,
}

pub type PieLoaderGroupContractSlice = PieLoaderSlice<PieLoaderGroupContractView>;

/// Everything one driver rank declares.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PieLoaderModelContractView {
    /// Byte alignment every materialized buffer must satisfy.
    pub alignment: u32,
    /// The node pool, shared by every tensor. Topologically sorted: a node may
    /// only reference nodes before it.
    pub nodes: PieLoaderExprNodeSlice,
    /// The declarations, in order. `Expr::Out` may only name an earlier one.
    pub tensors: PieLoaderTensorContractSlice,
    /// Interchangeable declaration sets. Empty for a contract that has none,
    /// which is every contract that predates them.
    pub groups: PieLoaderGroupContractSlice,
}

unsafe fn slice_of<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if ptr.is_null() {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

unsafe fn text(value: &PieLoaderBytes, what: &str) -> Result<String, String> {
    if value.ptr.is_null() {
        if value.len == 0 {
            return Err(format!("{what}: empty name"));
        }
        return Err(format!("{what}: null pointer with non-zero length"));
    }
    let bytes = unsafe { std::slice::from_raw_parts(value.ptr, value.len) };
    std::str::from_utf8(bytes)
        .map(str::to_string)
        .map_err(|err| format!("{what}: not valid UTF-8: {err}"))
}

fn dtype(value: u32, what: &str) -> Result<DType, String> {
    PieLoaderDType::try_from(value)
        .map(DType::from)
        .map_err(|v| format!("{what}: {v} is not a PieLoaderDType"))
}

fn read_quant(spec: &PieLoaderQuantSpecView, what: &str) -> Result<QuantSpec, String> {
    let channel_axis = if spec.channel_axis == PIE_LOADER_NO_AXIS {
        None
    } else {
        Some(Axis(u8::try_from(spec.channel_axis).map_err(|_| {
            format!("{what}.channel_axis: {} is not an axis", spec.channel_axis)
        })?))
    };
    let scheme = PieLoaderQuantScheme::try_from(spec.scheme)
        .map_err(|v| format!("{what}.scheme: {v} is not a PieLoaderQuantScheme"))?;
    Ok(QuantSpec {
        scheme: QuantScheme::from(scheme),
        logical_dtype: dtype(spec.logical_dtype, &format!("{what}.logical_dtype"))?,
        bits_per_element: spec.bits_per_element,
        group_size: spec.group_size,
        channel_axis,
    })
}

fn read_encoding(spec: &PieLoaderEncodingSpec, what: &str) -> Result<Encoding, String> {
    let kind = PieLoaderEncodingKind::try_from(spec.kind)
        .map_err(|v| format!("{what}.kind: {v} is not a PieLoaderEncodingKind"))?;
    Ok(match kind {
        PieLoaderEncodingKind::Raw => Encoding::Raw(dtype(spec.dtype, &format!("{what}.dtype"))?),
        PieLoaderEncodingKind::Quant => {
            Encoding::Quant(read_quant(&spec.quant, &format!("{what}.quant"))?)
        }
    })
}

/// The layout of a `Repack` node, which must name a kernel.
///
/// Zero is not a layout here even though it is a legal wire value elsewhere: an
/// all-zero node is what a forgotten field looks like, and a repack that names
/// no kernel would otherwise be discovered missing on the device.
fn read_repack(layout: u32, what: &str) -> Result<RepackLayout, String> {
    let wire = PieLoaderRepackLayout::try_from(layout)
        .map_err(|v| format!("{what}: {v} is not a PieLoaderRepackLayout"))?;
    RepackLayout::try_from(wire).map_err(|()| {
        format!("{what}: a Repack names a kernel, and PieLoaderRepackLayout_None names none")
    })
}

fn read_type(
    shape: &PieLoaderI64Slice,
    encoding: &PieLoaderEncodingSpec,
    what: &str,
) -> Result<TensorType, String> {
    Ok(TensorType {
        shape: unsafe { slice_of(shape.ptr, shape.len) }.to_vec(),
        encoding: read_encoding(encoding, what)?,
    })
}

/// Read the scale pairing a declaration states, if it states one.
///
/// # Safety
///
/// `view.of` must be valid for the duration of the call.
unsafe fn read_scales(view: &PieLoaderScalesView, what: &str) -> Result<Option<Scales>, String> {
    if view.of.len == 0 {
        return Ok(None);
    }
    let of = unsafe { text(&view.of, &format!("{what}.scales.of")) }?;
    let granularity = QuantGranularity::from(
        PieLoaderQuantGranularity::try_from(view.granularity).map_err(|v| {
            format!("{what}.scales.granularity: {v} is not a PieLoaderQuantGranularity")
        })?,
    );
    if granularity == QuantGranularity::PerGroup && view.group_size == 0 {
        return Err(format!(
            "{what}.scales is PerGroup with group_size 0, which describes no grouping"
        ));
    }
    Ok(Some(Scales {
        of,
        granularity,
        group_size: view.group_size,
        channel_axis: view.channel_axis,
        form: ScaleForm::from(
            PieLoaderScaleForm::try_from(view.form)
                .map_err(|v| format!("{what}.scales.form: {v} is not a PieLoaderScaleForm"))?,
        ),
    }))
}

/// Materialize one node, given the nodes already materialized.
///
/// Takes `done` rather than recursing, which is what makes the index discipline
/// load-bearing: a child is looked up, never visited, so an out-of-range or
/// forward reference is a message instead of a stack overflow.
fn read_expr(node: &PieLoaderExprNode, index: usize, done: &[Expr]) -> Result<Expr, String> {
    let what = format!("contract.nodes[{index}]");
    let kind = PieLoaderExprKind::try_from(node.kind)
        .map_err(|v| format!("{what}.kind: {v} is not a PieLoaderExprKind"))?;

    // A child must already be materialized. Requiring `< index` rather than
    // merely `< len` is what makes the array a topological order, and therefore
    // acyclic, without a traversal.
    let child = |slot: u32, field: &str| -> Result<Expr, String> {
        if slot == PIE_LOADER_NO_NODE {
            return Err(format!("{what}.{field}: {kind:?} needs an operand"));
        }
        done.get(slot as usize).cloned().ok_or_else(|| {
            format!(
                "{what}.{field}: {slot} is not an earlier node (this is node \
                 {index} of {}; operands must come first)",
                done.len()
            )
        })
    };
    let src = || child(node.src, "src").map(Box::new);
    let axis = Axis(node.axis);

    Ok(match kind {
        PieLoaderExprKind::Src => Expr::Src(unsafe { text(&node.name, &what) }?),
        PieLoaderExprKind::Out => Expr::Out(unsafe { text(&node.name, &what) }?),
        PieLoaderExprKind::SrcIndexed => Expr::SrcIndexed(unsafe { text(&node.name, &what) }?),
        PieLoaderExprKind::Select => Expr::Select {
            src: src()?,
            axis,
            // `start` carries the stride: a `Select` is the `Slice` whose start
            // is `index * stride`, so the field that would have been the start
            // is the multiplier instead. Reusing the slot keeps the node POD
            // one shape rather than growing a field only two kinds can use.
            stride: node.start,
            len: node.len,
        },
        PieLoaderExprKind::Slice => Expr::Slice {
            src: src()?,
            axis,
            start: node.start,
            len: node.len,
        },
        PieLoaderExprKind::Stride => Expr::Stride {
            src: src()?,
            axis,
            start: node.start,
            len: node.len,
            step: node.step,
        },
        PieLoaderExprKind::Gather => Expr::Gather {
            src: src()?,
            axis,
            indices: unsafe { slice_of(node.indices.ptr, node.indices.len) }.to_vec(),
        },
        PieLoaderExprKind::Concat => {
            let parts = unsafe { slice_of(node.parts.ptr, node.parts.len) };
            if parts.is_empty() {
                return Err(format!("{what}.parts: a Concat of nothing has no type"));
            }
            Expr::Concat {
                axis,
                parts: parts
                    .iter()
                    .map(|slot| child(*slot, "parts"))
                    .collect::<Result<_, _>>()?,
            }
        }
        PieLoaderExprKind::Transmute => Expr::Transmute {
            src: src()?,
            to: read_type(&node.out_shape, &node.out_encoding, &format!("{what}.out"))?,
        },
        PieLoaderExprKind::Fill => Expr::Fill {
            value: node.fill_bits,
            ty: read_type(&node.out_shape, &node.out_encoding, &format!("{what}.out"))?,
        },
        PieLoaderExprKind::Shard => Expr::Shard { src: src()?, axis },
        PieLoaderExprKind::Repack => Expr::Repack {
            src: src()?,
            layout: read_repack(node.repack_layout, &format!("{what}.repack_layout"))?,
            to: read_type(&node.out_shape, &node.out_encoding, &format!("{what}.out"))?,
        },
        PieLoaderExprKind::Cast => Expr::Cast {
            src: src()?,
            to: read_encoding(&node.out_encoding, &format!("{what}.out"))?,
        },
        PieLoaderExprKind::Scale => Expr::Scale {
            src: src()?,
            factor: if node.scale_by == PIE_LOADER_NO_NODE {
                ScaleFactor::Uniform(node.scale_factor_bits)
            } else {
                ScaleFactor::PerBlock {
                    by: Box::new(child(node.scale_by, "scale_by")?),
                }
            },
        },
    })
}

/// Materialize the whole contract.
///
/// # Safety
///
/// Every pointer in `view` and everything it reaches must be valid for the
/// duration of the call.
pub unsafe fn read_contract(view: &PieLoaderModelContractView) -> Result<ModelContract, String> {
    let nodes = unsafe { slice_of(view.nodes.ptr, view.nodes.len) };
    let tensors = unsafe { slice_of(view.tensors.ptr, view.tensors.len) };
    if tensors.is_empty() && view.groups.len == 0 {
        return Err(
            "contract.tensors is empty; a contract that declares nothing \
                    would compile to a plan that loads nothing"
                .to_string(),
        );
    }
    if view.alignment == 0 {
        return Err("contract.alignment is 0; state 1 for unaligned".to_string());
    }

    let mut built: Vec<Expr> = Vec::with_capacity(nodes.len());
    for (index, node) in nodes.iter().enumerate() {
        let expr = read_expr(node, index, &built)?;
        built.push(expr);
    }

    let declared = unsafe { read_tensors(tensors, &built, "contract.tensors") }?;

    let mut groups = Vec::with_capacity(view.groups.len);
    for (index, group) in unsafe { slice_of(view.groups.ptr, view.groups.len) }
        .iter()
        .enumerate()
    {
        let what = format!("contract.groups[{index}]");
        let name = unsafe { text(&group.name, &what) }?;
        let tensors = unsafe { slice_of(group.tensors.ptr, group.tensors.len) };
        groups.push(GroupContract {
            arity: group.arity,
            tensors: unsafe { read_tensors(tensors, &built, &format!("{what}.tensors")) }?,
            name,
        });
    }

    Ok(ModelContract {
        alignment: view.alignment,
        tensors: declared,
        groups,
    })
}

/// Materialize a run of declarations against the node pool they share.
///
/// Shared by the contract's own tensors and each group's, because a group's
/// declarations are declarations -- the only thing that differs is that their
/// expressions may name the instance.
///
/// # Safety
///
/// Every pointer in `tensors` must be valid for the duration of the call.
unsafe fn read_tensors(
    tensors: &[PieLoaderTensorContractView],
    built: &[Expr],
    whose: &str,
) -> Result<Vec<TensorContract>, String> {
    let mut declared = Vec::with_capacity(tensors.len());
    for (index, tensor) in tensors.iter().enumerate() {
        let what = format!("{whose}[{index}]");
        let name = unsafe { text(&tensor.name, &what) }?;
        let expr = built
            .get(tensor.root as usize)
            .cloned()
            .ok_or_else(|| format!("{what}.root: {} is not a node", tensor.root))?;
        let shape = unsafe { slice_of(tensor.shape.ptr, tensor.shape.len) };
        let scales = unsafe { read_scales(&tensor.scales, &what) }?;
        declared.push(TensorContract {
            name,
            expr,
            shape: (!shape.is_empty()).then(|| shape.to_vec()),
            encoding: read_encoding(&tensor.encoding, &what)?,
            scales,
            visibility: PieLoaderVisibility::try_from(tensor.visibility as u32)
                .map_err(|v| format!("{what}.visibility: {v} is not a PieLoaderVisibility"))?
                .into(),
        });
    }
    Ok(declared)
}

// ── writing PODs the loader owns ───────────────────────
//
// The loader writes two of these formats in production: a checkpoint's tensor
// table (`ffi::checkpoint`) and, in tests, a whole contract
// (`crate::testkit::contract_writer`). Flattening a *contract* is not on the load path
// and lives in the latter.

pub(crate) fn write_quant(spec: &QuantSpec) -> PieLoaderQuantSpecView {
    PieLoaderQuantSpecView {
        scheme: PieLoaderQuantScheme::from(spec.scheme) as u32,
        logical_dtype: PieLoaderDType::from(spec.logical_dtype) as u32,
        bits_per_element: spec.bits_per_element,
        group_size: spec.group_size,
        channel_axis: spec
            .channel_axis
            .map_or(PIE_LOADER_NO_AXIS, |axis| i32::from(axis.0)),
    }
}

pub(crate) fn write_encoding(encoding: &Encoding) -> PieLoaderEncodingSpec {
    let mut spec = PieLoaderEncodingSpec::default();
    match encoding {
        Encoding::Raw(dtype) => {
            spec.kind = PieLoaderEncodingKind::Raw as u32;
            spec.dtype = PieLoaderDType::from(*dtype) as u32;
        }
        Encoding::Quant(quant) => {
            spec.kind = PieLoaderEncodingKind::Quant as u32;
            spec.quant = write_quant(quant);
        }
    }
    spec
}
