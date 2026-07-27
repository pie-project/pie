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

use crate::contract::{Expr, ModelContract, TensorContract, TensorType};
use crate::ffi::types::{
    PieLoaderBytes, PieLoaderDType, PieLoaderEncodingKind, PieLoaderI64Slice, PieLoaderQuantScheme,
    PieLoaderRepackLayout, PieLoaderRowMap, PieLoaderSlice, PieLoaderU32Slice,
};
use crate::types::{
    Axis, DType, Encoding, QuantScheme, QuantSpec, RepackLayout, RepackSpec, RowMap,
};

/// `PieLoaderExprNode::src` when the node has no single operand.
pub const PIE_LOADER_NO_NODE: u32 = u32::MAX;

/// Which constructor a node is. Mirrors [`crate::contract::Expr`] exactly; a
/// variant added there without a variant here is a compile error in
/// [`read_expr`].
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderExprKind {
    Src = 0,
    Out = 1,
    Slice = 2,
    Cat = 3,
    Reshape = 4,
    Pad = 5,
    Shard = 6,
    Repack = 7,
    Quantize = 8,
    Bitcast = 9,
}

impl TryFrom<u32> for PieLoaderExprKind {
    type Error = u32;
    fn try_from(value: u32) -> Result<Self, u32> {
        Ok(match value {
            0 => Self::Src,
            1 => Self::Out,
            2 => Self::Slice,
            3 => Self::Cat,
            4 => Self::Reshape,
            5 => Self::Pad,
            6 => Self::Shard,
            7 => Self::Repack,
            8 => Self::Quantize,
            9 => Self::Bitcast,
            other => return Err(other),
        })
    }
}

/// An optional [`PieLoaderDType`], as a signed integer so that `-1` is "unset".
///
/// `QuantSpec` has three of these. Encoding them as a sentinel rather than a
/// second `bool` field per member keeps the struct a plain list of numbers,
/// which is what a designated initializer is good at.
pub type PieLoaderOptDType = i32;
pub const PIE_LOADER_NO_DTYPE: PieLoaderOptDType = -1;
/// The same convention for `QuantSpec::channel_axis`.
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
    pub scale_dtype: PieLoaderOptDType,
    pub zero_point_dtype: PieLoaderOptDType,
    pub block_shape: PieLoaderI64Slice,
}

impl Default for PieLoaderQuantSpecView {
    fn default() -> Self {
        Self {
            scheme: PieLoaderQuantScheme::None as u32,
            logical_dtype: PieLoaderDType::BF16 as u32,
            bits_per_element: 0,
            group_size: 0,
            channel_axis: PIE_LOADER_NO_AXIS,
            scale_dtype: PIE_LOADER_NO_DTYPE,
            zero_point_dtype: PIE_LOADER_NO_DTYPE,
            block_shape: PieLoaderI64Slice::default(),
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

/// [`crate::types::RepackSpec`], flattened. All eleven fields, because a repack
/// is opaque to the type checker and therefore has to state everything.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderRepackSpecView {
    /// A `PieLoaderRepackLayout` value, as `uint32_t`.
    pub layout: u32,
    /// A `PieLoaderRowMap` value, as `uint32_t`.
    pub row_map: u32,
    pub batch: u32,
    pub source_rows: u32,
    pub source_row_offset: u32,
    pub target_rows: u32,
    pub valid_rows: u32,
    pub source_stride_cols: u32,
    pub source_col_offset: u32,
    pub source_cols: u32,
    pub target_cols: u32,
}

impl Default for PieLoaderRepackSpecView {
    fn default() -> Self {
        Self {
            layout: PieLoaderRepackLayout::None as u32,
            row_map: PieLoaderRowMap::Identity as u32,
            batch: 0,
            source_rows: 0,
            source_row_offset: 0,
            target_rows: 0,
            valid_rows: 0,
            source_stride_cols: 0,
            source_col_offset: 0,
            source_cols: 0,
            target_cols: 0,
        }
    }
}

/// One node of the expression graph.
///
/// Every field is read by exactly the kinds that need it and ignored by the
/// rest; see [`read_expr`] for the mapping, which is the only place it is
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
    /// `Cat`.
    pub src: u32,
    /// `Cat`: the operands, in order. Same index rule.
    pub parts: PieLoaderU32Slice,
    /// `Slice`, `Cat`, `Pad`, `Shard`.
    pub axis: u8,
    /// `Slice`.
    pub start: i64,
    pub len: i64,
    /// `Slice`. `0` is rejected; state `1` for a contiguous run.
    pub step: i64,
    /// `Pad`.
    pub before: i64,
    pub after: i64,
    /// `Reshape`: the new shape. At most one extent may be `-1`.
    pub shape: PieLoaderI64Slice,
    /// `Repack`, `Bitcast`: the declared output type, which the checker takes on
    /// trust because it cannot see through either.
    pub out_shape: PieLoaderI64Slice,
    pub out_encoding: PieLoaderEncodingSpec,
    /// `Repack`.
    pub repack: PieLoaderRepackSpecView,
    /// `Quantize`.
    pub quant: PieLoaderQuantSpecView,
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
            before: 0,
            after: 0,
            shape: PieLoaderI64Slice::default(),
            out_shape: PieLoaderI64Slice::default(),
            out_encoding: PieLoaderEncodingSpec::default(),
            repack: PieLoaderRepackSpecView::default(),
            quant: PieLoaderQuantSpecView::default(),
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
    /// is what `LogicalShape` in `model_contracts.hpp` was working around.
    pub shape: PieLoaderI64Slice,
    /// What the driver wants the tensor to *be*. Unlike the shape this is never
    /// optional, because it is not a prediction: the loader inserts whatever
    /// cast, decode or encode is needed to reach it.
    pub encoding: PieLoaderEncodingSpec,
}

pub type PieLoaderTensorContractSlice = PieLoaderSlice<PieLoaderTensorContractView>;

/// Everything one driver rank declares.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PieLoaderModelContractView {
    pub abi_version: u32,
    /// Byte alignment every materialized buffer must satisfy.
    pub alignment: u32,
    /// The node pool, shared by every tensor. Topologically sorted: a node may
    /// only reference nodes before it.
    pub nodes: PieLoaderExprNodeSlice,
    /// The declarations, in order. `Expr::Out` may only name an earlier one.
    pub tensors: PieLoaderTensorContractSlice,
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

fn opt_dtype(value: PieLoaderOptDType, what: &str) -> Result<Option<DType>, String> {
    if value == PIE_LOADER_NO_DTYPE {
        return Ok(None);
    }
    let raw = u32::try_from(value).map_err(|_| format!("{what}: {value} is not a dtype"))?;
    dtype(raw, what).map(Some)
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
        scale_dtype: opt_dtype(spec.scale_dtype, &format!("{what}.scale_dtype"))?,
        zero_point_dtype: opt_dtype(spec.zero_point_dtype, &format!("{what}.zero_point_dtype"))?,
        block_shape: unsafe { slice_of(spec.block_shape.ptr, spec.block_shape.len) }.to_vec(),
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

fn read_repack(spec: &PieLoaderRepackSpecView, what: &str) -> Result<RepackSpec, String> {
    let layout = PieLoaderRepackLayout::try_from(spec.layout)
        .map_err(|v| format!("{what}.layout: {v} is not a PieLoaderRepackLayout"))?;
    let row_map = PieLoaderRowMap::try_from(spec.row_map)
        .map_err(|v| format!("{what}.row_map: {v} is not a PieLoaderRowMap"))?;
    Ok(RepackSpec {
        layout: RepackLayout::from(layout),
        row_map: RowMap::from(row_map),
        batch: spec.batch,
        source_rows: spec.source_rows,
        source_row_offset: spec.source_row_offset,
        target_rows: spec.target_rows,
        valid_rows: spec.valid_rows,
        source_stride_cols: spec.source_stride_cols,
        source_col_offset: spec.source_col_offset,
        source_cols: spec.source_cols,
        target_cols: spec.target_cols,
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
        PieLoaderExprKind::Slice => Expr::Slice {
            src: src()?,
            axis,
            start: node.start,
            len: node.len,
            step: node.step,
        },
        PieLoaderExprKind::Cat => {
            let parts = unsafe { slice_of(node.parts.ptr, node.parts.len) };
            if parts.is_empty() {
                return Err(format!("{what}.parts: a Cat of nothing has no type"));
            }
            Expr::Cat {
                axis,
                parts: parts
                    .iter()
                    .map(|slot| child(*slot, "parts"))
                    .collect::<Result<_, _>>()?,
            }
        }
        PieLoaderExprKind::Reshape => Expr::Reshape {
            src: src()?,
            shape: unsafe { slice_of(node.shape.ptr, node.shape.len) }.to_vec(),
        },
        PieLoaderExprKind::Pad => Expr::Pad {
            src: src()?,
            axis,
            before: node.before,
            after: node.after,
        },
        PieLoaderExprKind::Shard => Expr::Shard { src: src()?, axis },
        PieLoaderExprKind::Repack => Expr::Repack {
            src: src()?,
            spec: read_repack(&node.repack, &format!("{what}.repack"))?,
            out: read_type(&node.out_shape, &node.out_encoding, &format!("{what}.out"))?,
        },
        PieLoaderExprKind::Quantize => Expr::Quantize {
            src: src()?,
            spec: read_quant(&node.quant, &format!("{what}.quant"))?,
        },
        PieLoaderExprKind::Bitcast => Expr::Bitcast {
            src: src()?,
            out: read_type(&node.out_shape, &node.out_encoding, &format!("{what}.out"))?,
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
    if tensors.is_empty() {
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

    let mut declared = Vec::with_capacity(tensors.len());
    for (index, tensor) in tensors.iter().enumerate() {
        let what = format!("contract.tensors[{index}]");
        let name = unsafe { text(&tensor.name, &what) }?;
        let expr = built
            .get(tensor.root as usize)
            .cloned()
            .ok_or_else(|| format!("{what}.root: {} is not a node", tensor.root))?;
        let shape = unsafe { slice_of(tensor.shape.ptr, tensor.shape.len) };
        declared.push(TensorContract {
            name,
            expr,
            shape: (!shape.is_empty()).then(|| shape.to_vec()),
            encoding: read_encoding(&tensor.encoding, &what)?,
        });
    }

    Ok(ModelContract {
        abi_version: view.abi_version,
        alignment: view.alignment,
        tensors: declared,
    })
}

// ── writing PODs the loader owns ───────────────────────
//
// The loader writes two of these formats in production: a checkpoint's tensor
// table (`ffi::checkpoint`) and, in tests, a whole contract
// (`crate::contract_writer`). Flattening a *contract* is not on the load path
// and lives in the latter.

/// Somewhere to keep the `i64` runs a POD encoding points into.
///
/// A quantization spec can carry a `block_shape`, so writing one needs
/// somewhere stable to put it. Two things marshal encodings — a contract being
/// flattened and a checkpoint being opened — and they own their storage
/// differently, so the writer takes the store rather than being a method on
/// either.
pub(crate) trait ShapeStore {
    fn store_shape(&mut self, values: &[i64]) -> PieLoaderI64Slice;
}

pub(crate) fn write_quant<S: ShapeStore + ?Sized>(
    store: &mut S,
    spec: &QuantSpec,
) -> PieLoaderQuantSpecView {
    PieLoaderQuantSpecView {
        scheme: PieLoaderQuantScheme::from(spec.scheme) as u32,
        logical_dtype: PieLoaderDType::from(spec.logical_dtype) as u32,
        bits_per_element: spec.bits_per_element,
        group_size: spec.group_size,
        channel_axis: spec
            .channel_axis
            .map_or(PIE_LOADER_NO_AXIS, |axis| i32::from(axis.0)),
        scale_dtype: write_opt_dtype(spec.scale_dtype),
        zero_point_dtype: write_opt_dtype(spec.zero_point_dtype),
        block_shape: store.store_shape(&spec.block_shape),
    }
}

pub(crate) fn write_encoding<S: ShapeStore + ?Sized>(
    store: &mut S,
    encoding: &Encoding,
) -> PieLoaderEncodingSpec {
    let mut spec = PieLoaderEncodingSpec::default();
    match encoding {
        Encoding::Raw(dtype) => {
            spec.kind = PieLoaderEncodingKind::Raw as u32;
            spec.dtype = PieLoaderDType::from(*dtype) as u32;
        }
        Encoding::Quant(quant) => {
            spec.kind = PieLoaderEncodingKind::Quant as u32;
            spec.quant = write_quant(store, quant);
        }
    }
    spec
}

fn write_opt_dtype(dtype: Option<DType>) -> PieLoaderOptDType {
    dtype.map_or(PIE_LOADER_NO_DTYPE, |dtype| {
        PieLoaderDType::from(dtype) as PieLoaderOptDType
    })
}
