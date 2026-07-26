use serde::{Deserialize, Serialize};

use crate::types::{DType, ExprId, QuantScheme, RepackSpec, TensorDecl, TensorId};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayoutPlan {
    pub exprs: Vec<LayoutExpr>,
    pub outputs: Vec<ExprId>,
}

impl LayoutPlan {
    pub fn new() -> Self {
        Self {
            exprs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn push(&mut self, expr: LayoutExpr) -> ExprId {
        let id = ExprId(self.exprs.len() as u32);
        self.exprs.push(expr);
        id
    }

    pub fn expr(&self, id: ExprId) -> Option<&LayoutExpr> {
        self.exprs.get(id.0 as usize)
    }

    pub fn expr_mut(&mut self, id: ExprId) -> Option<&mut LayoutExpr> {
        self.exprs.get_mut(id.0 as usize)
    }

    pub fn decl(&self, id: ExprId) -> Option<&TensorDecl> {
        Some(self.expr(id)?.decl())
    }
}

impl Default for LayoutPlan {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayoutExpr {
    Source {
        tensor: TensorId,
        decl: TensorDecl,
    },
    /// A compiled affine expression: a list of rectangular copies over
    /// `inputs`, plus an optional zero fill.
    ///
    /// This is the single node the whole affine fragment of `contract::Expr`
    /// lowers to — slicing, sharding, concatenation, stacking, reshaping and
    /// padding all arrive here already resolved into strides, by
    /// `contract::compile`. There is nothing left for the optimizer to rewrite
    /// and nothing left for the planner to interpret: each piece is one
    /// `StridedExtent`.
    Gather {
        inputs: Vec<ExprId>,
        pieces: Vec<GatherPiece>,
        zero_fill: bool,
        decl: TensorDecl,
    },
    Cast {
        input: ExprId,
        dtype: DType,
        decl: TensorDecl,
    },
    Decode {
        scheme: QuantScheme,
        data: ExprId,
        metadata: Vec<ExprId>,
        decl: TensorDecl,
    },
    Encode {
        scheme: QuantScheme,
        input: ExprId,
        metadata_outputs: Vec<TensorDecl>,
        decl: TensorDecl,
    },
    Transcode {
        from: QuantScheme,
        to: QuantScheme,
        data: ExprId,
        metadata: Vec<ExprId>,
        metadata_outputs: Vec<TensorDecl>,
        decl: TensorDecl,
    },
    Repack {
        input: ExprId,
        spec: RepackSpec,
        decl: TensorDecl,
    },
    Attach {
        data: ExprId,
        metadata: Vec<ExprId>,
        decl: TensorDecl,
    },
    Realize {
        input: ExprId,
        runtime_name: String,
        decl: TensorDecl,
    },
}

/// One rectangular copy inside a [`LayoutExpr::Gather`].
///
/// `input` indexes the gather's `inputs`. Offsets and strides are in **bytes**,
/// like everything below this point — the element width is a property of the
/// encoding, and a sub-byte encoding has no element addresses at all. `dims`
/// runs outermost first and its last entry is the contiguous inner block, so a
/// piece maps one-for-one onto `load_plan::StridedExtent`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatherPiece {
    pub input: u32,
    pub src_offset: u64,
    pub dst_offset: u64,
    pub dims: Vec<GatherDim>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatherDim {
    pub count: i64,
    pub src_stride: i64,
    pub dst_stride: i64,
}

impl GatherPiece {
    /// A single contiguous byte range.
    pub fn span(input: u32, src_offset: u64, dst_offset: u64, bytes: u64) -> Self {
        Self {
            input,
            src_offset,
            dst_offset,
            dims: vec![GatherDim {
                count: bytes as i64,
                src_stride: 1,
                dst_stride: 1,
            }],
        }
    }

    pub fn bytes(&self) -> u64 {
        self.dims.iter().map(|dim| dim.count).product::<i64>() as u64
    }

    /// Whether this piece moves one unbroken block, and so could be aliased
    /// rather than copied.
    pub fn is_contiguous(&self) -> bool {
        self.dims.len() == 1 && self.dims[0].src_stride == 1 && self.dims[0].dst_stride == 1
    }
}

impl LayoutExpr {
    pub fn inputs(&self) -> Vec<ExprId> {
        match self {
            Self::Source { .. } => Vec::new(),
            Self::Cast { input, .. }
            | Self::Encode { input, .. }
            | Self::Repack { input, .. }
            | Self::Realize { input, .. } => vec![*input],
            Self::Gather { inputs, .. } => inputs.clone(),
            Self::Decode { data, metadata, .. } | Self::Transcode { data, metadata, .. } => {
                let mut out = vec![*data];
                out.extend(metadata.iter().copied());
                out
            }
            Self::Attach { data, metadata, .. } => {
                let mut out = vec![*data];
                out.extend(metadata.iter().copied());
                out
            }
        }
    }

    pub fn decl(&self) -> &TensorDecl {
        match self {
            Self::Source { decl, .. }
            | Self::Gather { decl, .. }
            | Self::Cast { decl, .. }
            | Self::Decode { decl, .. }
            | Self::Encode { decl, .. }
            | Self::Transcode { decl, .. }
            | Self::Repack { decl, .. }
            | Self::Attach { decl, .. }
            | Self::Realize { decl, .. } => decl,
        }
    }

    pub fn decl_mut(&mut self) -> &mut TensorDecl {
        match self {
            Self::Source { decl, .. }
            | Self::Gather { decl, .. }
            | Self::Cast { decl, .. }
            | Self::Decode { decl, .. }
            | Self::Encode { decl, .. }
            | Self::Transcode { decl, .. }
            | Self::Repack { decl, .. }
            | Self::Attach { decl, .. }
            | Self::Realize { decl, .. } => decl,
        }
    }
}
