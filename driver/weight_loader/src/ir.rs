use serde::{Deserialize, Serialize};

use crate::types::{Axis, DType, ExprId, Layout, QuantScheme, TensorDecl, TensorId};

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
        self.expr(id)?.decl()
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
    Select {
        input: ExprId,
        axis: Axis,
        start: i64,
        length: i64,
        decl: TensorDecl,
    },
    Partition {
        input: ExprId,
        axis: Axis,
        parts: u32,
        index: u32,
        decl: TensorDecl,
    },
    Join {
        inputs: Vec<ExprId>,
        axis: Axis,
        decl: TensorDecl,
    },
    Stack {
        inputs: Vec<ExprId>,
        axis: Axis,
        decl: TensorDecl,
    },
    Unzip {
        input: ExprId,
        axis: Axis,
        outputs: Vec<TensorDecl>,
    },
    Reorder {
        input: ExprId,
        perm: Vec<u8>,
        decl: TensorDecl,
    },
    View {
        input: ExprId,
        layout: Layout,
        axis: Option<Axis>,
        start: i64,
        length: i64,
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

impl LayoutExpr {
    pub fn inputs(&self) -> Vec<ExprId> {
        match self {
            Self::Source { .. } => Vec::new(),
            Self::Select { input, .. }
            | Self::Partition { input, .. }
            | Self::Unzip { input, .. }
            | Self::Reorder { input, .. }
            | Self::View { input, .. }
            | Self::Cast { input, .. }
            | Self::Encode { input, .. }
            | Self::Realize { input, .. } => vec![*input],
            Self::Join { inputs, .. } | Self::Stack { inputs, .. } => inputs.clone(),
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

    pub fn decl(&self) -> Option<&TensorDecl> {
        match self {
            Self::Source { decl, .. }
            | Self::Select { decl, .. }
            | Self::Partition { decl, .. }
            | Self::Join { decl, .. }
            | Self::Stack { decl, .. }
            | Self::Reorder { decl, .. }
            | Self::View { decl, .. }
            | Self::Cast { decl, .. }
            | Self::Decode { decl, .. }
            | Self::Encode { decl, .. }
            | Self::Transcode { decl, .. }
            | Self::Attach { decl, .. }
            | Self::Realize { decl, .. } => Some(decl),
            Self::Unzip { .. } => None,
        }
    }

    pub fn decl_mut(&mut self) -> Option<&mut TensorDecl> {
        match self {
            Self::Source { decl, .. }
            | Self::Select { decl, .. }
            | Self::Partition { decl, .. }
            | Self::Join { decl, .. }
            | Self::Stack { decl, .. }
            | Self::Reorder { decl, .. }
            | Self::View { decl, .. }
            | Self::Cast { decl, .. }
            | Self::Decode { decl, .. }
            | Self::Encode { decl, .. }
            | Self::Transcode { decl, .. }
            | Self::Attach { decl, .. }
            | Self::Realize { decl, .. } => Some(decl),
            Self::Unzip { .. } => None,
        }
    }
}
