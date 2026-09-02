//! Traces built by hand, one statement per line — the test vocabulary for
//! every pass in this crate. `model-dsl` is a dev-dependency and cannot be
//! reached from a unit test here, so these say in `Def`, `Ty` and `Guard`
//! what a forward pass says in `split` and `Value::merge`.

use model_ir::ops::{Attention, Elementwise};
use model_ir::{
    CacheRow, Guard, Def, Dim, Dtype, Node, Trace, Platform, RuntimeInput, Seam, StructKind, Ty,
    ValueDecl, ValueId,
};

/// A trace under construction.
pub(crate) struct Build {
    pub(crate) trace: Trace,
    inputs: u32,
}

/// The ordinary activation rectangle: one row per token, `width` elements
/// wide, in the platform's activation element.
pub(crate) fn act(width: u64) -> Ty {
    Ty::Tensor {
        shape: vec![Dim::Tokens, Dim::Const(width)],
        dtype: Dtype::Bf16,
    }
}

/// The tower's rectangle: one row per patch, `width` elements wide (a
/// leading `Dim::Patches`).
pub(crate) fn patch(width: u64) -> Ty {
    Ty::Tensor {
        shape: vec![Dim::Patches, Dim::Const(width)],
        dtype: Dtype::Bf16,
    }
}

/// `Guard::Fact(bit)`, spelled short.
pub(crate) fn fact(bit: u8) -> Guard {
    Guard::Fact(bit)
}

impl Build {
    pub(crate) fn new() -> Build {
        Build {
            trace: Trace {
                name: "hand-built".to_string(),
                platform: Platform::Cuda,
                params: Vec::new(),
                caches: vec![CacheRow::State {
                    name: "state".to_string(),
                    slab: vec![1],
                    dtype: Dtype::Bf16,
                }],
                values: Vec::new(),
                nodes: Vec::new(),
                seams: Vec::new(),
            },
            inputs: 0,
        }
    }

    pub(crate) fn value(&mut self, def: Def, ty: Ty) -> ValueId {
        self.trace.values.push(ValueDecl { def, ty });
        ValueId((self.trace.values.len() - 1) as u32)
    }

    /// A demand sink the engine binds, distinct per call.
    pub(crate) fn input(&mut self, width: u64) -> ValueId {
        self.inputs += 1;
        let which = RuntimeInput::Mask {
            space: self.inputs - 1,
        };
        self.value(Def::Input(which), act(width))
    }

    pub(crate) fn cache(&mut self) -> ValueId {
        self.value(Def::Cache(0), act(1))
    }

    /// One guarded op over `x`, minting a fresh `width`-wide rectangle.
    /// `rmsnorm.no_scale` since it declares no in-place alias.
    pub(crate) fn op(&mut self, x: ValueId, width: u64, guard: Guard) -> ValueId {
        let node = self.trace.nodes.len() as u32;
        let y = self.value(Def::Op(node), act(width));
        self.push(
            Elementwise::RmsnormNoScale {
                x,
                head_dim: 1,
                eps: 1e-6,
                y,
            }
            .into(),
            guard,
        );
        y
    }

    /// The same op, minting a rectangle of the declared `ty` instead of the
    /// ordinary token-shaped one — for when the shape is under test.
    pub(crate) fn shaped(&mut self, x: ValueId, ty: Ty, guard: Guard) -> ValueId {
        let node = self.trace.nodes.len() as u32;
        let y = self.value(Def::Op(node), ty);
        self.push(
            Elementwise::RmsnormNoScale {
                x,
                head_dim: 1,
                eps: 1e-6,
                y,
            }
            .into(),
            guard,
        );
        y
    }

    /// The same plan build, but reading its `kv_indptr` from a value the
    /// caller names rather than from a fresh runtime input — the one shape
    /// `region::hoist` refuses, since a plan build over an activation has no
    /// instant to run in.
    pub(crate) fn prepare_over(&mut self, kv_indptr: ValueId, guard: Guard) -> ValueId {
        let kv_indices = self.input(1);
        let last_page_len = self.input(1);
        let kv_len = self.input(1);
        let node = self.trace.nodes.len() as u32;
        let plan = self.value(Def::Op(node), Ty::Struct(StructKind::AttnDecodePlan));
        self.push(
            Attention::PlanDecode {
                kv_indptr,
                kv_indices,
                last_page_len,
                kv_len,
                q_heads: 1,
                kv_heads: 1,
                head_dim: 4,
                window: None,
                plan,
            }
            .into(),
            guard,
        );
        plan
    }

    /// The attention that reads a prepare node's struct.
    pub(crate) fn decode(&mut self, q: ValueId, plan: ValueId, guard: Guard) -> ValueId {
        let cache = self.cache();
        let node = self.trace.nodes.len() as u32;
        let o = self.value(Def::Op(node), act(4));
        self.push(
            Attention::Decode {
                q,
                plan,
                cache,
                window: None,
                head_dim: 4,
                sm_scale: 1.0,
                o,
            }
            .into(),
            guard,
        );
        o
    }

    /// A cache write: an effect root, and it hands nothing back.
    pub(crate) fn append(&mut self, x: ValueId, guard: Guard) -> usize {
        let cache = self.cache();
        let write_page = self.input(1);
        let write_offset = self.input(1);
        self.push(
            Attention::KvAppendShared {
                plane: x,
                cache,
                write_page,
                write_offset,
            }
            .into(),
            guard,
        );
        self.trace.nodes.len() - 1
    }

    pub(crate) fn merge(&mut self, arms: &[(ValueId, Guard)], width: u64) -> ValueId {
        self.value(Def::Merge(arms.to_vec()), act(width))
    }

    /// The `"out"` seam: roots the demand walk and pins the arena to fire end.
    pub(crate) fn out(&mut self, v: ValueId) -> &mut Build {
        self.trace.seams.push(Seam {
            seam: "out".to_string(),
            values: vec![v],
            layer: None,
        });
        self
    }

    fn push(&mut self, op: model_ir::Operation, guard: Guard) {
        self.trace.nodes.push(Node {
            op,
            guard,
            layer: None,
        });
    }
}
