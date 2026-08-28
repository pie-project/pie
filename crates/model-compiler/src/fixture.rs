//! Traces built by hand, one statement per line — the test vocabulary for
//! every pass in this crate.
//!
//! `model-dsl` is the authoring surface and CANNOT be reached from a unit
//! test here: it is a dev-dependency, which means it exists for
//! `tests/every_sku_carves_an_arena.rs` and not for `src/`. So these say in
//! `Def`, `Ty` and `Guard` what a forward pass says in `split` and
//! `Value::merge`, the same way `model_ir::check::classes`' own tests do. The
//! catalog test is the one that checks the two agree.

use model_ir::ops::{Attention, Collective, Elementwise};
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

/// A rectangle whose leading dim is a CONSTANT — a bias plane, a fixed block.
/// No window cuts one (`RowExpr::cut_per_class`), which is what makes it the
/// shape two classes may never share a column of.
pub(crate) fn block(rows: u64, width: u64) -> Ty {
    Ty::Tensor {
        shape: vec![Dim::Const(rows), Dim::Const(width)],
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

    /// A demand sink the driver binds, distinct per call.
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
    /// `rmsnorm.no_scale` because it declares NO in-place alias — the ordinary
    /// case, where a result needs bytes of its own.
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
    /// ordinary token-shaped one — what a test reaches for when the SHAPE is
    /// the thing under test.
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

    /// One op that writes THROUGH its operand — `Operands::aliases` says so,
    /// and the carve is expected to fold the two onto one column.
    pub(crate) fn in_place(&mut self, x: ValueId, width: u64, guard: Guard) -> ValueId {
        let node = self.trace.nodes.len() as u32;
        let x_out = self.value(Def::Op(node), act(width));
        self.push(Elementwise::MulScalar { s: 2.0, x, x_out }.into(), guard);
        x_out
    }

    /// `y_out = y + x`, written through `y` — the residual ledger every
    /// transformer is, and the shape the carve's reuse claim is really about.
    pub(crate) fn residual_add(
        &mut self,
        x: ValueId,
        y: ValueId,
        width: u64,
        guard: Guard,
    ) -> ValueId {
        let node = self.trace.nodes.len() as u32;
        let y_out = self.value(Def::Op(node), act(width));
        self.push(Elementwise::ResidualAdd { x, y, y_out }.into(), guard);
        y_out
    }

    /// A prepare node: it defines a `Ty::Struct`, which is the whole rule P5
    /// reads. The reading it states is the one [`Build::decode`] restates —
    /// one head of width 4, no window — because a schedule and its reader
    /// disagreeing is a shell refusal rather than a fixture.
    pub(crate) fn prepare(&mut self, guard: Guard) -> ValueId {
        let kv_indptr = self.input(1);
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

    /// The same plan build, but reading its `kv_indptr` from a value the
    /// caller names rather than from a fresh runtime input.
    ///
    /// THE ONE SHAPE `region::hoist` REFUSES, when the caller hands it an op's
    /// output: a plan build over an activation is host work with no instant to
    /// run in — after the graph computed the number, and before the graph the
    /// build's own struct is read by.
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

    /// A collective — the family P3 may never elide.
    pub(crate) fn all_gather(&mut self, x: ValueId, width: u64, guard: Guard) -> ValueId {
        let node = self.trace.nodes.len() as u32;
        let y = self.value(Def::Op(node), act(width));
        self.push(Collective::AllGather { x, y }.into(), guard);
        y
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

    /// The `"out"` seam — what a trace writes the forward's return value as,
    /// and therefore what roots the demand walk and pins the arena to fire end.
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
