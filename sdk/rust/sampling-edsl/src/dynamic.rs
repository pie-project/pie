//! Runtime (length-erased) builder handles + the shared DAG helpers.
//!
//! [`DynValue`] is an SSA handle that carries its [`ir::ValueType`] as data and
//! emits onto the shared [`GraphInner`](crate::builder) flat op list. The vocab
//! is a runtime value (an inferlet learns it from the model), so this is the
//! authoring surface. Shape/dtype inference here is light; the canonical
//! `pie-sampling-ir` validator (run at [`Graph::build`](crate::builder::Graph::build))
//! is the authority.

use crate::builder::{Graph, NodeId};
use crate::ir;

/// A runtime, length-erased SSA handle.
#[derive(Clone)]
pub struct DynValue {
    id: NodeId,
    graph: Graph,
    ty: ir::ValueType,
    key: Option<ir::TensorKey>,
}

/// Result shape of a binary elementwise op (scalar operand broadcasts).
fn binary_shape(a: ir::Shape, b: ir::Shape) -> ir::Shape {
    if a.is_scalar() { b } else { a }
}

/// Result shape of a last-axis reduction (`[n] -> []`, `[m, n] -> [m]`).
fn reduce_shape(s: ir::Shape) -> ir::Shape {
    s.drop_last().unwrap_or(ir::Shape::SCALAR)
}

impl DynValue {
    pub(crate) fn new(
        id: NodeId,
        graph: Graph,
        ty: ir::ValueType,
        key: Option<ir::TensorKey>,
    ) -> Self {
        Self { id, graph, ty, key }
    }

    /// This value's runtime type.
    pub fn ty(&self) -> ir::ValueType {
        self.ty
    }
    /// This value's SSA id.
    pub fn node_id(&self) -> NodeId {
        self.id
    }
    /// The host tensor key, if this is a host-bound input.
    pub fn input_key(&self) -> Option<ir::TensorKey> {
        self.key
    }
    /// The owning graph handle.
    pub fn graph(&self) -> Graph {
        self.graph.clone()
    }

    fn emit(&self, op: ir::Op, ty: ir::ValueType) -> DynValue {
        let id = self.graph.inner_ref().borrow_mut().emit(op, &[ty]);
        DynValue::new(id, self.graph.clone(), ty, None)
    }
    fn f32(&self, shape: ir::Shape) -> ir::ValueType {
        ir::ValueType::new(shape, ir::DType::F32)
    }

    // -- unary --
    pub fn exp(&self) -> DynValue {
        self.emit(ir::Op::Exp(self.id), self.ty)
    }
    pub fn log(&self) -> DynValue {
        self.emit(ir::Op::Log(self.id), self.ty)
    }
    pub fn recip(&self) -> DynValue {
        self.emit(ir::Op::Recip(self.id), self.ty)
    }
    pub fn neg(&self) -> DynValue {
        self.emit(ir::Op::Neg(self.id), self.ty)
    }
    pub fn abs(&self) -> DynValue {
        self.emit(ir::Op::Abs(self.id), self.ty)
    }

    // -- binary f32 (scalar operand broadcasts) --
    fn bin(&self, rhs: &DynValue, mk: impl FnOnce(NodeId, NodeId) -> ir::Op) -> DynValue {
        let shape = binary_shape(self.ty.shape, rhs.ty.shape);
        self.emit(mk(self.id, rhs.id), self.f32(shape))
    }
    pub fn add(&self, rhs: &DynValue) -> DynValue {
        self.bin(rhs, ir::Op::Add)
    }
    pub fn sub(&self, rhs: &DynValue) -> DynValue {
        self.bin(rhs, ir::Op::Sub)
    }
    pub fn mul(&self, rhs: &DynValue) -> DynValue {
        self.bin(rhs, ir::Op::Mul)
    }
    pub fn div(&self, rhs: &DynValue) -> DynValue {
        self.bin(rhs, ir::Op::Div)
    }
    pub fn max_elem(&self, rhs: &DynValue) -> DynValue {
        self.bin(rhs, ir::Op::MaxElem)
    }
    pub fn min_elem(&self, rhs: &DynValue) -> DynValue {
        self.bin(rhs, ir::Op::MinElem)
    }

    // -- comparisons -> bool --
    fn cmp(&self, rhs: &DynValue, mk: impl FnOnce(NodeId, NodeId) -> ir::Op) -> DynValue {
        let shape = binary_shape(self.ty.shape, rhs.ty.shape);
        self.emit(mk(self.id, rhs.id), ir::ValueType::new(shape, ir::DType::Bool))
    }
    pub fn gt(&self, rhs: &DynValue) -> DynValue {
        self.cmp(rhs, ir::Op::Gt)
    }
    pub fn ge(&self, rhs: &DynValue) -> DynValue {
        self.cmp(rhs, ir::Op::Ge)
    }
    pub fn eq(&self, rhs: &DynValue) -> DynValue {
        self.cmp(rhs, ir::Op::Eq)
    }

    // -- reductions (last axis; rank>=2 -> per-row Vector{rows}) --
    pub fn reduce_max(&self) -> DynValue {
        self.emit(ir::Op::ReduceMax(self.id), self.f32(reduce_shape(self.ty.shape)))
    }
    pub fn reduce_sum(&self) -> DynValue {
        self.emit(ir::Op::ReduceSum(self.id), self.f32(reduce_shape(self.ty.shape)))
    }
    pub fn reduce_min(&self) -> DynValue {
        self.emit(ir::Op::ReduceMin(self.id), self.f32(reduce_shape(self.ty.shape)))
    }
    /// Argmax over the last axis -> I32 token id(s) (scalar, or per-row `[rows]`).
    pub fn argmax(&self) -> DynValue {
        self.emit(
            ir::Op::ReduceArgmax(self.id),
            ir::ValueType::new(reduce_shape(self.ty.shape), ir::DType::I32),
        )
    }

    // -- scans (last axis; shape-preserving) --
    pub fn cumsum(&self) -> DynValue {
        self.emit(ir::Op::CumSum(self.id), self.ty)
    }
    pub fn cumprod(&self) -> DynValue {
        self.emit(ir::Op::CumProd(self.id), self.ty)
    }

    // -- broadcast (Op::Broadcast, left-aligned replicate) --
    /// Replicate a scalar to a vector `[n]`.
    pub fn broadcast_vec(&self, n: u32) -> DynValue {
        let shape = ir::Shape::vector(n);
        self.emit(ir::Op::Broadcast { value: self.id, shape }, ir::ValueType::new(shape, self.ty.dtype))
    }
    /// Replicate a scalar to a matrix `[m, n]`.
    pub fn broadcast_matrix(&self, m: u32, n: u32) -> DynValue {
        let shape = ir::Shape::matrix(m, n);
        self.emit(ir::Op::Broadcast { value: self.id, shape }, ir::ValueType::new(shape, self.ty.dtype))
    }
    /// Per-row lift a `[m]` vector to a `[m, n]` matrix (`out[i, :] = self[i]`) —
    /// the per-row divide in a matrix softmax / residual renorm.
    pub fn row_broadcast(&self, n: u32) -> DynValue {
        let m = self.ty.shape.dims().first().copied().unwrap_or(0);
        let shape = ir::Shape::matrix(m, n);
        self.emit(ir::Op::Broadcast { value: self.id, shape }, ir::ValueType::new(shape, self.ty.dtype))
    }

    // -- pivot-threshold masks (f32 input) -> bool, same shape --
    /// `RankLe(k)` keeps the top-`k` by logit; since #25 `k` is an integer
    /// value-id (a `U32`/`I32` scalar input or per-row `[rows]` vector), not a
    /// baked immediate.
    pub fn pivot_rank_le(&self, k: &DynValue) -> DynValue {
        self.emit(
            ir::Op::PivotThreshold { input: self.id, predicate: ir::Predicate::RankLe(k.id) },
            ir::ValueType::new(self.ty.shape, ir::DType::Bool),
        )
    }
    pub fn pivot_cummass_le(&self, p: &DynValue) -> DynValue {
        self.emit(
            ir::Op::PivotThreshold { input: self.id, predicate: ir::Predicate::CummassLe(p.id) },
            ir::ValueType::new(self.ty.shape, ir::DType::Bool),
        )
    }
    pub fn pivot_prob_ge(&self, thr: &DynValue) -> DynValue {
        self.emit(
            ir::Op::PivotThreshold { input: self.id, predicate: ir::Predicate::ProbGe(thr.id) },
            ir::ValueType::new(self.ty.shape, ir::DType::Bool),
        )
    }

    // -- indexing --
    /// 1-D gather `out[j] = self[idx[j]]` (`self` is `[n]`, result `[k]`).
    ///
    /// Result dtype = the SOURCE dtype (matches the canonical validator,
    /// `validate.rs`: `ValueType::vector(k, ts.dtype)`) — so a gather of the i32
    /// `picked` intrinsic stays i32 and can be output as a `Token` (the Stage-2
    /// seed extract). The old hardcoded-f32 label was a latent inconsistency that
    /// never bit only because gather results were previously always compared or
    /// reduced, never output directly.
    pub fn gather(&self, idx: &DynValue) -> DynValue {
        // Result dtype = the SOURCE dtype (matches the validator + driver interp;
        // a gather of the i32 `picked` stays i32 so it can be output as a Token).
        // Re-applied from the mtp_specdecode fix (`6ee06bfc` on dev) — dedups at
        // the echo→pipe-audit fold.
        let k = idx.ty.shape.last_len().unwrap_or(0);
        let ty = ir::ValueType::new(ir::Shape::vector(k), self.ty.dtype);
        self.emit(ir::Op::Gather { src: self.id, idx: idx.id }, ty)
    }
    /// Per-row column pick `out[i] = self[i, idx[i]]` (`self` is `[m, n]`, `idx`
    /// `[m]` integer, result `[m]`). The lossless accept-ratio lookup.
    pub fn gather_row(&self, idx: &DynValue) -> DynValue {
        let m = self.ty.shape.dims().first().copied().unwrap_or(0);
        self.emit(ir::Op::GatherRow { src: self.id, idx: idx.id }, self.f32(ir::Shape::vector(m)))
    }

    /// Apply a packed allowed-token bitmask to `self` (logits `[n]`):
    /// `out[j] = bit_j(mask) ? self[j] : −∞`, `bit_j = (mask[j>>5] >> (j&31)) &
    /// 1` (bit 1 = allowed → pass-through, bit 0 = disallowed → `−∞`). `mask` is
    /// a packed `[ceil(n/32)]` u32 vector (the matcher's allowed-token bits).
    /// Result ≡ `self` (shape + dtype). The de-hardwired grammar mask op.
    pub fn mask_apply(&self, mask: &DynValue) -> DynValue {
        self.emit(
            ir::Op::MaskApply { logits: self.id, mask: mask.id },
            ir::ValueType::new(self.ty.shape, self.ty.dtype),
        )
    }
}

/// Elementwise `cond ? a : b` for runtime handles.
pub fn dselect(cond: &DynValue, a: &DynValue, b: &DynValue) -> DynValue {
    let shape = binary_shape(binary_shape(a.ty.shape, b.ty.shape), cond.ty.shape);
    let ty = ir::ValueType::new(shape, a.ty.dtype);
    let id = a
        .graph
        .inner_ref()
        .borrow_mut()
        .emit(ir::Op::Select { cond: cond.id, a: a.id, b: b.id }, &[ty]);
    DynValue::new(id, a.graph.clone(), ty, None)
}

// ============================================================================
// Input declarations + RNG on Graph
// ============================================================================

impl Graph {
    fn input(&self, ty: ir::ValueType, binding: ir::Binding) -> DynValue {
        let key = match binding {
            ir::Binding::Tensor { key, .. } => Some(key),
            _ => None,
        };
        let id = self.inner_ref().borrow_mut().add_input(ty, binding);
        DynValue::new(id, self.clone(), ty, key)
    }
    fn konst(&self, lit: ir::Literal) -> DynValue {
        let ty = ir::ValueType::scalar(lit.dtype());
        let id = self.inner_ref().borrow_mut().add_const(lit);
        DynValue::new(id, self.clone(), ty, None)
    }

    /// The intrinsic next-token logits as a `[vocab]` f32 vector.
    pub fn intrinsic_logits_dyn(&self) -> DynValue {
        self.input(ir::ValueType::vector(self.vocab(), ir::DType::F32), ir::Binding::Logits)
    }
    /// The intrinsic logits as a `[rows, vocab]` matrix (one row per draft
    /// position — v4 spec-decode verify block).
    pub fn intrinsic_logits_matrix_dyn(&self, rows: u32) -> DynValue {
        let ty = ir::ValueType::new(ir::Shape::matrix(rows, self.vocab()), ir::DType::F32);
        self.input(ty, ir::Binding::Logits)
    }
    /// The speculator's **draft** logits as a `[vocab]` f32 vector (de-hardwired
    /// speculation, M=1). Binds [`ir::Binding::MtpLogits`] → the driver
    /// source-selects the draft row of `ws.logits` (not a separate buffer). The
    /// program treats it like any logits vector; only the binding differs.
    pub fn intrinsic_mtp_logits_dyn(&self) -> DynValue {
        self.input(ir::ValueType::vector(self.vocab(), ir::DType::F32), ir::Binding::MtpLogits)
    }
    /// The speculator's draft logits as a **`[K, vocab]` matrix** (Stage 2
    /// PTIR-native MTP, overview §6.1): the MTP head's K next-step draft
    /// proposals, one row per draft position — `argmax` per row yields the
    /// K fresh drafts. K is trace-known (a different K = a different traced
    /// program). Binds [`ir::Binding::MtpLogits`]; the driver maps the K rows
    /// onto the pass's draft rows of `ws.logits` at fire time. Shape contract:
    /// `pie_sampling_ir::validate::intrinsic_decl_ok`.
    pub fn intrinsic_mtp_logits_matrix_dyn(&self, k: u32) -> DynValue {
        let ty = ir::ValueType::new(ir::Shape::matrix(k, self.vocab()), ir::DType::F32);
        self.input(ty, ir::Binding::MtpLogits)
    }

    /// The window's **draft token ids** as a device-resident `[k]` i32 vector —
    /// the CURRENT window's drafts (produced by the previous fire's MTP argmax,
    /// composed into the retained `[k+1]` window's rows `1..=k`). Binds
    /// [`ir::Binding::MtpDrafts`] → the driver source-selects the retained
    /// `mtp_drafts` `[k]` buffer (bravo's per-link retain). The device-resident
    /// analog of `spec_verify_greedy`'s host `draft` input: the verify compares
    /// `target.argmax().eq(&this)` with ZERO host round-trip on the drafts. `k`
    /// trace-known. Shape contract: `pie_sampling_ir::validate::intrinsic_decl_ok`
    /// (I32 ∧ `[k≥1]`).
    pub fn intrinsic_mtp_drafts_dyn(&self, k: u32) -> DynValue {
        self.input(ir::ValueType::vector(k, ir::DType::I32), ir::Binding::MtpDrafts)
    }

    pub fn constant_f32_dyn(&self, x: f32) -> DynValue {
        self.konst(ir::Literal::F32(x))
    }
    pub fn constant_i32_dyn(&self, x: i32) -> DynValue {
        self.konst(ir::Literal::I32(x))
    }
    pub fn constant_bool_dyn(&self, x: bool) -> DynValue {
        self.konst(ir::Literal::Bool(x))
    }

    /// A host-supplied scalar tensor input of `dtype`.
    pub fn host_scalar_dyn(&self, dtype: ir::DType, ready: ir::Readiness) -> DynValue {
        let key = self.next_key();
        self.input(ir::ValueType::scalar(dtype), ir::Binding::Tensor { key, ready })
    }
    /// A host-supplied vocab-length vector tensor input.
    pub fn host_vocab_vector_dyn(&self, dtype: ir::DType, ready: ir::Readiness) -> DynValue {
        let key = self.next_key();
        let ty = ir::ValueType::vector(self.vocab(), dtype);
        self.input(ty, ir::Binding::Tensor { key, ready })
    }
    /// A host-supplied fixed-length vector tensor input.
    pub fn host_vector_dyn(&self, dtype: ir::DType, len: u32, ready: ir::Readiness) -> DynValue {
        let key = self.next_key();
        self.input(ir::ValueType::vector(len, dtype), ir::Binding::Tensor { key, ready })
    }
    /// A host-supplied `[rows, len]` matrix tensor input.
    pub fn host_matrix_dyn(
        &self,
        dtype: ir::DType,
        rows: u32,
        len: u32,
        ready: ir::Readiness,
    ) -> DynValue {
        let key = self.next_key();
        let ty = ir::ValueType::new(ir::Shape::matrix(rows, len), dtype);
        self.input(ty, ir::Binding::Tensor { key, ready })
    }

    /// Emit an `Op::Rng{stream, shape, kind}` (no seed operand — the per-fire
    /// ambient seed is supplied by the runtime; `stream` is a static per-op salt
    /// decorrelating multiple draws). `stream = 0` reproduces today's hash.
    fn rng(&self, stream: u32, shape: ir::Shape, kind: ir::RngKind) -> DynValue {
        let ty = ir::ValueType::new(shape, ir::DType::F32);
        let id = self.inner_ref().borrow_mut().emit(ir::Op::Rng { stream, shape, kind }, &[ty]);
        DynValue::new(id, self.clone(), ty, None)
    }
    /// Uniform `[0,1)` noise vector `[len]` at `stream`.
    pub fn rng_uniform_vec(&self, stream: u32, len: u32) -> DynValue {
        self.rng(stream, ir::Shape::vector(len), ir::RngKind::Uniform)
    }
    /// Gumbel noise vector `[len]` at `stream`.
    pub fn rng_gumbel_vec(&self, stream: u32, len: u32) -> DynValue {
        self.rng(stream, ir::Shape::vector(len), ir::RngKind::Gumbel)
    }
    /// Gumbel noise matrix `[rows, len]` at `stream` (per-row spec-decode resample).
    pub fn rng_gumbel_matrix(&self, stream: u32, rows: u32, len: u32) -> DynValue {
        self.rng(stream, ir::Shape::matrix(rows, len), ir::RngKind::Gumbel)
    }
}

// ============================================================================
// Shared DAG helpers (one source for `program` + `sugar`)
// ============================================================================

/// `softmax(x)` over a `[n]` vector — max-shifted for stability.
pub fn dyn_softmax(x: &DynValue) -> DynValue {
    let m = x.reduce_max();
    let e = x.sub(&m).exp();
    let s = e.reduce_sum();
    e.div(&s)
}

/// Per-row softmax over a `[rows, vocab]` matrix (the `ReduceMax -> row_broadcast
/// -> Exp/Sub -> ReduceSum -> row_broadcast -> Div` recipe).
pub fn dyn_softmax_rows(m: &DynValue, vocab: u32) -> DynValue {
    let rowmax = m.reduce_max().row_broadcast(vocab);
    let e = m.sub(&rowmax).exp();
    let z = e.reduce_sum().row_broadcast(vocab);
    e.div(&z)
}

/// Temperature scaling `x * (1/T)`.
pub fn dyn_temperature_scale(x: &DynValue, temp: &DynValue) -> DynValue {
    x.mul(&temp.recip())
}

/// `Select{ keep, scores, -inf }` — non-kept lanes become `-inf`. Works for any
/// `scores` shape (the `-inf` const is broadcast to it).
pub fn dyn_mask_to_score(g: &Graph, keep: &DynValue, scores: &DynValue) -> DynValue {
    let neg_inf = broadcast_like(g, f32::NEG_INFINITY, scores);
    dselect(keep, scores, &neg_inf)
}

/// Min-p keep-mask in logit space: `logit >= max_logit + log(p)`.
pub fn dyn_min_p_mask(logits: &DynValue, p: &DynValue) -> DynValue {
    let thr = logits.reduce_max().add(&p.log());
    logits.ge(&thr)
}

/// Gumbel-max selection `argmax(scores + gumbel(stream))` -> token.
pub fn dyn_gumbel_argmax(g: &Graph, scores: &DynValue, stream: u32, len: u32) -> DynValue {
    scores.add(&g.rng_gumbel_vec(stream, len)).argmax()
}

/// Lossless residual resample over `[rows, vocab]` `p`/`q`:
/// `argmax_rows( log(max(0, p-q)) + Gumbel{stream} )` (per-row). `log(0)=-inf`
/// masks zero-residual lanes; per-row normalization is argmax-invariant.
pub fn dyn_residual_resample_rows(
    g: &Graph,
    p: &DynValue,
    q: &DynValue,
    stream: u32,
    rows: u32,
    vocab: u32,
) -> DynValue {
    let zero = g.constant_f32_dyn(0.0);
    let residual = p.sub(q).max_elem(&zero);
    let log_res = residual.log();
    log_res.add(&g.rng_gumbel_matrix(stream, rows, vocab)).argmax()
}

/// `Eq(x, const)` for a scalar f32 constant.
pub fn dyn_eq_const(g: &Graph, x: &DynValue, c: f32) -> DynValue {
    x.eq(&g.constant_f32_dyn(c))
}

/// A const `c` broadcast to the shape of `like` (scalar -> like's shape).
fn broadcast_like(g: &Graph, c: f32, like: &DynValue) -> DynValue {
    let k = g.constant_f32_dyn(c);
    let shape = like.ty().shape;
    let dims = shape.dims();
    match dims.len() {
        0 => k,
        1 => k.broadcast_vec(dims[0]),
        _ => k.broadcast_matrix(dims[0], dims[1]),
    }
}
