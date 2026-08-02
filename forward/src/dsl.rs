//! The declaration surface (north-star-dsl.md, "the v2 surface").
//!
//! plan.md's sketch, made real: values carry the tape, so ops are free
//! functions and operators rather than builder methods; weights are typed
//! handles from a per-layer namespace, so no declaration spells a string
//! or a width; per-layer state is an object (`Kv::append`); and a LOWERED
//! declaration calls **raw kernel signatures** ([`cuda`]) — functions
//! named for the driver's launcher symbols, whose parameters are the
//! launcher's semantic operands — recording [`OpKind::Launch`] ops.
//!
//! The recording substrate is [`TraceBuilder`], unchanged: this module is
//! surface, not IR. The one behavioral subtlety lives in `+=`:
//!
//! * `y += matmul(&a, &w.o_proj)` — if the matmul is the op just
//!   recorded and nothing else consumed it, the tape REWRITES it to the
//!   `beta_one` accumulate form (`matmul_add`), the hand-written passes'
//!   cuBLAS residual fold. Output id unchanged, so the fold is invisible
//!   to dataflow.
//! * `y += anything_else` — a [`OpKind::ResidualAdd`] launch, exactly the
//!   post-norm landing the hand-written pass makes explicit.
//!
//! Op `layer` tags derive from what an op touches — its weight handle,
//! its state handle, or its first input — rather than from a bracketing
//! closure; the semantic goldens pin that this derivation reproduces the
//! bracketed tagging byte for byte.

use std::cell::RefCell;
use std::rc::Rc;

use crate::facts::{LlamaLikeCudaFacts, LlamaLikeFacts, QkNorm};
use crate::trace::{
    DType, Dim, FireClass, ForwardPlan, NormVariant, RopeKind, Shape, StateRef, StateStore,
    TraceBuilder,
};

/// The shared tape a trace-in-progress records onto.
#[derive(Clone)]
pub struct Trace {
    inner: Rc<RefCell<TraceBuilder>>,
}

impl Trace {
    fn with<T>(&self, layer: Option<u32>, f: impl FnOnce(&mut TraceBuilder) -> T) -> T {
        let mut b = self.inner.borrow_mut();
        b.set_layer(layer);
        f(&mut b)
    }
}

/// A traced value: an SSA id carrying its tape and the layer of its
/// producer (the tag weightless ops inherit).
#[derive(Clone)]
pub struct Val {
    t: Trace,
    pub(crate) id: crate::trace::ValueId,
    layer: Option<u32>,
}

/// A matmul weight handle: name, out width, layer. Built by the
/// [`Layer`] namespace; the declaration never spells either.
#[derive(Clone)]
pub struct MatW {
    pub name: String,
    pub width: u32,
    pub layer: Option<u32>,
}

/// A norm weight handle. `per_head` carries the head_dim when this
/// weight's convention is per-head (qwen3-style qk-norm) — [`rmsnorm`]
/// dispatches on it, which is how `rmsnorm(q, &w.q_norm)` needs no
/// variant arguments: THE WEIGHT KNOWS.
#[derive(Clone)]
pub struct NormW {
    pub name: String,
    pub variant: NormVariant,
    pub per_head: Option<u32>,
    pub layer: Option<u32>,
}

/// A layer's paged KV cache.
#[derive(Clone)]
pub struct Kv {
    t: Trace,
    pub l: u32,
}

impl Kv {
    /// The layer's cache handle, for weight namespaces built outside
    /// [`M::layer`] (the qwen3_5 fragments build their own).
    pub(crate) fn at(t: &Trace, l: u32) -> Kv {
        Kv { t: t.clone(), l }
    }

    pub fn append(&self, k: &Val, v: &Val) {
        self.t
            .with(Some(self.l), |b| b.kv_append(self.l, k.id, v.id));
    }
}

/// A depthwise causal-conv weight handle: name, kernel width, layer. The
/// conv addresses the layer's per-request conv state, so the handle's
/// layer is both the op's tag and the state it touches — a `u32`, not an
/// `Option`: a conv never records outside a layer.
#[derive(Clone)]
pub struct ConvW {
    pub name: String,
    pub kernel: u32,
    pub layer: u32,
}

/// The GDN prep's weight pair (`a_log` + `dt_bias`) — one op, two names,
/// so one handle carries both, plus the layer tag they share.
#[derive(Clone)]
pub struct GdnPrepW {
    pub a_log: String,
    pub dt_bias: String,
    pub layer: u32,
}

/// A layer's per-request recurrent state (the GDN delta-rule store) —
/// the [`Kv`] of linear attention. The state is not a traced value
/// (see the trace module doc's "the per-request state axis"); the
/// handle exists so [`gated_delta`] can derive its layer the way
/// [`attention`] does from [`Kv`].
#[derive(Clone)]
pub struct Rs {
    t: Trace,
    pub l: u32,
}

impl Rs {
    /// The layer's recurrent-state handle, [`Kv::at`]-style.
    pub(crate) fn at(t: &Trace, l: u32) -> Rs {
        Rs { t: t.clone(), l }
    }
}

/// The per-layer weight namespace plus the layer's state, built by
/// [`M::layer`]. Handles are eager (a handful of strings); nothing is
/// interned until an op actually touches it, so unused handles cost
/// nothing and configs that never bind (say) `qkv` simply never read it.
pub struct Layer {
    pub qkv: MatW,
    pub q_proj: MatW,
    pub k_proj: MatW,
    pub v_proj: MatW,
    pub o_proj: MatW,
    pub gate_up: MatW,
    pub down: MatW,
    pub attn_norm: NormW,
    pub mlp_norm: NormW,
    pub q_norm: NormW,
    pub k_norm: NormW,
    pub kv: Kv,
}

/// The model context a declaration runs against: facts, the optional
/// lowering (backend facts + fire class), and the tape.
pub struct M {
    t: Trace,
    f: LlamaLikeFacts,
    lower: Option<(LlamaLikeCudaFacts, FireClass)>,
}

impl M {
    pub fn facts(&self) -> &LlamaLikeFacts {
        &self.f
    }

    /// The lowering in hand, if this is a lowered trace: the backend
    /// facts and the fire class the declaration's class arms match on.
    pub fn lowering(&self) -> Option<(&LlamaLikeCudaFacts, FireClass)> {
        self.lower.as_ref().map(|(c, class)| (c, *class))
    }

    pub fn embed(&self) -> Val {
        let id = self.t.with(None, |b| b.embed("embed", self.f.hidden));
        Val {
            t: self.t.clone(),
            id,
            layer: None,
        }
    }

    pub fn layer(&self, l: u32) -> Layer {
        let f = &self.f;
        let w = |name: &str| format!("layer.{l}.{name}");
        let mat = |name: &str, width: u32| MatW {
            name: w(name),
            width,
            layer: Some(l),
        };
        let row_norm = |name: &str| NormW {
            name: w(name),
            variant: f.norm_variant,
            per_head: None,
            layer: Some(l),
        };
        // The qk-norm handles carry the convention the facts state:
        // per-head (qwen3), row over the flattened projection (olmo2's
        // Global), or — under QkNorm::Off — handles no arm ever touches.
        let qk_norm = |name: &str| NormW {
            name: w(name),
            variant: f.norm_variant,
            per_head: (f.qk_norm == QkNorm::PerHead).then_some(f.head_dim),
            layer: Some(l),
        };
        Layer {
            qkv: mat("qkv", f.q_width() + 2 * f.kv_width()),
            q_proj: mat("q_proj", f.q_width()),
            k_proj: mat("k_proj", f.kv_width()),
            v_proj: mat("v_proj", f.kv_width()),
            o_proj: mat("o_proj", f.hidden),
            gate_up: mat("gate_up", 2 * f.intermediate),
            down: mat("down", f.hidden),
            attn_norm: row_norm("attn_norm"),
            mlp_norm: row_norm("mlp_norm"),
            q_norm: qk_norm("q_norm"),
            k_norm: qk_norm("k_norm"),
            kv: Kv {
                t: self.t.clone(),
                l,
            },
        }
    }

    pub fn final_norm(&self) -> NormW {
        NormW {
            name: "final_norm".to_string(),
            variant: self.f.norm_variant,
            per_head: None,
            layer: None,
        }
    }

    /// The epilogue: gather the sampled rows and project to logits
    /// (`OpKind::LmHead`, resolving the tied-embedding fact).
    pub fn logits(&self, x: &Val) {
        let name = if self.f.tied_embeddings {
            "embed"
        } else {
            "lm_head"
        };
        self.t.with(None, |b| b.lm_head(x.id, name, self.f.vocab));
    }
}

/// Run a declaration and return its traced form. `lower: None` is the
/// semantic trace; with a lowering the class arms run and the family
/// name records which class this launch form serves.
pub fn trace(
    facts: &LlamaLikeFacts,
    lower: Option<(&LlamaLikeCudaFacts, FireClass)>,
    body: impl FnOnce(&mut M),
) -> ForwardPlan {
    let family = match &lower {
        None => "llama_like".to_string(),
        Some((_, class)) => format!(
            "llama_like.cuda.{}",
            match class {
                FireClass::Decode => "decode",
                FireClass::Prefill => "prefill",
            }
        ),
    };
    let mut m = M {
        t: Trace {
            inner: Rc::new(RefCell::new(TraceBuilder::new(family))),
        },
        f: facts.clone(),
        lower: lower.map(|(c, class)| (c.clone(), class)),
    };
    body(&mut m);
    Rc::try_unwrap(m.t.inner)
        .ok()
        .expect("declaration must not hold values past its body")
        .into_inner()
        .finish()
}

/// Run a declaration under an explicit family name and return its traced
/// form — the entry for families that are not llama_like-shaped (the
/// qwen3_5 fragments and hybrid), which carry their own facts and build
/// their own weight namespaces rather than going through [`M`].
pub fn trace_named(family: &str, body: impl FnOnce(&Trace)) -> ForwardPlan {
    let t = Trace {
        inner: Rc::new(RefCell::new(TraceBuilder::new(family))),
    };
    body(&t);
    Rc::try_unwrap(t.inner)
        .ok()
        .expect("declaration must not hold values past its body")
        .into_inner()
        .finish()
}

/// Declare a fragment parameter ([`TraceBuilder::input`]): the residual
/// stream entering the block, `[Tokens, hidden]` bf16, produced by no op
/// (layer `None` — the first op that consumes it takes its tag from its
/// own weight handle).
pub fn input(t: &Trace, hidden: u32) -> Val {
    let id = t.with(None, |b| {
        b.input(Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)
    });
    Val {
        t: t.clone(),
        id,
        layer: None,
    }
}

/// The embedding gather under an explicit weight name — [`M::embed`] for
/// traces that run without an [`M`].
pub fn embed_with(t: &Trace, weight: &str, hidden: u32) -> Val {
    let id = t.with(None, |b| b.embed(weight, hidden));
    Val {
        t: t.clone(),
        id,
        layer: None,
    }
}

/// The epilogue under an explicit weight name — [`M::logits`] for traces
/// that run without an [`M`] (the caller resolves the tied-embedding
/// fact to a name).
pub fn lm_head_at(t: &Trace, x: &Val, weight: &str, vocab: u32) {
    t.with(None, |b| {
        b.lm_head(x.id, weight, vocab);
    });
}

// ── The semantic vocabulary, as free functions ─────────────────────────

pub fn matmul(x: &Val, w: &MatW) -> Val {
    let id = x.t.with(w.layer, |b| b.matmul(x.id, &w.name, w.width));
    Val {
        t: x.t.clone(),
        id,
        layer: w.layer,
    }
}

/// Row RMSNorm, or per-head RMSNorm when the weight handle says so.
pub fn rmsnorm(x: &Val, w: &NormW) -> Val {
    let id = x.t.with(w.layer, |b| match w.per_head {
        None => b.rmsnorm(x.id, &w.name, w.variant),
        Some(head_dim) => b.rmsnorm_per_head(x.id, &w.name, head_dim, w.variant),
    });
    Val {
        t: x.t.clone(),
        id,
        layer: w.layer,
    }
}

pub fn split_qkv(x: &Val, q_width: u32, kv_width: u32) -> (Val, Val, Val) {
    let (q, k, v) = x
        .t
        .with(x.layer, |b| b.split_qkv(x.id, q_width, kv_width));
    let mk = |id| Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    };
    (mk(q), mk(k), mk(v))
}

pub fn rope(q: &Val, k: &Val, kind: RopeKind) -> (Val, Val) {
    let (qo, ko) = q.t.with(q.layer, |b| b.rope(q.id, k.id, kind));
    let mk = |id| Val {
        t: q.t.clone(),
        id,
        layer: q.layer,
    };
    (mk(qo), mk(ko))
}

pub fn swiglu(x: &Val, inter: u32) -> Val {
    let id = x.t.with(x.layer, |b| b.swiglu(x.id, inter));
    Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    }
}

/// Paged attention over the layer's cache — the SEMANTIC form, opaque.
/// A lowered arm states a kernel from [`cuda`] instead.
pub fn attention(q: &Val, kv: &Kv, q_width: u32) -> Val {
    let id = q.t.with(Some(kv.l), |b| b.attention(kv.l, q.id, q_width));
    Val {
        t: q.t.clone(),
        id,
        layer: Some(kv.l),
    }
}

/// The expert-indexed matmul ([`TraceBuilder::matmul_per_token`]): `w` is
/// an `{e}`-templated bank handle and `selector` a [`topk`] index value.
/// Layer from the weight handle, like [`matmul`].
pub fn matmul_per_token(x: &Val, w: &MatW, selector: &Val) -> Val {
    let id = x.t.with(w.layer, |b| {
        b.matmul_per_token(x.id, &w.name, selector.id, w.width)
    });
    Val {
        t: x.t.clone(),
        id,
        layer: w.layer,
    }
}

/// Router top-k: `(indices, weights)`, softmaxed and renormalized
/// ([`TraceBuilder::topk`]). Weightless — layer from the logits.
pub fn topk(logits: &Val, k: u32) -> (Val, Val) {
    let (idx, w) = logits.t.with(logits.layer, |b| b.topk(logits.id, k));
    let mk = |id| Val {
        t: logits.t.clone(),
        id,
        layer: logits.layer,
    };
    (mk(idx), mk(w))
}

/// The top-k combine: collapse `x` under per-token `weights`
/// ([`TraceBuilder::weighted_sum`] — operand order weights-then-value is
/// the builder's). Layer from `weights`, the first input.
pub fn weighted_sum(weights: &Val, x: &Val) -> Val {
    let id = weights
        .t
        .with(weights.layer, |b| b.weighted_sum(weights.id, x.id));
    Val {
        t: weights.t.clone(),
        id,
        layer: weights.layer,
    }
}

/// The shared-expert landing: `base + sigmoid(gate) * x`. Layer from
/// `x`, the first input.
pub fn sigmoid_gate_add(x: &Val, gate: &Val, base: &Val) -> Val {
    let id = x
        .t
        .with(x.layer, |b| b.sigmoid_gate_add(x.id, gate.id, base.id));
    Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    }
}

/// The multiply-only output gate: `x * sigmoid(gate)`. Layer from `x`.
pub fn sigmoid_gate_mul(x: &Val, gate: &Val) -> Val {
    let id = x.t.with(x.layer, |b| b.sigmoid_gate_mul(x.id, gate.id));
    Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    }
}

/// The two-way GDN row split of a packed projection. Layer from `x`.
pub fn split_gdn(x: &Val, w0: u32, w1: u32) -> (Val, Val) {
    let (a, b) = x.t.with(x.layer, |b| b.split_gdn(x.id, w0, w1));
    let mk = |id| Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    };
    (mk(a), mk(b))
}

/// The interleaved per-head `[query | gate]` split of a 2×-wide gated q
/// projection. Layer from `x`.
pub fn split_q_gate(x: &Val, heads: u32, head_dim: u32) -> (Val, Val) {
    let (q, gate) = x
        .t
        .with(x.layer, |b| b.split_q_gate(x.id, heads, head_dim));
    let mk = |id| Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    };
    (mk(q), mk(gate))
}

/// The partial-rotary rope: only the first `rotary_dim` channels of each
/// head rotate. Layer from `q`, [`rope`]-style.
pub fn rope_partial(q: &Val, k: &Val, kind: RopeKind, rotary_dim: u32) -> (Val, Val) {
    let (qo, ko) = q
        .t
        .with(q.layer, |b| b.rope_partial(q.id, k.id, kind, rotary_dim));
    let mk = |id| Val {
        t: q.t.clone(),
        id,
        layer: q.layer,
    };
    (mk(qo), mk(ko))
}

/// Depthwise causal conv1d (+ fused SiLU) against the handle's layer and
/// that layer's per-request conv state. Layer from the weight handle.
pub fn causal_conv1d(x: &Val, w: &ConvW) -> Val {
    let id = x.t.with(Some(w.layer), |b| {
        b.causal_conv1d(w.layer, x.id, &w.name, w.kernel)
    });
    Val {
        t: x.t.clone(),
        id,
        layer: Some(w.layer),
    }
}

/// The post-conv GDN prep: `(q, k, v, g, beta)`, all f32, per-head
/// layouts from the four geometry params ([`TraceBuilder::gdn_prep`]).
/// Layer from the weight handle.
pub fn gdn_prep(
    qkv: &Val,
    a: &Val,
    b: &Val,
    w: &GdnPrepW,
    key_heads: u32,
    key_dim: u32,
    value_heads: u32,
    value_dim: u32,
) -> (Val, Val, Val, Val, Val) {
    let out = qkv.t.with(Some(w.layer), |bld| {
        bld.gdn_prep(
            qkv.id, a.id, b.id, &w.a_log, &w.dt_bias, key_heads, key_dim, value_heads, value_dim,
        )
    });
    let mk = |id| Val {
        t: qkv.t.clone(),
        id,
        layer: Some(w.layer),
    };
    (mk(out.0), mk(out.1), mk(out.2), mk(out.3), mk(out.4))
}

/// The gated-delta recurrence against the layer's per-request recurrent
/// state. Layer from the state handle, [`attention`]-style.
pub fn gated_delta(rs: &Rs, q: &Val, k: &Val, v: &Val, g: &Val, beta: &Val) -> Val {
    let id = rs.t.with(Some(rs.l), |b| {
        b.gated_delta(rs.l, q.id, k.id, v.id, g.id, beta.id)
    });
    Val {
        t: rs.t.clone(),
        id,
        layer: Some(rs.l),
    }
}

/// The gated RMSNorm landing: per-head norm of the rank-3 f32 core,
/// silu-gated by `gate`, flattened to `gate`'s bf16 shape. The op's fold
/// is Plain by construction ([`TraceBuilder::rmsnorm_gated`] takes only
/// the weight name), so the handle's `variant`/`per_head` are unread —
/// only its name and layer speak.
pub fn rmsnorm_gated(x: &Val, gate: &Val, w: &NormW) -> Val {
    let id = x
        .t
        .with(w.layer, |b| b.rmsnorm_gated(x.id, gate.id, &w.name));
    Val {
        t: x.t.clone(),
        id,
        layer: w.layer,
    }
}

// ── `+=`: the residual landing ─────────────────────────────────────────

impl std::ops::AddAssign<Val> for Val {
    /// `y += rhs`. If `rhs` is the op just recorded and it is a plain
    /// matmul nobody else consumed, rewrite it to the `beta_one`
    /// accumulate (the cuBLAS fold — id-neutral, so dataflow never sees
    /// the difference). Otherwise record the explicit
    /// [`OpKind::ResidualAdd`] launch, the post-norm landing.
    fn add_assign(&mut self, rhs: Val) {
        let folded_or_added = rhs.t.with(rhs.layer, |b| {
            if b.try_fold_residual(rhs.id, self.id) {
                rhs.id
            } else {
                b.residual_add(rhs.id, self.id)
            }
        });
        self.id = folded_or_added;
        self.layer = rhs.layer;
    }
}

// ── The runtime branch ─────────────────────────────────────────────────

/// Record a [`OpKind::Guard`]: `then_f`'s ops run when `pred` holds at
/// fire time, `else_f`'s when it does not. The ONE branch a lowered
/// declaration may write over a runtime input — the predicate vocabulary
/// is closed ([`GuardPred`]), the regions are flat, and neither region's
/// values may escape (side-effect launches only; the builder has no way
/// to stop a determined escape yet, so the discipline is reviewed, not
/// enforced — a value-producing guard is a later design).
pub fn guard(m: &M, pred: crate::trace::GuardPred, then_f: impl FnOnce(), else_f: impl FnOnce()) {
    let idx = {
        let mut b = m.t.inner.borrow_mut();
        b.set_layer(None);
        b.open_guard(pred)
    };
    then_f();
    let then_ops = {
        let b = m.t.inner.borrow();
        (b.op_count_now() - idx - 1) as u32
    };
    else_f();
    let mut b = m.t.inner.borrow_mut();
    let else_ops = (b.op_count_now() - idx - 1) as u32 - then_ops;
    b.close_guard(idx, then_ops, else_ops);
}

// ── Raw kernel signatures ──────────────────────────────────────────────

/// The CUDA launchers a lowered declaration may state, one function per
/// kernel, PARAMETERS = the launcher's semantic operands (tensors,
/// weights, state, tables). Mechanical parameters — stream, dims,
/// workspace scratch, plan caches — are the driver's binding, not a
/// choice, and do not appear. Each records one [`OpKind::Launch`] (or
/// the exact launch pair the hand-written arm makes); the doc comment is
/// the contract naming the C++ symbol.
///
/// Prepare-phase host work (decode-plan builds, XQA's fire-wide prepare)
/// is NOT stated here: the trace states the BODY's launches, and a
/// stated kernel obligates the driver to whatever prepare its contract
/// needs — the same prepare/body seam the graph work built.
pub mod cuda {
    use super::*;

    fn record(
        t: &Trace,
        layer: Option<u32>,
        kernel: &str,
        weights: Vec<String>,
        state: Option<StateRef>,
        inputs: Vec<crate::trace::ValueId>,
        out: Option<(Shape, DType)>,
    ) -> Option<Val> {
        let ids = t.with(layer, |b| {
            b.launch(kernel, weights, state, inputs, out.into_iter().collect())
        });
        ids.first().map(|&id| Val {
            t: t.clone(),
            id,
            layer,
        })
    }

    fn kv_state(kv: &Kv) -> Option<StateRef> {
        Some(StateRef {
            store: StateStore::KvCache,
            layer: kv.l,
        })
    }

    /// `kernels::launch_rope_standard_table`: build the fire's cos/sin
    /// table, once. A value, not a latch — the fused-QKV kernel consumes
    /// it as an operand.
    pub fn rope_standard_table(m: &M) -> Val {
        record(
            &m.t,
            None,
            "launch_rope_standard_table",
            vec![],
            None,
            vec![],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(m.facts().head_dim)]),
                DType::F32,
            )),
        )
        .expect("table launch produces a value")
    }

    /// `kernels::launch_qkv_decode_qk_norm_rope_write_kv_bf16`: the fused
    /// decode-QKV epilogue — split + per-head Plain q/k norms + Standard
    /// rope + KV append, one launch. Packed GEMM output in, roped Q out;
    /// K/V go straight to the cache and never exist as values. The
    /// general arm (`split_qkv` + `rmsnorm`×2 + `rope` + `Kv::append`) is
    /// this call's semantics, and the parity harness holds it there.
    pub fn qkv_decode_qk_norm_rope_write_kv(
        packed: &Val,
        q_norm: &NormW,
        k_norm: &NormW,
        kv: &Kv,
        table: Option<&Val>,
        q_width: u32,
    ) -> Val {
        let mut inputs = vec![packed.id];
        inputs.extend(table.map(|t| t.id));
        record(
            &packed.t,
            Some(kv.l),
            "launch_qkv_decode_qk_norm_rope_write_kv_bf16",
            vec![q_norm.name.clone(), k_norm.name.clone()],
            kv_state(kv),
            inputs,
            Some((Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16)),
        )
        .expect("fused post produces q")
    }

    /// `kernels::launch_qk_rmsnorm_rope_bf16`: the fused per-head q/k
    /// norm + Standard rope, one launch — the hand-written
    /// `fuse_qk_norm_rope` branch. bf16 rounding differs between this
    /// kernel and the norm+rope triple it replaces, so parity requires
    /// stating it wherever the hand-written path takes it: every lowered
    /// arm with per-head qk-norm and Standard rope that did not take the
    /// fully-fused decode post. In place on q and k; SSA-wise two fresh
    /// values.
    pub fn qk_rmsnorm_rope(q: &Val, k: &Val, q_norm: &NormW, k_norm: &NormW) -> (Val, Val) {
        let ids = q.t.with(q.layer, |b| {
            let q_sh = b.value_shape(q.id);
            let k_sh = b.value_shape(k.id);
            b.launch(
                "launch_qk_rmsnorm_rope_bf16",
                vec![q_norm.name.clone(), k_norm.name.clone()],
                None,
                vec![q.id, k.id],
                vec![(q_sh, DType::BF16), (k_sh, DType::BF16)],
            )
        });
        let mk = |id| Val {
            t: q.t.clone(),
            id,
            layer: q.layer,
        };
        (mk(ids[0]), mk(ids[1]))
    }

    /// `ops::launch_attention_xqa_decode_bf16_prepared` (whose contract
    /// includes the fire-wide XQA prepare).
    pub fn attention_xqa_decode(q: &Val, kv: &Kv, q_width: u32) -> Val {
        attn(q, kv, q_width, "launch_attention_xqa_decode_bf16_prepared")
    }

    /// `ops::dispatch_attention_flashinfer_decode` against the decode
    /// plan its contract obligates.
    pub fn attention_flashinfer_decode(q: &Val, kv: &Kv, q_width: u32) -> Val {
        attn(q, kv, q_width, "dispatch_attention_flashinfer_decode")
    }

    /// The decode-shaped fallback for GQA ratios outside the decode
    /// kernel set (`force_prefill_path`): the exact pair the hand-written
    /// arm launches — `kernels::launch_dequant_kv_cache_layer_to_bf16_active`
    /// then `ops::dispatch_attention_flashinfer_prefill_bf16`.
    pub fn attention_prefill_dequant(q: &Val, kv: &Kv, q_width: u32) -> Val {
        dequant(kv);
        attn(q, kv, q_width, "dispatch_attention_flashinfer_prefill_bf16")
    }

    /// The planned prefill every prefill-shaped fire runs: the same
    /// dequant + `ops::dispatch_attention_flashinfer_prefill_bf16` pair.
    pub fn attention_flashinfer_prefill(q: &Val, kv: &Kv, q_width: u32) -> Val {
        dequant(kv);
        attn(q, kv, q_width, "dispatch_attention_flashinfer_prefill_bf16")
    }

    /// `kernels::launch_write_kv_explicit_bf16`: the explicit-descriptor
    /// KV write (graph-replay steering; N cells, one per query token).
    /// Stated inside the `HasWriteDesc` guard's then-region.
    pub fn write_kv_explicit(k: &Val, v: &Val, kv: &Kv) {
        record(
            &kv.t,
            Some(kv.l),
            "launch_write_kv_explicit_bf16",
            vec![],
            kv_state(kv),
            vec![k.id, v.id],
            None,
        );
    }

    /// `kernels::launch_write_kv_to_pages`: the page-derived append
    /// (position re-derived from the page table). The `HasWriteDesc`
    /// guard's else-region.
    pub fn write_kv_to_pages(k: &Val, v: &Val, kv: &Kv) {
        record(
            &kv.t,
            Some(kv.l),
            "launch_write_kv_to_pages",
            vec![],
            kv_state(kv),
            vec![k.id, v.id],
            None,
        );
    }

    fn dequant(kv: &Kv) {
        record(
            &kv.t,
            Some(kv.l),
            "launch_dequant_kv_cache_layer_to_bf16_active",
            vec![],
            kv_state(kv),
            vec![],
            None,
        );
    }

    fn attn(q: &Val, kv: &Kv, q_width: u32, kernel: &str) -> Val {
        record(
            &q.t,
            Some(kv.l),
            kernel,
            vec![],
            kv_state(kv),
            vec![q.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16)),
        )
        .expect("attention produces a value")
    }
}
