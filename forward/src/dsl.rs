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

use crate::facts::{LlamaLikeFacts, QkNorm};
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
    pub q_bias: MatW,
    pub k_bias: MatW,
    pub v_bias: MatW,
    pub o_proj: MatW,
    pub gate_up: MatW,
    pub down: MatW,
    pub attn_norm: NormW,
    pub mlp_norm: NormW,
    pub q_norm: NormW,
    pub k_norm: NormW,
    pub kv: Kv,
}

/// The model context a declaration runs against: the facts and the tape.
///
/// It carries NO lowering. A model text is written for one backend
/// (`.wiki/tart/dsl.md` ③: the model file is
/// `families/<family>/<backend>.rs`), so "am I lowered?" is not a
/// question a body can ask — the semantic text and the CUDA text are two
/// texts, and each states its own kernels unconditionally. What used to
/// be `m.lowering()` is now the CUDA text's own parameters.
pub struct M {
    t: Trace,
    f: LlamaLikeFacts,
}

impl M {
    pub fn facts(&self) -> &LlamaLikeFacts {
        &self.f
    }

    /// The tape, for the few call sites (value-producing guards) that
    /// need it directly.
    pub fn trace(&self) -> &Trace {
        &self.t
    }

    /// V2 rung ②: the body STATES the depth axis (with its deployment
    /// gate beside it, like the mask peel's) instead of a caller
    /// painting roles on after the trace. Must precede the layer loop;
    /// recording assigns each layer-tagged op's [`DepthRole`] — the
    /// flashinfer decode dispatch swaps to the depth prefix plan on
    /// union tail layers, everything else windows.
    ///
    /// [`DepthRole`]: crate::trace::DepthRole
    pub fn depth_window(&self) {
        self.t.with(None, |b| b.declare_depth_window());
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
            q_bias: mat("q_bias", f.q_width()),
            k_bias: mat("k_bias", f.kv_width()),
            v_bias: mat("v_bias", f.kv_width()),
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

/// Run the SEMANTIC llama_like declaration: no kernel is stated, and the
/// consumer (Metal, the site table, `declared_dag`) chooses.
pub fn trace_semantic(facts: &LlamaLikeFacts, body: impl FnOnce(&mut M)) -> ForwardPlan {
    run("llama_like".to_string(), facts, body)
}

/// Run a LOWERED llama_like declaration — one per [`FireClass`], the
/// family name recording which launch form this trace serves. The body
/// takes the backend facts as its own parameter; nothing about the
/// lowering reaches it through [`M`].
pub fn trace_cuda(
    facts: &LlamaLikeFacts,
    class: FireClass,
    body: impl FnOnce(&mut M),
) -> ForwardPlan {
    let family = format!(
        "llama_like.cuda.{}",
        match class {
            FireClass::Decode => "decode",
            FireClass::Prefill => "prefill",
            // The service classes are qwen3_5's; llama_like has no
            // spec-decode repair pass. The ffi entry rejects them
            // before tracing; this is the same statement for direct
            // Rust callers.
            FireClass::CommitAdvance | FireClass::StateOnly | FireClass::FrozenVerify => {
                panic!("llama_like has no MTP service classes")
            }
        }
    );
    run(family, facts, body)
}

fn run(family: String, facts: &LlamaLikeFacts, body: impl FnOnce(&mut M)) -> ForwardPlan {
    let mut m = M {
        t: Trace {
            inner: Rc::new(RefCell::new(TraceBuilder::new(family))),
        },
        f: facts.clone(),
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

/// Broadcast bias add (`OpKind::AddBias`): `x[r, :] += bias`. The Qwen-2
/// family's qkv biases; the kernel is 1:1 so semantic and lowered traces
/// state the same op.
pub fn add_bias(x: &Val, w: &MatW) -> Val {
    let id = x.t.with(w.layer, |b| b.add_bias(x.id, &w.name));
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

/// An open [`OpKind::Guard`] chain: `.arm(pred, f)` regions are tried in
/// order at fire time, `.otherwise(f)` closes the chain with the
/// fallback region. The ONE branch a lowered declaration may write over
/// runtime inputs — the predicate vocabulary is closed ([`GuardPred`]),
/// the regions are flat and consecutive, and a region's OWN values may
/// not escape (its launches are lowerings of the guard's outputs, which
/// [`guarded_value`] created up front; the discipline is reviewed, not
/// enforced).
#[must_use = "a guard chain must be closed with .otherwise(..)"]
pub struct GuardCtx {
    t: Trace,
    idx: usize,
    arms: Vec<crate::trace::GuardArm>,
    emitted: u32,
}

impl GuardCtx {
    pub fn arm(mut self, pred: crate::trace::GuardPred, f: impl FnOnce()) -> Self {
        f();
        let total = {
            let b = self.t.inner.borrow();
            (b.op_count_now() - self.idx - 1) as u32
        };
        self.arms.push(crate::trace::GuardArm {
            pred,
            ops: total - self.emitted,
        });
        self.emitted = total;
        self
    }

    pub fn otherwise(self, f: impl FnOnce()) {
        f();
        let mut b = self.t.inner.borrow_mut();
        let total = (b.op_count_now() - self.idx - 1) as u32;
        b.close_guard(self.idx, self.arms, total - self.emitted);
    }
}

/// Open a side-effect-only guard chain.
pub fn guarded(m: &M) -> GuardCtx {
    guarded_on(&m.t)
}

pub(crate) fn guarded_on(t: &Trace) -> GuardCtx {
    let idx = {
        let mut b = t.inner.borrow_mut();
        b.set_layer(None);
        b.open_guard(vec![]).0
    };
    GuardCtx {
        t: t.clone(),
        idx,
        arms: Vec::new(),
        emitted: 0,
    }
}

/// Open a VALUE-PRODUCING guard chain: the returned [`Val`]s are the
/// guard's outputs — one producer whichever arm runs — and each region's
/// launches are their lowerings, binding the same output buffer and
/// recording no SSA outputs of their own.
pub fn guarded_value(t: &Trace, layer: Option<u32>, shape: (Shape, DType)) -> (GuardCtx, Val) {
    let (idx, outs) = {
        let mut b = t.inner.borrow_mut();
        b.set_layer(layer);
        b.open_guard(vec![shape])
    };
    (
        GuardCtx {
            t: t.clone(),
            idx,
            arms: Vec::new(),
            emitted: 0,
        },
        Val {
            t: t.clone(),
            id: outs[0],
            layer,
        },
    )
}

/// Two-way sugar over [`GuardCtx`] — the 4a form llama_like writes.
pub fn guard(m: &M, pred: crate::trace::GuardPred, then_f: impl FnOnce(), else_f: impl FnOnce()) {
    guarded(m).arm(pred, then_f).otherwise(else_f);
}

// ── The row partition ──────────────────────────────────────────────────

/// WHICH ROWS of the fire an arm's statements cover
/// (`.wiki/tart/dsl.md` ③'s `rows!(..)`).
///
/// A row predicate is not a deployment condition: it does not resolve at
/// trace time and vanish, it PARTITIONS the fire. Today's tree writes
/// both kinds as plain Rust `if`, which is why a reader cannot tell
/// which one disappears — naming the row kind is the first half of
/// fixing that.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowPred {
    /// Rows with nothing attached at a seam — the hook-free prefix.
    HookFree,
    /// Rows carrying no custom mask.
    Unmasked,
}

impl RowPred {
    /// The axis word today's IR carries. It is DERIVED here rather than
    /// passed: the arm's predicate already says which rows it covers, so
    /// stating the axis beside it was the same fact twice.
    fn window(self) -> crate::trace::PeelWindow {
        match self {
            RowPred::HookFree => crate::trace::PeelWindow::HookFreePrefix,
            RowPred::Unmasked => crate::trace::PeelWindow::UnmaskedPrefix,
        }
    }
}

/// The arms of a [`by_rows`] partition.
///
/// Each arm's statements record as the arm is written — the construct is
/// already open — and the axis word the IR carries is patched in at
/// close from the arm's predicate.
#[must_use = "a row partition must be closed with .rest(..)"]
pub struct RowsCtx<'t> {
    t: &'t Trace,
    idx: usize,
    prefix: Option<u32>,
    pred: Option<RowPred>,
}

impl RowsCtx<'_> {
    /// The rows `pred` names, and what runs over them.
    pub fn arm(&mut self, pred: RowPred, f: impl FnOnce()) {
        assert!(
            self.pred.is_none(),
            "by_rows takes one arm and a rest today — the IR's Peel is a \
             two-region op (`.wiki/tart/dsl.md` migration step 6 flattens it)"
        );
        f();
        let b = self.t.inner.borrow();
        self.prefix = Some((b.op_count_now() - self.idx - 1) as u32);
        drop(b);
        self.pred = Some(pred);
    }

    /// Every other row.
    pub fn rest(&mut self, f: impl FnOnce()) {
        let prefix = self
            .prefix
            .expect("a row partition states its arm before its rest");
        f();
        let mut b = self.t.inner.borrow_mut();
        let total = (b.op_count_now() - self.idx - 1) as u32;
        b.close_peel(self.idx, prefix, total - prefix);
        b.set_peel_window(
            self.idx,
            self.pred.expect("an arm was stated").window(),
        );
    }
}

/// THE row-partition construct (`.wiki/tart/dsl.md` ③'s `t.by_rows`):
/// the arms' statements each cover their own rows and ALL of them run,
/// which is what separates this from the fire-level [`GuardCtx`] chain
/// (first matching arm wins, whole fire).
///
/// `shape` present makes the partition value-producing: the [`Val`] is
/// the construct's, and each region's launches bind disjoint row windows
/// of it, recording no SSA outputs of their own.
///
/// It lowers to today's [`OpKind::Peel`] — one axis word, two regions —
/// so the goldens pin that this surface changed no traced byte. What it
/// removes is the axis word from the call site: `peel` and `peel_masked`
/// were two functions naming the same mechanism over two axes, and the
/// axis is now read off the arm's predicate.
///
/// [`OpKind::Peel`]: crate::trace::OpKind::Peel
pub fn by_rows(
    t: &Trace,
    layer: Option<u32>,
    shape: Option<(Shape, DType)>,
    build: impl FnOnce(&mut RowsCtx<'_>),
) -> Option<Val> {
    let (idx, outs) = {
        let mut b = t.inner.borrow_mut();
        b.set_layer(layer);
        // The axis is patched at close, once the arm has named it.
        b.open_peel(
            shape.into_iter().collect(),
            crate::trace::PeelWindow::HookFreePrefix,
        )
    };
    let mut ctx = RowsCtx {
        t,
        idx,
        prefix: None,
        pred: None,
    };
    build(&mut ctx);
    assert!(
        ctx.pred.is_some(),
        "a row partition must state an arm and a rest"
    );
    outs.first().map(|&id| Val {
        t: t.clone(),
        id,
        layer,
    })
}

/// [`guard`] for declarations that carry no [`M`] (the qwen3_5 bodies
/// build their own weight namespaces and run against a bare [`Trace`]):
/// the same two-way chain, opened on the tape directly.
pub(crate) fn guard_on(
    t: &Trace,
    pred: crate::trace::GuardPred,
    then_f: impl FnOnce(),
    else_f: impl FnOnce(),
) {
    guarded_on(t).arm(pred, then_f).otherwise(else_f);
}

/// Record a [`OpKind::HookSite`] (the HookSite slice): the layer's
/// attached programs run here at fire time, observing `q`. Since A2
/// (the class-collapse amendment) the sites live INSIDE the
/// `HasStageHooks` guard arm of the Decode/Prefill traces — the one
/// text carries them, and an unhooked fire's walk never reaches them,
/// which is the launch-list truth (the SITES' bracketing launches —
/// begin_layer, compact — exist only on hooked fires).
pub fn hook_site(stage: crate::trace::HookStage, q: &Val, layer: u32) {
    q.t.with(Some(layer), |b| {
        b.push_hook_site(stage, layer, q.id);
    });
}

// ── The seam surface (V2 rung ①) ───────────────────────────────────────

/// V2 (north-star-dsl.md "V2 — THE REDESIGN"): the seam vocabulary.
///
/// A seam is a named, typed, identity-by-default extension point in the
/// model text — the ONE surface behind what were three mechanisms (the
/// two [`crate::trace::HookStage`]s, the `HasLora` guard arm, and the
/// dispatch-side prologue/epilogue stages). At THIS rung only the
/// surface unifies: each seam lowers to exactly the op(s) the pre-seam
/// text recorded, and the goldens pin that byte-identity. The IR's own
/// Seam op and the signature-table ABI are later rungs; what changes
/// now is that the model text states extension points in one vocabulary
/// instead of naming mechanisms.
pub mod seam {
    /// What an attachment at a seam MAY do. Caps are the seam's
    /// interface; whether a given deployment can service a cap is a
    /// dispatch-table fact refused at load (the XQA-has-no-capture
    /// contract's future home — enforcement lands with the signature
    /// ABI). The vocabulary documents the gradient: pure expressions
    /// innermost, observation mid-body, full PTIR at the boundary.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Cap {
        /// Pure value rewrite `|x, y| y'` (the adapter family).
        Transform,
        /// Read the seam's value from an attached program.
        Observe,
        /// Read the attention scores the capturing dispatch published.
        Scores,
        /// Narrow the page list the stated attention kernel consumes.
        PageMaskSink,
        /// Device puts (embeds, channels) — boundary-only.
        Put,
        /// Draw samples from the logits — boundary-only.
        Sample,
        /// Emit host-visible outputs — boundary-only.
        Emit,
    }

    /// A seam's definition: the stable NAME the request surface keys on
    /// (`fwd.adapter("attn.qv", ..)`, `fwd.attach(..)`) and its caps.
    pub struct Def {
        pub name: &'static str,
        pub caps: &'static [Cap],
    }

    /// Pre-attention observation seam: sees the just-projected q; a
    /// page-mask-sink attachment narrows the page list the SAME stated
    /// attention kernel consumes as substituted arguments (today's
    /// `OnAttnProj`).
    pub const ATTN_Q: Def = Def {
        name: "attn.q",
        caps: &[Cap::Observe, Cap::PageMaskSink],
    };

    /// Post-attention observation seam: sees the scores the (possibly
    /// capturing) dispatch published through the sideband (today's
    /// `OnAttn`).
    pub const ATTN_OUT: Def = Def {
        name: "attn.out",
        caps: &[Cap::Observe, Cap::Scores],
    };

    /// The adapter value seam over the raw q/v projections — pure
    /// expressions of `(x, y)`, `fwd.adapter`'s site family (today's
    /// `HasLora` guard arm).
    pub const ATTN_QV: Def = Def {
        name: "attn.qv",
        caps: &[Cap::Transform],
    };

    /// Entry boundary seam (prologue's home). Boundary attachments
    /// never enter row signatures — they cause no divergence — which is
    /// why their dispatch-side lowering needs no trace op at any rung.
    pub const IN: Def = Def {
        name: "in",
        caps: &[Cap::Put, Cap::Emit],
    };

    /// Exit boundary seam (epilogue's home).
    pub const OUT: Def = Def {
        name: "out",
        caps: &[Cap::Observe, Cap::Sample, Cap::Put, Cap::Emit],
    };
}

/// An observation seam at `def`, watching `v` at `layer` — rung-①
/// lowering: exactly the [`OpKind::HookSite`] op the pre-seam text
/// recorded, so the traced form is byte-identical.
///
/// [`OpKind::HookSite`]: crate::trace::OpKind::HookSite
pub fn seam_observe(def: &seam::Def, v: &Val, layer: u32) {
    let stage = match def.name {
        "attn.q" => crate::trace::HookStage::OnAttnProj,
        "attn.out" => crate::trace::HookStage::OnAttn,
        other => unreachable!("no observation seam named {other}"),
    };
    hook_site(stage, v, layer);
}

/// The adapter value seam ([`seam::ATTN_QV`]) over the raw q/v
/// projections — rung-① lowering: exactly the `HasLora` guard with the
/// span-grouped correction arm and the EMPTY else (a fire with no
/// usable lanes launches nothing), byte-identical to the pre-seam text.
pub fn seam_adapter_qv(m: &M, q: &Val, v: &Val, layer: u32) {
    guard(
        m,
        crate::trace::GuardPred::HasLora,
        || cuda::lora_qkv_correction(q, v, layer),
        || {},
    );
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

    /// The GDN ops' state mark, [`kv_state`]-style: the layer's
    /// per-request conv/recurrent slabs.
    fn rs_state(rs: &Rs) -> Option<StateRef> {
        Some(StateRef {
            store: StateStore::RecurrentState,
            layer: rs.l,
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
    /// includes the fire-wide XQA prepare — and which is therefore
    /// declared `whole`; see [`crate::kernels`]).
    pub fn attention_xqa_decode(q: &Val, kv: &Kv) -> Option<Val> {
        attn_at(q, kv, "launch_attention_xqa_decode_bf16_prepared")
    }

    /// `ops::dispatch_attention_flashinfer_decode` against the decode
    /// plan its contract obligates.
    pub fn attention_flashinfer_decode(q: &Val, kv: &Kv) -> Option<Val> {
        attn_at(q, kv, "dispatch_attention_flashinfer_decode")
    }

    /// `ops::dispatch_attention_flashinfer_prefill_bf16` — the dispatch
    /// ALONE.
    ///
    /// Three wrappers used to differ here only by whether they also
    /// launched the dequant staging: llama_like's cache may be
    /// quantized, so its prefill-shaped arms dequant the layer first,
    /// while qwen3_5's full-attention path gates on a native-bf16 cache
    /// and launches only the dispatch. That is not a property of this
    /// kernel — it is a second STATEMENT the text either makes or does
    /// not, so the text makes it ([`dequant_only`] beside this call).
    pub fn attention_flashinfer_prefill(q: &Val, kv: &Kv) -> Option<Val> {
        attn_at(q, kv, "dispatch_attention_flashinfer_prefill_bf16")
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

    /// `kernels::launch_causal_conv1d_update_batched_bf16`: the
    /// slot-indirected decode conv update (+ fused SiLU) against the
    /// layer's per-request conv slab. Shape-preserving, like the
    /// semantic [`causal_conv1d`] it lowers.
    pub fn gdn_conv_update_batched(x: &Val, w: &ConvW, rs: &Rs) -> Val {
        gdn_conv(x, w, rs, "launch_causal_conv1d_update_batched_bf16")
    }

    /// `kernels::launch_causal_conv1d_prefill_batched_bf16`: the batched
    /// prefill conv walk (each request walking its qo_indptr window and
    /// persisting the trailing K-window into the slab).
    pub fn gdn_conv_prefill_batched(x: &Val, w: &ConvW, rs: &Rs) -> Val {
        gdn_conv(x, w, rs, "launch_causal_conv1d_prefill_batched_bf16")
    }

    fn gdn_conv(x: &Val, w: &ConvW, rs: &Rs, kernel: &str) -> Val {
        let ids = x.t.with(Some(w.layer), |b| {
            let shape = b.value_shape(x.id);
            b.launch(
                kernel,
                vec![w.name.clone()],
                rs_state(rs),
                vec![x.id],
                vec![(shape, DType::BF16)],
            )
        });
        Val {
            t: x.t.clone(),
            id: ids[0],
            layer: Some(w.layer),
        }
    }

    /// `kernels::launch_recurrent_gated_delta_step_batched[_gqa][_state_bf16]`:
    /// the one-token decode recurrence step against the layer's
    /// per-request recurrent state. `gqa` states the compact-K_h-indexing
    /// GQA variant (value heads != key heads); `state_bf16` the store
    /// dtype. Output = the semantic [`gated_delta`]'s: the core keeps v's
    /// `[Tokens, Vh, Vd]` f32 shape.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_step_batched(
        q: &Val,
        k: &Val,
        v: &Val,
        g: &Val,
        beta: &Val,
        rs: &Rs,
        gqa: bool,
        state_bf16: bool,
    ) -> Val {
        let kernel = match (gqa, state_bf16) {
            (true, true) => "launch_recurrent_gated_delta_step_batched_gqa_state_bf16",
            (true, false) => "launch_recurrent_gated_delta_step_batched_gqa",
            (false, true) => "launch_recurrent_gated_delta_step_batched_state_bf16",
            (false, false) => "launch_recurrent_gated_delta_step_batched",
        };
        let ids = q.t.with(Some(rs.l), |b| {
            let shape = b.value_shape(v.id);
            b.launch(
                kernel,
                vec![],
                rs_state(rs),
                vec![q.id, k.id, v.id, g.id, beta.id],
                vec![(shape, DType::F32)],
            )
        });
        Val {
            t: q.t.clone(),
            id: ids[0],
            layer: Some(rs.l),
        }
    }

    /// `kernels::launch_chunk_gated_delta_prefill_batched_warp_tiled[_gqa][_state_bf16]`:
    /// the warp-tiled small-N prefill recurrence. NOT a value producer:
    /// the three prefill recurrence signatures record launches with NO
    /// outputs, because each runs inside a value-producing guard chain
    /// ([`guarded_value`]) whose output IS the recurrence core — the
    /// region launches bind the guard's buffer and add no SSA values of
    /// their own.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_prefill_warp_tiled(
        q: &Val,
        k: &Val,
        v: &Val,
        g: &Val,
        beta: &Val,
        rs: &Rs,
        gqa: bool,
        state_bf16: bool,
    ) {
        let kernel = match (gqa, state_bf16) {
            (true, true) => "launch_chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16",
            (true, false) => "launch_chunk_gated_delta_prefill_batched_warp_tiled_gqa",
            (false, true) => "launch_chunk_gated_delta_prefill_batched_warp_tiled_state_bf16",
            (false, false) => "launch_chunk_gated_delta_prefill_batched_warp_tiled",
        };
        gdn_prefill(q, k, v, g, beta, rs, kernel);
    }

    /// `kernels::launch_chunk_gated_delta_prefill_batched_cached[_state_bf16]`:
    /// the env-gated cached prefill recurrence. No `_gqa` variant exists —
    /// this family indexes the REPEATED `[Vh]`-head layout, which is why
    /// its guard arm materializes [`repeat_interleave_heads`] first.
    /// Guard-region launch, output-less like the warp-tiled form.
    pub fn gdn_prefill_cached(
        q: &Val,
        k: &Val,
        v: &Val,
        g: &Val,
        beta: &Val,
        rs: &Rs,
        state_bf16: bool,
    ) {
        let kernel = if state_bf16 {
            "launch_chunk_gated_delta_prefill_batched_cached_state_bf16"
        } else {
            "launch_chunk_gated_delta_prefill_batched_cached"
        };
        gdn_prefill(q, k, v, g, beta, rs, kernel);
    }

    /// `kernels::launch_chunk_gated_delta_prefill_batched[_state_bf16]`:
    /// the batched GQA-aware FLA prefill recurrence — the fallback arm
    /// (it indexes the compact K_h layout directly, so no repeats).
    /// Guard-region launch, output-less like the warp-tiled form.
    pub fn gdn_prefill_fla(
        q: &Val,
        k: &Val,
        v: &Val,
        g: &Val,
        beta: &Val,
        rs: &Rs,
        state_bf16: bool,
    ) {
        let kernel = if state_bf16 {
            "launch_chunk_gated_delta_prefill_batched_state_bf16"
        } else {
            "launch_chunk_gated_delta_prefill_batched"
        };
        gdn_prefill(q, k, v, g, beta, rs, kernel);
    }

    fn gdn_prefill(q: &Val, k: &Val, v: &Val, g: &Val, beta: &Val, rs: &Rs, kernel: &str) {
        record(
            &q.t,
            Some(rs.l),
            kernel,
            vec![],
            rs_state(rs),
            vec![q.id, k.id, v.id, g.id, beta.id],
            None,
        );
    }

    /// `kernels::launch_repeat_interleave_heads_fp32`: materialize the
    /// K_h → V_h head repeat of a compact per-head f32 value into the
    /// workspace buffer the cached recurrence family reads. Output-less:
    /// where that buffer lives is the driver's binding, not dataflow —
    /// the same stance as the KV writes. Stated only inside the cached
    /// arm, because only that kernel family consumes the repeated layout
    /// (the decode-GQA step, warp-tiled and FLA kernels all index the
    /// compact layout directly).
    pub fn repeat_interleave_heads(x: &Val) {
        record(
            &x.t,
            x.layer,
            "launch_repeat_interleave_heads_fp32",
            vec![],
            None,
            vec![x.id],
            None,
        );
    }

    /// `"qwen35_verify_stash_load"`: replay the layer's stashed in-proj
    /// outputs — `[mixed_qkv | a | b]` from the verify hidden stash slab
    /// into the workspace buffers the following conv/prep read.
    ///
    /// A PSEUDO-SYMBOL, the first: it names an OPERATION the driver
    /// implements as a `cudaMemcpyAsync` trio, not a `__global__` entry
    /// point. The contract stands regardless — a launcher may be three
    /// API calls; the symbol names the operation, and the driver's
    /// name→launcher registry resolves it like any other. No inputs
    /// (the stash is the layer's per-request state, marked below);
    /// THREE outputs, the in-proj triple the GEMMs would have produced —
    /// mixed_qkv `[Tokens, conv_dim]`, a `[Tokens, value_heads]`, b
    /// `[Tokens, value_heads]`, all bf16 — so the CommitAdvance pass's
    /// dataflow into `gdn_prep` stays complete. WHERE those buffers
    /// live is the driver's binding, [`repeat_interleave_heads`]-style.
    pub fn verify_stash_load(t: &Trace, rs: &Rs, conv_dim: u32, value_heads: u32) -> (Val, Val, Val) {
        let ids = t.with(Some(rs.l), |b| {
            b.launch(
                "qwen35_verify_stash_load",
                vec![],
                rs_state(rs),
                vec![],
                vec![
                    (Shape(vec![Dim::Tokens, Dim::Const(conv_dim)]), DType::BF16),
                    (Shape(vec![Dim::Tokens, Dim::Const(value_heads)]), DType::BF16),
                    (Shape(vec![Dim::Tokens, Dim::Const(value_heads)]), DType::BF16),
                ],
            )
        });
        let mk = |id| Val {
            t: t.clone(),
            id,
            layer: Some(rs.l),
        };
        (mk(ids[0]), mk(ids[1]), mk(ids[2]))
    }

    /// `"qwen35_verify_stash_store"`: persist a linear layer's in-proj
    /// triple `[qkv, a, b]` into the layer's verify hidden stash slab —
    /// [`verify_stash_load`]'s writing half, the same pseudo-symbol
    /// contract (a memcpy trio behind one name). Output-less: the stash
    /// is the layer's per-request state, not dataflow.
    ///
    /// No class this rung states it: its consumer is the future
    /// frozen-verify class (the `write_state=false` verify pass that
    /// fills the stash the commit pass replays — semantic this rung).
    /// The pair is declared together because the load's contract is only
    /// meaningful against the store's layout.
    pub fn verify_stash_store(qkv: &Val, a: &Val, b: &Val, rs: &Rs) {
        record(
            &qkv.t,
            Some(rs.l),
            "qwen35_verify_stash_store",
            vec![],
            rs_state(rs),
            vec![qkv.id, a.id, b.id],
            None,
        );
    }

    /// `ops::dispatch_attention_flashinfer_decode_capture`: the
    /// score-capturing decode dispatch (the OnAttn sideband's producer;
    /// its contract includes the capture publish against the possibly
    /// page-mask-compacted CSR). Region launch of the WantsAttnScore
    /// guard — output-less; the guard owns the attention output.
    pub fn attention_flashinfer_decode_capture(q: &Val, kv: &Kv) -> Option<Val> {
        attn_at(q, kv, "dispatch_attention_flashinfer_decode_capture")
    }

    /// `ops::dispatch_attention_flashinfer_prefill_capture_bf16` — the
    /// prefill counterpart, same guard-region contract.
    pub fn attention_flashinfer_prefill_capture(q: &Val, kv: &Kv) -> Option<Val> {
        attn_at(q, kv, "dispatch_attention_flashinfer_prefill_capture_bf16")
    }

    /// Output-less [`qkv_decode_qk_norm_rope_write_kv`] for the Peel's
    /// prefix region (A3): the peel owns q; this launch binds its
    /// `[0, fast_rows)` rows. Same operands, same aux contract.
    pub fn qkv_decode_qk_norm_rope_write_kv_region(
        packed: &Val,
        q_norm: &NormW,
        k_norm: &NormW,
        kv: &Kv,
        table: Option<&Val>,
    ) {
        let mut inputs = vec![packed.id];
        if let Some(t) = table {
            inputs.push(t.id);
        }
        record(
            &packed.t,
            Some(kv.l),
            "launch_qkv_decode_qk_norm_rope_write_kv_bf16",
            vec![q_norm.name.clone(), k_norm.name.clone()],
            kv_state(kv),
            inputs,
            None,
        );
    }

    /// `ops::dispatch_attention_flashinfer_prefill_custom`: the
    /// custom-mask prefill dispatch — a genuinely distinct launcher, so
    /// no pseudo-symbol is needed. The mask data (BRLE bytes + indptr)
    /// crosses as runtime args of the stated kernel, commit_lens's peer.
    /// Since A1 (the class-collapse amendment) it is stated inside the
    /// `HasCustomMask` guard arm of the Decode/Prefill traces.
    pub fn attention_flashinfer_prefill_custom(q: &Val, kv: &Kv) -> Option<Val> {
        attn_at(q, kv, "dispatch_attention_flashinfer_prefill_custom")
    }

    /// `"pie_lora_qkv_correction"`: the §5.1 adapter correction — every
    /// usable lora lane's `x·Aᵀ·Bᵀ` delta landed on the materialized q/v
    /// projections, before anything consumes them (bias, norms, rope,
    /// KV append). A PSEUDO-SYMBOL, the stash pair's peer: the driver
    /// implements it as the per-lane / grouped GEMM-pair sequence of
    /// `LoraFireState::apply` — a launcher may be many calls; the symbol
    /// names the operation. Output-less and in place on q/v; stated
    /// inside the `HasLora` guard's then-region (the else is empty — a
    /// fire with no adapters launches nothing, which is the truth).
    pub fn lora_qkv_correction(q: &Val, v: &Val, l: u32) {
        record(
            &q.t,
            Some(l),
            "pie_lora_qkv_correction",
            vec![],
            None,
            vec![q.id, v.id],
            None,
        );
    }

    /// `kernels::launch_dequant_kv_cache_layer_to_bf16_active`: the
    /// staging launch a quantized cache needs before a prefill-shaped
    /// dispatch. Its OWN statement — see
    /// [`attention_flashinfer_prefill`] for why it is not folded into
    /// any attention wrapper.
    pub fn dequant_only(kv: &Kv) {
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

    /// ONE attention statement, whatever position it is written in
    /// (`.wiki/tart/dsl.md` ②, migration step 2).
    ///
    /// A dispatch inside a value-producing guard or peel region binds
    /// that construct's output and records no SSA output of its own; the
    /// same dispatch written as a plain statement produces its own
    /// value. That is a property of the STATEMENT'S POSITION, which the
    /// tape knows ([`crate::trace::TraceBuilder::inside_value_region`]),
    /// so it stops being spelled in the wrapper's name — the `_region`
    /// half of every attention wrapper is deleted by this one function.
    ///
    /// The output shape is q's own: these kernels are width-preserving
    /// on the query, which is what the retired `q_width` parameter was
    /// re-stating at each call site.
    fn attn_at(q: &Val, kv: &Kv, kernel: &str) -> Option<Val> {
        let out = q.t.inner.borrow().inside_value_region();
        let shape = (!out).then(|| q.t.inner.borrow().value_shape(q.id));
        record(
            &q.t,
            Some(kv.l),
            kernel,
            vec![],
            kv_state(kv),
            vec![q.id],
            shape.map(|s| (s, DType::BF16)),
        )
    }

}
