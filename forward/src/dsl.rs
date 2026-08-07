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

impl Val {
    /// The tape this value was recorded on — what a seam statement
    /// needs when the text has no [`M`] or bare [`Trace`] in hand.
    pub fn trace(&self) -> &Trace {
        &self.t
    }
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
    /// (`OpKind::LmHead`, resolving the tied-embedding fact). Returns
    /// the logits — what the `out` seam sees.
    pub fn logits(&self, x: &Val) -> Val {
        let name = if self.f.tied_embeddings {
            "embed"
        } else {
            "lm_head"
        };
        let id = self.t.with(None, |b| b.lm_head(x.id, name, self.f.vocab));
        Val {
            t: self.t.clone(),
            id,
            layer: None,
        }
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
    run(format!("llama_like.cuda.{}", class_word(class)), facts, body)
}

/// Run a LOWERED llama_like declaration for METAL.
///
/// The counterpart of [`trace_cuda`], and the reason the backend is a
/// first-class axis rather than a CUDA assumption: a model text is
/// written for one backend, so Metal gets its own text stating Metal's
/// kernels, checked against Metal's table
/// ([`crate::kernels::KERNELS_METAL`]).
///
/// Metal has NO such text yet — it consumes the semantic trace and
/// re-derives its dispatch selection in C++
/// (`driver/metal/src/model/llama_like/declared_dag.hpp`), which is the
/// same "the driver decides" shape the CUDA side is being cured of.
/// This entry is the seam that text will be written against; until it
/// is, nothing calls it, and the empty Metal kernel table means the
/// first thing that does must declare its kernels.
pub fn trace_metal(
    facts: &LlamaLikeFacts,
    class: FireClass,
    body: impl FnOnce(&mut M),
) -> ForwardPlan {
    run(format!("llama_like.metal.{}", class_word(class)), facts, body)
}

fn class_word(class: FireClass) -> &'static str {
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
pub fn lm_head_at(t: &Trace, x: &Val, weight: &str, vocab: u32) -> Val {
    let id = t.with(None, |b| b.lm_head(x.id, weight, vocab));
    Val {
        t: t.clone(),
        id,
        layer: None,
    }
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

    /// A seam's SIGNATURE (`.wiki/tart/dsl.md` ①): the stable NAME the
    /// request surface keys on (`fwd.adapter("attn.qv", ..)`,
    /// `fwd.attach(..)`), what an attachment SEES, what it MAY do, and —
    /// for the seams that have one — where it sits and where its output
    /// lands.
    ///
    /// `after` / `before` and `sink` are the two lines the doc singles
    /// out as carrying what is today only a comment. They are not
    /// documentation here: [`check_plan`] reads `after` / `before`.
    pub struct Def {
        pub name: &'static str,
        /// The value roles an attachment observes or rewrites, in
        /// operand order.
        pub sees: &'static [&'static str],
        pub caps: &'static [Cap],
        /// The seam's POSITION rule, for seams whose arithmetic depends
        /// on it. `after` names the op kinds that must have produced the
        /// values it sees; `before` names the op kinds that must not yet
        /// have consumed them.
        pub position: Option<Position>,
        /// Where a sink-writing attachment's output lands.
        pub sink: Option<&'static str>,
    }

    /// A seam's position rule, stated as op-kind names
    /// ([`crate::trace::OpKind`]'s discriminants, plus `"Launch:<symbol>"`
    /// for stated kernels).
    #[derive(Debug, Clone, Copy)]
    pub struct Position {
        pub after: &'static [&'static str],
        pub before: &'static [&'static str],
    }

    /// Pre-attention observation seam: sees the just-projected q; a
    /// page-mask-sink attachment narrows the page list the SAME stated
    /// attention kernel consumes as substituted arguments (today's
    /// `OnAttnProj`).
    pub const ATTN_Q: Def = Def {
        name: "attn.q",
        sees: &["q"],
        caps: &[Cap::Observe, Cap::PageMaskSink],
        position: None,
        // Where Quest's `attn_page_mask` lands. Hardcoded in
        // `emit_cuda::emit_masked_pages_bracket` today; declared here,
        // consumed when the launch ABI flattens (migration step 6).
        sink: Some("attention.pages"),
    };

    /// Post-attention observation seam: sees the scores the (possibly
    /// capturing) dispatch published through the sideband (today's
    /// `OnAttn`).
    pub const ATTN_OUT: Def = Def {
        name: "attn.out",
        sees: &["a"],
        caps: &[Cap::Observe, Cap::Scores],
        position: None,
        sink: None,
    };

    /// The adapter value seam over the raw q/v projections — pure
    /// expressions of `(x, y)`, `fwd.adapter`'s site family (today's
    /// `HasLora` guard arm).
    /// THE POSITION RULE IS THE POINT: the correction lands on the raw
    /// projections, before bias, norms, rope and the KV append. Applying
    /// it after rope is DIFFERENT ARITHMETIC — the bug the first live
    /// A/B caught. It was a comment until now.
    pub const ATTN_QV: Def = Def {
        name: "attn.qv",
        sees: &["q", "v"],
        caps: &[Cap::Transform],
        position: Some(Position {
            after: &["Matmul", "SplitQkv"],
            before: &["AddBias", "Rmsnorm", "Rope", "KvAppend", "Launch"],
        }),
        sink: None,
    };

    /// Entry boundary seam (prologue's home). Boundary attachments
    /// never enter row signatures — they cause no divergence — which is
    /// why their dispatch-side lowering needs no trace op at any rung.
    pub const IN: Def = Def {
        name: "in",
        sees: &[],
        caps: &[Cap::Put, Cap::Emit],
        position: None,
        sink: None,
    };

    /// Exit boundary seam (epilogue's home).
    pub const OUT: Def = Def {
        name: "out",
        sees: &["logits"],
        caps: &[Cap::Observe, Cap::Sample, Cap::Put, Cap::Emit],
        position: None,
        sink: None,
    };

    /// LOAD-TIME check of the seams a text stated.
    ///
    /// One rule today, and it is the one whose violation is silent:
    /// [`ATTN_QV`]'s position. The adapter's delta must land on the base
    /// projection, not on base + bias, and not after rope — so between
    /// the ops that PRODUCE the values the seam sees and the seam's own
    /// statement, nothing may consume them. A live A/B caught exactly
    /// this once; the rule stops being a comment here.
    pub fn check_plan(plan: &crate::trace::ForwardPlan) -> Vec<String> {
        let mut problems = Vec::new();
        for stmt in &plan.seams {
            let Some(def) = by_name(&stmt.seam) else {
                problems.push(format!(
                    "{}: states seam `{}`, which no seam! signature declares",
                    plan.family, stmt.seam
                ));
                continue;
            };
            let (Some(pos), Some(at)) = (def.position, stmt.op) else {
                continue;
            };
            let at = at as usize;
            // The values this statement sees are the inputs of the op
            // it carries (the adapter's guard opens at `at`; its
            // correction launch is the next op and names q and v).
            let Some(seen) = plan.ops.get(at + 1).map(|op| op.inputs.clone()) else {
                continue;
            };
            for &v in &seen {
                let produced_at = plan
                    .ops
                    .iter()
                    .position(|op| op.outputs.contains(&v));
                match produced_at {
                    None => problems.push(format!(
                        "{}: seam `{}` sees value {v}, which no op produces",
                        plan.family, def.name
                    )),
                    Some(from) => {
                        let producer = kind_name(&plan.ops[from].kind);
                        if !pos.after.contains(&producer) {
                            problems.push(format!(
                                "{}: seam `{}` must sit after {:?}, but value {v} \
                                 comes from {producer}",
                                plan.family, def.name, pos.after
                            ));
                        }
                        for (i, op) in plan.ops.iter().enumerate().take(at).skip(from + 1) {
                            if !op.inputs.contains(&v) {
                                continue;
                            }
                            let consumer = kind_name(&op.kind);
                            if pos.before.contains(&consumer) {
                                problems.push(format!(
                                    "{}: seam `{}` must sit before {consumer}, but op \
                                     {i} consumes value {v} first — different \
                                     arithmetic, not a reordering",
                                    plan.family, def.name
                                ));
                            }
                        }
                    }
                }
            }
        }
        problems
    }

    /// Every seam a model text may state.
    pub const ALL: &[&Def] = &[&IN, &ATTN_QV, &ATTN_Q, &ATTN_OUT, &OUT];

    pub fn by_name(name: &str) -> Option<&'static Def> {
        ALL.iter().copied().find(|d| d.name == name)
    }

    fn kind_name(kind: &crate::trace::OpKind) -> &'static str {
        use crate::trace::OpKind as K;
        match kind {
            K::Embed { .. } => "Embed",
            K::Matmul { .. } => "Matmul",
            K::SplitQkv { .. } => "SplitQkv",
            K::Rope { .. } => "Rope",
            K::Rmsnorm { .. } => "Rmsnorm",
            K::AddBias { .. } => "AddBias",
            K::KvAppend { .. } => "KvAppend",
            K::Launch { .. } => "Launch",
            K::Guard { .. } => "Guard",
            K::Peel { .. } => "Peel",
            K::HookSite { .. } => "HookSite",
            _ => "other",
        }
    }
}

/// THE SEAM STATEMENT — one construct for all five extension points
/// (`.wiki/tart/dsl.md` ①, migration step 4).
///
/// Until now three of the five lowered to ops through two different
/// functions and the other two lowered to NOTHING: the traced form did
/// not record that a text has a prologue or an epilogue at all, which is
/// what put those two stages in a different world from the rest. Every
/// seam is stated the same way here, and every statement is recorded
/// ([`crate::trace::SeamStatement`]) whichever way it lowers.
///
/// The LOWERINGS are unchanged, which is what keeps the goldens'
/// op streams byte-identical:
///
/// * `attn.q` / `attn.out` — one [`OpKind::HookSite`];
/// * `attn.qv` — the `HasLora` guard with the correction arm and an
///   EMPTY else (a fire with no usable lanes launches nothing);
/// * `in` / `out` — no op. A boundary attachment causes no divergence,
///   so it enters no row signature; what it needed was a DECLARATION,
///   and that is what the statement list now carries.
///
/// [`OpKind::HookSite`]: crate::trace::OpKind::HookSite
pub fn seam(t: &Trace, def: &seam::Def, sees: &[&Val], layer: Option<u32>) {
    assert_eq!(
        sees.len(),
        def.sees.len(),
        "seam `{}` sees {:?}",
        def.name,
        def.sees
    );
    match def.name {
        "attn.q" | "attn.out" => {
            let stage = if def.name == "attn.q" {
                crate::trace::HookStage::OnAttnProj
            } else {
                crate::trace::HookStage::OnAttn
            };
            let l = layer.expect("a body seam states its layer");
            hook_site(stage, sees[0], l);
            let at = t.inner.borrow().op_count_now() - 1;
            t.inner
                .borrow_mut()
                .push_seam(def.name, layer, Some(at as u32));
        }
        "attn.qv" => {
            let l = layer.expect("a body seam states its layer");
            // The index the guard is about to take, captured before it
            // opens: the statement points at the CONSTRUCT, and the
            // position check reads the correction's operands from
            // inside it.
            let at = t.inner.borrow().op_count_now() as u32;
            guard_on(
                t,
                crate::trace::GuardPred::HasLora,
                || cuda::lora_qkv_correction(sees[0], sees[1], l),
                || {},
            );
            t.inner.borrow_mut().push_seam(def.name, layer, Some(at));
        }
        "in" | "out" => {
            t.inner.borrow_mut().push_seam(def.name, layer, None);
        }
        other => unreachable!("no seam named {other}"),
    }
}

// ── Metal kernel signatures ────────────────────────────────────────────

/// The METAL launchers a lowered declaration may state — the `dsl::cuda`
/// of the second backend (`.wiki/tart/dsl.md` ②).
///
/// UNVERIFIED (2026-08-05). Every symbol here is an MSL entrypoint read
/// off the driver's source (`driver/metal/src/kernels/decode_psos.cpp`'s
/// `PsoSpec` table and `model/qwen3_5/decode_step.hpp`'s `Kernel` kinds),
/// not something a running deployment produced: the Metal driver cannot
/// build on the machine we have, because `xcrun --find metal` fails —
/// the shader compiler ships with full Xcode. Nothing consumes this yet.
/// `.wiki/tart/macos.md` rung 3 is where it gets proven, by showing
/// `declared_dag.hpp`'s emitted descriptors come out unchanged.
///
/// ONE DECISION worth stating, because it will look like an omission:
/// the quantized entrypoints are spelled by their BASE name
/// (`affine_qmv_fast`), not with the checkpoint's affine suffix
/// (`..._bfloat16_gs_64_b_4`, `AffineFormat::kernel_suffix()`). The
/// suffix is the driver's binding of a checkpoint fact, in the same
/// class as the stream and the workspace scratch — it selects no
/// different arithmetic and no different arm. What the text chooses is
/// the kernel FAMILY.
pub mod metal {
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

    fn same_shape(v: &Val) -> (Shape, DType) {
        (v.t.inner.borrow().value_shape(v.id), DType::BF16)
    }

    /// `embed_gather.metal::embed_gather_4bit` (M=1) /
    /// `embed_gather_mb_4bit` (M>1).
    pub fn embed_gather(t: &Trace, weight: &str, hidden: u32, multi_batch: bool) -> Val {
        let kernel = if multi_batch {
            "embed_gather_mb_4bit"
        } else {
            "embed_gather_4bit"
        };
        record(
            t,
            None,
            kernel,
            vec![weight.to_string()],
            None,
            vec![],
            Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
        )
        .expect("embed produces the residual stream")
    }

    /// `rms_norm.metal::rms_single_row_bfloat16` — ONE entrypoint for
    /// every norm this family states (attn_norm, mlp_norm, q_norm,
    /// k_norm, final_norm; the driver fans five `Kernel` kinds onto it).
    pub fn rms_norm(x: &Val, w: &NormW) -> Val {
        let out = same_shape(x);
        record(
            &x.t,
            w.layer,
            "rms_single_row_bfloat16",
            vec![w.name.clone()],
            None,
            vec![x.id],
            Some(out),
        )
        .expect("a norm produces its value")
    }

    /// `quantized_qmv.metal::affine_qmv_fast` — the projection GEMV,
    /// M=1. The driver fans every projection kind onto it.
    pub fn qmv(x: &Val, w: &MatW) -> Val {
        record(
            &x.t,
            w.layer,
            "affine_qmv_fast",
            vec![w.name.clone()],
            None,
            vec![x.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(w.width)]), DType::BF16)),
        )
        .expect("a projection produces its value")
    }

    /// `quantized_qmv.metal::affine_qmv_fast_residual` — the same GEMV
    /// with the block residual folded into its epilogue, which is what a
    /// `beta_one` matmul is on this backend.
    pub fn qmv_residual(x: &Val, w: &MatW, residual: &Val) -> Val {
        record(
            &x.t,
            w.layer,
            "affine_qmv_fast_residual",
            vec![w.name.clone()],
            None,
            vec![x.id, residual.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(w.width)]), DType::BF16)),
        )
        .expect("a folded projection produces its value")
    }

    /// `quantized_qmm_t.metal::affine_qmm_t` — MLX's steel quantized
    /// GEMM, the M>1 projection.
    pub fn qmm(x: &Val, w: &MatW) -> Val {
        record(
            &x.t,
            w.layer,
            "affine_qmm_t",
            vec![w.name.clone()],
            None,
            vec![x.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(w.width)]), DType::BF16)),
        )
        .expect("a projection produces its value")
    }

    /// `quantized_qmm_t.metal::affine_qmm_t_residual`.
    pub fn qmm_residual(x: &Val, w: &MatW, residual: &Val) -> Val {
        record(
            &x.t,
            w.layer,
            "affine_qmm_t_residual",
            vec![w.name.clone()],
            None,
            vec![x.id, residual.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(w.width)]), DType::BF16)),
        )
        .expect("a folded projection produces its value")
    }

    /// `residual_add.metal::residual_add_bfloat16` — the explicit
    /// landing, for the deployments and positions where no epilogue fold
    /// exists.
    pub fn residual_add(x: &Val, residual: &Val) -> Val {
        let out = same_shape(x);
        record(
            &x.t,
            x.layer,
            "residual_add_bfloat16",
            vec![],
            None,
            vec![x.id, residual.id],
            Some(out),
        )
        .expect("the residual landing produces its value")
    }

    /// `rope.metal::rope_neox_decode_bfloat16` (M=1) /
    /// `rope_neox_mb_bfloat16` (M>1). One dispatch for q and k together,
    /// as the plan states it (`declared_dag.hpp`'s `Kind::Rope`).
    pub fn rope(q: &Val, k: &Val, multi_batch: bool) -> (Val, Val) {
        let kernel = if multi_batch {
            "rope_neox_mb_bfloat16"
        } else {
            "rope_neox_decode_bfloat16"
        };
        let q_sh = same_shape(q);
        let k_sh = same_shape(k);
        let ids = q.t.with(q.layer, |b| {
            b.launch(kernel, vec![], None, vec![q.id, k.id], vec![q_sh, k_sh])
        });
        let mk = |id| Val {
            t: q.t.clone(),
            id,
            layer: q.layer,
        };
        (mk(ids[0]), mk(ids[1]))
    }

    /// `kv_append.metal::kv_append_bfloat16` (contiguous) /
    /// `kv_append_paged.metal::kv_append_paged_bfloat16` (page table).
    pub fn kv_append(k: &Val, v: &Val, kv: &Kv, paged: bool) {
        let kernel = if paged {
            "kv_append_paged_bfloat16"
        } else {
            "kv_append_bfloat16"
        };
        record(
            &kv.t,
            Some(kv.l),
            kernel,
            vec![],
            kv_state(kv),
            vec![k.id, v.id],
            None,
        );
    }

    /// `sdpa_vector.metal::sdpa_vector_decode_bfloat16_d_256` (M=1) /
    /// `sdpa_paged.metal::sdpa_paged_decode_bfloat16_d_256` (M>1).
    pub fn sdpa(q: &Val, kv: &Kv, q_width: u32, paged: bool) -> Option<Val> {
        let kernel = if paged {
            "sdpa_paged_decode_bfloat16_d_256"
        } else {
            "sdpa_vector_decode_bfloat16_d_256"
        };
        record(
            &q.t,
            Some(kv.l),
            kernel,
            vec![],
            kv_state(kv),
            vec![q.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16)),
        )
    }

    /// `silu_mul.metal::silu_mul_bfloat16` — the SwiGLU activation over
    /// the packed gate/up bank.
    pub fn silu_mul(x: &Val, intermediate: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "silu_mul_bfloat16",
            vec![],
            None,
            vec![x.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
                DType::BF16,
            )),
        )
        .expect("the activation produces its value")
    }

    /// `quantized_qmv.metal::affine_qmv_fast` against the lm head — the
    /// readout, `[Requests, vocab]` f32 like every family's.
    pub fn lm_head(x: &Val, weight: &str, vocab: u32) -> Val {
        record(
            &x.t,
            None,
            "affine_qmv_fast",
            vec![weight.to_string()],
            None,
            vec![x.id],
            Some((Shape(vec![Dim::Requests, Dim::Const(vocab)]), DType::F32)),
        )
        .expect("the readout produces the logits")
    }
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

    // ── gemma-4 ────────────────────────────────────────────────────
    //
    // The vocabulary the third family needs and the first two did not.
    // Every one of these is a kernel the hand-written `gemma4.cpp`
    // already fires; what is new is that a declaration can name it.

    /// `kernels::launch_{chunked_,}geglu_tanh_bf16`: gemma-4's MLP
    /// activation. `gelu_pytorch_tanh` on the gate, not SiLU — a
    /// different function, so a different kernel, and NOT a variant of
    /// [`swiglu`].
    ///
    /// `packed` splits the same way swiglu's does: a bound gate‖up bank
    /// lands one buffer and takes the chunked form. gemma-4 states the
    /// binding as a fact for the same reason llama_like does.
    pub fn geglu_tanh(x: &Val, intermediate: u32, packed: bool) -> Val {
        record(
            &x.t,
            x.layer,
            if packed {
                "launch_chunked_geglu_tanh_bf16"
            } else {
                "launch_geglu_tanh_bf16"
            },
            vec![],
            None,
            vec![x.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
                DType::BF16,
            )),
        )
        .expect("the activation produces its value")
    }

    /// `kernels::launch_geglu_tanh_bf16` in its PAIR form: the gate and
    /// the up operand are two buffers, not one packed bank.
    ///
    /// gemma-4's PLE epilogue needs it even on a checkpoint that bound a
    /// packed MLP bank, because the "up" operand there is the layer's
    /// slice of the per-layer table — a buffer that was never going to
    /// be adjacent to the gate. Same kernel as [`geglu_tanh`]'s unpacked
    /// arm; a different statement because the OPERANDS differ, which is
    /// what a reader needs to see.
    pub fn geglu_tanh_pair(gate: &Val, up: &Val, width: u32) -> Val {
        record(
            &gate.t,
            gate.layer,
            "launch_geglu_tanh_bf16",
            vec![],
            None,
            vec![gate.id, up.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
        )
        .expect("the activation produces its value")
    }

    /// `kernels::launch_rope_partial_bf16` rotating Q ALONE.
    ///
    /// A KV-shared layer's K was rotated at its SOURCE layer, where it
    /// was written to the cache, so rotating it again here would be
    /// wrong twice over — the value is not even in this layer's
    /// registers. The driver says that with `num_kv_heads = 0`; the
    /// trace says it by the statement having ONE operand.
    ///
    /// The semantic [`super::rope`] cannot: its shape is a (q, k) pair,
    /// and a pair with an empty slot is a different statement, not a
    /// degenerate one.
    pub fn rope_partial_q_only(q: &Val) -> Val {
        let out = (q.t.inner.borrow().value_shape(q.id), DType::BF16);
        record(
            &q.t,
            q.layer,
            "launch_rope_partial_bf16",
            vec![],
            None,
            vec![q.id],
            Some(out),
        )
        .expect("the rotation produces its value")
    }

    /// [`qk_rmsnorm_rope_rounded`] with K absent — the SHARED sliding
    /// layer's form.
    ///
    /// Same symbol, and that is the point: the driver reaches this by
    /// passing `k_norm = nullptr` and `num_kv_heads = 0` to the very
    /// same launcher, so a declaration that spelled it as a rope plus a
    /// separate norm would be naming a pair of kernels the pass never
    /// fires. One operand, one weight, one launch.
    pub fn qk_rmsnorm_rope_rounded_q_only(q: &Val, q_norm: &NormW) -> Val {
        let out = (q.t.inner.borrow().value_shape(q.id), DType::BF16);
        record(
            &q.t,
            q_norm.layer,
            "launch_qk_rmsnorm_rope_bf16_rounded",
            vec![q_norm.name.clone()],
            None,
            vec![q.id],
            Some(out),
        )
        .expect("the fused pair produces q")
    }

    /// `kernels::launch_rmsnorm_no_scale_bf16`: `v / rms(v)` per head,
    /// with NO learnable weight — gemma-4's V-norm.
    ///
    /// Weightless, so it takes no [`NormW`]: a norm handle contributes a
    /// name and a layer, and this kernel reads neither. That is also why
    /// it cannot be the semantic `Rmsnorm` with a variant — there is no
    /// gamma for a variant to describe.
    pub fn rmsnorm_no_scale(x: &Val) -> Val {
        let out = (x.t.inner.borrow().value_shape(x.id), DType::BF16);
        record(
            &x.t,
            x.layer,
            "launch_rmsnorm_no_scale_bf16",
            vec![],
            None,
            vec![x.id],
            Some(out),
        )
        .expect("the norm produces its value")
    }

    /// `kernels::launch_rmsnorm_residual_add_scale_rmsnorm_bf16`: FOUR
    /// statements in one launch — norm `x`, add it to the stream, scale
    /// the result, then norm THAT with the next weight.
    ///
    /// The last of those four is the next block's input norm, which is
    /// why gemma-4's per-layer body appears to be missing one: the fused
    /// kernel already produced it. A declaration that named the four
    /// separately would be naming a shape the driver does not run.
    ///
    /// Returns `(hidden, norm_out)` — the landed residual and the norm
    /// the next block consumes.
    pub fn norm_residual_scale_norm(
        x: &Val,
        w: &NormW,
        next: &NormW,
        hidden: u32,
    ) -> (Val, Val) {
        let shape = (Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16);
        let ids = x.t.with(w.layer, |b| {
            b.launch(
                "launch_rmsnorm_residual_add_scale_rmsnorm_bf16",
                vec![w.name.clone(), next.name.clone()],
                None,
                vec![x.id],
                vec![shape.clone(), shape],
            )
        });
        let mk = |id| Val {
            t: x.t.clone(),
            id,
            layer: w.layer,
        };
        (mk(ids[0]), mk(ids[1]))
    }

    /// `kernels::launch_rmsnorm_residual_add_bf16`: the two-statement
    /// form — norm, then land on the stream. gemma-4's
    /// post-feedforward norm, where no next-block norm follows to fuse.
    pub fn norm_residual_add(x: &Val, w: &NormW, hidden: u32) -> Val {
        record(
            &x.t,
            w.layer,
            "launch_rmsnorm_residual_add_bf16",
            vec![w.name.clone()],
            None,
            vec![x.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
        )
        .expect("the fused norm+residual produces its value")
    }

    /// `kernels::launch_scalar_mul_bf16`: multiply by a load-time
    /// constant, NAMED.
    ///
    /// gemma-4 fires this four times per fire with four different
    /// constants — `sqrt(hidden)` on the embedding, then
    /// `sqrt(ple_dim)`, `1/sqrt(hidden)` and `1/sqrt(2)` through the PLE
    /// prologue. All four are derived from dims, so none is an operand;
    /// but a statement that did not say WHICH would leave an executor
    /// with four identical launches and no way to tell them apart. This
    /// was written without the name first, and writing the arm is what
    /// found it.
    ///
    /// The name rides the weight slot because that is what a name slot
    /// is: `scale.` marks it as a constant rather than a tensor, so a
    /// binder never looks for it.
    pub fn scalar_mul(x: &Val, scale: &str) -> Val {
        let out = (x.t.inner.borrow().value_shape(x.id), DType::BF16);
        record(
            &x.t,
            x.layer,
            "launch_scalar_mul_bf16",
            vec![format!("scale.{scale}")],
            None,
            vec![x.id],
            Some(out),
        )
        .expect("the scale produces its value")
    }

    /// `kernels::launch_logit_softcap_bf16`: `cap * tanh(x / cap)` over
    /// the logits. A load-time fact decides whether it runs at all
    /// (`final_logit_softcapping`), so its presence is a trace-time
    /// match, not a branch.
    pub fn logit_softcap(x: &Val, vocab: u32) -> Val {
        record(
            &x.t,
            None,
            "launch_logit_softcap_bf16",
            vec![],
            None,
            vec![x.id],
            Some((Shape(vec![Dim::Requests, Dim::Const(vocab)]), DType::BF16)),
        )
        .expect("the softcap produces its value")
    }

    /// `kernels::launch_qkv_packed_qk_norm_rope_vnorm_write_kv_bf16`:
    /// gemma-4's decode post — split the packed projection, norm q and
    /// k, rope them, norm v, and write k/v straight to the pages. One
    /// launch, six statements, and the only value that survives it is q.
    ///
    /// Its eligibility is a per-FIRE question in the hand-written pass
    /// (`hooks == nullptr && !partial && !dump && native bf16 && a
    /// decode path`), and the terms split cleanly: `partial` and the
    /// cache format are load-time, hooks and the fire class are the
    /// declaration's own class/guard vocabulary. So a class trace states
    /// it or does not, and nothing reads a workspace to decide.
    ///
    /// Writes through the KV pages, so it carries the layer's cache
    /// state the way every write-side statement here does.
    pub fn qkv_packed_post(
        packed: &Val,
        q_norm: &NormW,
        k_norm: &NormW,
        kv: &Kv,
        q_width: u32,
    ) -> Val {
        record(
            &packed.t,
            q_norm.layer,
            "launch_qkv_packed_qk_norm_rope_vnorm_write_kv_bf16",
            vec![q_norm.name.clone(), k_norm.name.clone()],
            kv_state(kv),
            vec![packed.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16)),
        )
        .expect("the fused post produces q")
    }

    /// `kernels::launch_qk_rmsnorm_rope_bf16_rounded`: the per-head q/k
    /// norm + rope pair, in the ROUNDED form.
    ///
    /// gemma-4 rounds where qwen3_5 does not, and bf16 rounding is not
    /// an implementation detail between two kernels that compute the
    /// same function — it is which numbers come out. So the symbol is
    /// the statement, and a family states the one its hand-written pass
    /// fires. In place on q and k; SSA-wise two fresh values.
    pub fn qk_rmsnorm_rope_rounded(
        q: &Val,
        k: &Val,
        q_norm: &NormW,
        k_norm: &NormW,
    ) -> (Val, Val) {
        let shapes = {
            let b = q.t.inner.borrow();
            vec![
                (b.value_shape(q.id), DType::BF16),
                (b.value_shape(k.id), DType::BF16),
            ]
        };
        let ids = q.t.with(q_norm.layer, |b| {
            b.launch(
                "launch_qk_rmsnorm_rope_bf16_rounded",
                vec![q_norm.name.clone(), k_norm.name.clone()],
                None,
                vec![q.id, k.id],
                shapes,
            )
        });
        let mk = |id| Val {
            t: q.t.clone(),
            id,
            layer: q_norm.layer,
        };
        (mk(ids[0]), mk(ids[1]))
    }

    /// `kernels::launch_transpose_bf16_nld_to_lnd`: relay the PLE table
    /// from `[N, L, D]` to `[L, N, D]` so each layer reads a CONTIGUOUS
    /// slice.
    ///
    /// The whole point of the statement is addressing, not arithmetic —
    /// it replaces a per-layer slice-pack kernel with one relay per
    /// fire, which is the driver's own comment at the call site. The
    /// output's leading dim is the LAYER count, a load-time constant, so
    /// the shape is `[Const(layers), Tokens, Const(dim)]`.
    pub fn transpose_nld_to_lnd(x: &Val, layers: u32, dim: u32) -> Val {
        record(
            &x.t,
            None,
            "launch_transpose_bf16_nld_to_lnd",
            vec![],
            None,
            vec![x.id],
            Some((
                Shape(vec![Dim::Const(layers), Dim::Tokens, Dim::Const(dim)]),
                DType::BF16,
            )),
        )
        .expect("the relay produces its value")
    }

    /// `kernels::launch_topk_softmax_bf16`: the router's top-k + softmax +
    /// renormalize, one launch, two results — expert indices
    /// (`[Tokens, k]` i32, the `dyn` value every expert-indexed statement
    /// consumes) and routing weights (`[Tokens, k]` f32).
    ///
    /// The first statement of the MoE branch's CUDA text. The SEMANTIC
    /// [`super::topk`] stays opaque; this one names the kernel, which is
    /// what `lower()` needs before an expert-routed body can be a list of
    /// rectangles rather than residue.
    pub fn topk(logits: &Val, k: u32) -> (Val, Val) {
        let ids = logits.t.with(logits.layer, |b| {
            b.launch(
                "launch_topk_softmax_bf16",
                vec![],
                None,
                vec![logits.id],
                vec![
                    (Shape(vec![Dim::Tokens, Dim::Const(k)]), DType::I32),
                    (Shape(vec![Dim::Tokens, Dim::Const(k)]), DType::F32),
                ],
            )
        });
        let mk = |id| Val {
            t: logits.t.clone(),
            id,
            layer: logits.layer,
        };
        (mk(ids[0]), mk(ids[1]))
    }

    /// `kernels::launch_moe_gate_up_decode_gemv_bf16` /
    /// `..._moe_down_decode_gemv_bf16`: the routed projections of the
    /// decode GEMV leg, one launch each over the fire's `N * k` routes.
    ///
    /// The expert axis is INSIDE the value, not outside it: one launch
    /// reads `experts` and strides the stacked bank itself, so the
    /// declaration stays a rectangle even though the arithmetic is
    /// per-token-per-expert. That is why this leg is the one the CUDA
    /// text can state — see [`super::matmul_per_token`]'s other legs,
    /// which reach the same numbers by *host* routing (the general path)
    /// or by an aligned padding that gives the intermediate an extent no
    /// [`Dim`] spells (the grouped-GEMM path).
    ///
    /// Both projections carry the routed extent as a third dim: `k` is a
    /// load-time constant, so `[Tokens, k, width]` is exactly the
    /// `N * k`-row buffer the kernel writes, said without inventing a
    /// row space.
    pub fn moe_gate_up_gemv(x: &Val, w: &MatW, experts: &Val, top_k: u32) -> Val {
        moe_routed_gemv(
            "launch_moe_gate_up_decode_gemv_bf16",
            x,
            w,
            experts,
            top_k,
        )
    }

    pub fn moe_down_gemv(x: &Val, w: &MatW, experts: &Val, top_k: u32) -> Val {
        moe_routed_gemv("launch_moe_down_decode_gemv_bf16", x, w, experts, top_k)
    }

    fn moe_routed_gemv(kernel: &str, x: &Val, w: &MatW, experts: &Val, top_k: u32) -> Val {
        record(
            &x.t,
            w.layer,
            kernel,
            vec![w.name.clone()],
            None,
            vec![experts.id, x.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(top_k), Dim::Const(w.width)]),
                DType::BF16,
            )),
        )
        .expect("a routed projection produces its value")
    }

    /// `ops::flashinfer_cutlass_moe_bf16`: the whole routed block —
    /// permute, both grouped GEMMs, the activation, and the weighted
    /// finalize — as ONE call.
    ///
    /// This is the leg the decode path actually takes, and it is stated
    /// first because it is the only one that is a single rectangle. Its
    /// `bool` return reads like a runtime fallthrough, but every false
    /// it can produce is decided before the fire: null operands (a
    /// binding question) and `workspace_bytes < needed`, where `needed`
    /// is a pure function of the static dims and `num_rows`, and the
    /// caller has already required `N <= cutlass_max_rows` — the row
    /// count the workspace was sized for. So the leg is a FACT plus a
    /// row bound, not a gamble, and fires above the bound decline rather
    /// than the declaration guessing.
    ///
    /// Consumes the router's two outputs and both expert banks; produces
    /// the combined `[Tokens, hidden]` in one value, which is why the
    /// text that names it has no separate WeightedSum.
    pub fn moe_fused_cutlass(
        x: &Val,
        experts: &Val,
        weights: &Val,
        gate_up: &MatW,
        down: &MatW,
        hidden: u32,
    ) -> Val {
        record(
            &x.t,
            gate_up.layer,
            "ops::flashinfer_cutlass_moe_bf16",
            vec![gate_up.name.clone(), down.name.clone()],
            None,
            vec![x.id, experts.id, weights.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
        )
        .expect("the fused MoE produces its value")
    }

    /// `kernels::launch_residual_add_bf16`: the explicit stream add, for
    /// the legs whose producer wrote to scratch instead of folding.
    pub fn residual_add(x: &Val, residual: &Val, hidden: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "launch_residual_add_bf16",
            vec![],
            None,
            vec![x.id, residual.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
        )
        .expect("the residual add produces its value")
    }

    /// `kernels::launch_sigmoid_dot_scalar_gate_add_bf16`: the shared
    /// expert's landing with its gate logit folded in — one launch that
    /// dots `norm_x` with the `[1, H]` gate row, sigmoids the scalar, and
    /// accumulates `shared` into the stream.
    ///
    /// The general form is a `[Tokens, 1]` GEMM followed by
    /// `launch_sigmoid_scalar_gate_add_bf16`; this fused form runs when
    /// the gate weight is bound unquantized and `N` is within the decode
    /// fast path's bound (1024). Every fire this text covers is under
    /// `cutlass_max_rows` (<= 512), so within the declaration's own row
    /// range the fused form is not a guarded arm but the only arm.
    pub fn sigmoid_dot_scalar_gate_add(
        x: &Val,
        gate: &MatW,
        shared: &Val,
        base: &Val,
        hidden: u32,
    ) -> Val {
        record(
            &x.t,
            gate.layer,
            "launch_sigmoid_dot_scalar_gate_add_bf16",
            vec![gate.name.clone()],
            None,
            vec![x.id, base.id, shared.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
        )
        .expect("the shared-expert landing produces its value")
    }

    /// `kernels::launch_chunked_swiglu_bf16` over the routed rows — the
    /// same kernel [`swiglu`]'s packed arm names, launched with `N * k`
    /// rows instead of `N`. A separate statement because the SHAPE
    /// differs, not the kernel: the routed value keeps its expert dim.
    pub fn swiglu_routed(x: &Val, top_k: u32, intermediate: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "launch_chunked_swiglu_bf16",
            vec![],
            None,
            vec![x.id],
            Some((
                Shape(vec![
                    Dim::Tokens,
                    Dim::Const(top_k),
                    Dim::Const(intermediate),
                ]),
                DType::BF16,
            )),
        )
        .expect("the routed activation produces its value")
    }

    /// `kernels::launch_token_batched_weighted_sum_bf16`, or the
    /// `..._add_bf16` form when the residual folds into the same launch.
    ///
    /// The combine collapses `[Tokens, k, H]` to `[Tokens, H]` under the
    /// router's weights. `fold_residual` is the hand-written pass's
    /// `add_to_residual`: at tp=1 the MoE output lands straight on the
    /// residual stream, so the add is not a second launch. Stating it
    /// here is what lets the body emit ONE op where the semantic text
    /// emits a WeightedSum and a ResidualAdd — the fusion is a kernel
    /// fact, so it belongs in the CUDA reading, not in the trace shape.
    ///
    /// The per-expert `launch_scatter_add_weighted_bf16` loop is the
    /// OTHER combine, and it is not stated here: it runs once per expert
    /// with a row count the host learned from a device readback, which
    /// is a launch count no declaration fixes.
    pub fn weighted_sum(weights: &Val, x: &Val, hidden: u32, residual: Option<&Val>) -> Val {
        let mut inputs = vec![x.id, weights.id];
        if let Some(r) = residual {
            inputs.push(r.id);
        }
        record(
            &weights.t,
            weights.layer,
            if residual.is_some() {
                "launch_token_batched_weighted_sum_add_bf16"
            } else {
                "launch_token_batched_weighted_sum_bf16"
            },
            vec![],
            None,
            inputs,
            Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
        )
        .expect("the combine produces its value")
    }

    /// The MLP activation, stating which of the two swiglu kernels runs.
    ///
    /// `packed` is [`crate::facts::LlamaLikeCudaFacts::gate_up_fused`]: a
    /// checkpoint that bound the packed gate‖up bank lands the projection
    /// in one buffer and takes the CHUNKED kernel; one that did not lands
    /// two and takes the pair form. Same arithmetic, different addressing
    /// — which is exactly the kind of choice that used to sit in the
    /// executor (`declared::arm_swiglu`) and in the generated file (a
    /// per-layer `if (gate_up_fused_N)`), reading a workspace to decide
    /// what the binding had already decided at load.
    ///
    /// One value either way: the trace declares ONE packed matmul before
    /// this, and whether the binding materialised it as one buffer or two
    /// is a BUFFER question, which is `lower::Buffers`'.
    pub fn swiglu(x: &Val, intermediate: u32, packed: bool) -> Val {
        record(
            &x.t,
            x.layer,
            if packed {
                "launch_chunked_swiglu_bf16"
            } else {
                "launch_swiglu_bf16"
            },
            vec![],
            None,
            vec![x.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
                DType::BF16,
            )),
        )
        .expect("the activation produces its value")
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

    /// `ops::launch_attention_flashinfer_prefill` — the PLAN-FREE
    /// prefill wrapper, which builds its own R-shaped plan from the host
    /// indptrs on the way in.
    ///
    /// A DIFFERENT statement from [`attention_flashinfer_prefill`], not
    /// a spelling of it: that one names the dispatch alone and its
    /// caller owes it a plan, while this one owes nothing and cannot be
    /// given a row window (the plan it builds spans all R requests).
    /// gemma-4's prefill fires this; llama_like's fires the other. The
    /// two are one call apart in C++ and a whole contract apart here,
    /// which is why the table carries both.
    pub fn attention_flashinfer_prefill_planless(q: &Val, kv: &Kv) -> Option<Val> {
        attn_at(q, kv, "ops::launch_attention_flashinfer_prefill")
    }

    /// `ops::dispatch_attention_flashinfer_decode` asked for its LSE.
    ///
    /// The SAME symbol as [`attention_flashinfer_decode`] and a
    /// different call: `lse_out` is the last positional argument of
    /// every flashinfer entry point, and the driver passes it only on
    /// layers that carry attention sinks (`layer.attn_sinks != nullptr`,
    /// a load-time per-layer answer). Supplying it costs a per-layer
    /// write, which is why plain Mixtral layers do not — so whether this
    /// statement or the one-value one runs is a FACT, not a branch.
    ///
    /// Produces `(o, lse)`. The LSE is fp32 `[Tokens, q_heads]`, and it
    /// exists so [`attention_sink_rescale`] can apply the
    /// softmax-denominator extension flashinfer's DefaultAttention does
    /// not emit natively.
    pub fn attention_flashinfer_decode_lse(q: &Val, kv: &Kv, q_heads: u32) -> (Val, Val) {
        let shape = q.t.inner.borrow().value_shape(q.id);
        let ids = q.t.with(Some(kv.l), |b| {
            b.launch(
                "dispatch_attention_flashinfer_decode",
                vec![],
                kv_state(kv),
                vec![q.id],
                vec![
                    (shape, DType::BF16),
                    (Shape(vec![Dim::Tokens, Dim::Const(q_heads)]), DType::F32),
                ],
            )
        });
        let mk = |id| Val {
            t: q.t.clone(),
            id,
            layer: q.layer,
        };
        (mk(ids[0]), mk(ids[1]))
    }

    /// `kernels::launch_rope_yarn_original_bf16`: the YaRN-paper rope —
    /// a dim-index ramp between interpolated and extrapolated
    /// frequencies, plus an `attention_factor` magnitude scale.
    ///
    /// A different KERNEL from the plain rope, not a parameterisation:
    /// which one a deployment fires is decided by its config at load and
    /// erases here. The semantic [`super::rope`] carries a `RopeKind`
    /// the lowering refuses for anything but Standard, so a family that
    /// scales says so by naming the launcher.
    pub fn rope_yarn_original(q: &Val, k: &Val) -> (Val, Val) {
        let (q_sh, k_sh) = {
            let b = q.t.inner.borrow();
            (b.value_shape(q.id), b.value_shape(k.id))
        };
        let ids = q.t.with(q.layer, |b| {
            b.launch(
                "launch_rope_yarn_original_bf16",
                vec![],
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

    /// `ops::gemm_act_x_wt_bias_bf16`: a projection whose BIAS RIDES IN
    /// THE EPILOGUE.
    ///
    /// Not a `matmul` plus an [`super::add_bias`]. At decode this routes
    /// to the warp-per-row GEMV, whose epilogue absorbs the bias for
    /// free — so the folded form is one launch where the split form is
    /// two, and the two do not accumulate in the same order. A family
    /// whose driver folds must say so: mixtral folds q/k/v and the
    /// router, and adds `o_bias` separately, which is why gpt-oss's text
    /// uses both spellings and neither by default.
    pub fn gemm_bias(x: &Val, w: &MatW, bias: &MatW) -> Val {
        record(
            &x.t,
            w.layer,
            "ops::gemm_act_x_wt_bias_bf16",
            vec![w.name.clone(), bias.name.clone()],
            None,
            vec![x.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(w.width)]),
                DType::BF16,
            )),
        )
        .expect("a biased projection produces its value")
    }

    /// `ops::launch_attention_flashinfer_prefill` asked for its LSE —
    /// the prefill twin of [`attention_flashinfer_decode_lse`], and the
    /// same argument makes the difference.
    pub fn attention_flashinfer_prefill_lse(q: &Val, kv: &Kv, q_heads: u32) -> (Val, Val) {
        let shape = q.t.inner.borrow().value_shape(q.id);
        let ids = q.t.with(Some(kv.l), |b| {
            b.launch(
                "ops::launch_attention_flashinfer_prefill",
                vec![],
                kv_state(kv),
                vec![q.id],
                vec![
                    (shape, DType::BF16),
                    (Shape(vec![Dim::Tokens, Dim::Const(q_heads)]), DType::F32),
                ],
            )
        });
        let mk = |id| Val {
            t: q.t.clone(),
            id,
            layer: q.layer,
        };
        (mk(ids[0]), mk(ids[1]))
    }

    /// `kernels::launch_attention_sink_rescale_bf16`: `o *= sigmoid(lse
    /// - sink_h)`, in place, per (token, head).
    ///
    /// gpt-oss learns a per-head SINK logit that participates in the
    /// softmax denominator without contributing a value — so the whole
    /// effect is a rescale of the attention output by how much
    /// probability mass the sink would have taken. The sink weight is
    /// `[q_heads]`, which is why it rides in the weight slot.
    pub fn attention_sink_rescale(o: &Val, lse: &Val, sinks: &MatW) -> Val {
        let shape = o.t.inner.borrow().value_shape(o.id);
        record(
            &o.t,
            sinks.layer,
            "launch_attention_sink_rescale_bf16",
            vec![sinks.name.clone()],
            None,
            vec![o.id, lse.id],
            Some((shape, DType::BF16)),
        )
        .expect("the sink rescale produces its value")
    }

    /// `kernels::launch_bf16_to_fp16`: the activation cast the MXFP4
    /// routed GEMVs want on their input.
    ///
    /// A statement rather than an implementation detail of the GEMV
    /// because it is its own launch over its own extent — and because
    /// the routed leg casts TWICE, once on the block input and once on
    /// the post-activation routes, over different extents.
    pub fn bf16_to_fp16(x: &Val) -> Val {
        let shape = x.t.inner.borrow().value_shape(x.id);
        record(
            &x.t,
            x.layer,
            "launch_bf16_to_fp16",
            vec![],
            None,
            vec![x.id],
            Some((shape, DType::F16)),
        )
        .expect("the cast produces its value")
    }

    /// `kernels::launch_mxfp4_moe_gate_up_decode_bf16`: BOTH routed
    /// projections of gpt-oss's fused decode leg, in one launch,
    /// reading the packed 4-bit nibbles straight out of HBM.
    ///
    /// The weight slot names the layer's per-expert POINTER BANK, not a
    /// tensor: the kernel indexes experts through a device array of
    /// pointers plus a parallel scale array. That indirection is a
    /// BINDING — the executor resolves the name to whatever the layer
    /// holds, exactly as [`moe_fused_cutlass`] resolves its two banks —
    /// so it is not the obstacle it looks like. What would be an
    /// obstacle is the host-routed walk this leg replaces: its launch
    /// count depends on which experts the router picked, and no
    /// rectangle spells that.
    ///
    /// Produces `(gate, up)`, each `[Tokens, k, intermediate]` — the
    /// routed extent as a third dim, [`moe_gate_up_gemv`]'s convention.
    pub fn mxfp4_moe_gate_up_decode(
        x: &Val,
        experts: &Val,
        bank: &MatW,
        top_k: u32,
        intermediate: u32,
    ) -> (Val, Val) {
        let shape = || {
            (
                Shape(vec![
                    Dim::Tokens,
                    Dim::Const(top_k),
                    Dim::Const(intermediate),
                ]),
                DType::BF16,
            )
        };
        let ids = x.t.with(bank.layer, |b| {
            b.launch(
                "launch_mxfp4_moe_gate_up_decode_bf16",
                vec![bank.name.clone()],
                None,
                vec![experts.id, x.id],
                vec![shape(), shape()],
            )
        });
        let mk = |id| Val {
            t: x.t.clone(),
            id,
            layer: bank.layer,
        };
        (mk(ids[0]), mk(ids[1]))
    }

    /// `kernels::launch_mxfp4_moe_down_decode_bf16`: the routed down
    /// projection, the same bank convention as
    /// [`mxfp4_moe_gate_up_decode`].
    pub fn mxfp4_moe_down_decode(
        x: &Val,
        experts: &Val,
        bank: &MatW,
        top_k: u32,
        hidden: u32,
    ) -> Val {
        record(
            &x.t,
            bank.layer,
            "launch_mxfp4_moe_down_decode_bf16",
            vec![bank.name.clone()],
            None,
            vec![experts.id, x.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(top_k), Dim::Const(hidden)]),
                DType::BF16,
            )),
        )
        .expect("the routed down projection produces its value")
    }

    /// `kernels::launch_gpt_oss_glu_bf16`: SwiGLU with gpt-oss's CLAMP.
    ///
    /// A different kernel from [`swiglu`], not a parameterisation of it:
    /// `swiglu_limit` is a config constant, so which of the two runs is
    /// decided at load and erases here. Reading it as a runtime scalar
    /// would put a branch in every fire for an answer that never
    /// changes.
    /// Its extent is the ROUTED one — `[Tokens, k, intermediate]`, the
    /// shape of the operands it consumes, not `[Tokens, intermediate]`.
    /// Declaring the collapsed shape made the two `bf16_to_fp16` sites
    /// indistinguishable to anything reading the trace, and the second
    /// one re-cast the block input while the routed activations were
    /// never written — a live defect the ledger, the golden and the
    /// registry all passed.
    pub fn gpt_oss_glu(gate: &Val, up: &Val, top_k: u32, intermediate: u32) -> Val {
        record(
            &gate.t,
            gate.layer,
            "launch_gpt_oss_glu_bf16",
            vec![],
            None,
            vec![gate.id, up.id],
            Some((
                Shape(vec![
                    Dim::Tokens,
                    Dim::Const(top_k),
                    Dim::Const(intermediate),
                ]),
                DType::BF16,
            )),
        )
        .expect("the clamped GLU produces its value")
    }

    /// `ops::launch_attention_naive_paged` — the fallback prefill for a
    /// head dim flashinfer's TC prefill template rejects.
    ///
    /// gemma-4's FULL-attention layers run at head_dim 512, and
    /// flashinfer 0.6.x refuses to instantiate a prefill at
    /// `NUM_MMA_D_QK=32`. So the deployment states a naive paged kernel
    /// on exactly those layers — a per-layer HEAD DIM fact, erased at
    /// trace time, not a runtime fallback the executor discovers.
    pub fn attention_naive_paged(q: &Val, kv: &Kv) -> Option<Val> {
        attn_at(q, kv, "ops::launch_attention_naive_paged")
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

#[cfg(test)]
mod seam_tests {
    use super::seam;
    use crate::trace::{ForwardPlan, GuardArm, GuardPred, Op, OpKind, SeamStatement};

    fn op(kind: OpKind, inputs: Vec<u32>, outputs: Vec<u32>) -> Op {
        Op {
            kind,
            inputs,
            outputs,
            layer: Some(0),
        }
    }

    fn matmul() -> OpKind {
        OpKind::Matmul {
            weight: "layer.0.q_proj".to_string(),
            beta_one: false,
            selector: None,
        }
    }

    fn lora() -> OpKind {
        OpKind::Launch {
            kernel: "pie_lora_qkv_correction".to_string(),
            weights: vec![],
            state: None,
        }
    }

    fn guard() -> OpKind {
        OpKind::Guard {
            arms: vec![GuardArm {
                pred: GuardPred::HasLora,
                ops: 1,
            }],
            else_ops: 0,
        }
    }

    /// `q` and `v` from projections, seam immediately after: the shape
    /// every live text has.
    fn well_placed() -> Vec<Op> {
        vec![
            op(matmul(), vec![], vec![1]),
            op(matmul(), vec![], vec![2]),
            op(guard(), vec![], vec![]),
            op(lora(), vec![1, 2], vec![]),
        ]
    }

    fn plan(ops: Vec<Op>) -> ForwardPlan {
        ForwardPlan {
            family: "test".to_string(),
            values: vec![],
            ops,
            depth_window: false,
            seams: vec![SeamStatement {
                seam: "attn.qv".to_string(),
                layer: Some(0),
                op: Some(2),
            }],
        }
    }

    /// The adapter's position rule FIRES. Without this the live traces'
    /// clean check proves only that the walk found nothing to look at.
    #[test]
    fn the_adapter_position_rule_is_not_vacuous() {
        assert!(seam::check_plan(&plan(well_placed())).is_empty());

        // A bias consuming q BEFORE the seam: the delta would land on
        // base + bias. This is the shape the live A/B caught.
        let mut ops = well_placed();
        ops.insert(
            2,
            op(
                OpKind::AddBias {
                    weight: "layer.0.q_bias".to_string(),
                },
                vec![1],
                vec![3],
            ),
        );
        let mut p = plan(ops);
        p.seams[0].op = Some(3);
        let problems = seam::check_plan(&p);
        assert_eq!(problems.len(), 1, "{problems:#?}");
        assert!(problems[0].contains("AddBias"), "{}", problems[0]);

        // A seam placed after rope: different arithmetic, and the
        // producer is no longer a projection.
        let ops = vec![
            op(matmul(), vec![], vec![1]),
            op(matmul(), vec![], vec![2]),
            op(OpKind::Rope { kind: crate::trace::RopeKind::Standard, partial: None }, vec![1], vec![3]),
            op(guard(), vec![], vec![]),
            op(lora(), vec![3, 2], vec![]),
        ];
        let mut p = plan(ops);
        p.seams[0].op = Some(3);
        let problems = seam::check_plan(&p);
        assert_eq!(problems.len(), 1, "{problems:#?}");
        assert!(problems[0].contains("Rope"), "{}", problems[0]);
    }

    /// Every seam a text may state is declared, and its `sees` arity is
    /// what the statement passes.
    #[test]
    fn the_seam_table_is_complete() {
        for d in seam::ALL {
            assert_eq!(seam::by_name(d.name).map(|x| x.name), Some(d.name));
        }
        assert_eq!(seam::ATTN_QV.sees, &["q", "v"]);
        assert!(seam::ATTN_Q.sink.is_some(), "the page-mask sink is declared");
    }
}
