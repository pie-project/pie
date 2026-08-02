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
    pub fn append(&self, k: &Val, v: &Val) {
        self.t
            .with(Some(self.l), |b| b.kv_append(self.l, k.id, v.id));
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
