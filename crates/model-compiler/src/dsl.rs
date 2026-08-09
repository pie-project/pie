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

use crate::facts::QkNorm;
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

/// How a weight is STORED, and therefore which kernel can read it.
///
/// The polymorphism axis quantization needs, and it belongs on the
/// WEIGHT rather than on the statement for the reason [`NormW`] already
/// gives about its variant: the weight knows. `x @ Wᵀ` is one piece of
/// arithmetic whatever W is made of; what changes is the kernel that
/// can read W, and a kernel is what a declaration STATES.
///
/// Today the driver picks that kernel — `make_weight_view` builds a
/// descriptor from a per-layer struct the statement never mentions, and
/// `gemm::act_x_w` routes on it. Every defect this arc found was that
/// shape: the driver knowing something the statement did not.
///
/// The scales and zero-points are WEIGHTS, so a quantized statement
/// names more of them. A `Launch` already carries a list.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WeightRepr {
    /// Dense, read directly. The implicit `WeightView(const DeviceTensor&)`.
    #[default]
    Bf16,
    /// Scaled storage — the weight's own dtype says int4/int8/fp8, and
    /// these say where the scales live (`QuantMeta`'s three layouts).
    Scaled {
        layout: ScaleLayout,
        /// Elements per scale under [`ScaleLayout::PerGroup`]; 0 otherwise.
        group: u32,
        /// Which axis [`ScaleLayout::PerChannel`] runs along.
        axis: u32,
        /// The checkpoint carries zero-points beside the scales.
        zero_point: bool,
    },
    /// MXFP4 with E8M0 block scales — gpt-oss's expert banks, and the
    /// one representation whose scales are not a separate layout
    /// question.
    Mxfp4Marlin,
}

/// Where a scaled weight's scales apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ScaleLayout {
    PerTensor,
    PerChannel,
    PerGroup,
}

/// A matmul weight handle: name, out width, layer, and how it is
/// STORED. Built by the [`Layer`] namespace; the declaration never
/// spells any of them.
#[derive(Clone)]
pub struct MatW {
    pub name: String,
    pub width: u32,
    pub layer: Option<u32>,
    /// Dense unless the deployment's facts say otherwise. Defaulted so
    /// that a text which has never met a quantized checkpoint reads
    /// exactly as it did.
    pub repr: WeightRepr,
}

impl MatW {
    /// The dense handle, which is what every text builds today.
    pub fn dense(name: String, width: u32, layer: Option<u32>) -> MatW {
        MatW {
            name,
            width,
            layer,
            repr: WeightRepr::Bf16,
        }
    }

    /// The same weight, stored some other way. The scale and
    /// zero-point tensors are named by CONVENTION off the weight's own
    /// name, which is how the loader already finds them.
    pub fn with_repr(mut self, repr: WeightRepr) -> MatW {
        self.repr = repr;
        self
    }

    /// The extra tensors this representation makes the statement name.
    /// Empty for [`WeightRepr::Bf16`], which is why a dense statement
    /// carries one weight and a quantized one carries three.
    pub fn scale_names(&self) -> Vec<String> {
        match &self.repr {
            WeightRepr::Bf16 => Vec::new(),
            WeightRepr::Mxfp4Marlin => vec![format!("{}.scales", self.name)],
            WeightRepr::Scaled { zero_point, .. } => {
                let mut out = vec![format!("{}.scales", self.name)];
                if *zero_point {
                    out.push(format!("{}.zeros", self.name));
                }
                out
            }
        }
    }

    /// The launcher symbol a statement over this weight STATES.
    ///
    /// `None` for the dense case, which still records the semantic
    /// [`crate::trace::OpKind::Matmul`] — that kind fans to exactly one
    /// kernel per backend, so nothing is being chosen. A quantized one
    /// fans to several, so it names which.
    pub fn gemm_symbol(&self) -> Option<&'static str> {
        match &self.repr {
            WeightRepr::Bf16 => None,
            WeightRepr::Mxfp4Marlin => Some("gemm::act_x_wt_mxfp4_marlin"),
            WeightRepr::Scaled {
                layout: ScaleLayout::PerGroup,
                ..
            } => Some("gemm::act_x_wt_grouped_scaled"),
            WeightRepr::Scaled {
                layout: ScaleLayout::PerChannel,
                ..
            } => Some("gemm::act_x_wt_channel_scaled"),
            WeightRepr::Scaled {
                layout: ScaleLayout::PerTensor,
                ..
            } => Some("gemm::act_x_wt_tensor_scaled"),
        }
    }
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
    /// A layer's KV handle. `pub` because a declaration -- which lives in
    /// `crates/model` -- names it directly when it builds its own weight
    /// namespace rather than taking `M`'s.
    pub fn at(t: &Trace, l: u32) -> Kv {
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
    /// A layer's recurrent-state handle. `pub` for the reason `Kv::at` is.
    pub fn at(t: &Trace, l: u32) -> Rs {
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
    /// The PACKED gate‖up bank, for a deployment whose loader join
    /// materialised one.
    pub gate_up: MatW,
    /// The two halves, for a deployment whose did not. A text states
    /// either the packed handle or this pair — never both — and which
    /// is a binding fact (`gate_up_fused`).
    pub gate_proj: MatW,
    pub up_proj: MatW,
    pub down: MatW,
    /// The router: hidden -> one logit per expert.
    ///
    /// A MIXTURE's handles, and they live on the same `Layer` as the dense
    /// ones for the reason `ModelShape` gives about itself: a namespace is
    /// shared and a text states the shape its facts describe. A dense
    /// deployment simply never names these, exactly as a deployment whose
    /// loader did not join gate and up never names `gate_up`.
    pub router: MatW,
    /// The expert banks. `MatW::name` carries no expert index -- the routed
    /// kernel indexes the bank by the slot it read, which is what makes it
    /// ONE weight rather than `n_experts` of them.
    pub expert_gate: MatW,
    pub expert_up: MatW,
    pub expert_down: MatW,
    /// The dense expert a mixture may also have, and the scalar gate that
    /// blends it in (`shared_expert_combine`).
    pub shared_gate: MatW,
    pub shared_up: MatW,
    pub shared_down: MatW,
    pub shared_gate_proj: MatW,
    pub attn_norm: NormW,
    pub mlp_norm: NormW,
    pub q_norm: NormW,
    pub k_norm: NormW,
    pub kv: Kv,
}

/// The facts a DENSE TRANSFORMER's weight namespace is built from: an
/// embedding, a stack of `qkv`/`o_proj`/`gate_up`/`down` layers with their
/// norms, and a readout.
///
/// Deliberately not one family's facts type, and that is the whole point.
/// Every field here is true of any dense transformer; nothing about llama's
/// rope, its qk-norm placement or its fused-QKV binding appears, because
/// those reach a family's text as its own parameters. What [`M`] offers is
/// the namespace, and the namespace is shared.
///
/// It exists as a separate struct so a declaration can live OUTSIDE this
/// crate: the toolchain cannot name a family's facts type without the
/// dependency pointing the wrong way. Each family projects into it once —
/// see `LlamaLikeFacts::shape`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelShape {
    /// Width of the residual stream.
    pub hidden: u32,
    /// Width of the MLP's inner dimension. `gate_up` is twice this.
    pub intermediate: u32,
    /// Experts in a mixture's bank, and 0 for a dense deployment.
    ///
    /// The three mixture numbers live on the shared shape rather than one
    /// family's facts for the reason the rest of this struct does: a routed
    /// FFN is true of many architectures (qwen3-moe, gemma-4, gpt-oss) and
    /// nothing about how any of them ROUTES appears here.
    pub n_experts: u32,
    /// One expert's inner width.
    pub moe_intermediate: u32,
    /// The dense expert's inner width, and 0 for a mixture without one.
    pub shared_intermediate: u32,
    /// Readout width.
    pub vocab: u32,
    /// One attention head's width — the per-head qk-norm's row length.
    pub head_dim: u32,
    /// `q_heads * head_dim`.
    pub q_width: u32,
    /// `kv_heads * head_dim`.
    pub kv_width: u32,
    /// Whether the q/k norms are per-head, row-wise, or absent.
    pub qk_norm: QkNorm,
    /// Which norm every norm in the namespace is.
    pub norm_variant: NormVariant,
    /// The readout reads the embedding table rather than its own weight.
    pub tied_embeddings: bool,
    /// How this deployment STORES its linear projections.
    ///
    /// The namespace's field rather than each text's, because it is the
    /// same answer for every handle [`M::layer`] hands out — a
    /// checkpoint quantizes uniformly — and because a text that had to
    /// repeat it per projection would be a text that could get one
    /// wrong. `Bf16` is the reading every family had before the axis
    /// existed.
    pub proj_repr: WeightRepr,
}

/// The model context a declaration runs against: the shape and the tape.
///
/// It carries NO lowering. A model text is written for one backend
/// (`.wiki/tart/dsl.md` ③: the model file is
/// `families/<family>/<backend>.rs`), so "am I lowered?" is not a
/// question a body can ask — the semantic text and the CUDA text are two
/// texts, and each states its own kernels unconditionally. What used to
/// be `m.lowering()` is now the CUDA text's own parameters.
pub struct M {
    t: Trace,
    f: ModelShape,
}

impl Val {
    /// The tape this value was recorded on — what a seam statement
    /// needs when the text has no [`M`] or bare [`Trace`] in hand.
    pub fn trace(&self) -> &Trace {
        &self.t
    }

    /// The layer this value belongs to, or `None` in the prologue and
    /// epilogue. A text needs it when it opens a value-producing guard
    /// AROUND an existing value — the guard's own value must be tagged
    /// the same way, or the depth axis would treat the two differently.
    pub fn layer(&self) -> Option<u32> {
        self.layer
    }
}

impl M {
    pub fn shape(&self) -> &ModelShape {
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
            repr: f.proj_repr,
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
            qkv: mat("qkv", f.q_width + 2 * f.kv_width),
            q_proj: mat("q_proj", f.q_width),
            k_proj: mat("k_proj", f.kv_width),
            v_proj: mat("v_proj", f.kv_width),
            q_bias: mat("q_bias", f.q_width),
            k_bias: mat("k_bias", f.kv_width),
            v_bias: mat("v_bias", f.kv_width),
            o_proj: mat("o_proj", f.hidden),
            gate_up: mat("gate_up", 2 * f.intermediate),
            gate_proj: mat("gate_proj", f.intermediate),
            up_proj: mat("up_proj", f.intermediate),
            down: mat("down", f.hidden),
            router: mat("router", f.n_experts),
            expert_gate: mat("expert_gate", f.moe_intermediate),
            expert_up: mat("expert_up", f.moe_intermediate),
            expert_down: mat("expert_down", f.hidden),
            shared_gate: mat("shared_gate", f.shared_intermediate),
            shared_up: mat("shared_up", f.shared_intermediate),
            shared_down: mat("shared_down", f.hidden),
            // One number per row: how much of the shared expert to keep.
            shared_gate_proj: mat("shared_gate_proj", 1),
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

/// Run a SEMANTIC declaration: no kernel is stated, and the consumer (Metal,
/// the site table, `declared_dag`) chooses.
///
/// `family` is the plan's name and nothing more — the caller's, because the
/// caller is the family. This used to be the literal `"llama_like"`, which
/// was the last family name baked into the toolchain.
pub fn trace_semantic(family: &str, shape: &ModelShape, body: impl FnOnce(&mut M)) -> ForwardPlan {
    run(family.to_string(), shape, body)
}

/// Run a LOWERED declaration for CUDA — one per [`FireClass`], the family
/// name recording which launch form this trace serves. The body takes the
/// backend facts as its own parameter; nothing about the lowering reaches it
/// through [`M`].
///
/// The recorded name is `<family>.cuda.<class>`, which is what
/// [`Backend::of_family`](crate::kernels::Backend::of_family) reads the
/// backend out of.
pub fn trace_cuda(
    family: &str,
    shape: &ModelShape,
    class: FireClass,
    body: impl FnOnce(&mut M),
) -> ForwardPlan {
    run(format!("{family}.cuda.{}", class_word(class)), shape, body)
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
/// (`crates/driver-metal/csrc/src/model/llama_like/declared_dag.hpp`), which is the
/// same "the driver decides" shape the CUDA side is being cured of.
/// This entry is the seam that text will be written against; until it
/// is, nothing calls it, and the empty Metal kernel table means the
/// first thing that does must declare its kernels.
pub fn trace_metal(
    family: &str,
    shape: &ModelShape,
    class: FireClass,
    body: impl FnOnce(&mut M),
) -> ForwardPlan {
    run(format!("{family}.metal.{}", class_word(class)), shape, body)
}

fn class_word(class: FireClass) -> &'static str {
    match class {
        FireClass::Decode => "decode",
        FireClass::Prefill => "prefill",
        // The service classes belong to a text that declares an MTP repair
        // pass, and such a text composes its own trace rather than going
        // through `M`. A caller that reaches here has asked a whole-model
        // prologue/epilogue for a class it has no shape for.
        FireClass::CommitAdvance | FireClass::StateOnly | FireClass::FrozenVerify => {
            panic!("the {class:?} service class has no whole-model form here")
        }
    }
}

fn run(family: String, shape: &ModelShape, body: impl FnOnce(&mut M)) -> ForwardPlan {
    let mut m = M {
        t: Trace {
            inner: Rc::new(RefCell::new(TraceBuilder::new(family))),
        },
        f: *shape,
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

/// `x[index]` — the window of a value along its LEADING dim.
///
/// Launches nothing. The value it produces is the operand's bytes at an
/// offset, which `lower::Buffers` computes; `lower` emits no rectangle.
///
/// It exists because gemma3n's AltUp needs it and no other family does:
/// `altup_predict` produces all `k` streams, the layer body runs on ONE,
/// and in `gemma3n.cpp` that is `predictions + active * N * H` — a
/// pointer offset with no kernel behind it. Stating it as an op rather
/// than hiding it in a `Val` method is deliberate: which window the body
/// reads is a fact about the MODEL, and a reader following the dataflow
/// has to see it.
///
/// The shape drops the leading dim. Selecting from a rank-1 value is
/// refused — there would be nothing left to name.
pub fn select(x: &Val, index: u32) -> Val {
    let id = x.t.with(x.layer, |b| b.select(x.id, index));
    Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    }
}

/// `y = x @ Wᵀ`, over a weight stored however [`MatW::repr`] says.
///
/// POLYMORPHIC OVER THE REPRESENTATION, and that is the whole of the
/// quantization axis. A dense weight records the semantic
/// [`crate::trace::OpKind::Matmul`] — one arithmetic, one kernel per
/// backend, nothing chosen. Any other representation records a stated
/// `Launch` naming the kernel that can read it, plus the scale and
/// zero-point tensors as extra WEIGHTS.
///
/// So the driver never sees a descriptor and never routes: it binds the
/// names the statement gives it and calls the symbol the statement
/// names. `make_weight_view` and `gemm::act_x_w`'s internal dispatch
/// have nothing left to decide.
pub fn matmul(x: &Val, w: &MatW) -> Val {
    let id = match w.gemm_symbol() {
        None => x.t.with(w.layer, |b| b.matmul(x.id, &w.name, w.width)),
        Some(symbol) => {
            let mut weights = vec![w.name.clone()];
            weights.extend(w.scale_names());
            let shape = Shape(vec![Dim::Tokens, Dim::Const(w.width)]);
            let outs = x.t.with(w.layer, |b| {
                b.launch(
                    symbol,
                    weights,
                    None,
                    vec![x.id],
                    vec![(shape, DType::BF16)],
                )
            });
            outs[0]
        }
    };
    Val {
        t: x.t.clone(),
        id,
        layer: w.layer,
    }
}

/// Row RMSNorm, or per-head RMSNorm when the weight handle says so.
///
/// SEMANTIC: this is the backend-independent spelling, and it stays one
/// because a trace with no backend cannot name a CUDA symbol —
/// `check_plan` refuses it, correctly. The stated form is
/// [`cuda::rmsnorm`], which is what a `*.cuda.*` text calls.
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
///
/// Takes the TAPE, not a model context: a guard is a statement about what
/// gets recorded, and nothing about it is one family's. This used to have an
/// `&M`-taking twin whose whole body was `guarded(&m.t)`.
pub fn guarded(t: &Trace) -> GuardCtx {
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
pub fn guard(t: &Trace, pred: crate::trace::GuardPred, then_f: impl FnOnce(), else_f: impl FnOnce()) {
    guarded(t).arm(pred, then_f).otherwise(else_f);
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

/// Which discipline an arm of [`regions`] follows.
///
/// The Guard/Peel unification is of the SURFACE, and this enum is where
/// that is said out loud: one construct in the text, arms that state
/// their own rule. Read `lower::Lowering::select`'s doc before assuming
/// these can collapse into one predicate vocabulary — the obvious
/// generalisation (a fire fact is just a row predicate that holds for all
/// rows or none) was implemented once and shipped a real defect, caught
/// by the live shadow comparison. `.wiki/tart/dsl.md` migration step 2
/// carries the argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    /// EXCLUSIVE, over the whole window: the first `Fire` arm whose
    /// predicate holds runs, and the rest do not. A `GuardPred` names a
    /// property of the FIRE, and the arm it selects is a kernel choice
    /// for the whole op list.
    Fire(crate::trace::GuardPred),
    /// A PARTITION: this arm covers the rows its predicate names, the
    /// rest covers the others, and BOTH run. Moving an axis from `Fire`
    /// to `Rows` is a deliberate change in what the text says, never a
    /// reinterpretation a backend performs.
    Rows(RowPred),
}

/// The arms of a [`regions`] construct.
pub struct RegionsCtx<'t> {
    guard: Option<GuardCtx>,
    rows: Option<RowsCtx<'t>>,
    t: &'t Trace,
    layer: Option<u32>,
    shape: Option<(Shape, DType)>,
    out: Option<Val>,
}

impl<'t> RegionsCtx<'t> {
    /// One arm, and what runs in it.
    ///
    /// The FIRST arm fixes the construct's discipline, because the IR
    /// underneath is still two ops (`Guard` and `Peel`) and neither can
    /// express a mix. A mixed chain is a real thing to want — a fire
    /// choice inside one side of a row split — and the text already
    /// expresses it by NESTING, which is what the IR merge in migration
    /// step 6 is for. Asking for it in one flat chain is refused here
    /// rather than silently flattened into whichever op was opened first.
    pub fn arm(&mut self, pred: Region, f: impl FnOnce()) {
        match pred {
            Region::Fire(p) => {
                assert!(
                    self.rows.is_none(),
                    "regions: a Fire arm after a Rows arm — one flat chain                      cannot be both disciplines (nest instead; the IR merge                      is migration step 6)"
                );
                let g = self.guard.take().unwrap_or_else(|| {
                    let (idx, outs) = {
                        let mut b = self.t.inner.borrow_mut();
                        b.set_layer(self.layer);
                        b.open_guard(self.shape.clone().into_iter().collect())
                    };
                    self.out = outs.first().map(|v| Val {
                        t: self.t.clone(),
                        id: *v,
                        layer: self.layer,
                    });
                    GuardCtx {
                        t: self.t.clone(),
                        idx,
                        arms: Vec::new(),
                        emitted: 0,
                    }
                });
                self.guard = Some(g.arm(p, f));
            }
            Region::Rows(p) => {
                assert!(
                    self.guard.is_none(),
                    "regions: a Rows arm after a Fire arm — one flat chain                      cannot be both disciplines (nest instead; the IR merge                      is migration step 6)"
                );
                let ctx = self.rows.get_or_insert_with(|| {
                    let (idx, outs) = {
                        let mut b = self.t.inner.borrow_mut();
                        b.set_layer(self.layer);
                        b.open_peel(
                            self.shape.clone().into_iter().collect(),
                            crate::trace::PeelWindow::HookFreePrefix,
                        )
                    };
                    self.out = outs.first().map(|v| Val {
                        t: self.t.clone(),
                        id: *v,
                        layer: self.layer,
                    });
                    RowsCtx {
                        t: self.t,
                        idx,
                        prefix: None,
                        pred: None,
                    }
                });
                ctx.arm(p, f);
            }
        }
    }

    /// Every case the arms did not name.
    fn close(mut self, f: impl FnOnce()) -> Option<Val> {
        if let Some(g) = self.guard.take() {
            g.otherwise(f);
        } else if let Some(mut r) = self.rows.take() {
            r.rest(f);
        } else {
            panic!("regions states at least one arm before its rest");
        }
        self.out
    }
}

/// ONE construct for both region disciplines (`.wiki/tart/dsl.md`
/// migration step 2).
///
/// `by_rows` and `guarded_value` were two spellings of "some statements
/// run and some do not", and a reader had to know which mechanism a
/// family had reached for before they could read the arm. This is the
/// single surface: arms, each stating its own discipline, and a rest.
///
/// It lowers to today's two IR ops unchanged — a `Fire`-armed chain to
/// `Guard`, a `Rows`-armed one to `Peel` — so the goldens pin that this
/// surface changed no traced byte. Merging THOSE is migration step 6, and
/// it is a separate change with a separate gate: the IR carries the
/// discipline, so nothing here has to guess it later.
pub fn regions(
    t: &Trace,
    layer: Option<u32>,
    shape: Option<(Shape, DType)>,
    build: impl FnOnce(&mut RegionsCtx<'_>),
    rest: impl FnOnce(),
) -> Option<Val> {
    let mut ctx = RegionsCtx {
        guard: None,
        rows: None,
        t,
        layer,
        shape,
        out: None,
    };
    build(&mut ctx);
    ctx.close(rest)
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
pub fn guard_on(
    t: &Trace,
    pred: crate::trace::GuardPred,
    then_f: impl FnOnce(),
    else_f: impl FnOnce(),
) {
    guarded(t).arm(pred, then_f).otherwise(else_f);
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
    // The values the statement NAMES. Carried onto the record so buffer
    // assignment can pin exactly these, rather than guessing from the
    // operands of whatever op the seam points at.
    let ids: Vec<crate::trace::ValueId> = sees.iter().map(|v| v.id).collect();
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
                .push_seam(def.name, layer, Some(at as u32), ids);
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
            t.inner.borrow_mut().push_seam(def.name, layer, Some(at), ids);
        }
        "in" | "out" => {
            t.inner.borrow_mut().push_seam(def.name, layer, None, ids);
        }
        other => unreachable!("no seam named {other}"),
    }
}

// ── Metal kernel signatures ────────────────────────────────────────────

/// The METAL launchers a lowered declaration may state — the `dsl::cuda`
/// of the second backend (`.wiki/tart/dsl.md` ②).
///
/// UNVERIFIED (2026-08-05). Every symbol here is an MSL entrypoint read
/// off the driver's source (`crates/driver-metal/csrc/src/batch/decode_psos.cpp`'s
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
        with_params(t, layer, kernel, weights, state, Vec::new(), inputs, out)
    }

    /// [`record`], plus the scalars the symbol's row names.
    ///
    /// A kernel takes numbers no operand shape gives — a projection's two
    /// extents, a norm's epsilon, an attention's strides. The row says which
    /// slot wants which; this is where the statement supplies them, and the
    /// order is the row's `Param(i)` order.
    ///
    /// A float rides as its bits (`f32::to_bits`) and the row reads it back
    /// with `ParamF32`: the channel is untyped `u32` and what each slot means
    /// is the symbol's contract, which is the row.
    #[allow(clippy::too_many_arguments)]
    fn with_params(
        t: &Trace,
        layer: Option<u32>,
        kernel: &str,
        weights: Vec<String>,
        state: Option<StateRef>,
        params: Vec<u32>,
        inputs: Vec<crate::trace::ValueId>,
        out: Option<(Shape, DType)>,
    ) -> Option<Val> {
        let ids = t.with(layer, |b| {
            b.launch_with_params(
                kernel,
                weights,
                state,
                params,
                inputs,
                out.into_iter().collect(),
            )
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
    pub fn embed_gather(
        t: &Trace,
        weight: &str,
        hidden: u32,
        multi_batch: bool,
        repr: WeightRepr,
        point: &str,
    ) -> Val {
        // ALWAYS the M>1 symbol, and `multi_batch` is deliberately unread.
        //
        // `embed_gather_4bit` reads `id[0]` and writes `out[hidden]` — one row,
        // by construction, whatever grid it is handed. The class is not the
        // question: a DECODE of four requests is four rows, so a text that
        // picks by class names the single-row gather for a four-row fire and
        // three lanes get nothing. Measured against a real checkpoint: one of
        // four readout lanes held anything, and bisecting the fire put the
        // stop at statement ZERO.
        //
        // The mb variant's own comment says it "reduces to embed_gather_4bit
        // at N=1", so naming it unconditionally is not a widening — it is the
        // same kernel with the row read from the grid instead of assumed.
        let _ = multi_batch;
        let stem = "embed_gather_mb_4bit";
        with_params(
            t,
            None,
            &format!("{stem}{point}"),
            quant_table(weight, repr),
            None,
            vec![hidden],
            vec![],
            Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
        )
        .expect("embed produces the residual stream")
    }

    /// `rms_norm.metal::rms_single_row_bfloat16` — ONE entrypoint for
    /// every norm this family states (attn_norm, mlp_norm, q_norm,
    /// k_norm, final_norm; the driver fans five `Kernel` kinds onto it).
    pub fn rms_norm(x: &Val, w: &NormW, row: u32, eps: f32) -> Val {
        let out = same_shape(x);
        with_params(
            &x.t,
            w.layer,
            "rms_single_row_bfloat16",
            vec![w.name.clone()],
            None,
            // `RmsParams`, field for field: eps, axis_size, w_stride,
            // plus_one, gain.
            //
            // `w_stride` is ONE, and the distance between this and the `row`
            // it used to say is the whole of a wrong answer. It is the stride
            // between consecutive CHANNELS of the gain vector -- `ws[w_stride
            // * i]` in the shader -- and a contiguous row's channels are one
            // apart. `rms.metal`'s own header says `w_stride=1`.
            //
            // Passing the axis made every norm read `w[2048 * i]`: it strode
            // out of the gain vector on the second channel and multiplied by
            // whatever followed it in the checkpoint. Measured against MLX at
            // position zero, channel 1 came out -0.016 where the reference
            // says +0.052 -- the wrong SIGN, from the wrong tensor, on the
            // second statement of the fire.
            //
            // `plus_one` is the `(1 + w)` reading gemma takes and this family
            // does not; the gain is unity.
            vec![
                eps.to_bits(),
                row,
                1,
                u32::from(w.variant == crate::trace::NormVariant::Gemma),
                1.0f32.to_bits(),
            ],
            vec![x.id],
            Some(out),
        )
        .expect("a norm produces its value")
    }

    /// The tensors a quantized projection reads: the packed weight, then
    /// its scales and zero point.
    ///
    /// An affine kernel takes THREE buffers and the statements here used to
    /// name one, which left the driver to derive the other two from a naming
    /// convention it had to know. `dsl::matmul` already states the triplet
    /// for the same reason its own doc gives — *"the driver never sees a
    /// descriptor and never routes: it binds the names the statement gives it
    /// and calls the symbol the statement names"* — and the Metal statements
    /// now say the same thing.
    /// The instantiation point an affine entrypoint is compiled at.
    ///
    /// `quantized_qmv.metal` stamps one template over
    /// `(activation dtype × group size × bit width)`, so the symbol a
    /// statement names is `affine_qmv_fast_bfloat16_gs_64_b_4` and not the
    /// stem. A stem does not resolve — which is the GOOD failure: the runtime
    /// compiler reports it by listing what the shader does export, where a
    /// WRONG point would compile and read the wrong bytes (the `_d_256`
    /// defect, one axis over).
    ///
    /// Both numbers come from the deployment's facts. Nothing here derives
    /// them: g64/b8 and g128/b4 pack to identical shapes, so no tensor can be
    /// asked.
    /// The GEMM's instantiation point: [`affine_point`] plus its tile.
    ///
    /// `affine_qmm_t` is stamped over `(group × bits × bm × bn)`, so its
    /// symbol carries two more numbers than the GEMV's.
    #[must_use]
    pub fn affine_gemm_point(repr: WeightRepr, bits: u32, tile: (u32, u32)) -> String {
        let (bm, bn) = tile;
        format!("{}_bm_{bm}_bn_{bn}", affine_point(repr, bits))
    }

    #[must_use]
    pub fn affine_point(repr: WeightRepr, bits: u32) -> String {
        let group = match repr {
            WeightRepr::Scaled { group, .. } => group,
            _ => 0,
        };
        format!("_bfloat16_gs_{group}_b_{bits}")
    }

    /// A value's row width, from the shape the trace already carries.
    ///
    /// A projection's INPUT extent, which no fact states and no operand
    /// carries — the statement's own operand does, and this reads it. Zero for
    /// a shape whose trailing dim is not a constant, which is a shape no
    /// projection here has.
    fn in_width(x: &Val) -> u32 {
        match x.t.inner.borrow().value_shape(x.id).0.last() {
            Some(Dim::Const(n)) => *n,
            _ => 0,
        }
    }

    fn quant_weights(w: &MatW) -> Vec<String> {
        let mut out = vec![w.name.clone()];
        out.extend(w.scale_names());
        out
    }

    /// The same triplet for a table the text names by STRING rather than
    /// through a [`MatW`] handle — the embedding and the readout.
    ///
    /// They take a `repr` for the same reason a projection does: the symbols
    /// are `embed_gather_4bit` and `affine_qmv_fast`, both affine, both
    /// reading three tensors.
    fn quant_table(name: &str, repr: WeightRepr) -> Vec<String> {
        quant_weights(&MatW {
            name: name.to_string(),
            width: 0,
            layer: None,
            repr,
        })
    }

    /// `quantized_qmv.metal::affine_qmv_fast` — the projection GEMV,
    /// M=1. The driver fans every projection kind onto it.
    pub fn qmv(x: &Val, w: &MatW, point: &str) -> Val {
        with_params(
            &x.t,
            w.layer,
            &format!("affine_qmv_fast{point}"),
            quant_weights(w),
            None,
            // The GEMV's two extents: the row it reads and the row it writes.
            // A projection told its output is zero wide computes nothing and
            // reports success, which is why these are stated and not derived.
            vec![in_width(x), w.width],
            vec![x.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(w.width)]), DType::BF16)),
        )
        .expect("a projection produces its value")
    }

    /// `quantized_qmv.metal::affine_qmv_fast_residual` — the same GEMV
    /// with the block residual folded into its epilogue, which is what a
    /// `beta_one` matmul is on this backend.
    pub fn qmv_residual(x: &Val, w: &MatW, residual: &Val, point: &str) -> Val {
        with_params(
            &x.t,
            w.layer,
            &format!("affine_qmv_fast_residual{point}"),
            quant_weights(w),
            None,
            vec![in_width(x), w.width],
            vec![x.id, residual.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(w.width)]), DType::BF16)),
        )
        .expect("a folded projection produces its value")
    }

    /// `quantized_qmm_t.metal::affine_qmm_t` — MLX's steel quantized
    /// GEMM, the M>1 projection.
    pub fn qmm(x: &Val, w: &MatW, point: &str) -> Val {
        with_params(
            &x.t,
            w.layer,
            &format!("affine_qmm_t{point}"),
            quant_weights(w),
            None,
            // The GEMV's two extents: the row it reads and the row it writes.
            // A projection told its output is zero wide computes nothing and
            // reports success, which is why these are stated and not derived.
            vec![in_width(x), w.width],
            vec![x.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(w.width)]), DType::BF16)),
        )
        .expect("a projection produces its value")
    }

    /// `quantized_qmm_t.metal::affine_qmm_t_residual`.
    pub fn qmm_residual(x: &Val, w: &MatW, residual: &Val, point: &str) -> Val {
        with_params(
            &x.t,
            w.layer,
            &format!("affine_qmm_t_residual{point}"),
            quant_weights(w),
            None,
            vec![in_width(x), w.width],
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

    /// `rope/rope.metal::neox_decode_bfloat16` (M=1) /
    /// `neox_mb_bfloat16` (M>1). One dispatch for q and k together,
    /// as the plan states it (`declared_dag.hpp`'s `Kind::Rope`).
    pub fn rope(
        q: &Val,
        k: &Val,
        multi_batch: bool,
        theta: f32,
        scale: f32,
        head_dim: u32,
        table: bool,
    ) -> (Val, Val) {
        (
            rope_one(q, multi_batch, theta, scale, head_dim, table),
            rope_one(k, multi_batch, theta, scale, head_dim, table),
        )
    }

    /// One tensor's rotation — which is what the kernel does.
    ///
    /// `rope_neox_decode` takes ONE `device T* x` and rotates it in place.
    /// This helper used to state a single launch carrying q and k, two inputs
    /// and two results, on the strength of a comment saying the DAG spells it
    /// as one `Kind::Rope`. The DAG spells one KIND and dispatches it twice;
    /// the trace stated one LAUNCH, so the second tensor was never rotated.
    ///
    /// Nothing could see it until the rows carried their operands: a statement
    /// whose shape disagrees with its kernel's is invisible to every check
    /// that only asks whether the symbol exists.
    ///
    /// In place, so the result is the operand: the row states `x` as its
    /// `Out(0)` and the same buffer is read and written.
    fn rope_one(
        x: &Val,
        multi_batch: bool,
        theta: f32,
        scale: f32,
        head_dim: u32,
        table: bool,
    ) -> Val {
        // A deployment that RESCALES its frequency ladder cannot state a base:
        // llama-3 rescales piecewise and YaRN rescales differently, and both
        // are tables. The driver derives one at load and answers it as
        // `Source::RopeFrequencies`, so the statement's job is only to say
        // WHICH form this deployment takes.
        if table {
            return with_params(
                &x.t,
                x.layer,
                "neox_freqs_decode_bfloat16",
                vec![],
                None,
                // Scale, head width, and YaRN's `mscale` -- one for llama-3,
                // whose rescaling lives entirely in the frequencies.
                vec![scale.to_bits(), head_dim, 1.0f32.to_bits()],
                vec![x.id],
                Some(same_shape(x)),
            )
            .expect("a rotation produces its value");
        }
        let kernel = if multi_batch {
            "neox_mb_bfloat16"
        } else {
            "neox_decode_bfloat16"
        };
        with_params(
            &x.t,
            x.layer,
            kernel,
            vec![],
            None,
            // The rotation's scale, its log2 base and the head width. The base
            // is `log2(theta)` because the shader raises two to it —
            // `rope_neox_geometric_body` — and handing it theta rotates by a
            // frequency ladder that is wrong from the second channel on.
            vec![scale.to_bits(), theta.log2().to_bits(), head_dim],
            vec![x.id],
            Some(same_shape(x)),
        )
        .expect("a rotation produces its value")
    }

    /// `attn/split_qkv.metal::split_qkv_bf16`: deinterleave the packed QKV
    /// projection `[rows, q_width + 2*kv_width]` into three buffers.
    ///
    /// # Why this exists beside `dsl::split_qkv`
    ///
    /// The generic `split_qkv` records an `OpKind::SplitQkv`, which carries
    /// the two widths *in the op kind*. A driver could read them — by
    /// matching on `OpKind`, which is exactly what "nothing in the driver may
    /// choose a kernel" forbids: the widths would reach the kernel because the
    /// driver knew what a QKV split is.
    ///
    /// So the Metal text states the launch outright, and the widths ride the
    /// channel built for them — [`OpKind::Launch::params`], whose own doc says
    /// *"a scalar that has nowhere to ride is a scalar the DRIVER re-derives
    /// from its config. That is the thing this arc removes."* The driver then
    /// forwards `params` to every kernel that states them, knowing nothing
    /// about what they mean.
    ///
    /// [`OpKind::Launch::params`]: crate::trace::OpKind::Launch
    pub fn split_qkv(packed: &Val, q_width: u32, kv_width: u32) -> (Val, Val, Val) {
        let rows = packed.t.inner.borrow().value_shape(packed.id).0[0];
        let out = |w: u32| (Shape(vec![rows, Dim::Const(w)]), DType::BF16);
        let ids = packed.t.with(packed.layer, |b| {
            b.launch_with_params(
                "split_qkv_bf16",
                vec![],
                None,
                vec![q_width, kv_width],
                vec![packed.id],
                vec![out(q_width), out(kv_width), out(kv_width)],
            )
        });
        let mk = |id| Val {
            t: packed.t.clone(),
            id,
            layer: packed.layer,
        };
        (mk(ids[0]), mk(ids[1]), mk(ids[2]))
    }

    /// `kv_append.metal::kv_append_bfloat16` (contiguous) /
    /// `kv_append_paged.metal::kv_append_paged_bfloat16` (page table).
    pub fn kv_append(k: &Val, v: &Val, kv: &Kv, paged: bool, head_dim: u32, kv_heads: u32) {
        let kernel = if paged {
            "kv_append_paged_bfloat16"
        } else {
            "kv_append_bfloat16"
        };
        with_params(
            &kv.t,
            Some(kv.l),
            kernel,
            vec![],
            kv_state(kv),
            // The model's two: how wide a head is and how many there are. The
            // pool's strides come from the ROW (`KvHeadStride`, `KvSeqStride`)
            // because they are the shape the driver allocated, not the shape
            // the model has.
            vec![head_dim, kv_heads],
            vec![k.id, v.id],
            None,
        );
    }

    /// `sdpa_vector.metal::sdpa_vector_decode_bfloat16_d_<head_dim>` (M=1) /
    /// `sdpa_paged.metal::sdpa_paged_decode_bfloat16_d_<head_dim>` (M>1).
    ///
    /// The width is the deployment's, not a literal. It used to be `_d_256`
    /// unconditionally, which is wrong for every checkpoint whose heads are
    /// narrower — `qwen3_0_6b`'s are 128 — and wrong in the way that does not
    /// fault: a 256-wide kernel over 128-wide heads reads past the end of
    /// every head and answers with whatever is there. `PARITY-BATCH.md`
    /// records the same defect in the C++ llama walk, where `_d128` was a
    /// literal that strode 64-wide heads past their end.
    ///
    /// Both kernels instantiate `_d_64`, `_d_128` and `_d_256`; the paged one
    /// also `_d_512`. A width neither carries has no kernel, and the symbol
    /// this returns will simply not resolve — which the driver's
    /// `every_symbol_the_lowering_names_has_a_row` check reports by name.
    #[allow(clippy::too_many_arguments)]
    pub fn sdpa(
        q: &Val,
        kv: &Kv,
        q_width: u32,
        head_dim: u32,
        paged: bool,
        gqa_factor: u32,
        kv_heads: u32,
        window: i32,
    ) -> Option<Val> {
        let kernel = if paged {
            format!("sdpa_paged_decode_bfloat16_d_{head_dim}")
        } else {
            format!("sdpa_vector_decode_bfloat16_d_{head_dim}")
        };
        let kernel = kernel.as_str();
        // The model's scalars, in the order both rows name them. The strides
        // and the page size are the POOL's and come from the row; the mask
        // stride is zero because this text states no custom mask.
        //
        // The scale is `1/sqrt(head_dim)` — the softmax temperature, and the
        // one number here a reader is most likely to assume the kernel knows.
        // It does not: it takes it, and a zero makes every logit zero and
        // every attention uniform.
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        with_params(
            &q.t,
            Some(kv.l),
            kernel,
            vec![],
            kv_state(kv),
            vec![
                gqa_factor,
                kv_heads,
                scale.to_bits(),
                0,
                window as u32,
            ],
            vec![q.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16)),
        )
    }

    /// `silu_mul.metal::silu_mul_bfloat16` — the SwiGLU activation over
    /// the packed gate/up bank.
    pub fn silu_mul(gate: &Val, up: &Val, intermediate: u32) -> Val {
        record(
            &gate.t,
            gate.layer,
            "silu_mul_bfloat16",
            vec![],
            None,
            vec![gate.id, up.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
                DType::BF16,
            )),
        )
        .expect("the activation produces its value")
    }

    /// `quantized_qmv.metal::affine_qmv_fast` against the lm head — the
    /// readout, `[Requests, vocab]` f32 like every family's.
    pub fn lm_head(x: &Val, weight: &str, vocab: u32, repr: WeightRepr, point: &str) -> Val {
        with_params(
            &x.t,
            None,
            &format!("affine_qmv_fast{point}"),
            quant_table(weight, repr),
            None,
            vec![in_width(x), vocab],
            vec![x.id],
            // BF16, because that is what the kernel WRITES. `affine_qmv_fast`
            // is instantiated at bfloat and its output is `device T*`; the
            // readout is not special-cased to widen.
            //
            // Stating F32 here sized the arena slot for four bytes an element
            // and the kernel filled two, so the logits region came back
            // EXACTLY half zero -- 64128 of 128256 -- with every surviving
            // value a fraction of its real magnitude. A dtype the trace states
            // and the kernel disagrees with is not a rounding difference; it
            // is a stride, and every value after the first is at the wrong
            // address.
            Some((Shape(vec![Dim::Requests, Dim::Const(vocab)]), DType::BF16)),
        )
        .expect("the readout produces the logits")
    }

    // ── The mixture. ──
    //
    // Six statements, and the reason they are six rather than one is the
    // reason a mixture is interesting at all: a routed FFN's SHAPE depends on
    // a value the fire computes. The router picks experts, the sort groups
    // rows by the expert they picked, the gather materializes those groups
    // contiguously, the matmuls run over the groups, and the combine puts the
    // rows back where they started weighted by the router's confidence.
    //
    // Nothing here is a per-family branch. The executor walks these exactly as
    // it walks a projection: symbol, row, file, rule, grid, operands. What is
    // different is only that `LaunchRule::RouteRows` and `RoutedQmv` read
    // `n_experts` and `experts_per_token` off the dims -- which is the same
    // way `Qmv` reads `width`.

    /// `moe/route.metal::router_topk` — which experts a row goes to, and how
    /// much of each.
    ///
    /// Two outputs: the expert slots and their weights. Both are read by name
    /// downstream, which is why this returns the pair rather than folding them.
    pub fn router_topk(
        logits: &Val,
        n_experts: u32,
        experts_per_token: u32,
        scaled: bool,
    ) -> (Val, Val) {
        let sym = if scaled { "router_topk_scaled_bfloat16" } else { "router_topk_bfloat16" };
        let slots = Dim::Const(experts_per_token);
        let ids = logits.t.with(logits.layer, |b| {
            b.launch_with_params(
                sym,
                vec![],
                None,
                // `RouterParams`, packed: the shader takes a struct pointer.
                vec![n_experts, experts_per_token],
                vec![logits.id],
                vec![
                    (Shape(vec![Dim::Tokens, slots]), DType::I32),
                    (Shape(vec![Dim::Tokens, slots]), DType::BF16),
                ],
            )
        });
        let mk = |id| Val { t: logits.t.clone(), id, layer: logits.layer };
        (mk(ids[0]), mk(ids[1]))
    }

    /// `moe/route.metal::route_sort` — group the rows by expert.
    ///
    /// FOUR outputs, and a text that named fewer would leave the combine
    /// reading whatever was in the buffer: the permutation, the per-row
    /// expert, the per-tile expert, and the inverse the combine reads back.
    pub fn route_sort(
        expert_ids: &Val,
        n_experts: u32,
        experts_per_token: u32,
        tile_rows: u32,
        padded: u32,
        width: u32,
    ) -> (Val, Val, Val, Val) {
        let pad = Dim::Const(padded);
        let ids = expert_ids.t.with(expert_ids.layer, |b| {
            b.launch_with_params(
                "route_sort",
                vec![],
                None,
                // `MoeRouteParams`, packed and SHARED with the gather so the
                // sort's padding and the gather's bounds cannot disagree.
                vec![padded, n_experts, experts_per_token, tile_rows, padded, width, width],
                vec![expert_ids.id],
                vec![
                    (Shape(vec![pad]), DType::I32),
                    (Shape(vec![pad]), DType::I32),
                    (Shape(vec![Dim::Const(padded.div_ceil(tile_rows.max(1)))]), DType::I32),
                    (Shape(vec![pad]), DType::I32),
                ],
            )
        });
        let mk = |id| Val { t: expert_ids.t.clone(), id, layer: expert_ids.layer };
        (mk(ids[0]), mk(ids[1]), mk(ids[2]), mk(ids[3]))
    }

    /// `moe/route.metal::route_gather` — the rows, in expert order.
    #[allow(clippy::too_many_arguments)]
    pub fn route_gather(
        x: &Val,
        perm: &Val,
        n_experts: u32,
        experts_per_token: u32,
        tile_rows: u32,
        padded: u32,
        width: u32,
    ) -> Val {
        with_params(
            &x.t,
            x.layer,
            "route_gather",
            vec![],
            None,
            vec![padded, n_experts, experts_per_token, tile_rows, padded, width, width],
            vec![x.id, perm.id],
            Some((Shape(vec![Dim::Const(padded), Dim::Const(width)]), DType::BF16)),
        )
        .expect("the gather produces its rows")
    }

    /// `quant/qmv.metal::affine_qmv_routed` — the expert-selecting GEMV.
    ///
    /// `sel = row * slots_per_row + slot`, which is why the launch's row and
    /// slot axes are not interchangeable and why `slots_per_row` is stated.
    pub fn routed_qmv(
        x: &Val,
        expert_ids: &Val,
        w: &MatW,
        experts_per_token: u32,
        biased: bool,
    ) -> Val {
        let sym = if biased {
            "affine_qmv_routed_bias_bfloat16_gs_64_b_4"
        } else {
            "affine_qmv_routed_bfloat16_gs_64_b_4"
        };
        let in_w = in_width(x);
        with_params(
            &x.t,
            w.layer,
            sym,
            quant_weights(w),
            None,
            vec![in_w, w.width, 0, in_w, experts_per_token],
            vec![x.id, expert_ids.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(w.width)]), DType::BF16)),
        )
        .expect("a routed projection produces its value")
    }

    /// `moe/route.metal::combine_sorted` — the rows back where they started,
    /// weighted by the router.
    pub fn combine_sorted(
        y: &Val,
        expert_weights: &Val,
        inv: &Val,
        experts_per_token: u32,
        width: u32,
    ) -> Val {
        with_params(
            &y.t,
            y.layer,
            "combine_sorted",
            vec![],
            None,
            // `ExpertCombineParams`, packed.
            vec![width, experts_per_token],
            vec![y.id, expert_weights.id, inv.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
        )
        .expect("the combine produces its rows")
    }

    /// `moe/route.metal::shared_expert_combine` — `routed + sigmoid(gate) *
    /// shared`, the landing for a mixture that also has a dense expert.
    pub fn shared_expert_combine(
        routed: &Val,
        shared: &Val,
        gate: &Val,
        width: u32,
    ) -> Val {
        with_params(
            &routed.t,
            routed.layer,
            "shared_expert_combine",
            vec![],
            None,
            vec![width],
            vec![routed.id, shared.id, gate.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
        )
        .expect("the shared landing produces its rows")
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

    /// A launch that produces MORE THAN ONE value.
    ///
    /// `TraceBuilder::launch` always returned a `Vec`; [`record`] narrowed it
    /// to the first, which was right for every statement until MLA. Its
    /// prepare splits a latent KV row into four -- `kv_c`, `k_pe`, `q_nope`,
    /// `q_pe` -- and a statement returning one of them would leave the other
    /// three unnamed on the tape, which is exactly the silent dataflow gap
    /// the trace exists to make visible.
    /// [`record_many`], plus the scalar arguments — [`record_with_params`]
    /// for a statement with more than one result.
    fn record_many_with_params(
        t: &Trace,
        layer: Option<u32>,
        kernel: &str,
        weights: Vec<String>,
        params: Vec<u32>,
        inputs: Vec<crate::trace::ValueId>,
        outs: Vec<(Shape, DType)>,
    ) -> Vec<Val> {
        let n = outs.len();
        let ids = t.with(layer, |b| {
            b.launch_with_params(kernel, weights, None, params, inputs, outs)
        });
        assert_eq!(ids.len(), n, "the tape recorded a different arity than stated");
        ids.into_iter()
            .map(|id| Val { t: t.clone(), id, layer })
            .collect()
    }

    fn record_many(
        t: &Trace,
        layer: Option<u32>,
        kernel: &str,
        weights: Vec<String>,
        inputs: Vec<crate::trace::ValueId>,
        outs: Vec<(Shape, DType)>,
    ) -> Vec<Val> {
        let n = outs.len();
        let ids = t.with(layer, |b| b.launch(kernel, weights, None, inputs, outs));
        assert_eq!(ids.len(), n, "the tape recorded a different arity than stated");
        ids.into_iter()
            .map(|id| Val { t: t.clone(), id, layer })
            .collect()
    }

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

    /// [`record`], plus the SCALAR ARGUMENTS the symbol takes that no
    /// operand shape gives ([`crate::trace::OpKind::Launch`]'s params).
    ///
    /// Signed values ride as their two's complement: `window_left = -1`
    /// is `0xFFFFFFFF`, and the executor casts back. The channel is
    /// untyped on purpose -- what each slot means is the SYMBOL's
    /// contract, exactly as `aux_names`' slots are.
    fn record_with_params(
        t: &Trace,
        layer: Option<u32>,
        kernel: &str,
        weights: Vec<String>,
        state: Option<StateRef>,
        params: Vec<u32>,
        inputs: Vec<crate::trace::ValueId>,
        out: Option<(Shape, DType)>,
    ) -> Option<Val> {
        let ids = t.with(layer, |b| {
            b.launch_with_params(
                kernel,
                weights,
                state,
                params,
                inputs,
                out.into_iter().collect(),
            )
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

    /// `kernels::rope::rope_standard_table`: build the fire's cos/sin
    /// table, once. A value, not a latch — the fused-QKV kernel consumes
    /// it as an operand.
    pub fn rope_standard_table(t: &Trace, head_dim: u32) -> Val {
        record(
            t,
            None,
            "rope::rope_standard_table",
            vec![],
            None,
            vec![],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(head_dim)]),
                DType::F32,
            )),
        )
        .expect("table launch produces a value")
    }

    /// `kernels::attn::qkv_decode_qk_norm_rope_write_kv_bf16`: the fused
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
            "attn::qkv_decode_qk_norm_rope_write_kv_bf16",
            vec![q_norm.name.clone(), k_norm.name.clone()],
            kv_state(kv),
            inputs,
            Some((Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16)),
        )
        .expect("fused post produces q")
    }

    // ── the DEVICE-WINDOW forms ────────────────────────────────────
    //
    // A hooked pure-decode fire is graph-CAPTURED, and its hook split rides a
    // DEVICE word (`win_d`) rather than a host row range. That is what makes
    // these their own statements: the window is not a number the lowering
    // knows, so it cannot be expressed as a rectangle -- every one is
    // `whole`, and for a reason no other `whole` row in this table gives.
    //
    // `.wiki/tart/dsl.md`'s step 2e found this path by surveying before
    // deleting; these are the statements that survey was about.

    /// `kernels::rope::qk_rmsnorm_rope_bf16_devwin`: the fused q/k norm and
    /// rope, over a device-carried window.
    pub fn qk_rmsnorm_rope_devwin(q: &Val, k: &Val, q_w: &str, k_w: &str, q_width: u32) -> Val {
        record(
            &q.t,
            q.layer,
            "rope::qk_rmsnorm_rope_bf16_devwin",
            vec![q_w.to_string(), k_w.to_string()],
            None,
            vec![q.id, k.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16)),
        )
        .expect("the norm+rope produces its value")
    }

    /// `kernels::attn::qkv_decode_qk_norm_rope_write_kv_bf16_devwin`: the
    /// fused decode QKV epilogue, over a device-carried window.
    pub fn qkv_decode_fused_devwin(packed: &Val, l: u32, q_width: u32) -> Val {
        record(
            &packed.t,
            Some(l),
            "attn::qkv_decode_qk_norm_rope_write_kv_bf16_devwin",
            vec![],
            Some(StateRef { store: StateStore::KvCache, layer: l }),
            vec![packed.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16)),
        )
        .expect("the fused epilogue produces its value")
    }

    /// `kernels::attn::write_kv_to_pages_bf16_devwin`: the page write, over
    /// a device-carried window.
    pub fn write_kv_to_pages_devwin(k: &Val, v: &Val, l: u32) {
        record(
            &k.t,
            Some(l),
            "attn::write_kv_to_pages_bf16_devwin",
            vec![],
            Some(StateRef { store: StateStore::KvCache, layer: l }),
            vec![k.id, v.id],
            None,
        );
    }

    /// `kernels::attn::write_kv_explicit_bf16_devwin`: the explicit-slot
    /// write, over a device-carried window.
    pub fn write_kv_explicit_devwin(k: &Val, v: &Val, l: u32) {
        record(
            &k.t,
            Some(l),
            "attn::write_kv_explicit_bf16_devwin",
            vec![],
            Some(StateRef { store: StateStore::KvCache, layer: l }),
            vec![k.id, v.id],
            None,
        );
    }

    // ── head-dim padding, and the rest ─────────────────────────────

    /// `kernels::attn::pad_head_dim_bf16` / `kernels::attn::strip_head_dim_bf16`:
    /// widen each head to the padded width a kernel demands, and narrow back.
    ///
    /// The pair is what `head_dim_padded` COSTS, and stating it is what turns
    /// `if (c.head_dim_padded)` in the model body into a fact the trace
    /// carries. Row-shaped: each token's heads are padded independently.
    pub fn pad_head_dim(x: &Val, heads: u32, head_dim_padded: u32) -> Val {
        record(
            &x.t, x.layer, "attn::pad_head_dim_bf16", vec![], None, vec![x.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(heads), Dim::Const(head_dim_padded)]),
                DType::BF16,
            )),
        )
        .expect("the pad produces its value")
    }

    /// The inverse of [`Self::pad_head_dim`].
    pub fn strip_head_dim(x: &Val, heads: u32, head_dim: u32) -> Val {
        record(
            &x.t, x.layer, "attn::strip_head_dim_bf16", vec![], None, vec![x.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(heads), Dim::Const(head_dim)]),
                DType::BF16,
            )),
        )
        .expect("the strip produces its value")
    }

    /// `kernels::attn::merge_attention_states_bf16`: merge partial attention outputs by
    /// their log-sum-exps.
    ///
    /// The KV-split's other half. `whole` -- it merges `num_index_sets`
    /// partials whose boundaries are the split's, not a row range's.
    pub fn merge_attention_states(v: &Val, s: &Val, heads: u32, head_dim: u32) -> (Val, Val) {
        let outs = record_many(
            &v.t, v.layer, "attn::merge_attention_states_bf16", vec![], vec![v.id, s.id],
            vec![
                (Shape(vec![Dim::Tokens, Dim::Const(heads), Dim::Const(head_dim)]), DType::BF16),
                (Shape(vec![Dim::Tokens, Dim::Const(heads)]), DType::F32),
            ],
        );
        let mut it = outs.into_iter();
        let merged = it.next().expect("the merge states two outputs");
        let lse = it.next().expect("the merge states two outputs");
        (merged, lse)
    }

    /// `kernels::attn::compact_page_csr`: drop the pages a keep-mask
    /// excludes, rewriting the CSR.
    ///
    /// `whole`: it rewrites `[R+1]` indptr arrays, so a row window would
    /// compact the wrong requests' page lists.
    pub fn compact_page_csr(t: &Trace, l: u32, keep: &Val) -> Val {
        record(
            t, Some(l), "attn::compact_page_csr", vec![],
            Some(StateRef { store: StateStore::KvCache, layer: l }),
            vec![keep.id],
            Some((Shape(vec![Dim::Requests]), DType::I32)),
        )
        .expect("the compaction produces its value")
    }

    /// `kernels::attn::attn_score_fold_heads`: fold the captured per-head scores
    /// into the per-request form an observer reads.
    pub fn attn_score_fold_heads(scores: &Val, heads: u32) -> Val {
        record(
            &scores.t, scores.layer, "attn::attn_score_fold_heads", vec![], None,
            vec![scores.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(heads)]), DType::F32)),
        )
        .expect("the fold produces its value")
    }

    /// `kernels::gemm::mla_absorb_q_to_latent_bf16`: project the query into the latent
    /// space MLA attends in.
    ///
    /// A cuBLAS op, not a raw launch -- and that is why the vocabulary audit
    /// missed it twice: a launcher is anything that issues DEVICE work, and
    /// there are two ways to do that here.
    pub fn mla_absorb_q_to_latent(q_nope: &Val, w: &str, heads: u32, kv_lora_rank: u32) -> Val {
        record(
            &q_nope.t, q_nope.layer, "gemm::mla_absorb_q_to_latent_bf16",
            vec![w.to_string()], None, vec![q_nope.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(heads), Dim::Const(kv_lora_rank)]),
                DType::BF16,
            )),
        )
        .expect("the absorb produces its value")
    }

    /// `kernels::gemm::mla_absorb_latent_to_v_bf16`: project the latent attention
    /// output back to the value space.
    pub fn mla_absorb_latent_to_v(latent: &Val, w: &str, heads: u32, v_head_dim: u32) -> Val {
        record(
            &latent.t, latent.layer, "gemm::mla_absorb_latent_to_v_bf16",
            vec![w.to_string()], None, vec![latent.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(heads), Dim::Const(v_head_dim)]),
                DType::BF16,
            )),
        )
        .expect("the absorb produces its value")
    }

    /// `kernels::ssm::flashinfer_mamba_ssu_bf16`: FlashInfer's selective state update.
    ///
    /// The other mamba scan -- nemotron_h takes this on sm90+ and its own
    /// batched kernel elsewhere. `whole` for the reasons every state scan is.
    pub fn flashinfer_mamba_ssu(conv_out: &Val, dt: &Val, l: u32, intermediate: u32) -> Val {
        record(
            &conv_out.t, Some(l), "ssm::flashinfer_mamba_ssu_bf16", vec![],
            Some(StateRef { store: StateStore::RecurrentState, layer: l }),
            vec![conv_out.id, dt.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(intermediate)]), DType::BF16)),
        )
        .expect("the scan produces its value")
    }

    /// `kernels::gemm::act_x_wt_bf16_cublas`: the plain cuBLAS GEMM, named.
    pub fn gemm_cublas(act: &Val, w: &str, n: u32) -> Val {
        record(
            &act.t, act.layer, "gemm::act_x_wt_bf16_cublas",
            vec![w.to_string()], None, vec![act.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(n)]), DType::BF16)),
        )
        .expect("the gemm produces its value")
    }

    /// `kernels::gemm::act_x_wt_bf16_out_fp32`: the same, accumulating to fp32.
    pub fn gemm_out_fp32(act: &Val, w: &str, n: u32) -> Val {
        record(
            &act.t, act.layer, "gemm::act_x_wt_bf16_out_fp32",
            vec![w.to_string()], None, vec![act.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(n)]), DType::F32)),
        )
        .expect("the gemm produces its value")
    }

    /// `kernels::gemm::act_x_wt_bf16`: the plain `x · Wᵀ`.
    ///
    /// Stated because families FIRE it — glm5's projections, nemotron_h's,
    /// qwen3_5's router. It went missing from the table for as long as it
    /// did because it is an `inline void` forwarder in `ops/gemm.hpp`, and
    /// the audit's launcher regex required the return type to start the
    /// line; the fix to that regex is what surfaced this.
    ///
    /// Distinct from the ordinary `Matmul` op: this is the CUDA reading
    /// for a projection whose weight the family names directly rather
    /// than through the `layer.{l}.{field}` binding — the DSA indexer's,
    /// for one.
    pub fn gemm_xwt(act: &Val, w: &str, n: u32) -> Val {
        record(
            &act.t, act.layer, "gemm::act_x_wt_bf16",
            vec![w.to_string()], None, vec![act.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(n)]), DType::BF16)),
        )
        .expect("the gemm produces its value")
    }

    /// `kernels::gemm::batched_act_x_wt_bf16`: one GEMM per pointer-array entry.
    ///
    /// `whole` for the same reason `gemm_grouped` is, and then some: the
    /// batch is addressed through DEVICE pointer arrays built for the
    /// whole fire (`kernels::moe::build_moe_ptrs_aligned_bf16` fills them), so a
    /// row window would leave every pointer aimed at a row the window does
    /// not own. This is the MoE aligned leg's projection on a deployment
    /// whose shape the grouped kernel refuses.
    pub fn gemm_batched_xwt(act: &Val, w: &str, n: u32) -> Val {
        record(
            &act.t, act.layer, "gemm::batched_act_x_wt_bf16",
            vec![w.to_string()], None, vec![act.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(n)]), DType::BF16)),
        )
        .expect("the gemm produces its value")
    }

    /// `kernels::gemm::grouped_act_x_wt_bf16`: one GEMM per group, batched.
    ///
    /// `whole`: the group boundaries (`M_array`) are fire-global, so a row
    /// window would cut a group in half.
    pub fn gemm_grouped(act: &Val, w: &str, n: u32) -> Val {
        record(
            &act.t, act.layer, "gemm::grouped_act_x_wt_bf16",
            vec![w.to_string()], None, vec![act.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(n)]), DType::BF16)),
        )
        .expect("the gemm produces its value")
    }

    /// `kernels::mlp::sigmoid_scalar_gate_add_bf16`: add `x` onto `out`,
    /// each row scaled by its own sigmoid gate.
    pub fn sigmoid_scalar_gate_add(out: &Val, x: &Val, gate: &Val, hidden: u32) -> Val {
        record(
            &out.t, out.layer, "mlp::sigmoid_scalar_gate_add_bf16", vec![], None,
            vec![out.id, x.id, gate.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
        )
        .expect("the gated add produces its value")
    }

    /// `kernels::layout::split_bf16_rows`: split `[N, l+r]` into `[N, l]` and
    /// `[N, r]`. The inverse of [`Self::concat_rows`].
    pub fn split_rows(src: &Val, left_dim: u32, right_dim: u32) -> (Val, Val) {
        let outs = record_many(
            &src.t, src.layer, "layout::split_bf16_rows", vec![], vec![src.id],
            vec![
                (Shape(vec![Dim::Tokens, Dim::Const(left_dim)]), DType::BF16),
                (Shape(vec![Dim::Tokens, Dim::Const(right_dim)]), DType::BF16),
            ],
        );
        let mut it = outs.into_iter();
        let l = it.next().expect("the split states two outputs");
        let r = it.next().expect("the split states two outputs");
        (l, r)
    }

    /// `kernels::layout::split_qwen_gdn_ba_bf16`: split the GDN `ba`
    /// projection into its beta and alpha halves.
    pub fn split_qwen_gdn_ba(ba: &Val, v_h: u32) -> (Val, Val) {
        let outs = record_many(
            &ba.t, ba.layer, "layout::split_qwen_gdn_ba_bf16", vec![], vec![ba.id],
            vec![
                (Shape(vec![Dim::Tokens, Dim::Const(v_h)]), DType::BF16),
                (Shape(vec![Dim::Tokens, Dim::Const(v_h)]), DType::BF16),
            ],
        );
        let mut it = outs.into_iter();
        let b = it.next().expect("the split states two outputs");
        let a = it.next().expect("the split states two outputs");
        (b, a)
    }

    // ── qwen3_5: multi-token prediction ────────────────────────────
    //
    // MTP drafts several tokens per step and repairs when a draft is
    // rejected, which needs two things the rest of the model does not: an
    // attention that can see a HISTORY buffer alongside the pages (the
    // rejected tokens are not committed), and a per-slot pending-hidden
    // shuffle. All four address through `slot_ids` or `qo_indptr`, so all
    // four are `whole`.

    /// `kernels::attn::attention_mtp_paged_history_bf16`: attend the pages AND
    /// an uncommitted history buffer.
    ///
    /// The draft's own tokens are not in the cache yet -- committing them
    /// before they are accepted is the thing MTP must not do -- so they are
    /// passed beside it and the kernel reads both.
    pub fn attention_mtp_paged_history(q: &Val, l: u32, heads: u32, head_dim: u32) -> Val {
        record(
            &q.t,
            Some(l),
            "attn::attention_mtp_paged_history_bf16",
            vec![],
            Some(StateRef {
                store: StateStore::KvCache,
                layer: l,
            }),
            vec![q.id],
            Some((
                Shape(vec![
                    Dim::Tokens,
                    Dim::Const(heads),
                    Dim::Const(head_dim),
                ]),
                DType::BF16,
            )),
        )
        .expect("the attention produces its value")
    }

    /// `kernels::attn::mtp_shift_hidden_bf16`: the previous step's pending
    /// hidden, shifted into this step's rows.
    pub fn mtp_shift_hidden(target: &Val, pending: &Val, hidden: u32) -> Val {
        record(
            &target.t,
            target.layer,
            "attn::mtp_shift_hidden_bf16",
            vec![],
            None,
            vec![target.id, pending.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
        )
        .expect("the shift produces its value")
    }

    /// `kernels::attn::mtp_update_pending_hidden_bf16`: stash each request's
    /// last hidden for the next step.
    pub fn mtp_update_pending_hidden(target: &Val, l: u32) {
        record(
            &target.t,
            Some(l),
            "attn::mtp_update_pending_hidden_bf16",
            vec![],
            Some(StateRef {
                store: StateStore::RecurrentState,
                layer: l,
            }),
            vec![target.id],
            None,
        );
    }

    /// `kernels::layout::copy_if_valid_slot`: a copy that skips requests
    /// whose slot id is invalid.
    ///
    /// The graph-safe shape: the launch happens for every request every
    /// time, and the slot id decides whether it does anything -- so the
    /// dispatch is fixed and a CUDA graph replays.
    pub fn copy_if_valid_slot(src: &Val, l: u32, width: u32) -> Val {
        record(
            &src.t,
            Some(l),
            "layout::copy_if_valid_slot",
            vec![],
            Some(StateRef {
                store: StateStore::RecurrentState,
                layer: l,
            }),
            vec![src.id],
            Some((
                Shape(vec![Dim::Requests, Dim::Const(width)]),
                DType::BF16,
            )),
        )
        .expect("the copy produces its value")
    }

    // ── qwen3_5: the single-request GDN entries ────────────────────
    //
    // The unbatched twins of the `_batched` forms above: a legacy parity
    // entrypoint and a single-request fast path. Same recurrence, one
    // request, so they are not `whole` for the reason the batched ones are
    // not -- their `B` is the batch, not a window into one.

    /// `kernels::ssm::recurrent_gated_delta_step`: one decode step,
    /// single request.
    pub fn gdn_step_single(q: &Val, l: u32, heads: u32, v_dim: u32) -> Val {
        record(
            &q.t,
            Some(l),
            "ssm::recurrent_gated_delta_step",
            vec![],
            Some(StateRef {
                store: StateStore::RecurrentState,
                layer: l,
            }),
            vec![q.id],
            Some((
                Shape(vec![
                    Dim::Requests,
                    Dim::Const(heads),
                    Dim::Const(v_dim),
                ]),
                DType::F32,
            )),
        )
        .expect("the step produces its value")
    }

    /// The same, with the state kept in bf16.
    ///
    /// A precision BINDING, not a variant: which one a deployment uses is a
    /// load-time fact, exactly as the `_batched` pair above states it.
    pub fn gdn_step_single_state_bf16(q: &Val, l: u32, heads: u32, v_dim: u32) -> Val {
        record(
            &q.t,
            Some(l),
            "ssm::recurrent_gated_delta_step_state_bf16",
            vec![],
            Some(StateRef {
                store: StateStore::RecurrentState,
                layer: l,
            }),
            vec![q.id],
            Some((
                Shape(vec![
                    Dim::Requests,
                    Dim::Const(heads),
                    Dim::Const(v_dim),
                ]),
                DType::F32,
            )),
        )
        .expect("the step produces its value")
    }

    /// `kernels::ssm::chunk_gated_delta_prefill`: the chunked prefill,
    /// single request.
    pub fn gdn_prefill_single(q: &Val, l: u32, heads: u32, v_dim: u32) -> Val {
        record(
            &q.t,
            Some(l),
            "ssm::chunk_gated_delta_prefill",
            vec![],
            Some(StateRef {
                store: StateStore::RecurrentState,
                layer: l,
            }),
            vec![q.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(heads), Dim::Const(v_dim)]),
                DType::F32,
            )),
        )
        .expect("the prefill produces its value")
    }

    /// The same, with the state kept in bf16.
    pub fn gdn_prefill_single_state_bf16(q: &Val, l: u32, heads: u32, v_dim: u32) -> Val {
        record(
            &q.t,
            Some(l),
            "ssm::chunk_gated_delta_prefill_state_bf16",
            vec![],
            Some(StateRef {
                store: StateStore::RecurrentState,
                layer: l,
            }),
            vec![q.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(heads), Dim::Const(v_dim)]),
                DType::F32,
            )),
        )
        .expect("the prefill produces its value")
    }

    /// `kernels::ssm::causal_conv1d_prefill_bf16`: the prefill conv,
    /// single request.
    pub fn causal_conv1d_prefill_single(x: &Val, weight: &str, l: u32, channels: u32) -> Val {
        record(
            &x.t,
            Some(l),
            "ssm::causal_conv1d_prefill_bf16",
            vec![weight.to_string()],
            Some(StateRef {
                store: StateStore::RecurrentState,
                layer: l,
            }),
            vec![x.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(channels)]),
                DType::BF16,
            )),
        )
        .expect("the conv produces its value")
    }

    // ── qwen3_5: the rest ──────────────────────────────────────────

    /// `kernels::norm::rmsnorm_gated_bf16`: the gated RMS norm, in its own
    /// launch rather than folded into a projection.
    pub fn rmsnorm_gated_launch(x: &Val, gate: &Val, weight: &str, width: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "norm::rmsnorm_gated_bf16",
            vec![weight.to_string()],
            None,
            vec![x.id, gate.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
        )
        .expect("the norm produces its value")
    }

    /// `kernels::moe::moe_grouped_gemm_bf16`: the grouped expert GEMM.
    /// The bank is named, like any other matmul's weight. It is ONE tensor
    /// (`[E, N, K]`) that the kernel indexes by the block's expert id, not a
    /// per-expert selection, so the traced name carries the `{e}` the family
    /// spells it with and the binding resolves to the whole bank. Without
    /// this the statement said "a grouped GEMM" and left which weights
    /// entirely to the executor -- readable, but not a declaration.
    pub fn moe_grouped_gemm(
        act: &Val,
        expert_ids: &Val,
        stage: &Val,
        aligned: Dim,
        width: u32,
        bank: &str,
    ) -> Val {
        record(
            &act.t,
            act.layer,
            "moe::moe_grouped_gemm_bf16",
            vec![bank.to_string()],
            None,
            // The second operand is the ALIGN's per-block expert id --
            // what the kernel indexes the bank by. It used to be the
            // sorted route order, which the kernel never reads: the
            // statement named one array and the executor bound another
            // (`mw.aligned_expert_ids`), so the declaration could not be
            // checked against the call. The third is the DESTINATION,
            // named by the pointer build above and written in place.
            vec![act.id, expert_ids.id, stage.id],
            // Block-major rows, not tokens: the operand this multiplies is
            // the gathered aligned bank, and saying `Tokens` here made the
            // routed leg's values indistinguishable from the shared
            // expert's -- which is exactly the question an executor has to
            // answer to pick a buffer.
            Some((Shape(vec![aligned, Dim::Const(width)]), DType::BF16)),
        )
        .expect("the gemm produces its value")
    }

    /// `kernels::mlp::chunked_swiglu_strided_bf16`: chunked swiglu over
    /// strided rows.
    pub fn chunked_swiglu_strided(x: &Val, intermediate: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "mlp::chunked_swiglu_strided_bf16",
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

    /// `kernels::mlp::sigmoid_scalar_gate_strided_add_bf16`: the shared
    /// expert's sigmoid-gated add, into a strided destination.
    pub fn sigmoid_scalar_gate_strided_add(x: &Val, y: &Val, gate: &Val, width: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "mlp::sigmoid_scalar_gate_strided_add_bf16",
            vec![],
            None,
            vec![x.id, y.id, gate.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
        )
        .expect("the gated add produces its value")
    }

    /// `kernels::layout::concat_bf16_rows`: join two row-aligned tensors
    /// along the channel axis.
    pub fn concat_rows(left: &Val, right: &Val, left_dim: u32, right_dim: u32) -> Val {
        record(
            &left.t,
            left.layer,
            "layout::concat_bf16_rows",
            vec![],
            None,
            vec![left.id, right.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(left_dim + right_dim)]),
                DType::BF16,
            )),
        )
        .expect("the concat produces its value")
    }

    /// `kernels::sample::lm_head_gemv_argmax_int8`: the readout and the argmax
    /// in one launch, over an int8 head with a per-channel scale.
    ///
    /// It produces TOKEN IDS, not logits. A greedy-decode fast path that
    /// never materializes the vocab-wide row -- which is why it is its own
    /// statement rather than `lm_head` followed by an argmax.
    pub fn lm_head_gemv_argmax_int8(x: &Val, weight: &str, scale: &str) -> Val {
        record(
            &x.t,
            None,
            "sample::lm_head_gemv_argmax_int8",
            vec![weight.to_string(), scale.to_string()],
            None,
            vec![x.id],
            Some((Shape(vec![Dim::Requests]), DType::I32)),
        )
        .expect("the readout produces its value")
    }

    // ── kimi: the WNA16 quantized MoE path ─────────────────────────
    //
    // 4-bit weights with a bf16 scale per group of `group_size` along K.
    // Distinct from MXFP4 (whose scale is an E8M0 exponent byte per 32) and
    // from fp8 -- three quantizations, three statements, because which one a
    // checkpoint ships is a fact the declaration reads.

    /// `kernels::quant::dequant_wna16_int4b8_to_bf16`: widen a packed
    /// int4-b8 weight to bf16.
    ///
    /// Weight-shaped: `[out_dim, in_dim/8]` packed to `[out_dim, in_dim]`,
    /// no token extent.
    pub fn dequant_wna16_int4b8(t: &Trace, l: u32, w: &str, out_dim: u32, in_dim: u32) -> Val {
        record(
            t,
            Some(l),
            "quant::dequant_wna16_int4b8_to_bf16",
            vec![w.to_string()],
            None,
            vec![],
            Some((
                Shape(vec![Dim::Const(out_dim), Dim::Const(in_dim)]),
                DType::BF16,
            )),
        )
        .expect("the dequant produces its value")
    }

    /// `kernels::quant::wna16_gate_up_decode_bf16`: the gate and up
    /// projections, decode-shaped, straight off the packed weights.
    ///
    /// `topk_idx` here is `[N, K]` in TOKEN order -- not the route-major
    /// order the aligned path sorts into -- so a row window keeps each
    /// token's routing intact and this is not `whole`.
    /// `bank` names the layer's expert weights; the statement records the
    /// FOUR per-expert tables the launcher actually reads
    /// (`<bank>.gate_packed` / `.gate_scale` / `.up_packed` /
    /// `.up_scale`). They were unnamed once, and a driver whose executor
    /// is model-agnostic could not reach them at all: with no name in the
    /// trace there is nothing to resolve, and the only way in was a
    /// family's private layer struct — the convention this whole
    /// direction exists to remove.
    pub fn wna16_gate_up_decode(
        act: &Val,
        topk_idx: &Val,
        intermediate: u32,
        bank: &str,
    ) -> (Val, Val) {
        let outs = record_many(
            &act.t,
            act.layer,
            "quant::wna16_gate_up_decode_bf16",
            vec![
                format!("{bank}.gate_packed"),
                format!("{bank}.gate_scale"),
                format!("{bank}.up_packed"),
                format!("{bank}.up_scale"),
            ],
            vec![act.id, topk_idx.id],
            vec![
                (
                    Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
                    DType::BF16,
                ),
                (
                    Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
                    DType::BF16,
                ),
            ],
        );
        let mut it = outs.into_iter();
        let gate = it.next().expect("the projection states two outputs");
        let up = it.next().expect("the projection states two outputs");
        (gate, up)
    }

    /// `kernels::quant::wna16_down_decode_bf16`: the down projection, same
    /// shape.
    pub fn wna16_down_decode(
        act: &Val,
        topk_idx: &Val,
        hidden: u32,
        bank: &str,
    ) -> Val {
        record(
            &act.t,
            act.layer,
            "quant::wna16_down_decode_bf16",
            vec![format!("{bank}.down_packed"), format!("{bank}.down_scale")],
            None,
            vec![act.id, topk_idx.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
        )
        .expect("the projection produces its value")
    }

    /// `kernels::norm::rmsnorm_strided_bf16`: the norm, reading and writing
    /// a prefix of wider rows.
    ///
    /// How a fused projection's halves get normed in place without a copy:
    /// the stride says where the row really ends.
    pub fn rmsnorm_strided(x: &Val, weight: &str, hidden: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "norm::rmsnorm_strided_bf16",
            vec![weight.to_string()],
            None,
            vec![x.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
        )
        .expect("the norm produces its value")
    }

    // ── the rope variants, and three small shapes ──────────────────

    /// `kernels::rope::rope_yarn_bf16`: YaRN-scaled rope.
    ///
    /// A different statement from [`Self::rope_yarn_original`], not a
    /// parameterization of it: the two interpolate frequencies differently,
    /// and which a checkpoint wants is a load-time fact.
    pub fn rope_yarn(q: &Val, k: &Val, q_width: u32) -> Val {
        record(
            &q.t,
            q.layer,
            "rope::rope_yarn_bf16",
            vec![],
            None,
            vec![q.id, k.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16)),
        )
        .expect("the rope produces its value")
    }

    /// `kernels::rope::qk_rmsnorm_mrope_bf16`: per-head q/k norms and MROPE.
    ///
    /// MROPE takes `[num_tokens, 3]` positions — a `(t, h, w)` triple rather
    /// than one index — because a vision model's tokens sit in a grid. That
    /// is why it cannot be the plain `qk_rmsnorm_rope` with a different
    /// theta.
    pub fn qk_rmsnorm_mrope(q: &Val, k: &Val, q_weight: &str, k_weight: &str, q_width: u32) -> Val {
        record(
            &q.t,
            q.layer,
            "rope::qk_rmsnorm_mrope_bf16",
            vec![q_weight.to_string(), k_weight.to_string()],
            None,
            vec![q.id, k.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16)),
        )
        .expect("the norm+rope produces its value")
    }

    /// `kernels::layout::split_gate_up_bf16`: split a packed `[N, 2·I]` bank.
    ///
    /// By HALVES, unlike [`Self::deinterleave_rows`], which splits by parity.
    /// Same shape, different layout, and the checkpoint decides which.
    pub fn split_gate_up(packed: &Val, intermediate: u32) -> (Val, Val) {
        let outs = record_many(
            &packed.t,
            packed.layer,
            "layout::split_gate_up_bf16",
            vec![],
            vec![packed.id],
            vec![
                (
                    Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
                    DType::BF16,
                ),
                (
                    Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
                    DType::BF16,
                ),
            ],
        );
        let mut it = outs.into_iter();
        let gate = it.next().expect("the split states two outputs");
        let up = it.next().expect("the split states two outputs");
        (gate, up)
    }

    /// `kernels::quant::scale_rows_bf16`: scale each row by its own factor.
    pub fn scale_rows(x: &Val, scale: &Val, width: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "quant::scale_rows_bf16",
            vec![],
            None,
            vec![x.id, scale.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
        )
        .expect("the scale produces its value")
    }

    /// `kernels::quant::cast_fp32_to_bf16`: narrow.
    pub fn cast_f32_to_bf16(x: &Val, width: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "quant::cast_fp32_to_bf16",
            vec![],
            None,
            vec![x.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
        )
        .expect("the cast produces its value")
    }

    /// `kernels::moe::apply_per_expert_scale_bf16`: multiply each route's
    /// weight by its expert's scale, in place.
    pub fn apply_per_expert_scale(topk_idx: &Val, topk_w: &Val, scale: &str, top_k: u32) -> Val {
        record(
            &topk_w.t,
            topk_w.layer,
            "moe::apply_per_expert_scale_bf16",
            vec![scale.to_string()],
            None,
            vec![topk_idx.id, topk_w.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(top_k)]), DType::F32)),
        )
        .expect("the scale produces its value")
    }

    /// `kernels::norm::residual_add_scale_rmsnorm_bf16`: residual add, a
    /// scalar scale, and the next pre-norm, fused.
    ///
    /// gemma-4's end-of-layer shape. The scale sits BETWEEN the add and the
    /// norm, which is why it is not [`Self::residual_add_rmsnorm`] with an
    /// extra multiply somewhere.
    pub fn residual_add_scale_rmsnorm(x: &Val, residual: &Val, weight: &str, hidden: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "norm::residual_add_scale_rmsnorm_bf16",
            vec![weight.to_string()],
            None,
            vec![x.id, residual.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
        )
        .expect("the fused norm produces its value")
    }

    /// `kernels::attn::dispatch_attention_flashinfer_prefill_sm90_bf16`: the FA3
    /// prefill, on Hopper.
    pub fn flashinfer_prefill_sm90(q: &Val, l: u32, heads: u32, head_dim: u32) -> Val {
        record(
            &q.t,
            Some(l),
            "attn::dispatch_attention_flashinfer_prefill_sm90_bf16",
            vec![],
            Some(StateRef {
                store: StateStore::KvCache,
                layer: l,
            }),
            vec![q.id],
            Some((
                Shape(vec![
                    Dim::Tokens,
                    Dim::Const(heads),
                    Dim::Const(head_dim),
                ]),
                DType::BF16,
            )),
        )
        .expect("the attention produces its value")
    }

    // ── mixtral / gpt-oss: the MXFP4 MoE path ──────────────────────
    //
    // gpt-oss ships its experts as MXFP4 -- 4-bit values with an E8M0
    // exponent byte per block of 32 -- and mixtral's shell runs them through
    // Marlin. Several statements here operate on WEIGHTS rather than
    // activations (repacking a scale layout, splitting a fused bias) and have
    // no token extent at all. They are stated because they are launches the
    // fire performs, and a reader tracing where an operand came from should
    // find them on the tape.

    /// `kernels::norm::add_bias_bf16_strided`: add a bias row into a strided
    /// destination.
    pub fn add_bias_strided(x: &Val, bias: &str, width: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "norm::add_bias_bf16_strided",
            vec![bias.to_string()],
            None,
            vec![x.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
        )
        .expect("the bias add produces its value")
    }

    /// `kernels::moe::add_moe_route_bias_bf16`: add each route's EXPERT
    /// bias, indexed by that route's expert.
    ///
    /// `whole`: `topk_idx` is route-global, so a row window would pick the
    /// wrong experts' biases.
    pub fn add_moe_route_bias(x: &Val, topk_idx: &Val, bias: &str, width: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "moe::add_moe_route_bias_bf16",
            vec![bias.to_string()],
            None,
            vec![x.id, topk_idx.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
        )
        .expect("the bias add produces its value")
    }

    /// `kernels::attn::build_window_page_view`: a page view keeping only the
    /// last `keep_pages` of each request.
    ///
    /// How sliding-window attention is expressed without a second cache: the
    /// window is a VIEW over the same pages. `whole` -- it walks
    /// `src_indptr[R+1]`.
    pub fn build_window_page_view(t: &Trace, l: u32, keep_pages: u32) -> Val {
        record(
            t,
            Some(l),
            "attn::build_window_page_view",
            vec![],
            Some(StateRef {
                store: StateStore::KvCache,
                layer: l,
            }),
            vec![],
            Some((
                Shape(vec![Dim::Requests, Dim::Const(keep_pages)]),
                DType::I32,
            )),
        )
        .expect("the view produces its value")
    }

    /// `kernels::attn::build_full_split_view`: describe one request's page
    /// range as `splits` separate one-token requests.
    ///
    /// The KV-split decode shape: the same pages, presented as several
    /// requests so the attention kernel parallelises over them, with the
    /// partials merged afterwards by [`Self::combine_attn_outputs`].
    pub fn build_full_split_view(t: &Trace, l: u32, splits: u32) -> Val {
        record(
            t,
            Some(l),
            "attn::build_full_split_view",
            vec![],
            Some(StateRef {
                store: StateStore::KvCache,
                layer: l,
            }),
            vec![],
            Some((Shape(vec![Dim::Const(splits + 1)]), DType::I32)),
        )
        .expect("the view produces its value")
    }

    /// `kernels::layout::deinterleave_rows_bf16`: split a fused `[2·I, H]`
    /// weight into its gate and up halves BY PARITY.
    ///
    /// A weight-layout fact: gpt-oss interleaves the two projections row by
    /// row, so this is not the same as slicing the tensor in half. No token
    /// extent — it transforms a weight.
    pub fn deinterleave_rows(t: &Trace, l: u32, w: &str, i: u32, h: u32) -> (Val, Val) {
        let outs = record_many(
            t,
            Some(l),
            "layout::deinterleave_rows_bf16",
            vec![w.to_string()],
            vec![],
            vec![
                (Shape(vec![Dim::Const(i), Dim::Const(h)]), DType::BF16),
                (Shape(vec![Dim::Const(i), Dim::Const(h)]), DType::BF16),
            ],
        );
        let mut it = outs.into_iter();
        let gate = it.next().expect("the split states two outputs");
        let up = it.next().expect("the split states two outputs");
        (gate, up)
    }

    /// `kernels::layout::deinterleave_vec_bf16`: the same, for the fused
    /// per-expert bias vector.
    pub fn deinterleave_vec(t: &Trace, l: u32, w: &str, i: u32) -> (Val, Val) {
        let outs = record_many(
            t,
            Some(l),
            "layout::deinterleave_vec_bf16",
            vec![w.to_string()],
            vec![],
            vec![
                (Shape(vec![Dim::Const(i)]), DType::BF16),
                (Shape(vec![Dim::Const(i)]), DType::BF16),
            ],
        );
        let mut it = outs.into_iter();
        let gate = it.next().expect("the split states two outputs");
        let up = it.next().expect("the split states two outputs");
        (gate, up)
    }

    /// `kernels::gemm::gemv3_bf16`: three GEMVs against one activation, in
    /// one launch.
    ///
    /// The decode-shaped q/k/v projection: `N == 1` means each projection is
    /// a matrix-vector product, and three of them share the activation read.
    pub fn gemv3(act: &Val, w0: &str, w1: &str, w2: &str, n0: u32, n1: u32, n2: u32) -> (Val, Val, Val) {
        let outs = record_many(
            &act.t,
            act.layer,
            "gemm::gemv3_bf16",
            vec![w0.to_string(), w1.to_string(), w2.to_string()],
            vec![act.id],
            vec![
                (Shape(vec![Dim::Tokens, Dim::Const(n0)]), DType::BF16),
                (Shape(vec![Dim::Tokens, Dim::Const(n1)]), DType::BF16),
                (Shape(vec![Dim::Tokens, Dim::Const(n2)]), DType::BF16),
            ],
        );
        let mut it = outs.into_iter();
        let o0 = it.next().expect("gemv3 states three outputs");
        let o1 = it.next().expect("gemv3 states three outputs");
        let o2 = it.next().expect("gemv3 states three outputs");
        (o0, o1, o2)
    }

    /// `kernels::mlp::gpt_oss_glu_strided_bf16`: gpt-oss's clamped GLU,
    /// reading and writing strided.
    pub fn gpt_oss_glu_strided(gate: &Val, up: &Val, width: u32) -> Val {
        record(
            &gate.t,
            gate.layer,
            "mlp::gpt_oss_glu_strided_bf16",
            vec![],
            None,
            vec![gate.id, up.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
        )
        .expect("the activation produces its value")
    }

    /// `kernels::norm::rmsnorm_bf16_with_fp16`: the norm, published in both
    /// bf16 and fp16.
    ///
    /// The fp16 copy is what the MXFP4 grouped GEMM below consumes; producing
    /// it here rather than casting afterwards is the binding, so the
    /// declaration states it.
    pub fn rmsnorm_with_fp16(x: &Val, weight: &str, hidden: u32) -> (Val, Val) {
        let outs = record_many(
            &x.t,
            x.layer,
            "norm::rmsnorm_bf16_with_fp16",
            vec![weight.to_string()],
            vec![x.id],
            vec![
                (Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16),
                (Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::F16),
            ],
        );
        let mut it = outs.into_iter();
        let bf16 = it.next().expect("the norm states two outputs");
        let fp16 = it.next().expect("the norm states two outputs");
        (bf16, fp16)
    }

    /// `kernels::rope::rope_write_kv_bf16`: rope q and k, then commit k/v to
    /// the pages, in one launch.
    pub fn rope_write_kv(q: &Val, k: &Val, v: &Val, l: u32, q_width: u32) -> Val {
        record(
            &q.t,
            Some(l),
            "rope::rope_write_kv_bf16",
            vec![],
            Some(StateRef {
                store: StateStore::KvCache,
                layer: l,
            }),
            vec![q.id, k.id, v.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16)),
        )
        .expect("the fused rope+write produces its value")
    }

    /// `kernels::quant::mxfp4_scales_to_marlin_e8m0`: repack the checkpoint's
    /// E8M0 scale layout into the one Marlin walks.
    pub fn mxfp4_scales_to_marlin(t: &Trace, l: u32, w: &str, groups: u32, rows: u32) -> Val {
        record(
            t,
            Some(l),
            "quant::mxfp4_scales_to_marlin_e8m0",
            vec![w.to_string()],
            None,
            vec![],
            Some((
                Shape(vec![Dim::Const(groups), Dim::Const(rows)]),
                DType::I32,
            )),
        )
        .expect("the repack produces its value")
    }

    /// `kernels::moe::transpose_expert_scales_u8`: the per-expert group
    /// scales, `[E, n, k/32]` -> `[E, k/32, n]`.
    pub fn transpose_expert_scales(t: &Trace, l: u32, w: &str, experts: u32, k_groups: u32, n: u32) -> Val {
        record(
            t,
            Some(l),
            "moe::transpose_expert_scales_u8",
            vec![w.to_string()],
            None,
            vec![],
            Some((
                Shape(vec![
                    Dim::Const(experts),
                    Dim::Const(k_groups),
                    Dim::Const(n),
                ]),
                DType::I32,
            )),
        )
        .expect("the transpose produces its value")
    }

    /// `kernels::quant::mxfp4_moe_gate_up_decode_grouped_bf16`: the gate and
    /// up projections for every route, grouped by expert.
    pub fn mxfp4_moe_gate_up_decode_grouped(
        act: &Val,
        sorted_route_ids: &Val,
        counts: &Val,
        intermediate: u32,
    ) -> (Val, Val) {
        let outs = record_many(
            &act.t,
            act.layer,
            "quant::mxfp4_moe_gate_up_decode_grouped_bf16",
            vec![],
            vec![act.id, sorted_route_ids.id, counts.id],
            vec![
                (
                    Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
                    DType::BF16,
                ),
                (
                    Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
                    DType::BF16,
                ),
            ],
        );
        let mut it = outs.into_iter();
        let gate = it.next().expect("the projection states two outputs");
        let up = it.next().expect("the projection states two outputs");
        (gate, up)
    }

    /// `marlin_moe::launch_mxfp4_moe_gemm_w4a16_bf16`: the Marlin W4A16
    /// grouped MoE GEMM.
    ///
    /// Namespaced in the symbol because it lives in the vendored `marlin_moe`
    /// tree, the same way `ops::` entries do.
    pub fn mxfp4_moe_gemm_w4a16(act: &Val, sorted_route_ids: &Val, width: u32) -> Val {
        record(
            &act.t,
            act.layer,
            "marlin_moe::launch_mxfp4_moe_gemm_w4a16_bf16",
            vec![],
            None,
            vec![act.id, sorted_route_ids.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
        )
        .expect("the gemm produces its value")
    }

    /// `kernels::attn::dispatch_attention_flashinfer_decode_bf16`: the bf16-typed
    /// decode dispatch.
    pub fn flashinfer_decode_bf16(q: &Val, l: u32, heads: u32, head_dim: u32) -> Val {
        record(
            &q.t,
            Some(l),
            "attn::dispatch_attention_flashinfer_decode_bf16",
            vec![],
            Some(StateRef {
                store: StateStore::KvCache,
                layer: l,
            }),
            vec![q.id],
            Some((
                Shape(vec![
                    Dim::Tokens,
                    Dim::Const(heads),
                    Dim::Const(head_dim),
                ]),
                DType::BF16,
            )),
        )
        .expect("the attention produces its value")
    }

    // ── deepseek_v4: hyper-connections ─────────────────────────────
    //
    // The SECOND rank-K residual scheme in this table, and it is not AltUp's.
    // gemma-3n predicts each stream from a learned linear combination and
    // corrects from one active stream; HC mixes with a per-token matrix that
    // has been sinkhorn-normalized, and there is no active stream — every
    // layer reads a weighted collapse of all of them and writes back to all
    // of them. Two answers to "what if the residual had a rank", worth being
    // able to state separately.

    /// `kernels::norm::hc_rmsnorm_to_f32`: norm the flattened multi-stream
    /// residual into the fp32 the mixing GEMM wants.
    pub fn hc_rmsnorm_to_f32(residual: &Val, weight: &str, width: u32) -> Val {
        record(
            &residual.t,
            residual.layer,
            "norm::hc_rmsnorm_to_f32",
            vec![weight.to_string()],
            None,
            vec![residual.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::F32)),
        )
        .expect("the norm produces its value")
    }

    /// `kernels::norm::hc_expand_bf16`: replicate the embedding into K
    /// streams, at the top of the stack.
    ///
    /// Where a rank-K residual BEGINS. AltUp's equivalent is implicit in how
    /// gemma-3n lays out its workspace; HC states it, which is the better of
    /// the two and the one a declaration can read.
    pub fn hc_expand(x: &Val, hc_mult: u32, hidden: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "norm::hc_expand_bf16",
            vec![],
            None,
            vec![x.id],
            Some((
                Shape(vec![
                    Dim::Tokens,
                    Dim::Const(hc_mult),
                    Dim::Const(hidden),
                ]),
                DType::BF16,
            )),
        )
        .expect("the expand produces its value")
    }

    /// `kernels::norm::hc_pre_postprocess_bf16`: turn the mixing GEMM's
    /// output into `(post_mix, comb_mix, layer_input)`.
    ///
    /// `comb_mix` is a `[hc_mult, hc_mult]` matrix PER TOKEN, sinkhorn-
    /// normalized so the mixing is doubly stochastic. `layer_input` is the
    /// single stream the layer body actually runs on.
    pub fn hc_pre(mixes: &Val, residual: &Val, hc_mult: u32, hidden: u32) -> (Val, Val, Val) {
        let outs = record_many(
            &mixes.t,
            mixes.layer,
            "norm::hc_pre_postprocess_bf16",
            vec![],
            vec![mixes.id, residual.id],
            vec![
                (Shape(vec![Dim::Tokens, Dim::Const(hc_mult)]), DType::F32),
                (
                    Shape(vec![
                        Dim::Tokens,
                        Dim::Const(hc_mult),
                        Dim::Const(hc_mult),
                    ]),
                    DType::F32,
                ),
                (Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16),
            ],
        );
        let mut it = outs.into_iter();
        let post_mix = it.next().expect("hc_pre states three outputs");
        let comb_mix = it.next().expect("hc_pre states three outputs");
        let layer_input = it.next().expect("hc_pre states three outputs");
        (post_mix, comb_mix, layer_input)
    }

    /// `kernels::norm::hc_post_bf16`: fold the layer's output back into all
    /// K streams — `new_residual_j = comb_mix_ij · residual_i + post_mix_j · x`.
    pub fn hc_post(
        x: &Val,
        residual: &Val,
        post_mix: &Val,
        comb_mix: &Val,
        hc_mult: u32,
        hidden: u32,
    ) -> Val {
        record(
            &x.t,
            x.layer,
            "norm::hc_post_bf16",
            vec![],
            None,
            vec![x.id, residual.id, post_mix.id, comb_mix.id],
            Some((
                Shape(vec![
                    Dim::Tokens,
                    Dim::Const(hc_mult),
                    Dim::Const(hidden),
                ]),
                DType::BF16,
            )),
        )
        .expect("the fold produces its value")
    }

    /// `kernels::norm::hc_head_postprocess_bf16`: collapse the K streams to
    /// one, for the readout.
    pub fn hc_head(mixes: &Val, residual: &Val, hidden: u32) -> Val {
        record(
            &mixes.t,
            mixes.layer,
            "norm::hc_head_postprocess_bf16",
            vec![],
            None,
            vec![mixes.id, residual.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
        )
        .expect("the collapse produces its value")
    }

    /// `kernels::norm::per_head_rmsnorm_bf16`: an RMS norm whose rows are
    /// heads rather than the residual width.
    pub fn per_head_rmsnorm(x: &Val, weight: &str, heads: u32, head_dim: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "norm::per_head_rmsnorm_bf16",
            vec![weight.to_string()],
            None,
            vec![x.id],
            Some((
                Shape(vec![
                    Dim::Tokens,
                    Dim::Const(heads),
                    Dim::Const(head_dim),
                ]),
                DType::BF16,
            )),
        )
        .expect("the norm produces its value")
    }

    /// `kernels::norm::attn_sink_correction_bf16`: the learned per-head sink
    /// term, applied as a correction to the attention output.
    pub fn attn_sink_correction(o: &Val, lse: &Val, sink: &str, heads: u32, head_dim: u32) -> Val {
        record(
            &o.t,
            o.layer,
            "norm::attn_sink_correction_bf16",
            vec![sink.to_string()],
            None,
            vec![o.id, lse.id],
            Some((
                Shape(vec![
                    Dim::Tokens,
                    Dim::Const(heads),
                    Dim::Const(head_dim),
                ]),
                DType::BF16,
            )),
        )
        .expect("the correction produces its value")
    }

    // ── deepseek_v4: compressed attention ──────────────────────────
    //
    // A SECOND KV cache beside the fine-grained one, holding a single entry
    // per `ratio` tokens. Every query attends both, and the two outputs are
    // merged by their log-sum-exps -- which is what makes it exact rather
    // than an approximation: the merge is the same algebra flashinfer's own
    // KV-split uses.

    /// `kernels::attn::dsv4_boundary_meta_decode`: which decode tokens close
    /// a compression window, CUDA-graph-safely.
    ///
    /// Returns `(pos, req, rope)`. A token whose position is not a boundary
    /// gets `pos = -1`, which the gather zero-fills and the store skips --
    /// so the shape is fixed and the graph replays.
    pub fn dsv4_boundary_meta_decode(positions: &Val) -> (Val, Val, Val) {
        let outs = record_many(
            &positions.t,
            positions.layer,
            "attn::dsv4_boundary_meta_decode",
            vec![],
            vec![positions.id],
            vec![
                (Shape(vec![Dim::Requests]), DType::I32),
                (Shape(vec![Dim::Requests]), DType::I32),
                (Shape(vec![Dim::Requests]), DType::I32),
            ],
        );
        let mut it = outs.into_iter();
        let pos = it.next().expect("the meta states three outputs");
        let req = it.next().expect("the meta states three outputs");
        let rope = it.next().expect("the meta states three outputs");
        (pos, req, rope)
    }

    /// `kernels::attn::dsv4_compress_gather_paged_bf16`: build one compressed
    /// entry per boundary token, by a per-dimension softmax over the gate
    /// scores of the window ending there.
    pub fn dsv4_compress_gather_paged(boundary_pos: &Val, l: u32, head_dim: u32) -> Val {
        record(
            &boundary_pos.t,
            Some(l),
            "attn::dsv4_compress_gather_paged_bf16",
            vec![],
            Some(StateRef {
                store: StateStore::KvCache,
                layer: l,
            }),
            vec![boundary_pos.id],
            Some((
                Shape(vec![Dim::Requests, Dim::Const(head_dim)]),
                DType::BF16,
            )),
        )
        .expect("the gather produces its value")
    }

    /// `kernels::attn::dsv4_store_comp_entries_bf16`: commit those entries to
    /// the compressed cache, at the boundary token's own slot.
    pub fn dsv4_store_comp_entries(entries: &Val, boundary_pos: &Val, l: u32) {
        record(
            &entries.t,
            Some(l),
            "attn::dsv4_store_comp_entries_bf16",
            vec![],
            Some(StateRef {
                store: StateStore::KvCache,
                layer: l,
            }),
            vec![entries.id, boundary_pos.id],
            None,
        );
    }

    /// `kernels::attn::attention_compressed_paged_bf16`: attend the
    /// compressed cache, causally.
    ///
    /// Entry `c` lives at absolute position `(c+1)·ratio - 1`, and a query at
    /// `p` may attend it iff that boundary is `<= p`.
    pub fn attention_compressed_paged(q: &Val, l: u32, heads: u32, head_dim: u32) -> (Val, Val) {
        let outs = record_many(
            &q.t,
            Some(l),
            "attn::attention_compressed_paged_bf16",
            vec![],
            vec![q.id],
            vec![
                (
                    Shape(vec![
                        Dim::Tokens,
                        Dim::Const(heads),
                        Dim::Const(head_dim),
                    ]),
                    DType::BF16,
                ),
                (Shape(vec![Dim::Tokens, Dim::Const(heads)]), DType::F32),
            ],
        );
        let mut it = outs.into_iter();
        let o = it.next().expect("the attention states two outputs");
        let lse = it.next().expect("the attention states two outputs");
        (o, lse)
    }

    /// `kernels::attn::combine_attn_outputs_bf16`: merge two partial
    /// attention results by their log-sum-exps.
    ///
    /// Exact, not an approximation — the same algebra flashinfer's own
    /// KV-split merge uses, which is why the fine and compressed halves can
    /// be attended independently at all.
    pub fn combine_attn_outputs(
        o1: &Val,
        lse1: &Val,
        o2: &Val,
        lse2: &Val,
        heads: u32,
        head_dim: u32,
    ) -> (Val, Val) {
        let outs = record_many(
            &o1.t,
            o1.layer,
            "attn::combine_attn_outputs_bf16",
            vec![],
            vec![o1.id, lse1.id, o2.id, lse2.id],
            vec![
                (
                    Shape(vec![
                        Dim::Tokens,
                        Dim::Const(heads),
                        Dim::Const(head_dim),
                    ]),
                    DType::BF16,
                ),
                (Shape(vec![Dim::Tokens, Dim::Const(heads)]), DType::F32),
            ],
        );
        let mut it = outs.into_iter();
        let o = it.next().expect("the combine states two outputs");
        let lse = it.next().expect("the combine states two outputs");
        (o, lse)
    }

    /// `kernels::attn::lse_log2_to_ln`: rebase a log-sum-exp from log2 to
    /// natural log.
    ///
    /// FlashInfer publishes its LSE in log2; the combine above works in ln.
    /// A unit conversion, and it is a launch, so the trace says so rather
    /// than leaving a reader to wonder which base an LSE is in.
    pub fn lse_log2_to_ln(lse: &Val, heads: u32) -> Val {
        record(
            &lse.t,
            lse.layer,
            "attn::lse_log2_to_ln",
            vec![],
            None,
            vec![lse.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(heads)]), DType::F32)),
        )
        .expect("the rebase produces its value")
    }

    // ── deepseek_v4: routing, activation, dequant ──────────────────

    /// `kernels::moe::topk_sqrtsoftplus_bf16`: the router, scored by
    /// `sqrt(softplus(·))`.
    pub fn topk_sqrtsoftplus(logits: &Val, bias: &str, top_k: u32) -> (Val, Val) {
        let outs = record_many(
            &logits.t,
            logits.layer,
            "moe::topk_sqrtsoftplus_bf16",
            vec![bias.to_string()],
            vec![logits.id],
            vec![
                (Shape(vec![Dim::Tokens, Dim::Const(top_k)]), DType::I32),
                (Shape(vec![Dim::Tokens, Dim::Const(top_k)]), DType::F32),
            ],
        );
        let mut it = outs.into_iter();
        let idx = it.next().expect("the router states two outputs");
        let w = it.next().expect("the router states two outputs");
        (idx, w)
    }

    /// `kernels::moe::hash_route_lookup`: expert INDICES from a hash table
    /// keyed by token id; weights still from the router logits.
    ///
    /// A route that is a pure function of the token, not of its activations —
    /// which is why it reads `token_ids` and a `tid2eid` table rather than
    /// scoring anything. The weights are not hashed, so the logits GEMM above
    /// it does not go away.
    pub fn hash_route_lookup(token_ids: &Val, logits: &Val, table: &str, top_k: u32) -> (Val, Val) {
        let outs = record_many(
            &logits.t,
            logits.layer,
            "moe::hash_route_lookup",
            vec![table.to_string()],
            vec![token_ids.id, logits.id],
            vec![
                (Shape(vec![Dim::Tokens, Dim::Const(top_k)]), DType::I32),
                (Shape(vec![Dim::Tokens, Dim::Const(top_k)]), DType::F32),
            ],
        );
        let mut it = outs.into_iter();
        let idx = it.next().expect("the lookup states two outputs");
        let w = it.next().expect("the lookup states two outputs");
        (idx, w)
    }

    /// `kernels::mlp::swiglu_clamp_bf16` / `kernels::mlp::chunked_swiglu_clamp_bf16`:
    /// swiglu with the gate clamped.
    ///
    /// `packed` picks the chunked form, the binding choice [`Self::swiglu`]
    /// carries.
    pub fn swiglu_clamp(x: &Val, intermediate: u32, packed: bool) -> Val {
        record(
            &x.t,
            x.layer,
            if packed {
                "mlp::chunked_swiglu_clamp_bf16"
            } else {
                "mlp::swiglu_clamp_bf16"
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

    /// `kernels::rope::rope_partial_last_bf16`: rope the LAST `rope_dim`
    /// channels rather than the first.
    ///
    /// A different statement from [`Self::rope_partial_q_only`], not a flag on
    /// it: which end of the channel axis carries position is a property of
    /// the checkpoint's layout.
    pub fn rope_partial_last(x: &Val, heads: u32, head_dim: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "rope::rope_partial_last_bf16",
            vec![],
            None,
            vec![x.id],
            Some((
                Shape(vec![
                    Dim::Tokens,
                    Dim::Const(heads),
                    Dim::Const(head_dim),
                ]),
                DType::BF16,
            )),
        )
        .expect("the rope produces its value")
    }

    /// `kernels::attn::write_kv_to_pages_bf16`: the bf16-typed page write.
    pub fn write_kv_to_pages_bf16(k: &Val, v: &Val, l: u32) {
        record(
            &k.t,
            Some(l),
            "attn::write_kv_to_pages_bf16",
            vec![],
            Some(StateRef {
                store: StateStore::KvCache,
                layer: l,
            }),
            vec![k.id, v.id],
            None,
        );
    }

    /// `kernels::attn::attention_naive_paged_bf16`: the bf16-typed naive paged
    /// attention.
    pub fn attention_naive_paged_bf16(q: &Val, l: u32, heads: u32, head_dim: u32) -> Val {
        record(
            &q.t,
            Some(l),
            "attn::attention_naive_paged_bf16",
            vec![],
            Some(StateRef {
                store: StateStore::KvCache,
                layer: l,
            }),
            vec![q.id],
            Some((
                Shape(vec![
                    Dim::Tokens,
                    Dim::Const(heads),
                    Dim::Const(head_dim),
                ]),
                DType::BF16,
            )),
        )
        .expect("the attention produces its value")
    }

    /// Which shape a quantized weight's SCALE has.
    ///
    /// Three fp8 forms, and the difference is not a tuning knob: one scale
    /// per tensor, one per output channel, or one per group of `group_size`
    /// along K. It is a property of the checkpoint, so the declaration states
    /// which — a driver that guessed would dequantize correctly on one
    /// checkpoint and silently wrongly on another.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Fp8Scale {
        /// One scale for the whole tensor.
        PerTensor,
        /// One per output channel.
        PerChannel,
        /// One per group along the reduction axis.
        PerGroup,
    }

    /// `kernels::quant::dequant_fp8_e4m3_to_bf16[_per_channel|_per_group]`:
    /// widen an fp8 weight to bf16.
    pub fn dequant_fp8_e4m3(
        t: &Trace,
        l: u32,
        weight: &str,
        rows: u32,
        cols: u32,
        scale: Fp8Scale,
    ) -> Val {
        record(
            t,
            Some(l),
            match scale {
                Fp8Scale::PerTensor => "quant::dequant_fp8_e4m3_to_bf16",
                Fp8Scale::PerChannel => "quant::dequant_fp8_e4m3_to_bf16_per_channel",
                Fp8Scale::PerGroup => "quant::dequant_fp8_e4m3_to_bf16_per_group",
            },
            vec![weight.to_string()],
            None,
            vec![],
            Some((
                Shape(vec![Dim::Const(rows), Dim::Const(cols)]),
                DType::BF16,
            )),
        )
        .expect("the dequant produces its value")
    }

    /// `kernels::quant::dequant_mxfp4_to_bf16`: the same for MXFP4, whose
    /// scale is an E8M0 exponent byte per block of 32.
    pub fn dequant_mxfp4(t: &Trace, l: u32, weight: &str, rows: u32, cols: u32) -> Val {
        record(
            t,
            Some(l),
            "quant::dequant_mxfp4_to_bf16",
            vec![weight.to_string()],
            None,
            vec![],
            Some((
                Shape(vec![Dim::Const(rows), Dim::Const(cols)]),
                DType::BF16,
            )),
        )
        .expect("the dequant produces its value")
    }

    // ── nemotron_h: mamba ──────────────────────────────────────────
    //
    // The other linear-attention shape, and it is not GDN's or KDA's. Mamba
    // carries a `[head_dim, state_size]` slab per head and advances it with a
    // scalar `dA` derived from a per-token `dt` -- a selective scan, not a
    // delta rule. The state is a different SHAPE, which is why nothing above
    // stands in for it and why the todo lists it as its own missing algebra.

    /// `kernels::ssm::nemotron_mamba_split_bf16`: split the fused input
    /// projection into `(gate, conv_in, dt)`.
    pub fn nemotron_mamba_split(
        projected: &Val,
        intermediate: u32,
        conv_dim: u32,
        heads: u32,
    ) -> (Val, Val, Val) {
        let outs = record_many(
            &projected.t,
            projected.layer,
            "ssm::nemotron_mamba_split_bf16",
            vec![],
            vec![projected.id],
            vec![
                (
                    Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
                    DType::BF16,
                ),
                (Shape(vec![Dim::Tokens, Dim::Const(conv_dim)]), DType::BF16),
                (Shape(vec![Dim::Tokens, Dim::Const(heads)]), DType::BF16),
            ],
        );
        let mut it = outs.into_iter();
        let gate = it.next().expect("the split states three outputs");
        let conv_in = it.next().expect("the split states three outputs");
        let dt = it.next().expect("the split states three outputs");
        (gate, conv_in, dt)
    }

    /// `kernels::ssm::nemotron_prepare_mamba_params`: widen `A_log`, `D`
    /// and `dt_bias` to fp32, storing `A = -exp(A_log)`.
    ///
    /// Per HEAD, with no token extent at all — it transforms weights, not
    /// activations. Stated because it is a launch the fire performs, and a
    /// reader following where `A` comes from should find it on the tape.
    pub fn nemotron_prepare_mamba_params(
        t: &Trace,
        l: u32,
        a_log: &str,
        d: &str,
        dt_bias: &str,
        heads: u32,
    ) -> (Val, Val, Val) {
        let outs = record_many(
            t,
            Some(l),
            "ssm::nemotron_prepare_mamba_params",
            vec![a_log.to_string(), d.to_string(), dt_bias.to_string()],
            vec![],
            vec![
                (Shape(vec![Dim::Const(heads)]), DType::F32),
                (Shape(vec![Dim::Const(heads)]), DType::F32),
                (Shape(vec![Dim::Const(heads)]), DType::F32),
            ],
        );
        let mut it = outs.into_iter();
        let a = it.next().expect("the prepare states three outputs");
        let d_f32 = it.next().expect("the prepare states three outputs");
        let bias = it.next().expect("the prepare states three outputs");
        (a, d_f32, bias)
    }

    /// `kernels::ssm::nemotron_prepare_mamba_dt_da`: the per-token step
    /// size and its decay, `(dt, dA)`.
    pub fn nemotron_prepare_mamba_dt_da(dt_raw: &Val, a: &Val, heads: u32) -> (Val, Val) {
        let outs = record_many(
            &dt_raw.t,
            dt_raw.layer,
            "ssm::nemotron_prepare_mamba_dt_da",
            vec![],
            vec![dt_raw.id, a.id],
            vec![
                (Shape(vec![Dim::Tokens, Dim::Const(heads)]), DType::F32),
                (Shape(vec![Dim::Tokens, Dim::Const(heads)]), DType::F32),
            ],
        );
        let mut it = outs.into_iter();
        let dt = it.next().expect("the prepare states two outputs");
        let da = it.next().expect("the prepare states two outputs");
        (dt, da)
    }

    /// `kernels::ssm::nemotron_mamba_ssm_batched_bf16`: the selective scan.
    ///
    /// `whole` for both reasons the table collects: it addresses through
    /// `slot_ids` and `qo_indptr`, AND the scan carries state from token to
    /// token, so a row window would resume from the wrong slab.
    pub fn nemotron_mamba_ssm(
        conv_out: &Val,
        dt: &Val,
        l: u32,
        intermediate: u32,
    ) -> Val {
        record(
            &conv_out.t,
            Some(l),
            "ssm::nemotron_mamba_ssm_batched_bf16",
            vec![],
            Some(StateRef {
                store: StateStore::RecurrentState,
                layer: l,
            }),
            vec![conv_out.id, dt.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
                DType::BF16,
            )),
        )
        .expect("the scan produces its value")
    }

    /// `kernels::ssm::causal_conv1d_update_bf16`: the decode-step conv,
    /// reading and advancing the per-request conv window.
    ///
    /// `whole`: it advances a slot's state in place, so a row window would
    /// advance the wrong ones.
    pub fn causal_conv1d_update(x: &Val, weight: &str, bias: &str, l: u32, channels: u32) -> Val {
        record(
            &x.t,
            Some(l),
            "ssm::causal_conv1d_update_bf16",
            vec![weight.to_string(), bias.to_string()],
            Some(StateRef {
                store: StateStore::RecurrentState,
                layer: l,
            }),
            vec![x.id],
            Some((
                Shape(vec![Dim::Requests, Dim::Const(channels)]),
                DType::BF16,
            )),
        )
        .expect("the conv produces its value")
    }

    /// `kernels::ssm::zamba_rmsnorm_gated_bf16`: the grouped, gated output
    /// norm mamba's block ends with.
    pub fn zamba_rmsnorm_gated(x: &Val, gate: &Val, weight: &str, hidden: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "ssm::zamba_rmsnorm_gated_bf16",
            vec![weight.to_string()],
            None,
            vec![x.id, gate.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
        )
        .expect("the norm produces its value")
    }

    /// `kernels::mlp::relu2_bf16`: `relu(x)²`, nemotron_h's MLP activation.
    pub fn relu2(x: &Val, width: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "mlp::relu2_bf16",
            vec![],
            None,
            vec![x.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
        )
        .expect("the activation produces its value")
    }

    // ── nemotron_h: its own MoE dispatch ───────────────────────────

    /// `kernels::moe::topk_sigmoid_bias_fp32`: the router, over fp32
    /// logits and with a per-expert correction bias.
    pub fn topk_sigmoid_bias(logits: &Val, bias: &str, top_k: u32) -> (Val, Val) {
        let outs = record_many(
            &logits.t,
            logits.layer,
            "moe::topk_sigmoid_bias_fp32",
            vec![bias.to_string()],
            vec![logits.id],
            vec![
                (Shape(vec![Dim::Tokens, Dim::Const(top_k)]), DType::I32),
                (Shape(vec![Dim::Tokens, Dim::Const(top_k)]), DType::F32),
            ],
        );
        let mut it = outs.into_iter();
        let idx = it.next().expect("the router states two outputs");
        let w = it.next().expect("the router states two outputs");
        (idx, w)
    }

    /// `kernels::moe::moe_bucket_exact`: bucket routes by expert WITHOUT
    /// padding to fixed blocks.
    ///
    /// The unpadded counterpart of [`Self::moe_align`], writing exact
    /// per-expert counts the host reads to build cuBLAS grouped shapes.
    /// `whole` for the same reason: the sort is over all routes.
    pub fn moe_bucket_exact(topk_idx: &Val, num_experts: u32, top_k: u32) -> (Val, Val) {
        let outs = record_many(
            &topk_idx.t,
            topk_idx.layer,
            "moe::moe_bucket_exact",
            vec![],
            vec![topk_idx.id],
            vec![
                (Shape(vec![Dim::Tokens, Dim::Const(top_k)]), DType::I32),
                (Shape(vec![Dim::Const(num_experts)]), DType::I32),
            ],
        );
        let mut it = outs.into_iter();
        let sorted = it.next().expect("the bucket states two outputs");
        let counts = it.next().expect("the bucket states two outputs");
        (sorted, counts)
    }

    /// `kernels::ssm::build_nemotron_moe_ptrs_aligned_bf16`: the pointer
    /// arrays for the block-aligned batched GEMM.
    pub fn build_nemotron_moe_ptrs_aligned(expert_ids: &Val, aligned_in: &Val, l: u32) {
        record(
            &expert_ids.t,
            Some(l),
            "ssm::build_nemotron_moe_ptrs_aligned_bf16",
            vec![],
            None,
            vec![expert_ids.id, aligned_in.id],
            None,
        );
    }

    /// `kernels::ssm::build_nemotron_moe_ptrs_decode_batched_bf16`: the
    /// same, for the decode path that skips the permutation entirely.
    pub fn build_nemotron_moe_ptrs_decode(topk_idx: &Val, topk_w: &Val, x: &Val, l: u32) {
        record(
            &topk_idx.t,
            Some(l),
            "ssm::build_nemotron_moe_ptrs_decode_batched_bf16",
            vec![],
            None,
            vec![topk_idx.id, topk_w.id, x.id],
            None,
        );
    }

    /// `kernels::moe::token_batched_weighted_sum_aligned_bf16`: combine the
    /// aligned expert outputs back per token.
    pub fn token_batched_weighted_sum_aligned(
        aligned_out: &Val,
        topk_w: &Val,
        hidden: u32,
    ) -> Val {
        record(
            &aligned_out.t,
            aligned_out.layer,
            "moe::token_batched_weighted_sum_aligned_bf16",
            vec![],
            None,
            vec![aligned_out.id, topk_w.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
        )
        .expect("the combine produces its value")
    }

    // ── KDA: Kimi Delta Attention ──────────────────────────────────
    //
    // The linear-attention half of kimi_k3. Same gated delta rule qwen3_5
    // runs, with one difference that changes every kernel: the decay is per
    // KEY CHANNEL, not per head. Qwen3.5 multiplies the whole `[K_d, V_d]`
    // state slab by one scalar `exp(g_h)`; KDA multiplies column `k` by
    // `exp(gate[h, k])`. That is the "delta" in the name -- a fine-grained
    // forget gate -- and it is why these are their own kernels rather than
    // GDN's with a broadcast.
    //
    // All the arithmetic is fp32; bf16 operands are widened first, which is
    // why the dtype casts below are statements rather than annotations.

    /// `kernels::ssm::kda_gate_beta_bf16`: the forget gate and the write
    /// strength, from their raw projections.
    ///
    /// Returns `(gate, beta)`, both fp32. `A_log` is per head and `dt_bias`
    /// per head-channel, so both are WEIGHTS the launch reads.
    pub fn kda_gate_beta(
        raw_g: &Val,
        raw_beta: &Val,
        a_log: &str,
        dt_bias: &str,
        heads: u32,
        head_dim: u32,
    ) -> (Val, Val) {
        let outs = record_many(
            &raw_g.t,
            raw_g.layer,
            "ssm::kda_gate_beta_bf16",
            vec![a_log.to_string(), dt_bias.to_string()],
            vec![raw_g.id, raw_beta.id],
            vec![
                (
                    Shape(vec![Dim::Tokens, Dim::Const(heads * head_dim)]),
                    DType::F32,
                ),
                (Shape(vec![Dim::Tokens, Dim::Const(heads)]), DType::F32),
            ],
        );
        let mut it = outs.into_iter();
        let gate = it.next().expect("the gate states two outputs");
        let beta = it.next().expect("the gate states two outputs");
        (gate, beta)
    }

    /// `kernels::ssm::kda_recurrent_step_batched`: one decode token per
    /// request, advancing each request's state slot.
    ///
    /// `whole`: `slot_ids` is indexed `0..R` against the fire's request
    /// order, so a row window would advance the wrong slots.
    pub fn kda_recurrent_step(
        q: &Val,
        k: &Val,
        v: &Val,
        gate: &Val,
        beta: &Val,
        l: u32,
        heads: u32,
        head_dim: u32,
    ) -> Val {
        record(
            &q.t,
            Some(l),
            "ssm::kda_recurrent_step_batched",
            vec![],
            Some(StateRef {
                store: StateStore::RecurrentState,
                layer: l,
            }),
            vec![q.id, k.id, v.id, gate.id, beta.id],
            Some((
                Shape(vec![
                    Dim::Requests,
                    Dim::Const(heads),
                    Dim::Const(head_dim),
                ]),
                DType::F32,
            )),
        )
        .expect("the recurrence produces its value")
    }

    /// `kernels::ssm::kda_prefill_batched`: the same recurrence over a
    /// prefill window, one block per (request, head).
    ///
    /// `whole` twice over: it walks windows out of `qo_indptr`, AND the
    /// recurrence has a strict per-token state dependency -- the block walks
    /// its window one token at a time because token `t`'s state is token
    /// `t-1`'s output. A row window would start the scan from the wrong
    /// state, which is a different answer rather than a misaddressed one.
    pub fn kda_prefill(
        q: &Val,
        k: &Val,
        v: &Val,
        gate: &Val,
        beta: &Val,
        l: u32,
        heads: u32,
        head_dim: u32,
    ) -> Val {
        record(
            &q.t,
            Some(l),
            "ssm::kda_prefill_batched",
            vec![],
            Some(StateRef {
                store: StateStore::RecurrentState,
                layer: l,
            }),
            vec![q.id, k.id, v.id, gate.id, beta.id],
            Some((
                Shape(vec![
                    Dim::Tokens,
                    Dim::Const(heads),
                    Dim::Const(head_dim),
                ]),
                DType::F32,
            )),
        )
        .expect("the recurrence produces its value")
    }

    /// `kernels::ssm::kda_o_norm_gated_bf16`: the output norm and gate.
    pub fn kda_o_norm_gated(x: &Val, gate: &Val, weight: &str, width: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "ssm::kda_o_norm_gated_bf16",
            vec![weight.to_string()],
            None,
            vec![x.id, gate.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16))
        )
        .expect("the norm produces its value")
    }

    // ── kimi_k3: SiTU, and the fp32 widenings its recurrence needs ─

    /// `kernels::mlp::situ_bf16` / `kernels::mlp::chunked_situ_bf16`: Moonshot's
    /// `SituAndMul`.
    ///
    /// Not a swiglu variant. The tanh saturates far enough out (beta 4,
    /// linear_beta 25 on K3) that a bf16 intermediate loses the distinction
    /// the gate exists to make, so the kernel evaluates in fp32 and narrows
    /// once. `packed` picks the chunked form, the same binding choice
    /// [`Self::swiglu`] carries.
    pub fn situ(x: &Val, intermediate: u32, packed: bool) -> Val {
        record(
            &x.t,
            x.layer,
            if packed {
                "mlp::chunked_situ_bf16"
            } else {
                "mlp::situ_bf16"
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

    /// `kernels::ssm::l2norm_scale_bf16_to_fp32`: l2-norm each row and
    /// scale, widening to fp32.
    ///
    /// `y[r,c] = x[r,c] / sqrt(Σ_c x[r,c]² + eps) · scale`. The recurrence
    /// above wants q pre-scaled by `K_d^(-1/2)`; this is where that happens.
    pub fn l2norm_scale_to_f32(x: &Val, width: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "ssm::l2norm_scale_bf16_to_fp32",
            vec![],
            None,
            vec![x.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::F32)),
        )
        .expect("the norm produces its value")
    }

    /// `kernels::ssm::bf16_to_fp32`: widen.
    ///
    /// A statement rather than a dtype annotation because it is a launch, and
    /// the trace records launches. KDA's arithmetic is fp32 throughout, so an
    /// operand that lives in bf16 in the workspace crosses here.
    pub fn bf16_to_f32(x: &Val, width: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "ssm::bf16_to_fp32",
            vec![],
            None,
            vec![x.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::F32)),
        )
        .expect("the cast produces its value")
    }

    /// `kernels::ssm::fp32_to_bf16`: narrow, on the way back out.
    pub fn f32_to_bf16(x: &Val, width: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "ssm::fp32_to_bf16",
            vec![],
            None,
            vec![x.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
        )
        .expect("the cast produces its value")
    }

    /// `kernels::attn::attn_res_blend_bf16`: blend the prefix output with
    /// the open blocks', weighted by a learned score.
    ///
    /// With no open blocks the softmax is over a single row and the output IS
    /// the prefix, which is what the tail blend of a model with no open
    /// blocks means -- the kernel's own header says so, and it is why this
    /// needs no guard around it.
    pub fn attn_res_blend(
        prefix: &Val,
        blocks: &Val,
        norm_weight: &str,
        proj_weight: &str,
        width: u32,
    ) -> Val {
        record(
            &prefix.t,
            prefix.layer,
            "attn::attn_res_blend_bf16",
            vec![norm_weight.to_string(), proj_weight.to_string()],
            None,
            vec![prefix.id, blocks.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
        )
        .expect("the blend produces its value")
    }

    // ── tensor-parallel shapes ─────────────────────────────────────

    /// `kernels::layout::embed_bf16_vocab_shard`: gather from a vocab-SHARDED
    /// embedding table.
    ///
    /// Under tensor parallelism the table is split along the vocabulary, so a
    /// rank holds `[local_vocab, hidden]` starting at `vocab_offset` and
    /// writes zeros for tokens outside its shard; the all-reduce that follows
    /// makes the row whole. Row-shaped, so not `whole` — the shard is a
    /// property of the WEIGHT, not of the row range.
    pub fn embed_vocab_shard(t: &Trace, weight: &str, hidden: u32) -> Val {
        record(
            t,
            None,
            "layout::embed_bf16_vocab_shard",
            vec![weight.to_string()],
            None,
            vec![],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(hidden)]),
                DType::BF16,
            )),
        )
        .expect("the gather produces its value")
    }

    /// `kernels::norm::residual_add_rmsnorm_bf16`: the residual add and the
    /// next block's pre-norm, fused.
    ///
    /// `hidden = round_bf16(hidden + residual)` then
    /// `norm_out = rmsnorm(hidden, weight)`. The kernel's own header states
    /// that the rounding matches `kernels::norm::residual_add_bf16`'s, so this is
    /// numerically the two-kernel sequence and not an approximation of it —
    /// which is what makes it a BINDING choice a declaration may state rather
    /// than a different computation.
    pub fn residual_add_rmsnorm(x: &Val, residual: &Val, weight: &str, hidden: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "norm::residual_add_rmsnorm_bf16",
            vec![weight.to_string()],
            None,
            vec![x.id, residual.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(hidden)]),
                DType::BF16,
            )),
        )
        .expect("the fused norm produces its value")
    }

    // ── MLA: the kimi splits ───────────────────────────────────────
    //
    // The unfused counterpart of [`mla_prepare`]. Kimi projects `q_a` and
    // `kv_a` in one GEMM and splits afterwards; both statements are row-shaped
    // (`tokens` is their only extent), so unlike the fused prepare they are
    // NOT `whole` — which is the whole reason a deployment might bind them
    // instead.

    /// `kernels::attn::kimi_split_kv_a_norm_bf16`: split the latent KV
    /// projection into `(kv_c, k_pe)`, norming the compressed half on the way.
    ///
    /// The norm is folded in, so this is one statement where the semantic
    /// reading is two (`rmsnorm` then a split).
    pub fn kimi_split_kv_a_norm(
        kv_a: &Val,
        norm_weight: &str,
        kv_lora_rank: u32,
        qk_rope_dim: u32,
    ) -> (Val, Val) {
        let outs = record_many(
            &kv_a.t,
            kv_a.layer,
            "attn::kimi_split_kv_a_norm_bf16",
            vec![norm_weight.to_string()],
            vec![kv_a.id],
            vec![
                (
                    Shape(vec![Dim::Tokens, Dim::Const(kv_lora_rank)]),
                    DType::BF16,
                ),
                (
                    Shape(vec![Dim::Tokens, Dim::Const(qk_rope_dim)]),
                    DType::BF16,
                ),
            ],
        );
        let mut it = outs.into_iter();
        let kv_c = it.next().expect("the split states two outputs");
        let k_pe = it.next().expect("the split states two outputs");
        (kv_c, k_pe)
    }

    /// `kernels::attn::kimi_split_q_b_bf16`: split the query projection into
    /// its nope and rope halves.
    pub fn kimi_split_q_b(
        q_b: &Val,
        heads: u32,
        qk_nope_dim: u32,
        qk_rope_dim: u32,
    ) -> (Val, Val) {
        let outs = record_many(
            &q_b.t,
            q_b.layer,
            "attn::kimi_split_q_b_bf16",
            vec![],
            vec![q_b.id],
            vec![
                (
                    Shape(vec![
                        Dim::Tokens,
                        Dim::Const(heads),
                        Dim::Const(qk_nope_dim),
                    ]),
                    DType::BF16,
                ),
                (
                    Shape(vec![
                        Dim::Tokens,
                        Dim::Const(heads),
                        Dim::Const(qk_rope_dim),
                    ]),
                    DType::BF16,
                ),
            ],
        );
        let mut it = outs.into_iter();
        let q_nope = it.next().expect("the split states two outputs");
        let q_pe = it.next().expect("the split states two outputs");
        (q_nope, q_pe)
    }

    // ── DSA: the lightning indexer ─────────────────────────────────
    //
    // glm5 attends SPARSELY: a small side network scores every (query, key)
    // pair, and only the top-k keys per query are attended. The two rope
    // statements prepare that indexer's own q and k; the third scores and
    // thresholds, and its output is the mask MLA's `index_mask` reads.
    //
    // The mask is the one statement here that is `whole`, and the reason is
    // the algebra rather than the addressing: query `i` scores keys `0..=i`,
    // so a row window that starts anywhere but zero cannot see the keys it
    // must rank against.

    /// `kernels::attn::dsa_index_q_rope_bf16`: interleaved rope on each
    /// index head of the indexer's queries.
    pub fn dsa_index_q_rope(idx_q: &Val, heads: u32, head_dim: u32) -> Val {
        record(
            &idx_q.t,
            idx_q.layer,
            "attn::dsa_index_q_rope_bf16",
            vec![],
            None,
            vec![idx_q.id],
            Some((
                Shape(vec![
                    Dim::Tokens,
                    Dim::Const(heads),
                    Dim::Const(head_dim),
                ]),
                DType::BF16,
            )),
        )
        .expect("the rope produces its value")
    }

    /// `kernels::attn::dsa_index_knorm_rope_bf16`: layernorm then rope on the
    /// indexer's keys.
    ///
    /// A LayerNorm with a bias, not the RMS norm the rest of the model uses —
    /// which is why it is its own statement rather than `rmsnorm` followed by
    /// `rope`.
    pub fn dsa_index_knorm_rope(idx_k: &Val, head_dim: u32) -> Val {
        record(
            &idx_k.t,
            idx_k.layer,
            "attn::dsa_index_knorm_rope_bf16",
            vec![],
            None,
            vec![idx_k.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(head_dim)]),
                DType::BF16,
            )),
        )
        .expect("the norm+rope produces its value")
    }

    /// `kernels::attn::dsa_index_topk_mask`: score every causal (query, key)
    /// pair and keep the top-k per query.
    ///
    /// `logit[i,j] = Σ_h relu(idx_q[i,h,·] · idx_k[j,·]) · idx_w[i,h]`, then
    /// the mask is 1 for the top-`k` of `j <= i`. The output is `[T, T]`, and
    /// it is what MLA's `index_mask` consumes.
    pub fn dsa_index_topk_mask(idx_q: &Val, idx_k: &Val, idx_w: &Val) -> Val {
        record(
            &idx_q.t,
            idx_q.layer,
            "attn::dsa_index_topk_mask",
            vec![],
            None,
            vec![idx_q.id, idx_k.id, idx_w.id],
            Some((Shape(vec![Dim::Tokens, Dim::Tokens]), DType::I32)),
        )
        .expect("the indexer produces its mask")
    }

    // ── MoE: the ALIGNED dispatch path ─────────────────────────────
    //
    // glm5 and kimi_k3 route through a permutation, not a loop. Every
    // (token, expert) pair is a ROUTE; the routes are bucketed by expert and
    // padded to fixed-size blocks so one batched GEMM covers every expert at
    // once, and the permutation is undone afterwards.
    //
    // Five of the six are `whole`, and it is the same reason each time: the
    // permutation is computed over ALL routes in the fire. `sorted_route_ids`
    // is a global order, so a statement addressed through it cannot be handed
    // a row window -- the window would name different routes than the sort
    // did. This is the `dyn` axis the trace module doc describes, at the one
    // point where it stops being expressible as a row range.

    /// `kernels::moe::topk_sigmoid_bf16`: the router — each token's top-k
    /// experts and their weights, gated by sigmoid rather than softmax.
    ///
    /// Returns `(topk_idx, topk_w)`. The ONE statement here that is not
    /// `whole`: a token's routing reads only its own logits row.
    pub fn topk_sigmoid(logits: &Val, top_k: u32) -> (Val, Val) {
        let outs = record_many(
            &logits.t,
            logits.layer,
            "moe::topk_sigmoid_bf16",
            vec![],
            vec![logits.id],
            vec![
                (Shape(vec![Dim::Tokens, Dim::Const(top_k)]), DType::I32),
                (Shape(vec![Dim::Tokens, Dim::Const(top_k)]), DType::F32),
            ],
        );
        let mut it = outs.into_iter();
        let idx = it.next().expect("the router states two outputs");
        let w = it.next().expect("the router states two outputs");
        (idx, w)
    }

    /// `kernels::moe::moe_align_decode`: bucket the routes by expert and
    /// pad each bucket to a block.
    ///
    /// Returns `(sorted_route_ids, expert_ids, route_to_aligned_row)` — the
    /// permutation, which expert each block belongs to, and the inverse map
    /// the combine reads.
    /// The three load-time numbers ride the param channel. They are the
    /// permutation's own shape — how many experts to bucket into, how
    /// wide a block is, how many blocks the padding admits — and the
    /// executor was reading two of them out of a config struct and one
    /// out of its MoE workspace.
    pub fn moe_align(
        topk_idx: &Val,
        max_blocks: u32,
        block_size: u32,
        top_k: u32,
        num_experts: u32,
    ) -> (Val, Val, Val) {
        let routes = Dim::Const(top_k);
        let outs = record_many_with_params(
            &topk_idx.t,
            topk_idx.layer,
            "moe::moe_align_decode",
            vec![],
            vec![num_experts, block_size, max_blocks],
            vec![topk_idx.id],
            vec![
                (
                    Shape(vec![Dim::Const(max_blocks * block_size)]),
                    DType::I32,
                ),
                (Shape(vec![Dim::Const(max_blocks)]), DType::I32),
                (Shape(vec![Dim::Tokens, routes]), DType::I32),
            ],
        );
        let mut it = outs.into_iter();
        let sorted = it.next().expect("the align states three outputs");
        let experts = it.next().expect("the align states three outputs");
        let inverse = it.next().expect("the align states three outputs");
        (sorted, experts, inverse)
    }

    /// `kernels::moe::gather_moe_aligned_inputs_bf16`: the block-major
    /// operand, gathered in the sorted order.
    pub fn gather_moe_aligned_inputs(
        x: &Val,
        sorted_route_ids: &Val,
        aligned: Dim,
        hidden: u32,
    ) -> Val {
        record(
            &x.t,
            x.layer,
            "moe::gather_moe_aligned_inputs_bf16",
            vec![],
            None,
            vec![x.id, sorted_route_ids.id],
            Some((Shape(vec![aligned, Dim::Const(hidden)]), DType::BF16)),
        )
        .expect("the gather produces its value")
    }

    /// `kernels::moe::build_moe_ptrs_aligned_bf16`: the aligned leg's
    /// staging, and the pointer arrays one batched GEMM per projection
    /// needs into it.
    ///
    /// It DECLARES the staging, which is the only SSA-valid way to say
    /// what this call does. The kernel bakes the three staging buffers'
    /// BASE ADDRESSES into device pointer arrays, so it has to know
    /// where they are before anything writes them — and a statement
    /// cannot take an operand that a later statement produces. So the
    /// build is what fixes where the aligned staging lives, and the two
    /// grouped GEMMs and the swiglu between them fill buffers it named:
    /// each takes its destination as an operand and writes it in place.
    /// Before this, all three were `mw.aligned_*` in the executor and
    /// the declaration ended at "a pointer build happens here".
    ///
    /// `(gate_up, act, out)` — `[aligned, 2·I]`, `[aligned, I]`,
    /// `[aligned, H]`, all bf16, all block-major.
    ///
    /// The six POINTER ARRAYS are still the driver's: an array of device
    /// addresses has no dtype in this vocabulary, and inventing one to
    /// hold `void*` is a wider change than this statement needs. They
    /// are reachable only from the two GEMMs that this call also serves,
    /// so the gap is bounded — see the executor's fallback arm.
    pub fn build_moe_ptrs_aligned(
        expert_ids: &Val,
        aligned_in: &Val,
        l: u32,
        gate_up_bank: &str,
        down_bank: &str,
        aligned: Dim,
        hidden: u32,
        moe_intermediate: u32,
    ) -> (Val, Val, Val) {
        let outs = record_many(
            &expert_ids.t,
            Some(l),
            "moe::build_moe_ptrs_aligned_bf16",
            vec![gate_up_bank.to_string(), down_bank.to_string()],
            vec![expert_ids.id, aligned_in.id],
            vec![
                (
                    Shape(vec![aligned, Dim::Const(2 * moe_intermediate)]),
                    DType::BF16,
                ),
                (
                    Shape(vec![aligned, Dim::Const(moe_intermediate)]),
                    DType::BF16,
                ),
                (Shape(vec![aligned, Dim::Const(hidden)]), DType::BF16),
            ],
        );
        let mut it = outs.into_iter();
        let gate_up = it.next().expect("the ptr build states three stages");
        let act = it.next().expect("the ptr build states three stages");
        let out = it.next().expect("the ptr build states three stages");
        (gate_up, act, out)
    }

    /// `kernels::moe::reorder_moe_aligned_output_bf16`: undo the block
    /// permutation, back to route order.
    pub fn reorder_moe_aligned_output(
        aligned_out: &Val,
        sorted_route_ids: &Val,
        top_k: u32,
        hidden: u32,
    ) -> Val {
        record(
            &aligned_out.t,
            aligned_out.layer,
            "moe::reorder_moe_aligned_output_bf16",
            vec![],
            None,
            vec![aligned_out.id, sorted_route_ids.id],
            Some((
                Shape(vec![
                    Dim::Tokens,
                    Dim::Const(top_k),
                    Dim::Const(hidden),
                ]),
                DType::BF16,
            )),
        )
        .expect("the reorder produces its value")
    }

    /// `kernels::moe::scatter_add_weighted_bf16`: fold the routed rows back
    /// onto the residual stream, each scaled by its router weight.
    ///
    /// `out[dst_idx[i]] += src[i] · row_weights[i]`. `whole` because
    /// `dst_idx` is route-global: a window over output ROWS is not a window
    /// over routes.
    pub fn scatter_add_weighted(
        out: &Val,
        src: &Val,
        dst_idx: &Val,
        row_weights: &Val,
        hidden: u32,
    ) -> Val {
        record(
            &out.t,
            out.layer,
            "moe::scatter_add_weighted_bf16",
            vec![],
            None,
            vec![out.id, src.id, dst_idx.id, row_weights.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(hidden)]),
                DType::BF16,
            )),
        )
        .expect("the combine produces its value")
    }

    // ── MLA: latent attention ──────────────────────────────────────
    //
    // deepseek_v4, glm5 and kimi_k3 all attend through a LATENT KV: the
    // cache stores a `kv_lora_rank`-wide compressed row plus a small
    // rope-carrying `qk_rope_head_dim` row, and the heads are reconstructed
    // on the way in. It is a different attention algebra, not a different
    // head count -- which is why none of the flashinfer statements above can
    // stand in for it, and why it gets its own [`Prepare::MlaPlan`].

    /// `kernels::attn::mla_prepare_bf16`: one launch that turns the two
    /// projections into the four operands MLA attends over.
    ///
    /// Returns `(kv_c, k_pe, q_nope, q_pe)` — the compressed KV row, its
    /// rope-carrying companion, and the query split the same way. It is one
    /// statement rather than four because the kernel is one launch, and the
    /// trace records launches.
    ///
    /// `whole`: it addresses through `qo_indptr` / `kv_page_indptr` /
    /// `kv_last_page_lens`, which are R-shaped. A row window would leave that
    /// arithmetic pointing at the wrong request.
    pub fn mla_prepare(
        kv_a: &Val,
        q_b: &Val,
        heads: u32,
        kv_lora_rank: u32,
        qk_nope_dim: u32,
        qk_rope_dim: u32,
    ) -> (Val, Val, Val, Val) {
        let outs = record_many(
            &kv_a.t,
            kv_a.layer,
            "attn::mla_prepare_bf16",
            vec![],
            vec![kv_a.id, q_b.id],
            vec![
                (
                    Shape(vec![Dim::Tokens, Dim::Const(kv_lora_rank)]),
                    DType::BF16,
                ),
                (
                    Shape(vec![Dim::Tokens, Dim::Const(qk_rope_dim)]),
                    DType::BF16,
                ),
                (
                    Shape(vec![
                        Dim::Tokens,
                        Dim::Const(heads),
                        Dim::Const(qk_nope_dim),
                    ]),
                    DType::BF16,
                ),
                (
                    Shape(vec![
                        Dim::Tokens,
                        Dim::Const(heads),
                        Dim::Const(qk_rope_dim),
                    ]),
                    DType::BF16,
                ),
            ],
        );
        let mut it = outs.into_iter();
        let kv_c = it.next().expect("mla_prepare states four outputs");
        let k_pe = it.next().expect("mla_prepare states four outputs");
        let q_nope = it.next().expect("mla_prepare states four outputs");
        let q_pe = it.next().expect("mla_prepare states four outputs");
        (kv_c, k_pe, q_nope, q_pe)
    }

    /// `kernels::attn::write_mla_to_pages`: commit the compressed KV row and
    /// its rope companion to the paged latent cache.
    ///
    /// The MLA counterpart of `write_kv_to_pages`, and `whole` for the same
    /// reason `mla_prepare` is: page addressing is per-request.
    pub fn write_mla_to_pages(kv_c: &Val, k_pe: &Val, l: u32) {
        record(
            &kv_c.t,
            Some(l),
            "attn::write_mla_to_pages",
            vec![],
            Some(StateRef {
                store: StateStore::KvCache,
                layer: l,
            }),
            vec![kv_c.id, k_pe.id],
            None,
        );
    }

    /// `kernels::attn::dispatch_attention_mla_bf16`: attention over the latent cache.
    ///
    /// `needs` [`Prepare::MlaPlan`] — its own kind of plan, built from the
    /// latent geometry (`kv_lora_rank`, `qk_rope_head_dim`) that no other
    /// prepare has a field for, and cached in an `MlaPlanCache` rather than
    /// in the shared attention workspace.
    ///
    /// `lacks Scores`: there is no capture variant of this dispatch, so a
    /// program whose `attn.out` seam wants the score matrix cannot be served
    /// over rows this kernel covers. It publishes an LSE, which is a
    /// different thing and not what the capability names.
    pub fn attention_mla(q_nope: &Val, q_pe: &Val, l: u32, heads: u32, kv_lora_rank: u32) -> Val {
        record(
            &q_nope.t,
            Some(l),
            "attn::dispatch_attention_mla_bf16",
            vec![],
            Some(StateRef {
                store: StateStore::KvCache,
                layer: l,
            }),
            vec![q_nope.id, q_pe.id],
            Some((
                Shape(vec![
                    Dim::Tokens,
                    Dim::Const(heads),
                    Dim::Const(kv_lora_rank),
                ]),
                DType::BF16,
            )),
        )
        .expect("the attention produces its value")
    }

    /// `kernels::attn::attention_flashinfer_prefill_custom`: the custom-mask
    /// prefill in its PLAN-FREE form.
    ///
    /// The counterpart of [`Self::flashinfer_prefill_planless`]'s reasoning:
    /// it takes the indptr arrays and the mask directly and builds its
    /// R-shaped plan on the way in, so it owes its caller no prepare and
    /// cannot be handed a row window. `whole`, and `FireWide` for the same
    /// reason XQA is.
    ///
    /// gemma-3n binds this rather than the planned `flashinfer_custom` above,
    /// which is a deployment fact and therefore something a declaration
    /// states.
    pub fn flashinfer_prefill_custom_planless(q: &Val, l: u32, heads: u32, head_dim: u32) -> Val {
        record(
            &q.t,
            Some(l),
            "attn::attention_flashinfer_prefill_custom",
            vec![],
            Some(StateRef {
                store: StateStore::KvCache,
                layer: l,
            }),
            vec![q.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(heads * head_dim)]),
                DType::BF16,
            )),
        )
        .expect("the attention produces its value")
    }

    // ── gemma-3n: AltUp ────────────────────────────────────────────
    //
    // Gemma-3n carries K = `altup_num_inputs` PARALLEL residual streams
    // instead of one. Each layer predicts the post-layer state of all K
    // from a learned per-token combination of them, runs the real layer on
    // one ACTIVE stream, and corrects the other K-1 from the difference.
    // That is a residual stream with a rank, and it is why gemma-3n cannot
    // be written as `llama_like` with different facts: `Dim::Tokens` rows
    // are still rows, but the value under them is `[K, T, H]`.
    //
    // None of these carry a contract clause. Every one is row-shaped —
    // token `t`'s output reads only token `t`'s inputs — so a peel may
    // split them, no host plan is obligated, and there is no seam
    // capability for one to refuse.

    /// `kernels::norm::altup_predict_bf16`: the K streams' post-layer
    /// state, predicted.
    ///
    /// `predictions[k, t, h] = streams[k, t, h] + Σ_j coefs[t, j, k]·streams[j, t, h]`
    ///
    /// `coefs` is fp32 and stays fp32: the K-summation accumulates
    /// round-off that bf16 cannot absorb, which the kernel's own header
    /// says is why it takes a float pointer.
    pub fn altup_predict(streams: &Val, coefs: &Val, k: u32, hidden: u32) -> Val {
        record(
            &streams.t,
            streams.layer,
            "norm::altup_predict_bf16",
            vec![],
            None,
            vec![streams.id, coefs.id],
            Some((
                Shape(vec![Dim::Const(k), Dim::Tokens, Dim::Const(hidden)]),
                DType::BF16,
            )),
        )
        .expect("the prediction produces its value")
    }

    /// `kernels::norm::altup_correct_bf16`: the other K-1 streams,
    /// corrected from what the active one actually computed.
    ///
    /// `corrected[k] = predictions[k] + (activated - predictions[active])·(coefs[t,k] + 1)`
    ///
    /// The `+1` is folded into the coefficient by
    /// [`altup_unpack_correct_coefs`], not by this kernel.
    pub fn altup_correct(
        predictions: &Val,
        activated: &Val,
        correction_coefs: &Val,
        k: u32,
        hidden: u32,
    ) -> Val {
        record(
            &predictions.t,
            predictions.layer,
            "norm::altup_correct_bf16",
            vec![],
            None,
            vec![predictions.id, activated.id, correction_coefs.id],
            Some((
                Shape(vec![Dim::Const(k), Dim::Tokens, Dim::Const(hidden)]),
                DType::BF16,
            )),
        )
        .expect("the correction produces its value")
    }

    /// `kernels::norm::altup_unpack_predict_coefs`: the router's bf16
    /// `[T, K*K]` output as the fp32 `[T, K, K]` [`altup_predict`] reads.
    ///
    /// Not a cast. It also applies the transpose HF spells
    /// `.reshape(*, K, K).permute(0, 1, 3, 2)`, so the statement is a
    /// distinct op rather than a dtype annotation on the matmul above it.
    pub fn altup_unpack_predict_coefs(packed: &Val, k: u32) -> Val {
        record(
            &packed.t,
            packed.layer,
            "norm::altup_unpack_predict_coefs",
            vec![],
            None,
            vec![packed.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(k), Dim::Const(k)]),
                DType::F32,
            )),
        )
        .expect("the unpack produces its value")
    }

    /// `kernels::norm::altup_unpack_correct_coefs`: the same for the
    /// correction's `[T, K]`, with HF's `+ 1.0` folded in.
    pub fn altup_unpack_correct_coefs(packed: &Val, k: u32) -> Val {
        record(
            &packed.t,
            packed.layer,
            "norm::altup_unpack_correct_coefs",
            vec![],
            None,
            vec![packed.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(k)]), DType::F32)),
        )
        .expect("the unpack produces its value")
    }

    /// `kernels::norm::mean_streams_bf16`: the K streams averaged into
    /// one — `out[t, h] = (1/K) Σ_k streams[k, t, h]`.
    ///
    /// How a rank-K residual stream is read by anything that expects one.
    pub fn mean_streams(streams: &Val, hidden: u32) -> Val {
        record(
            &streams.t,
            streams.layer,
            "norm::mean_streams_bf16",
            vec![],
            None,
            vec![streams.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(hidden)]),
                DType::BF16,
            )),
        )
        .expect("the mean produces its value")
    }

    /// `kernels::norm::compute_rms_bf16`: each row's RMS, as fp32.
    ///
    /// A MEASUREMENT, not a normalization: it produces the target that
    /// [`magnitude_rescale`] then holds another tensor to. The pair exists
    /// because gemma-3n keeps a stream's magnitude fixed across a
    /// projection rather than re-norming it.
    pub fn compute_rms(x: &Val) -> Val {
        record(
            &x.t,
            x.layer,
            "norm::compute_rms_bf16",
            vec![],
            None,
            vec![x.id],
            Some((Shape(vec![Dim::Tokens]), DType::F32)),
        )
        .expect("the measurement produces its value")
    }

    /// `kernels::norm::magnitude_rescale_bf16`: scale each row of `x` so
    /// its RMS equals `target`'s.
    ///
    /// In place in the kernel; a value here, because a trace records what
    /// a statement produces and the reader should not have to know which
    /// buffer it landed in.
    pub fn magnitude_rescale(x: &Val, target_rms: &Val, hidden: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "norm::magnitude_rescale_bf16",
            vec![],
            None,
            vec![x.id, target_rms.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(hidden)]),
                DType::BF16,
            )),
        )
        .expect("the rescale produces its value")
    }

    /// `kernels::norm::tanh_bf16` on AltUp's modality-router output.
    ///
    /// HF computes this in fp32 and casts back; the kernel folds both, so
    /// the trace states one op where the reference states three.
    /// The result is the OPERAND's shape, read off the trace rather than
    /// respelled: this kernel takes one pointer and rewrites it, so the
    /// two are one buffer and a second spelling can only disagree.
    ///
    /// It did. `[Tokens, width]` was the spelling, and gemma-3n's altup
    /// coefficients run over a `Select`ed stream slice whose leading dim
    /// is the STREAM count, not the fire's tokens — so the operand was
    /// `[4, 4]` and the result claimed `[Tokens, 4]`. Nothing compared
    /// them until the row said in place, at which point the arena put
    /// one buffer where two shapes disagreed and
    /// `an_alias_lands_inside_its_owner` refused it.
    pub fn tanh(x: &Val) -> Val {
        let out = (x.t.inner.borrow().value_shape(x.id), DType::BF16);
        record(
            &x.t,
            x.layer,
            "norm::tanh_bf16",
            vec![],
            None,
            vec![x.id],
            Some(out),
        )
        .expect("the activation produces its value")
    }

    /// `kernels::mlp::gaussian_topk_bf16`: gemma-3n's activation
    /// sparsity — zero every element below `mean + std_multiplier·std` of
    /// its own row.
    ///
    /// A top-k by THRESHOLD rather than by count, which is what lets it be
    /// row-shaped: no sort, no cross-row comparison, so a peel may split
    /// it like any other elementwise statement.
    pub fn gaussian_topk(x: &Val, width: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "mlp::gaussian_topk_bf16",
            vec![],
            None,
            vec![x.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
        )
        .expect("the sparsifier produces its value")
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
                "mlp::chunked_geglu_tanh_bf16"
            } else {
                "mlp::geglu_tanh_bf16"
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

    /// `kernels::mlp::geglu_tanh_bf16` in its PAIR form: the gate and
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
            "mlp::geglu_tanh_bf16",
            vec![],
            None,
            vec![gate.id, up.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
        )
        .expect("the activation produces its value")
    }

    /// `kernels::rope::rope_partial_bf16` rotating Q ALONE.
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
    pub fn rope_partial_q_only(q: &Val, rotary_dim: u32) -> Val {
        let out = (q.t.inner.borrow().value_shape(q.id), DType::BF16);
        record_with_params(
            &q.t,
            q.layer,
            "rope::rope_partial_bf16",
            vec![],
            None,
            vec![rotary_dim],
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
            "rope::qk_rmsnorm_rope_bf16_rounded",
            vec![q_norm.name.clone()],
            None,
            vec![q.id],
            Some(out),
        )
        .expect("the fused pair produces q")
    }

    /// `kernels::norm::rmsnorm_no_scale_bf16`: `v / rms(v)` per head,
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
            "norm::rmsnorm_no_scale_bf16",
            vec![],
            None,
            vec![x.id],
            Some(out),
        )
        .expect("the norm produces its value")
    }

    /// `kernels::norm::rmsnorm_residual_add_scale_rmsnorm_bf16`: FOUR
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
        y: &Val,
        w: &NormW,
        next: &NormW,
        hidden: u32,
    ) -> (Val, Val) {
        let shape = (Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16);
        let ids = x.t.with(w.layer, |b| {
            b.launch(
                "norm::rmsnorm_residual_add_scale_rmsnorm_bf16",
                vec![w.name.clone(), next.name.clone()],
                None,
                // The STREAM is an operand. The kernel reads it and
                // accumulates into it, so a statement that named only `x`
                // left SSA with no edge from the old stream to the new
                // one -- and an executor binding buffers from the edges
                // then handed the launch a fresh buffer to land on.
                vec![x.id, y.id],
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

    /// `kernels::norm::rmsnorm_residual_add_bf16`: the two-statement
    /// form — norm, then land on the stream. gemma-4's
    /// post-feedforward norm, where no next-block norm follows to fuse.
    pub fn norm_residual_add(x: &Val, y: &Val, w: &NormW, hidden: u32) -> Val {
        record(
            &x.t,
            w.layer,
            "norm::rmsnorm_residual_add_bf16",
            vec![w.name.clone()],
            None,
            // `y` is the residual stream this lands on: read, accumulated
            // into, and returned. Naming it is what gives the new stream
            // an SSA edge to the old one.
            vec![x.id, y.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
        )
        .expect("the fused norm+residual produces its value")
    }

    /// `kernels::norm::scalar_mul_bf16`: multiply by a load-time
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
    /// The NUMBER rides the param channel, in the bits an untyped `u32`
    /// slot already has room for. The name stays because a reader wants
    /// it; the driver used to need it, and that is the difference. It
    /// held a name-to-arithmetic table — `sqrt(hidden)`, `sqrt(ple_dim)`,
    /// `1/sqrt(hidden)`, `1/sqrt(2)` — recomputing on the device side
    /// what the host had already derived from its own dims, and an
    /// unrecognised name was a runtime refusal rather than a number.
    /// `by` is OPTIONAL, and a `None` is a family saying its facts do
    /// not carry the number yet — gemma-3n's altup and laurel scales and
    /// gemma-2's query scale are per-layer constants nothing on the host
    /// side has derived, and inventing one here would be worse than the
    /// name it replaces. A statement without the param falls through the
    /// generated branch's arity guard to whatever arm knows better,
    /// which for those two families is the hand-written pass.
    pub fn scalar_mul(x: &Val, scale: &str, by: Option<f32>) -> Val {
        let out = (x.t.inner.borrow().value_shape(x.id), DType::BF16);
        record_with_params(
            &x.t,
            x.layer,
            "norm::scalar_mul_bf16",
            vec![format!("scale.{scale}")],
            None,
            by.map(f32::to_bits).into_iter().collect(),
            vec![x.id],
            Some(out),
        )
        .expect("the scale produces its value")
    }

    /// `kernels::attn::logit_softcap_bf16`: `cap * tanh(x / cap)` over
    /// the logits. A load-time fact decides whether it runs at all
    /// (`final_logit_softcapping`), so its presence is a trace-time
    /// match, not a branch.
    pub fn logit_softcap(x: &Val, vocab: u32) -> Val {
        record(
            &x.t,
            None,
            "attn::logit_softcap_bf16",
            vec![],
            None,
            vec![x.id],
            Some((Shape(vec![Dim::Requests, Dim::Const(vocab)]), DType::BF16)),
        )
        .expect("the softcap produces its value")
    }

    /// `kernels::attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16`:
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
            "attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16",
            vec![q_norm.name.clone(), k_norm.name.clone()],
            kv_state(kv),
            vec![packed.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16)),
        )
        .expect("the fused post produces q")
    }

    /// `kernels::rope::qk_rmsnorm_rope_bf16_rounded`: the per-head q/k
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
                "rope::qk_rmsnorm_rope_bf16_rounded",
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

    /// `kernels::layout::transpose_bf16_nld_to_lnd`: relay the PLE table
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
            "layout::transpose_bf16_nld_to_lnd",
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

    /// `kernels::moe::topk_softmax_bf16`: the router's top-k + softmax +
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
                "moe::topk_softmax_bf16",
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

    /// `kernels::moe::moe_gate_up_decode_gemv_bf16` /
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
            "moe::moe_gate_up_decode_gemv_bf16",
            x,
            w,
            experts,
            top_k,
        )
    }

    pub fn moe_down_gemv(x: &Val, w: &MatW, experts: &Val, top_k: u32) -> Val {
        moe_routed_gemv("moe::moe_down_decode_gemv_bf16", x, w, experts, top_k)
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

    /// `kernels::moe::flashinfer_cutlass_moe_bf16`: the whole routed block —
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
            "moe::flashinfer_cutlass_moe_bf16",
            vec![gate_up.name.clone(), down.name.clone()],
            None,
            vec![x.id, experts.id, weights.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
        )
        .expect("the fused MoE produces its value")
    }

    /// `kernels::norm::residual_add_bf16`: the explicit stream add, for
    /// the legs whose producer wrote to scratch instead of folding.
    pub fn residual_add(x: &Val, residual: &Val, hidden: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "norm::residual_add_bf16",
            vec![],
            None,
            vec![x.id, residual.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
        )
        .expect("the residual add produces its value")
    }

    /// Row RMSNorm, STATING which fold it runs.
    ///
    /// Gemma folds `(1 + w)` instead of `w` — different arithmetic, so a
    /// different kernel — and the fold is a property of the WEIGHT,
    /// which is why [`NormW`] carries it and why the caller passes no
    /// variant.
    ///
    /// The semantic [`super::rmsnorm`] carries the variant as a param
    /// instead, and four drivers read it and pick; three had hard-coded
    /// their own deployment's answer. A `*.cuda.*` text calls this one
    /// and nothing downstream chooses.
    ///
    /// PER-HEAD is not here yet: its row count is the operand's width
    /// over the head dim, and `head_dim` has nowhere to ride on a
    /// `Launch`. It moves when it states a kernel that takes it.
    pub fn rmsnorm(x: &Val, w: &NormW) -> Val {
        let id = x.t.with(w.layer, |b| match w.per_head {
            // PER-HEAD falls through to the semantic kind, and the call
            // site does not have to know which it got: the handle
            // decides, and the same site is per-head on qwen3 and
            // row-wise on olmo2.
            Some(head_dim) => b.rmsnorm_per_head(x.id, &w.name, head_dim, w.variant),
            None => {
                let symbol = match w.variant {
                    NormVariant::Gemma => "norm::rmsnorm_gemma_bf16",
                    _ => "norm::rmsnorm_bf16",
                };
                let shape = b.value_shape(x.id);
                b.launch(
                    symbol,
                    vec![w.name.clone()],
                    None,
                    vec![x.id],
                    vec![(shape, DType::BF16)],
                )[0]
            }
        });
        Val {
            t: x.t.clone(),
            id,
            layer: w.layer,
        }
    }

    // ── TENSOR PARALLELISM ─────────────────────────────────────────
    //
    // A collective is a STATEMENT. It is real device work with operands
    // and a result, and the only reason it has not been one is that the
    // hand-written passes reached for `tp->` directly.
    //
    // Sharding itself needs no vocabulary: a rank's trace states ITS
    // widths, and the text divides by `tp_size` from the facts the way
    // it already divides by anything else. What needs vocabulary is the
    // point where the shards are recombined, because that is a launch.

    /// `comm::all_reduce_bf16`: the NVLink P2P sum, out of place.
    ///
    /// ONE ARM OF A CHOICE THE DRIVER USED TO MAKE. `NcclComm::
    /// all_reduce_bf16` asks `can_handle(bytes)` and routes to this
    /// kernel below the threshold, `ncclAllReduce` above it — an `if`
    /// inside a driver method picking between two implementations,
    /// which is the shape this whole arc removes.
    ///
    /// So a text states the pair as a GUARD, the way qwen3.5's
    /// recurrence states its three spellings and the fused landing
    /// states its two. The predicate is the message size, which is
    /// `TokensLE` — the threshold is bytes, and a row of `hidden` bf16
    /// elements is a fixed number of them, so the token count IS the
    /// test once the deployment's hidden size is known.
    ///
    /// What does NOT reduce to the predicate is buffer REGISTRATION:
    /// the P2P kernel reads only buffers handed to `register_buffer`,
    /// which is a placement fact of the deployment rather than a
    /// property of the fire. It belongs on the facts beside
    /// `gate_up_fused` — a load-time answer that erases into the trace.
    pub fn all_reduce_p2p(x: &Val, hidden: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "comm::all_reduce_bf16",
            vec![],
            None,
            vec![x.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
        )
        .expect("the collective produces its value")
    }

    /// `dist::all_reduce_bf16`: sum this value across ranks, in place.
    ///
    /// The in-place form, which is what a post-norm landing takes: the
    /// partial is summed where it lies and the statement's result is the
    /// same bytes.
    ///
    /// THE OTHER ARM of [`all_reduce_p2p`]'s choice, and the one that is
    /// not a kernel: NCCL is the comm plane, and `custom_all_reduce.hpp`
    /// says in as many words where that knowledge belongs — with the
    /// caller, not with a compute kernel. So this symbol has no
    /// `kernel!` operand signature and cannot get one without moving
    /// NCCL down a layer, which is a decision that was already made in
    /// the other direction once.
    ///
    /// It is still a STATEMENT. What a symbol needs to be stated is a
    /// name the declaration can choose and an arm that binds it; a
    /// generated ABI entry point is a separate benefit that this one
    /// does not get.
    pub fn all_reduce(x: &Val, hidden: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "dist::all_reduce_bf16",
            vec![],
            None,
            vec![x.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
        )
        .expect("the collective produces its value")
    }

    /// `dist::all_reduce_bf16_out`: sum this value across ranks into a
    /// SEPARATE destination.
    ///
    /// The two-step landing's first half. It reads as the same
    /// collective and it is; what differs is that the result is not the
    /// operand's bytes, because the residual add downstream needs both.
    pub fn all_reduce_out(x: &Val, hidden: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "dist::all_reduce_bf16_out",
            vec![],
            None,
            vec![x.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
        )
        .expect("the collective produces its value")
    }

    /// `dist::all_gather_bf16`: concatenate this value's shards along
    /// its row width. The result is `parts` times as wide.
    pub fn all_gather(x: &Val, parts: u32, width: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "dist::all_gather_bf16",
            vec![],
            None,
            vec![x.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(width * parts)]),
                DType::BF16,
            )),
        )
        .expect("the collective produces its value")
    }

    /// `comm::all_reduce_residual_rmsnorm_bf16`: the FUSED landing —
    /// sum the shards, add the residual, and norm, in one launch.
    ///
    /// TWO results, because the kernel has two effects: the residual
    /// stream is updated IN PLACE (operand 1, which the `kernel!` row
    /// aliases output 0 over) and the normed activation is written
    /// fresh. Returned in that order.
    ///
    /// WHETHER TO FUSE IS A GUARD, not a driver test. The hand-written
    /// pass asks `can_fuse_residual_rmsnorm(tokens, hidden, stream)` at
    /// fire time; `hidden` and the buffer registration are load-time
    /// facts that resolve into the trace, and what is left —
    /// `tokens` — is exactly `GuardPred::TokensLE`. So a text states
    /// the fused arm under that predicate and the two-step form as the
    /// else, the same shape qwen3.5's recurrence uses for its three
    /// spellings.
    pub fn all_reduce_residual_rmsnorm(
        x: &Val,
        residual: &Val,
        weight: &NormW,
        hidden: u32,
    ) -> (Val, Val) {
        let shape = (Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16);
        let outs = x.t.with(x.layer, |b| {
            b.launch(
                "comm::all_reduce_residual_rmsnorm_bf16",
                vec![weight.name.clone()],
                None,
                vec![x.id, residual.id],
                vec![shape.clone(), shape],
            )
        });
        let mk = |id| Val {
            t: x.t.clone(),
            id,
            layer: x.layer,
        };
        (mk(outs[0]), mk(outs[1]))
    }

    /// `kernels::mlp::sigmoid_dot_scalar_gate_add_bf16`: the shared
    /// expert's landing with its gate logit folded in — one launch that
    /// dots `norm_x` with the `[1, H]` gate row, sigmoids the scalar, and
    /// accumulates `shared` into the stream.
    ///
    /// The general form is a `[Tokens, 1]` GEMM followed by
    /// `kernels::mlp::sigmoid_scalar_gate_add_bf16`; this fused form runs when
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
            "mlp::sigmoid_dot_scalar_gate_add_bf16",
            vec![gate.name.clone()],
            None,
            vec![x.id, base.id, shared.id],
            Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
        )
        .expect("the shared-expert landing produces its value")
    }

    /// `kernels::mlp::chunked_swiglu_bf16` over the routed rows — the
    /// same kernel [`swiglu`]'s packed arm names, launched with `N * k`
    /// rows instead of `N`. A separate statement because the SHAPE
    /// differs, not the kernel: the routed value keeps its expert dim.
    pub fn swiglu_routed(x: &Val, top_k: u32, intermediate: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "mlp::chunked_swiglu_bf16",
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

    /// `kernels::moe::token_batched_weighted_sum_bf16`, or the
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
    /// The per-expert `kernels::moe::scatter_add_weighted_bf16` loop is the
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
                "moe::token_batched_weighted_sum_add_bf16"
            } else {
                "moe::token_batched_weighted_sum_bf16"
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
                "mlp::chunked_swiglu_bf16"
            } else {
                "mlp::swiglu_bf16"
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

    /// `kernels::mlp::chunked_swiglu_bf16` over the ALIGNED leg's
    /// block-major staging: [`Self::swiglu`]'s shape, plus the
    /// destination the pointer build named.
    ///
    /// Its own statement because the destination is not this call's to
    /// choose. The aligned staging's addresses are baked into the
    /// pointer arrays, so the activation has to land on the buffer
    /// `build_moe_ptrs_aligned` declared — an operand, written in place,
    /// exactly as the two grouped GEMMs around it do. Stating it any
    /// other way puts the activation somewhere the down projection's
    /// pointers do not point.
    pub fn swiglu_aligned(x: &Val, stage: &Val, aligned: Dim, intermediate: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "mlp::chunked_swiglu_bf16",
            vec![],
            None,
            vec![x.id, stage.id],
            Some((
                Shape(vec![aligned, Dim::Const(intermediate)]),
                DType::BF16,
            )),
        )
        .expect("the aligned activation produces its value")
    }

    /// `kernels::mlp::swiglu_bf16` in its PAIR form: two operands, the
    /// gate and the up projection, into one activation.
    ///
    /// The spelling an UNFUSED gate_up binding actually fires, and the
    /// one the declaration could not carry until now. [`Self::swiglu`]
    /// above states one packed operand either way and lets `packed` pick
    /// the kernel — which left the pair form reading two workspace
    /// buffers (`ws.gate`, `ws.up`) that no traced value described, so
    /// the executor had to keep that convention and cross-check it
    /// against the fact on every launch.
    ///
    /// With the projections stated as two matmuls the two operands ARE
    /// values, and the whole `gate_up_used_fused` correspondence between
    /// the Matmul arm and this one disappears: each statement says what
    /// it reads.
    pub fn swiglu_pair(gate: &Val, up: &Val, intermediate: u32) -> Val {
        record(
            &gate.t,
            gate.layer,
            "mlp::swiglu_bf16",
            vec![],
            None,
            vec![gate.id, up.id],
            Some((
                Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
                DType::BF16,
            )),
        )
        .expect("the activation produces its value")
    }

    /// `kernels::rope::qk_rmsnorm_rope_bf16`: the fused per-head q/k
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
                "rope::qk_rmsnorm_rope_bf16",
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

    /// `kernels::attn::attention_xqa_decode_bf16_prepared` (whose contract
    /// includes the fire-wide XQA prepare — and which is therefore
    /// declared `whole`; see [`crate::kernels`]).
    pub fn attention_xqa_decode(q: &Val, kv: &Kv, window_left: i32) -> Option<Val> {
        attn_at(q, kv, "attn::attention_xqa_decode_bf16_prepared", window_left)
    }

    /// `kernels::attn::dispatch_attention_flashinfer_decode` against the decode
    /// plan its contract obligates.
    pub fn attention_flashinfer_decode(q: &Val, kv: &Kv, window_left: i32) -> Option<Val> {
        attn_at(q, kv, "attn::dispatch_attention_flashinfer_decode", window_left)
    }

    /// `kernels::attn::dispatch_attention_flashinfer_prefill_bf16` — the dispatch
    /// ALONE.
    ///
    /// Three wrappers used to differ here only by whether they also
    /// launched the dequant staging: llama_like's cache may be
    /// quantized, so its prefill-shaped arms dequant the layer first,
    /// while qwen3_5's full-attention path gates on a native-bf16 cache
    /// and launches only the dispatch. That is not a property of this
    /// kernel — it is a second STATEMENT the text either makes or does
    /// not, so the text makes it ([`dequant_only`] beside this call).
    pub fn attention_flashinfer_prefill(q: &Val, kv: &Kv, window_left: i32) -> Option<Val> {
        attn_at(q, kv, "attn::dispatch_attention_flashinfer_prefill_bf16", window_left)
    }

    /// `kernels::attn::attention_flashinfer_prefill` — the PLAN-FREE
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
    pub fn attention_flashinfer_prefill_planless(q: &Val, kv: &Kv, window_left: i32) -> Option<Val> {
        attn_at(q, kv, "attn::attention_flashinfer_prefill", window_left)
    }

    /// `kernels::attn::dispatch_attention_flashinfer_decode` asked for its LSE.
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
                "attn::dispatch_attention_flashinfer_decode",
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

    /// `kernels::rope::rope_yarn_original_bf16`: the YaRN-paper rope —
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
                "rope::rope_yarn_original_bf16",
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

    /// `kernels::rope::rope_bf16`: the full rotation, named.
    ///
    /// The semantic [`super::rope`] carries a `RopeKind` and a rotary
    /// width, and the driver's arm asked whether the width was zero to
    /// decide between two launchers. That is a KERNEL CHOICE — a full
    /// rotation and a partial one are different arithmetic — so it
    /// belongs in the statement, and the pair below is that statement.
    pub fn rope(q: &Val, k: &Val) -> (Val, Val) {
        rope_launch(q, k, "rope::rope_bf16", vec![0])
    }

    /// `kernels::rope::rope_partial_bf16`: only the first `rotary_dim`
    /// channels of each head rotate.
    ///
    /// `rotary_dim` rides the statement's PARAMS
    /// ([`crate::trace::OpKind::Launch`]), not the executor's config.
    ///
    /// The THETA does not, yet, and the reason is worth writing down
    /// rather than leaving as an omission: gemma-4 alternates it per
    /// layer between its local and global attention, so a driver
    /// reading the single `cfg.rope_theta` reads the wrong one for half
    /// that model's layers — the fact belongs here. What blocks it is
    /// the emission fixtures, which would have to state each target's
    /// real theta, and inventing those numbers is worse than a driver
    /// reading a config value that is uniform for every family but one.
    /// It is a property of this rotation and no operand shape spells it
    /// — the operands are full-width q and k either way — which is
    /// exactly what that channel is for.
    pub fn rope_partial(q: &Val, k: &Val, rotary_dim: u32) -> (Val, Val) {
        assert!(
            rotary_dim > 0,
            "a partial rotation with no channels is the full one; state \
             `cuda::rope`"
        );
        rope_launch(q, k, "rope::rope_partial_bf16", vec![rotary_dim])
    }

    /// The shape both rotations share: two operands in, two results out,
    /// each landing where its operand lies (the `kernel!` rows alias
    /// both pairs).
    fn rope_launch(q: &Val, k: &Val, symbol: &str, params: Vec<u32>) -> (Val, Val) {
        let (q_sh, k_sh) = {
            let b = q.t.inner.borrow();
            (b.value_shape(q.id), b.value_shape(k.id))
        };
        let ids = q.t.with(q.layer, |b| {
            b.launch_with_params(
                symbol,
                vec![],
                None,
                params,
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

    /// `kernels::gemm::act_x_wt_bias_bf16`: a projection whose BIAS RIDES IN
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
            "gemm::act_x_wt_bias_bf16",
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

    /// `kernels::attn::attention_flashinfer_prefill` asked for its LSE —
    /// the prefill twin of [`attention_flashinfer_decode_lse`], and the
    /// same argument makes the difference.
    pub fn attention_flashinfer_prefill_lse(q: &Val, kv: &Kv, q_heads: u32) -> (Val, Val) {
        let shape = q.t.inner.borrow().value_shape(q.id);
        let ids = q.t.with(Some(kv.l), |b| {
            b.launch(
                "attn::attention_flashinfer_prefill",
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

    /// `kernels::attn::attention_sink_rescale_bf16`: `o *= sigmoid(lse
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
            "attn::attention_sink_rescale_bf16",
            vec![sinks.name.clone()],
            None,
            vec![o.id, lse.id],
            Some((shape, DType::BF16)),
        )
        .expect("the sink rescale produces its value")
    }

    /// `kernels::quant::bf16_to_fp16`: the activation cast the MXFP4
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
            "quant::bf16_to_fp16",
            vec![],
            None,
            vec![x.id],
            Some((shape, DType::F16)),
        )
        .expect("the cast produces its value")
    }

    /// `kernels::quant::mxfp4_moe_gate_up_decode_bf16`: BOTH routed
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
                "quant::mxfp4_moe_gate_up_decode_bf16",
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

    /// `kernels::quant::mxfp4_moe_down_decode_bf16`: the routed down
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
            "quant::mxfp4_moe_down_decode_bf16",
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

    /// `kernels::mlp::gpt_oss_glu_bf16`: SwiGLU with gpt-oss's CLAMP.
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
    /// `limit` is the deployment's `swiglu_limit`, and it rides the
    /// param channel for the reason [`Self::scalar_mul`]'s scale does:
    /// it is a load-time number the host has, and the executor was
    /// reaching into a config struct for it.
    pub fn gpt_oss_glu(
        gate: &Val,
        up: &Val,
        top_k: u32,
        intermediate: u32,
        limit: f32,
    ) -> Val {
        record_with_params(
            &gate.t,
            gate.layer,
            "mlp::gpt_oss_glu_bf16",
            vec![],
            None,
            vec![limit.to_bits()],
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

    /// `kernels::attn::attention_naive_paged` — the fallback prefill for a
    /// head dim flashinfer's TC prefill template rejects.
    ///
    /// gemma-4's FULL-attention layers run at head_dim 512, and
    /// flashinfer 0.6.x refuses to instantiate a prefill at
    /// `NUM_MMA_D_QK=32`. So the deployment states a naive paged kernel
    /// on exactly those layers — a per-layer HEAD DIM fact, erased at
    /// trace time, not a runtime fallback the executor discovers.
    pub fn attention_naive_paged(q: &Val, kv: &Kv, window_left: i32) -> Option<Val> {
        attn_at(q, kv, "attn::attention_naive_paged", window_left)
    }

    /// `kernels::attn::write_kv_explicit_bf16`: the explicit-descriptor
    /// KV write (graph-replay steering; N cells, one per query token).
    /// Stated inside the `HasWriteDesc` guard's then-region.
    pub fn write_kv_explicit(k: &Val, v: &Val, kv: &Kv) {
        record(
            &kv.t,
            Some(kv.l),
            "attn::write_kv_explicit_bf16",
            vec![],
            kv_state(kv),
            vec![k.id, v.id],
            None,
        );
    }

    /// `kernels::attn::write_kv_to_pages`: the page-derived append
    /// (position re-derived from the page table). The `HasWriteDesc`
    /// guard's else-region.
    pub fn write_kv_to_pages(k: &Val, v: &Val, kv: &Kv) {
        record(
            &kv.t,
            Some(kv.l),
            "attn::write_kv_to_pages",
            vec![],
            kv_state(kv),
            vec![k.id, v.id],
            None,
        );
    }

    /// `kernels::ssm::causal_conv1d_update_batched_bf16`: the
    /// slot-indirected decode conv update (+ fused SiLU) against the
    /// layer's per-request conv slab. Shape-preserving, like the
    /// semantic [`causal_conv1d`] it lowers.
    pub fn gdn_conv_update_batched(x: &Val, w: &ConvW, rs: &Rs) -> Val {
        gdn_conv(x, w, rs, "ssm::causal_conv1d_update_batched_bf16")
    }

    /// `kernels::ssm::causal_conv1d_prefill_batched_bf16`: the batched
    /// prefill conv walk (each request walking its qo_indptr window and
    /// persisting the trailing K-window into the slab).
    pub fn gdn_conv_prefill_batched(x: &Val, w: &ConvW, rs: &Rs) -> Val {
        gdn_conv(x, w, rs, "ssm::causal_conv1d_prefill_batched_bf16")
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

    /// `kernels::ssm::recurrent_gated_delta_step_batched[_gqa][_state_bf16]`:
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
            (true, true) => "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16",
            (true, false) => "ssm::recurrent_gated_delta_step_batched_gqa",
            (false, true) => "ssm::recurrent_gated_delta_step_batched_state_bf16",
            (false, false) => "ssm::recurrent_gated_delta_step_batched",
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

    /// `kernels::ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa[_state_bf16]`:
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
        state_bf16: bool,
    ) {
        // ONE arm per state dtype, and nothing else to choose. The GQA
        // kernel's `repeat` is 1 when `K_h == V_h`, its `qk_h` is `h`, and
        // its index reduces to the non-GQA one exactly — so that pair was
        // a second copy of the same arithmetic and upstream deleted it.
        // Keeping a statement for a symbol the driver no longer exports
        // would be a declaration that cannot load.
        let kernel = if state_bf16 {
            "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16"
        } else {
            "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa"
        };
        gdn_prefill(q, k, v, g, beta, rs, kernel);
    }

    /// `kernels::ssm::chunk_gated_delta_prefill_batched_cached[_state_bf16]`:
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
            "ssm::chunk_gated_delta_prefill_batched_cached_state_bf16"
        } else {
            "ssm::chunk_gated_delta_prefill_batched_cached"
        };
        gdn_prefill(q, k, v, g, beta, rs, kernel);
    }

    /// `kernels::ssm::chunk_gated_delta_prefill_batched[_state_bf16]`:
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
            "ssm::chunk_gated_delta_prefill_batched_state_bf16"
        } else {
            "ssm::chunk_gated_delta_prefill_batched"
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

    /// `kernels::ssm::repeat_interleave_heads_fp32`: materialize the
    /// K_h → V_h head repeat of a compact per-head f32 value. Stated
    /// only inside the cached arm, because only that kernel family
    /// consumes the repeated layout (the decode-GQA step, warp-tiled and
    /// FLA kernels all index the compact layout directly).
    ///
    /// It DECLARES its result, which it did not use to. Output-less, the
    /// stance was "where that buffer lives is the driver's binding, not
    /// dataflow" — and the cost of that stance was paid twice over: the
    /// driver kept a `repeat_next_is_k` toggle to decide which of two
    /// workspace fields a launch meant, the emitter kept the SAME toggle
    /// to decide it statically, and the recurrence below could not name
    /// its own q/k operands because the value between them had no id.
    /// A repeat is dataflow; it took a value to say so.
    ///
    /// `[Tokens, value_heads, key_dim]` f32 — the compact `[Tokens,
    /// key_heads, key_dim]` operand with each key head repeated to fill
    /// the value-head count.
    pub fn repeat_interleave_heads(x: &Val, value_heads: u32, key_dim: u32) -> Val {
        record(
            &x.t,
            x.layer,
            "ssm::repeat_interleave_heads_fp32",
            vec![],
            None,
            vec![x.id],
            Some((
                Shape(vec![
                    Dim::Tokens,
                    Dim::Const(value_heads),
                    Dim::Const(key_dim),
                ]),
                DType::F32,
            )),
        )
        .expect("the head repeat produces its value")
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

    /// `kernels::attn::dispatch_attention_flashinfer_decode_capture`: the
    /// score-capturing decode dispatch (the OnAttn sideband's producer;
    /// its contract includes the capture publish against the possibly
    /// page-mask-compacted CSR). Region launch of the WantsAttnScore
    /// guard — output-less; the guard owns the attention output.
    pub fn attention_flashinfer_decode_capture(q: &Val, kv: &Kv, window_left: i32) -> Option<Val> {
        attn_at(q, kv, "attn::dispatch_attention_flashinfer_decode_capture", window_left)
    }

    /// `kernels::attn::dispatch_attention_flashinfer_prefill_capture_bf16` — the
    /// prefill counterpart, same guard-region contract.
    pub fn attention_flashinfer_prefill_capture(q: &Val, kv: &Kv, window_left: i32) -> Option<Val> {
        attn_at(q, kv, "attn::dispatch_attention_flashinfer_prefill_capture_bf16", window_left)
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
            "attn::qkv_decode_qk_norm_rope_write_kv_bf16",
            vec![q_norm.name.clone(), k_norm.name.clone()],
            kv_state(kv),
            inputs,
            None,
        );
    }

    /// `kernels::attn::dispatch_attention_flashinfer_prefill_custom`: the
    /// custom-mask prefill dispatch — a genuinely distinct launcher, so
    /// no pseudo-symbol is needed. The mask data (BRLE bytes + indptr)
    /// crosses as runtime args of the stated kernel, commit_lens's peer.
    /// Since A1 (the class-collapse amendment) it is stated inside the
    /// `HasCustomMask` guard arm of the Decode/Prefill traces.
    pub fn attention_flashinfer_prefill_custom(q: &Val, kv: &Kv, window_left: i32) -> Option<Val> {
        attn_at(q, kv, "attn::dispatch_attention_flashinfer_prefill_custom", window_left)
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

    /// `kernels::attn::dequant_kv_cache_layer_to_bf16_active`: the
    /// staging launch a quantized cache needs before a prefill-shaped
    /// dispatch. Its OWN statement — see
    /// [`attention_flashinfer_prefill`] for why it is not folded into
    /// any attention wrapper.
    pub fn dequant_only(kv: &Kv) {
        record(
            &kv.t,
            Some(kv.l),
            "attn::dequant_kv_cache_layer_to_bf16_active",
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
    /// Every FlashInfer/XQA dispatch's shape: one query in, the layer's
    /// cache as state, and the attention output — or none, inside a
    /// value-producing guard region, where the guard owns the value.
    ///
    /// `window_left` is the SLIDING WINDOW this layer attends over,
    /// `-1` for none. It is a load-time fact (a config's
    /// `sliding_window`, or its per-layer list where the architecture
    /// alternates), and it used to be derived inside every executor:
    /// eleven copies of the same three lines across four families,
    /// reaching into `fwd_cfg.per_layer_window_left` — a per-layer array
    /// no statement mentioned.
    ///
    /// It rides the statement's PARAMS because no operand shape gives
    /// it. What is NOT closed by this is the per-FIRE override
    /// (`runtime_window_left`), which is a runtime input and wants a
    /// guard predicate; `DeclineReason::SlidingWindow` still names it.
    fn attn_at(q: &Val, kv: &Kv, kernel: &str, window_left: i32) -> Option<Val> {
        let out = q.t.inner.borrow().inside_value_region();
        let shape = (!out).then(|| q.t.inner.borrow().value_shape(q.id));
        record_with_params(
            &q.t,
            Some(kv.l),
            kernel,
            vec![],
            kv_state(kv),
            vec![window_left as u32],
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
            params: vec![],
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
                // This fixture exists for the adapter POSITION rule, which
                // reads `layer`/`op` and never the exposed set.
                values: vec![],
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

