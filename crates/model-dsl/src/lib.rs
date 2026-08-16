//! THE DECLARATION SURFACE — what a forward pass is WRITTEN IN
//! (north-star-dsl.md, "the v2 surface").
//!
//! ```text
//! declaration  ──trace──▶  forward plan  ──lower──▶  driver executes
//!  THIS CRATE              `model-ir`              `model-compiler`
//! ```
//!
//! ## The declarations are not here
//!
//! They are in `crates/model`, one per generation, beside that model's chat
//! template and its load contract — `.wiki/tart-todo.md` item 1, and the shape
//! `.wiki/tart/dsl.md` ③ always described ("the model file is
//! `families/<family>/<backend>.rs`"). What is here is the vocabulary they are
//! written in, and it names no family.
//!
//! This crate is the half of the old `model-compiler` that only `crates/model`
//! ever reached: a driver lowers a traced form and never writes one, so every
//! `dsl::` path in a driver was a doc link and not a call. Holding it with the
//! lowering meant `driver-metal`, `driver-vulkan` and `driver-wgpu` each
//! compiled [`cuda`]'s 4,469 lines — an authoring surface for a backend they
//! are not — to reach code that never mentions them. Now nothing but
//! `crates/model` depends on this crate at all.
//!
//! ## The surface
//!
//! plan.md's sketch, made real: values carry the tape, so ops are free
//! functions and operators rather than builder methods; weights are typed
//! handles from a per-layer namespace, so no declaration spells a string
//! or a width; per-layer state is an object (`Kv::append`); and a LOWERED
//! declaration calls **raw kernel signatures** ([`cuda`]) — functions
//! named for the driver's launcher symbols, whose parameters are the
//! launcher's semantic operands — recording [`OpKind::Launch`](model_ir::trace::OpKind::Launch) ops.
//!
//! The recording substrate is [`TraceBuilder`], and it is `model-ir`'s: this
//! crate is surface, not IR. The one behavioral subtlety lives in `+=`:
//!
//! * `y += matmul(&a, &w.o_proj)` — if the matmul is the op just
//!   recorded and nothing else consumed it, the tape REWRITES it to the
//!   `beta_one` accumulate form (`matmul_add`), the hand-written passes'
//!   cuBLAS residual fold. Output id unchanged, so the fold is invisible
//!   to dataflow.
//! * `y += anything_else` — a [`OpKind::ResidualAdd`](model_ir::trace::OpKind::ResidualAdd) launch, exactly the
//!   post-norm landing the hand-written pass makes explicit.
//!
//! Op `layer` tags derive from what an op touches — its weight handle,
//! its state handle, or its first input — rather than from a bracketing
//! closure; the semantic goldens pin that this derivation reproduces the
//! bracketed tagging byte for byte.
//!
//! # On arity
//!
//! Seven functions here take eight arguments, and the paragraph above is
//! why: a declaration's parameters are the launcher's semantic operands,
//! so the arity is the KERNEL's and not this file's to choose. `rope`
//! takes a theta, a scale, a head dim, a rotary dim and a table flag
//! because rope does. Bundling them behind a struct would put a layer
//! between the surface and the symbol it is named for, which is the one
//! thing this surface exists not to do.
//!
//! The allow is module-wide rather than seven copies because the reason
//! is the module's, not any function's. A function here that is wide for
//! some OTHER reason is a real finding, and this hides it — the guard
//! against that is that a declaration which does not mirror a launcher
//! does not belong in this file at all.
#![allow(clippy::too_many_arguments)]

use std::cell::RefCell;
use std::rc::Rc;

use model_ir::facts::QkNorm;
use model_ir::trace::{
    DType, Dim, FireClass, ForwardPlan, NormVariant, RopeKind, Shape, StateRef, StateStore,
    TraceBuilder,
};

/// The tracer's fingerprint: an FNV-1a content hash of this crate's `src/`
/// AND `model-ir`'s, computed by `build.rs`.
///
/// The traced form is a pure function of (declaration code, facts), so this
/// number plus the facts identifies a plan exactly. `model`'s FFI stamps it
/// into every plan header so a consumer can key a cache or a golden on
/// `PieForwardPlan::compiler_version` and have it invalidate itself when the
/// tracer changes.
///
/// It covers BOTH source trees because the tracer is both of them since the
/// split — this crate decides what a statement records, `model-ir` decides
/// what recording one means — and a fingerprint over half a tracer is worse
/// than none: it would report "unchanged" across a change it cannot see.
///
/// It is a function rather than a constant because `env!` only reads the
/// environment of the crate being compiled, and the crate that needs the
/// number is not this one.
pub fn compiler_version() -> u64 {
    env!("PIE_FORWARD_COMPILER_HASH")
        .parse::<u64>()
        .unwrap_or(0)
}

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
    pub(crate) id: model_ir::trace::ValueId,
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
///
/// Two layouts, mirroring `model_loader::types::QuantGranularity` exactly —
/// which is the only vocabulary a checkpoint can state a scale in, whether it
/// SHIPS the scales (`contract::Scales::granularity`) or the loader encodes
/// them (`plan/build.rs::ScaleLayout::for_encode`). A third variant,
/// `PerTensor`, was here and named `gemm::act_x_wt_tensor_scaled`; it had no
/// constructor anywhere in the workspace outside this file, and no checkpoint
/// format in the tree could have grown one without the loader growing a
/// granularity first. It is deleted rather than left as a route nothing can
/// take. The entry point it named was `gemm::act_x_wt_tensor_scaled` in
/// `kernels-cuda/csrc/src/gemm/gemm.hpp`, and THAT FILE IS DELETED TOO — the
/// whole 2,216-line host program is `driver-cuda/src/fire/gemm.rs` now, and
/// the PerTensor arm went with it because nothing constructed it. So
/// re-stating this variant is no longer "a variant, an arm and a row": it is
/// a variant, an arm, a row, AND a Rust body in `fire::gemm`, plus a loader
/// granularity for a checkpoint to state it in. Read `fire::gemm`'s FP8
/// notes first — the `returned == 0` heuristic latch that reached the FP8
/// PerTensor path is recorded there, and is why this was never one more
/// enum arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ScaleLayout {
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
    /// [`model_ir::trace::OpKind::Matmul`] — that kind fans to exactly one
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
    /// The attention landing's bias, for a family that publishes one.
    ///
    /// gpt-oss does, and its width is the model's, not the projection's --
    /// `o_proj` maps heads back to `hidden`, so the bias is one number per
    /// hidden channel. Separate from [`Layer::o_proj`] rather than folded
    /// into a `gemm_bias` because the folded form is a different kernel with
    /// a different accumulation order; see that function's doc.
    pub o_bias: MatW,
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
    /// The router's bias -- one number per EXPERT, added to the logits
    /// before the top-k.
    ///
    /// The most consequential bias in a mixture and the least forgiving. A
    /// projection bias shifts an activation the next norm largely absorbs;
    /// this one shifts a ranking, so a text that drops it does not compute
    /// a slightly different answer, it routes to different experts.
    pub router_bias: MatW,
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
    /// The norm on the ATTENTION's output, before the residual add.
    ///
    /// `NormPlacement::Sandwich` only — gemma's `post_attention_layernorm`
    /// under a placement where `mlp_norm` is already spoken for by
    /// `pre_feedforward_layernorm`. A `Pre` or `Post` text never names it.
    pub post_attn_norm: NormW,
    /// The norm on the FFN's output, before the residual add. gemma's
    /// `post_feedforward_layernorm`; see [`Layer::post_attn_norm`].
    pub post_mlp_norm: NormW,
    /// The MIXTURE layer's per-LEG norms, which only gemma-4's MoE rows name.
    ///
    /// A gemma-4 mixture layer runs a dense FFN and a routed one SIDE BY
    /// SIDE off the same post-attention stream, norms each leg's output on
    /// its own, adds the two, and only then applies [`Layer::post_mlp_norm`]
    /// to the sum. So the layer ships seven norms where a dense gemma ships
    /// four: `_1` is the dense leg's output norm, `mlp_norm_2` the routed
    /// leg's INPUT norm, `post_mlp_norm_2` its output norm.
    ///
    /// `mlp_norm_2` and [`Layer::mlp_norm`] are the same width and different
    /// tensors. Reusing one for both is a substitution no shape check can
    /// see, which is why the routed leg names its own.
    pub post_mlp_norm_1: NormW,
    pub mlp_norm_2: NormW,
    pub post_mlp_norm_2: NormW,
    /// The router's own RMS-norm weight, `[hidden]`.
    ///
    /// gemma-4 norms the router's input at a scale that is neither leg's,
    /// then projects. Distinct from the router's quantisation scales, which
    /// ride [`Layer::router`]'s own affine point.
    pub router_scale: NormW,
    /// The router's learned per-expert gain, `[n_experts]`.
    ///
    /// Multiplies the weights AFTER the top-k softmax, which is why
    /// `moe/route.metal` carries it as `router_topk_scaled`'s fifth buffer
    /// rather than folding it into the logits: applied before the softmax it
    /// would move the ranking.
    pub router_expert_scale: MatW,
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
    /// recording assigns each layer-tagged op's depth role — the
    /// flashinfer decode dispatch swaps to the depth prefix plan on
    /// union tail layers, everything else windows.
    ///
    /// `DepthRole` is named as a type here no longer: it was RETIRED into
    /// two questions the plan answers directly
    /// ([`ForwardPlan::depth_windowed`](model_ir::trace::ForwardPlan::depth_windowed),
    /// [`ForwardPlan::depth_prefix_plan`](model_ir::trace::ForwardPlan::depth_prefix_plan)),
    /// and the dangling link outlived it.
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
            o_bias: mat("o_bias", f.hidden),
            gate_up: mat("gate_up", 2 * f.intermediate),
            gate_proj: mat("gate_proj", f.intermediate),
            up_proj: mat("up_proj", f.intermediate),
            down: mat("down", f.hidden),
            router: mat("router", f.n_experts),
            router_bias: mat("router_bias", f.n_experts),
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
            post_attn_norm: row_norm("post_attn_norm"),
            post_mlp_norm: row_norm("post_mlp_norm"),
            post_mlp_norm_1: row_norm("post_mlp_norm_1"),
            mlp_norm_2: row_norm("mlp_norm_2"),
            post_mlp_norm_2: row_norm("post_mlp_norm_2"),
            router_scale: row_norm("router_scale"),
            router_expert_scale: mat("router_expert_scale", f.n_experts),
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
/// [`Backend::of_family`](model_ir::kernels::Backend::of_family) reads the
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
/// kernels, checked against what Metal states
/// ([`model_ir::kernels::stated_in`]).
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

/// The epilogue, with the TIED-EMBEDDING fact resolved here rather than
/// by each caller.
///
/// The fork is one line — `if tied { "embed" } else { "lm_head" }` — and
/// four families wrote it: llama_like, gemma-4, qwen3.5 and gpt-oss. A
/// FIFTH did not, and that is why this exists rather than being left as
/// four tidy duplicates.
///
/// gemma-2 carries `tied_embeddings: true` in its facts, correctly (the
/// checkpoint ships no `lm_head.weight`), and its text named `"lm_head"`
/// unconditionally. A fact declared and never read — so the trace asked
/// the binder for a tensor the checkpoint does not contain, and the
/// family sits in `NOT_YET_OPENABLE` where nothing could notice.
///
/// A one-line fork repeated per family is a fact each family can forget
/// to read. Taking the BOOLEAN instead of the resolved name means the
/// forgetting has nowhere to happen.
pub fn lm_head_tied(t: &Trace, x: &Val, tied: bool, vocab: u32) -> Val {
    lm_head_at(t, x, if tied { "embed" } else { "lm_head" }, vocab)
}


// ── The surface, by file ───────────────────────────────────────────────
//
// One module per KIND of statement, and `pub use` on the three neutral ones
// so the flat paths a declaration spells (`matmul(..)`, `guarded(t)`,
// `rows!(..)`) are unchanged -- a file boundary is not an API.
//
// `cuda` and `metal` stay NAMED, because a declaration names them: a lowered
// text calls `dsl::cuda::rope_partial_last`, and which backend it is written
// against is the one thing that surface is for.
pub mod cuda;
pub mod metal;
mod guard;
mod ops;
mod rows;

pub use guard::*;
pub use ops::*;
pub use rows::*;
// ── The seam surface (V2 rung ①) ───────────────────────────────────────

/// The seam vocabulary, which is [`model_ir`]'s.
///
/// It was declared here while `dsl`, `trace` and `lower` were one crate,
/// and it was the only module of this file that BOTH the tracer and the
/// lowering reached — `TraceBuilder::finish` validates against it and
/// `model_compiler::lower` reads [`seam::OUT`]. That made it IR wearing an
/// authoring module's path, so it moved. Re-exported rather than merely
/// used, because the statements below spell `seam::IN` / `seam::OUT` /
/// `seam::ATTN_OUT` and a declaration reading this surface should not have
/// to know which crate the word came from.
pub use model_ir::seam;

/// THE SEAM STATEMENT — one construct for all five extension points
/// (`.wiki/tart/dsl.md` ①, migration step 4).
///
/// Until now three of the five lowered to ops through two different
/// functions and the other two lowered to NOTHING: the traced form did
/// not record that a text has a prologue or an epilogue at all, which is
/// what put those two stages in a different world from the rest. Every
/// seam is stated the same way here, and every statement is recorded
/// ([`model_ir::trace::SeamStatement`]) whichever way it lowers.
///
/// The LOWERINGS are unchanged, which is what keeps the goldens'
/// op streams byte-identical:
///
/// * `attn.q` / `attn.out` — one [`OpKind::HookSite`](model_ir::trace::OpKind::HookSite);
/// * `attn.qv` — the `HasLora` guard with the correction arm and an
///   EMPTY else (a fire with no usable lanes launches nothing);
/// * `in` / `out` — no op. A boundary attachment causes no divergence,
///   so it enters no row signature; what it needed was a DECLARATION,
///   and that is what the statement list now carries.
///
/// [`OpKind::HookSite`](model_ir::trace::OpKind::HookSite): model_ir::trace::OpKind::HookSite
pub fn seam(t: &Trace, def: &seam::Def, sees: &[&Val], layer: Option<u32>) {
    // The values the statement NAMES. Carried onto the record so buffer
    // assignment can pin exactly these, rather than guessing from the
    // operands of whatever op the seam points at.
    let ids: Vec<model_ir::trace::ValueId> = sees.iter().map(|v| v.id).collect();
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
                model_ir::trace::HookStage::OnAttnProj
            } else {
                model_ir::trace::HookStage::OnAttn
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
                model_ir::trace::GuardPred::HasLora,
                || cuda::lora_qkv_correction(sees[0], sees[1], l),
                || {},
            );
            t.inner
                .borrow_mut()
                .push_seam(def.name, layer, Some(at), ids);
        }
        "in" | "out" => {
            t.inner.borrow_mut().push_seam(def.name, layer, None, ids);
        }
        other => unreachable!("no seam named {other}"),
    }
}
