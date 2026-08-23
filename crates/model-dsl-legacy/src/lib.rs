//! DSL surface for tracing model declarations into `model-ir` plans.
//!
//! Lowered declarations may state raw backend launch symbols; semantic
//! declarations record backend-neutral ops. Layer tags derive from touched
//! handles or first inputs.

//! Function arity mirrors launcher operands; do not hide them in structs.
#![allow(clippy::too_many_arguments)]

use std::cell::RefCell;
use std::rc::Rc;

use model_ir::facts::QkNorm;
use model_ir::trace::{
    DType, Dim, FireClass, ForwardPlan, NormVariant, RopeKind, Shape, StateRef, StateStore,
    TraceBuilder,
};

/// FNV-1a hash over this crate and `model-ir`; plan-cache key material.
pub fn compiler_version() -> u64 {
    env!("PIE_FORWARD_COMPILER_HASH")
        .parse::<u64>()
        .unwrap_or(0)
}

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

    /// The backend this trace's family names, or `None` for a description
    /// trace. What tier-1 resolution branches on.
    #[must_use]
    pub fn backend(&self) -> Option<model_ir::kernels::Backend> {
        model_ir::kernels::Backend::of_family(self.inner.borrow().family())
    }

    /// The symbol this trace's backend claims for a canon point, or the
    /// `canon::<claim>` spelling for a description trace.
    ///
    /// Panics — at TRACE time, which is load time, the same moment
    /// `check_plan` refuses — when the backend claims nothing: an unclaimed
    /// role is a missing `#[routine(canon = ..)]`, and stating a guessed
    /// symbol instead would move the failure to the fire.
    #[must_use]
    pub fn canon(&self, claim: &str) -> String {
        match self.backend() {
            None => format!("canon::{claim}"),
            Some(b) => model_ir::kernels::canon_symbol(b, claim)
                .unwrap_or_else(|| {
                    panic!(
                        "no {b:?} routine claims `canon = {claim}`; \
                         tier-1 resolution has nothing to state"
                    )
                })
                .to_string(),
        }
    }
}

#[derive(Clone)]
pub struct Val {
    t: Trace,
    pub(crate) id: model_ir::trace::ValueId,
    layer: Option<u32>,
}

/// A RAGGED OPERAND PAIRING: a row stream and the boundary CSR that windows
/// it, travelling as one value at the authoring surface.
///
/// Raggedness is not a shape. The lowering's `Tokens`/`Requests` machinery
/// is untouched, and a statement taking one of these still records the two
/// halves as two operands in the order it always did — the wire and the
/// routines see no change. What the pair closes is the seam between the
/// halves: a wrapper that took the data stream here and minted its CSR
/// there was free to take them inconsistently, and a routine that needed
/// the boundary count had it restated as a spliced `Const` beside the
/// operand that already carries it (`indptr.rows`).
#[derive(Clone)]
pub struct RaggedVal {
    /// The row stream the CSR windows — attention's q, the token stream.
    pub data: Val,
    /// The boundary CSR, `[Requests]` i32. The driver stages the `+1` row
    /// the CSR convention implies.
    pub indptr: Val,
}

/// Storage representation; non-dense forms state a launch symbol and extra weights.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WeightRepr {
    #[default]
    Bf16,
    Scaled {
        layout: ScaleLayout,
        group: u32,
        axis: u32,
        zero_point: bool,
    },
    Mxfp4Marlin,
}

/// Scale granularity as checkpoints/loaders state it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ScaleLayout {
    PerChannel,
    PerGroup,
}

/// Matmul weight handle: name, output width, layer tag, storage.
#[derive(Clone)]
pub struct MatW {
    pub name: String,
    pub width: u32,
    pub layer: Option<u32>,
    /// Dense default keeps old declarations unchanged.
    pub repr: WeightRepr,
}

impl MatW {
    pub fn dense(name: String, width: u32, layer: Option<u32>) -> MatW {
        MatW {
            name,
            width,
            layer,
            repr: WeightRepr::Bf16,
        }
    }

    pub fn with_repr(mut self, repr: WeightRepr) -> MatW {
        self.repr = repr;
        self
    }

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

/// Norm handle; `per_head` carries head dim for q/k norm conventions.
#[derive(Clone)]
pub struct NormW {
    pub name: String,
    pub variant: NormVariant,
    pub per_head: Option<u32>,
    pub layer: Option<u32>,
    /// The epsilon THIS NORM's statements carry, on the handle because a
    /// handle is how `model/src/{family}/forward` — which owns every value a
    /// statement needs — hands one to the DSL. It rides the params run as
    /// its bits; an unstated epsilon is a zero, and a zero epsilon divides
    /// an all-zero row by nothing.
    pub eps: f32,
}

#[derive(Clone)]
pub struct Kv {
    t: Trace,
    pub l: u32,
}

impl Kv {
    /// `pub`: external declarations may build their own layer namespace.
    pub fn at(t: &Trace, l: u32) -> Kv {
        Kv { t: t.clone(), l }
    }

    pub fn append(&self, k: &Val, v: &Val) {
        self.t.with(Some(self.l), |b| {
            b.launch(
                "attn::write_kv_to_pages",
                vec![],
                Some(StateRef {
                    store: StateStore::KvCache,
                    layer: self.l,
                }),
                vec![k.id, v.id],
                vec![],
            );
        });
    }

    /// The state this layer's KV statements address — what every hand
    /// wrapper's `kv_state` computed in private; a generated statement
    /// takes it as its `state` argument (design-no-ask §10).
    #[must_use]
    pub fn state(&self) -> StateRef {
        StateRef {
            store: StateStore::KvCache,
            layer: self.l,
        }
    }

    /// The layer's paged-KV VIEW as an operand — the tier-1 `kv_cache`
    /// object, minted through the trace's dedup.
    #[must_use]
    pub fn cache(&self) -> Val {
        runtime::kv_cache(&self.t, self.l)
    }
}

/// Depthwise-conv handle; layer is both tag and conv-state key.
#[derive(Clone)]
pub struct ConvW {
    pub name: String,
    /// `None` means the checkpoint has no bias plane.
    pub bias: Option<String>,
    pub kernel: u32,
    pub layer: u32,
}

/// GDN prep uses two weight names in one op.
#[derive(Clone)]
pub struct GdnPrepW {
    pub a_log: String,
    pub dt_bias: String,
    pub layer: u32,
}

/// Per-layer recurrent-state handle for GDN delta rule.
#[derive(Clone)]
pub struct Rs {
    t: Trace,
    pub l: u32,
}

impl Rs {
    /// `pub`: same external-namespace reason as `Kv::at`.
    pub fn at(t: &Trace, l: u32) -> Rs {
        Rs { t: t.clone(), l }
    }

    /// The state this layer's recurrent statements address —
    /// [`Kv::state`]'s twin on the recurrent store.
    #[must_use]
    pub fn state(&self) -> StateRef {
        StateRef {
            store: StateStore::RecurrentState,
            layer: self.l,
        }
    }

    /// The layer's recurrent-state VIEW as an operand — the tier-1
    /// `recurrent_state` object, minted through the trace's dedup.
    #[must_use]
    pub fn view(&self) -> Val {
        runtime::recurrent(&self.t, self.l)
    }
}

/// Per-layer namespace of eager handles.
pub struct Layer {
    pub qkv: MatW,
    pub q_proj: MatW,
    pub k_proj: MatW,
    pub v_proj: MatW,
    pub q_bias: MatW,
    pub k_bias: MatW,
    pub v_bias: MatW,
    pub o_proj: MatW,
    /// Attention-output bias; separate from `o_proj` to avoid changing accumulation order.
    pub o_bias: MatW,
    /// Packed gate‖up bank when loader joined it.
    pub gate_up: MatW,
    /// Split gate/up handles when loader did not join them.
    pub gate_proj: MatW,
    pub up_proj: MatW,
    pub down: MatW,
    /// Router logits: hidden -> expert.
    pub router: MatW,
    /// Per-expert logit bias; affects ranking.
    pub router_bias: MatW,
    /// Expert-bank weights; routed kernels index expert slots.
    pub expert_gate: MatW,
    pub expert_up: MatW,
    pub expert_down: MatW,
    /// Optional dense expert and its blend gate.
    pub shared_gate: MatW,
    pub shared_up: MatW,
    pub shared_down: MatW,
    pub shared_gate_proj: MatW,
    pub attn_norm: NormW,
    pub mlp_norm: NormW,
    /// Sandwich-norm attention output.
    pub post_attn_norm: NormW,
    /// Sandwich-norm FFN output.
    pub post_mlp_norm: NormW,
    /// Gemma-4 dense-leg output norm.
    pub post_mlp_norm_1: NormW,
    /// Gemma-4 routed-leg input norm; distinct tensor from `mlp_norm`.
    pub mlp_norm_2: NormW,
    pub post_mlp_norm_2: NormW,
    /// Router-input RMS scale.
    pub router_scale: NormW,
    /// Post-softmax per-expert gain; before softmax would change ranking.
    pub router_expert_scale: MatW,
    pub q_norm: NormW,
    pub k_norm: NormW,
    pub kv: Kv,
}

/// Shared dense/mixture transformer shape used to build the namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelShape {
    pub hidden: u32,
    pub intermediate: u32,
    pub n_experts: u32,
    pub moe_intermediate: u32,
    pub shared_intermediate: u32,
    pub vocab: u32,
    pub head_dim: u32,
    /// `q_heads * head_dim`.
    pub q_width: u32,
    /// `kv_heads * head_dim`.
    pub kv_width: u32,
    pub qk_norm: QkNorm,
    pub norm_variant: NormVariant,
    /// The family's RMS epsilon in MILLIONTHS (`1e-5` is `10`), an integer
    /// because this struct derives `Eq` and a float field would end that.
    /// [`ModelShape::norm_eps`] gives it back as the f32 every handle takes.
    pub norm_eps_micro: u32,
    pub tied_embeddings: bool,
    /// Uniform projection storage for handles from `M::layer`.
    pub proj_repr: WeightRepr,
}

impl ModelShape {
    /// The stored millionths, as the f32 the handles carry.
    #[must_use]
    pub fn norm_eps(&self) -> f32 {
        self.norm_eps_micro as f32 / 1.0e6
    }
}

/// Model context: shape plus tape, no lowering state.
pub struct M {
    t: Trace,
    f: ModelShape,
}

impl Val {
    pub fn trace(&self) -> &Trace {
        &self.t
    }

    /// Preserve tag when opening a value-producing guard around this value.
    pub fn layer(&self) -> Option<u32> {
        self.layer
    }

    /// SSA identity only; used to stage one activation shared by q/k/v.
    #[must_use]
    pub fn key(&self) -> model_ir::trace::ValueId {
        self.id
    }
}

impl M {
    pub fn shape(&self) -> &ModelShape {
        &self.f
    }

    pub fn trace(&self) -> &Trace {
        &self.t
    }

    /// Must precede the layer loop; assigns layer-tagged ops to depth windows.
    pub fn depth_window(&self) {
        self.t.with(None, |b| b.declare_depth_window());
    }

    pub fn embed(&self) -> Val {
        embed_with(&self.t, "embed", self.f.hidden, self.f.vocab)
    }

    pub fn layer(&self, l: u32) -> Layer {
        let f = &self.f;
        let eps = f.norm_eps();
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
            eps,
        };
        let qk_norm = |name: &str| NormW {
            name: w(name),
            variant: f.norm_variant,
            per_head: (f.qk_norm == QkNorm::PerHead).then_some(f.head_dim),
            layer: Some(l),
            eps,
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
            eps: self.f.norm_eps(),
        }
    }

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

/// Trace backend-neutral ops; consumer chooses kernels.
pub fn trace_semantic(family: &str, shape: &ModelShape, body: impl FnOnce(&mut M)) -> ForwardPlan {
    run(family.to_string(), shape, body)
}

/// Trace CUDA-lowered declarations as `<family>.cuda.<class>`.
pub fn trace_cuda(
    family: &str,
    shape: &ModelShape,
    class: FireClass,
    body: impl FnOnce(&mut M),
) -> ForwardPlan {
    run(format!("{family}.cuda.{}", class_word(class)), shape, body)
}

/// Trace Metal-lowered declarations as `<family>.metal.<class>`; currently unused.
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

/// Trace a family that supplies its own namespace/facts.
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

/// Fragment input: `[Tokens, hidden]` bf16, layer `None`.
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

/// `layout::embed_bf16`: the token-id stream is minted by name and the
/// vocab rides the params run — the swept signature's appended pair.
pub fn embed_with(t: &Trace, weight: &str, hidden: u32, vocab: u32) -> Val {
    let id = t.with(None, |b| {
        let token_ids = b.runtime_tensor("token_ids", None, Shape(vec![Dim::Tokens]), DType::I32);
        b.launch_with_params(
            weight_launch::EMBED,
            vec![weight.to_string()],
            None,
            vec![vocab],
            vec![token_ids],
            vec![(Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)],
        )[0]
    });
    Val {
        t: t.clone(),
        id,
        layer: None,
    }
}

/// The tier-1 symbols `ops.rs` and the handles state; named once.
pub mod weight_launch {
    pub const EMBED: &str = "layout::embed_bf16";
}

pub fn lm_head_at(t: &Trace, x: &Val, weight: &str, vocab: u32) -> Val {
    let id = t.with(None, |b| b.lm_head(x.id, weight, vocab));
    Val {
        t: t.clone(),
        id,
        layer: None,
    }
}

/// Resolve tied embedding here so each family cannot forget the fact.
pub fn lm_head_tied(t: &Trace, x: &Val, tied: bool, vocab: u32) -> Val {
    lm_head_at(t, x, if tied { "embed" } else { "lm_head" }, vocab)
}

/// The runtime vocabulary's veneer: every driver-owned value a text may
/// name, minted here and answered by the driver's resolver at bind. The
/// names are `kernels::runtime`'s (identity in the floor crate, carrier in
/// the plane, answer in the driver); this module is only the authoring
/// surface's spelling of them.
pub mod runtime {
    use super::*;

    fn tensor(t: &Trace, name: &str, shape: Shape, dtype: DType) -> Val {
        let id = t.with(None, |b| b.runtime_tensor(name, None, shape, dtype));
        Val {
            t: t.clone(),
            id,
            layer: None,
        }
    }

    /// Per-token absolute positions, `[Tokens]` i32.
    pub fn positions(t: &Trace) -> Val {
        tensor(t, "positions", Shape(vec![Dim::Tokens]), DType::I32)
    }

    /// The fire's token ids, `[Tokens]` i32.
    pub fn token_ids(t: &Trace) -> Val {
        tensor(t, "token_ids", Shape(vec![Dim::Tokens]), DType::I32)
    }

    /// Which request each token row belongs to, `[Tokens]` i32.
    pub fn request_of_token(t: &Trace) -> Val {
        tensor(t, "request_of_token", Shape(vec![Dim::Tokens]), DType::I32)
    }

    /// The query-window CSR, `[Requests + 1]` i32 (stated as `[Requests]`;
    /// the driver stages the +1 row the CSR convention implies).
    pub fn qo_indptr(t: &Trace) -> Val {
        tensor(t, "qo_indptr", Shape(vec![Dim::Requests]), DType::I32)
    }

    /// The fire's token stream WITH its request boundaries: `token_ids`
    /// (`[Tokens]` i32) paired with the qo CSR — the ragged pair, minted
    /// together so the halves cannot come from two different fires.
    pub fn token_stream(t: &Trace) -> RaggedVal {
        RaggedVal {
            data: token_ids(t),
            indptr: qo_indptr(t),
        }
    }

    /// Pair a token-rowed stream the caller computed — attention's q —
    /// with the fire's query-window CSR. The statement-level spelling of
    /// "these rows are ragged over the fire's requests".
    pub fn query_windows(data: &Val) -> RaggedVal {
        RaggedVal {
            data: data.clone(),
            indptr: qo_indptr(&data.t),
        }
    }

    /// Row-validity mask, `[Tokens]` i32.
    pub fn row_valid(t: &Trace) -> Val {
        tensor(t, "row_valid", Shape(vec![Dim::Tokens]), DType::I32)
    }

    /// Rows the fire samples, `[Requests]` i32.
    pub fn sampling_indices(t: &Trace) -> Val {
        tensor(
            t,
            "sampling_indices",
            Shape(vec![Dim::Requests]),
            DType::I32,
        )
    }

    /// First-token flags, `[Requests]` i32.
    pub fn first_token(t: &Trace) -> Val {
        tensor(t, "first_token", Shape(vec![Dim::Requests]), DType::I32)
    }

    /// The layer's paged-KV view — an OBJECT, not a tensor: resolved by
    /// name, read by the routine through `In<Struct<KvCache>>`.
    pub fn kv_cache(t: &Trace, l: u32) -> Val {
        let id = t.with(Some(l), |b| b.runtime_object("kv_cache", Some(l)));
        Val {
            t: t.clone(),
            id,
            layer: Some(l),
        }
    }

    /// The layer's recurrent-state view (slab + conv half).
    pub fn recurrent(t: &Trace, l: u32) -> Val {
        let id = t.with(Some(l), |b| b.runtime_object("recurrent_state", Some(l)));
        Val {
            t: t.clone(),
            id,
            layer: Some(l),
        }
    }

    /// The custom-mask view; null/false inside when the fire has none.
    pub fn attention_mask(t: &Trace) -> Val {
        let id = t.with(None, |b| b.runtime_object("attention_mask", None));
        Val {
            t: t.clone(),
            id,
            layer: None,
        }
    }
}

/// The fire's runtime values as a HANDLE the forward mints once and passes
/// into statements — design-no-ask §10's `let rt = dsl::rt(t);`.
///
/// A generated wrapper takes one argument per mark with no secret mints,
/// runtime streams included: a reader of
/// `generated::rope_bf16(&q, &k, .., &rt.positions(), ..)` can SEE the
/// statement depends on `positions`. Each accessor mints through
/// `TraceBuilder`'s `(name, layer)` dedup, so a stream named by twenty
/// statements is one value — exactly what the hand wrappers' hidden
/// `rt_tokens` calls recorded, now in the caller's hand.
///
/// The per-layer handles fold in beside it ([`Rt::kv`]/[`Rt::rs`]); the
/// handles themselves stay — they answer layer identity, this answers
/// fire-scope identity.
#[derive(Clone)]
pub struct Rt {
    t: Trace,
}

/// Mint the runtime handle for this trace scope.
#[must_use]
pub fn rt(t: &Trace) -> Rt {
    Rt { t: t.clone() }
}

impl Rt {
    /// Per-token absolute positions, `[Tokens]` i32.
    #[must_use]
    pub fn positions(&self) -> Val {
        runtime::positions(&self.t)
    }

    /// The fire's token ids, `[Tokens]` i32.
    #[must_use]
    pub fn token_ids(&self) -> Val {
        runtime::token_ids(&self.t)
    }

    /// Which request each token row belongs to, `[Tokens]` i32.
    #[must_use]
    pub fn request_of_token(&self) -> Val {
        runtime::request_of_token(&self.t)
    }

    /// The query-window CSR, `[Requests]` i32 (the driver stages the `+1`
    /// row the CSR convention implies).
    #[must_use]
    pub fn qo_indptr(&self) -> Val {
        runtime::qo_indptr(&self.t)
    }

    /// Row-validity mask, `[Tokens]` i32.
    #[must_use]
    pub fn row_valid(&self) -> Val {
        runtime::row_valid(&self.t)
    }

    /// Rows the fire samples, `[Requests]` i32.
    #[must_use]
    pub fn sampling_indices(&self) -> Val {
        runtime::sampling_indices(&self.t)
    }

    /// First-token flags, `[Requests]` i32.
    #[must_use]
    pub fn first_token(&self) -> Val {
        runtime::first_token(&self.t)
    }

    /// The custom-mask view; null/false inside when the fire has none.
    #[must_use]
    pub fn attention_mask(&self) -> Val {
        runtime::attention_mask(&self.t)
    }

    /// The layer's KV handle — [`Kv::at`], folded beside the runtime
    /// handle so a forward mints both from one place.
    #[must_use]
    pub fn kv(&self, l: u32) -> Kv {
        Kv::at(&self.t, l)
    }

    /// The layer's recurrent handle — [`Rs::at`]'s fold, as [`Rt::kv`].
    #[must_use]
    pub fn rs(&self, l: u32) -> Rs {
        Rs::at(&self.t, l)
    }

    /// A TIER-2 runtime OBJECT by its dotted, plane-declared name
    /// (`"fa2.prefill"`, `"moe.banks"`, `"dsv4.ape"`, …) — the driver
    /// answers the name at bind, and `check_plan` refuses a spelling
    /// outside the vocabulary. Tier-1 objects have their own accessors;
    /// this is the operand form of what a plane declares for itself.
    #[must_use]
    pub fn object(&self, name: &str, layer: Option<u32>) -> Val {
        let id = self.t.with(layer, |b| b.runtime_object(name, layer));
        Val {
            t: self.t.clone(),
            id,
            layer,
        }
    }
}

/// The dtype AXES a catalogued forward is generic over — the no-ask
/// contract's S2c (`.wiki/designs/design-no-ask.md` §9).
///
/// A forward declares `<W1: DtypeAxis, W2: DtypeAxis, A: DtypeAxis,
/// K: KvAxis>` — projection repr, expert repr, activation dtype, KV
/// scheme: the four things real deployments vary. The CATALOG enumerates
/// the shipping points (`catalogue!`), and every consumer derives from the
/// axes: the text's `MatW.repr`, the manifest's tensor claims, the load
/// contract's checks. One declaration, three readers — a checkpoint that
/// ships mxfp4 experts refuses the bf16 row and matches the mxfp4 row,
/// which is load-time dtype dispatch as contract matching.
///
/// Markers, not values, so a catalog row is a TYPE INSTANTIATION the
/// compiler monomorphizes — the closure the routine level must not make
/// (`Routine::point` selects there; the demand set is whatever the
/// enumerated rows reach, held by the coverage test).
pub mod axes {
    use super::{DType, ScaleLayout, WeightRepr};

    /// One weight-bearing axis: what the checkpoint holds and the text
    /// states for a bank of projections.
    pub trait Dtype: 'static {
        /// The activation/storage dtype statements read and write.
        const DTYPE: DType;
        /// The tensor representation the manifest claims and `MatW` states.
        const REPR: WeightRepr;
        /// The axis's name, joined into the catalogued family string.
        const NAME: &'static str;
    }

    /// The KV cache axis: which scheme the pages hold. Subsumes the
    /// `kv_native_bf16` boolean and `Boot::route`'s bf16-vs-quantised
    /// choice — a load-time fact, stated where the SKU is named.
    pub trait KvDtype: 'static {
        /// The pages hold the model's own bf16.
        const NATIVE_BF16: bool;
        /// The axis's name.
        const NAME: &'static str;
    }

    /// Plain bf16: the repr every dense row ships today.
    pub enum Bf16 {}
    impl Dtype for Bf16 {
        const DTYPE: DType = DType::BF16;
        const REPR: WeightRepr = WeightRepr::Bf16;
        const NAME: &'static str = "bf16";
    }

    /// MXFP4 experts in Marlin layout (gpt-oss's shipped form).
    pub enum Mxfp4 {}
    impl Dtype for Mxfp4 {
        const DTYPE: DType = DType::BF16;
        const REPR: WeightRepr = WeightRepr::Mxfp4Marlin;
        const NAME: &'static str = "mxfp4";
    }

    /// W4A16 experts in kimi-k2's shipped packing: eight 4-bit codes per
    /// i32 word, one bf16 scale per 32-code group along the input axis
    /// (`weight_scale: [out, in/32]`), and the `code - 8` bias the
    /// scheme's own (`QuantScheme::Int4B8`) — no zero-point tensor, so
    /// `zero_point` is false. The group here is the ONE number the
    /// family's wna16 statements read (`wna16_{gate_up,down}_decode`
    /// take it as a param); the rest is the load contract's metadata.
    pub enum Wna16 {}
    impl Dtype for Wna16 {
        const DTYPE: DType = DType::BF16;
        const REPR: WeightRepr = WeightRepr::Scaled {
            layout: ScaleLayout::PerGroup,
            group: 32,
            axis: 0,
            zero_point: false,
        };
        const NAME: &'static str = "wna16";
    }

    pub use Bf16 as Bf16Ax;
    pub use Dtype as DtypeAxis;
    pub use KvDtype as KvAxis;
    pub use Mxfp4 as Mxfp4Ax;
    pub use Wna16 as Wna16Ax;

    /// Native bf16 KV pages.
    pub enum NativeKv {}
    impl KvDtype for NativeKv {
        const NATIVE_BF16: bool = true;
        const NAME: &'static str = "kv-bf16";
    }
}

/// Enumerate one shipping SKU: a name and the monomorphized forward it
/// instantiates. The row is a `(name, fn)` pair in a plain table the
/// family exposes; the coverage test walks every family's table, traces
/// each point, and `TraceBuilder::finish`'s `check_plan` refuses a point
/// whose statements reach a routine row that does not exist — the closure
/// of the demand set, checked rather than hoped.
#[macro_export]
macro_rules! catalog {
    ($( ($name:literal, $trace:path, $m:expr $(,)?) ),+ $(,)?) => {
        &[ $( ($name, (|plane| $trace($name, &$m, plane)) as _) ),+ ]
    };
}

#[macro_export]
macro_rules! catalogue {
    ($( ($name:literal, $f:path $(,)?) ),+ $(,)?) => {
        &[ $( ($name, $f as _) ),+ ]
    };
}

/// State one routine by its MARKER — the no-ask contract's auto-registration.
///
/// `#[routine]` emits a `Signature` (namespace, trace name) and a
/// `Derivation` (the source column) beside every routine, so a routine
/// added to a plane is a routine this fn can state with no wrapper
/// written: the symbol and the argument PLACEMENT come off the marker, and
/// the call is validated against the column at trace time — which is load
/// time, the same moment `check_plan` re-checks the statement it records.
///
/// The named wrappers in [`cuda`] remain as readable forms; this is the
/// floor they could each become one line over.
pub mod fire {
    use super::*;
    use ::kernels::{Kind, Source};
    use model_ir::trace::ValueId;

    /// One statement's operands, in each run's own order.
    #[derive(Default)]
    pub struct Call {
        /// The `In`/`InOut` slots, in signature order — activation `Val`s
        /// and runtime mints alike.
        pub inputs: Vec<ValueId>,
        /// The `Const<Tensor<..>>` weight slots, by name, in order.
        pub weights: Vec<String>,
        /// The scalar `Const` slots, as bits, in signature order.
        pub params: Vec<u32>,
        /// One `(Shape, DType)` per `Out`/`InOut` slot, in order.
        pub outs: Vec<(Shape, DType)>,
        /// The per-layer state this statement addresses, if any.
        pub state: Option<StateRef>,
        /// The layer tag.
        pub layer: Option<u32>,
        /// Fire-extent slots, as [`OpKind::Launch::param_extents`].
        pub extents: Vec<(u8, Shape)>,
    }

    /// What `R`'s column claims, counted: `(required, optional)` per run,
    /// in the order in / out / weight / param.
    ///
    /// A NULLABLE mark (`Option<In<..>>`, `Option<Out<..>>`,
    /// `Option<Const<Tensor<..>>>`) may be left out of its run — omission
    /// is how a statement spells absence, exactly as the hand wrappers do
    /// (`check_plan` re-checks the recorded statement either way).
    /// `SOURCES` and `DERIVED` are both one entry per parameter with the
    /// `Ctx` dropped, so the zip is aligned by construction.
    fn counts(sources: &[Option<Source>], derived: &[::kernels::Derived]) -> [(usize, usize); 4] {
        let mut runs = [(0usize, 0usize); 4];
        for (i, s) in sources.iter().enumerate() {
            let nullable = derived.get(i).is_some_and(|d| d.nullable);
            let mut add = |run: usize| {
                if nullable {
                    runs[run].1 += 1;
                } else {
                    runs[run].0 += 1;
                }
            };
            match s {
                Some(Source::Slot(Kind::In, _)) => add(0),
                Some(Source::Slot(Kind::Out, _)) => add(1),
                Some(Source::Alias(..)) => {
                    add(0);
                    add(1);
                }
                Some(Source::Slot(Kind::Weight, _)) => add(2),
                Some(Source::Slot(Kind::Param | Kind::ParamF32, _)) => add(3),
                _ => {}
            }
        }
        runs
    }

    /// Record one launch of `R`. Panics — at trace time, which is load
    /// time — when the call does not fill the column: a statement short of
    /// its signature must not become a statement at all. A run with
    /// nullable slots may be filled short by exactly the absent ones.
    pub fn fire<R: ::kernels::Signature + ::kernels::Derivation>(t: &Trace, call: Call) -> Vec<Val> {
        let symbol = format!("{}::{}", R::NAMESPACE, R::NAME);
        checked::<R>(t, &symbol, call)
    }

    /// [`fire()`], with the statement's SYMBOL supplied by the caller — the
    /// SHADER-plane spelling. A CUDA statement states `namespace::name` and
    /// the marker fixes it; a metal/vulkan/wgpu statement states an
    /// instantiated ENTRYPOINT (`rms_single_row_bfloat16`,
    /// `affine_qmv_fast_bfloat16_gs_64_b_4`), resolved back to its routine
    /// by the census stem (`kernels_metal::kernel_of`) — so the symbol
    /// carries the routine's own dtype point plus whatever instantiation
    /// axes the caller chose, and cannot come off the marker alone.
    ///
    /// The column validation is [`fire()`]'s, and one check is added: the
    /// symbol must resolve to `R`'s own name, so a wrapper cannot fire one
    /// routine's marker under another routine's entrypoint.
    pub fn fire_at<R: ::kernels::Signature + ::kernels::Derivation>(
        t: &Trace,
        symbol: &str,
        call: Call,
    ) -> Vec<Val> {
        assert!(
            kernels_metal::kernel_of(symbol).is_some_and(|n| n == R::NAME),
            "`{symbol}` does not resolve to routine `{}` in the shader planes' \
             census (it resolves to {:?}); the statement would name one \
             routine's symbol under another routine's column",
            R::NAME,
            kernels_metal::kernel_of(symbol),
        );
        checked::<R>(t, symbol, call)
    }

    /// The shared half of [`fire()`]/[`fire_at`]: the column check and the
    /// recording.
    fn checked<R: ::kernels::Signature + ::kernels::Derivation>(
        t: &Trace,
        symbol: &str,
        call: Call,
    ) -> Vec<Val> {
        let runs = counts(R::SOURCES, R::DERIVED);
        let filled = [
            ("operand", call.inputs.len()),
            ("result", call.outs.len()),
            ("weight", call.weights.len()),
            ("param", call.params.len()),
        ];
        for ((run, got), (required, optional)) in filled.into_iter().zip(runs) {
            assert!(
                (required..=required + optional).contains(&got),
                "`{}::{}`'s column claims {required}{} {run} slots; the call fills {got}",
                R::NAMESPACE,
                R::NAME,
                if optional == 0 {
                    String::new()
                } else {
                    format!("..={}", required + optional)
                },
            );
        }
        let layer = call.layer;
        let ids = t.with(layer, |b| {
            b.launch_devwin(
                symbol,
                call.weights,
                call.state,
                call.params,
                call.extents,
                None,
                call.inputs,
                call.outs,
            )
        });
        ids.into_iter()
            .map(|id| Val {
                t: t.clone(),
                id,
                layer,
            })
            .collect()
    }
}

/// One result's geometry, resolved from the ROUTINE's stated `out(..)` rule
/// against the statement's own operands — the trace-time half of B4-gen
/// (design-no-ask §10). `inputs` is the statement's operand run in slot
/// order; the rule's ordinals index into it. Shared by both generated
/// planes (`cuda::generated`, `metal::generated`).
///
/// Panics — at trace time, which is load time — when the rule does not
/// resolve, because a result the rule cannot shape must not become a
/// statement. `Unstated` never reaches here: an unruled result keeps its
/// `(Shape, DType)` parameter on the generated wrapper.
pub(crate) fn ruled_out(
    t: &Trace,
    routine: &str,
    rule: ::kernels::OutRule,
    inputs: &[model_ir::trace::ValueId],
    params: &[u32],
) -> (Shape, DType) {
    let b = t.inner.borrow();
    // A raise (a runtime OBJECT operand) has no rectangle; a rule must
    // never name one, so it enters the slot table as the empty shape,
    // which every constructor refuses rather than reads.
    let shapes: Vec<Shape> = inputs
        .iter()
        .map(|&id| {
            if b.is_raised(id) {
                Shape(vec![])
            } else {
                b.value_shape(id)
            }
        })
        .collect();
    let dtypes: Vec<DType> = inputs.iter().map(|&id| b.value_dtype(id)).collect();
    let refs: Vec<&Shape> = shapes.iter().collect();
    model_ir::kernels::out_shape(rule, &refs, &dtypes, params).unwrap_or_else(|| {
        panic!("`{routine}`'s out rule does not resolve against this statement's operands")
    })
}

pub mod cuda;
pub mod declare;
pub mod facts;
pub mod forward;
pub mod kernels;
pub mod load;
mod guard;
mod record;
pub mod metal;
mod ops;
mod rows;

pub use model_dsl_macros::Facts;

pub use declare::*;
pub use record::{Value, Windows};
pub use model_ir::plan::Plan;
pub use facts::*;
pub use forward::*;
pub use guard::*;
pub use ops::*;
pub use rows::*;

/// The plane a trace is bound to; vulkan and wgpu consume the metal-shaped
/// table, so two tables is two planes.
pub use model_ir::kernels::Backend as Plane;


/// The seam vocabulary plus the one per-layer tap statement.
pub mod seam {
    pub use model_ir::seam::*;

    use crate::record::Value;

    pub trait Sees {
        fn values(&self) -> Vec<&Value>;
    }

    impl Sees for (&Value,) {
        fn values(&self) -> Vec<&Value> {
            vec![self.0]
        }
    }

    impl Sees for (&Value, &Value) {
        fn values(&self) -> Vec<&Value> {
            vec![self.0, self.1]
        }
    }

    pub fn at<S: Sees>(def: Def, sees: S, layer: u32) {
        let values = sees.values();
        values[0].rec.seam(def.name, &values, Some(layer));
    }
}

/// Record a seam statement; `sees` order must match `def.sees`.
pub fn seam(t: &Trace, def: &seam::Def, sees: &[&Val], layer: Option<u32>) {
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
            // Capture guard op index before opening it; seam points at the construct.
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
