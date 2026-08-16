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
}

#[derive(Clone)]
pub struct Val {
    t: Trace,
    pub(crate) id: model_ir::trace::ValueId,
    layer: Option<u32>,
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
        self.t
            .with(Some(self.l), |b| b.kv_append(self.l, k.id, v.id));
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
    pub tied_embeddings: bool,
    /// Uniform projection storage for handles from `M::layer`.
    pub proj_repr: WeightRepr,
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

pub fn embed_with(t: &Trace, weight: &str, hidden: u32) -> Val {
    let id = t.with(None, |b| b.embed(weight, hidden));
    Val {
        t: t.clone(),
        id,
        layer: None,
    }
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

pub mod cuda;
mod guard;
pub mod metal;
mod ops;
mod rows;

pub use guard::*;
pub use ops::*;
pub use rows::*;

/// Re-exported so declarations spell the seam vocabulary from this surface.
pub use model_ir::seam;

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
