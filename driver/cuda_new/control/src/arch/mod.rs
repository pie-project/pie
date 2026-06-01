//! The per-arch contract — replaces `bound_model.cpp` and the 55
//! `is_*_arch` branches in `entry.cpp::run_impl`.
//!
//! Arch-specific knowledge lives as *data* (`ArchSpec`) plus a small
//! trait impl, not a branching cascade. Adding an arch is one `impl Arch`
//! here + one `device/src/forward/<arch>` TU on the C++ side. No central
//! file grows.

use anyhow::{Result, bail};
use serde::Deserialize;

use crate::ffi::PieArchId;

/// Capability flags the executor consults (graph capture, fused argmax,
/// compact logits). Mirrors driver/cuda's `model::ModelCapabilities`.
#[derive(Copy, Clone, Default, Debug)]
pub struct Capabilities {
    pub graph_safe: bool,
    pub supports_tp_greedy_argmax: bool,
    pub supports_compact_logits: bool,
    pub supports_small_prefill_graph: bool,
    pub supports_fused_lmhead_argmax: bool,
}

/// What each attention layer is, for heterogeneous stacks (Nemotron-H
/// Mamba/attention mix, Qwen3.5 linear/full mix, Gemma sliding/full).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LayerKind {
    FullAttention,
    SlidingAttention,
    LinearAttention, // Mamba2 / gated-delta-net
    Mla,             // DeepSeek-V4 / Kimi / GLM5
    Mlp,             // no token mixer — FFN-only layer (Nemotron-H '-' layers)
}

/// MLA (multi-head latent attention) latent dims — DeepSeek-V3/V4, Kimi, GLM.
/// `None` on `ArchSpec.mla` for non-MLA archs. Mirrors `mla_block.cuh`'s
/// per-head contract (the q up-proj splits into a NoPE + RoPE part; the
/// compressed-KV latent is `kv_lora_rank` wide).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MlaDims {
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
}

/// Mamba-2 / SSD state dims — Nemotron-H (and the Mamba family). `None` on
/// `ArchSpec.mamba` for non-recurrent archs. Mirrors `ssm_scan.cuh` /
/// `nemotron_block.cuh` (intermediate = num_heads*head_dim; conv_dim =
/// intermediate + 2*n_groups*state_size).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MambaDims {
    pub num_heads: usize,
    pub head_dim: usize,
    pub state_size: usize,
    pub n_groups: usize,
    pub conv_kernel: usize,
}

/// Declarative description of a model, derived once from the HF config.
/// The builder turns this into `PieKvLayout` + `PieWorkspaceDims` and
/// drives the C++ alloc primitives.
#[derive(Clone, Debug)]
pub struct ArchSpec {
    pub id: PieArchId,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_model_len: usize,
    /// Per-layer kind. Homogeneous archs fill this with one value.
    pub layer_kinds: Vec<LayerKind>,
    /// Per-layer head_dim override (Gemma-4 dual 256/512). Empty = scalar.
    pub per_layer_head_dim: Vec<i32>,
    /// KV-cache sharing map (Gemma-4). Empty = each layer is its own source.
    pub kv_source_layer: Vec<i32>,
    pub moe_experts: usize, // 0 = dense
    pub recurrent_state_slots: usize, // 0 = no Mamba/linear state
    /// MLA latent dims (DeepSeek-V4 / Kimi / GLM). `None` for non-MLA archs.
    pub mla: Option<MlaDims>,
    /// Mamba-2 / SSD state dims (Nemotron-H). `None` for non-recurrent archs.
    pub mamba: Option<MambaDims>,
    /// Per-expert FFN width for MoE archs (`0` = dense; falls back to
    /// `intermediate_size`). DeepSeek/Qwen-MoE/GPT-OSS set this < the dense width.
    pub moe_intermediate_size: usize,
    /// Experts routed per token (top-k). `0` for dense.
    pub num_experts_per_tok: usize,
    /// Always-on shared experts added to the routed MoE output (DeepSeek /
    /// Qwen-MoE). `0` = none.
    pub n_shared_experts: usize,
    /// Attention logit soft-cap (Gemma-2/3 `attn_logit_softcapping`). `None` = off.
    pub attn_logit_softcap: Option<f32>,
    /// Final lm_head logit soft-cap (Gemma-2 `final_logit_softcapping`). `None` = off.
    pub final_logit_softcap: Option<f32>,
    /// AltUp parallel residual streams (Gemma-3n/4 `altup_num_inputs`). `0`/`1`
    /// = no AltUp.
    pub altup_num_inputs: usize,
    /// RoPE base frequency (`rope_theta`). Carried through so the builder
    /// can hand it to the per-arch forward without re-parsing the config.
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Sliding-window left context. `None` = full causal across the stack;
    /// `Some(w)` = a sliding window of `w` tokens (mirrors HF
    /// `sliding_window`). Per-layer overrides live in `layer_kinds`.
    pub sliding_window: Option<usize>,
    /// `lm_head` shares storage with the embedding table.
    pub tie_word_embeddings: bool,
    /// Per-head q-norm / k-norm RMSNorm before RoPE (Qwen3 / Gemma-3 /
    /// OLMo-3). The Llama-like C++ body skips the extra norm when this is
    /// false (the weight pointers are simply left null).
    pub use_qk_norm: bool,
    /// QKV projections carry an additive bias term (Qwen-2 / OLMo-3 /
    /// GPT-OSS). Llama-3 / Qwen-3 / Phi-3 / Mistral leave it false.
    pub use_qkv_bias: bool,
}

/// Workspace sizing — derived from an `ArchSpec` + the planned forward
/// capacity. Maps onto `ffi::PieWorkspaceDims`.
#[derive(Clone, Debug)]
pub struct WorkspaceDims {
    pub max_tokens: usize,
    pub max_requests: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub recurrent_state_slots: usize,
    pub moe_experts: usize,
}

/// Parsed HuggingFace `config.json` — the narrow slice the arch registry
/// needs to derive an `ArchSpec`. Ports the equivalent fields of
/// driver/cuda's `loader/hf_config.hpp::HfConfig`; phase 2 will populate
/// these from `pie-weight-loader` rather than re-parsing JSON here.
///
/// `Deserialize` is wired so a config.json can be read directly in tests
/// and early bring-up; the field names match HF's snake_case keys, and
/// `#[serde(default)]` lets the many optional fields fall back cleanly.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct HfConfig {
    /// First entry of HF's `architectures` list (e.g. "Qwen3ForCausalLM").
    /// Kept verbatim for capability reporting and as a dispatch fallback.
    pub architectures: Vec<String>,
    /// Lower-case `model_type` ("qwen3", "llama", …) — the primary
    /// registry key (some configs ship multiple architectures).
    pub model_type: String,

    // ── Transformer dimensions ──────────────────────────────────────
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// GQA group count. Equal to `num_attention_heads` for MHA.
    pub num_key_value_heads: usize,
    /// Some configs imply `head_dim = hidden_size / num_attention_heads`;
    /// Qwen3 sets it explicitly. `0` here means "derive it" (see
    /// `HfConfig::resolved_head_dim`).
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,

    // ── Norm / RoPE ────────────────────────────────────────────────
    pub rms_norm_eps: f32,
    pub rope_theta: f32,

    // ── Quirks ─────────────────────────────────────────────────────
    /// `-1` (or absent) means full causal. Stored as the raw HF value;
    /// `HfConfig::sliding_window_opt` turns it into the `Option` the
    /// `ArchSpec` carries.
    pub sliding_window: Option<i64>,
    pub tie_word_embeddings: bool,
    /// QKV/O biases (Qwen2 yes, Qwen3 no).
    pub attention_bias: bool,
    /// Qwen3 / Gemma-3 / OLMo-3 per-head q/k RMSNorm.
    pub use_qk_norm: bool,

    // ── Sparse MoE (dense models leave this 0) ─────────────────────
    /// HF `num_local_experts` / `num_experts` / DeepSeek `n_routed_experts`.
    #[serde(alias = "num_local_experts", alias = "n_routed_experts")]
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    /// Per-expert FFN width (DeepSeek/Qwen-MoE/GPT-OSS `moe_intermediate_size`);
    /// `0` falls back to the dense `intermediate_size`.
    pub moe_intermediate_size: usize,
    /// Always-on shared experts (DeepSeek `n_shared_experts`).
    pub n_shared_experts: usize,
    /// First K layers are dense, not MoE (DeepSeek `first_k_dense_replace`).
    pub first_k_dense_replace: usize,

    // ── MLA (DeepSeek-V3/V4 / Kimi / GLM) ──────────────────────────
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,

    // ── Gemma (sandwich norms, soft-cap, AltUp) ────────────────────
    pub attn_logit_softcapping: Option<f32>,
    pub final_logit_softcapping: Option<f32>,
    /// Gemma-3n/4 AltUp parallel residual streams (`0`/`1` = none).
    pub altup_num_inputs: usize,
    /// Gemma-3: every Nth layer is global attention, the rest sliding (`0` = n/a).
    pub sliding_window_pattern: usize,

    // ── Mamba-2 / Nemotron-H ───────────────────────────────────────
    pub mamba_num_heads: usize,
    pub mamba_head_dim: usize,
    #[serde(alias = "mamba_d_state")]
    pub ssm_state_size: usize,
    #[serde(alias = "mamba_n_groups")]
    pub n_groups: usize,
    #[serde(alias = "mamba_d_conv")]
    pub conv_kernel: usize,
    /// Nemotron-H per-layer schedule, e.g. "M-M-*-M-…" (`M`=Mamba, `*`=attention,
    /// `-`=MLP/MoE). Empty for non-hybrid archs.
    pub hybrid_override_pattern: String,

    /// Multimodal checkpoints (e.g. gemma-4-E4B) nest the LM's transformer dims
    /// under `text_config`; the top level only carries `model_type` + the
    /// vision/audio sub-configs. `resolve_multimodal()` lifts these up.
    pub text_config: Option<Box<HfConfig>>,
}

impl HfConfig {
    /// For multimodal checkpoints whose transformer dims live under
    /// `text_config` (the top level having `num_hidden_layers == 0`), lift the
    /// nested text config up — but keep the TOP-LEVEL `model_type` (e.g.
    /// "gemma4", not the nested "gemma4_text") so the registry key matches.
    /// A no-op for plain single-tower configs.
    pub fn resolve_multimodal(self) -> HfConfig {
        if self.num_hidden_layers == 0 {
            if let Some(tc) = self.text_config.clone() {
                let mut inner = *tc;
                if !self.model_type.is_empty() {
                    inner.model_type = self.model_type.clone();
                }
                if self.architectures.len() >= inner.architectures.len() {
                    inner.architectures = self.architectures.clone();
                }
                inner.text_config = None;
                return inner;
            }
        }
        self
    }

    /// `head_dim` if the config sets it, else `hidden_size /
    /// num_attention_heads` (matching driver/cuda's loader default).
    pub fn resolved_head_dim(&self) -> usize {
        if self.head_dim > 0 {
            self.head_dim
        } else if self.num_attention_heads > 0 {
            self.hidden_size / self.num_attention_heads
        } else {
            0
        }
    }

    /// `num_key_value_heads` if set, else fall back to MHA
    /// (`num_attention_heads`).
    pub fn resolved_num_kv_heads(&self) -> usize {
        if self.num_key_value_heads > 0 {
            self.num_key_value_heads
        } else {
            self.num_attention_heads
        }
    }

    /// HF stores `sliding_window = -1` (or omits it) for full causal.
    /// Turn any positive value into the window; everything else is `None`.
    pub fn sliding_window_opt(&self) -> Option<usize> {
        match self.sliding_window {
            Some(w) if w > 0 => Some(w as usize),
            _ => None,
        }
    }

    /// The dispatch key: lower-cased `model_type`, falling back to the
    /// first `architectures` entry when `model_type` is empty (some
    /// older checkpoints only ship `architectures`).
    pub fn dispatch_key(&self) -> String {
        if !self.model_type.is_empty() {
            self.model_type.to_ascii_lowercase()
        } else {
            self.architectures
                .first()
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_default()
        }
    }
}

/// The polymorphic per-arch interface used by the builder + executor.
pub trait Arch {
    fn spec(&self, cfg: &HfConfig) -> Result<ArchSpec>;
    fn workspace_dims(&self, spec: &ArchSpec, max_tokens: usize) -> WorkspaceDims;
    fn id(&self) -> PieArchId;
    fn caps(&self) -> Capabilities;
    // Spec-decode opt-in. Most archs return None; Gemma4-MTP / Qwen3.5
    // override (see spec.rs).
    fn drafter(&self) -> Option<()> {
        None
    }
}

/// Shared `spec()` body for the homogeneous full-attention transformer
/// shape (Llama / Qwen3 / Mistral / Phi-3 / OLMo). The two concrete archs
/// differ only in `id`, `use_qk_norm`, and `use_qkv_bias`, which they
/// thread through here.
fn build_homogeneous_spec(
    cfg: &HfConfig,
    id: PieArchId,
    use_qk_norm: bool,
    use_qkv_bias: bool,
) -> Result<ArchSpec> {
    if cfg.num_hidden_layers == 0 {
        bail!("HfConfig: num_hidden_layers is 0 — config missing or unparsed");
    }
    if cfg.num_attention_heads == 0 {
        bail!("HfConfig: num_attention_heads is 0");
    }
    let head_dim = cfg.resolved_head_dim();
    if head_dim == 0 {
        bail!("HfConfig: could not resolve head_dim (hidden_size/num_heads is 0)");
    }

    let num_layers = cfg.num_hidden_layers;
    let sliding_window = cfg.sliding_window_opt();
    let layer_kind = if sliding_window.is_some() {
        LayerKind::SlidingAttention
    } else {
        LayerKind::FullAttention
    };

    Ok(ArchSpec {
        id,
        num_layers,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        num_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.resolved_num_kv_heads(),
        head_dim,
        vocab_size: cfg.vocab_size,
        max_model_len: cfg.max_position_embeddings,
        layer_kinds: vec![layer_kind; num_layers],
        per_layer_head_dim: Vec::new(),
        kv_source_layer: Vec::new(),
        moe_experts: 0,
        recurrent_state_slots: 0,
        mla: None,
        mamba: None,
        moe_intermediate_size: 0,
        num_experts_per_tok: 0,
        n_shared_experts: 0,
        attn_logit_softcap: None,
        final_logit_softcap: None,
        altup_num_inputs: 0,
        rope_theta: cfg.rope_theta,
        rms_norm_eps: cfg.rms_norm_eps,
        sliding_window,
        tie_word_embeddings: cfg.tie_word_embeddings,
        use_qk_norm,
        use_qkv_bias,
    })
}

/// Shared `workspace_dims()` — pure projection of an `ArchSpec` plus the
/// planned forward capacity onto the `PieWorkspaceDims` fields.
fn build_workspace_dims(spec: &ArchSpec, max_tokens: usize) -> WorkspaceDims {
    WorkspaceDims {
        max_tokens,
        // One logit row per request is the steady-state decode capacity;
        // mem.rs refines this. `max_tokens` is a safe upper bound on the
        // number of concurrent requests in a fire.
        max_requests: max_tokens,
        hidden_size: spec.hidden_size,
        intermediate_size: spec.intermediate_size,
        num_heads: spec.num_heads,
        num_kv_heads: spec.num_kv_heads,
        head_dim: spec.head_dim,
        vocab_size: spec.vocab_size,
        num_layers: spec.num_layers,
        recurrent_state_slots: spec.recurrent_state_slots,
        moe_experts: spec.moe_experts,
    }
}

/// Llama-family decoder: a homogeneous "pre-norm + QKV/o + gate-up-down"
/// transformer. Covers llama, mistral/ministral3, phi3, and olmo2/3 — the
/// shapes that bind `Qwen3Weights` and route through
/// `llama_like_forward_paged` in driver/cuda. No q/k-norm.
pub struct LlamaLikeArch;

impl Arch for LlamaLikeArch {
    fn spec(&self, cfg: &HfConfig) -> Result<ArchSpec> {
        // OLMo-3 / Qwen-2 carry QKV bias; key on the HF flag so we don't
        // re-encode the per-model table here. `use_qk_norm` stays false:
        // it's the Qwen3-distinguishing quirk handled by `Qwen3Arch`.
        build_homogeneous_spec(
            cfg,
            PieArchId::LlamaLike,
            /*use_qk_norm=*/ false,
            /*use_qkv_bias=*/ cfg.attention_bias,
        )
    }

    fn workspace_dims(&self, spec: &ArchSpec, max_tokens: usize) -> WorkspaceDims {
        build_workspace_dims(spec, max_tokens)
    }

    fn id(&self) -> PieArchId {
        PieArchId::LlamaLike
    }

    fn caps(&self) -> Capabilities {
        // Mirrors LlamaLikeModel's ctor (llama_like_model.cpp):
        //   graph_safe              = native-bf16 KV (we assume bf16 here;
        //                             mem.rs/builder gates on the real KV
        //                             format once the cache is allocated),
        //   supports_compact_logits = true,
        //   supports_tp_greedy_argmax = only under TP>1 with a sharded
        //                             lm_head (off by default).
        Capabilities {
            graph_safe: true,
            supports_compact_logits: true,
            supports_tp_greedy_argmax: false,
            supports_small_prefill_graph: false,
            supports_fused_lmhead_argmax: false,
        }
    }
}

/// Qwen3: the Llama-like shape plus per-head q-norm / k-norm RMSNorm
/// applied before RoPE (the one structural quirk vs Llama). In driver/cuda
/// Qwen3 binds `Qwen3Weights` and runs the same `llama_like_forward_paged`
/// with `use_qk_norm = true`; here it gets its own `PieArchId::Qwen3` so
/// the C++ body can select the qk-norm variant explicitly.
pub struct Qwen3Arch;

impl Arch for Qwen3Arch {
    fn spec(&self, cfg: &HfConfig) -> Result<ArchSpec> {
        // Qwen3's signature quirk: q/k-norm is on regardless of whether
        // the config sets `use_qk_norm` (driver/cuda infers it from
        // model_type == "qwen3"). Bias follows the HF `attention_bias`
        // flag (Qwen3 is false; Qwen2-style checkpoints set it).
        build_homogeneous_spec(
            cfg,
            PieArchId::Qwen3,
            /*use_qk_norm=*/ true,
            /*use_qkv_bias=*/ cfg.attention_bias,
        )
    }

    fn workspace_dims(&self, spec: &ArchSpec, max_tokens: usize) -> WorkspaceDims {
        build_workspace_dims(spec, max_tokens)
    }

    fn id(&self) -> PieArchId {
        PieArchId::Qwen3
    }

    fn caps(&self) -> Capabilities {
        // Qwen3 routes through the same llama-like model in driver/cuda,
        // so the capability shape is identical.
        Capabilities {
            graph_safe: true,
            supports_compact_logits: true,
            supports_tp_greedy_argmax: false,
            supports_small_prefill_graph: false,
            supports_fused_lmhead_argmax: false,
        }
    }
}

/// Standard MoE capabilities (routing is dynamic → not graph-safe yet; compact
/// logits still apply). Shared by the MoE/MLA frontier archs.
fn moe_caps() -> Capabilities {
    Capabilities { graph_safe: false, supports_compact_logits: true, ..Default::default() }
}

/// Layer the routed-MoE sizing onto a base spec from `cfg` (experts, top-k,
/// per-expert width, shared experts). `moe_intermediate_size==0` falls back to
/// the dense `intermediate_size`.
fn apply_moe(spec: &mut ArchSpec, cfg: &HfConfig) {
    spec.moe_experts = cfg.num_experts;
    spec.num_experts_per_tok = cfg.num_experts_per_tok;
    spec.moe_intermediate_size = if cfg.moe_intermediate_size > 0 {
        cfg.moe_intermediate_size
    } else {
        cfg.intermediate_size
    };
    spec.n_shared_experts = cfg.n_shared_experts;
}

/// Qwen3.5-MoE: the Qwen3 shape (full attention + per-head q/k-norm) with the
/// dense MLP replaced by a top-K routed MoE FFN. Routes to `moe_forward`.
pub struct Qwen3MoeArch;

impl Arch for Qwen3MoeArch {
    fn spec(&self, cfg: &HfConfig) -> Result<ArchSpec> {
        let mut s = build_homogeneous_spec(cfg, PieArchId::Qwen3_5Moe, true, cfg.attention_bias)?;
        if cfg.num_experts == 0 {
            bail!("qwen3-moe: num_experts is 0 (not a MoE config)");
        }
        apply_moe(&mut s, cfg);
        Ok(s)
    }
    fn workspace_dims(&self, spec: &ArchSpec, max_tokens: usize) -> WorkspaceDims {
        build_workspace_dims(spec, max_tokens)
    }
    fn id(&self) -> PieArchId {
        PieArchId::Qwen3_5Moe
    }
    fn caps(&self) -> Capabilities {
        moe_caps()
    }
}

/// DeepSeek-V3/V4 · Kimi · GLM: MLA (compressed-latent) attention on every
/// layer, plus a routed MoE FFN (DeepSeek runs the first `first_k_dense_replace`
/// layers dense — recorded for the builder, not yet specialized). Routes to
/// `mla_forward`. `id` selects which frontier label the caller bound.
pub struct DeepseekMlaArch {
    pub id: PieArchId,
}

impl Arch for DeepseekMlaArch {
    fn spec(&self, cfg: &HfConfig) -> Result<ArchSpec> {
        let mut s = build_homogeneous_spec(cfg, self.id, /*qk_norm=*/ false, /*bias=*/ false)?;
        if cfg.kv_lora_rank == 0 {
            bail!("MLA arch: kv_lora_rank is 0 (config missing the MLA latent dims)");
        }
        s.mla = Some(MlaDims {
            q_lora_rank: cfg.q_lora_rank,
            kv_lora_rank: cfg.kv_lora_rank,
            qk_nope_head_dim: cfg.qk_nope_head_dim,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
            v_head_dim: cfg.v_head_dim,
        });
        // The q "head_dim" for MLA is the NoPE+RoPE width (spec.head_dim is a
        // placeholder for the generic path; the MLA forward uses `mla` dims).
        if cfg.qk_nope_head_dim + cfg.qk_rope_head_dim > 0 {
            s.head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
        }
        for k in s.layer_kinds.iter_mut() {
            *k = LayerKind::Mla;
        }
        if cfg.num_experts > 0 {
            apply_moe(&mut s, cfg);
        }
        Ok(s)
    }
    fn workspace_dims(&self, spec: &ArchSpec, max_tokens: usize) -> WorkspaceDims {
        build_workspace_dims(spec, max_tokens)
    }
    fn id(&self) -> PieArchId {
        self.id
    }
    fn caps(&self) -> Capabilities {
        moe_caps()
    }
}

/// GPT-OSS: routed MoE (mxfp4 weights) over attention layers carrying QKV bias
/// and attention sinks, with alternating sliding/full windows. First pass keys
/// the window off the HF `sliding_window`; the per-layer alternation is a
/// forward refinement. Routes to `moe_forward`.
pub struct GptOssArch;

impl Arch for GptOssArch {
    fn spec(&self, cfg: &HfConfig) -> Result<ArchSpec> {
        let mut s = build_homogeneous_spec(cfg, PieArchId::GptOss, /*qk_norm=*/ false, /*bias=*/ true)?;
        if cfg.num_experts == 0 {
            bail!("gpt_oss: num_experts is 0 (not a MoE config)");
        }
        apply_moe(&mut s, cfg);
        Ok(s)
    }
    fn workspace_dims(&self, spec: &ArchSpec, max_tokens: usize) -> WorkspaceDims {
        build_workspace_dims(spec, max_tokens)
    }
    fn id(&self) -> PieArchId {
        PieArchId::GptOss
    }
    fn caps(&self) -> Capabilities {
        moe_caps()
    }
}

/// Gemma-3/4: sandwich-norm transformer with per-head q/k-norm, attention +
/// final logit soft-cap, AltUp parallel residual streams (3n/4), and a
/// global/sliding window alternation (`sliding_window_pattern` = every Nth
/// layer is global). Dense FFN (geglu). `id` selects Gemma4 vs Gemma3n.
pub struct Gemma4Arch {
    pub id: PieArchId,
}

impl Arch for Gemma4Arch {
    fn spec(&self, cfg: &HfConfig) -> Result<ArchSpec> {
        let mut s = build_homogeneous_spec(cfg, self.id, /*qk_norm=*/ true, /*bias=*/ false)?;
        s.attn_logit_softcap = cfg.attn_logit_softcapping;
        s.final_logit_softcap = cfg.final_logit_softcapping;
        s.altup_num_inputs = cfg.altup_num_inputs;
        // Gemma-3 alternation: every `sliding_window_pattern`-th layer is global
        // attention, the rest sliding (only when a window is configured).
        if s.sliding_window.is_some() && cfg.sliding_window_pattern > 0 {
            let p = cfg.sliding_window_pattern;
            for (i, k) in s.layer_kinds.iter_mut().enumerate() {
                *k = if (i + 1) % p == 0 {
                    LayerKind::FullAttention
                } else {
                    LayerKind::SlidingAttention
                };
            }
        }
        Ok(s)
    }
    fn workspace_dims(&self, spec: &ArchSpec, max_tokens: usize) -> WorkspaceDims {
        build_workspace_dims(spec, max_tokens)
    }
    fn id(&self) -> PieArchId {
        self.id
    }
    fn caps(&self) -> Capabilities {
        // AltUp + sandwich norms → not graph-safe in the first pass.
        Capabilities { graph_safe: false, supports_compact_logits: true, ..Default::default() }
    }
}

/// Nemotron-H: a hybrid stack interleaving Mamba-2 mixer, attention, and
/// MLP/MoE layers per the `hybrid_override_pattern` (`M`=Mamba, `*`=attention,
/// `-`=MLP). Carries the Mamba state dims; routes mixer layers to the
/// Nemotron Mamba block and attention layers to the paged-attention path.
pub struct NemotronHArch;

impl Arch for NemotronHArch {
    fn spec(&self, cfg: &HfConfig) -> Result<ArchSpec> {
        let mut s = build_homogeneous_spec(cfg, PieArchId::NemotronH, /*qk_norm=*/ false, /*bias=*/ cfg.attention_bias)?;
        let pat = cfg.hybrid_override_pattern.trim();
        if pat.is_empty() {
            bail!("nemotron_h: hybrid_override_pattern is empty (need the per-layer M/*/- schedule)");
        }
        let kinds: Vec<LayerKind> = pat
            .chars()
            .filter(|c| !c.is_whitespace())
            .map(|c| match c {
                'M' | 'm' => Ok(LayerKind::LinearAttention),
                '*' => Ok(LayerKind::FullAttention),
                '-' => Ok(LayerKind::Mlp),
                other => Err(other),
            })
            .collect::<std::result::Result<_, _>>()
            .map_err(|c| anyhow::anyhow!("nemotron_h: unknown layer code '{c}' in hybrid_override_pattern"))?;
        if kinds.len() != s.num_layers {
            bail!(
                "nemotron_h: hybrid_override_pattern has {} layers but num_hidden_layers is {}",
                kinds.len(),
                s.num_layers
            );
        }
        let mamba_layers = kinds.iter().filter(|&&k| k == LayerKind::LinearAttention).count();
        s.layer_kinds = kinds;
        if cfg.ssm_state_size == 0 {
            bail!("nemotron_h: ssm_state_size is 0 (config missing the Mamba state dims)");
        }
        s.mamba = Some(MambaDims {
            num_heads: cfg.mamba_num_heads,
            head_dim: cfg.mamba_head_dim,
            state_size: cfg.ssm_state_size,
            n_groups: cfg.n_groups,
            conv_kernel: cfg.conv_kernel,
        });
        // One recurrent-state slot per Mamba layer (the builder/mem refines this
        // into per-request capacity).
        s.recurrent_state_slots = mamba_layers;
        if cfg.num_experts > 0 {
            apply_moe(&mut s, cfg);
        }
        Ok(s)
    }
    fn workspace_dims(&self, spec: &ArchSpec, max_tokens: usize) -> WorkspaceDims {
        build_workspace_dims(spec, max_tokens)
    }
    fn id(&self) -> PieArchId {
        PieArchId::NemotronH
    }
    fn caps(&self) -> Capabilities {
        // Recurrent state + heterogeneous layers → not graph-safe in v1.
        Capabilities { graph_safe: false, supports_compact_logits: true, ..Default::default() }
    }
}

/// Map an HF config to an `Arch`. Ports the `model_type` string matching
/// from `bound_model.cpp::bind_cuda_model` for the families wired so far
/// (llama-like + qwen3); the if/else cascade and `is_*_arch` predicates
/// collapse into this one table walk. Unknown types error clearly so the
/// seam stays honest about what's ported.
pub fn detect(cfg: &HfConfig) -> Result<Box<dyn Arch>> {
    let key = cfg.dispatch_key();
    match key.as_str() {
        // ── dense llama base (Llama-3/4 dense + the frontier dense shape) ──
        // qwen2 is llama-like-shaped (it carries QKV biases, applied by the
        // loader/forward when present) — kept as the small tokenizer-complete
        // model used for end-to-end bring-up.
        "llama" | "llama3" | "llama4" | "qwen2" => Ok(Box::new(LlamaLikeArch)),

        // ── Qwen3 / Qwen3.5 dense (per-head q/k-norm) ──
        "qwen3" | "qwen3_5" | "qwen3.5" => Ok(Box::new(Qwen3Arch)),

        // ── Qwen3.5-MoE (qk-norm + routed MoE FFN) ──
        "qwen3_moe" | "qwen3_5_moe" | "qwen2_moe" => Ok(Box::new(Qwen3MoeArch)),

        // ── DeepSeek / Kimi / GLM — MLA attention + routed MoE ──
        "deepseek" | "deepseek_v2" | "deepseek_v3" | "deepseek_v4" => {
            Ok(Box::new(DeepseekMlaArch { id: PieArchId::DeepseekV4 }))
        }
        "kimi" | "kimi_k2" | "kimi_v2" | "kimi_v3" => {
            Ok(Box::new(DeepseekMlaArch { id: PieArchId::Kimi }))
        }
        "glm" | "glm4_moe" | "glm5" | "glm_moe_dsa" => {
            Ok(Box::new(DeepseekMlaArch { id: PieArchId::Glm5 }))
        }

        // ── GPT-OSS (mxfp4 MoE, QKV bias, sliding/full alternation) ──
        "gpt_oss" | "gptoss" => Ok(Box::new(GptOssArch)),

        // ── Gemma-3/4 (sandwich norms, soft-cap, AltUp, sliding pattern) ──
        "gemma3" | "gemma3_text" | "gemma4" | "gemma4_text" => {
            Ok(Box::new(Gemma4Arch { id: PieArchId::Gemma4 }))
        }
        "gemma3n" => Ok(Box::new(Gemma4Arch { id: PieArchId::Gemma3n })),

        // ── Nemotron-H (hybrid Mamba/attention/MLP) ──
        "nemotron_h" => Ok(Box::new(NemotronHArch)),

        // ── legacy families DROPPED from cuda_new (frontier-only scope) ──
        "llama2" | "mistral" | "mistral3" | "ministral3" | "phi3" | "olmo2"
        | "olmo3" | "gemma" | "gemma2" | "mixtral" => bail!(
            "arch '{key}' is a legacy family dropped from cuda_new — frontier-only scope \
             (keep MLA / modern-MoE / Gemma-3/4 / Nemotron-H + dense Llama-3/4 · Qwen3.5)"
        ),

        other => bail!(
            "arch '{other}' not recognized by cuda_new (frontier set: llama · qwen3[_moe] · \
             deepseek/kimi/glm · gpt_oss · gemma3/4 · nemotron_h)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Llama-3-8B-ish config.
    fn llama3_8b() -> HfConfig {
        HfConfig {
            architectures: vec!["LlamaForCausalLM".to_string()],
            model_type: "llama".to_string(),
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 0, // derive: 4096/32 = 128
            vocab_size: 128256,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            sliding_window: None,
            tie_word_embeddings: false,
            attention_bias: false,
            use_qk_norm: false,
            num_experts: 0,
            num_experts_per_tok: 0,
            ..Default::default()
        }
    }

    /// Qwen3-8B-ish config (explicit head_dim, qk-norm).
    fn qwen3_8b() -> HfConfig {
        HfConfig {
            architectures: vec!["Qwen3ForCausalLM".to_string()],
            model_type: "qwen3".to_string(),
            hidden_size: 4096,
            intermediate_size: 12288,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 151936,
            max_position_embeddings: 40960,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            sliding_window: None,
            tie_word_embeddings: false,
            attention_bias: false,
            use_qk_norm: true,
            num_experts: 0,
            num_experts_per_tok: 0,
            ..Default::default()
        }
    }

    #[test]
    fn detect_llama_returns_llama_like() {
        let cfg = llama3_8b();
        let arch = detect(&cfg).expect("llama should detect");
        assert_eq!(arch.id(), PieArchId::LlamaLike);

        let spec = arch.spec(&cfg).expect("spec");
        assert_eq!(spec.id, PieArchId::LlamaLike);
        assert_eq!(spec.num_layers, 32);
        assert_eq!(spec.hidden_size, 4096);
        assert_eq!(spec.intermediate_size, 14336);
        assert_eq!(spec.num_heads, 32);
        assert_eq!(spec.num_kv_heads, 8);
        assert_eq!(spec.head_dim, 128); // derived from hidden/heads
        assert_eq!(spec.vocab_size, 128256);
        assert_eq!(spec.max_model_len, 8192);
        assert!(!spec.use_qk_norm);
        assert_eq!(spec.layer_kinds.len(), 32);
        assert!(spec.layer_kinds.iter().all(|&k| k == LayerKind::FullAttention));
        assert_eq!(spec.sliding_window, None);
        assert_eq!(spec.moe_experts, 0);
    }

    #[test]
    fn detect_qwen3_returns_qwen3() {
        let cfg = qwen3_8b();
        let arch = detect(&cfg).expect("qwen3 should detect");
        assert_eq!(arch.id(), PieArchId::Qwen3);

        let spec = arch.spec(&cfg).expect("spec");
        assert_eq!(spec.id, PieArchId::Qwen3);
        assert_eq!(spec.num_layers, 36);
        assert_eq!(spec.hidden_size, 4096);
        assert_eq!(spec.num_heads, 32);
        assert_eq!(spec.num_kv_heads, 8);
        assert_eq!(spec.head_dim, 128); // explicit
        assert!(spec.use_qk_norm, "qwen3 must enable q/k norm");
        assert_eq!(spec.layer_kinds.len(), 36);
        assert!(spec.layer_kinds.iter().all(|&k| k == LayerKind::FullAttention));
    }

    #[test]
    fn detect_unknown_errors() {
        let mut cfg = llama3_8b();
        cfg.model_type = "totally_made_up_arch".to_string();
        cfg.architectures.clear();
        let err = detect(&cfg).err().expect("unknown arch must error");
        let msg = err.to_string();
        assert!(msg.contains("totally_made_up_arch"), "msg was: {msg}");
        assert!(msg.contains("not recognized"), "msg was: {msg}");
    }

    #[test]
    fn detect_dropped_legacy_errors() {
        // gemma2/mistral/qwen2/… are legacy families dropped from the frontier
        // scope; they must error clearly rather than mis-route.
        for legacy in ["gemma2", "mistral", "phi3", "mixtral", "olmo2"] {
            let mut cfg = llama3_8b();
            cfg.model_type = legacy.to_string();
            let err = detect(&cfg).err().unwrap_or_else(|| panic!("{legacy} must error"));
            let msg = err.to_string();
            assert!(msg.contains(legacy), "msg was: {msg}");
            assert!(msg.contains("legacy"), "msg was: {msg}");
        }
    }

    #[test]
    fn dispatch_key_falls_back_to_architectures() {
        let mut cfg = llama3_8b();
        cfg.model_type.clear();
        // architectures[0] = "LlamaForCausalLM" — lower-cased, not a key,
        // so this should error rather than mis-dispatch (we only key on
        // model_type strings). Confirms the fallback path is exercised.
        let err = detect(&cfg)
            .err()
            .expect("arch_name is not a model_type key");
        assert!(err.to_string().contains("llamaforcausallm"));
    }

    #[test]
    fn workspace_dims_projects_spec() {
        let cfg = qwen3_8b();
        let arch = detect(&cfg).unwrap();
        let spec = arch.spec(&cfg).unwrap();
        let ws = arch.workspace_dims(&spec, 2048);
        assert_eq!(ws.max_tokens, 2048);
        assert_eq!(ws.hidden_size, 4096);
        assert_eq!(ws.num_heads, 32);
        assert_eq!(ws.num_kv_heads, 8);
        assert_eq!(ws.head_dim, 128);
        assert_eq!(ws.num_layers, 36);
        assert_eq!(ws.vocab_size, 151936);
        assert_eq!(ws.moe_experts, 0);
        assert_eq!(ws.recurrent_state_slots, 0);
    }

    #[test]
    fn caps_match_llama_like_model() {
        let arch = LlamaLikeArch;
        let c = arch.caps();
        assert!(c.graph_safe);
        assert!(c.supports_compact_logits);
        assert!(!c.supports_tp_greedy_argmax);
        // Qwen3 shares the shape.
        let qc = Qwen3Arch.caps();
        assert!(qc.graph_safe);
        assert!(qc.supports_compact_logits);
    }

    #[test]
    fn sliding_window_marks_layers() {
        // Use a still-supported family (llama); mistral is dropped now.
        let mut cfg = llama3_8b();
        cfg.sliding_window = Some(4096);
        let arch = detect(&cfg).unwrap();
        let spec = arch.spec(&cfg).unwrap();
        assert_eq!(spec.sliding_window, Some(4096));
        assert!(spec.layer_kinds.iter().all(|&k| k == LayerKind::SlidingAttention));
    }

    #[test]
    fn parse_from_json() {
        // A trimmed config.json deserializes straight into HfConfig.
        let json = r#"{
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": true,
            "use_qk_norm": true,
            "some_unknown_future_field": 42
        }"#;
        let cfg: HfConfig = serde_json::from_str(json).expect("parse");
        assert_eq!(cfg.model_type, "qwen3");
        assert_eq!(cfg.num_hidden_layers, 28);
        assert!(cfg.tie_word_embeddings);
        let arch = detect(&cfg).unwrap();
        assert_eq!(arch.id(), PieArchId::Qwen3);
        let spec = arch.spec(&cfg).unwrap();
        assert_eq!(spec.num_layers, 28);
        assert!(spec.tie_word_embeddings);
    }

    #[test]
    fn zero_layers_errors() {
        let mut cfg = llama3_8b();
        cfg.num_hidden_layers = 0;
        let arch = LlamaLikeArch;
        assert!(arch.spec(&cfg).is_err());
    }

    // ── frontier arch detection (B1) ───────────────────────────────────

    #[test]
    fn detect_deepseek_mla_moe() {
        let mut cfg = llama3_8b();
        cfg.model_type = "deepseek_v3".to_string();
        cfg.num_attention_heads = 128;
        cfg.q_lora_rank = 1536;
        cfg.kv_lora_rank = 512;
        cfg.qk_nope_head_dim = 128;
        cfg.qk_rope_head_dim = 64;
        cfg.v_head_dim = 128;
        cfg.num_experts = 256;
        cfg.num_experts_per_tok = 8;
        cfg.moe_intermediate_size = 2048;
        cfg.n_shared_experts = 1;
        let arch = detect(&cfg).expect("deepseek detects");
        assert_eq!(arch.id(), PieArchId::DeepseekV4);
        let s = arch.spec(&cfg).unwrap();
        assert_eq!(s.id, PieArchId::DeepseekV4);
        let mla = s.mla.expect("MLA dims present");
        assert_eq!(mla.kv_lora_rank, 512);
        assert_eq!(mla.qk_rope_head_dim, 64);
        assert_eq!(s.head_dim, 192); // qk_nope + qk_rope
        assert!(s.layer_kinds.iter().all(|&k| k == LayerKind::Mla));
        assert_eq!(s.moe_experts, 256);
        assert_eq!(s.moe_intermediate_size, 2048);
        assert_eq!(s.n_shared_experts, 1);
        assert!(!arch.caps().graph_safe, "MLA not graph-safe in v1");
    }

    #[test]
    fn detect_qwen3_moe() {
        let mut cfg = qwen3_8b();
        cfg.model_type = "qwen3_moe".to_string();
        cfg.num_experts = 128;
        cfg.num_experts_per_tok = 8;
        cfg.moe_intermediate_size = 768;
        let arch = detect(&cfg).expect("qwen3_moe detects");
        assert_eq!(arch.id(), PieArchId::Qwen3_5Moe);
        let s = arch.spec(&cfg).unwrap();
        assert_eq!(s.moe_experts, 128);
        assert_eq!(s.num_experts_per_tok, 8);
        assert_eq!(s.moe_intermediate_size, 768);
        assert!(s.use_qk_norm, "qwen-moe keeps qk-norm");
        assert!(s.mla.is_none());
    }

    #[test]
    fn detect_gpt_oss() {
        let mut cfg = llama3_8b();
        cfg.model_type = "gpt_oss".to_string();
        cfg.attention_bias = true;
        cfg.num_experts = 32;
        cfg.num_experts_per_tok = 4;
        cfg.moe_intermediate_size = 2880;
        let arch = detect(&cfg).expect("gpt_oss detects");
        assert_eq!(arch.id(), PieArchId::GptOss);
        let s = arch.spec(&cfg).unwrap();
        assert_eq!(s.moe_experts, 32);
        assert!(s.use_qkv_bias, "gpt-oss carries QKV bias");
    }

    #[test]
    fn detect_gemma4() {
        let mut cfg = llama3_8b();
        cfg.model_type = "gemma4".to_string();
        cfg.num_hidden_layers = 12;
        cfg.head_dim = 256;
        cfg.attn_logit_softcapping = Some(50.0);
        cfg.final_logit_softcapping = Some(30.0);
        cfg.altup_num_inputs = 4;
        cfg.sliding_window = Some(1024);
        cfg.sliding_window_pattern = 6;
        let arch = detect(&cfg).expect("gemma4 detects");
        assert_eq!(arch.id(), PieArchId::Gemma4);
        let s = arch.spec(&cfg).unwrap();
        assert!(s.use_qk_norm);
        assert_eq!(s.altup_num_inputs, 4);
        assert_eq!(s.attn_logit_softcap, Some(50.0));
        assert_eq!(s.final_logit_softcap, Some(30.0));
        assert_eq!(s.head_dim, 256);
        // every 6th layer global, the rest sliding.
        assert_eq!(s.layer_kinds[5], LayerKind::FullAttention);
        assert_eq!(s.layer_kinds[0], LayerKind::SlidingAttention);
        assert_eq!(s.layer_kinds[11], LayerKind::FullAttention);
    }

    #[test]
    fn detect_nemotron_h_hybrid() {
        let mut cfg = llama3_8b();
        cfg.model_type = "nemotron_h".to_string();
        cfg.num_hidden_layers = 6;
        cfg.hybrid_override_pattern = "M-M*M-".to_string();
        cfg.mamba_num_heads = 8;
        cfg.mamba_head_dim = 64;
        cfg.ssm_state_size = 128;
        cfg.n_groups = 2;
        cfg.conv_kernel = 4;
        let arch = detect(&cfg).expect("nemotron_h detects");
        assert_eq!(arch.id(), PieArchId::NemotronH);
        let s = arch.spec(&cfg).unwrap();
        let m = s.mamba.expect("Mamba dims present");
        assert_eq!(m.state_size, 128);
        assert_eq!(m.n_groups, 2);
        assert_eq!(m.conv_kernel, 4);
        use LayerKind::*;
        assert_eq!(s.layer_kinds, vec![LinearAttention, Mlp, LinearAttention, FullAttention, LinearAttention, Mlp]);
        assert_eq!(s.recurrent_state_slots, 3); // three 'M' layers
    }

    #[test]
    fn nemotron_pattern_length_mismatch_errors() {
        let mut cfg = llama3_8b();
        cfg.model_type = "nemotron_h".to_string();
        cfg.num_hidden_layers = 6;
        cfg.hybrid_override_pattern = "M-M".to_string(); // only 3, need 6
        cfg.ssm_state_size = 128;
        assert!(detect(&cfg).unwrap().spec(&cfg).is_err());
    }
}
