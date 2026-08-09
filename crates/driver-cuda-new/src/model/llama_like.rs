//! The llama-like forward's configuration and plan state — slice A of
//! gate-plan-state.
//!
//! Ports the host-pure surface of `model/llama_like/llama_like.hpp`:
//! [`LlamaLikeForwardCfg`], [`LlamaLikePlanState`], the rope-config
//! mapping, the fused-decode-post env gate, and the three graph-layout
//! functions the capture keys are built from. The prepare hook
//! (`prepare_llama_like_decode_plan`) and the forward body are later
//! slices.
//!
//! # The plan-cache seam
//!
//! `LlamaLikePlanState`'s plan slots hold flashinfer plan caches, which are
//! opaque even to the C++ — `attention_flashinfer.hpp` forward-declares
//! them precisely so their definition can live beside the kernels. The port
//! keeps them opaque the same way: the state is generic over the plan
//! handle types, and everything the driver ever asks OF a plan goes through
//! [`PlanLayouts`]. The real implementation will answer from
//! `kernels-cuda`; the parity test answers from a recorder, which is what
//! lets the branch structure be proven without a GPU.

use std::ffi::{OsStr, c_void};
use std::sync::OnceLock;

use super::config::{HfConfig, RopeScaling};

/// How many depth bands a banded fire may carry.
///
/// Mirrors `pie::driver::fire::kMaxDepthBands` (`region_plans.hpp`). The
/// parity transcript pins the array lengths that use it.
pub const MAX_DEPTH_BANDS: usize = 3;

/// Which rope launch family the checkpoint's scaling resolves to.
///
/// Mirrors `model::RopeKind` (`llama_like.hpp`), discriminants included.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum RopeKind {
    /// Pure theta-based rope — Qwen 2/3, Phi-3, Mistral.
    Standard = 0,
    /// Llama-3 smoothed-interpolation YaRN.
    YaRN = 1,
    /// Original YaRN (OLMo-3, gpt-oss): dim-index ramp plus
    /// attention-factor mscale.
    YaRNOriginal = 2,
    /// Qwen3-VL interleaved 3-axis M-RoPE.
    MRopeInterleaved = 3,
}

/// Where each sub-layer's norm sits relative to it.
///
/// Mirrors `model::NormPlacement`, discriminants included.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum NormPlacement {
    /// Norm before the sub-layer — Llama, Qwen, Mistral, Phi.
    Pre = 0,
    /// Norm after the sub-layer, then the residual add — OLMo-3.
    Post = 1,
}

/// Per-architecture knobs for the shared llama-like forward.
///
/// Mirrors `model::LlamaLikeForwardCfg` field for field. [`Default`]
/// reproduces every C++ member initializer; the parity transcript pins each
/// one.
#[derive(Debug)]
pub struct LlamaLikeForwardCfg {
    /// Per-head q/k RMSNorm before rope — Qwen3, Gemma-3, OLMo-3.
    pub use_qk_norm: bool,
    /// Bias terms on the QKV projections — Qwen-2, OLMo-3, GPT-OSS.
    pub use_qkv_bias: bool,
    /// Pre- or post-norm layer shape.
    pub norm_placement: NormPlacement,
    /// Which rope launch family the forward applies.
    pub rope_kind: RopeKind,
    /// YaRN context-extension factor. Consumed only under
    /// [`RopeKind::YaRN`] / [`RopeKind::YaRNOriginal`].
    pub yarn_factor: f32,
    /// Llama-3 YaRN low-frequency corner.
    pub yarn_low_freq_factor: f32,
    /// Llama-3 YaRN high-frequency corner.
    pub yarn_high_freq_factor: f32,
    /// The pre-extension maximum position.
    pub yarn_original_max_position: i32,
    /// Original-YaRN fast-dimension ramp bound.
    pub yarn_beta_fast: f32,
    /// Original-YaRN slow-dimension ramp bound.
    pub yarn_beta_slow: f32,
    /// Original-YaRN attention magnitude factor.
    pub yarn_attention_factor: f32,
    /// Sliding-window width; `-1` means full causal for every layer.
    pub sliding_window: i32,
    /// Per-layer `window_left` override; empty defers to the scalar.
    pub per_layer_window_left: Vec<i32>,
    /// Route decode batches through the prefill kernel — models whose GQA
    /// group size is outside flashinfer's decode dispatch table.
    pub force_prefill_path: bool,
    /// Use the XQA decode kernel where its preconditions hold.
    pub use_xqa_decode: bool,
    /// Plan the decode path in CUDA-graph mode.
    pub decode_plan_cuda_graph: bool,
    /// Plan pure-decode fires through the prefill planner.
    pub use_prefill_decode_plan: bool,
    /// Full-attention variant floor on request count for the
    /// prefill-decode plan.
    pub prefill_decode_full_attention_min_requests: i32,
    /// Full-attention variant floor on KV pages for the prefill-decode
    /// plan.
    pub prefill_decode_full_attention_min_kv_pages: i32,
    /// KV-page floor below which the prefill-decode plan is not used.
    pub prefill_decode_min_kv_pages: i32,
    /// Tensor-parallel world size; `1` keeps the single-GPU forward.
    pub tp_size: i32,
    /// The TP communicator; must be non-null whenever `tp_size > 1`.
    /// An opaque pointer until the distributed layer is ported.
    pub tp_comm: *mut c_void,
    /// Whether this rank publishes logits. TP followers skip them.
    pub emit_logits: bool,
    /// Per-fire chunked lm-head argmax width; `0` disables (§20.37).
    pub logits_argmax_chunk_tokens: i32,
    /// M-RoPE time-axis section width. Consumed only under
    /// [`RopeKind::MRopeInterleaved`].
    pub mrope_section_t: i32,
    /// M-RoPE height-axis section width.
    pub mrope_section_h: i32,
    /// M-RoPE width-axis section width.
    pub mrope_section_w: i32,
}

impl Default for LlamaLikeForwardCfg {
    fn default() -> Self {
        Self {
            use_qk_norm: false,
            use_qkv_bias: false,
            norm_placement: NormPlacement::Pre,
            rope_kind: RopeKind::Standard,
            yarn_factor: 1.0,
            yarn_low_freq_factor: 1.0,
            yarn_high_freq_factor: 4.0,
            yarn_original_max_position: 8192,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            yarn_attention_factor: 1.0,
            sliding_window: -1,
            per_layer_window_left: Vec::new(),
            force_prefill_path: false,
            use_xqa_decode: false,
            decode_plan_cuda_graph: true,
            use_prefill_decode_plan: false,
            prefill_decode_full_attention_min_requests: 0,
            prefill_decode_full_attention_min_kv_pages: 0,
            prefill_decode_min_kv_pages: 0,
            tp_size: 1,
            tp_comm: std::ptr::null_mut(),
            emit_logits: true,
            logits_argmax_chunk_tokens: 0,
            mrope_section_t: 0,
            mrope_section_h: 0,
            mrope_section_w: 0,
        }
    }
}

/// Persistent decode-plan state, refreshed by the prepare hook outside any
/// capture region and read — never written — by the forward body.
///
/// Mirrors `model::LlamaLikePlanState` field for field. `D` and `P` are the
/// decode- and prefill-plan handle types (see [`PlanLayouts`]); `L` is the
/// staged-lora handle, opaque until the lora slice lands.
#[derive(Debug)]
pub struct LlamaLikePlanState<D, P, L = ()> {
    /// The pure-decode flashinfer plan.
    pub decode_plan: Option<D>,
    /// The reusable causal prefill plan.
    pub prefill_plan: Option<P>,
    /// The decode-shaped prefill plan (`use_prefill_decode_plan`
    /// deployments).
    pub prefill_decode_plan: Option<P>,
    /// The custom-mask pure-decode fires' OWN plan slot — the supergraph
    /// axiom S3: an arm may not share a mutable plan slot with a foreign
    /// fire class.
    pub mask_decode_plan: Option<P>,
    /// STRUCTURAL S-2: the depth union's prefix decode plan, planned
    /// against the secondary workspace.
    pub depth_prefix_decode_plan: Option<D>,
    /// V2 rung ④ Act 1: one prefix decode plan per distinct-k band.
    pub depth_band_plans: [Option<D>; MAX_DEPTH_BANDS],
    /// Prefill-family band plans (force-prefill / prefill-decode
    /// deployments).
    pub depth_band_prefill_plans: [Option<P>; 3],
    /// Each band's layer depth, deepest first.
    pub depth_band_k: [u32; 3],
    /// Each band's request-row count.
    pub depth_band_rows: [u32; 3],
    /// How many bands are live; `0` = unbanded fire.
    pub depth_band_count: u32,
    /// NO-DEMOTION: the plain-decode middle's decode plan of a mixed
    /// 3-way fire.
    pub mixed_mid_decode_plan: Option<D>,
    /// First request of the plain-decode middle; `-1` = no middle.
    pub mixed_mid_start: i32,
    /// NS-2: the request index this fire's attention splits at; `-1` =
    /// fire-level plans.
    pub spatial_mask_split: i32,
    /// M-2: the split's token-row offset; equals `spatial_mask_split` on
    /// pure-decode fires.
    pub spatial_mask_row_split: i32,
    /// The body dispatches through `prefill_plan`.
    pub use_prefill_plan: bool,
    /// The body dispatches through `prefill_decode_plan`.
    pub use_prefill_decode_plan: bool,
    /// The body dispatches through `mask_decode_plan`.
    pub use_mask_decode_plan: bool,
    /// Non-zero when the prefill plan was built for the FA2
    /// score-capturing dispatch — a plan-time choice the body only honours.
    pub prefill_score_window: u32,
    /// Lora campaign 3a: the fire's pre-staged lora state.
    pub lora_staged: Option<L>,
    /// The table `lora_staged` was staged from. An opaque pointer until
    /// the lora slice lands.
    pub lora_staged_table: *const c_void,
    /// The body dispatches through the XQA decode kernel.
    pub use_xqa_decode: bool,
    /// XQA's planned per-sequence page bound.
    pub xqa_max_pages_per_seq: i32,
    /// Host copy of the prefill-decode plan's qo indptr.
    pub prefill_decode_qo_indptr_h: Vec<u32>,
}

impl<D, P, L> Default for LlamaLikePlanState<D, P, L> {
    fn default() -> Self {
        Self {
            decode_plan: None,
            prefill_plan: None,
            prefill_decode_plan: None,
            mask_decode_plan: None,
            depth_prefix_decode_plan: None,
            depth_band_plans: [None, None, None],
            depth_band_prefill_plans: [None, None, None],
            depth_band_k: [0; 3],
            depth_band_rows: [0; 3],
            depth_band_count: 0,
            mixed_mid_decode_plan: None,
            mixed_mid_start: -1,
            spatial_mask_split: -1,
            spatial_mask_row_split: -1,
            use_prefill_plan: false,
            use_prefill_decode_plan: false,
            use_mask_decode_plan: false,
            prefill_score_window: 0,
            lora_staged: None,
            lora_staged_table: std::ptr::null(),
            use_xqa_decode: false,
            xqa_max_pages_per_seq: 0,
            prefill_decode_qo_indptr_h: Vec::new(),
        }
    }
}

/// What the driver asks of a flashinfer plan cache.
///
/// The C++ reaches these as free functions in `kernels::attn`; the real
/// Rust implementation will answer over FFI from `kernels-cuda`, and the
/// parity test answers from a recorder. A trait rather than direct calls
/// for the same reason `page_mask` has `MaskOps`: the logic under test is
/// the DRIVER's, and the boundary is where the proof swaps in its probe.
pub trait PlanLayouts {
    /// The decode-plan handle type.
    type DecodePlan;
    /// The prefill-plan handle type.
    type PrefillPlan;

    /// `kernels::attn::decode_plan_graph_layout`.
    fn decode_plan_graph_layout(&self, plan: &Self::DecodePlan) -> u32;
    /// `kernels::attn::prefill_plan_graph_layout`.
    fn prefill_plan_graph_layout(&self, plan: &Self::PrefillPlan) -> u32;
    /// `kernels::attn::prefill_plan_graph_capturable`.
    fn prefill_plan_graph_capturable(&self, plan: &Self::PrefillPlan) -> bool;
    /// `kernels::attn::xqa_decode_graph_layout` — pure of the page bound.
    fn xqa_decode_graph_layout(&self, max_pages_per_seq: i32) -> u8;
}

/// splitmix64's finalizer — the mix the NS-3 spatial layout key uses.
fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Compact graph-layout class for the state's decode-side dispatch.
///
/// Ports `llama_like_decode_graph_layout`, branch for branch: the NS-3
/// spatial-split key first (both plans' grids and the split join the
/// layout), then XQA, then the three prefill-shaped slots in dispatch
/// order, then the plain decode plan.
pub fn llama_like_decode_graph_layout<O: PlanLayouts, L>(
    ops: &O,
    state: &LlamaLikePlanState<O::DecodePlan, O::PrefillPlan, L>,
) -> u32 {
    if state.spatial_mask_split >= 0
        && state.use_mask_decode_plan
        && let Some(mask) = &state.mask_decode_plan
    {
        let mut h = splitmix(0x5350_414c ^ state.spatial_mask_split as u64);
        h = splitmix(h ^ u64::from(ops.prefill_plan_graph_layout(mask)));
        if state.spatial_mask_split > 0
            && let Some(decode) = &state.decode_plan
        {
            h = splitmix(h ^ u64::from(ops.decode_plan_graph_layout(decode)));
        }
        return (h & 0x00ff_ffff) as u32;
    }
    if state.use_xqa_decode {
        return u32::from(ops.xqa_decode_graph_layout(state.xqa_max_pages_per_seq));
    }
    if state.use_prefill_decode_plan
        && let Some(plan) = &state.prefill_decode_plan
    {
        return ops.prefill_plan_graph_layout(plan);
    }
    if state.use_prefill_plan
        && let Some(plan) = &state.prefill_plan
    {
        return ops.prefill_plan_graph_layout(plan);
    }
    if state.use_mask_decode_plan
        && let Some(plan) = &state.mask_decode_plan
    {
        return ops.prefill_plan_graph_layout(plan);
    }
    match &state.decode_plan {
        None => 0,
        Some(plan) => ops.decode_plan_graph_layout(plan),
    }
}

/// The supergraph UNION key's layout (S3): both arms' kernel
/// configurations, mixed so the pair cannot alias a plain layout.
///
/// Ports `llama_like_supergraph_graph_layout`.
pub fn llama_like_supergraph_graph_layout<O: PlanLayouts, L>(
    ops: &O,
    state: &LlamaLikePlanState<O::DecodePlan, O::PrefillPlan, L>,
) -> u32 {
    let decode_side = if state.use_xqa_decode {
        u32::from(ops.xqa_decode_graph_layout(state.xqa_max_pages_per_seq))
    } else {
        state
            .decode_plan
            .as_ref()
            .map_or(0, |p| ops.decode_plan_graph_layout(p))
    };
    let mask_side = state
        .mask_decode_plan
        .as_ref()
        .map_or(0, |p| ops.prefill_plan_graph_layout(p));
    let mut h = decode_side.wrapping_add(0x9e37_79b9);
    h ^= mask_side
        .wrapping_add(0x85eb_ca6b)
        .wrapping_add(h << 6)
        .wrapping_add(h >> 2);
    h
}

/// True when the fire carries a prefill the executor may capture.
///
/// Ports `llama_like_prefill_graph_capturable`: only the true prefill plan
/// answers — the decode-shaped prefill plan is admitted by the pure-decode
/// rules, and conflating the two would hide which rule let a wave through.
pub fn llama_like_prefill_graph_capturable<O: PlanLayouts, L>(
    ops: &O,
    state: &LlamaLikePlanState<O::DecodePlan, O::PrefillPlan, L>,
) -> bool {
    if state.use_prefill_plan
        && let Some(plan) = &state.prefill_plan
    {
        return ops.prefill_plan_graph_capturable(plan);
    }
    false
}

/// The `PIE_CUDA_DECODE_FUSED_POST` kill switch's decision, as a pure
/// function of the variable's value.
///
/// The C++ gate reads the environment once into a function-local static;
/// [`decode_fused_post_enabled`] is that cached form, and this is the
/// testable one. Unset and empty both mean ON; anything else is ON unless
/// its first byte is `'0'` — the C++ checks `v[0] != '0'`, one byte, so
/// `"01"` is OFF and `"10"` is ON.
pub fn decode_fused_post_enabled_from(value: Option<&OsStr>) -> bool {
    match value {
        None => true,
        Some(v) => match v.as_encoded_bytes().first() {
            None => true,
            Some(b) => *b != b'0',
        },
    }
}

/// `PIE_CUDA_DECODE_FUSED_POST`, read once and cached — the shape the
/// forward body and the declared executor's peephole both consult.
///
/// Ports `decode_fused_post_enabled`.
pub fn decode_fused_post_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        decode_fused_post_enabled_from(
            std::env::var_os("PIE_CUDA_DECODE_FUSED_POST").as_deref(),
        )
    })
}

/// Map the checkpoint's rope-scaling kind onto the driver's launch family.
///
/// Ports `rope_kind_from_hf_config`: Llama3-style frequency scaling maps
/// to YaRN; `OriginalYaRN` keeps HuggingFace's original formulation.
pub fn rope_kind_from_hf_config(hf: &HfConfig) -> RopeKind {
    match hf.rope_scaling_kind {
        RopeScaling::Llama3 => RopeKind::YaRN,
        RopeScaling::OriginalYarn => RopeKind::YaRNOriginal,
        RopeScaling::None => RopeKind::Standard,
    }
}

/// Populate the rope block of a [`LlamaLikeForwardCfg`] from the HF config
/// in one place — every arch that builds one pulls the same eight fields.
///
/// Ports `apply_rope_config`.
pub fn apply_rope_config(fwd_cfg: &mut LlamaLikeForwardCfg, hf: &HfConfig) {
    fwd_cfg.rope_kind = rope_kind_from_hf_config(hf);
    fwd_cfg.yarn_factor = hf.rope_factor;
    fwd_cfg.yarn_low_freq_factor = hf.rope_low_freq_factor;
    fwd_cfg.yarn_high_freq_factor = hf.rope_high_freq_factor;
    fwd_cfg.yarn_original_max_position = hf.rope_original_max_position;
    fwd_cfg.yarn_beta_fast = hf.rope_beta_fast;
    fwd_cfg.yarn_beta_slow = hf.rope_beta_slow;
    fwd_cfg.yarn_attention_factor = hf.rope_attention_factor;
}

// ───────────────────────────────────────────────────────────────────────────
// Slice B: the prepare hook.
// ───────────────────────────────────────────────────────────────────────────

use crate::launch::AttentionWorkspaceView;

/// What the prepare hook asks of the flashinfer planner boundary.
///
/// The C++ reaches these as free functions in `kernels::attn` plus two
/// lazy workspace singletons in `llama_like.cpp`'s anonymous namespace; the
/// singletons are driver-global state, which is why they live behind the
/// ops rather than as parameters. The decode planner's `stream` and
/// `window_left` are omitted: the prepare hook passes `nullptr` and the
/// default `-1` at every site, so a trait slot would be a constant.
pub trait PlannerOps {
    /// The decode-plan handle.
    type DecodePlan;
    /// The prefill-plan handle.
    type PrefillPlan;

    /// `kernels::attn::make_decode_plan`.
    fn make_decode_plan(&mut self) -> Self::DecodePlan;
    /// `kernels::attn::make_prefill_plan`.
    fn make_prefill_plan(&mut self) -> Self::PrefillPlan;
    /// `kernels::attn::plan_attention_flashinfer_decode`. Slices carry the
    /// C++ pointer semantics: only the first `num_requests + 1` entries of
    /// `kv_page_indptr_h` are the plan's input.
    #[allow(clippy::too_many_arguments)]
    fn plan_decode(
        &mut self,
        plan: &mut Self::DecodePlan,
        kv_page_indptr_h: &[u32],
        num_requests: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        workspace: AttentionWorkspaceView,
        enable_cuda_graph: bool,
        full_attention_variant: bool,
        hnd_layout: bool,
    );
    /// `kernels::attn::plan_attention_flashinfer_prefill_bf16`, the full
    /// nineteen-slot signature.
    #[allow(clippy::too_many_arguments)]
    fn plan_prefill(
        &mut self,
        plan: &mut Self::PrefillPlan,
        qo_indptr_h: &[u32],
        kv_page_indptr_h: &[u32],
        kv_last_page_lens_h: &[u32],
        total_tokens: i32,
        num_requests: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        workspace: AttentionWorkspaceView,
        enable_cuda_graph: bool,
        window_left: i32,
        full_attention_variant: bool,
        hnd_layout: bool,
        causal_mask: bool,
        custom_mask: bool,
        wants_prefill_score: bool,
    );
    /// `kernels::attn::xqa_decode_page_bucket`.
    fn xqa_decode_page_bucket(&mut self, max_pages_per_seq: i32) -> i32;
    /// The mixed fire's dedicated suffix-plan workspace
    /// (`spatial_suffix_attn_ws`) — two same-family plans must not share
    /// one workspace's scheduling buffers.
    fn spatial_suffix_ws_view(&mut self) -> AttentionWorkspaceView;
    /// Band slot `i`'s dedicated workspace (`depth_band_attn_ws_public`).
    fn depth_band_ws_view(&mut self, band: usize) -> AttentionWorkspaceView;
}

/// The KV-cache geometry the prepare hook reads.
///
/// The C++ takes `KvCache&` and calls three accessors; the live cache port
/// answers this from its layout (`KvCache::plan_geom`), and carrying the
/// three answers instead of the object keeps the hook free of the cache's
/// elastic-pool generic.
#[derive(Debug, Clone, Copy)]
pub struct KvGeom {
    /// `KvCache::page_size()`.
    pub page_size: i32,
    /// `KvCache::hnd_layout()`.
    pub hnd_layout: bool,
    /// `KvCache::format().is_native_bf16()`.
    pub native_bf16: bool,
}

/// The prepare hook's four environment gates, read once per process in the
/// C++ (function-local statics) and carried as a value here so a test can
/// sweep them without owning the process environment.
#[derive(Debug, Clone, Copy)]
pub struct PrepareGates {
    /// `PIE_SPATIAL_MASK` — unset or first byte not `'0'` means armed.
    pub spatial_mask_on: bool,
    /// `PIE_MIXED_MID` — same shape; the NO-DEMOTION middle's A/B switch.
    pub mixed_mid_on: bool,
    /// `PIE_PREFILL_GRAPH_PLAN` — set, non-empty, first byte not `'0'`.
    /// Defaults OFF; see the C++'s measurement log for why it stays off.
    pub prefill_graph_plan: bool,
    /// `PIE_REGION_TRACE` — set at all.
    pub region_trace: bool,
}

impl PrepareGates {
    /// Read all four variables, with each gate's own parsing shape.
    #[must_use]
    pub fn from_env() -> Self {
        let on_unless_zero = |name: &str| match std::env::var_os(name) {
            None => true,
            Some(v) => v.as_encoded_bytes().first() != Some(&b'0'),
        };
        let off_unless_set = |name: &str| match std::env::var_os(name) {
            None => false,
            Some(v) => {
                let b = v.as_encoded_bytes();
                !b.is_empty() && b[0] != b'0'
            }
        };
        Self {
            spatial_mask_on: on_unless_zero("PIE_SPATIAL_MASK"),
            mixed_mid_on: on_unless_zero("PIE_MIXED_MID"),
            prefill_graph_plan: off_unless_set("PIE_PREFILL_GRAPH_PLAN"),
            region_trace: std::env::var_os("PIE_REGION_TRACE").is_some(),
        }
    }
}

/// The prepare call's optional tail — the C++ signature's eight defaulted
/// parameters, as one value so the recursive prefix build can spell "all
/// defaults" without eight underscores.
#[derive(Debug, Clone, Copy)]
pub struct PrepareParams<'a> {
    /// Non-zero when the fire's PTIR programs read `AttnScore`.
    pub attn_score_window: u32,
    /// NS-2: the planned unmasked wire-row prefix; `u32::MAX` = no split.
    pub unmasked_prefix_rows: u32,
    /// NS-2: resolver-threaded suffix page counts, when composed.
    pub mask_suffix_page_counts_h: Option<&'a [u32]>,
    /// NS-2: resolver-threaded suffix last-page lengths.
    pub mask_suffix_last_lens_h: Option<&'a [u32]>,
    /// S-2: the depth union's request split; `u32::MAX` = uniform fire.
    pub full_depth_rows: u32,
    /// ④: per-band layer depths, deepest first.
    pub depth_band_k: &'a [u32],
    /// ④: per-band request-row counts.
    pub depth_band_rows: &'a [u32],
    /// ④: how many bands; `0` = unbanded.
    pub depth_band_count: u32,
}

impl Default for PrepareParams<'_> {
    fn default() -> Self {
        Self {
            attn_score_window: 0,
            unmasked_prefix_rows: u32::MAX,
            mask_suffix_page_counts_h: None,
            mask_suffix_last_lens_h: None,
            full_depth_rows: u32::MAX,
            depth_band_k: &[],
            depth_band_rows: &[],
            depth_band_count: 0,
        }
    }
}

/// `decode_full_attention_variant_enabled` — a constant `true` in the C++.
const DECODE_FULL_ATTENTION_VARIANT: bool = true;

/// The kvpp SENTRY: host plan inputs are validated before any planner
/// consumes them, and a violation dumps the arrays and refuses the fire —
/// the heisenbug becomes a self-documenting event instead of a deep
/// planner assert. Ports `kvpp_sentry`, stderr included: the whole point
/// is the UNCONDITIONAL dump at the next occurrence.
#[allow(clippy::print_stderr)]
fn kvpp_sentry(what: &str, qo_indptr_h: &[u32], kv_page_indptr_h: &[u32], num_requests: i32) {
    let r_max = usize::try_from(num_requests.max(0)).unwrap_or(0);
    for r in 0..r_max {
        let qo_bad = qo_indptr_h.len() > r + 1 && qo_indptr_h[r + 1] < qo_indptr_h[r];
        let kv_bad =
            kv_page_indptr_h.len() > r + 1 && kv_page_indptr_h[r + 1] < kv_page_indptr_h[r];
        if !qo_bad && !kv_bad {
            continue;
        }
        eprintln!(
            "[kvpp-sentry] {what}: NON-MONOTONE host plan input at lane {r} of {num_requests}"
        );
        eprintln!("  qo={qo_indptr_h:?}");
        eprintln!("  kvpp={kv_page_indptr_h:?}");
    }
}

/// Refresh the decode plan for the current fire — the prepare hook, run
/// OUTSIDE any capture region so the captured body replays against plans
/// it only reads.
///
/// Ports `prepare_llama_like_decode_plan` branch for branch: the NS-2
/// spatial split with its RECURSIVE prefix build, the M-2 mixed fire with
/// the NO-DEMOTION middle, fire-level custom masks, XQA, the single
/// global-window prefill plan, force-prefill, the prefill-decode plan with
/// its page floors, the plain decode plan, the S-2 depth prefix, and the
/// ④ banded-depth stamps. Shape guards DECLINE (fall through or leave a
/// slot null) rather than throw, exactly as the C++ does.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
// stderr is the C++'s own trace channel (`PIE_REGION_TRACE`); routing it
// anywhere else would change what the operator sees.
#[allow(clippy::print_stderr)]
pub fn prepare_llama_like_decode_plan<O: PlannerOps, L>(
    ops: &mut O,
    gates: &PrepareGates,
    state: &mut LlamaLikePlanState<O::DecodePlan, O::PrefillPlan, L>,
    main_ws: AttentionWorkspaceView,
    cache: &KvGeom,
    cfg: &HfConfig,
    fwd_cfg: &LlamaLikeForwardCfg,
    qo_indptr_h: &[u32],
    kv_page_indptr_h: &[u32],
    kv_last_page_lens_h: &[u32],
    total_tokens: i32,
    num_requests: i32,
    is_pure_decode: bool,
    have_custom_mask: bool,
    params: &PrepareParams<'_>,
) {
    state.use_xqa_decode = false;
    state.xqa_max_pages_per_seq = 0;
    state.use_prefill_plan = false;
    state.use_prefill_decode_plan = false;
    state.use_mask_decode_plan = false;
    state.prefill_score_window = 0;
    state.spatial_mask_split = -1;
    state.spatial_mask_row_split = -1;

    let r_usize = usize::try_from(num_requests.max(0)).unwrap_or(0);

    // NS-2: a masked pure-decode fire with a planned 0 <= prefix < R
    // builds BOTH plans — the decode side over the prefix (a recursive
    // prepare over the same host CSRs truncated to `split`), then the mask
    // plan over the REBASED suffix.
    if gates.spatial_mask_on
        && have_custom_mask
        && is_pure_decode
        && params.unmasked_prefix_rows != u32::MAX
        && i64::from(params.unmasked_prefix_rows) < i64::from(num_requests)
        && cfg.head_dim == cfg.head_dim_kernel
        && !(fwd_cfg.use_xqa_decode && cache.native_bf16 && !cache.hnd_layout)
    {
        let split = i32::try_from(params.unmasked_prefix_rows).unwrap_or(i32::MAX);
        let split_u = usize::try_from(split).unwrap_or(0);
        let rs = num_requests - split;
        let rs_u = usize::try_from(rs.max(0)).unwrap_or(0);
        if split > 0 {
            prepare_llama_like_decode_plan(
                ops,
                gates,
                state,
                main_ws,
                cache,
                cfg,
                fwd_cfg,
                qo_indptr_h,
                kv_page_indptr_h,
                kv_last_page_lens_h,
                split,
                split,
                /*is_pure_decode=*/ true,
                /*have_custom_mask=*/ false,
                &PrepareParams {
                    attn_score_window: params.attn_score_window,
                    ..PrepareParams::default()
                },
            );
        }
        // The recursion reset the state flags; restore the split AFTER it.
        let qo_suffix: Vec<u32> = (0..=u32::try_from(rs.max(0)).unwrap_or(0)).collect();
        let mut kvpp_suffix = vec![0u32; rs_u + 1];
        if let Some(counts) = params.mask_suffix_page_counts_h {
            for i in 0..rs_u {
                kvpp_suffix[i + 1] = kvpp_suffix[i] + counts[i];
            }
        } else {
            let page_base = kv_page_indptr_h[split_u];
            for i in 0..=rs_u {
                kvpp_suffix[i] = kv_page_indptr_h[split_u + i] - page_base;
            }
        }
        if state.mask_decode_plan.is_none() {
            state.mask_decode_plan = Some(ops.make_prefill_plan());
        }
        let t = if fwd_cfg.tp_size > 0 { fwd_cfg.tp_size } else { 1 };
        let suffix_ws = ops.spatial_suffix_ws_view();
        let plan = state.mask_decode_plan.as_mut().expect("just made");
        ops.plan_prefill(
            plan,
            &qo_suffix,
            &kvpp_suffix,
            params
                .mask_suffix_last_lens_h
                .unwrap_or(&kv_last_page_lens_h[split_u..]),
            rs,
            rs,
            cfg.num_attention_heads / t,
            cfg.num_key_value_heads / t,
            cfg.head_dim_kernel,
            cache.page_size,
            suffix_ws,
            fwd_cfg.decode_plan_cuda_graph,
            /*window_left=*/ -1,
            /*full_attention_variant=*/ false,
            cache.hnd_layout,
            /*causal_mask=*/ false,
            /*custom_mask=*/ true,
            /*wants_prefill_score=*/ false,
        );
        state.use_mask_decode_plan = true;
        state.spatial_mask_split = split;
        state.spatial_mask_row_split = split;
        return;
    }
    // THE MIXED FIRE (M-2): a prefill-shaped masked fire with a planned
    // unmasked prefix. The planned word counts TOKEN ROWS; the request
    // split derives from the host qo indptr. Shape guards decline the
    // split rather than throw.
    if gates.spatial_mask_on
        && have_custom_mask
        && !is_pure_decode
        && params.unmasked_prefix_rows != u32::MAX
        && params.unmasked_prefix_rows > 0
        && i64::from(params.unmasked_prefix_rows) < i64::from(total_tokens)
        && cfg.head_dim == cfg.head_dim_kernel
        && fwd_cfg.per_layer_window_left.is_empty()
    {
        let split_req = if i64::from(params.unmasked_prefix_rows) < i64::from(num_requests) {
            i32::try_from(params.unmasked_prefix_rows).unwrap_or(-1)
        } else {
            -1
        };
        let mut suffix_decode = split_req > 0 && split_req < num_requests;
        if suffix_decode {
            let s = usize::try_from(split_req).unwrap_or(0);
            for r in s..r_usize {
                if qo_indptr_h[r + 1] - qo_indptr_h[r] != 1 {
                    suffix_decode = false;
                    break;
                }
            }
        }
        if suffix_decode {
            let split_u = usize::try_from(split_req).unwrap_or(0);
            let rs = num_requests - split_req;
            let rs_u = usize::try_from(rs.max(0)).unwrap_or(0);
            let t = if fwd_cfg.tp_size > 0 { fwd_cfg.tp_size } else { 1 };
            let num_q_heads_local = cfg.num_attention_heads / t;
            let num_kv_heads_local = cfg.num_key_value_heads / t;
            if state.prefill_plan.is_none() {
                state.prefill_plan = Some(ops.make_prefill_plan());
            }
            {
                let plan = state.prefill_plan.as_mut().expect("just made");
                ops.plan_prefill(
                    plan,
                    qo_indptr_h,
                    kv_page_indptr_h,
                    kv_last_page_lens_h,
                    i32::try_from(qo_indptr_h[split_u]).unwrap_or(0),
                    split_req,
                    num_q_heads_local,
                    num_kv_heads_local,
                    cfg.head_dim_kernel,
                    cache.page_size,
                    main_ws,
                    fwd_cfg.decode_plan_cuda_graph,
                    fwd_cfg.sliding_window,
                    /*full_attention_variant=*/ false,
                    cache.hnd_layout,
                    /*causal_mask=*/ true,
                    /*custom_mask=*/ false,
                    /*wants_prefill_score=*/ false,
                );
            }
            state.use_prefill_plan = true;
            // NO-DEMOTION: the plain-decode middle [P, split_req) gets the
            // decode kernel instead of demoting to the causal prefill.
            state.mixed_mid_decode_plan = None;
            state.mixed_mid_start = -1;
            {
                let mut p = split_req;
                for r in 0..usize::try_from(split_req).unwrap_or(0) {
                    if qo_indptr_h[r + 1] - qo_indptr_h[r] == 1 {
                        p = i32::try_from(r).unwrap_or(i32::MAX);
                        break;
                    }
                }
                let mid = split_req - p;
                if gates.mixed_mid_on
                    && mid > 0
                    && p > 0
                    && !fwd_cfg.force_prefill_path
                    && !fwd_cfg.use_prefill_decode_plan
                {
                    let p_u = usize::try_from(p).unwrap_or(0);
                    let mid_u = usize::try_from(mid).unwrap_or(0);
                    let mid_base = kv_page_indptr_h[p_u];
                    let kvpp_mid: Vec<u32> = (0..=mid_u)
                        .map(|i| kv_page_indptr_h[p_u + i] - mid_base)
                        .collect();
                    if state.mixed_mid_decode_plan.is_none() {
                        state.mixed_mid_decode_plan = Some(ops.make_decode_plan());
                    }
                    let plan = state.mixed_mid_decode_plan.as_mut().expect("just made");
                    ops.plan_decode(
                        plan,
                        &kvpp_mid,
                        mid,
                        num_q_heads_local,
                        num_kv_heads_local,
                        cfg.head_dim_kernel,
                        cache.page_size,
                        main_ws,
                        fwd_cfg.decode_plan_cuda_graph,
                        DECODE_FULL_ATTENTION_VARIANT
                            && fwd_cfg.sliding_window < 0
                            && fwd_cfg.per_layer_window_left.is_empty(),
                        cache.hnd_layout,
                    );
                    state.mixed_mid_start = p;
                    // Re-plan the prefix CAUSAL to the prefill lanes only.
                    let plan = state.prefill_plan.as_mut().expect("planned above");
                    ops.plan_prefill(
                        plan,
                        qo_indptr_h,
                        kv_page_indptr_h,
                        kv_last_page_lens_h,
                        i32::try_from(qo_indptr_h[p_u]).unwrap_or(0),
                        p,
                        num_q_heads_local,
                        num_kv_heads_local,
                        cfg.head_dim_kernel,
                        cache.page_size,
                        main_ws,
                        fwd_cfg.decode_plan_cuda_graph,
                        fwd_cfg.sliding_window,
                        /*full_attention_variant=*/ false,
                        cache.hnd_layout,
                        /*causal_mask=*/ true,
                        /*custom_mask=*/ false,
                        /*wants_prefill_score=*/ false,
                    );
                }
            }
            // The suffix mask plan: identity qo over the 1-token rows.
            let qo_suffix: Vec<u32> = (0..=u32::try_from(rs.max(0)).unwrap_or(0)).collect();
            let mut kvpp_suffix = vec![0u32; rs_u + 1];
            if let Some(counts) = params.mask_suffix_page_counts_h {
                for i in 0..rs_u {
                    kvpp_suffix[i + 1] = kvpp_suffix[i] + counts[i];
                }
            } else {
                let page_base = kv_page_indptr_h[split_u];
                for i in 0..=rs_u {
                    kvpp_suffix[i] = kv_page_indptr_h[split_u + i] - page_base;
                }
            }
            if state.mask_decode_plan.is_none() {
                state.mask_decode_plan = Some(ops.make_prefill_plan());
            }
            let suffix_ws = ops.spatial_suffix_ws_view();
            let plan = state.mask_decode_plan.as_mut().expect("just made");
            ops.plan_prefill(
                plan,
                &qo_suffix,
                &kvpp_suffix,
                params
                    .mask_suffix_last_lens_h
                    .unwrap_or(&kv_last_page_lens_h[split_u..]),
                rs,
                rs,
                num_q_heads_local,
                num_kv_heads_local,
                cfg.head_dim_kernel,
                cache.page_size,
                suffix_ws,
                fwd_cfg.decode_plan_cuda_graph,
                /*window_left=*/ -1,
                /*full_attention_variant=*/ false,
                cache.hnd_layout,
                /*causal_mask=*/ false,
                /*custom_mask=*/ true,
                /*wants_prefill_score=*/ false,
            );
            state.use_mask_decode_plan = true;
            state.spatial_mask_split = split_req;
            state.spatial_mask_row_split = i32::try_from(qo_indptr_h[split_u]).unwrap_or(0);
            return;
        }
    }
    if have_custom_mask {
        // Pure-decode custom-mask fires plan into their DEDICATED slot;
        // prefill-shaped custom-mask fires keep the prefill slot.
        let t = if fwd_cfg.tp_size > 0 { fwd_cfg.tp_size } else { 1 };
        let num_q_heads_local = cfg.num_attention_heads / t;
        let num_kv_heads_local = cfg.num_key_value_heads / t;
        let slot = if is_pure_decode {
            &mut state.mask_decode_plan
        } else {
            &mut state.prefill_plan
        };
        if slot.is_none() {
            *slot = Some(ops.make_prefill_plan());
        }
        let plan = slot.as_mut().expect("just made");
        ops.plan_prefill(
            plan,
            qo_indptr_h,
            kv_page_indptr_h,
            kv_last_page_lens_h,
            total_tokens,
            num_requests,
            num_q_heads_local,
            num_kv_heads_local,
            cfg.head_dim_kernel,
            cache.page_size,
            main_ws,
            fwd_cfg.decode_plan_cuda_graph,
            /*window_left=*/ -1,
            /*full_attention_variant=*/ false,
            cache.hnd_layout,
            /*causal_mask=*/ false,
            /*custom_mask=*/ true,
            /*wants_prefill_score=*/ false,
        );
        if is_pure_decode {
            state.use_mask_decode_plan = true;
        } else {
            state.use_prefill_plan = true;
        }
        return;
    }
    if is_pure_decode && fwd_cfg.use_xqa_decode && cache.native_bf16 && !cache.hnd_layout {
        let mut max_pages: i32 = 1;
        for r in 0..r_usize {
            let pages = i32::try_from(kv_page_indptr_h[r + 1] - kv_page_indptr_h[r]).unwrap_or(0);
            max_pages = max_pages.max(pages);
        }
        state.use_xqa_decode = true;
        state.xqa_max_pages_per_seq = ops.xqa_decode_page_bucket(max_pages);
        // ④ envelope banding: this deployment's band walk is PLAN-FREE, so
        // stamping k/rows is all the prepare owes.
        state.depth_band_count = 0;
        if (1..=3).contains(&params.depth_band_count) && !have_custom_mask {
            for j in 0..params.depth_band_count as usize {
                state.depth_band_k[j] = params.depth_band_k[j];
                state.depth_band_rows[j] = params.depth_band_rows[j];
            }
            state.depth_band_count = params.depth_band_count;
        }
        if gates.region_trace {
            eprintln!(
                "[band-prep] xqa-branch in={} stamped={}",
                params.depth_band_count, state.depth_band_count
            );
        }
        return;
    }
    if !is_pure_decode {
        // Real prefill/mixed batches share one attention schedule across
        // all layers when the model has a single global window; alternating
        // sliding-window layouts keep the per-layer planner path.
        if fwd_cfg.per_layer_window_left.is_empty() {
            if state.prefill_plan.is_none() {
                state.prefill_plan = Some(ops.make_prefill_plan());
            }
            let t = if fwd_cfg.tp_size > 0 { fwd_cfg.tp_size } else { 1 };
            // SnapKV observes the tail of the prompt, so the capture is
            // decided HERE: only FA2 is instrumented, and SM90-vs-FA2 is a
            // plan-time choice. A sliding-window model is excluded.
            let score_window = if fwd_cfg.sliding_window < 0 {
                params.attn_score_window
            } else {
                0
            };
            let plan = state.prefill_plan.as_mut().expect("just made");
            ops.plan_prefill(
                plan,
                qo_indptr_h,
                kv_page_indptr_h,
                kv_last_page_lens_h,
                total_tokens,
                num_requests,
                cfg.num_attention_heads / t,
                cfg.num_key_value_heads / t,
                cfg.head_dim_kernel,
                cache.page_size,
                main_ws,
                // Inert by default; see the C++'s measurement log for the
                // A/B that keeps PIE_PREFILL_GRAPH_PLAN off.
                gates.prefill_graph_plan && fwd_cfg.decode_plan_cuda_graph,
                fwd_cfg.sliding_window,
                /*full_attention_variant=*/ false,
                cache.hnd_layout,
                /*causal_mask=*/ true,
                /*custom_mask=*/ false,
                score_window > 0,
            );
            state.use_prefill_plan = true;
            state.prefill_score_window = score_window;
        }
        return;
    }
    if fwd_cfg.force_prefill_path {
        state.use_prefill_plan = false;
        state.use_prefill_decode_plan = false;
        // ④ banded depth, force_prefill deployment: plan-free; k/rows is
        // all the prepare owes.
        state.depth_band_count = 0;
        if (1..=3).contains(&params.depth_band_count) && is_pure_decode && !have_custom_mask {
            for j in 0..params.depth_band_count as usize {
                state.depth_band_k[j] = params.depth_band_k[j];
                state.depth_band_rows[j] = params.depth_band_rows[j];
            }
            state.depth_band_count = params.depth_band_count;
        }
        if gates.region_trace {
            eprintln!(
                "[band-prep] force-prefill-branch in={} stamped={}",
                params.depth_band_count, state.depth_band_count
            );
        }
        return;
    }
    let min_prefill_decode_pages = fwd_cfg.prefill_decode_min_kv_pages.max(0);
    kvpp_sentry("prepare", qo_indptr_h, kv_page_indptr_h, num_requests);
    // ④ Act 1: bands are re-stamped per fire.
    state.depth_band_count = 0;
    let mut total_kv_pages: u64 = 0;
    for r in 0..r_usize {
        total_kv_pages += u64::from(kv_page_indptr_h[r + 1] - kv_page_indptr_h[r]);
    }
    let avg_kv_pages = if num_requests > 0 {
        i32::try_from(total_kv_pages.div_ceil(u64::try_from(num_requests).unwrap_or(1)))
            .unwrap_or(i32::MAX)
    } else {
        0
    };
    state.use_prefill_decode_plan = fwd_cfg.use_prefill_decode_plan
        && (min_prefill_decode_pages == 0 || avg_kv_pages >= min_prefill_decode_pages);
    if state.use_prefill_decode_plan {
        if state.prefill_decode_plan.is_none() {
            state.prefill_decode_plan = Some(ops.make_prefill_plan());
        }
        let t = if fwd_cfg.tp_size > 0 { fwd_cfg.tp_size } else { 1 };
        let num_q_heads_local = cfg.num_attention_heads / t;
        let num_kv_heads_local = cfg.num_key_value_heads / t;
        state.prefill_decode_qo_indptr_h =
            (0..=u32::try_from(num_requests.max(0)).unwrap_or(0)).collect();
        let min_full_attention_pages = fwd_cfg.prefill_decode_full_attention_min_kv_pages.max(0);
        let full_attention_variant = fwd_cfg.prefill_decode_full_attention_min_requests > 0
            && num_requests >= fwd_cfg.prefill_decode_full_attention_min_requests
            && (min_full_attention_pages == 0 || avg_kv_pages >= min_full_attention_pages)
            && fwd_cfg.sliding_window < 0
            && fwd_cfg.per_layer_window_left.is_empty();
        let identity_qo = state.prefill_decode_qo_indptr_h.clone();
        let plan = state.prefill_decode_plan.as_mut().expect("just made");
        ops.plan_prefill(
            plan,
            &identity_qo,
            kv_page_indptr_h,
            kv_last_page_lens_h,
            /*total_tokens=*/ num_requests,
            num_requests,
            num_q_heads_local,
            num_kv_heads_local,
            cfg.head_dim_kernel,
            cache.page_size,
            main_ws,
            fwd_cfg.decode_plan_cuda_graph,
            fwd_cfg.sliding_window,
            full_attention_variant,
            cache.hnd_layout,
            /*causal_mask=*/ false,
            /*custom_mask=*/ false,
            /*wants_prefill_score=*/ false,
        );
        // ④ banded depth, prefill family: one plan per boundary,
        // identity-qo prefix restriction, each in its OWN workspace.
        if (1..=3).contains(&params.depth_band_count) && is_pure_decode && !have_custom_mask {
            for j in 0..params.depth_band_count as usize {
                let rows = params.depth_band_rows[j];
                state.depth_band_k[j] = params.depth_band_k[j];
                state.depth_band_rows[j] = rows;
                if rows == 0 {
                    continue;
                }
                if state.depth_band_prefill_plans[j].is_none() {
                    state.depth_band_prefill_plans[j] = Some(ops.make_prefill_plan());
                }
                let band_ws = ops.depth_band_ws_view(j);
                let plan = state.depth_band_prefill_plans[j].as_mut().expect("just made");
                ops.plan_prefill(
                    plan,
                    &identity_qo,
                    kv_page_indptr_h,
                    kv_last_page_lens_h,
                    /*total_tokens=*/ i32::try_from(rows).unwrap_or(0),
                    i32::try_from(rows).unwrap_or(0),
                    num_q_heads_local,
                    num_kv_heads_local,
                    cfg.head_dim_kernel,
                    cache.page_size,
                    band_ws,
                    fwd_cfg.decode_plan_cuda_graph,
                    fwd_cfg.sliding_window,
                    full_attention_variant,
                    cache.hnd_layout,
                    /*causal_mask=*/ false,
                    /*custom_mask=*/ false,
                    /*wants_prefill_score=*/ false,
                );
            }
            state.depth_band_count = params.depth_band_count;
        }
        if gates.region_trace {
            eprintln!(
                "[band-prep] prefill-branch in={} stamped={}",
                params.depth_band_count, state.depth_band_count
            );
        }
        return;
    }
    if state.decode_plan.is_none() {
        state.decode_plan = Some(ops.make_decode_plan());
    }
    let t = if fwd_cfg.tp_size > 0 { fwd_cfg.tp_size } else { 1 };
    let num_q_heads_local = cfg.num_attention_heads / t;
    let num_kv_heads_local = cfg.num_key_value_heads / t;
    {
        let plan = state.decode_plan.as_mut().expect("just made");
        ops.plan_decode(
            plan,
            kv_page_indptr_h,
            num_requests,
            num_q_heads_local,
            num_kv_heads_local,
            cfg.head_dim_kernel,
            cache.page_size,
            main_ws,
            fwd_cfg.decode_plan_cuda_graph,
            DECODE_FULL_ATTENTION_VARIANT
                && fwd_cfg.sliding_window < 0
                && fwd_cfg.per_layer_window_left.is_empty(),
            cache.hnd_layout,
        );
    }
    // S-2: the depth union's PREFIX plan, against the SECONDARY workspace.
    // Shape guards DECLINE rather than throw.
    if params.full_depth_rows != u32::MAX
        && params.full_depth_rows > 0
        && i64::from(params.full_depth_rows) < i64::from(num_requests)
        && is_pure_decode
        && !have_custom_mask
    {
        if state.depth_prefix_decode_plan.is_none() {
            state.depth_prefix_decode_plan = Some(ops.make_decode_plan());
        }
        let suffix_ws = ops.spatial_suffix_ws_view();
        let plan = state.depth_prefix_decode_plan.as_mut().expect("just made");
        ops.plan_decode(
            plan,
            kv_page_indptr_h,
            i32::try_from(params.full_depth_rows).unwrap_or(0),
            num_q_heads_local,
            num_kv_heads_local,
            cfg.head_dim_kernel,
            cache.page_size,
            suffix_ws,
            fwd_cfg.decode_plan_cuda_graph,
            DECODE_FULL_ATTENTION_VARIANT
                && fwd_cfg.sliding_window < 0
                && fwd_cfg.per_layer_window_left.is_empty(),
            cache.hnd_layout,
        );
    }
    // ④ banded depth: one prefix decode plan per band, deepest first, each
    // against its own workspace. A zero-row band needs no plan.
    state.depth_band_count = 0;
    if (1..=3).contains(&params.depth_band_count) && is_pure_decode && !have_custom_mask {
        for j in 0..params.depth_band_count as usize {
            let rows = params.depth_band_rows[j];
            state.depth_band_k[j] = params.depth_band_k[j];
            state.depth_band_rows[j] = rows;
            if rows == 0 {
                continue;
            }
            // The XQA deployment is PLAN-FREE; its band walk reads the
            // staged device CSRs directly.
            if state.use_xqa_decode {
                continue;
            }
            if state.depth_band_plans[j].is_none() {
                state.depth_band_plans[j] = Some(ops.make_decode_plan());
            }
            let band_ws = ops.depth_band_ws_view(j);
            let plan = state.depth_band_plans[j].as_mut().expect("just made");
            ops.plan_decode(
                plan,
                kv_page_indptr_h,
                i32::try_from(rows).unwrap_or(0),
                num_q_heads_local,
                num_kv_heads_local,
                cfg.head_dim_kernel,
                cache.page_size,
                band_ws,
                fwd_cfg.decode_plan_cuda_graph,
                DECODE_FULL_ATTENTION_VARIANT
                    && fwd_cfg.sliding_window < 0
                    && fwd_cfg.per_layer_window_left.is_empty(),
                cache.hnd_layout,
            );
        }
        state.depth_band_count = params.depth_band_count;
    }
    if gates.region_trace {
        eprintln!(
            "[band-prep] decode-branch in={} stamped={}",
            params.depth_band_count, state.depth_band_count
        );
    }
}
