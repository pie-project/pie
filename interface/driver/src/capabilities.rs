//! Driver capability handshake — NOT a rkyv wire type.
//!
//! Each driver advertises its capabilities (page geometry, forward limits,
//! tokenizer arch, etc.) at startup via a JSON blob delivered to the runtime.
//! The runtime parses it into [`DriverCapabilities`] for scheduling and
//! batching decisions. The shape is wire-stable — fields appear verbatim
//! in the driver's ready-callback JSON (see `driver/cuda/src/entry.cpp`,
//! `driver/metal/src/entry.cpp`, `driver/dummy/src/lib.rs`).
//!
//! This is intentionally not under `#[schema]` — the handshake happens
//! once at driver startup and uses JSON over a side channel, not the
//! rkyv ring.

use serde::{Deserialize, Serialize};

/// Static driver capabilities reported at handshake time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriverCapabilities {
    /// Total KV pages available for context residency.
    pub total_pages: u32,
    /// KV page size in tokens.
    pub kv_page_size: u32,
    /// Number of CPU-resident swap-pool pages (0 if no swap support).
    pub swap_pool_size: u32,
    /// True when the model needs runtime-assigned recurrent-state slots.
    #[serde(default)]
    pub rs_cache_required: bool,
    /// Number of GPU-resident recurrent-state slots (0 if unsupported).
    #[serde(default)]
    pub rs_cache_slots: u32,
    /// Bytes per recurrent-state slot, for accounting/telemetry.
    #[serde(default)]
    pub rs_cache_slot_bytes: u64,
    /// True when the driver repairs recurrent state internally after
    /// system-speculative draft rejection.
    #[serde(default)]
    pub rs_cache_spec_rollback: bool,
    /// True when the driver wired a system drafter and can verify/return
    /// system-provided speculative drafts (the capability signal).
    #[serde(default)]
    pub system_speculation_supported: bool,
    /// Operator opt-in for system speculation (deployment config). The runtime
    /// combines this with `system_speculation_supported` to decide whether to
    /// drive drafts. Default false = off unless explicitly enabled.
    #[serde(default)]
    pub enable_system_speculation: bool,
    /// Maximum forward-pass tokens accepted in one driver fire.
    pub max_forward_tokens: u32,
    /// Maximum forward-pass requests accepted in one driver fire.
    pub max_forward_requests: u32,
    /// Maximum page references accepted in one driver fire.
    pub max_page_refs: u32,
    /// Maximum logits rows the driver can return in one fire.
    pub max_logit_rows: u32,
    /// Maximum probability rows the driver can return in one fire.
    pub max_prob_rows: u32,
    /// Maximum custom-mask bytes accepted in one fire.
    pub max_custom_mask_bytes: u32,
    /// Maximum sampler rows accepted in one fire.
    pub max_sampler_rows: u32,
    /// Maximum logprob label count accepted in one fire.
    pub max_logprob_labels: u32,
    /// Architecture name (e.g. `llama3`, `qwen3`) — used for tokenizer dispatch.
    pub arch_name: String,
    /// Vocabulary size — pinned by the loaded model.
    pub vocab_size: u32,
    /// Maximum model context length (positions). Drives scheduler ceiling.
    pub max_model_len: u32,
    /// Activation dtype on the driver side (`bf16` / `f16` / `f32`).
    pub activation_dtype: String,
    /// Optional snapshot directory the driver can use to persist state.
    #[serde(default)]
    pub snapshot_dir: String,
    /// Optional shmem region name; only Some when the driver speaks shmem
    /// (subprocess flavors). In-process drivers leave this absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shmem_name: Option<String>,
}
