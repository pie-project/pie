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
    // ---- Device storage-target hints (weight-loader Variant A) ----
    // These describe how the device wants persistent weights laid out so the
    // in-process storage compiler can plan residency. All are `serde(default)`
    // so drivers that predate the handshake (or don't care) parse cleanly with
    // neutral values.
    /// Device storage backend tag (e.g. `cuda`); empty = unspecified/neutral.
    #[serde(default)]
    pub storage_backend: String,
    /// Maximum single-copy tile size in bytes the device prefers for staged
    /// H2D weight transfers; 0 = no limit / driver default.
    #[serde(default)]
    pub max_tile_bytes: u64,
    /// Preferred byte alignment for persistent weight buffers; 0 = neutral
    /// (compiler falls back to its own default).
    #[serde(default)]
    pub preferred_alignment: u32,
    /// MXFP4 MoE handling policy tag the device advertises (e.g.
    /// `routed_decode` / `native_gemm` / `eager_bf16`); empty = neutral.
    #[serde(default)]
    pub mxfp4_moe_policy: String,
    /// True when the device has a native MXFP4 MoE GEMM path (so packed
    /// blocks/scales can stay quantized on device rather than dequantizing).
    #[serde(default)]
    pub native_mxfp4_moe: bool,
    /// Optional shmem region name; only Some when the driver speaks shmem
    /// (subprocess flavors). In-process drivers leave this absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shmem_name: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal handshake JSON carrying only the required (non-default)
    /// fields — i.e. what an older driver that predates the storage-target
    /// hints would emit.
    fn legacy_caps_json() -> &'static str {
        r#"{
            "total_pages": 1024,
            "kv_page_size": 16,
            "swap_pool_size": 0,
            "max_forward_tokens": 512,
            "max_forward_requests": 32,
            "max_page_refs": 4096,
            "max_logit_rows": 512,
            "max_prob_rows": 512,
            "max_custom_mask_bytes": 0,
            "max_sampler_rows": 512,
            "max_logprob_labels": 0,
            "arch_name": "qwen3",
            "vocab_size": 151936,
            "max_model_len": 4096,
            "activation_dtype": "bf16"
        }"#
    }

    #[test]
    fn storage_fields_default_to_neutral_when_absent() {
        let caps: DriverCapabilities = serde_json::from_str(legacy_caps_json()).unwrap();
        assert_eq!(caps.storage_backend, "");
        assert_eq!(caps.max_tile_bytes, 0);
        assert_eq!(caps.preferred_alignment, 0);
        assert_eq!(caps.mxfp4_moe_policy, "");
        assert!(!caps.native_mxfp4_moe);
    }

    #[test]
    fn storage_fields_round_trip_through_json() {
        let mut caps: DriverCapabilities = serde_json::from_str(legacy_caps_json()).unwrap();
        caps.storage_backend = "cuda".to_string();
        caps.max_tile_bytes = 64 * 1024 * 1024;
        caps.preferred_alignment = 256;
        caps.mxfp4_moe_policy = "native_gemm".to_string();
        caps.native_mxfp4_moe = true;
        let json = serde_json::to_string(&caps).unwrap();
        let back: DriverCapabilities = serde_json::from_str(&json).unwrap();
        assert_eq!(back.storage_backend, "cuda");
        assert_eq!(back.max_tile_bytes, 64 * 1024 * 1024);
        assert_eq!(back.preferred_alignment, 256);
        assert_eq!(back.mxfp4_moe_policy, "native_gemm");
        assert!(back.native_mxfp4_moe);
    }
}
