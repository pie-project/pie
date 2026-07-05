//! Cold create-time driver capabilities.
//!
//! Drivers return a versioned JSON document once at create time. The runtime uses
//! the parsed [`DriverCapabilities`] for allocation ceilings, batching limits,
//! model metadata, and weight-layout planning. Legacy IPC-only fields are
//! intentionally gone from this final local contract.

use serde::{Deserialize, Serialize};

/// Static driver capabilities reported at handshake time.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DriverCapabilities {
    /// Local direct-FFI ABI version used by the capability payload.
    pub abi_version: u32,
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
    /// Maximum forward-pass tokens accepted in one driver fire.
    pub max_forward_tokens: u32,
    /// Maximum forward-pass requests accepted in one driver fire.
    pub max_forward_requests: u32,
    /// Maximum page references accepted in one driver fire.
    pub max_page_refs: u32,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal handshake JSON carrying only the required (non-default)
    /// fields — i.e. what an older driver that predates the storage-target
    /// hints would emit.
    fn legacy_caps_json() -> &'static str {
        r#"{
            "abi_version": 1,
            "total_pages": 1024,
            "kv_page_size": 16,
            "swap_pool_size": 0,
            "max_forward_tokens": 512,
            "max_forward_requests": 32,
            "max_page_refs": 4096,
            "arch_name": "qwen3",
            "vocab_size": 151936,
            "max_model_len": 4096,
            "activation_dtype": "bf16"
        }"#
    }

    #[test]
    fn storage_fields_default_to_neutral_when_absent() {
        let caps: DriverCapabilities = serde_json::from_str(legacy_caps_json()).unwrap();
        assert_eq!(caps.abi_version, 1);
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

    #[test]
    fn deleted_legacy_fields_are_rejected() {
        let json = r#"{
            "abi_version": 1,
            "total_pages": 1024,
            "kv_page_size": 16,
            "swap_pool_size": 0,
            "max_forward_tokens": 512,
            "max_forward_requests": 32,
            "max_page_refs": 4096,
            "arch_name": "qwen3",
            "vocab_size": 151936,
            "max_model_len": 4096,
            "activation_dtype": "bf16",
            "shmem_name": "/legacy"
        }"#;
        assert!(serde_json::from_str::<DriverCapabilities>(json).is_err());
    }
}
