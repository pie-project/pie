//! Cold driver facts returned by the two boot calls.
//!
//! `create` returns [`DeviceFacts`], which contains only properties that can be
//! queried before a model exists. `load_model` returns [`DriverCapabilities`],
//! which contains model-derived limits and metadata.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const KV_COPY_DEVICE_TO_DEVICE: u32 = 1 << 0;
pub const KV_COPY_DEVICE_TO_HOST: u32 = 1 << 1;
pub const KV_COPY_HOST_TO_DEVICE: u32 = 1 << 2;
pub const KV_COPY_HOST_TO_HOST: u32 = 1 << 3;

/// Runtime-owned payload for the blocking model-load boot call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelLoadDesc {
    pub load_plan_bytes: Vec<u8>,
    pub snapshot_dir: PathBuf,
    pub compiler_version: u64,
    pub component: crate::ModelComponent,
}

/// Create-time device properties used by the runtime storage compiler.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DeviceFacts {
    pub abi_version: u32,
    pub backend: String,
    pub unified_memory: bool,
    pub fp8_native: bool,
    pub native_mxfp4_moe: bool,
    pub storage_alignment: u32,
    pub storage_max_tile_bytes: u64,
    pub storage_tile_map_mask: u32,
    pub page_size: u32,
}

/// Model-derived capabilities returned after the LoadPlan is executed.
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
    /// Supported whole-page KV copy directions.
    #[serde(default)]
    pub kv_copy_domain_mask: u32,
    /// True when the model needs runtime-assigned recurrent-state slots.
    #[serde(default)]
    pub rs_cache_required: bool,
    /// Number of GPU-resident recurrent-state slots (0 if unsupported).
    #[serde(default)]
    pub rs_cache_slots: u32,
    /// Bytes per recurrent-state slot, for accounting/telemetry.
    #[serde(default)]
    pub rs_cache_slot_bytes: u64,
    /// Shared elastic-memory accounting page size in bytes (0 if unsupported).
    #[serde(default)]
    pub elastic_page_bytes: u64,
    /// Total pages in the device-wide elastic physical budget.
    #[serde(default)]
    pub elastic_budget_pages: u64,
    /// The loaded model exposes native MTP draft-logit rows to PTIR.
    #[serde(default)]
    pub has_mtp_logits: bool,
    /// The loaded model exposes device-resident MTP draft token IDs to PTIR.
    #[serde(default)]
    pub has_mtp_drafts: bool,
    /// The loaded model exposes a scalar value-head result to PTIR.
    #[serde(default)]
    pub has_value_head: bool,
    /// Descriptor-port tags the driver can resolve on-device for decode envelopes.
    #[serde(default)]
    pub device_geometry_port_mask: u32,
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
    #[serde(default)]
    pub hidden_size: u32,
    #[serde(default)]
    pub supports_media_encode: bool,
    /// Optional snapshot directory the driver can use to persist state.
    #[serde(default)]
    pub snapshot_dir: String,
    #[serde(default)]
    pub kv_handle: Option<crate::transfer::KvHandle>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps_json() -> &'static str {
        r#"{
            "abi_version": 4,
            "total_pages": 1024,
            "kv_page_size": 16,
            "swap_pool_size": 0,
            "kv_copy_domain_mask": 0,
            "max_forward_tokens": 512,
            "max_forward_requests": 32,
            "max_page_refs": 4096,
            "arch_name": "qwen3",
            "vocab_size": 151936,
            "max_model_len": 4096,
            "activation_dtype": "bf16",
            "has_mtp_logits": true,
            "has_mtp_drafts": false,
            "has_value_head": false
        }"#
    }

    #[test]
    fn capabilities_round_trip() {
        let caps: DriverCapabilities = serde_json::from_str(caps_json()).unwrap();
        assert!(caps.has_mtp_logits);
        assert!(!caps.has_mtp_drafts);
        let json = serde_json::to_string(&caps).unwrap();
        assert_eq!(
            serde_json::from_str::<DriverCapabilities>(&json).unwrap(),
            caps
        );
    }

    #[test]
    fn device_facts_round_trip() {
        let facts = DeviceFacts {
            abi_version: 4,
            backend: "metal".to_string(),
            unified_memory: true,
            fp8_native: false,
            native_mxfp4_moe: false,
            storage_alignment: 256,
            storage_max_tile_bytes: 64 << 20,
            storage_tile_map_mask: 0,
            page_size: 16 << 10,
        };
        let json = serde_json::to_string(&facts).unwrap();
        assert_eq!(serde_json::from_str::<DeviceFacts>(&json).unwrap(), facts);
    }

    #[test]
    fn deleted_legacy_fields_are_rejected() {
        let json = r#"{
            "abi_version": 4,
            "total_pages": 1024,
            "kv_page_size": 16,
            "swap_pool_size": 0,
            "kv_copy_domain_mask": 0,
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
