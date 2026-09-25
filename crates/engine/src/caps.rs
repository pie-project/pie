use eta_ir::registry::{GeometryClass, ModelProfile, PortMask};
use serde::{Deserialize, Serialize};

use crate::transfer::{KvHandle, MemoryDomain};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceFacts {
    pub backend: String,
    pub domain: MemoryDomain,
    pub sms: u32,
    pub unified_memory: bool,
    pub fp8_native: bool,
    pub native_mxfp4_moe: bool,
    pub storage_alignment: u32,
    pub storage_max_tile_bytes: u64,
    pub codegen_backend: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCopyDomains {
    pub device_to_device: bool,
    pub device_to_host: bool,
    pub host_to_device: bool,
    pub host_to_host: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FireLimits {
    pub max_lanes: u32,
    pub max_tokens: u32,
    pub max_page_refs: u32,
    pub max_context: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolFacts {
    pub kv_pages: u32,
    pub kv_page_size: u32,
    pub state_slots: u32,
    pub state_slot_bytes: u64,
    pub adapter_banks: u32,
    pub elastic_page_bytes: u64,
    pub elastic_budget_pages: u64,
    /// The windowed kv pool, page 0 its null page, and the window its rows
    /// are read through; zero when the model declares no windowed space.
    #[serde(default)]
    pub window_pages: u32,
    #[serde(default)]
    pub window_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Capabilities {
    pub device: DeviceFacts,
    pub pools: PoolFacts,
    pub limits: FireLimits,
    pub profile: ModelProfile,
    pub ports: PortMask,
    pub geometry: GeometryClass,
    pub kv_copy: KvCopyDomains,
    pub kv_handle: Option<KvHandle>,
    pub media_encode: bool,
    #[serde(default)]
    pub device_channel_commit: bool,

    #[serde(default)]
    pub rs_verbs: bool,

    #[serde(default)]
    pub bidirectional_attention: bool,
}

impl Capabilities {
    #[must_use]
    pub fn admits(&self, wanted: GeometryClass) -> bool {
        self.ports.covers(wanted.ports())
    }
}
