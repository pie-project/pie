//! What a device is ([`DeviceFacts`], stable for the process), and what a
//! load of it can do ([`Capabilities`], what [`Loaded`](crate::load::Loaded)
//! carries back).

use serde::{Deserialize, Serialize};
use eta_ir::registry::{GeometryClass, ModelProfile, PortMask};

use crate::transfer::{KvHandle, MemoryDomain};

/// What the machine is. Stable for the life of the process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceFacts {
    /// Which shell this is: `"cuda"`, `"metal"`, `"mock"`.
    pub backend: String,
    /// Where this device's memory lives, including its ordinal if it has one.
    pub domain: MemoryDomain,
    /// The device's parallel width; the one number the compiler's layout
    /// pass currently reads.
    pub sms: u32,
    /// Host and device see one address space.
    pub unified_memory: bool,
    /// The device has FP8 arithmetic, not just FP8 storage.
    pub fp8_native: bool,
    /// The device has an MXFP4 MoE GEMM (as against dequantizing to bf16).
    pub native_mxfp4_moe: bool,
    /// Byte alignment every storage binding must satisfy.
    pub storage_alignment: u32,
    /// The largest single tile a storage binding may map.
    pub storage_max_tile_bytes: u64,
    /// Which backend name the guest-program codegen emits for, when this shell
    /// compiles guest programs at all.
    pub codegen_backend: Option<String>,
}

/// Which directions this engine's `copy_kv` serves.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCopyDomains {
    /// Device to device — peer copy or NVLink.
    pub device_to_device: bool,
    /// Device to host — an eviction or an export.
    pub device_to_host: bool,
    /// Host to device — a restore or an import.
    pub host_to_device: bool,
    /// Host to host — staging between pinned buffers.
    pub host_to_host: bool,
}

/// The ceilings one fire may not exceed, baked in at load.
///
/// A submission past any of them is
/// [`Error::Impossible`](crate::Error::Impossible), not
/// [`Exhausted`](crate::Error::Exhausted): freeing pages cannot make a graph
/// recorded for 8192 rows take 9000.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FireLimits {
    /// The most lanes one fire may carry — `Dim::Lanes`.
    pub max_lanes: u32,
    /// The most token rows one fire may carry — `Dim::Tokens`.
    pub max_tokens: u32,
    /// The most page references one fire's geometry may name.
    pub max_page_refs: u32,
    /// The most tokens one sequence may hold.
    pub max_context: u32,
}

/// How much room a load's pools came out with.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolFacts {
    /// KV pages the pool seats.
    pub kv_pages: u32,
    /// Tokens per KV page.
    pub kv_page_size: u32,
    /// Recurrent-state slots the pool seats.
    pub state_slots: u32,
    /// Bytes one recurrent-state slot occupies.
    pub state_slot_bytes: u64,
    /// Adapter banks the pool seats — a budget, not an admission cap.
    pub adapter_banks: u32,
    /// Bytes one elastic page occupies, when the pools are virtual. Zero
    /// means they are not — a load whose pools are one fixed reservation.
    pub elastic_page_bytes: u64,
    /// The most elastic pages this load may ever map.
    pub elastic_budget_pages: u64,
}

/// What this load can do — the record `Engine::load` answers with.
///
/// `PartialEq` but not `Eq`: `ModelProfile::activation` reaches a `Dtype`, so
/// comparison is structural, not reflexive (float-shaped facts).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Capabilities {
    /// The machine underneath.
    pub device: DeviceFacts,
    /// The pools this load reserved.
    pub pools: PoolFacts,
    /// The ceilings a fire is baked against.
    pub limits: FireLimits,
    /// The guest-visible model profile — vocabulary, page size, layer count,
    /// activation dtype, and which model-gated intrinsics a guest program may
    /// name. The bind-time input `eta_compiler::validate` takes.
    pub profile: ModelProfile,
    /// The descriptor ports this shell resolves on the device.
    pub ports: PortMask,
    /// The most demanding geometry class those ports admit.
    pub geometry: GeometryClass,
    /// Which `copy_kv` directions are served.
    pub kv_copy: KvCopyDomains,
    /// This load's KV pool, if it is exportable to a peer.
    pub kv_handle: Option<KvHandle>,
    /// Whether `encode` is served — i.e. whether this load carries a
    /// multimodal encoder.
    pub media_encode: bool,
    /// True: channel rings advance on-device (predictable cursors, published
    /// pinned mirror). False (Metal): caller/engine own separate host/device
    /// rings, cells cross at the fire boundary.
    #[serde(default)]
    pub device_channel_commit: bool,

    /// True: [`RsVerb::Buffer`](crate::RsVerb::Buffer)/
    /// [`RsVerb::FoldBuffered`](crate::RsVerb::FoldBuffered) are served.
    /// False (Metal): only [`RsVerb::Fold`](crate::RsVerb::Fold) is; the
    /// others are refused by name, not silently folded.
    #[serde(default)]
    pub rs_verbs: bool,
}

impl Capabilities {
    /// Does this load serve everything `wanted` asks for?
    ///
    /// The one negotiation the contract does: a caller states the class it
    /// wants to submit in, and a load either resolves those ports or does not.
    #[must_use]
    pub fn admits(&self, wanted: GeometryClass) -> bool {
        self.ports.covers(wanted.ports())
    }
}
