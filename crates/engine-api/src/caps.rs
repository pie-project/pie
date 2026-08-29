//! What a device is, and what a load of it can do.
//!
//! Two records with two different lifetimes, which is why they are two records
//! and not the one `DriverCapabilities` struct that stood here:
//!
//! * [`DeviceFacts`] is about the **machine**. It is answerable before
//!   anything is loaded, it does not change while the process runs, and the
//!   compiler reads it to decide a layout.
//! * [`Capabilities`] is about **this load**: how big the pools came out, what
//!   the fire ceilings are, which descriptor ports the shell serves. It is
//!   what [`Loaded`](crate::load::Loaded) carries back.
//!
//! # What died on the way here
//!
//! `DriverCapabilities` was a 30-field flat struct with `#[serde(default)]` on
//! two thirds of it, and it mixed three subjects: the device, the load, and
//! the MODEL. The model half — `vocab_size`, `max_model_len`,
//! `activation_dtype: String`, and eight `has_*` booleans naming
//! model-gated PTIR intrinsics — is [`tensor_ir::registry::ModelProfile`],
//! which is the type `tensor-compiler` binds a guest program against. The
//! runtime used to rebuild a `ModelProfile` out of those booleans at bind time
//! (`pipeline::program::profile_from`); carrying the profile itself removes
//! the second copy and the mapping between them.
//!
//! Also gone: `abi_version` (see [`crate::error`] on `BAD_ABI_VERSION`),
//! `kv_copy_domain_mask` as four `1 << n` constants (now [`KvCopyDomains`]),
//! `device_geometry_port_mask` as a private thirteen-bit numbering (now
//! [`tensor_ir::registry::PortMask`], in the registry that owns the ports —
//! decision 19), and `snapshot_dir`/`model_id`/`arch_name`, which say where
//! the caller's own checkpoint came from.

use serde::{Deserialize, Serialize};
use tensor_ir::registry::{GeometryClass, ModelProfile, PortMask};

use crate::transfer::{KvHandle, MemoryDomain};

/// What the machine is. Stable for the life of the process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceFacts {
    /// Which shell this is: `"cuda"`, `"metal"`, `"mock"`.
    pub backend: String,
    /// Where this device's memory lives — and, for the domains that have one,
    /// which ordinal it is. Replaces the `DeviceDomain = u32` alias and the
    /// separate `device_ordinal: u32` that always travelled beside it.
    pub domain: MemoryDomain,
    /// The device's parallel width. The one `DeviceProfile` number the
    /// compiler's layout pass currently reads, and the reason a shell that
    /// probes its device can bake a better plan than one that does not.
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
///
/// Four `bool`s, which is what `kv_copy_domain_mask: u32` and its four
/// `KV_COPY_*` bit constants were saying. A caller reads the field it is about
/// to need instead of remembering which bit it is.
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

/// The ceilings one fire may not exceed.
///
/// These are the budgets the load was baked against, read back. A submission
/// past any of them is [`Error::Impossible`](crate::Error::Impossible)
/// rather than [`Exhausted`](crate::Error::Exhausted): freeing pages
/// cannot make a graph that was recorded for 8192 rows take 9000.
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
    /// Adapter banks the pool seats (design §8 — a budget, not an admission
    /// cap).
    pub adapter_banks: u32,
    /// Bytes one elastic page occupies, when the pools are virtual. Zero
    /// means they are not — a load whose pools are one fixed reservation.
    pub elastic_page_bytes: u64,
    /// The most elastic pages this load may ever map.
    pub elastic_budget_pages: u64,
}

/// What THIS LOAD can do.
///
/// The record `load` answers with, and what a controller republishes to its
/// peers (`controller_api::WorkerInfo::capability`).
///
/// `PartialEq` but not `Eq`: [`ModelProfile::activation`] reaches a `DType`
/// and the profile is compared structurally, which is what a caller wants
/// ("did this worker's capability change?") — but `Eq` would be a claim about
/// reflexivity that a profile carrying float-shaped facts should not make.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Capabilities {
    /// The machine underneath.
    pub device: DeviceFacts,
    /// The pools this load reserved.
    pub pools: PoolFacts,
    /// The ceilings a fire is baked against.
    pub limits: FireLimits,
    /// The guest-visible model profile — vocabulary, page size, layer count,
    /// activation dtype, and which model-gated intrinsics and second-party
    /// kernels a guest program may name. The bind-time input
    /// `tensor_compiler::validate` takes, carried rather than reconstructed.
    pub profile: ModelProfile,
    /// The descriptor ports this shell resolves on the device. Named in the
    /// registry's own numbering (decision 19).
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
    /// **Does this engine advance its channel rings on the DEVICE, through
    /// the predicated commit kernel?** (alto design §1 article 3, wave F2a.)
    ///
    /// An engine that answers `true` states three things at once, and the
    /// caller's whole channel plane turns on them:
    ///
    /// ```text
    /// * it accepts a lane's `channels` TICKETS and validates them where the
    ///   data is, so the caller may predict cursors by counting
    /// * its host-visible channels live in mapped pinned memory it publishes
    ///   at registration (`RegisteredChannel::mirror`), so a guest's cells
    ///   cross by device access and never by a copy through the contract
    /// * `publish_channel`/`take_channel` are therefore a CONVENIENCE and not
    ///   the path: a caller that adopted the mirror pumps nothing
    /// ```
    ///
    /// `false` is the shape every engine had before F2a and the shape Metal
    /// still has: the caller owns the host ring, the engine owns the device
    /// ring, and cells cross at the fire's boundary through the two verbs.
    #[serde(default)]
    pub device_channel_commit: bool,

    /// **Does this engine serve the recurrent verbs beyond the plain fold?**
    /// (alto design §6, wave F3.)
    ///
    /// An engine that answers `true` states that
    /// [`RsVerb::Buffer`](crate::RsVerb::Buffer) scatters the recurrent ops'
    /// in-projection inputs into a buffered-activation pool it allocated at
    /// load and leaves the folded state untouched, and that
    /// [`RsVerb::FoldBuffered`](crate::RsVerb::FoldBuffered) replays that pool
    /// through conv+recurrence truncated at a device-resolved accepted length,
    /// from the buffer head the last fold left behind.
    ///
    /// It states the MIXED ROW too (wave F3b): a `Buffer` whose fold is
    /// non-zero lands the durable state on a row of the window it is writing,
    /// cutting the row at an interior boundary into the segment that folds and
    /// the segment that continues from it.
    ///
    /// `false` is the shape every engine had before F3, and the shape Metal
    /// still has: only [`RsVerb::Fold`](crate::RsVerb::Fold) is served, and a
    /// lane that asks for either other verb is refused by name
    /// ([`Lane::validate_for`](crate::Lane::validate_for)) rather than
    /// silently folded — a speculative draft handed a destructive fold would
    /// corrupt the state it was speculating over.
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
