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

/// The runtime's MXFP4 MoE lowering request.
///
/// `Auto` lets the loader pick from what the driver says the device can do;
/// the explicit variants pin a lowering and fail the load if the device cannot
/// provide it. Mirrors `PieLoaderMxfp4MoeRequest` and must keep the same
/// discriminants — the value is forwarded to the loader unchanged.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Mxfp4MoeRequest {
    #[default]
    Auto = 0,
    RoutedDecode = 1,
    NativeGemm = 2,
    EagerBf16 = 3,
}

impl Mxfp4MoeRequest {
    /// Parse the spelling used in worker config and CLI flags.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "" | "auto" => Some(Self::Auto),
            "routed_dequant" | "packed" | "routed_decode" => Some(Self::RoutedDecode),
            "bf16" | "dequant" | "eager_bf16" => Some(Self::EagerBf16),
            "native" | "native_gemm" => Some(Self::NativeGemm),
            _ => None,
        }
    }
}

/// What the CHECKPOINT is, for a driver that cannot read one.
///
/// # Why this exists
///
/// Every driver answers a [`DriverCapabilities`], and two thirds of that
/// struct is a statement about the DEVICE — how many pages, which copy
/// directions, which sinks it can honour, how wide a fire it can run. The
/// remaining third is a statement about the checkpoint: its architecture, its
/// vocabulary, how long a context it was published for.
///
/// `driver-metal` answers the whole thing itself, because it identifies the
/// checkpoint and so knows both halves. `driver-vulkan` and `driver-wgpu`
/// cannot: they keep `model` and `model-loader` as **dev**-dependencies, and
/// `tests/pure.rs` asserts that closure — a driver that depended on a
/// checkpoint FORMAT would be a driver that could not be handed bytes.
///
/// So the identification happens once, on the side that already reads
/// catalogs, and its result crosses as this. What it replaced was worse than
/// a missing type: the engine's seams built the ENTIRE `DriverCapabilities`
/// on the driver's behalf — sixty lines each of `has_attn_score: false`,
/// `kv_copy_domain_mask: ...`, `max_forward_tokens: 4096` — so a fact about
/// what a device could do was written down by the crate that dispatches to
/// it, in two copies that had already drifted (one said
/// `PIE_DECODE_ENVELOPE_PORTS`, the other `0`).
///
/// That is the same shape `DriverSpec::device_domain` records as having cost
/// a hardcoded `PIE_MEMORY_DOMAIN_CUDA_DEVICE` at nine sites, and the same
/// one `Driver::kind` and `Driver::device_domain` fixed for the two facts
/// that were `match`es in the engine. This is the third and largest.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelFacts {
    /// The architecture the catalog row advertises, e.g. `"llama"`.
    pub arch_name: String,
    /// The catalog row's id.
    pub model_id: String,
    /// Tokens in the vocabulary.
    pub vocab_size: u32,
    /// The longest context the row was published for.
    pub max_model_len: u32,
    /// The residual width.
    pub hidden_size: u32,
    /// Where the payload was read from, for diagnostics.
    pub snapshot_dir: String,
}

/// Runtime-owned payload for the blocking model-load boot call.
///
/// Carries the *request*, not a compiled plan: the driver compiles the load
/// plan itself, because only the driver can measure the device it will run on
/// (`loader/architecture.md` §3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelLoadDesc {
    pub snapshot_dir: PathBuf,
    /// The runtime's own quantization request (e.g. `"fp8"`), not a checkpoint
    /// fact. Empty means "whatever the checkpoint is".
    pub runtime_quant: String,
    pub mxfp4_moe: Mxfp4MoeRequest,
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

/// One model-structural expert-selection site of the driver's declared
/// plan: an MoE trace's per-token expert-indexed matmul group, stated as
/// exactly the parameters the engine's fire planner vocabulary takes
/// (`expert_weights_site(experts, top_k)`). One entry per distinct
/// parameterization, in the plan's first-appearance order.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExpertSiteSummary {
    /// The expert count the per-token selector indexes (the router logits
    /// width — a plan fact, not the weight-template cardinality).
    pub experts: u32,
    /// `k` of the `TopK` op producing the selector.
    pub top_k: u32,
}

/// The site summary of the driver's traced + validated declared plan — the
/// model-structural divergence sites the plan's own structure states,
/// reported through the capabilities handshake so the engine's scheduler
/// can thread them to fire planning without re-tracing from binding facts
/// it does not have (the engine's `fire_plan::site_table` module doc
/// records that analysis).
///
/// Empty (the default) means "no model-structural sites known": the driver
/// did not trace (`PIE_DECLARED_FORWARD` off), the validation refused the
/// configuration, or the plan is dense. Absence of the field parses to the
/// same — old payloads keep today's behavior.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelSiteSummary {
    #[serde(default)]
    pub expert_sites: Vec<ExpertSiteSummary>,
}

impl ModelSiteSummary {
    pub fn is_empty(&self) -> bool {
        self.expert_sites.is_empty()
    }
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
    /// The driver can maintain per-page KV key envelopes and execute the
    /// `envelope_dot` second-party kernel at an attention stage (Quest).
    /// Requires a native-bf16 NHD paged KV cache AND a query hook that fires
    /// post-rope, so the score compares against the keys as cached.
    #[serde(default)]
    pub has_kv_envelopes: bool,
    /// The driver can observe per-position softmax attention weights at an
    /// `OnAttn` tap (`IntrinsicId::AttnScore`), for H2O/TOVA-style eviction.
    /// Requires a score-observing attention kernel, and is refused for
    /// soft-capped or sliding-window attention, where the captured row is not
    /// the softmax those policies are defined over.
    #[serde(default)]
    pub has_attn_score: bool,
    /// The driver can HONOUR an `attn_page_mask` sink: it compacts the fire's
    /// page table to the kept pages before the layer's attention. Advertised
    /// separately from `has_attn_score` because observing scores and enforcing
    /// a selection are independent backend abilities -- a driver may well have
    /// one without the other.
    #[serde(default)]
    pub has_attn_page_mask: bool,
    /// The driver can HONOUR a `lora` sink: it consumes the sink's A/B/SITES
    /// configuration and applies the low-rank delta at the declared projection
    /// sites for the whole forward. First-party name, so like
    /// `has_attn_page_mask` this must gate at bind -- a backend that cannot
    /// apply the delta would otherwise run the program as a silent no-op
    /// adapter.
    #[serde(default)]
    pub has_lora: bool,
    /// Site summary of the driver's traced + validated declared plan
    /// ([`ModelSiteSummary`]); empty when no plan was traced/validated or
    /// the plan declares no model-structural sites (every dense model).
    #[serde(default)]
    pub model_site_summary: ModelSiteSummary,
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
    /// Catalog id of the row the driver loaded (e.g. `qwen3-0.6b`).
    ///
    /// This is the identity, and it is a whole answer where `arch_name` is
    /// a family. `arch_name` came from the checkpoint's `config.json`
    /// `model_type` and was therefore a string the host had to interpret
    /// a second time, with its own table, under its own defaults — which
    /// is how a chat template could be chosen that the model's vocabulary
    /// did not contain. An id names a row in the `const` catalog that the
    /// host and the driver both link, so the answer the host reaches is
    /// the answer the driver used.
    ///
    /// Empty from a driver that has not been moved onto the catalog yet.
    #[serde(default)]
    pub model_id: String,
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
    /// Which backend's source this driver wants in
    /// [`ProgramRegistration::emitted_kernels`](crate::plan::ProgramRegistration),
    /// or empty when it generates its own (or needs none).
    ///
    /// Emitted kernels are tens of kilobytes each and a program can have dozens,
    /// so the host only runs code generation for a driver that says it will read
    /// the result. This is what lets the two backends move off their in-driver
    /// emitters one at a time.
    #[serde(default)]
    pub codegen_backend: String,
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
        // The fixture predates the site summary; absence parses to empty
        // ("no model-structural sites known").
        assert!(caps.model_site_summary.is_empty());
        let json = serde_json::to_string(&caps).unwrap();
        assert_eq!(
            serde_json::from_str::<DriverCapabilities>(&json).unwrap(),
            caps
        );
    }

    /// A populated site summary — the shape the CUDA driver emits from its
    /// validated MoE plan — round-trips, entry order preserved.
    #[test]
    fn model_site_summary_round_trips() {
        let mut caps: DriverCapabilities = serde_json::from_str(caps_json()).unwrap();
        caps.model_site_summary = ModelSiteSummary {
            expert_sites: vec![ExpertSiteSummary {
                experts: 256,
                top_k: 8,
            }],
        };
        let json = serde_json::to_string(&caps).unwrap();
        let parsed: DriverCapabilities = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, caps);
        assert_eq!(parsed.model_site_summary.expert_sites.len(), 1);
        assert_eq!(parsed.model_site_summary.expert_sites[0].experts, 256);
        assert_eq!(parsed.model_site_summary.expert_sites[0].top_k, 8);

        // And the driver-side JSON spelling parses to the same summary.
        let driver_row = r#"{"expert_sites": [{"experts": 256, "top_k": 8}]}"#;
        assert_eq!(
            serde_json::from_str::<ModelSiteSummary>(driver_row).unwrap(),
            caps.model_site_summary
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
