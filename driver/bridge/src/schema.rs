//! Canonical wire schema for the pie driver/runtime interface.
//!
//! These Rust types ARE the schema — no codegen, no IDL, no mirror. The
//! `#[schema(...)]` attribute macro derives `rkyv::{Archive, Serialize,
//! Deserialize}` and emits the entire C-ABI + PyO3 surface (readers,
//! parse entry points, descriptor types, builders) mechanically from
//! the type name. Adding a field → readers and the descriptor's mirror
//! field appear automatically; no hand-written accessors anywhere.
//!
//! ## Symbol naming
//!
//! Every emitted symbol follows the same mechanical pattern, derived
//! from the type name in `snake_case`:
//!
//!   * Reader:     `pie_<type>_<field>(...)`
//!   * Parse:      `pie_parse_<type>(bytes, len) -> *const ArchivedT`
//!   * Descriptor: `Pie<T>Desc` (C-friendly mirror)
//!   * Builder:    `pie_build_<type>(*const Pie<T>Desc, out, cap) -> usize`
//!   * Enum kind:  `pie_<type>_kind(p) -> u8` (data enums)
//!   * Enum value: `pie_<type>_value(p) -> u8` (unit enums)
//!   * Enum cast:  `pie_<type>_as_<variant>(p) -> *const ArchivedV`
//!
//! ## Evolution rules
//!
//!   * Append fields only. Removing or reordering changes the archived
//!     layout and [`crate::SCHEMA_HASH`] will catch the mismatch at
//!     connect time.
//!   * For enums, append variants only — the discriminant byte is on
//!     the wire.
//!   * Type changes (e.g. `u32` → `u64`) are protocol breaks. Bump
//!     [`crate::ipc::MAGIC`] to force older binaries to refuse
//!     the connection.

// Re-export so the `schema_module!` macro and the schema's nested-type
// references resolve `crate::schema::__py_brle`, `ArchivedBrle`,
// `PieBrleDesc`, etc. (it expects everything under `schema::`).
pub use crate::brle::*;
use pie_bridge_macros::schema;

// =============================================================================
// Top-level frames
// =============================================================================

#[schema]
pub struct Frame {
    pub driver_id: u32,
    pub payload: RequestPayload,
}

#[schema]
pub struct ResponseFrame {
    pub driver_id: u32,
    pub aborted: bool,
    pub payload: ResponsePayload,
}

#[schema]
#[allow(clippy::large_enum_variant)] // Boxing would change the wire schema and add hot-path indirection.
pub enum RequestPayload {
    Forward(ForwardRequest),
    Copy(CopyRequest),
    Adapter(AdapterRequest),
    Health,
}

#[schema]
#[allow(clippy::large_enum_variant)] // Boxing would change the wire schema and add hot-path indirection.
pub enum ResponsePayload {
    Forward(ForwardResponse),
    Status(StatusResponse),
}

// =============================================================================
// Forward pass
// =============================================================================

/// Batched forward-pass request. All vector fields are SoA: one entry
/// per token (token_ids, position_ids), per page (kv_page_indices), per
/// request (qo_indptr boundaries), or per slot (samplers +
/// sampler_indptr boundaries).
#[derive(Default)]
#[schema]
pub struct ForwardRequest {
    pub token_ids: Vec<u32>,
    pub position_ids: Vec<u32>,

    pub kv_page_indices: Vec<u32>,
    pub kv_page_indptr: Vec<u32>,
    pub kv_last_page_lens: Vec<u32>,
    pub qo_indptr: Vec<u32>,

    /// Runtime-managed recurrent-state cache slots, one per request
    /// for linear-attention models. Empty for models without rs_cache.
    pub rs_slot_ids: Vec<u32>,
    /// Per-request rs_cache flags. Bit 0 means "reset this slot before
    /// executing the request" (fresh context or replay-from-zero).
    pub rs_slot_flags: Vec<u8>,

    /// Per-row BRLE attention masks, flattened across the batch.
    /// `mask_indptr[r..r+1]` is the half-open range of rows in `masks`
    /// belonging to request `r`.
    pub masks: Vec<Brle>,
    pub mask_indptr: Vec<u32>,

    /// Per-request logit BRLE masks. Each request contributes 0 or 1
    /// entries; `logit_mask_indptr[r..r+1]` partitions per request.
    pub logit_masks: Vec<Brle>,
    pub logit_mask_indptr: Vec<u32>,

    pub sampling_indices: Vec<u32>,
    pub sampling_indptr: Vec<u32>,

    pub samplers: Vec<Sampler>,
    /// CSR boundary into `samplers` — `sampler_indptr[r..r+1]` is the
    /// half-open range of sampler slots for request `r`.
    pub sampler_indptr: Vec<u32>,

    pub adapter_bindings: Vec<AdapterBinding>,

    pub spec_token_ids: Vec<u32>,
    pub spec_position_ids: Vec<u32>,
    pub spec_indptr: Vec<u32>,
    pub output_spec_flags: Vec<bool>,

    pub context_ids: Vec<u64>,

    pub single_token_mode: bool,
    pub has_user_mask: bool,
}

/// Sampler configuration. Variants align with the kernel-dispatch types
/// in pie's sampling module. With rkyv, the Rust enum IS the wire
/// format — no mirror, no `*Args` wrapping.
///
/// Optional values use sentinels rather than `Option<T>` so the macro
/// can emit `Pie*Desc` in pure source order (no Option-last reshuffle)
/// and the C header mirrors the Rust struct byte-for-byte.
#[derive(PartialEq)]
#[schema]
pub enum Sampler {
    /// `seed = 0` → use a fresh per-fire random seed (no caller-provided
    /// seed). Any other value is the caller-provided PRNG seed.
    Multinomial {
        temperature: f32,
        seed: u32,
    },
    TopK {
        temperature: f32,
        k: u32,
    },
    TopP {
        temperature: f32,
        p: f32,
    },
    MinP {
        temperature: f32,
        p: f32,
    },
    TopKTopP {
        temperature: f32,
        k: u32,
        p: f32,
    },
    Embedding,
    Dist {
        temperature: f32,
        num_tokens: u32,
    },
    RawLogits,
    Logprob {
        token_id: u32,
    },
    Logprobs {
        token_ids: Vec<u32>,
    },
    Entropy,
}

/// Per-slot adapter binding. `-1` sentinels mean "unbound" — both fields
/// are signed so the wire form matches what portable's legacy SoA path
/// already consumes (`.as<int64_t>()`), no shim conversion needed.
#[derive(Default, PartialEq, Eq)]
#[schema]
pub struct AdapterBinding {
    /// `-1` means no adapter bound for this slot.
    pub adapter_id: i64,
    /// `-1` means no caller-provided adapter seed.
    pub seed: i64,
}

#[derive(Default)]
#[schema]
pub struct ForwardResponse {
    pub num_requests: u32,

    pub tokens_indptr: Vec<u32>,
    pub tokens: Vec<u32>,

    pub dists_req_indptr: Vec<u32>,
    pub dists_kv_indptr: Vec<u32>,
    pub dists_ids: Vec<u32>,
    pub dists_probs: Vec<f32>,

    pub logits_req_indptr: Vec<u32>,
    pub logits_byte_indptr: Vec<u32>,
    /// Opaque native-endian f32 bytes; one concatenated buffer indexed
    /// by `logits_byte_indptr`.
    pub logits_bytes: Vec<u8>,

    pub logprobs_req_indptr: Vec<u32>,
    pub logprobs_val_indptr: Vec<u32>,
    pub logprobs_values: Vec<f32>,

    pub entropies_indptr: Vec<u32>,
    pub entropies: Vec<f32>,
}

// =============================================================================
// Cold methods
// =============================================================================

#[schema]
pub struct CopyRequest {
    pub dir: CopyDir,
    pub srcs: Vec<u32>,
    pub dsts: Vec<u32>,
    pub resource: CopyResource,
}

#[derive(Copy, PartialEq, Eq)]
#[schema]
pub enum CopyDir {
    D2H,
    H2D,
    D2D,
    H2H,
}

#[derive(Copy, PartialEq, Eq)]
#[schema]
pub enum CopyResource {
    Kv,
    Rs,
}

#[schema]
pub struct AdapterRequest {
    pub op: AdapterOp,
    pub adapter_id: u64,
    /// Only meaningful when `op == Load`. Empty string for the other ops.
    pub path: String,
}

#[derive(Copy, PartialEq, Eq)]
#[schema]
pub enum AdapterOp {
    Load,
    Save,
    ZoInit,
    ZoUpdate,
}

/// Generic status response. Convention:
///   * 0 = success
///   * negative = error
///   * -1 reserved for abort sentinel (the server writes one of these
///     when a lease drops without commit; see [`ResponseFrame::aborted`]).
///   * positive = method-specific
#[derive(Copy)]
#[schema]
pub struct StatusResponse {
    pub status: i32,
}
