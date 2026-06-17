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

    // ── Multimodal visual spans (vision / video) ────────────────────
    // Appended per the schema evolution rule (append-only). A "visual
    // span" is one still image or one video clip; the driver runs the
    // vision encoder over it and scatters the projected rows into the
    // hidden state. All fields are empty for text-only passes. See
    // MULTIMODAL.md §4.
    //
    // CSR across the batch: `image_indptr[r..r+1]` is the half-open range
    // of images belonging to request `r` (one extra leading 0, like
    // `mask_indptr`).
    pub image_indptr: Vec<u32>,
    /// `(t, h, w)` merged-token grid per image — 3 entries per image.
    pub image_grids: Vec<u32>,
    /// Sequence anchor position per image (1 entry per image).
    pub image_anchor_positions: Vec<u32>,
    /// Staged pixel bytes per image, concatenated. NOTE: until the host
    /// preprocessor lands (Phase 2) these are the *encoded* image bytes,
    /// not a normalized pixel tensor; the driver does not consume them
    /// yet. `image_pixel_indptr[i..i+1]` is image `i`'s byte range.
    pub image_pixels: Vec<u8>,
    pub image_pixel_indptr: Vec<u32>,
    /// Precomputed M-RoPE `(t, h, w)` position ids — 3 per image token,
    /// concatenated. Empty for 1-D-RoPE archs (Gemma). The host owns the
    /// M-RoPE math; `image_mrope_indptr[i..i+1]` is image `i`'s range.
    pub image_mrope_positions: Vec<u32>,
    pub image_mrope_indptr: Vec<u32>,

    // ── Option-B pixel path (inferlet pre-patchified; see MULTIMODAL.md) ──
    // `image_pixels` carries the f32 `pixel_values` tensor as little-endian
    // bytes (`n_patch · patch_dim · 4` per image; ranges = `image_pixel_indptr`).
    /// Per-patch `(x, y)` positions for every image, concatenated (2 per patch).
    /// Image `i`'s patch count is derivable from its pixel byte range.
    pub image_patch_positions: Vec<u32>,
    /// Batch row offset where each image's soft-token rows begin in `token_ids`
    /// (one per image). The driver runs the vision encoder and writes the
    /// projected rows' hidden state at `[anchor_row .. anchor_row + n_soft]`.
    pub image_anchor_rows: Vec<u32>,

    // ── Multimodal audio spans (gemma4_audio) ───────────────────────
    // Direct analog of the image block. A "clip" is one audio span; the
    // driver runs the audio (USM/Conformer) encoder over its log-mel
    // features and scatters the projected soft-token rows into the hidden
    // state. All fields empty for non-audio passes. See audio_frontend.md.
    /// Per-clip log-mel features as little-endian f32 bytes, concatenated.
    /// Clip `i`'s byte range is `audio_feature_indptr[i..i+1]`
    /// (`n_frames_i * 128 * 4` bytes).
    pub audio_features: Vec<u8>,
    pub audio_feature_indptr: Vec<u32>,
    /// Batch row offset where each clip's soft-token rows begin in `token_ids`
    /// (one per clip). The driver writes the projected rows at
    /// `[anchor_row .. anchor_row + n_audio_tok]`.
    pub audio_anchor_rows: Vec<u32>,
    /// Per-request CSR: `audio_indptr[r..r+1]` is the half-open range of clips
    /// belonging to request `r` (one extra leading 0, like `image_indptr`).
    pub audio_indptr: Vec<u32>,

    /// Per-request additive logit bias applied before sampling.
    /// `(logit_bias_tokens[i], logit_bias_values[i])` is one (token_id, bias) pair.
    pub logit_bias_tokens: Vec<u32>,
    pub logit_bias_values: Vec<f32>,
    /// Per-request CSR: `logit_bias_indptr[r..r+1]` is the half-open range of
    /// pairs belonging to request `r` (one extra leading 0, like `mask_indptr`).
    pub logit_bias_indptr: Vec<u32>,
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

    /// Per-request speculative draft side channel. `spec_indptr` has
    /// `num_requests + 1` entries and partitions both `spec_tokens` and
    /// `spec_positions`.
    pub spec_indptr: Vec<u32>,
    pub spec_tokens: Vec<u32>,
    pub spec_positions: Vec<u32>,

    pub probe_wire_parse_us: u32,
    pub probe_plan_us: u32,
    pub probe_h2d_us: u32,
    pub probe_kernel_launch_us: u32,
    pub probe_sync_us: u32,
    pub probe_response_build_us: u32,
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
    // CSM native audio output (pie:core/audio-out). Appended per the enum
    // evolution rule (append-only — the discriminant byte is on the wire). The
    // `AdapterRequest.path` carries a JSON request
    // `{"prompt":[u32,...],"max_frames":u32,"out_path":"..."}`; the CSM driver
    // runs the full generation (backbone prefill + per-frame depth loop + Mimi
    // decode), writes the raw little-endian f32 PCM to `out_path`, and returns
    // `StatusResponse.status` = number of Mimi frames produced (negative on
    // error). Reuses the Adapter cold-path transport so no new wire payload
    // variant is needed. See AUDIO_OUTPUT.md.
    GenerateAudio,
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
