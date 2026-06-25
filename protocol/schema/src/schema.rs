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
//!     the IPC `MAGIC` constant (in the in-proc IPC crate) to force
//!     older binaries to refuse the connection.

// Re-export so the `schema_module!` macro and the schema's nested-type
// references resolve `crate::schema::__py_brle`, `ArchivedBrle`,
// `PieBrleDesc`, etc. (it expects everything under `schema::`).
pub use crate::brle::*;
// Flat-POD wire types embedded by value via `#[schema(pod)]` (see `crate::pod`).
// Re-exported so the historical `pie_schema::schema::{CopyDir, ...}` paths keep
// resolving for downstream consumers that imported them from here.
pub use crate::pod::{AdapterBinding, AdapterOp, CopyDir, CopyResource, StatusResponse};
use pie_schema_derive::schema;

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
    Status(#[schema(pod)] StatusResponse),
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

    // ── Samplers (SoA, #14 Phase 1) ─────────────────────────────────
    // One entry per sampler slot (length N = `sampler_indptr[last]`). Replaces
    // the former `Vec<Sampler>` AoS tagged-union. The wire carries the RAW
    // request values per slot (per-variant field, 0/empty where N/A) — driver
    // conventions (kind remap, Dist→top_k packing, p→top_p/min_p with 1.0/0.0
    // defaults) live in the driver's `view.hpp`, NOT on this neutral floor.
    /// Per-slot sampler kind: a `PIE_SAMPLER_*` discriminant.
    pub sampler_kinds: Vec<u8>,
    /// Sampling temperature (Multinomial/TopK/TopP/MinP/TopKTopP/Dist); 0 else.
    pub sampler_temperatures: Vec<f32>,
    /// Top-k cutoff (TopK/TopKTopP); 0 = N/A.
    pub sampler_top_k: Vec<u32>,
    /// Unified p value (TopP/MinP/TopKTopP — one kind per slot disambiguates); 0 = N/A.
    pub sampler_p: Vec<f32>,
    /// PRNG seed (Multinomial); 0 = fresh per-fire seed.
    pub sampler_seeds: Vec<u32>,
    /// Number of distribution tokens (Dist); 0 = N/A.
    pub sampler_num_tokens: Vec<u32>,
    /// Logprob.token_id (1 entry) / Logprobs.token_ids (M entries), concatenated;
    /// `sampler_token_ids_indptr[s..s+1]` is slot `s`'s range (empty for others).
    pub sampler_token_ids: Vec<u32>,
    pub sampler_token_ids_indptr: Vec<u32>,
    /// CSR boundary into the sampler slots — `sampler_indptr[r..r+1]` is the
    /// half-open range of sampler slots for request `r`.
    pub sampler_indptr: Vec<u32>,

    #[schema(pod)]
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
}

/// Per-slot sampler kind discriminants. Carried on the wire as the `u8`
/// elements of [`ForwardRequest::sampler_kinds`]; the value equals the
/// declaration index of the matching [`Sampler`] variant. Kept as explicit
/// `pub const`s (not a `#[repr(u8)]` enum) so cbindgen emits them verbatim as
/// the `PIE_SAMPLER_*` C consts that the driver's wire→driver kind remap
/// (`kNewToOldSamplerKind`) sources. Single source of truth for the producer
/// (`ForwardRequest::set_samplers`) and the C++ consumer.
pub const PIE_SAMPLER_MULTINOMIAL: u8 = 0;
pub const PIE_SAMPLER_TOP_K: u8 = 1;
pub const PIE_SAMPLER_TOP_P: u8 = 2;
pub const PIE_SAMPLER_MIN_P: u8 = 3;
pub const PIE_SAMPLER_TOP_K_TOP_P: u8 = 4;
pub const PIE_SAMPLER_EMBEDDING: u8 = 5;
pub const PIE_SAMPLER_DIST: u8 = 6;
pub const PIE_SAMPLER_RAW_LOGITS: u8 = 7;
pub const PIE_SAMPLER_LOGPROB: u8 = 8;
pub const PIE_SAMPLER_LOGPROBS: u8 = 9;
pub const PIE_SAMPLER_ENTROPY: u8 = 10;

/// Sampler configuration. As of #14 Phase 1, `Sampler` is **no longer a wire
/// type** — `ForwardRequest` carries the flattened SoA arrays
/// (`sampler_kinds`/`sampler_temperatures`/…). This plain enum survives as the
/// runtime's domain representation: the runtime builds a `Vec<Sampler>`, flattens
/// it into the request via [`ForwardRequest::set_samplers`], and keeps the list
/// to interpret response slots by kind. The driver reads only the SoA arrays.
#[derive(Debug, Clone, PartialEq)]
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

impl ForwardRequest {
    /// Flatten a slice of [`Sampler`]s into this request's SoA arrays (the wire
    /// form). Writes the RAW per-variant values (0/empty where not applicable);
    /// driver conventions (kind remap, Dist→top_k packing, p→top_p/min_p
    /// defaults) stay in the driver. Sets `sampler_kinds`/`temperatures`/`top_k`/
    /// `p`/`seeds`/`num_tokens`/`token_ids`(+`indptr`); leaves `sampler_indptr`
    /// (the per-request CSR) to the caller.
    pub fn set_samplers(&mut self, samplers: &[Sampler]) {
        let n = samplers.len();
        self.sampler_kinds = Vec::with_capacity(n);
        self.sampler_temperatures = Vec::with_capacity(n);
        self.sampler_top_k = Vec::with_capacity(n);
        self.sampler_p = Vec::with_capacity(n);
        self.sampler_seeds = Vec::with_capacity(n);
        self.sampler_num_tokens = Vec::with_capacity(n);
        self.sampler_token_ids = Vec::new();
        self.sampler_token_ids_indptr = Vec::with_capacity(n + 1);
        self.sampler_token_ids_indptr.push(0);
        for s in samplers {
            self.push_sampler(s);
        }
    }

    /// Append a single [`Sampler`] onto the SoA arrays, extending the
    /// `sampler_token_ids` CSR — the incremental form of [`set_samplers`] used by
    /// the host `sampler()` accumulator. Ensures the CSR carries its leading 0;
    /// the caller updates `sampler_indptr` once the slots are in.
    pub fn push_sampler(&mut self, s: &Sampler) {
        if self.sampler_token_ids_indptr.is_empty() {
            self.sampler_token_ids_indptr.push(0);
        }
        let (kind, temperature, k, p, seed, num_tokens) = match s {
            Sampler::Multinomial { temperature, seed } => {
                (PIE_SAMPLER_MULTINOMIAL, *temperature, 0, 0.0, *seed, 0)
            }
            Sampler::TopK { temperature, k } => (PIE_SAMPLER_TOP_K, *temperature, *k, 0.0, 0, 0),
            Sampler::TopP { temperature, p } => (PIE_SAMPLER_TOP_P, *temperature, 0, *p, 0, 0),
            Sampler::MinP { temperature, p } => (PIE_SAMPLER_MIN_P, *temperature, 0, *p, 0, 0),
            Sampler::TopKTopP { temperature, k, p } => {
                (PIE_SAMPLER_TOP_K_TOP_P, *temperature, *k, *p, 0, 0)
            }
            Sampler::Embedding => (PIE_SAMPLER_EMBEDDING, 0.0, 0, 0.0, 0, 0),
            Sampler::Dist {
                temperature,
                num_tokens,
            } => (PIE_SAMPLER_DIST, *temperature, 0, 0.0, 0, *num_tokens),
            Sampler::RawLogits => (PIE_SAMPLER_RAW_LOGITS, 0.0, 0, 0.0, 0, 0),
            Sampler::Logprob { token_id } => {
                self.sampler_token_ids.push(*token_id);
                (PIE_SAMPLER_LOGPROB, 0.0, 0, 0.0, 0, 0)
            }
            Sampler::Logprobs { token_ids } => {
                self.sampler_token_ids.extend_from_slice(token_ids);
                (PIE_SAMPLER_LOGPROBS, 0.0, 0, 0.0, 0, 0)
            }
            Sampler::Entropy => (PIE_SAMPLER_ENTROPY, 0.0, 0, 0.0, 0, 0),
        };
        self.sampler_kinds.push(kind);
        self.sampler_temperatures.push(temperature);
        self.sampler_top_k.push(k);
        self.sampler_p.push(p);
        self.sampler_seeds.push(seed);
        self.sampler_num_tokens.push(num_tokens);
        self.sampler_token_ids_indptr
            .push(self.sampler_token_ids.len() as u32);
    }

    /// Number of sampler slots (`= sampler_indptr[last]`).
    #[inline]
    pub fn n_samplers(&self) -> usize {
        self.sampler_kinds.len()
    }

    /// Reconstruct every [`Sampler`] as an owned `Vec`, the AoS view of the SoA
    /// arrays. For read-all consumers (chunk completion, response assembly) that
    /// want the old `samplers` field shape; the hot decode path reads the SoA
    /// arrays (or [`sampler_at`](Self::sampler_at)) directly instead.
    pub fn samplers(&self) -> Vec<Sampler> {
        (0..self.n_samplers())
            .map(|s| self.sampler_at(s).expect("slot in range"))
            .collect()
    }

    /// Reconstruct the [`Sampler`] at slot `s` from the SoA arrays — the inverse
    /// of [`set_samplers`](Self::set_samplers). Used by request-splitting paths
    /// that re-group sampler slots. Returns `None` if `s` is out of range.
    pub fn sampler_at(&self, s: usize) -> Option<Sampler> {
        let kind = *self.sampler_kinds.get(s)?;
        let temperature = self.sampler_temperatures[s];
        let k = self.sampler_top_k[s];
        let p = self.sampler_p[s];
        let seed = self.sampler_seeds[s];
        let num_tokens = self.sampler_num_tokens[s];
        Some(match kind {
            PIE_SAMPLER_MULTINOMIAL => Sampler::Multinomial { temperature, seed },
            PIE_SAMPLER_TOP_K => Sampler::TopK { temperature, k },
            PIE_SAMPLER_TOP_P => Sampler::TopP { temperature, p },
            PIE_SAMPLER_MIN_P => Sampler::MinP { temperature, p },
            PIE_SAMPLER_TOP_K_TOP_P => Sampler::TopKTopP { temperature, k, p },
            PIE_SAMPLER_EMBEDDING => Sampler::Embedding,
            PIE_SAMPLER_DIST => Sampler::Dist {
                temperature,
                num_tokens,
            },
            PIE_SAMPLER_RAW_LOGITS => Sampler::RawLogits,
            PIE_SAMPLER_LOGPROB => {
                let lo = self.sampler_token_ids_indptr[s] as usize;
                Sampler::Logprob {
                    token_id: self.sampler_token_ids[lo],
                }
            }
            PIE_SAMPLER_LOGPROBS => {
                let lo = self.sampler_token_ids_indptr[s] as usize;
                let hi = self.sampler_token_ids_indptr[s + 1] as usize;
                Sampler::Logprobs {
                    token_ids: self.sampler_token_ids[lo..hi].to_vec(),
                }
            }
            _ => Sampler::Entropy,
        })
    }

    /// Append one request's sampler SoA slots onto this (batch) request,
    /// offsetting the `sampler_token_ids` CSR. Mirrors the AoS
    /// `batch.samplers.extend(req.samplers)` it replaces; the caller pushes the
    /// per-request boundary onto `sampler_indptr` afterward.
    pub fn extend_samplers_from(&mut self, req: &ForwardRequest) {
        self.sampler_kinds.extend_from_slice(&req.sampler_kinds);
        self.sampler_temperatures
            .extend_from_slice(&req.sampler_temperatures);
        self.sampler_top_k.extend_from_slice(&req.sampler_top_k);
        self.sampler_p.extend_from_slice(&req.sampler_p);
        self.sampler_seeds.extend_from_slice(&req.sampler_seeds);
        self.sampler_num_tokens
            .extend_from_slice(&req.sampler_num_tokens);
        let tok_base = self.sampler_token_ids.len() as u32;
        self.sampler_token_ids
            .extend_from_slice(&req.sampler_token_ids);
        // req's CSR starts at 0; skip its leading 0, offset the rest.
        for &off in req.sampler_token_ids_indptr.iter().skip(1) {
            self.sampler_token_ids_indptr.push(tok_base + off);
        }
    }
}

// `AdapterBinding` (flat `#[repr(C)]` POD, embedded as `Vec<AdapterBinding>` in
// `ForwardRequest` via `#[schema(pod)]`) now lives in `crate::pod`.

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
    #[schema(pod)]
    pub dir: CopyDir,
    pub srcs: Vec<u32>,
    pub dsts: Vec<u32>,
    #[schema(pod)]
    pub resource: CopyResource,
}

#[schema]
pub struct AdapterRequest {
    #[schema(pod)]
    pub op: AdapterOp,
    pub adapter_id: u64,
    /// Only meaningful when `op == Load`. Empty string for the other ops.
    pub path: String,
}

// `CopyDir`, `CopyResource`, `AdapterOp` (flat `#[repr(u8)]` enums) and
// `StatusResponse` / `AdapterBinding` (flat `#[repr(C)]` structs) are flat-POD
// wire types — see `crate::pod`. They are embedded by value via `#[schema(pod)]`
// above (and `ForwardRequest.adapter_bindings` / `ResponsePayload::Status`).
