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
// Re-exported so the historical `pie_driver_abi::schema::{CopyDir, ...}` paths keep
// resolving for downstream consumers that imported them from here.
pub use crate::pod::{AdapterBinding, AdapterOp, CopyDir, CopyResource, StatusResponse};
use pie_driver_abi_derive::schema;

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
    /// Per-request rs_cache flags. See `RS_FLAG_*`. Bit 0 (`RS_FLAG_RESET`)
    /// resets the slot before executing (fresh context / replay-from-zero);
    /// bit 1 (`RS_FLAG_FOLD`) folds buffered recurrent state into the slot's
    /// folded state after the pass (count of tokens given by `rs_fold_lens`).
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

    // ── Programmable sampling (Sampling IR carrier) ─────────────────
    // A forward pass may carry a sampling *program* — flat versioned L0
    // bytecode (`pie-sampling-ir`) that the driver JIT-compiles and runs in
    // place of the legacy per-slot samplers for the rows it covers. The bridge
    // is OPAQUE to the bytecode: it carries the bytes, the submit-bound input
    // buffers, and the declared late-bound keys; the driver parses + caches by
    // hash. All fields empty for the legacy sampler path. A pass uses EITHER
    // sampler slots OR a program (enforced host-side). Outputs surface through
    // the existing `ForwardResponse` slots in the program's declared output
    // order — the bridge stays output-agnostic.
    //
    // Batched like the image/audio side-channels: a per-REQUEST count CSR plus
    // nested per-PROGRAM CSRs. MVP carries 0 or 1 program per request; the
    // shape generalizes to per-slot programs without a wire break.

    /// Per-request CSR over the program list: `sampling_program_indptr[r..r+1]`
    /// is the half-open range of programs belonging to request `r` (leading 0,
    /// cumulative program count — like `image_indptr`). Empty = no programs.
    pub sampling_program_indptr: Vec<u32>,
    /// Concatenated L0 bytecode for every program in the batch; opaque to the
    /// bridge. `sampling_program_bytes_indptr[p..p+1]` is program `p`'s range.
    pub sampling_program_bytes: Vec<u8>,
    /// Per-program byte CSR partitioning `sampling_program_bytes` (leading 0).
    pub sampling_program_bytes_indptr: Vec<u32>,

    // Submit-bound host inputs: values known at submit time, bound into the
    // program's `host{key, submit-bound}` inputs. One opaque blob + a
    // key/offset/len index table. Refreshed each fire; no recompile.
    /// Concatenated submit-bound input buffer bytes for every program.
    pub sampling_input_blob: Vec<u8>,
    /// Index table: the program input key each entry binds to.
    pub sampling_input_keys: Vec<u32>,
    /// Index table: each entry's byte offset into `sampling_input_blob`.
    pub sampling_input_offsets: Vec<u32>,
    /// Index table: each entry's byte length within `sampling_input_blob`.
    pub sampling_input_lens: Vec<u32>,
    /// Per-program CSR into the input index table (the `keys`/`offsets`/`lens`
    /// arrays): `sampling_input_indptr[p..p+1]` is program `p`'s entries
    /// (leading 0).
    pub sampling_input_indptr: Vec<u32>,

    // Late-bound channel: keys the program declared `host{key, late-bound}` —
    // supplied after submit, before the first consuming kernel (mirostat μ,
    // grammar mask). `sampling_late_keys` lists the declared host-late keys per
    // program; the parallel `sampling_late_{blob,offsets,lens}` value table
    // (mirroring `sampling_input_*`, keyed by `sampling_late_keys`) carries the
    // host-supplied value bytes for the correctness path (value known by submit
    // time). A key with `len == 0` has no staged host value → the driver falls
    // to the device-resident alias (output-ref) or skips (miss policy = skip,
    // spec §7.4). Output-ref late inputs (`Binding::Output`) are resolved
    // device-side and carry NO host value here.
    /// Declared host-late input keys for every program, concatenated.
    pub sampling_late_keys: Vec<u32>,
    /// Per-program CSR into `sampling_late_keys`:
    /// `sampling_late_indptr[p..p+1]` is program `p`'s keys (leading 0).
    pub sampling_late_indptr: Vec<u32>,
    /// Concatenated host-late value bytes, parallel to `sampling_late_keys`.
    pub sampling_late_blob: Vec<u8>,
    /// Per-late-key byte offset into `sampling_late_blob` (parallel to
    /// `sampling_late_keys`).
    pub sampling_late_offsets: Vec<u32>,
    /// Per-late-key value byte length (parallel to `sampling_late_keys`);
    /// `0` = no host value staged for this key (device alias or skip-on-miss).
    pub sampling_late_lens: Vec<u32>,

    // Per-input-slot binding-map (the WIT `input-binding`). For each program
    // input slot — in `Op::Input(i)` order — how it is bound: the LM-head
    // `Logits` intrinsic (positions resolved host-side into `sampling_indices`)
    // or a submit `Tensor` keyed into the submit-input table. The binding-free
    // bytecode no longer carries this, so the carrier does: it lets the driver
    // wire each `Op::Input(i)` to the logits buffer or its keyed submit value.
    /// Per-slot binding discriminant: `0` = Logits, `1` = Tensor (see
    /// [`SamplingBinding`]). Concatenated across programs.
    pub sampling_binding_kind: Vec<u8>,
    /// Per-slot TensorKey for a `Tensor` slot (`0` for a `Logits` slot), parallel
    /// to `sampling_binding_kind`.
    pub sampling_binding_key: Vec<u32>,
    /// Per-program CSR into `sampling_binding_{kind,key}`:
    /// `sampling_binding_indptr[p..p+1]` is program `p`'s slots (leading 0).
    pub sampling_binding_indptr: Vec<u32>,

    // WS8 P2 device-resident next-input link (`forward-pass.next-input`). Sources
    // some of this request's input tokens device-side from a *prior* forward's
    // sampled tokens (`pi.sampled`, the producer), instead of host-injecting them
    // (P1). Per-row + mergeable: under continuous batching one (batched) request
    // carries N sequences, and a re-formed consumer batch can have rows whose
    // producers sit in *different* prior batches — so the link is keyed per-row by
    // a global producer link id, never per-request.
    //
    // RETENTION-UNTIL-DRAIN (load-bearing): a producer's `pi.sampled` MUST be
    // retained until *all* its consumer links drain — the device/batched analog of
    // P1's drain-before-producer-drop, per-row and across batch re-formation. The
    // executor / page-manager owns this lifetime (retain on `pipeline_source_link`,
    // free on drain); the driver's inject only *reads* the resolved retained
    // pointer. Missing it = use-after-free on a re-formed batch.
    /// Non-zero ⇒ retain this (batched) forward's `pi.sampled[N]` as a
    /// device-resident next-input source under this **global** link id, until all
    /// its consumer links drain (executor/page-manager owned). `0` = not a source.
    pub pipeline_source_link: u32,
    /// Per fed consumer input: the global producer link id whose retained
    /// `pi.sampled` is the source (rows may name different producers ⇒ different
    /// ids; the executor groups by id like `partition_by_program` keyed on
    /// producer). Parallel to `next_input_src_rows`/`next_input_dest_slots`.
    pub next_input_producer_links: Vec<u32>,
    /// Per fed consumer input: the source row within that producer's `pi.sampled`
    /// — the producer sequence's `sampling_row` identity (echo's `h_sample_idx`).
    pub next_input_src_rows: Vec<u32>,
    /// Per fed consumer input: the destination slot in THIS request's input token
    /// buffer (`token_ids`) that the producer's sampled token fills device-side
    /// (`u32::MAX` = skip the lane, the `-1` sentinel). Rebased on batch merge by
    /// the token offset (the only non-global field).
    pub next_input_dest_slots: Vec<u32>,

    // ── Recurrent-state fold (RS_FLAG_FOLD) ─────────────────────────
    // Appended per the schema evolution rule (append-only). One entry per
    // request, parallel to `rs_slot_ids`/`rs_slot_flags`: the number of
    // buffered recurrent-state tokens to fold into the slot's folded state
    // after the pass. Only meaningful where `RS_FLAG_FOLD` is set in
    // `rs_slot_flags`; 0 elsewhere. The runtime derives this from the
    // inferlet's explicit `inference.fold(set, n)` call; the executor lowers
    // it onto the model's existing fold primitive (e.g. cuda GDN
    // `commit_len`). Empty for models without rs_cache.
    pub rs_fold_lens: Vec<u32>,

    // ── Recurrent-state fold-from-buffer (RS_FLAG_FOLD, real-driver path) ──
    // Appended per the schema evolution rule (append-only). The buffered-slab
    // physical slot ids a fold pass folds *from*, as a CSR over the batch:
    // `rs_buffer_slot_indptr[r..r+1]` is request `r`'s half-open range of
    // buffered-slab ids in `rs_buffer_slot_ids` (one extra leading 0, like
    // `kv_page_indptr`). For a request with `RS_FLAG_FOLD` set, the executor's
    // fold-from-buffer kernel reads `rs_fold_lens[r]` tokens across those
    // buffered slabs and folds them into the request's folded slot
    // (`rs_slot_ids[r]`). The runtime resolves the buffered suffix's first
    // `rs_fold_lens[r]` tokens to physical slab ids during fold-txn prepare.
    // Both empty for non-fold passes / models without rs_cache. (v1 mock folds
    // in-pass via `commit_len` and ignores these; the real cuda GDN
    // fold-from-buffered-slabs kernel consumes them.)
    pub rs_buffer_slot_ids: Vec<u32>,
    pub rs_buffer_slot_indptr: Vec<u32>,

    // ── Device-resident next-input free signal (#6 WS8 P2) ──────────
    // Appended per the schema evolution rule (append-only). Producer link ids
    // (`pipeline_source_link`) whose LAST consumer drains on THIS pass: after
    // the pass's stream sync the executor frees each link's retained
    // `pi.sampled` copy + its sample-done event (the consumer inject that read
    // it has completed, so the free is hazard-free). Host-emitted once a link's
    // consumer refcount reaches 0; the driver is count-agnostic and frees
    // strictly on this signal. Empty when no retained source is released here.
    pub next_input_free_links: Vec<u32>,

    // ── Programmable-sampling output fast-path (#27 cut #1) ──────────
    // Appended per the schema evolution rule (append-only). Per-output
    // eager-D2H destination: the driver copies each declared program output
    // VALUE directly into a host pinned buffer at the submit-provided pointer
    // (skipping the `ForwardResponse` SoA channels + the host re-marshal in
    // `build_output_tensors`). The host `pie_pinned_alloc`s one buffer per
    // fast-path output (sizes are submit-time-known: Token/Scalar = 4 B,
    // Logits = vocab·4, Logprobs = k·4), threads the raw host pointer here, and
    // wraps the same buffer as the WIT `tensor` zero-copy in `output()`.
    //
    // IN-PROC ONLY: the pointer is a raw host address valid in the (in-proc)
    // driver's address space; an out-of-process driver MUST fall to the legacy
    // path. EMPTY ⇒ legacy `ForwardResponse` channels (opt-in per pass) — so
    // this is fully back-compatible and the marshal stays the fallback for the
    // output kinds not yet on the fast-path (Distribution/Embedding).
    //
    // SoA + CSR, mirroring the `sampling_input_*` index tables: the dst arrays
    // are flattened across programs in declared output order (the program's
    // `Op` output-slot order — the same order `build_output_tensors` walks),
    // partitioned per-program by `sampling_output_indptr`. The driver, walking
    // program `p`'s declared outputs, copies output `j` into
    // `dst_ptrs[indptr[p] + j]` for `dst_lens[indptr[p] + j]` bytes on the copy
    // stream, behind the case-(b) WAR guard. (MVP M=1: one value per output
    // slot; batched M>1 rides a per-row stride — a follow-up.)
    /// Host pinned-buffer destination pointer per output value (raw host
    /// address, u64). Parallel to `sampling_output_dst_lens`.
    pub sampling_output_dst_ptrs: Vec<u64>,
    /// Byte capacity at each destination (the driver bounds-checks the D2H copy
    /// against this). Parallel to `sampling_output_dst_ptrs`.
    pub sampling_output_dst_lens: Vec<u32>,
    /// Per-program CSR into the dst arrays (one extra leading 0, cumulative —
    /// like `sampling_input_indptr`): `sampling_output_indptr[p..p+1]` is
    /// program `p`'s output values. Empty ⇒ no fast-path outputs (legacy path).
    pub sampling_output_indptr: Vec<u32>,
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

// =============================================================================
// Recurrent-state slot flags
// =============================================================================

/// Per-slot recurrent-state flags packed into [`ForwardRequest::rs_slot_flags`]
/// (one `u8` per request). Bit flags — combine with bitwise OR. Kept as
/// explicit `pub const`s (like the `PIE_SAMPLER_*` set) so cbindgen emits them
/// verbatim as C consts for the driver's rs_cache kernels — single source of
/// truth for the runtime producer and the driver consumer.
///
/// `RS_FLAG_RESET` (bit 0): reset the slot's recurrent state before executing
/// the request (fresh context / replay-from-zero).
pub const RS_FLAG_RESET: u8 = 1;
/// `RS_FLAG_FOLD` (bit 1): after the pass, fold buffered recurrent state into
/// the slot's folded state. The number of buffered tokens to fold is carried
/// per request in [`ForwardRequest::rs_fold_lens`]. Set by the runtime in
/// response to an explicit `inference.fold(set, n)` call.
pub const RS_FLAG_FOLD: u8 = 2;

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

/// One submit-bound host input buffer for a sampling program: the program input
/// `key` it binds to and its raw little-endian bytes (dtype/shape fixed by the
/// program's input declaration). The AoS view of one entry in the
/// `ForwardRequest::sampling_input_*` index table; not itself a wire type.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingInput {
    pub key: u32,
    pub bytes: Vec<u8>,
}

/// How one program input slot is bound (the WIT `input-binding`): the AoS view
/// of one entry in `ForwardRequest::sampling_binding_*`. Not a wire type — the
/// runtime flattens it into the SoA via [`ForwardRequest::push_sampling_program`]
/// and the driver reads `sampling_binding_{kind,key}` to wire each `Op::Input(i)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingBinding {
    /// The LM-head logits intrinsic (sampling positions resolved host-side into
    /// `sampling_indices`).
    Logits,
    /// A submit-bound tensor value keyed into the submit-input table.
    Tensor { key: u32 },
}

impl SamplingBinding {
    /// Wire discriminant for a `Logits` slot.
    pub const KIND_LOGITS: u8 = 0;
    /// Wire discriminant for a `Tensor` slot.
    pub const KIND_TENSOR: u8 = 1;

    /// The `sampling_binding_kind` discriminant for this binding.
    pub fn kind(self) -> u8 {
        match self {
            SamplingBinding::Logits => Self::KIND_LOGITS,
            SamplingBinding::Tensor { .. } => Self::KIND_TENSOR,
        }
    }

    /// The `sampling_binding_key` value (the TensorKey, or `0` for `Logits`).
    pub fn key(self) -> u32 {
        match self {
            SamplingBinding::Logits => 0,
            SamplingBinding::Tensor { key } => key,
        }
    }

    /// Reconstruct from the wire `(kind, key)` pair (unknown kind ⇒ `Logits`).
    pub fn from_parts(kind: u8, key: u32) -> SamplingBinding {
        match kind {
            Self::KIND_TENSOR => SamplingBinding::Tensor { key },
            _ => SamplingBinding::Logits,
        }
    }
}

/// A sampling program submitted on a [`ForwardRequest`]: the opaque L0 bytecode,
/// its submit-bound input buffers, the keys it declared host-late, and the
/// host-late values supplied for the correctness path. This is the host-side AoS
/// view of the per-program SoA carrier
/// (`sampling_program_*`/`sampling_input_*`/`sampling_late_*`); like [`Sampler`]
/// it is **not** a wire type — the runtime flattens it into a `ForwardRequest`
/// via [`ForwardRequest::push_sampling_program`] and the driver reads the SoA
/// arrays. Distinct from `pie_sampling_ir::SamplingProgram` (the typed IR this
/// bytecode lowers from).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplingProgramSubmission {
    pub bytecode: Vec<u8>,
    pub inputs: Vec<SamplingInput>,
    /// Per-input-slot binding-map (the WIT `input-binding`), in `Op::Input(i)`
    /// order — one entry per program input slot. Empty for a legacy/no-binding
    /// submission; the driver then falls back to the keyed submit table.
    pub bindings: Vec<SamplingBinding>,
    /// Declared host-late keys (`host{late-bound}`), in declaration order.
    pub late_keys: Vec<u32>,
    /// Host-late values supplied for the correctness path (a subset of
    /// `late_keys`, by key). A late key absent here has no staged host value
    /// (device-resident output-ref alias, or skip-on-miss).
    pub late_inputs: Vec<SamplingInput>,
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

    // ── Sampling-program carrier helpers ─────────────────────────────
    // Mirror the sampler SoA helpers (`push_sampler`/`n_samplers`/`sampler_at`/
    // `extend_samplers_from`) for the programmable-sampling carrier.

    /// Append one [`SamplingProgramSubmission`] onto the per-program SoA carrier,
    /// extending the bytecode, the submit-bound input blob + index table, and
    /// the late-bound key channel (each with its leading-0 nested CSR). The
    /// caller pushes the per-request boundary onto `sampling_program_indptr`
    /// once the request's program(s) are in — exactly as `push_sampler` leaves
    /// `sampler_indptr` to the caller.
    pub fn push_sampling_program(&mut self, program: &SamplingProgramSubmission) {
        // Ensure the nested per-program CSRs carry their leading 0.
        if self.sampling_program_bytes_indptr.is_empty() {
            self.sampling_program_bytes_indptr.push(0);
        }
        if self.sampling_input_indptr.is_empty() {
            self.sampling_input_indptr.push(0);
        }
        if self.sampling_late_indptr.is_empty() {
            self.sampling_late_indptr.push(0);
        }
        if self.sampling_binding_indptr.is_empty() {
            self.sampling_binding_indptr.push(0);
        }

        // Bytecode (opaque) + its byte-range boundary.
        self.sampling_program_bytes
            .extend_from_slice(&program.bytecode);
        self.sampling_program_bytes_indptr
            .push(self.sampling_program_bytes.len() as u32);

        // Submit-bound inputs: append each buffer to the blob and record its
        // (key, offset, len) in the index table.
        for input in &program.inputs {
            let offset = self.sampling_input_blob.len() as u32;
            self.sampling_input_blob.extend_from_slice(&input.bytes);
            self.sampling_input_keys.push(input.key);
            self.sampling_input_offsets.push(offset);
            self.sampling_input_lens.push(input.bytes.len() as u32);
        }
        self.sampling_input_indptr
            .push(self.sampling_input_keys.len() as u32);

        // Late-bound channel: the declared host-late keys plus a parallel value
        // table — for each declared key, stage its supplied host value (if any;
        // matched by key) or record `len == 0` (no host value → device alias /
        // skip-on-miss).
        for &key in &program.late_keys {
            self.sampling_late_keys.push(key);
            let offset = self.sampling_late_blob.len() as u32;
            self.sampling_late_offsets.push(offset);
            match program.late_inputs.iter().find(|i| i.key == key) {
                Some(input) => {
                    self.sampling_late_blob.extend_from_slice(&input.bytes);
                    self.sampling_late_lens.push(input.bytes.len() as u32);
                }
                None => self.sampling_late_lens.push(0),
            }
        }
        self.sampling_late_indptr
            .push(self.sampling_late_keys.len() as u32);

        // Per-slot binding-map: append each slot's (kind, key) and close the
        // per-program boundary.
        for binding in &program.bindings {
            self.sampling_binding_kind.push(binding.kind());
            self.sampling_binding_key.push(binding.key());
        }
        self.sampling_binding_indptr
            .push(self.sampling_binding_kind.len() as u32);
    }

    /// Number of sampling programs carried (`= sampling_program_bytes_indptr`
    /// boundaries minus the leading 0). Zero for the legacy sampler path.
    #[inline]
    pub fn n_sampling_programs(&self) -> usize {
        self.sampling_program_bytes_indptr.len().saturating_sub(1)
    }

    /// Mark this (batched) forward as a device-resident next-input source: its
    /// `pi.sampled[N]` is retained under the **global** link id `link` until all
    /// its consumer links drain (executor/page-manager owned). `0` clears it.
    #[inline]
    pub fn set_pipeline_source_link(&mut self, link: u32) {
        self.pipeline_source_link = link;
    }

    /// Record one per-row device-resident next-input link (WS8 P2): the producer's
    /// sampled token at `pi.sampled[src_row]` — in the forward retained under
    /// producer link `producer_link` — fills this request's input token slot
    /// `dest_slot` device-side. `dest_slot == u32::MAX` skips the lane. The host
    /// emits these instead of the P1 host inject when device pipelining is active.
    #[inline]
    pub fn push_next_input_link(&mut self, producer_link: u32, src_row: u32, dest_slot: u32) {
        self.next_input_producer_links.push(producer_link);
        self.next_input_src_rows.push(src_row);
        self.next_input_dest_slots.push(dest_slot);
    }

    /// Number of per-row device-resident next-input links on this request.
    #[inline]
    pub fn n_next_input_links(&self) -> usize {
        self.next_input_producer_links.len()
    }

    /// Signal that producer link `link`'s LAST consumer drains on this pass: the
    /// executor frees its retained `pi.sampled` copy + sample-done event after
    /// the pass's stream sync. The host emits this once the link's consumer
    /// refcount reaches 0 (the driver is count-agnostic — it frees on signal).
    #[inline]
    pub fn push_next_input_free_link(&mut self, link: u32) {
        self.next_input_free_links.push(link);
    }

    /// Reconstruct the [`SamplingProgramSubmission`] at program index `p` from
    /// the SoA carrier — the inverse of [`push_sampling_program`]. Returns
    /// `None` if `p` is out of range.
    pub fn sampling_program_at(&self, p: usize) -> Option<SamplingProgramSubmission> {
        let byte_lo = *self.sampling_program_bytes_indptr.get(p)? as usize;
        let byte_hi = *self.sampling_program_bytes_indptr.get(p + 1)? as usize;
        let bytecode = self.sampling_program_bytes.get(byte_lo..byte_hi)?.to_vec();

        let entry_lo = self.sampling_input_indptr[p] as usize;
        let entry_hi = self.sampling_input_indptr[p + 1] as usize;
        let inputs = (entry_lo..entry_hi)
            .map(|e| {
                let off = self.sampling_input_offsets[e] as usize;
                let len = self.sampling_input_lens[e] as usize;
                SamplingInput {
                    key: self.sampling_input_keys[e],
                    bytes: self.sampling_input_blob[off..off + len].to_vec(),
                }
            })
            .collect();

        let late_lo = self.sampling_late_indptr[p] as usize;
        let late_hi = self.sampling_late_indptr[p + 1] as usize;
        let late_keys = self.sampling_late_keys[late_lo..late_hi].to_vec();
        // Reconstruct the host-late values (those with a non-zero length).
        let late_inputs = (late_lo..late_hi)
            .filter_map(|e| {
                let len = self.sampling_late_lens[e] as usize;
                if len == 0 {
                    return None;
                }
                let off = self.sampling_late_offsets[e] as usize;
                Some(SamplingInput {
                    key: self.sampling_late_keys[e],
                    bytes: self.sampling_late_blob[off..off + len].to_vec(),
                })
            })
            .collect();

        // Reconstruct the per-slot binding-map (empty if this program carried
        // none — e.g. a legacy submission predating the binding-map field).
        let bindings = match (
            self.sampling_binding_indptr.get(p),
            self.sampling_binding_indptr.get(p + 1),
        ) {
            (Some(&blo), Some(&bhi)) => (blo as usize..bhi as usize)
                .map(|s| {
                    SamplingBinding::from_parts(
                        self.sampling_binding_kind[s],
                        self.sampling_binding_key[s],
                    )
                })
                .collect(),
            _ => Vec::new(),
        };

        Some(SamplingProgramSubmission {
            bytecode,
            inputs,
            bindings,
            late_keys,
            late_inputs,
        })
    }

    /// Append another request's sampling-program SoA slots onto this (batch)
    /// request, offsetting every nested CSR (byte ranges, input index table,
    /// late keys). Mirrors [`extend_samplers_from`]; the caller pushes the
    /// per-request boundary onto `sampling_program_indptr` afterward.
    pub fn extend_sampling_programs_from(&mut self, req: &ForwardRequest) {
        // Bytecode: concat, offset the per-program byte CSR.
        let byte_base = self.sampling_program_bytes.len() as u32;
        self.sampling_program_bytes
            .extend_from_slice(&req.sampling_program_bytes);
        for &off in req.sampling_program_bytes_indptr.iter().skip(1) {
            self.sampling_program_bytes_indptr.push(byte_base + off);
        }

        // Submit-bound inputs: concat blob + index table, rebasing the byte
        // offsets by the blob base and the per-program CSR by the table base.
        let blob_base = self.sampling_input_blob.len() as u32;
        let entry_base = self.sampling_input_keys.len() as u32;
        self.sampling_input_blob
            .extend_from_slice(&req.sampling_input_blob);
        self.sampling_input_keys
            .extend_from_slice(&req.sampling_input_keys);
        for &off in &req.sampling_input_offsets {
            self.sampling_input_offsets.push(blob_base + off);
        }
        self.sampling_input_lens
            .extend_from_slice(&req.sampling_input_lens);
        for &off in req.sampling_input_indptr.iter().skip(1) {
            self.sampling_input_indptr.push(entry_base + off);
        }

        // Late-bound channel: concat the keys + the parallel value table,
        // rebasing the value byte-offsets by the blob base and the per-program
        // CSR by the key base. `lens` (incl. the 0 = "no host value" sentinel)
        // copy verbatim.
        let late_base = self.sampling_late_keys.len() as u32;
        let late_blob_base = self.sampling_late_blob.len() as u32;
        self.sampling_late_keys
            .extend_from_slice(&req.sampling_late_keys);
        self.sampling_late_blob
            .extend_from_slice(&req.sampling_late_blob);
        for &off in &req.sampling_late_offsets {
            self.sampling_late_offsets.push(late_blob_base + off);
        }
        self.sampling_late_lens
            .extend_from_slice(&req.sampling_late_lens);
        for &off in req.sampling_late_indptr.iter().skip(1) {
            self.sampling_late_indptr.push(late_base + off);
        }

        // Per-slot binding-map: concat the (kind, key) slots and rebase the
        // per-program CSR by the slot base.
        let binding_base = self.sampling_binding_kind.len() as u32;
        self.sampling_binding_kind
            .extend_from_slice(&req.sampling_binding_kind);
        self.sampling_binding_key
            .extend_from_slice(&req.sampling_binding_key);
        for &off in req.sampling_binding_indptr.iter().skip(1) {
            self.sampling_binding_indptr.push(binding_base + off);
        }

        // Output fast-path dst table (#27 cut #1): concat the per-output-value
        // pinned dst pointers + byte capacities, rebasing the per-program CSR by
        // the table base — exactly parallel to the `sampling_input_*` merge.
        // Without this the batch-merge would silently drop the host-populated
        // `sampling_output_*` → the driver's view sees an empty fast-path carrier.
        let output_base = self.sampling_output_dst_ptrs.len() as u32;
        self.sampling_output_dst_ptrs
            .extend_from_slice(&req.sampling_output_dst_ptrs);
        self.sampling_output_dst_lens
            .extend_from_slice(&req.sampling_output_dst_lens);
        for &off in req.sampling_output_indptr.iter().skip(1) {
            self.sampling_output_indptr.push(output_base + off);
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
