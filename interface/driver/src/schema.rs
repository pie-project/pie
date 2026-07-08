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

    // ── PTIR trace carrier (thrust-3 P2c) ───────────────────────────
    // A fired pass may carry PTIR *trace containers* (the programmable-dataflow
    // generalization of the Sampling-IR program): stage-tagged programs +
    // channel decls + descriptor ports, canonical bytes (`pie-sampling-ir::ptir`)
    // that the driver decodes + JIT-compiles + caches by `container_hash`.
    // Opaque to the bridge like `sampling_program_*`. All empty for the legacy /
    // Sampling-IR paths.
    //
    // Bytes-first-fire-of-hash: `ptir_program_hashes` is ALWAYS present (the C3
    // identity + driver compile-cache key). For program `p`, an EMPTY byte range
    // (`ptir_program_bytes_indptr[p+1] == [p]`) ⇒ the driver serves it from its
    // hash-keyed cache; a non-empty range ⇒ decode + compile + cache under that
    // hash. First-fire ships bytes; steady-state fires carry hashes only. An
    // empty-bytes cache MISS is a HARD protocol error (loud), and the host
    // re-ships bytes whenever ITS registry re-registers a hash (a fresh first
    // fire). Seeds vs host-puts stay SEPARATE tables — different lifecycle (seed
    // = per-instance init, host-put = per-fire input; D1-coalesced).
    //
    // INSTANCE IDENTITY (persistent-instance model): `ptir_program_instances` is
    // ALSO always present, one per program — the identity of the STATEFUL
    // instance this fire drives (the driver's per-instance channel arena). The
    // driver caches arenas keyed by this id: it constructs the arena on the first
    // fire of an instance (applying the `seeds` that ride that first fire) and
    // REUSES it thereafter, so channel state (counters, beam state) SURVIVES
    // across fires. `hash` selects the compiled program; `instance` selects the
    // arena — many instances of one `hash` (e.g. 8 concurrent beams) share the
    // compiled program but each hold independent channel state. Seeds ride the
    // instance's first fire (not the hash's); host-puts ride every fire.

    /// Per-program `container_hash` (C3 identity + driver compile-cache key);
    /// always present, `len` = programs in the fire's PTIR set.
    pub ptir_program_hashes: Vec<u64>,
    /// Per-program stateful-instance identity (the driver's channel-arena cache
    /// key); always present, parallel to `ptir_program_hashes`. Stable across an
    /// instance's fires; seeds ride its first fire, arena persists thereafter.
    pub ptir_program_instances: Vec<u64>,
    /// Concatenated container bytes; opaque to the bridge. Empty range for a
    /// program ⇒ hash-cache hit (steady state).
    pub ptir_program_bytes: Vec<u8>,
    /// Per-program byte CSR partitioning `ptir_program_bytes` (leading 0).
    pub ptir_program_bytes_indptr: Vec<u32>,
    /// Concatenated PTIB sidecars (`encode_bound` of the `BoundTrace` — the
    /// typed wire form the driver's stage-runner reads). Shipped ONLY on the
    /// first fire of a hash, alongside the container bytes (seed-independent,
    /// hash-keyed — same lifecycle as the bytes); empty range thereafter ⇒
    /// hash-cache hit.
    pub ptir_program_sidecar_bytes: Vec<u8>,
    /// Per-program byte CSR partitioning `ptir_program_sidecar_bytes` (leading 0).
    pub ptir_program_sidecar_indptr: Vec<u32>,

    /// Dense-channel-index → global channel id map, concatenated across
    /// programs. For program `p`, the global ids of its declared channels in
    /// dense order occupy `ptir_program_channel_ids[indptr[p]..indptr[p+1]]`;
    /// the driver binds the trace's dense channel references to the global
    /// device channel registry through it (multi-pass channel sharing). Always
    /// present alongside `ptir_program_hashes` under the global-id ABI.
    pub ptir_program_channel_ids: Vec<u64>,
    /// Per-program CSR into `ptir_program_channel_ids` (leading 0).
    pub ptir_program_channel_ids_indptr: Vec<u32>,

    /// Seed table (per-instance `Channel::from` values for `seeded` channels;
    /// per-instance data D2, NOT in the hash). Global channel id per entry.
    pub ptir_program_seed_channels: Vec<u64>,
    /// Concatenated seed value bytes.
    pub ptir_program_seed_blob: Vec<u8>,
    /// Byte length per seed entry (parallel to `ptir_program_seed_channels`).
    pub ptir_program_seed_lens: Vec<u32>,
    /// Per-program CSR into the seed entries (leading 0).
    pub ptir_program_seed_indptr: Vec<u32>,

    /// Host-put table (host `channel.put` inputs, D1-coalesced before submit).
    /// Global channel id per entry.
    pub ptir_program_host_put_channels: Vec<u64>,
    /// Concatenated host-put value bytes.
    pub ptir_program_host_put_blob: Vec<u8>,
    /// Byte length per host-put entry (parallel to `ptir_program_host_put_channels`).
    pub ptir_program_host_put_lens: Vec<u32>,
    /// Per-program CSR into the host-put entries (leading 0).
    pub ptir_program_host_put_indptr: Vec<u32>,

    /// Release markers (W0.3): global channel ids whose device storage the
    /// driver must free after serving this fire — sent when the runtime drops
    /// the last handle to a channel. Rides any request (or a heartbeat submit);
    /// order-independent, deduped host-side.
    pub ptir_release_channel_ids: Vec<u64>,
    /// Release markers: PTIR instance ids whose persistent channel-arena views
    /// the driver must free (fixes the pre-existing instance-map leak). Sent
    /// when the runtime drops a forward-pass.
    pub ptir_release_instance_ids: Vec<u64>,

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
    /// Drafts-channel routing (§9): which device SOURCE this producer link retains.
    /// `0 = PrevSample` (retain `pi.sampled[N]`, the single-token carrier — the
    /// back-compat default) / `1 = PrevDrafts` (retain the composed `[k+1]`
    /// `[seed, drafts]` window from the program's out[2]/out[1] — the MTP
    /// drafts-channel). Additive `u8` tag, pure routing (mirrors
    /// `SamplingBinding::MtpLogits(2)`-vs-`Logits(0)`). `pipeline_source_kinds`
    /// (below) is the per-request batched form.
    pub pipeline_source_kind: u8,
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

    // ── Programmable-sampling late-input device-alias channel (#27 cut #2) ───
    // Appended per the schema evolution rule (append-only). The input-side mirror
    // of `sampling_output_*`: for a program input declared `host{key, late-bound}`
    // (e.g. a grammar mask), the host pre-uploads the value DIRECTLY to a device
    // buffer (`pie_device_alloc` + `pie_tensor_write_async`, direct-FFI from the
    // guest's WASM-memory slice — no IPC staging, the input mirror of cut #1's
    // output D2H) and passes the resulting DEVICE pointer here instead of staging
    // host bytes via `sampling_late_blob`. The driver's `HostLate` resolution
    // reads `sampling_late_device_ptrs[late_key]` → the device-resident value,
    // gated on the R12 self-arm flag (`sampling_late_device_flags[late_key]`,
    // set stream-ordered after the H2D). `0` ⇒ no device-alias for that late key
    // (fall back to the staged `sampling_late_blob` path). Parallel to
    // `sampling_late_keys` (one entry per late key, concatenated across programs).
    // Security: the guest never sees a device pointer — it hands the host a slice,
    // the host does the memcpy.
    /// Per-late-key device-resident value pointer (raw device address, u64; `0` =
    /// no device-alias, use the staged blob). Parallel to `sampling_late_keys`.
    pub sampling_late_device_ptrs: Vec<u64>,
    /// Per-late-key R12 self-arm flag pointer (raw device address of the
    /// `pie_device_alloc`-co-allocated `u32` flag, set after the H2D; `0` = none).
    /// The driver waits this flag before the consuming kernel reads the value, so
    /// a not-yet-arrived value is a loud miss, never a stale-buffer silent read.
    pub sampling_late_device_flags: Vec<u64>,
    /// Per-late-key device-resident value byte length (parallel to
    /// `sampling_late_device_ptrs`; `0` when no device-alias). The staged-blob
    /// `sampling_late_lens` records `0` for device-alias entries (no host value),
    /// so this carries the device value's size — the executor memcpy's `len` bytes
    /// per row for the merged per-row gather (`[N, len]`) without coupling to the
    /// program's `InputDecl`/dtype (`elem_count` × dtype) to derive it.
    pub sampling_late_device_lens: Vec<u32>,

    // ── Length column (PTIR Thrust-1 M2a / contract C1) ─────────────────────
    // Appended per the schema evolution rule (append-only). Per-request KV
    // length (overview §5.1's `kv_len [B]`): the PHYSICAL span each lane
    // attends, in tokens = `(pages_r − 1)·page_size + kv_last_page_lens[r]`
    // (0 when the lane has no pages). Today the driver reconstructs this from
    // `kv_page_indptr` + `kv_last_page_lens`; this makes it a first-class NAMED
    // column so thrusts 2/3 can bind/produce it directly (device-resident in
    // M5). Frozen fork pages are counted FULL — sub-page validity rides the
    // attention mask, never this total (W6). Parallel to `kv_last_page_lens`
    // (one entry per request); empty ⇒ derive from page metadata (back-compat).
    pub kv_len: Vec<u32>,

    // ── Per-request next-input producer links (PTIR Thrust-2 Bug#2 fix) ──────
    // Appended per the schema evolution rule (append-only). Supersedes the SCALAR
    // `pipeline_source_link` for R>1 co-batched fires: one entry per request (0 =
    // not a producer), so EACH co-batched producer row retains its own sampled
    // token under its own global link — not just the last (the scalar merge kept
    // only the final request's link, so every earlier co-batched producer's token
    // was never retained → its consumer retain-missed → placeholder 0, the R>1
    // concurrent-decode corruption). The executor retains request `r`'s producer
    // row (`qo_indptr[r+1]-1`) under `pipeline_source_links[r]` when non-zero.
    // Empty ⇒ fall back to the scalar `pipeline_source_link` (back-compat / R=1).
    pub pipeline_source_links: Vec<u32>,
    // ── Per-request drafts-channel routing (§9) ──────────────────────────────
    // Parallel to `pipeline_source_links`: `pipeline_source_kinds[r]` routes the
    // retain SOURCE for request `r`'s producer link (`0 = PrevSample` / `1 =
    // PrevDrafts`). Empty ⇒ all `PrevSample(0)` (back-compat / the scalar
    // `pipeline_source_kind` for R=1). Additive; a `PrevDrafts` link retains the
    // program-composed `[k+1]` `[seed, drafts]` window instead of `pi.sampled[N]`.
    pub pipeline_source_kinds: Vec<u8>,

    // ── Geometry-as-data device handle (PTIR Thrust-1 M5 / contract C1) ──────
    // Appended per the schema evolution rule (append-only). C1-FINAL length
    // column: the base DEVICE address (u64) of a packed `[R]` `u32` vector whose
    // `[r]` entry is request `r`'s `kv_len`, PRODUCED by a PRIOR pass's kernel
    // (driver `launch_derive_kv_len`) into a device buffer the host never reads
    // (geometry-as-data, overview §6.1) — so a forward's geometry can be a
    // program value, never a host round-trip. ONE handle for the whole forward
    // (indexed `[r]` at bind, mirroring how the paged-KV page table is one bound
    // buffer): a per-request handle would fragment the bind (C1 co-design).
    // Carried as a 0-or-1-element `Vec<u64>` — NOT a bare `u64` — so the archived
    // `ForwardRequest` stays 4-aligned (rkyv's `ArchivedVec` is 4-aligned for any
    // element type; a bare `u64` field would force the whole struct to 8-align
    // and break the all-vecs ABI invariant, `pie-ipc` archived_layout). Empty ⇒
    // all lanes host-fed via the scalar `kv_len` column above (back-compat);
    // `[base]` ⇒ the single forward-level handle. The executor's late-bind reads
    // `((u32*)kv_len_device[0])[r]` in place of the host-fed scalar, stream-
    // ordered AFTER the producer kernel (same-stream ordering; a cross-stream
    // seam would add a self-arm flag like `sampling_late_device_flags`). Per-
    // request length `R` is implicit from the request count. Frozen fork pages
    // count FULL (W6), same convention as the host-fed `kv_len`.
    pub kv_len_device: Vec<u64>,

    // ── (a) BRIDGE carry-descriptor channel (X2/X3, guru-ruled) ──────────────────
    // Appended per the append-only schema evolution rule; all-vecs (4-aligned ABI —
    // see `kv_len_device` above; a bare scalar would 8-align + break `pie-ipc`
    // archived_layout, so even the single version rides a 0-or-1 `Vec`). The
    // per-request carrier descriptor for the direct driver↔inferlet frame transport:
    // `enqueue` (runtime) populates these; the CUDA executor reads them at a2
    // fire-commit and calls `pie_frame_carry(instance, carry_word_index[r],
    // committed_head[r], sample_done, <once-registered done>, carry_user_ptr[r])`.
    // ALL EMPTY ⇒ no carry channel (every non-bridge path / the mock — zero behavior
    // change). The completion callback (`cuda_carry_done`) is a stable static
    // registered ONCE (`pie_frame_set_carry_done`), NOT threaded per-request, so
    // only `{user_data, word_index}` are per-request here.
    //
    // PROVISIONAL shape (guru's shared-schema nod pending): SoA cols + a version
    // guard (this preserves guru's "unknown layouts loud-reject" ABI rule — the
    // executor validates `carry_abi_version[0] == CARRY_DESCRIPTOR_VERSION` before
    // trusting the cols — while giving the a2 read plain u64 cols, not a `Vec<u8>`
    // blob). Populate + the enqueue→ForwardRequest relocation is the bridge capstone.
    /// `[CARRY_DESCRIPTOR_VERSION]` when a carry channel is present, else empty. The
    /// executor loud-rejects a mismatch (guru's version guard). 0-or-1 element.
    pub carry_abi_version: Vec<u32>,
    /// Boxed `CarryWake` raw pointer per request (the `user_data` the carrier hands
    /// back to `cuda_carry_done`). Parallel to the request axis (like the a2
    /// `sampling_output_dst_ptrs`). Empty ⇒ no carry.
    pub carry_user_ptr: Vec<u64>,
    /// Pinned head `word_index` per request (`2*c` for channel `c`; word 0 today).
    /// Parallel to `carry_user_ptr`.
    pub carry_word_index: Vec<u64>,
    /// Bound instance id per request. a2 batches R requests each under its OWN bound
    /// instance, so the carrier's `pie_frame_carry(instance, …)` needs the
    /// per-request instance (NOT one per-fire). Parallel to `carry_user_ptr`.
    pub carry_instance: Vec<u64>,
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
///
/// ## Intrinsic-kind flow contract (edsl → manifest → driver)
///
/// The intrinsic KIND is carried out-of-band in the manifest — the bytecode is
/// binding-free and `Logits`/`MtpLogits`/`MtpDrafts` all dedup to the same
/// program identity — so this discriminant is the ONLY channel that distinguishes
/// them. The single wire path, slot-by-slot in slot order:
///
/// 1. **edsl → IR:** the builder carries each slot's `Binding` into
///    `Built::bindings` (intrinsics are excluded from host inputs, kept as
///    bindings). — `sdk/rust/sampling-edsl/src/builder.rs`
/// 2. **IR → wire:** the runtime maps `ir::Binding` → `SamplingBinding`, then
///    `push_sampling_program` emits [`SamplingBinding::kind()`](SamplingBinding::kind) into
///    `sampling_binding_kind` (Logits→0, Tensor→1, MtpLogits→2, MtpDrafts→3) and
///    [`key()`](SamplingBinding::key) into `sampling_binding_key`.
///    — `runtime/src/api/inference.rs`, `push_sampling_program` below.
/// 3. **wire → driver:** the executor reads `sampling_binding_kind[i]` and stamps
///    `InputBind.intrinsic_kind` (bk 2 ⇒ `Intrinsic::MtpLogits`); the JIT backend
///    (`manifest_to_slot_bindings`) MUST propagate that `intrinsic_kind` into the
///    compiled `BufferDecl` — dropping it collapses every intrinsic to the sampled
///    `Logits` row. — `driver/cuda/src/executor/executor.cpp`,
///    `driver/cuda/src/sampling_ir/codegen.cpp`.
///
/// Regression-locked by `sampling_binding_intrinsic_kind_encoded_per_slot`
/// (`runtime/src/inference/request.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingBinding {
    /// The LM-head logits intrinsic (sampling positions resolved host-side into
    /// `sampling_indices`).
    Logits,
    /// The speculator's **draft** logits intrinsic (de-hardwired speculation):
    /// source-selects the draft rows of `ws.logits` (M=1 ⇒ row 0) rather than a
    /// separate buffer. A distinct manifest kind (not a flag on `Logits`) so a
    /// stale reader loud-rejects rather than misparsing — mirrors
    /// [`Binding::MtpLogits`](pie_sampling_ir::Binding). Payload-less (key `0`).
    MtpLogits,
    /// The retained window's **draft token ids** intrinsic (device-resident MTP
    /// spec-decode): source-selects the retained `mtp_drafts` `[k]` i32 buffer
    /// (bravo's per-link retain, rows `1..=k` of the `[k+1]` window) as the
    /// verify's `draft` operand — the device analog of a submit `draft` tensor.
    /// Distinct manifest kind (mirrors `MtpLogits`); payload-less (key `0`).
    /// I32 `[k]` (NOT `[k,vocab]` f32 like `MtpLogits`).
    MtpDrafts,
    /// A submit-bound tensor value keyed into the submit-input table.
    Tensor { key: u32 },
}

impl SamplingBinding {
    /// Wire discriminant for a `Logits` slot.
    pub const KIND_LOGITS: u8 = 0;
    /// Wire discriminant for a `Tensor` slot.
    pub const KIND_TENSOR: u8 = 1;
    /// Wire discriminant for a `MtpLogits` (draft-logits intrinsic) slot.
    pub const KIND_MTP_LOGITS: u8 = 2;
    /// Wire discriminant for a `MtpDrafts` (draft-token-ids intrinsic) slot.
    pub const KIND_MTP_DRAFTS: u8 = 3;

    /// The `sampling_binding_kind` discriminant for this binding.
    pub fn kind(self) -> u8 {
        match self {
            SamplingBinding::Logits => Self::KIND_LOGITS,
            SamplingBinding::MtpLogits => Self::KIND_MTP_LOGITS,
            SamplingBinding::MtpDrafts => Self::KIND_MTP_DRAFTS,
            SamplingBinding::Tensor { .. } => Self::KIND_TENSOR,
        }
    }

    /// The `sampling_binding_key` value (the TensorKey, or `0` for the
    /// payload-less `Logits`/`MtpLogits`/`MtpDrafts` intrinsics).
    pub fn key(self) -> u32 {
        match self {
            SamplingBinding::Logits | SamplingBinding::MtpLogits | SamplingBinding::MtpDrafts => 0,
            SamplingBinding::Tensor { key } => key,
        }
    }

    /// Reconstruct from the wire `(kind, key)` pair (unknown kind ⇒ `Logits`).
    pub fn from_parts(kind: u8, key: u32) -> SamplingBinding {
        match kind {
            Self::KIND_TENSOR => SamplingBinding::Tensor { key },
            Self::KIND_MTP_LOGITS => SamplingBinding::MtpLogits,
            Self::KIND_MTP_DRAFTS => SamplingBinding::MtpDrafts,
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

/// One per-channel value (a seed or a host-put) in a [`PtirProgramSubmission`].
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct PtirChannelValue {
    /// Global channel id (runtime-minted, inferlet-scoped) this value targets.
    /// Re-keyed from the old program-local dense index so a channel bound to
    /// several passes routes to one device cell (multi-pass channels).
    pub channel: u64,
    /// Raw little-endian value bytes (packed per the channel's dtype; `bool`
    /// travels packed, D1).
    pub bytes: Vec<u8>,
}

/// One PTIR trace container fired on a pass (thrust-3 P2c). Carries the identity
/// hash always; the container bytes only on a first fire of that hash
/// (steady-state fires ship `bytes = None` ⇒ the driver serves from its
/// hash-keyed compile cache). Seeds (per-instance init) and host-puts (per-fire,
/// D1-coalesced) ride separate tables — different lifecycle.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct PtirProgramSubmission {
    /// `container_hash` — the C3 identity + driver compile-cache key.
    pub hash: u64,
    /// The stateful instance this fire drives — the driver's channel-arena cache
    /// key (persistent-instance model). Stable across the instance's fires; the
    /// arena is built on the instance's first fire (binding `seeds`) and reused
    /// thereafter. Many instances of one `hash` hold independent channel state.
    pub instance: u64,
    /// Container bytes on a first fire; `None` ⇒ steady-state (hash-cache hit).
    pub bytes: Option<Vec<u8>>,
    /// PTIB sidecar (`encode_bound`) on a first fire, alongside `bytes` (same
    /// hash-keyed lifecycle); `None` ⇒ steady state.
    pub sidecar: Option<Vec<u8>>,
    /// Dense-channel-index → global channel id map (one entry per declared
    /// channel, in dense order). Lets the driver bind this program's trace-local
    /// channel references to the global device channel registry; identical for
    /// every fire of the instance. Empty on a legacy submission that predates
    /// global ids (the driver then treats seed/host-put `channel` as dense).
    pub channel_ids: Vec<u64>,
    /// Per-instance seed values for `seeded` channels (D2; not in the hash).
    pub seeds: Vec<PtirChannelValue>,
    /// Host-put channel inputs (coalesced before submit).
    pub host_puts: Vec<PtirChannelValue>,
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

    /// Append one PTIR trace container to this request's SoA carrier (thrust-3
    /// P2c). Ships the hash always; the bytes only when `prog.bytes` is `Some`
    /// (first fire) — a `None` records an EMPTY byte range so the driver serves
    /// the hash from its compile cache. Seeds + host-puts go to their separate
    /// per-program tables.
    pub fn push_ptir_program(&mut self, prog: &PtirProgramSubmission) {
        // Leading-0 for the nested per-program CSRs.
        if self.ptir_program_bytes_indptr.is_empty() {
            self.ptir_program_bytes_indptr.push(0);
        }
        if self.ptir_program_sidecar_indptr.is_empty() {
            self.ptir_program_sidecar_indptr.push(0);
        }
        if self.ptir_program_channel_ids_indptr.is_empty() {
            self.ptir_program_channel_ids_indptr.push(0);
        }
        if self.ptir_program_seed_indptr.is_empty() {
            self.ptir_program_seed_indptr.push(0);
        }
        if self.ptir_program_host_put_indptr.is_empty() {
            self.ptir_program_host_put_indptr.push(0);
        }

        self.ptir_program_hashes.push(prog.hash);
        self.ptir_program_instances.push(prog.instance);

        // Bytes + sidecar: first fire extends; steady-state leaves an empty range.
        if let Some(bytes) = &prog.bytes {
            self.ptir_program_bytes.extend_from_slice(bytes);
        }
        self.ptir_program_bytes_indptr
            .push(self.ptir_program_bytes.len() as u32);
        if let Some(sidecar) = &prog.sidecar {
            self.ptir_program_sidecar_bytes.extend_from_slice(sidecar);
        }
        self.ptir_program_sidecar_indptr
            .push(self.ptir_program_sidecar_bytes.len() as u32);

        // Dense-index → global-id map (one entry per declared channel).
        self.ptir_program_channel_ids
            .extend_from_slice(&prog.channel_ids);
        self.ptir_program_channel_ids_indptr
            .push(self.ptir_program_channel_ids.len() as u32);

        for s in &prog.seeds {
            self.ptir_program_seed_channels.push(s.channel);
            self.ptir_program_seed_blob.extend_from_slice(&s.bytes);
            self.ptir_program_seed_lens.push(s.bytes.len() as u32);
        }
        self.ptir_program_seed_indptr
            .push(self.ptir_program_seed_channels.len() as u32);

        for h in &prog.host_puts {
            self.ptir_program_host_put_channels.push(h.channel);
            self.ptir_program_host_put_blob.extend_from_slice(&h.bytes);
            self.ptir_program_host_put_lens.push(h.bytes.len() as u32);
        }
        self.ptir_program_host_put_indptr
            .push(self.ptir_program_host_put_channels.len() as u32);
    }

    /// Number of PTIR containers carried (`= ptir_program_hashes.len()`). Zero
    /// for the legacy / Sampling-IR paths.
    #[inline]
    pub fn ptir_program_count(&self) -> usize {
        self.ptir_program_hashes.len()
    }

    /// Reconstruct PTIR program `p`'s submission (round-trip of
    /// [`push_ptir_program`](Self::push_ptir_program); the driver's reference
    /// reader). `None` if `p` is out of range.
    pub fn ptir_program_at(&self, p: usize) -> Option<PtirProgramSubmission> {
        let hash = *self.ptir_program_hashes.get(p)?;
        let instance = *self.ptir_program_instances.get(p)?;
        let blo = *self.ptir_program_bytes_indptr.get(p)? as usize;
        let bhi = *self.ptir_program_bytes_indptr.get(p + 1)? as usize;
        let bytes = if bhi > blo { Some(self.ptir_program_bytes[blo..bhi].to_vec()) } else { None };
        let slo = *self.ptir_program_sidecar_indptr.get(p)? as usize;
        let shi = *self.ptir_program_sidecar_indptr.get(p + 1)? as usize;
        let sidecar =
            if shi > slo { Some(self.ptir_program_sidecar_bytes[slo..shi].to_vec()) } else { None };

        // Dense-index → global-id map for this program.
        let clo = *self.ptir_program_channel_ids_indptr.get(p).unwrap_or(&0) as usize;
        let chi = *self.ptir_program_channel_ids_indptr.get(p + 1).unwrap_or(&0) as usize;
        let channel_ids = self
            .ptir_program_channel_ids
            .get(clo..chi)
            .map(|s| s.to_vec())
            .unwrap_or_default();

        let read_table = |indptr: &[u32], channels: &[u64], blob: &[u8], lens: &[u32]| {
            let lo = *indptr.get(p).unwrap_or(&0) as usize;
            let hi = *indptr.get(p + 1).unwrap_or(&0) as usize;
            let mut byte_off: usize =
                lens.iter().take(lo).map(|&l| l as usize).sum();
            let mut out = Vec::new();
            for e in lo..hi {
                let len = lens[e] as usize;
                out.push(PtirChannelValue {
                    channel: channels[e],
                    bytes: blob[byte_off..byte_off + len].to_vec(),
                });
                byte_off += len;
            }
            out
        };
        let seeds = read_table(
            &self.ptir_program_seed_indptr,
            &self.ptir_program_seed_channels,
            &self.ptir_program_seed_blob,
            &self.ptir_program_seed_lens,
        );
        let host_puts = read_table(
            &self.ptir_program_host_put_indptr,
            &self.ptir_program_host_put_channels,
            &self.ptir_program_host_put_blob,
            &self.ptir_program_host_put_lens,
        );
        Some(PtirProgramSubmission { hash, instance, bytes, sidecar, channel_ids, seeds, host_puts })
    }

    /// Concatenate every PTIR program from `req` into `self` — the batch fold's
    /// ptir analogue of [`extend_sampling_programs_from`]. Reuses
    /// [`push_ptir_program`]'s cumulative-CSR offsetting via [`ptir_program_at`].
    /// Without this the scheduler's per-request batch merge (`append_request`)
    /// drops `ptir_program_*` and the driver sees an EMPTY carrier (hook never
    /// fires — the §6.2 e2e's `ptir_hashes=0` at forward-serve entry).
    pub fn extend_ptir_programs_from(&mut self, req: &ForwardRequest) {
        for p in 0..req.ptir_program_count() {
            if let Some(sub) = req.ptir_program_at(p) {
                self.push_ptir_program(&sub);
            }
        }
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

    /// Route this producer link's retain SOURCE (drafts-channel §9): `0 =
    /// PrevSample` (retain `pi.sampled[N]`) / `1 = PrevDrafts` (retain the composed
    /// `[k+1]` `[seed, drafts]` window). The guest surface is echo's
    /// `carrier::next_inputs_drafts` + `set_pipeline_source_kind(1)`; a producer
    /// that never calls it defaults to `PrevSample(0)` (byte-identical).
    #[inline]
    pub fn set_pipeline_source_kind(&mut self, kind: u8) {
        self.pipeline_source_kind = kind;
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
        // #27 cut #2 device-alias channel: parallel to `sampling_late_keys` (one
        // device ptr + R12 flag per late key) → concat verbatim, exactly like
        // `sampling_late_lens`. Empty (legacy staged path) stays empty.
        self.sampling_late_device_ptrs
            .extend_from_slice(&req.sampling_late_device_ptrs);
        self.sampling_late_device_flags
            .extend_from_slice(&req.sampling_late_device_flags);
        self.sampling_late_device_lens
            .extend_from_slice(&req.sampling_late_device_lens);

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

        // (a) BRIDGE carry-descriptor cols: concat per-request. `carry_abi_version`
        // is a single per-fire value (same ABI across reqs) — adopt the req's when
        // present. Empty req cols ⇒ nothing appended (that lane has no carry).
        self.carry_user_ptr.extend_from_slice(&req.carry_user_ptr);
        self.carry_word_index
            .extend_from_slice(&req.carry_word_index);
        self.carry_instance.extend_from_slice(&req.carry_instance);
        if self.carry_abi_version.is_empty() && !req.carry_abi_version.is_empty() {
            self.carry_abi_version = req.carry_abi_version.clone();
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
    /// Thrust-2 bubble-p50 gate: the DEVICE-idle inter-fire bubble measured
    /// AT THE DRIVER — the gap from the previous fire's kernel-retire (post final
    /// stream sync) to this fire's entry, before any GPU op. Unlike the runtime's
    /// host proxy (`inter_batch_bubble_us`, stamped when the scheduler *receives*
    /// the completion → IPC-lagged, over-counts), this is stamped inside the driver
    /// so it isolates the true GPU-idle gap for the `< 100 µs` gate. `0` on the
    /// first fire (no prior retire to measure against).
    pub probe_device_idle_us: u32,

    /// #32 per-(request,output) `[k]`-Token output CSR — a **two-level CSR**
    /// (mirrors `logprobs_req_indptr`/`logprobs_val_indptr`/`logprobs_values`), so
    /// `extract_per_request` can slice it per request like every other channel.
    ///
    /// `program_tokens_req_indptr` (`num_requests + 1`) partitions the output-slot
    /// axis by request: request `r`'s output slots are
    /// `[program_tokens_req_indptr[r] .. program_tokens_req_indptr[r+1])`.
    /// `program_tokens_indptr` then has one entry per (request, output) slot in
    /// row-major (request-then-output) order, partitioning `program_tokens`:
    /// output slot `s`'s tokens are `program_tokens[indptr[s]..indptr[s+1]]`.
    /// Non-empty ONLY for `[k]`-Token outputs (`elem_count > 1`); single-Token /
    /// Scalar / Logits leave an empty segment (single-Token stays the dense
    /// `tokens` path). Routes `[k]`-Token off the spec-decode `spec_tokens` channel.
    pub program_tokens_req_indptr: Vec<u32>,
    pub program_tokens_indptr: Vec<u32>,
    /// `[k]`-Token output values, concatenated in `program_tokens_indptr` order.
    pub program_tokens: Vec<u32>,

    // ── PTIR program outputs (thrust-3 P2c-fire) ────────────────────────
    // The host-visible Reader-channel cells a PTIR pass produced, harvested by
    // the driver's stage-runner (`PtirInstance::harvest_outputs`) and marshaled
    // back to the host channel store (`ChannelStore::marshal_response`). SoA +
    // per-program CSR, mirroring the REQUEST's host-put table exactly; program
    // `p` here corresponds 1:1 to program `p` in the request's PTIR set
    // (`ptir_program_hashes[p]`/`ptir_program_instances[p]`). Empty for the
    // legacy / Sampling-IR paths. Bool cells travel packed (D1), same as the
    // host-put wire.
    /// Produced-channel global id per output entry (a host-facing Reader channel).
    pub ptir_output_channels: Vec<u64>,
    /// Concatenated produced-cell value bytes (bool packed per the dtype, D1).
    pub ptir_output_blob: Vec<u8>,
    /// Byte length per output entry (parallel to `ptir_output_channels`).
    pub ptir_output_lens: Vec<u32>,
    /// Per-program CSR into the output entries (leading 0), one boundary per
    /// program in the fire's PTIR set.
    pub ptir_output_indptr: Vec<u32>,
}

impl ForwardResponse {
    /// Append one PTIR program's produced Reader-channel cells to the output SoA
    /// (thrust-3 P2c-fire) — the response-side mirror of
    /// [`ForwardRequest::push_ptir_program`]'s host-put table. Called once per
    /// program in the fire's PTIR set, in the same order as the request.
    pub fn push_ptir_output(&mut self, outputs: &[PtirChannelValue]) {
        if self.ptir_output_indptr.is_empty() {
            self.ptir_output_indptr.push(0);
        }
        for o in outputs {
            self.ptir_output_channels.push(o.channel);
            self.ptir_output_blob.extend_from_slice(&o.bytes);
            self.ptir_output_lens.push(o.bytes.len() as u32);
        }
        self.ptir_output_indptr.push(self.ptir_output_channels.len() as u32);
    }

    /// Number of PTIR programs whose outputs are carried (`= ptir_output_indptr`
    /// boundaries minus the leading 0). Zero for the legacy / Sampling-IR paths.
    #[inline]
    pub fn ptir_output_count(&self) -> usize {
        self.ptir_output_indptr.len().saturating_sub(1)
    }

    /// Decode PTIR program `p`'s produced Reader-channel cells (round-trip of
    /// [`push_ptir_output`](Self::push_ptir_output); the host's reference reader,
    /// fed straight into `ChannelStore::marshal_response`). `None` if `p` is out
    /// of range.
    pub fn ptir_output_at(&self, p: usize) -> Option<Vec<PtirChannelValue>> {
        let lo = *self.ptir_output_indptr.get(p)? as usize;
        let hi = *self.ptir_output_indptr.get(p + 1)? as usize;
        let mut byte_off: usize =
            self.ptir_output_lens.iter().take(lo).map(|&l| l as usize).sum();
        let mut out = Vec::with_capacity(hi - lo);
        for e in lo..hi {
            let len = self.ptir_output_lens[e] as usize;
            out.push(PtirChannelValue {
                channel: self.ptir_output_channels[e],
                bytes: self.ptir_output_blob[byte_off..byte_off + len].to_vec(),
            });
            byte_off += len;
        }
        Some(out)
    }
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

#[cfg(test)]
mod ptir_carrier_tests {
    use super::{ForwardRequest, PtirChannelValue, PtirProgramSubmission};

    fn cv(channel: u64, bytes: &[u8]) -> PtirChannelValue {
        PtirChannelValue { channel, bytes: bytes.to_vec() }
    }

    #[test]
    fn push_ptir_program_roundtrips() {
        let a = PtirProgramSubmission {
            hash: 0xAAAA_0000_0000_0001,
            instance: 1, // instance A's first fire ships bytes + sidecar + seeds
            bytes: Some(vec![1, 2, 3, 4, 5]), // first fire ships bytes
            sidecar: Some(vec![9, 8, 7]),     // + the PTIB sidecar, same lifecycle
            channel_ids: vec![100, 101, 102, 103], // dense idx -> global id
            seeds: vec![cv(100, &10u32.to_le_bytes()), cv(103, &[1, 0])],
            host_puts: vec![cv(102, &[0xFF, 0x00, 0x00, 0x00])],
        };
        // steady-state fire of another instance/hash: no bytes/seeds, driver
        // serves the program from cache + drives the already-built arena.
        let b = PtirProgramSubmission {
            hash: 0xBBBB_0000_0000_0002,
            instance: 2,
            bytes: None,
            sidecar: None,
            channel_ids: vec![200, 201],
            seeds: vec![],
            host_puts: vec![cv(201, &7u32.to_le_bytes())],
        };

        let mut req = ForwardRequest::default();
        req.push_ptir_program(&a);
        req.push_ptir_program(&b);

        assert_eq!(req.ptir_program_count(), 2);
        // `hashes` + `instances` always present; `b`'s empty byte range = hit.
        assert_eq!(req.ptir_program_hashes, vec![a.hash, b.hash]);
        assert_eq!(req.ptir_program_instances, vec![a.instance, b.instance]);
        assert_eq!(*req.ptir_program_bytes_indptr.last().unwrap(), 5, "only `a` shipped bytes");

        assert_eq!(req.ptir_program_at(0), Some(a), "program 0 round-trips exactly");
        assert_eq!(req.ptir_program_at(1), Some(b), "program 1 (steady-state, bytes=None)");
        assert_eq!(req.ptir_program_at(2), None, "out of range");
    }

    #[test]
    fn empty_carrier_is_legacy_clean() {
        // A request with no PTIR programs (legacy / Sampling-IR path) reads empty.
        let req = ForwardRequest::default();
        assert_eq!(req.ptir_program_count(), 0);
        assert_eq!(req.ptir_program_at(0), None);
    }

    #[test]
    fn push_ptir_output_roundtrips() {
        use super::ForwardResponse;
        // program 0 produced 2 cells (out=token, plus a scalar), program 1 one.
        let mut resp = ForwardResponse::default();
        resp.push_ptir_output(&[cv(1, &5u32.to_le_bytes()), cv(2, &[0xAB])]);
        resp.push_ptir_output(&[cv(1, &9u32.to_le_bytes())]);

        assert_eq!(resp.ptir_output_count(), 2);
        assert_eq!(
            resp.ptir_output_at(0),
            Some(vec![cv(1, &5u32.to_le_bytes()), cv(2, &[0xAB])]),
            "program 0's produced cells round-trip in order"
        );
        assert_eq!(resp.ptir_output_at(1), Some(vec![cv(1, &9u32.to_le_bytes())]));
        assert_eq!(resp.ptir_output_at(2), None, "out of range");
    }

    #[test]
    fn empty_ptir_output_is_legacy_clean() {
        use super::ForwardResponse;
        let resp = ForwardResponse::default();
        assert_eq!(resp.ptir_output_count(), 0);
        assert_eq!(resp.ptir_output_at(0), None);
    }
}
