//! Per-request and batched forward-pass helpers built on the canonical
//! schema types.
//!
//! The wire schema lives in `pie_driver_abi` (rkyv-derived Rust structs).
//! Pie reuses [`pie_driver_abi::ForwardRequest`] and [`pie_driver_abi::ForwardResponse`]
//! directly at every stage:
//!
//! - [`new_per_request`] builds a single-request `ForwardRequest`
//!   (indptrs `[0, N]`, empty kv pages).
//! - [`new_batched_forward_request`] + [`append_request`] fold per-request
//!   forms into a batched `ForwardRequest`.
//! - [`forward_frame`] wraps the batched request in a routable Frame.
//! - [`extract_per_request`] scatters a batched `ForwardResponse` back
//!   into single-request slices, one per inferlet response future.

use smallvec::{SmallVec, smallvec};

/// Stable per-adapter id (was `crate::adapter::AdapterId`; the adapter service
/// was removed — LoRA is absorbed into PTIR). Kept as the driver-ABI
/// `AdapterBinding.adapter_id` carrier type.
type AdapterId = u64;
use crate::arena::PhysicalPageId;
use crate::driver::DriverId;
use pie_driver_abi::Brle;

/// Build a per-request [`pie_driver_abi::ForwardRequest`].
///
/// `kv_page_indices` and `kv_last_page_lens` are left empty — the
/// scheduler fills them in during batching from the physical page list
/// it resolves out-of-band. `qo_indptr` and the other indptrs encode
/// the single-element shape (`[0, N]`) so the value is a valid
/// `ForwardRequest` on its own should anyone want to send it that way.
#[allow(clippy::too_many_arguments)]
pub fn new_per_request(
    context_id: u64,
    tokens: Vec<u32>,
    positions: Vec<u32>,
    masks: Vec<Brle>,
    has_user_mask: bool,
    logit_mask: Option<Brle>,
    sampling_indices: Vec<u32>,
    samplers: Vec<pie_driver_abi::Sampler>,
    speculative_tokens: Vec<u32>,
    speculative_positions: Vec<u32>,
    output_speculative_tokens: bool,
    adapter_id: Option<AdapterId>,
    adapter_seed: Option<i64>,
) -> pie_driver_abi::ForwardRequest {
    let n_tokens = tokens.len() as u32;
    let n_masks = masks.len() as u32;
    let n_sampling = sampling_indices.len() as u32;
    let n_samplers = samplers.len() as u32;
    let n_spec = speculative_tokens.len() as u32;
    let logit_masks: Vec<Brle> = logit_mask.into_iter().collect();
    let n_logit = logit_masks.len() as u32;
    let single_token_mode = !has_user_mask && n_tokens <= 1;

    let mut fr = pie_driver_abi::ForwardRequest {
        token_ids: tokens,
        position_ids: positions,
        kv_page_indices: Vec::new(),
        kv_page_indptr: vec![0],
        kv_last_page_lens: Vec::new(),
        qo_indptr: vec![0, n_tokens],
        rs_slot_ids: Vec::new(),
        rs_slot_flags: Vec::new(),
        rs_fold_lens: Vec::new(),
        masks,
        mask_indptr: vec![0, n_masks],
        logit_masks,
        logit_mask_indptr: vec![0, n_logit],
        sampling_indices,
        sampling_indptr: vec![0, n_sampling],
        // Sampler SoA filled by `set_samplers` below (Default leaves them empty).
        sampler_indptr: vec![0, n_samplers],
        adapter_bindings: vec![pie_driver_abi::AdapterBinding {
            adapter_id: adapter_id.map(|id| id as i64).unwrap_or(-1),
            seed: adapter_seed.unwrap_or(-1),
        }],
        spec_token_ids: speculative_tokens,
        spec_position_ids: speculative_positions,
        spec_indptr: vec![0, n_spec],
        output_spec_flags: vec![output_speculative_tokens],
        context_ids: vec![context_id],
        single_token_mode,
        has_user_mask,
        // Text-only builder: no visual spans. CSR roots start with a leading 0;
        // image_indptr is the single-request form [0, 0].
        image_indptr: vec![0, 0],
        image_grids: Vec::new(),
        image_anchor_positions: Vec::new(),
        image_pixels: Vec::new(),
        image_pixel_indptr: vec![0],
        image_mrope_positions: Vec::new(),
        image_mrope_indptr: vec![0],
        image_patch_positions: Vec::new(),
        image_anchor_rows: Vec::new(),
        // Text-only builder: no audio spans. CSR roots start with a leading 0.
        audio_features: Vec::new(),
        audio_feature_indptr: vec![0],
        audio_anchor_rows: Vec::new(),
        audio_indptr: vec![0, 0],
        ..Default::default()
    };
    fr.set_samplers(&samplers);
    fr
}

/// Inline storage for the page-trim bitmap. Sized to cover up to 1024 pages
/// (16 u64 words = 128 bytes, fits in one cache line per word) without ever
/// touching the heap. Larger contexts spill to the heap transparently.
const TRIM_INLINE_WORDS: usize = 16;
type TrimBits = SmallVec<[u64; TRIM_INLINE_WORDS]>;

// =============================================================================
// Page-trim plan
// =============================================================================
//
// When every query row of a request's attention mask agrees that an entire
// page's worth of KV positions is False, that physical page can be excluded
// from the wire-format `kv_page_indices` — the kernel reads fewer KV slots
// and the BRLE rows get sliced down to match. This is a pure performance
// optimization with no semantic change: position IDs of input tokens are
// unaffected (RoPE is independent of page list shape) and the page-hash
// chain used by the radix-trie dedup operates on `req.masks` upstream of
// this point, so trimming the wire copy doesn't perturb caching.
//
// The eligibility window stops at `first_writeable_page` — pages that the
// kernel will write new K/V into this pass cannot be dropped even if the
// mask says all-False, because the kernel's write target is determined by
// position-in-`kv_page_indices`.
//
// Eligibility math:
//   total_kv = (num_pages - 1) * page_size + last_page_len   (post-pass)
//   kv_before = total_kv - tokens.len()                       (pre-pass)
//   first_writeable_page = kv_before / page_size

/// A computed trim plan for a single request: which pages to drop and the
/// corresponding bit ranges to slice out of every BRLE row.
struct TrimPlan {
    /// Bitmask over `[0, num_pages)`: bit p set ⇒ page p is dropped.
    dropped_bits: TrimBits,
    /// Sorted disjoint `[s, e)` ranges in original-coord space, one per
    /// dropped page: `[p*page_size, (p+1)*page_size)`. Passed to
    /// `Brle::write_skipping` for each row.
    skip_ranges: Vec<(u32, u32)>,
}

impl TrimPlan {
    /// Compute the trim plan, or return `None` if no pages can be dropped.
    /// Returning `None` means the caller should take the fast path with
    /// zero extra allocations.
    fn compute(
        masks: &[Brle],
        num_pages: u32,
        last_page_len: u32,
        page_size: u32,
        num_input_tokens: u32,
    ) -> Option<Self> {
        if num_pages == 0 || page_size == 0 || masks.is_empty() {
            return None;
        }

        // Eligibility window: only pages strictly before the first page that
        // receives new K/V writes are candidates. last_page_len reflects the
        // post-pass state for non-spec input tokens; subtracting num_input_tokens
        // yields the pre-pass kv length. Speculative tokens write past
        // last_page_len into reserved pages, which are also writeable, but
        // they live in pages >= first_writeable_page either way so they don't
        // affect the cutoff.
        let total_kv = (num_pages - 1) * page_size + last_page_len;
        let kv_before = total_kv.saturating_sub(num_input_tokens);
        let first_writeable_page = kv_before / page_size;
        if first_writeable_page == 0 {
            return None;
        }

        let total_seq_len = total_kv;
        let num_words = ((num_pages as usize) + 63) / 64;

        // Running eligibility: AND-reduction across rows, seeded with the
        // writeable-window mask. SmallVec keeps both bitmaps inline on the
        // stack for typical `num_pages <= TRIM_INLINE_WORDS * 64` (1024).
        let mut eligible: TrimBits = smallvec![0u64; num_words];
        pie_driver_abi::brle::set_bits(&mut eligible, 0, first_writeable_page);

        let mut row_bits: TrimBits = smallvec![0u64; num_words];
        for mask in masks {
            for w in row_bits.iter_mut() {
                *w = 0;
            }
            mask.droppable_page_bits(page_size, num_pages, total_seq_len, &mut row_bits);
            for (e, r) in eligible.iter_mut().zip(row_bits.iter()) {
                *e &= *r;
            }
            // Early exit: once eligibility hits zero, no further rows can
            // bring it back. Common case for non-causal masks where rows
            // disagree on which pages are reachable.
            if eligible.iter().all(|&w| w == 0) {
                return None;
            }
        }

        // Materialize skip_ranges in page order. Walk set bits LSB-first per
        // word; each set bit p contributes [p*page_size, (p+1)*page_size).
        let mut skip_ranges: Vec<(u32, u32)> = Vec::new();
        for (w_idx, &word) in eligible.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let lsb = bits.trailing_zeros();
                let p = (w_idx as u32) * 64 + lsb;
                if p >= num_pages {
                    break;
                }
                skip_ranges.push((p * page_size, (p + 1) * page_size));
                bits &= bits.wrapping_sub(1);
            }
        }

        Some(TrimPlan {
            dropped_bits: eligible,
            skip_ranges,
        })
    }

    #[inline]
    fn is_page_dropped(&self, p: u32) -> bool {
        let w = (p / 64) as usize;
        let b = p % 64;
        self.dropped_bits
            .get(w)
            .map(|word| (word >> b) & 1 != 0)
            .unwrap_or(false)
    }
}

// =============================================================================
// Batched-request accumulator (free functions on pie_driver_abi::ForwardRequest)
// =============================================================================

/// Initialize a `pie_driver_abi::ForwardRequest` for the empty-batch state:
/// indptrs seeded with `[0]` so subsequent `append_request` calls can
/// push the rolling totals. `single_token_mode` starts at `true`; the
/// first per-request append that needs `custom_mask` flips it to false.
pub fn new_batched_forward_request() -> pie_driver_abi::ForwardRequest {
    new_batched_forward_request_with_capacity(0)
}

/// Same as [`new_batched_forward_request`] but pre-allocates Vec
/// capacities based on an expected request count, eliminating
/// per-append reallocations during `batch_build_us`. Pass 0 if you
/// don't know.
pub fn new_batched_forward_request_with_capacity(n_requests: usize) -> pie_driver_abi::ForwardRequest {
    // Per-request indptrs grow by exactly 1 entry. Pages, tokens,
    // samplers grow by at most a small multiple per request; the
    // estimates here are upper bounds for typical decode/prefill
    // shapes (page_size 16, ≤16 input tokens, ≤32 pages per req).
    let indptr_cap = n_requests + 1;
    let req_cap = n_requests;
    let token_cap = n_requests.saturating_mul(16);
    let page_cap = n_requests.saturating_mul(32);
    let indptr = |cap: usize| {
        let mut v = Vec::with_capacity(cap);
        v.push(0);
        v
    };
    pie_driver_abi::ForwardRequest {
        token_ids: Vec::with_capacity(token_cap),
        position_ids: Vec::with_capacity(token_cap),
        kv_page_indices: Vec::with_capacity(page_cap),
        kv_page_indptr: indptr(indptr_cap),
        kv_last_page_lens: Vec::with_capacity(req_cap),
        kv_len: Vec::with_capacity(req_cap),
        kv_len_device: Vec::new(),
        qo_indptr: indptr(indptr_cap),
        rs_slot_ids: Vec::with_capacity(req_cap),
        rs_slot_flags: Vec::with_capacity(req_cap),
        rs_fold_lens: Vec::with_capacity(req_cap),
        rs_buffer_slot_ids: Vec::new(),
        rs_buffer_slot_indptr: indptr(indptr_cap),
        masks: Vec::new(),
        mask_indptr: indptr(indptr_cap),
        logit_masks: Vec::new(),
        logit_mask_indptr: indptr(indptr_cap),
        sampling_indices: Vec::with_capacity(req_cap),
        sampling_indptr: indptr(indptr_cap),
        // Sampler SoA — one entry per slot, grown by `extend_samplers_from`.
        sampler_kinds: Vec::with_capacity(req_cap),
        sampler_temperatures: Vec::with_capacity(req_cap),
        sampler_top_k: Vec::with_capacity(req_cap),
        sampler_p: Vec::with_capacity(req_cap),
        sampler_seeds: Vec::with_capacity(req_cap),
        sampler_num_tokens: Vec::with_capacity(req_cap),
        sampler_token_ids: Vec::new(),
        sampler_token_ids_indptr: indptr(req_cap + 1),
        sampler_indptr: indptr(indptr_cap),
        adapter_bindings: Vec::with_capacity(req_cap),
        spec_token_ids: Vec::new(),
        spec_position_ids: Vec::new(),
        spec_indptr: indptr(indptr_cap),
        output_spec_flags: Vec::with_capacity(req_cap),
        context_ids: Vec::with_capacity(req_cap),
        single_token_mode: true,
        has_user_mask: false,
        // Image side-channel: per-request `image_indptr` and the per-image
        // pixel/mrope indptrs each start with a leading 0 (grown by the merge).
        image_indptr: indptr(indptr_cap),
        image_grids: Vec::new(),
        image_anchor_positions: Vec::new(),
        image_pixels: Vec::new(),
        image_pixel_indptr: indptr(indptr_cap),
        image_mrope_positions: Vec::new(),
        image_mrope_indptr: indptr(indptr_cap),
        image_patch_positions: Vec::new(),
        image_anchor_rows: Vec::new(),
        // Audio side-channel: per-request `audio_indptr` + per-clip feature
        // indptr each start with a leading 0 (grown by the merge).
        audio_features: Vec::new(),
        audio_feature_indptr: indptr(indptr_cap),
        audio_anchor_rows: Vec::new(),
        audio_indptr: indptr(indptr_cap),
        // Sampling-program carrier: the per-request count CSR and the nested
        // per-program byte / input-table / late-key CSRs each start with a
        // leading 0 (grown by `extend_sampling_programs_from` + the per-request
        // boundary push). All payload vecs stay empty for the legacy path.
        sampling_program_indptr: indptr(indptr_cap),
        sampling_program_bytes: Vec::new(),
        sampling_program_bytes_indptr: indptr(indptr_cap),
        sampling_input_blob: Vec::new(),
        sampling_input_keys: Vec::new(),
        sampling_input_offsets: Vec::new(),
        sampling_input_lens: Vec::new(),
        sampling_input_indptr: indptr(indptr_cap),
        sampling_late_keys: Vec::new(),
        sampling_late_indptr: indptr(indptr_cap),
        sampling_late_blob: Vec::new(),
        sampling_late_offsets: Vec::new(),
        sampling_late_lens: Vec::new(),
        // #27 cut #2 device-alias late channel: parallel to sampling_late_keys
        // (device value ptr + R12 flag per late key), grown by the batch-merge.
        sampling_late_device_ptrs: Vec::new(),
        sampling_late_device_flags: Vec::new(),
        sampling_late_device_lens: Vec::new(),
        // Per-slot binding-map: per-program CSR with a leading 0 (grown by the
        // merge + per-request boundary); the (kind, key) payload stays empty for
        // the legacy/no-program path.
        sampling_binding_kind: Vec::new(),
        sampling_binding_key: Vec::new(),
        sampling_binding_indptr: indptr(indptr_cap),
        // #27 cut #1 output fast-path dst table: payload arrays empty, the
        // per-program CSR seeded with a leading 0 (grown by the batch-merge in
        // `extend_sampling_programs_from`), exactly like `sampling_input_indptr`.
        sampling_output_dst_ptrs: Vec::new(),
        sampling_output_dst_lens: Vec::new(),
        sampling_output_indptr: indptr(indptr_cap),
        // P3 recurrent-state fold fields (rs_fold_lens / rs_buffer_slot_*)
        // default to empty — the de-hardwiring forward path does not use them.
        ..Default::default()
    }
}

/// Wrap a batched [`pie_driver_abi::ForwardRequest`] in a routable Frame.
pub fn forward_frame(driver_id: DriverId, req: pie_driver_abi::ForwardRequest) -> pie_driver_abi::Frame {
    pie_driver_abi::Frame {
        driver_id: driver_id as u32,
        payload: pie_driver_abi::RequestPayload::Forward(req),
    }
}

/// Append the request's physical page IDs to `kv_page_indices`,
/// honoring the trim plan if present.
fn emit_kv_pages(
    batch: &mut pie_driver_abi::ForwardRequest,
    physical_page_ids: &[PhysicalPageId],
    trim: Option<&TrimPlan>,
) {
    match trim {
        None => batch.kv_page_indices.extend(physical_page_ids),
        Some(plan) => {
            for (idx, &pid) in physical_page_ids.iter().enumerate() {
                if !plan.is_page_dropped(idx as u32) {
                    batch.kv_page_indices.push(pid);
                }
            }
        }
    }
}

/// Append one BRLE per row into `batch.masks`, applying the trim plan's
/// skip ranges if present. `mask_indptr` is per-request (one entry
/// pushed at the end), so each request contributes `masks.len()` Brle
/// rows.
fn emit_attention_masks(
    batch: &mut pie_driver_abi::ForwardRequest,
    masks: &[Brle],
    trim: Option<&TrimPlan>,
) {
    match trim {
        None => {
            batch.masks.extend_from_slice(masks);
        }
        Some(plan) => {
            for mask in masks {
                let mut buf = Vec::new();
                let new_total = mask.write_skipping(&plan.skip_ranges, &mut buf);
                batch.masks.push(Brle {
                    buffer: buf,
                    total_size: new_total as u64,
                });
            }
        }
    }
    batch.mask_indptr.push(batch.masks.len() as u32);
}

/// Append a per-request [`pie_driver_abi::ForwardRequest`] into the batched form.
/// `req` is the single-element shape produced by [`new_per_request`]
/// (indptrs `[0, N]`, empty kv pages). The scheduler resolved
/// `physical_page_ids` and `last_page_len` out-of-band; this call
/// folds them in along with the page-trim plan derived from
/// `req.masks`. See the file-level docs for trim criteria.
pub fn append_request(
    batch: &mut pie_driver_abi::ForwardRequest,
    req: &pie_driver_abi::ForwardRequest,
    physical_page_ids: &[PhysicalPageId],
    last_page_len: u32,
    page_size: u32,
) {
    append_request_with_options(
        batch,
        req,
        physical_page_ids,
        last_page_len,
        page_size,
        false,
    );
}

/// Append a per-request [`pie_driver_abi::ForwardRequest`] with caller-selected
/// decode mask elision. `elide_decode_masks` is only valid when the entire
/// batch is pure single-token decode; mixed prefill/decode batches need one
/// flattened mask row per query row for the bridge's custom-mask view.
pub fn append_request_with_options(
    batch: &mut pie_driver_abi::ForwardRequest,
    req: &pie_driver_abi::ForwardRequest,
    physical_page_ids: &[PhysicalPageId],
    last_page_len: u32,
    page_size: u32,
    elide_decode_masks: bool,
) {
    // Row offset of this request's tokens within the batch — image anchor rows
    // shift by this when merged (captured before extending `token_ids`).
    let row_base = batch.token_ids.len() as u32;
    // Tokens and positions
    batch.token_ids.extend(&req.token_ids);
    batch.position_ids.extend(&req.position_ids);

    let elide_decode_mask = elide_decode_masks
        && req.single_token_mode
        && !req.has_user_mask
        && req.token_ids.len() <= 1
        && req.spec_token_ids.is_empty();

    let synthesized_masks;
    let masks = if !elide_decode_mask && req.masks.is_empty() && !req.position_ids.is_empty() {
        synthesized_masks = req
            .position_ids
            .iter()
            .map(|&pos| Brle::all_true((pos + 1) as usize))
            .collect::<Vec<_>>();
        synthesized_masks.as_slice()
    } else {
        req.masks.as_slice()
    };

    let trim = if elide_decode_mask {
        None
    } else {
        TrimPlan::compute(
            masks,
            physical_page_ids.len() as u32,
            last_page_len,
            page_size,
            req.token_ids.len() as u32,
        )
    };

    // KV cache layout.
    emit_kv_pages(batch, physical_page_ids, trim.as_ref());
    // Length column (M2a / C1): per-request physical KV span, derived from the
    // EMITTED (post-trim) page count + this request's last-page length — so it
    // matches exactly what the driver reconstructs from the two arrays below.
    let page_start = *batch.kv_page_indptr.last().unwrap_or(&0);
    let page_count = batch.kv_page_indices.len() as u32 - page_start;
    let kv_len = if page_count == 0 {
        0
    } else {
        (page_count - 1) * page_size + last_page_len
    };
    batch.kv_len.push(kv_len);
    // Geometry-as-data (M5 / C1-FINAL): the device-resident kv_len source is ONE
    // forward-level handle (`batch.kv_len_device[0]` = base of a packed `[R]` u32
    // device buffer, `[r]` = request r's kv_len), NOT accumulated per-request —
    // a per-request handle would fragment the bind. It is set post-assembly by
    // the executor seam that wires a prior pass's producer output (empty ⇒
    // all lanes host-fed via the scalar `kv_len` above). append() must not touch
    // it, so a seam-set handle survives the merge unchanged.
    batch
        .kv_page_indptr
        .push(batch.kv_page_indices.len() as u32);
    batch.kv_last_page_lens.push(last_page_len);
    batch.qo_indptr.push(batch.token_ids.len() as u32);

    // Recurrent-state cache layout. For rs_cache models this carries
    // one runtime-assigned slot and one flag byte per request; for
    // regular KV-only models both vectors stay empty.
    batch.rs_slot_ids.extend(&req.rs_slot_ids);
    batch.rs_slot_flags.extend(&req.rs_slot_flags);

    // Attention masks. The runtime synthesizes causal masks for every
    // request so context lineage remains explicit, but pure single-token
    // decode does not need to send those masks to the driver: decode kernels
    // use KV lengths/page metadata directly and `single_token_mode` keeps the
    // custom-mask path disabled.
    if elide_decode_mask {
        batch.mask_indptr.push(batch.masks.len() as u32);
    } else {
        emit_attention_masks(batch, masks, trim.as_ref());
    }

    // Logit mask. Per-request: each request contributes 0 or 1 Brle
    // entries (carried in `req.logit_masks`).
    batch.logit_masks.extend_from_slice(&req.logit_masks);
    batch.logit_mask_indptr.push(batch.logit_masks.len() as u32);

    // Sampling indices.
    batch.sampling_indices.extend(&req.sampling_indices);
    batch
        .sampling_indptr
        .push(batch.sampling_indices.len() as u32);

    // Samplers (SoA): concat the per-slot arrays (offsetting the token_ids CSR),
    // then push this request's cumulative slot boundary.
    batch.extend_samplers_from(req);
    batch.sampler_indptr.push(batch.n_samplers() as u32);

    // Adapter binding (per-request has exactly one).
    batch
        .adapter_bindings
        .extend(req.adapter_bindings.iter().cloned());

    // Speculative decoding.
    batch.spec_token_ids.extend(&req.spec_token_ids);
    batch.spec_position_ids.extend(&req.spec_position_ids);
    batch.spec_indptr.push(batch.spec_token_ids.len() as u32);
    batch.output_spec_flags.extend(&req.output_spec_flags);

    // Context.
    batch.context_ids.extend(&req.context_ids);

    // Multimodal visual spans. Pure side-channel — does not affect the token,
    // KV-page, or qo layout above. Per-image data (grids, anchors) is appended
    // directly; the nested pixel/mrope CSRs are offset by the batch's running
    // lengths; `image_indptr` gets one boundary per request (cumulative image
    // count), matching the `mask_indptr` convention.
    batch.image_grids.extend(&req.image_grids);
    batch
        .image_anchor_positions
        .extend(&req.image_anchor_positions);
    batch
        .image_patch_positions
        .extend(&req.image_patch_positions);
    // Anchor rows shift by this request's row offset in the batch.
    for &r in &req.image_anchor_rows {
        batch.image_anchor_rows.push(row_base + r);
    }

    let pixel_base = batch.image_pixels.len() as u32;
    batch.image_pixels.extend(&req.image_pixels);
    for &off in req.image_pixel_indptr.iter().skip(1) {
        batch.image_pixel_indptr.push(pixel_base + off);
    }

    let mrope_base = batch.image_mrope_positions.len() as u32;
    batch.image_mrope_positions.extend(&req.image_mrope_positions);
    for &off in req.image_mrope_indptr.iter().skip(1) {
        batch.image_mrope_indptr.push(mrope_base + off);
    }

    batch.image_indptr.push((batch.image_grids.len() / 3) as u32);

    // Multimodal audio spans — direct analog of the image merge. Per-clip
    // log-mel features are appended; the feature CSR is offset by the batch's
    // running byte length; `audio_indptr` gets one boundary per request
    // (cumulative clip count). Anchor rows shift by this request's row offset.
    for &r in &req.audio_anchor_rows {
        batch.audio_anchor_rows.push(row_base + r);
    }
    let audio_feat_base = batch.audio_features.len() as u32;
    batch.audio_features.extend(&req.audio_features);
    for &off in req.audio_feature_indptr.iter().skip(1) {
        batch.audio_feature_indptr.push(audio_feat_base + off);
    }
    batch
        .audio_indptr
        .push(batch.audio_anchor_rows.len() as u32);

    // Sampling-program carrier — per-request side-channel, analogous to the
    // image/audio merges. `extend_sampling_programs_from` concatenates this
    // request's program bytecode, submit-bound input table, and late-key
    // channel, offsetting every nested CSR; `sampling_program_indptr` then gets
    // one boundary per request (cumulative program count, like `image_indptr`).
    batch.extend_sampling_programs_from(req);
    batch
        .sampling_program_indptr
        .push(batch.n_sampling_programs() as u32);

    // PTIR-program carrier (thrust-3 P2c) — per-request fold, analogous to the
    // sampling-program merge above. Concatenates this request's ptir containers
    // (hash/instance/bytes/sidecar/seeds/host-puts, cumulative CSRs) into the
    // batch so the driver's forward-serve sees `ptir_program_*` and the executor
    // hook fires. Without it the merged batch drops the carrier (ptir_hashes=0).
    batch.extend_ptir_programs_from(req);


    // Inference hint: prefill kernel when ANY request needs `custom_mask`.
    if req.token_ids.len() > 1 || req.has_user_mask {
        batch.single_token_mode = false;
    }
    if req.has_user_mask {
        batch.has_user_mask = true;
    }
}

// =============================================================================
// Per-request response extraction
// =============================================================================

/// Extract request `r`'s slice from a batched `pie_driver_abi::ForwardResponse`
/// into a single-request `ForwardResponse` (with `num_requests = 1` and
/// indptrs offset to zero). This is the "scatter" step that lets each
/// inferlet's response future see only its own request's data.
pub fn extract_per_request(
    fr: &pie_driver_abi::ForwardResponse,
    r: usize,
) -> pie_driver_abi::ForwardResponse {
    let mut out = pie_driver_abi::ForwardResponse {
        num_requests: 1,
        ..Default::default()
    };

    // Tokens: one indptr range per request.
    let (tok_lo, tok_hi) = indptr_range(&fr.tokens_indptr, r);
    let (spec_lo, spec_hi) = indptr_range(&fr.spec_indptr, r);
    if spec_hi > spec_lo {
        out.spec_indptr = vec![0, (spec_hi - spec_lo) as u32];
        out.spec_tokens = fr.spec_tokens[spec_lo..spec_hi].to_vec();
        out.spec_positions = fr.spec_positions[spec_lo..spec_hi].to_vec();
    } else if !fr.spec_indptr.is_empty() {
        out.spec_indptr = vec![0, 0];
    }

    // Hot path for normal generation: token samples only, no probe payloads.
    // Avoid allocating several empty indptr vectors for every request in
    // every decode batch.
    let token_payload_only = fr.dists_ids.is_empty()
        && fr.dists_probs.is_empty()
        && fr.logits_bytes.is_empty()
        && fr.logprobs_values.is_empty()
        && fr.entropies.is_empty()
        && fr.program_tokens.is_empty()
        && out.spec_tokens.is_empty();
    if token_payload_only {
        if tok_hi == tok_lo + 1 {
            out.tokens = vec![fr.tokens[tok_lo]];
        } else {
            out.tokens = fr.tokens[tok_lo..tok_hi].to_vec();
        }
        return out;
    }

    out.tokens = fr.tokens[tok_lo..tok_hi].to_vec();
    out.tokens_indptr = vec![0, (tok_hi - tok_lo) as u32];

    // Dists: per-request range of (ids,probs) pairs indexed by kv_indptr.
    if fr.dists_req_indptr.len() >= 2 && fr.dists_kv_indptr.len() >= 2 {
        let kv_lo = fr.dists_req_indptr[r] as usize;
        let kv_hi = fr.dists_req_indptr[r + 1] as usize;
        let val_lo = fr.dists_kv_indptr[kv_lo] as usize;
        let val_hi = fr.dists_kv_indptr[kv_hi] as usize;
        out.dists_req_indptr = vec![0, (kv_hi - kv_lo) as u32];
        out.dists_kv_indptr = (kv_lo..=kv_hi)
            .map(|k| fr.dists_kv_indptr[k] - fr.dists_kv_indptr[kv_lo])
            .collect();
        out.dists_ids = fr.dists_ids[val_lo..val_hi].to_vec();
        out.dists_probs = fr.dists_probs[val_lo..val_hi].to_vec();
    } else {
        out.dists_req_indptr = vec![0];
        out.dists_kv_indptr = vec![0];
    }

    // Logits: per-request range of byte blobs indexed by byte_indptr.
    if fr.logits_req_indptr.len() >= 2 && fr.logits_byte_indptr.len() >= 2 {
        let blob_lo = fr.logits_req_indptr[r] as usize;
        let blob_hi = fr.logits_req_indptr[r + 1] as usize;
        let byte_lo = fr.logits_byte_indptr[blob_lo] as usize;
        let byte_hi = fr.logits_byte_indptr[blob_hi] as usize;
        out.logits_req_indptr = vec![0, (blob_hi - blob_lo) as u32];
        out.logits_byte_indptr = (blob_lo..=blob_hi)
            .map(|b| fr.logits_byte_indptr[b] - fr.logits_byte_indptr[blob_lo])
            .collect();
        out.logits_bytes = fr.logits_bytes[byte_lo..byte_hi].to_vec();
    } else {
        out.logits_req_indptr = vec![0];
        out.logits_byte_indptr = vec![0];
    }

    // Logprobs: per-request range of slot vectors indexed by val_indptr.
    if fr.logprobs_req_indptr.len() >= 2 && fr.logprobs_val_indptr.len() >= 2 {
        let slot_lo = fr.logprobs_req_indptr[r] as usize;
        let slot_hi = fr.logprobs_req_indptr[r + 1] as usize;
        let val_lo = fr.logprobs_val_indptr[slot_lo] as usize;
        let val_hi = fr.logprobs_val_indptr[slot_hi] as usize;
        out.logprobs_req_indptr = vec![0, (slot_hi - slot_lo) as u32];
        out.logprobs_val_indptr = (slot_lo..=slot_hi)
            .map(|s| fr.logprobs_val_indptr[s] - fr.logprobs_val_indptr[slot_lo])
            .collect();
        out.logprobs_values = fr.logprobs_values[val_lo..val_hi].to_vec();
    } else {
        out.logprobs_req_indptr = vec![0];
        out.logprobs_val_indptr = vec![0];
    }

    // Entropies: one indptr range per request.
    let (ent_lo, ent_hi) = indptr_range(&fr.entropies_indptr, r);
    out.entropies = fr.entropies[ent_lo..ent_hi].to_vec();
    out.entropies_indptr = vec![0, (ent_hi - ent_lo) as u32];

    // Program tokens (#32 `[k]`-Token): two-level CSR like logprobs —
    // `program_tokens_req_indptr` partitions the per-(request,output) slot ranges,
    // `program_tokens_indptr` partitions the values. Slice to this request's slots,
    // resetting both indptrs to request-local offsets.
    if fr.program_tokens_req_indptr.len() >= 2 && fr.program_tokens_indptr.len() >= 2 {
        let slot_lo = fr.program_tokens_req_indptr[r] as usize;
        let slot_hi = fr.program_tokens_req_indptr[r + 1] as usize;
        let val_lo = fr.program_tokens_indptr[slot_lo] as usize;
        let val_hi = fr.program_tokens_indptr[slot_hi] as usize;
        out.program_tokens_req_indptr = vec![0, (slot_hi - slot_lo) as u32];
        out.program_tokens_indptr = (slot_lo..=slot_hi)
            .map(|s| fr.program_tokens_indptr[s] - fr.program_tokens_indptr[slot_lo])
            .collect();
        out.program_tokens = fr.program_tokens[val_lo..val_hi].to_vec();
    } else {
        out.program_tokens_req_indptr = vec![0];
        out.program_tokens_indptr = vec![0];
    }

    out
}

#[inline]
fn indptr_range(indptr: &[u32], r: usize) -> (usize, usize) {
    if indptr.len() >= r + 2 {
        (indptr[r] as usize, indptr[r + 1] as usize)
    } else {
        (0, 0)
    }
}

