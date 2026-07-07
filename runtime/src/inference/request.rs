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

use crate::adapter::AdapterId;
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
        // WS8 P2 device-resident next-input link: empty/0 for the host-inject
        // (P1) and no-pipelining paths; populated by the device-pipeline emission
        // at integration (per-row arrays + the producer source-link).
        pipeline_source_link: 0,
        next_input_producer_links: Vec::new(),
        next_input_src_rows: Vec::new(),
        next_input_dest_slots: Vec::new(),
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

    // Run-ahead next-input carrier (WS8 P2) — per-request device-resident link
    // fold. Each fed consumer input names a producer link id + the source row in
    // that producer's retained `pi.sampled`, and a dest slot in THIS request's
    // `pi.tokens`. On batch merge: producer link ids + source rows are GLOBAL (a
    // prior fire's per-link buffer) → verbatim; dest slots index this fire's own
    // token buffer → rebased by `row_base` (the batch token-offset). The
    // `u32::MAX` dest sentinel (skip-lane) is preserved un-rebased.
    for i in 0..req.n_next_input_links() {
        let dest = req.next_input_dest_slots[i];
        let rebased_dest = if dest == u32::MAX { u32::MAX } else { dest + row_base };
        batch.push_next_input_link(
            req.next_input_producer_links[i],
            req.next_input_src_rows[i],
            rebased_dest,
        );
    }
    // All-consumers-drained free signals — global link ids, verbatim.
    for &link in &req.next_input_free_links {
        batch.push_next_input_free_link(link);
    }
    // Producer source-link — PER-REQUEST (thrust-2 Bug#2 fix). Each co-batched
    // producer must retain its OWN sampled token under its OWN global link. Append
    // one `pipeline_source_links` entry per request (0 = not a producer) so an R>1
    // co-batched PRODUCER fire keeps EVERY request's link, not just the last.
    //
    // (Supersedes the earlier batch-level `set_pipeline_source_link` overwrite,
    // which resolved Open Q2 as batch-level on the false premise that depth-1
    // admits at most one producer per batch. It does not: multiple DISTINCT
    // requests co-batch as producers under continuous batching, and the overwrite
    // kept only the final request's link → the others' producers never retained →
    // their consumers retain-missed → placeholder token 0, the concurrent-decode
    // corruption charlie root-caused. The executor retains request `r`'s producer
    // row `qo_indptr[r+1]-1` under `pipeline_source_links[r]`.) The scalar
    // `pipeline_source_link` is still set (last non-zero) for R=1 back-compat.
    batch.pipeline_source_links.push(req.pipeline_source_link);
    if req.pipeline_source_link != 0 {
        batch.set_pipeline_source_link(req.pipeline_source_link);
    }
    // Drafts-channel routing (§9): fold the per-request source-kind parallel to the
    // link. `0 = PrevSample` (retain `pi.sampled`) / `1 = PrevDrafts` (retain the
    // composed `[k+1]` window). Empty ⇒ all PrevSample (the pushed 0s are harmless).
    batch.pipeline_source_kinds.push(req.pipeline_source_kind);
    if req.pipeline_source_kind != 0 {
        batch.set_pipeline_source_kind(req.pipeline_source_kind);
    }

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
        // A PTIR stage-program response carries its committed Reader-channel cells
        // in `ptir_output_*` with EMPTY tokens/probes — it must NOT take the
        // token-only early return, which would drop the outputs before the
        // `ptir_output_*` slice below (§6.2 `out.take: no cell available` root).
        && fr.ptir_output_indptr.is_empty()
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

    // PTIR outputs (§6.2 stage programs): a per-PROGRAM CSR — `ptir_output_indptr`
    // partitions the committed READER-channel slots by program (program p ↔
    // request p in the single-program-per-request fire). Slice this request's
    // program slots + the matching blob range (blob offset = cumulative Σ lens;
    // there is no byte_indptr, the blob is concatenated in slot order). Without
    // this the driver-filled `ptir_output_*` is dropped by `..Default` and the
    // guest's `out.take()` reads nothing.
    if fr.ptir_output_indptr.len() >= 2 {
        let (slot_lo, slot_hi) = indptr_range(&fr.ptir_output_indptr, r);
        if slot_hi > slot_lo {
            let byte_lo: usize =
                fr.ptir_output_lens[..slot_lo].iter().map(|&n| n as usize).sum();
            let byte_hi: usize = byte_lo
                + fr.ptir_output_lens[slot_lo..slot_hi]
                    .iter()
                    .map(|&n| n as usize)
                    .sum::<usize>();
            out.ptir_output_channels = fr.ptir_output_channels[slot_lo..slot_hi].to_vec();
            out.ptir_output_lens = fr.ptir_output_lens[slot_lo..slot_hi].to_vec();
            out.ptir_output_blob = fr.ptir_output_blob[byte_lo..byte_hi].to_vec();
            out.ptir_output_indptr = vec![0, (slot_hi - slot_lo) as u32];
        } else {
            out.ptir_output_indptr = vec![0, 0];
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_per_request_slices_program_tokens_two_level_csr() {
        // #32 two-level CSR (bravo's round-trip vectors): 2 requests, each n_out=2
        // (output 0 single-Token/empty, output 1 a `[k]`-Token of 2).
        // req_indptr=[0,2,4], indptr=[0,0,2,2,4], tokens=[279,280,555,556].
        // extract_per_request slices request r to its slots, resetting both
        // indptrs request-local so build_output_tensors walks seg=o within it.
        let fr = pie_driver_abi::ForwardResponse {
            num_requests: 2,
            program_tokens_req_indptr: vec![0, 2, 4],
            program_tokens_indptr: vec![0, 0, 2, 2, 4],
            program_tokens: vec![279, 280, 555, 556],
            ..Default::default()
        };
        // request 0: slots [0,2) → output 0 empty, output 1 = [279,280].
        let r0 = extract_per_request(&fr, 0);
        assert_eq!(r0.program_tokens, vec![279, 280]);
        assert_eq!(r0.program_tokens_indptr, vec![0, 0, 2]);
        // request 1: slots [2,4) → output 0 empty, output 1 = [555,556].
        let r1 = extract_per_request(&fr, 1);
        assert_eq!(r1.program_tokens, vec![555, 556]);
        assert_eq!(r1.program_tokens_indptr, vec![0, 0, 2]);
    }

    #[test]
    fn extract_per_request_preserves_ptir_output_over_token_fast_path() {
        // §6.2 regression: a PTIR stage-program response carries its committed
        // Reader-channel cells in `ptir_output_*` with EMPTY tokens/probes. The
        // `token_payload_only` fast-path (all probe channels empty) must NOT take
        // its early return for such a response, or the `ptir_output_*` slice is
        // dropped and the guest's `out.take()` reads "no cell available".
        let fr = pie_driver_abi::ForwardResponse {
            num_requests: 1,
            // greedy_argmax: 1 program, 1 committed Reader channel (chan 1) = i32 token.
            ptir_output_channels: vec![1],
            ptir_output_blob: 19148i32.to_le_bytes().to_vec(),
            ptir_output_lens: vec![4],
            ptir_output_indptr: vec![0, 1],
            ..Default::default()
        };
        let r0 = extract_per_request(&fr, 0);
        assert_eq!(r0.ptir_output_indptr, vec![0, 1]);
        assert_eq!(r0.ptir_output_channels, vec![1]);
        assert_eq!(r0.ptir_output_lens, vec![4]);
        assert_eq!(r0.ptir_output_blob, 19148i32.to_le_bytes().to_vec());
        // and `ptir_output_at(0)` reconstructs the (channel, bytes) tuple.
        let outs = r0.ptir_output_at(0).unwrap();
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].channel, 1);
        assert_eq!(outs[0].bytes, 19148i32.to_le_bytes().to_vec());
    }

    // -- Page-trim integration tests -----------------------------------------

    fn make_request(
        tokens: Vec<u32>,
        positions: Vec<u32>,
        masks: Vec<Brle>,
    ) -> pie_driver_abi::ForwardRequest {
        let has_user_mask = !masks.is_empty();
        new_per_request(
            0,
            tokens,
            positions,
            masks,
            has_user_mask,
            None,
            vec![],
            vec![],
            vec![],
            vec![],
            false,
            None,
            None,
        )
    }

    /// Attach one visual span to `req`, mirroring the `input_image` host
    /// handler: append grid + anchor + pixels + mrope, then push exactly one
    /// entry onto each per-image indptr (unconditionally, even when there are
    /// no mrope positions) so every image contributes one CSR slot.
    fn with_image(
        mut req: pie_driver_abi::ForwardRequest,
        grid: [u32; 3],
        anchor: u32,
        pixels: &[u8],
        mrope: &[u32],
    ) -> pie_driver_abi::ForwardRequest {
        req.image_grids.extend_from_slice(&grid);
        req.image_anchor_positions.push(anchor);
        req.image_pixels.extend_from_slice(pixels);
        req.image_pixel_indptr.push(req.image_pixels.len() as u32);
        req.image_mrope_positions.extend_from_slice(mrope);
        req.image_mrope_indptr
            .push(req.image_mrope_positions.len() as u32);
        req
    }

    #[test]
    fn image_side_channel_merges_with_csr_offsets() {
        // Two single-token decode requests, each carrying one image. Request 0
        // is an M-RoPE image (6 position components); request 1 is a 1-D-RoPE
        // image (no mrope). Merging must offset the nested pixel/mrope CSRs by
        // the batch's running lengths and push one image_indptr boundary per
        // request.
        let req0 = with_image(
            make_request(vec![1], vec![10], vec![]),
            [1, 4, 4],
            10,
            &[1, 2, 3],
            &[10, 10, 10, 10, 10, 11],
        );
        let req1 = with_image(
            make_request(vec![2], vec![20], vec![]),
            [1, 2, 2],
            100,
            &[9],
            &[],
        );

        let mut batch = new_batched_forward_request();
        append_request(&mut batch, &req0, &[100], 1, 16);
        append_request(&mut batch, &req1, &[200], 1, 16);

        // Per-image data concatenated in request order.
        assert_eq!(batch.image_grids, vec![1, 4, 4, 1, 2, 2]);
        assert_eq!(batch.image_anchor_positions, vec![10, 100]);
        assert_eq!(batch.image_pixels, vec![1, 2, 3, 9]);
        // Nested pixel CSR: leading 0, then offset cumulative byte ends.
        assert_eq!(batch.image_pixel_indptr, vec![0, 3, 4]);
        // M-RoPE positions only from the first image; the second contributes an
        // empty (but present) CSR slot.
        assert_eq!(batch.image_mrope_positions, vec![10, 10, 10, 10, 10, 11]);
        assert_eq!(batch.image_mrope_indptr, vec![0, 6, 6]);
        // Per-request CSR: one image each.
        assert_eq!(batch.image_indptr, vec![0, 1, 2]);
    }

    #[test]
    fn next_input_carrier_merges_with_dest_rebase() {
        // Run-ahead next-input carrier fold: producer link ids + source rows are
        // global (verbatim); dest slots index each fire's own token buffer (rebased
        // by the batch token-offset); `u32::MAX` skip-lane preserved; free-links
        // verbatim; producer source-link propagated.
        let mut req0 = make_request(vec![10, 11], vec![0, 1], vec![]); // 2 tokens
        req0.set_pipeline_source_link(7);
        req0.push_next_input_link(5, 0, 1); // dest 1 -> row_base(0)+1 = 1
        req0.push_next_input_free_link(5);

        let mut req1 = make_request(vec![12], vec![2], vec![]); // 1 token
        req1.set_pipeline_source_link(9);
        req1.push_next_input_link(7, 3, 0); // dest 0 -> row_base(2)+0 = 2
        req1.push_next_input_link(7, 4, u32::MAX); // skip-lane preserved
        req1.push_next_input_free_link(7);

        let mut batch = new_batched_forward_request();
        append_request(&mut batch, &req0, &[100], 1, 16); // row_base 0
        append_request(&mut batch, &req1, &[200], 1, 16); // row_base 2 (req0 had 2 tokens)

        assert_eq!(batch.next_input_producer_links, vec![5, 7, 7]); // global, verbatim
        assert_eq!(batch.next_input_src_rows, vec![0, 3, 4]); // global, verbatim
        assert_eq!(batch.next_input_dest_slots, vec![1, 2, u32::MAX]); // rebased + sentinel
        assert_eq!(batch.next_input_free_links, vec![5, 7]); // verbatim concat
        // Bug#2 fix: EACH co-batched producer retains its own link (per-request),
        // not just the last. The scalar stays set (last) for R=1 back-compat.
        assert_eq!(batch.pipeline_source_links, vec![7, 9]); // per-request, both preserved
        assert_eq!(batch.pipeline_source_link, 9); // scalar back-compat = last
    }

    #[test]
    fn co_batched_producers_retain_per_request_links_not_just_last() {
        // **Bug#2 repro/guard (thrust-2, bravo).** The R>1 concurrent-decode
        // corruption: two DISTINCT requests co-batch as run-ahead PRODUCERS in one
        // fire, each sampling a token its own consumer will inject next step. The
        // producer link is assigned PER-REQUEST (`apply_next_input_carrier` bumps a
        // fresh link per pass), so request A gets link 7, request B link 9. The
        // merge MUST retain BOTH — a batch-level scalar keeps only the last (9), so
        // request A's producer token is never retained → A's consumer retain-misses
        // → the device injects placeholder token 0 (or a cross-request token) → wrong
        // Q&K → garbage. This asserts the per-request `pipeline_source_links` fold so
        // each co-batched producer's token is retained under its own link.
        let mut prod_a = make_request(vec![100], vec![5], vec![]); // decode row, producer A
        prod_a.set_pipeline_source_link(7);
        let mut prod_b = make_request(vec![200], vec![9], vec![]); // decode row, producer B
        prod_b.set_pipeline_source_link(9);
        // A non-producer request in the same batch contributes a 0 (no retain).
        let plain = make_request(vec![300], vec![3], vec![]);

        let mut batch = new_batched_forward_request();
        append_request(&mut batch, &prod_a, &[100], 1, 16);
        append_request(&mut batch, &plain, &[200], 1, 16);
        append_request(&mut batch, &prod_b, &[300], 1, 16);

        // One entry per request, aligned with `qo_indptr`: both producers' links
        // survive (0 for the non-producer). A batch-level scalar would have lost 7.
        assert_eq!(batch.pipeline_source_links, vec![7, 0, 9]);
        // And each is attributable to its request's producer row via qo_indptr.
        assert_eq!(batch.qo_indptr, vec![0, 1, 2, 3]);
    }

    /// Attach one audio clip to `req`, mirroring the `input_audio` host handler:
    /// append the row anchor + feature bytes, then push one feature-CSR slot.
    fn with_audio(
        mut req: pie_driver_abi::ForwardRequest,
        anchor_row: u32,
        features: &[u8],
    ) -> pie_driver_abi::ForwardRequest {
        req.audio_anchor_rows.push(anchor_row);
        req.audio_features.extend_from_slice(features);
        req.audio_feature_indptr
            .push(req.audio_features.len() as u32);
        req
    }

    #[test]
    fn audio_side_channel_merges_with_csr_offsets() {
        // Two single-token requests, each carrying one audio clip. Merging must
        // offset the per-clip feature CSR by the batch's running byte length,
        // shift anchor rows by the request's batch-row offset, and push one
        // audio_indptr boundary per request.
        let req0 = with_audio(make_request(vec![1], vec![10], vec![]), 0, &[1, 2, 3]);
        let req1 = with_audio(make_request(vec![2], vec![20], vec![]), 0, &[4, 5]);

        let mut batch = new_batched_forward_request();
        append_request(&mut batch, &req0, &[100], 1, 16);
        append_request(&mut batch, &req1, &[200], 1, 16);

        // Per-clip features concatenated in request order.
        assert_eq!(batch.audio_features, vec![1, 2, 3, 4, 5]);
        // Nested feature CSR: leading 0, then offset cumulative byte ends.
        assert_eq!(batch.audio_feature_indptr, vec![0, 3, 5]);
        // Anchor rows shift by each request's row offset (req0 at row 0, req1
        // at row 1 since req0 contributed one token).
        assert_eq!(batch.audio_anchor_rows, vec![0, 1]);
        // Per-request CSR: one clip each.
        assert_eq!(batch.audio_indptr, vec![0, 1, 2]);
    }

    #[test]
    fn text_only_requests_carry_no_audio() {
        let req = make_request(vec![1], vec![10], vec![]);
        let mut batch = new_batched_forward_request();
        append_request(&mut batch, &req, &[100], 1, 16);
        assert!(batch.audio_features.is_empty());
        assert!(batch.audio_anchor_rows.is_empty());
        // audio_indptr still advances (one boundary per request, 0 clips).
        assert_eq!(batch.audio_indptr, vec![0, 0]);
        assert_eq!(batch.audio_feature_indptr, vec![0]);
    }

    #[test]
    fn text_only_requests_carry_no_images() {
        let req = make_request(vec![1], vec![10], vec![]);
        let mut batch = new_batched_forward_request();
        append_request(&mut batch, &req, &[100], 1, 16);
        assert!(batch.image_grids.is_empty());
        assert!(batch.image_pixels.is_empty());
        // image_indptr still advances (one boundary per request, 0 images).
        assert_eq!(batch.image_indptr, vec![0, 0]);
        assert_eq!(batch.image_pixel_indptr, vec![0]);
        assert_eq!(batch.image_mrope_indptr, vec![0]);
    }

    #[test]
    fn add_request_causal_decode_no_trim() {
        // Single-token decode at position 47, page_size=16, num_pages=3,
        // last_page_len=16. Causal mask (all-true [0,48]) → no false runs,
        // no pages can be dropped. Wire format must match the pre-optimization
        // layout exactly: all 3 pages present, mask buffer untouched.
        let causal = Brle::all_true(48);
        let req = make_request(vec![999], vec![47], vec![causal.clone()]);
        let pages: Vec<PhysicalPageId> = vec![100, 101, 102];

        let mut batch = new_batched_forward_request();
        append_request(&mut batch, &req, &pages, 16, 16);

        assert_eq!(batch.kv_page_indices, vec![100, 101, 102]);
        assert_eq!(batch.kv_page_indptr, vec![0, 3]);
        assert_eq!(batch.kv_last_page_lens, vec![16]);
        // M2a length column: physical span = (3-1)*16 + 16 = 48 (= seq_len).
        assert_eq!(batch.kv_len, vec![48]);
        // Mask buffer is the original BRLE (no rewrite).
        assert_eq!(batch.masks, vec![causal.clone()]);
        assert_eq!(batch.mask_indptr, vec![0, 1]);
    }

    /// A single-token decode request with a real per-request sampling row
    /// (`sampling_indices = [0]`, request-relative — the driver rebases it to
    /// `qo_lo + 0`). Mirrors what the API prepares for a decode step.
    fn decode_request(token: u32, position: u32) -> pie_driver_abi::ForwardRequest {
        new_per_request(
            0,
            vec![token],
            vec![position],
            vec![],
            false,
            None,
            vec![0], // request-relative sampling row
            vec![],
            vec![],
            vec![],
            false,
            None,
            None,
        )
    }

    /// **M5 (geometry-as-data / C1-FINAL): `kv_len_device` is a forward-level
    /// handle that survives the merge.** The device-resident `kv_len` source is
    /// ONE base pointer for the whole forward (base of a packed `[R]` u32 buffer,
    /// `[r]` = request r's kv_len), set post-assembly by the executor seam — a
    /// per-request handle would fragment the bind (C1 co-design). This proves the
    /// merge does NOT clobber a seam-set handle and the host-fed scalar column is
    /// unaffected; today the seam is unwired, so the handle stays `0` and the
    /// legacy path is byte-identical.
    #[test]
    fn m5_kv_len_device_survives_merge() {
        let req0 = decode_request(11, 47); // 3 pages, host kv_len 48
        let req1 = decode_request(22, 31); // 2 pages, host kv_len 32
        let req2 = decode_request(33, 15); // 1 page, host kv_len 16

        let reqs = [
            (req0, vec![100u32, 101, 102]),
            (req1, vec![200u32, 201]),
            (req2, vec![300u32]),
        ];
        let elide = reqs.iter().all(|(r, _)| {
            r.single_token_mode
                && !r.has_user_mask
                && r.token_ids.len() <= 1
                && r.spec_token_ids.is_empty()
        });

        let mut batch = new_batched_forward_request();
        // The executor seam wires a prior pass's producer output as the forward's
        // device kv_len source BEFORE the per-request merge folds the lanes in.
        batch.kv_len_device = vec![0xDEAD_BEEF_0000_1000];
        for (req, pages) in &reqs {
            append_request_with_options(&mut batch, req, pages, 16, 16, elide);
        }

        // Host-fed scalar column is unchanged (back-compat), one entry per request.
        assert_eq!(batch.kv_len, vec![48, 32, 16]);
        // The forward-level device handle is a single 1-element Vec the merge leaves
        // untouched (never fragmented/accumulated per request).
        assert_eq!(batch.kv_len_device, vec![0xDEAD_BEEF_0000_1000]);
    }

    /// M5 back-compat: an unwired seam leaves `kv_len_device` EMPTY — the legacy
    /// path is byte-identical modulo the appended (empty) handle, so the driver
    /// keeps deriving each lane's length from the scalar `kv_len` column.
    #[test]
    fn m5_kv_len_device_absent_is_all_host_fed() {
        let req = decode_request(11, 47);
        let mut batch = new_batched_forward_request();
        append_request_with_options(&mut batch, &req, &[100, 101, 102], 16, 16, true);
        assert_eq!(batch.kv_len, vec![48]);
        assert!(batch.kv_len_device.is_empty(), "unwired seam ⇒ host-fed (empty handle)");
    }

    /// **R>1 batch-merge page-attribution guard** (bravo, concurrent-decode hunt).
    /// The concurrent-garbage signature is *all-zero KV reads under R>1* — a
    /// request reads pages that were never written, i.e. its page-table entry is
    /// wrong in the *merged* batch (a semantic rebasing bug that would still pass
    /// the byte-round-trip S1 test). This dumps the merged per-request page tables
    /// for a 3-request decode batch with **distinct multi-page contexts** and
    /// asserts each request's KV span is correctly delimited AND pairwise
    /// DISJOINT: no request can alias/read another's pages purely from the host
    /// merge. Folds via the real `elide_decode_masks=true` decode path
    /// `build_batch_request` uses.
    #[test]
    fn r_gt_1_decode_batch_attributes_pages_disjointly() {
        // Three decodes with distinct physical pages + distinct context lengths:
        //   req0: pos 47 → seq_len 48 → 3 pages [100,101,102]
        //   req1: pos 31 → seq_len 32 → 2 pages [200,201]
        //   req2: pos 15 → seq_len 16 → 1 page  [300]
        let reqs = [
            (decode_request(11, 47), vec![100u32, 101, 102]),
            (decode_request(22, 31), vec![200u32, 201]),
            (decode_request(33, 15), vec![300u32]),
        ];

        // Reproduce `build_batch_request`: pure single-token decode ⇒ elide masks.
        let elide = reqs.iter().all(|(r, _)| {
            r.single_token_mode
                && !r.has_user_mask
                && r.token_ids.len() <= 1
                && r.spec_token_ids.is_empty()
        });
        assert!(elide, "a pure single-token decode batch must elide decode masks");

        let mut batch = new_batched_forward_request();
        for (req, pages) in &reqs {
            append_request_with_options(&mut batch, req, pages, 16, 16, elide);
        }

        // Page columns: concatenated in request order, CSR delimiting each run.
        assert_eq!(batch.kv_page_indices, vec![100, 101, 102, 200, 201, 300]);
        assert_eq!(batch.kv_page_indptr, vec![0, 3, 5, 6]);
        assert_eq!(batch.kv_last_page_lens, vec![16, 16, 16]);
        // M2a length column, per request: (num_pages-1)*16 + last_page_len.
        assert_eq!(batch.kv_len, vec![48, 32, 16]);
        // Token rows: one per request, in order.
        assert_eq!(batch.token_ids, vec![11, 22, 33]);
        assert_eq!(batch.position_ids, vec![47, 31, 15]);
        assert_eq!(batch.qo_indptr, vec![0, 1, 2, 3]);
        // Sampling rows stay request-relative ([0] each); the driver rebases them
        // per request via `qo_indptr[r] + sampling_indices[k]` — so NOT rebasing
        // in the merge is correct. The CSR delimits one row per request.
        assert_eq!(batch.sampling_indices, vec![0, 0, 0]);
        assert_eq!(batch.sampling_indptr, vec![0, 1, 2, 3]);

        // The anti-contamination invariant: reconstruct each request's page set
        // from the CSR exactly as the driver does, and assert the sets are
        // pairwise DISJOINT and their union is the whole array — so no request
        // can read a page belonging to another (the all-zero/cross-read root, if
        // the merge were the culprit, is structurally impossible here).
        let n = batch.kv_page_indptr.len() - 1;
        let spans: Vec<&[u32]> = (0..n)
            .map(|r| {
                let lo = batch.kv_page_indptr[r] as usize;
                let hi = batch.kv_page_indptr[r + 1] as usize;
                &batch.kv_page_indices[lo..hi]
            })
            .collect();
        for i in 0..n {
            // Reconstructed physical span matches the length column.
            let pc = spans[i].len() as u32;
            assert_eq!((pc - 1) * 16 + batch.kv_last_page_lens[i], batch.kv_len[i]);
            for j in (i + 1)..n {
                for &p in spans[i] {
                    assert!(
                        !spans[j].contains(&p),
                        "request {i} and {j} alias physical page {p} in the merged batch"
                    );
                }
            }
        }
    }

    #[test]
    fn add_request_attention_sink_trims_middle_pages() {
        // Decode at position 319 with sink+window mask: sink=4, gap=252,
        // window=64, total seq_len=320. page_size=16, num_pages=20,
        // last_page_len=16. The single new token writes to page 19 (the last
        // page), so first_writeable_page = 319/16 = 19 → eligible window is
        // pages 0..=18.
        //
        // Per-row droppable: false run [4, 256) covers pages 1..=15 fully.
        // After AND with eligibility window {0..=18}: pages 1..=15 dropped.
        let mask = Brle::from_vec(vec![0, 4, 252, 64]); // sink+window
        assert_eq!(mask.len(), 320);

        let req = make_request(vec![999], vec![319], vec![mask]);
        let pages: Vec<PhysicalPageId> = (0..20).map(|i| 1000 + i as PhysicalPageId).collect();

        let mut batch = new_batched_forward_request();
        append_request(&mut batch, &req, &pages, 16, 16);

        // Surviving pages: 0, 16, 17, 18, 19 (5 pages).
        let expected_pages: Vec<u32> = vec![1000, 1016, 1017, 1018, 1019];
        assert_eq!(batch.kv_page_indices, expected_pages);
        assert_eq!(batch.kv_page_indptr, vec![0, 5]);
        // last_page_len unchanged — last page is never dropped.
        assert_eq!(batch.kv_last_page_lens, vec![16]);
        // M2a length column: the physical span tracks the EMITTED (trimmed)
        // pages, not the logical seq_len — (5-1)*16 + 16 = 80, matching what the
        // driver reconstructs from the two arrays above (frozen/dropped pages are
        // out of the physical span; the mask carries any sub-page validity, W6).
        assert_eq!(batch.kv_len, vec![80]);

        // Trimmed BRLE: original false run [4, 256) shrinks by 15*16 = 240
        // bits (15 dropped pages). Layout becomes:
        //   sink(4) | gap'(12) | window(64)
        // i.e., BRLE buffer = [0, 4, 12, 64], total_size = 80 = 5*16.
        let trimmed = Brle {
            buffer: vec![0, 4, 12, 64],
            total_size: 80,
        };
        assert_eq!(batch.masks, vec![trimmed]);
        // Per-request indptr: one request, one mask row.
        assert_eq!(batch.mask_indptr, vec![0, 1]);
    }

    #[test]
    fn add_request_window_only_trims_leading_pages() {
        // Sliding-window mask: gap=240 (false), window=80 (true), seq_len=320.
        // page_size=16, num_pages=20, last_page_len=16. Decode at position 319.
        //   eligible window: pages 0..=18 (writeable = page 19).
        //   row droppable: pages 0..=14 (false run [0, 240) covers them fully).
        // Drop pages 0..=14 (15 pages); pages 15..=19 remain.
        let mask = Brle::from_vec(vec![240, 80]);
        assert_eq!(mask.len(), 320);

        let req = make_request(vec![999], vec![319], vec![mask]);
        let pages: Vec<PhysicalPageId> = (0..20).map(|i| 2000 + i as PhysicalPageId).collect();

        let mut batch = new_batched_forward_request();
        append_request(&mut batch, &req, &pages, 16, 16);

        let expected_pages: Vec<u32> = (15..20).map(|i| 2000 + i).collect();
        assert_eq!(batch.kv_page_indices, expected_pages);
        // After dropping 15 leading false pages, remaining mask is:
        //   false: 240 - 15*16 = 0  →  zero-length false prefix preserved
        //   true: 80
        // Buffer: [0, 80], total_size = 80.
        let trimmed = Brle {
            buffer: vec![0, 80],
            total_size: 80,
        };
        assert_eq!(batch.masks, vec![trimmed]);
    }

    #[test]
    fn add_request_writeable_pages_are_protected() {
        // Pathological: a request whose mask is all-False, but kv_before is
        // entirely contained in a single non-final page. The writeable-window
        // guard must protect that page even though the mask agrees it's
        // droppable.
        //
        // page_size=16, kv_before=10 (one partial page), tokens.len()=6
        // (filling the page). num_pages=1, last_page_len=16, total_kv=16.
        // first_writeable_page = 10/16 = 0 → no eligible pages.
        let mask = Brle::from_vec(vec![16]); // all false, total 16
        let req = make_request(
            vec![1, 2, 3, 4, 5, 6],
            vec![10, 11, 12, 13, 14, 15],
            vec![mask.clone()],
        );
        let pages: Vec<PhysicalPageId> = vec![777];

        let mut batch = new_batched_forward_request();
        append_request(&mut batch, &req, &pages, 16, 16);

        // No pages dropped; original mask preserved.
        assert_eq!(batch.kv_page_indices, vec![777]);
        assert_eq!(batch.masks, vec![mask.clone()]);
    }

    #[test]
    fn add_request_rows_disagree_no_drops() {
        // Two-token prefill, page_size=16, num_pages=4 (seq_len=64),
        // last_page_len=16, kv_before=62 → first_writeable_page=3, eligible
        // window {0,1,2}. Two rows whose droppable sets are disjoint within
        // that window:
        //   row 0: false [0, 32)  → pages 0,1 droppable
        //   row 1: false [32, 48) → page 2 droppable
        // AND-reduction collapses to ∅, so the trim path bails. Verify the
        // wire format is byte-identical to the no-trim layout.
        let row0 = Brle::from_vec(vec![32, 32]); // false 32, true 32
        let row1 = Brle::from_vec(vec![0, 32, 16, 16]); // true 32, false 16, true 16
        assert_eq!(row0.len(), 64);
        assert_eq!(row1.len(), 64);

        let req = make_request(vec![1, 2], vec![62, 63], vec![row0.clone(), row1.clone()]);
        let pages: Vec<PhysicalPageId> = vec![10, 11, 12, 13];

        let mut batch = new_batched_forward_request();
        append_request(&mut batch, &req, &pages, 16, 16);

        // Fast path: original pages and masks unmodified.
        assert_eq!(batch.kv_page_indices, vec![10, 11, 12, 13]);
        assert_eq!(batch.masks, vec![row0.clone(), row1.clone()]);
        // Per-request indptr: one request contributing 2 mask rows.
        assert_eq!(batch.mask_indptr, vec![0, 2]);
    }

    #[test]
    fn add_request_multi_row_identical_sink_pattern() {
        // Prefill with multiple input tokens, every row has the same
        // sink+window mask (a common inferlet pattern). Verify that the trim
        // applies uniformly across rows and the per-row mask offsets in
        // `mask_indptr` track the trimmed buffer correctly.
        let mask = Brle::from_vec(vec![0, 4, 252, 64]); // seq_len 320
        let req = make_request(
            vec![10, 20, 30],
            vec![317, 318, 319],
            vec![mask.clone(), mask.clone(), mask.clone()],
        );
        let pages: Vec<PhysicalPageId> = (0..20).map(|i| 5000 + i as PhysicalPageId).collect();

        let mut batch = new_batched_forward_request();
        // Three new tokens at positions 317..319, kv_before=317 →
        // first_writeable_page=19. Pages 1..=15 still dropped by the mask.
        append_request(&mut batch, &req, &pages, 16, 16);

        let expected_pages: Vec<u32> = vec![5000, 5016, 5017, 5018, 5019];
        assert_eq!(batch.kv_page_indices, expected_pages);

        // Three identical rows trimmed identically: each row's BRLE shrinks
        // to [0, 4, 12, 64], total_size 80. The Vec<Brle> has 3 entries.
        let trimmed_row = Brle {
            buffer: vec![0, 4, 12, 64],
            total_size: 80,
        };
        assert_eq!(
            batch.masks,
            vec![trimmed_row.clone(), trimmed_row.clone(), trimmed_row]
        );
        // Per-request indptr: one request contributing 3 rows.
        assert_eq!(batch.mask_indptr, vec![0, 3]);
    }

    /// Faithful production-path durable guard for the "new carrier field silently
    /// dropped by the batch-merge" class (2nd occurrence: #6 next-input, then #27
    /// `sampling_output_*`). Exercises the REAL merge path —
    /// `new_batched_forward_request` (init) + `append_request` (per-request fold)
    /// — and asserts the #27 cut #1 `sampling_output_*` fast-path dst table
    /// survives (the field the merge originally dropped → driver `view=0` →
    /// zeros) ALONGSIDE `next_input_*` + `pipeline_source_link`. Catches a regress
    /// in EITHER the init (missing leading-0 CSR) or the merge (missing concat).
    #[test]
    fn batch_merge_preserves_sampling_output_and_carriers() {
        // req0: 1-token decode + 1 fast-path output + a producer source-link + a
        // consumer next-input link.
        let mut req0 = make_request(vec![1], vec![10], vec![]);
        req0.sampling_output_dst_ptrs = vec![0x1000];
        req0.sampling_output_dst_lens = vec![4];
        req0.sampling_output_indptr = vec![0, 1];
        req0.set_pipeline_source_link(7);
        req0.push_next_input_link(3, 0, 0);

        // req1: 1-token decode + 2 fast-path outputs.
        let mut req1 = make_request(vec![2], vec![20], vec![]);
        req1.sampling_output_dst_ptrs = vec![0x2000, 0x3000];
        req1.sampling_output_dst_lens = vec![4, 8];
        req1.sampling_output_indptr = vec![0, 2];

        let mut batch = new_batched_forward_request();
        append_request(&mut batch, &req0, &[100], 1, 16);
        append_request(&mut batch, &req1, &[200], 1, 16);

        // #27 cut #1: the fast-path dst table survives — concatenated payloads +
        // the per-program CSR offset by the table base (p0's 1 output, p1's 2 at
        // base 1). Before the fix this was empty → driver view=0 → all-zeros.
        assert_eq!(
            batch.sampling_output_dst_ptrs,
            vec![0x1000u64, 0x2000, 0x3000]
        );
        assert_eq!(batch.sampling_output_dst_lens, vec![4u32, 4, 8]);
        assert_eq!(batch.sampling_output_indptr, vec![0u32, 1, 3]);

        // Other carriers survive the same merge (regression baseline).
        assert_eq!(batch.pipeline_source_link, 7);
        assert_eq!(batch.next_input_producer_links, vec![3]);
        assert_eq!(batch.next_input_src_rows, vec![0]);
        assert_eq!(batch.next_input_dest_slots, vec![0]);
    }

    /// S1 exit criterion: a `ForwardRequest` carrying the sampling-program
    /// carrier (`sampling_program_*` / `sampling_input_*` / `sampling_late_*` /
    /// `sampling_binding_*`) survives batching byte-for-byte modulo the
    /// documented CSR offsets. Two requests each carry one program with
    /// distinct bytecode, submit-bound inputs, late keys (one with a host
    /// value, one device-alias with none), and a per-slot binding map. After
    /// the real `append_request` fold, `sampling_program_at(p)` must reconstruct
    /// each original submission exactly, and the offset CSRs must be correct.
    #[test]
    fn sampling_program_carrier_survives_batching_byte_for_byte() {
        use pie_driver_abi::{SamplingBinding, SamplingInput, SamplingProgramSubmission};

        let prog0 = SamplingProgramSubmission {
            bytecode: vec![0xDE, 0xAD],
            inputs: vec![
                SamplingInput { key: 10, bytes: vec![1, 2, 3] },
                SamplingInput { key: 11, bytes: vec![4, 5] },
            ],
            bindings: vec![SamplingBinding::Logits, SamplingBinding::Tensor { key: 11 }],
            late_keys: vec![100, 200],
            // key 100 has a staged host value; key 200 is a device-alias (no value).
            late_inputs: vec![SamplingInput { key: 100, bytes: vec![9, 9, 9, 9] }],
        };
        let prog1 = SamplingProgramSubmission {
            bytecode: vec![0xBE, 0xEF, 0x01],
            inputs: vec![SamplingInput { key: 20, bytes: vec![7] }],
            bindings: vec![SamplingBinding::Tensor { key: 20 }],
            late_keys: vec![300],
            late_inputs: vec![],
        };

        let mut req0 = make_request(vec![1], vec![10], vec![]);
        req0.push_sampling_program(&prog0);
        let mut req1 = make_request(vec![2], vec![20], vec![]);
        req1.push_sampling_program(&prog1);

        let mut batch = new_batched_forward_request();
        append_request(&mut batch, &req0, &[100], 1, 16);
        append_request(&mut batch, &req1, &[200], 1, 16);

        // Byte-for-byte round-trip through the merged SoA + offset CSRs.
        assert_eq!(batch.n_sampling_programs(), 2);
        assert_eq!(batch.sampling_program_at(0), Some(prog0));
        assert_eq!(batch.sampling_program_at(1), Some(prog1));

        // Documented offsets: per-request boundary CSR, then the nested
        // per-program CSRs with req1's ranges rebased onto req0's tail.
        assert_eq!(batch.sampling_program_indptr, vec![0, 1, 2]);
        assert_eq!(batch.sampling_program_bytes, vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01]);
        assert_eq!(batch.sampling_program_bytes_indptr, vec![0, 2, 5]);
        // Input blob concatenated; req1's offset rebased by req0's blob len (5).
        assert_eq!(batch.sampling_input_blob, vec![1, 2, 3, 4, 5, 7]);
        assert_eq!(batch.sampling_input_keys, vec![10, 11, 20]);
        assert_eq!(batch.sampling_input_offsets, vec![0, 3, 5]);
        assert_eq!(batch.sampling_input_lens, vec![3, 2, 1]);
        assert_eq!(batch.sampling_input_indptr, vec![0, 2, 3]);
        // Late channel: keys concatenated, value offsets rebased by the late
        // blob len (4), the len-0 sentinel (device-alias) preserved.
        assert_eq!(batch.sampling_late_keys, vec![100, 200, 300]);
        assert_eq!(batch.sampling_late_blob, vec![9, 9, 9, 9]);
        assert_eq!(batch.sampling_late_offsets, vec![0, 4, 4]);
        assert_eq!(batch.sampling_late_lens, vec![4, 0, 0]);
        assert_eq!(batch.sampling_late_indptr, vec![0, 2, 3]);
        // Binding map: (kind, key) slots concatenated, CSR rebased.
        assert_eq!(batch.sampling_binding_kind, vec![0, 1, 1]);
        assert_eq!(batch.sampling_binding_key, vec![0, 11, 20]);
        assert_eq!(batch.sampling_binding_indptr, vec![0, 2, 3]);
    }

    /// The intrinsic-kind ENCODING contract: `push_sampling_program` must emit
    /// the manifest binding kind (`SamplingBinding::kind()`) into
    /// `sampling_binding_kind` per slot — Logits→0, Tensor→1, **MtpLogits→2**,
    /// **MtpDrafts→3**. This is the wire byte charlie's driver decode
    /// (`manifest_to_slot_bindings`) reads to set `InputBind.intrinsic_kind`;
    /// if the encoding collapsed the intrinsic kinds here, every intrinsic would
    /// resolve as the sampled Logits row (the (b) A/B regression). Locks the
    /// contract so a future binding shuffle can't silently drop the kind.
    #[test]
    fn sampling_binding_intrinsic_kind_encoded_per_slot() {
        use pie_driver_abi::{SamplingBinding, SamplingProgramSubmission};

        // One program binding all four kinds, in slot order.
        let prog = SamplingProgramSubmission {
            bytecode: vec![0x01],
            inputs: vec![],
            bindings: vec![
                SamplingBinding::Logits,
                SamplingBinding::MtpLogits,
                SamplingBinding::MtpDrafts,
                SamplingBinding::Tensor { key: 42 },
            ],
            late_keys: vec![],
            late_inputs: vec![],
        };

        let mut req = make_request(vec![1], vec![10], vec![]);
        req.push_sampling_program(&prog);

        // Emitted wire kinds/keys, in slot order: the intrinsics are payload-less
        // (key 0); the tensor carries its TensorKey.
        assert_eq!(req.sampling_binding_kind, vec![0, 2, 3, 1]);
        assert_eq!(req.sampling_binding_key, vec![0, 0, 0, 42]);
        assert_eq!(req.sampling_binding_indptr, vec![0, 4]);

        // Round-trips byte-for-byte through a merged batch (decode ≡ encode).
        let mut batch = new_batched_forward_request();
        append_request(&mut batch, &req, &[], 1, 16);
        assert_eq!(batch.sampling_program_at(0), Some(prog));
        assert_eq!(batch.sampling_binding_kind, vec![0, 2, 3, 1]);
    }

    /// The #27 cut #2 device-alias late channel (`sampling_late_device_*`) is
    /// value-parallel to `sampling_late_keys` (one device ptr + R12 flag + len
    /// per late key), so the batch fold concatenates it verbatim — no CSR
    /// offsetting. A miss here silently strands the device-resident late value.
    #[test]
    fn sampling_late_device_alias_channel_merges_verbatim() {
        let mut req0 = make_request(vec![1], vec![10], vec![]);
        req0.sampling_late_keys = vec![100, 200];
        req0.sampling_late_indptr = vec![0, 2];
        req0.sampling_late_device_ptrs = vec![0xAAAA, 0];
        req0.sampling_late_device_flags = vec![1, 0];
        req0.sampling_late_device_lens = vec![8, 0];

        let mut req1 = make_request(vec![2], vec![20], vec![]);
        req1.sampling_late_keys = vec![300];
        req1.sampling_late_indptr = vec![0, 1];
        req1.sampling_late_device_ptrs = vec![0xBBBB];
        req1.sampling_late_device_flags = vec![1];
        req1.sampling_late_device_lens = vec![4];

        let mut batch = new_batched_forward_request();
        append_request(&mut batch, &req0, &[100], 1, 16);
        append_request(&mut batch, &req1, &[200], 1, 16);

        assert_eq!(batch.sampling_late_keys, vec![100, 200, 300]);
        // Parallel device arrays concatenated verbatim (no rebasing).
        assert_eq!(batch.sampling_late_device_ptrs, vec![0xAAAA, 0, 0xBBBB]);
        assert_eq!(batch.sampling_late_device_flags, vec![1, 0, 1]);
        assert_eq!(batch.sampling_late_device_lens, vec![8, 0, 4]);
    }

    /// Legacy sampler SoA is unchanged by the program carrier: a batch of
    /// plain-sampler requests (no programs) folds its sampler SoA + per-request
    /// `sampler_indptr` exactly as before, and carries no program payload. This
    /// guards the S1 "legacy sampler SoA unchanged" invariant.
    #[test]
    fn legacy_sampler_soa_unchanged_alongside_program_carrier() {
        use pie_driver_abi::Sampler;

        let mut req0 = make_request(vec![1], vec![10], vec![]);
        req0.sampling_indices = vec![0];
        req0.set_samplers(&[Sampler::Multinomial { temperature: 1.0, seed: 42 }]);
        let mut req1 = make_request(vec![2], vec![20], vec![]);
        req1.sampling_indices = vec![0];
        req1.set_samplers(&[
            Sampler::TopK { temperature: 0.7, k: 40 },
            Sampler::MinP { temperature: 1.0, p: 0.05 },
        ]);

        let mut batch = new_batched_forward_request();
        append_request(&mut batch, &req0, &[100], 1, 16);
        append_request(&mut batch, &req1, &[200], 1, 16);

        // Per-request sampler CSR: req0 contributes 1 slot, req1 contributes 2.
        assert_eq!(batch.sampler_indptr, vec![0, 1, 3]);
        assert_eq!(batch.n_samplers(), 3);
        // Round-trip each merged sampler slot back to its origin.
        assert_eq!(
            batch.sampler_at(0),
            Some(Sampler::Multinomial { temperature: 1.0, seed: 42 })
        );
        assert_eq!(batch.sampler_at(1), Some(Sampler::TopK { temperature: 0.7, k: 40 }));
        assert_eq!(batch.sampler_at(2), Some(Sampler::MinP { temperature: 1.0, p: 0.05 }));
        // No program carrier populated by the legacy path.
        assert_eq!(batch.n_sampling_programs(), 0);
        assert!(batch.sampling_program_bytes.is_empty());
    }
}
