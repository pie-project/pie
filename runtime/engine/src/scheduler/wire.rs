//! Per-request and batched runtime launch-plan helpers.
//!
//! Owned values stay in the runtime and are lowered to borrowed FFI descriptors
//! only at the scheduler's native-driver invocation boundary.
//! Batched plans preserve geometry, channel-backed masks, recurrent state, and
//! multimodal side data without recreating a request protocol.

use smallvec::{SmallVec, smallvec};

use pie_driver_abi::EncodedMask;
use pie_grammar::brle::RunMask;

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
    /// `RunMask::write_skipping` for each row.
    skip_ranges: Vec<(u32, u32)>,
}

impl TrimPlan {
    /// Compute the trim plan, or return `None` if no pages can be dropped.
    /// Returning `None` means the caller should take the fast path with
    /// zero extra allocations.
    fn compute(
        masks: &[RunMask],
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
        pie_grammar::brle::set_bits(&mut eligible, 0, first_writeable_page);

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
// Batched-request accumulator (free functions on crate::driver::LaunchPlan)
// =============================================================================

/// Initialize a `crate::driver::LaunchPlan` for the empty-batch state:
/// indptrs seeded with `[0]` so subsequent `append_request` calls can
/// push the rolling totals. `single_token_mode` starts at `true`; the
/// first per-request append that needs `custom_mask` flips it to false.
#[allow(dead_code)] // batch.rs always calls `new_batched_forward_request_with_capacity` directly.
pub fn new_batched_forward_request() -> crate::driver::LaunchPlan {
    new_batched_forward_request_with_capacity(0)
}

/// Same as [`new_batched_forward_request`] but pre-allocates Vec
/// capacities based on an expected request count, eliminating
/// per-append reallocations during `batch_build_us`. Pass 0 if you
/// don't know.
pub fn new_batched_forward_request_with_capacity(n_requests: usize) -> crate::driver::LaunchPlan {
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
    crate::driver::LaunchPlan {
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
        sampling_indices: Vec::with_capacity(req_cap),
        sampling_indptr: indptr(indptr_cap),
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
        embed_rows: Vec::new(),
        embed_indptr: indptr(indptr_cap),
        embed_shapes: Vec::new(),
        embed_dtypes: Vec::new(),
        embed_anchor_rows: Vec::new(),
        embed_block_indptr: indptr(indptr_cap),
        ..Default::default()
    }
}

/// Append the request's `kv_page_indices` to the batch's `kv_page_indices`,
/// honoring the trim plan if present.
fn emit_kv_pages(
    batch: &mut crate::driver::LaunchPlan,
    pages: &[u32],
    trim: Option<&TrimPlan>,
) {
    match trim {
        None => batch.kv_page_indices.extend(pages),
        Some(plan) => {
            for (idx, &pid) in pages.iter().enumerate() {
                if !plan.is_page_dropped(idx as u32) {
                    batch.kv_page_indices.push(pid);
                }
            }
        }
    }
}

/// Append one BRLE per row into `batch.masks`, applying the trim plan's
/// skip ranges if present. `mask_indptr` is per-request (one entry
/// pushed at the end), so each request contributes `masks.len()` RunMask
/// rows.
fn emit_attention_masks(
    batch: &mut crate::driver::LaunchPlan,
    masks: &[RunMask],
    trim: Option<&TrimPlan>,
) {
    match trim {
        None => {
            batch.masks.extend(
                masks
                    .iter()
                    .map(|mask| EncodedMask::new(mask.buffer.clone(), mask.total_size)),
            );
        }
        Some(plan) => {
            for mask in masks {
                let mut buf = Vec::new();
                let new_total = mask.write_skipping(&plan.skip_ranges, &mut buf);
                batch.masks.push(EncodedMask::new(buf, new_total as u64));
            }
        }
    }
    batch.mask_indptr.push(batch.masks.len() as u32);
}

/// Append a per-request [`crate::driver::LaunchPlan`] into the batched form.
/// `req` is the single-element shape produced by [`new_per_request`]
/// (indptrs `[0, N]`). The request carries its own explicit
/// `kv_page_indices` (single-explicit-arity contract); the scheduler
/// resolved `last_page_len` out-of-band. This call folds them in along
/// with the page-trim plan derived from `req.masks`. See the file-level
/// docs for trim criteria.
#[allow(dead_code)] // batch.rs always calls `append_request_with_options` directly.
pub fn append_request(
    batch: &mut crate::driver::LaunchPlan,
    req: &crate::driver::LaunchPlan,
    last_page_len: u32,
    page_size: u32,
) {
    append_request_with_options(batch, req, last_page_len, page_size, false);
}

/// Append a per-request [`crate::driver::LaunchPlan`] with caller-selected
/// decode mask elision. `elide_decode_masks` is only valid when the entire
/// batch is pure single-token decode; mixed prefill/decode batches need one
/// flattened mask row per query row for the bridge's custom-mask view.
pub fn append_request_with_options(
    batch: &mut crate::driver::LaunchPlan,
    req: &crate::driver::LaunchPlan,
    last_page_len: u32,
    page_size: u32,
    elide_decode_masks: bool,
) {
    if req.qo_indptr.len().saturating_sub(1) > 1 {
        append_multi_row_request(batch, req, last_page_len, page_size);
        return;
    }
    // Row offset of this request's tokens within the batch — image anchor rows
    // shift by this when merged (captured before extending `token_ids`).
    let row_base = batch.token_ids.len() as u32;
    // Tokens and positions
    batch.token_ids.extend(&req.token_ids);
    batch.position_ids.extend(&req.position_ids);

    let elide_decode_mask = req.device_resolved_geometry
        || (elide_decode_masks
            && req.single_token_mode
            && !req.has_user_mask
            && req.token_ids.len() <= 1);

    let synthesized_masks;
    let decoded_masks;
    let masks = if !elide_decode_mask && req.masks.is_empty() && !req.position_ids.is_empty() {
        synthesized_masks = req
            .position_ids
            .iter()
            .map(|&pos| RunMask::all_true((pos + 1) as usize))
            .collect::<Vec<_>>();
        synthesized_masks.as_slice()
    } else {
        decoded_masks = req
            .masks
            .iter()
            .map(|mask| RunMask {
                buffer: mask.runs.clone(),
                total_size: mask.total_size,
            })
            .collect::<Vec<_>>();
        decoded_masks.as_slice()
    };

    let trim = if elide_decode_mask {
        None
    } else {
        TrimPlan::compute(
            masks,
            req.kv_page_indices.len() as u32,
            last_page_len,
            page_size,
            req.token_ids.len() as u32,
        )
    };

    // KV cache layout.
    emit_kv_pages(batch, &req.kv_page_indices, trim.as_ref());
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
    batch.rs_fold_lens.extend(&req.rs_fold_lens);
    batch.rs_buffer_slot_ids.extend(&req.rs_buffer_slot_ids);
    batch
        .rs_buffer_slot_indptr
        .push(batch.rs_buffer_slot_ids.len() as u32);

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

    fn append_multi_row_request(
        batch: &mut crate::driver::LaunchPlan,
        req: &crate::driver::LaunchPlan,
        fallback_last_page_len: u32,
        page_size: u32,
    ) {
        let rows = req.qo_indptr.len() - 1;
        assert_eq!(
            req.qo_indptr[0], 0,
            "multi-row query CSR must start at zero"
        );
        assert_eq!(
            req.qo_indptr[rows] as usize,
            req.token_ids.len(),
            "multi-row query CSR must cover every token"
        );
        assert_eq!(
            req.position_ids.len(),
            req.token_ids.len(),
            "multi-row positions must cover every token"
        );
        let deferred_geometry = req.device_resolved_geometry && req.kv_page_indptr.is_empty();
        if !deferred_geometry {
            assert_eq!(
                req.kv_page_indptr.len(),
                rows + 1,
                "multi-row KV CSR must cover every row"
            );
        }
        assert_eq!(
            req.sampling_indptr.len(),
            rows + 1,
            "multi-row sampling CSR must cover every row"
        );

        let token_base = batch.token_ids.len() as u32;
        batch.token_ids.extend(&req.token_ids);
        batch.position_ids.extend(&req.position_ids);
        for &boundary in req.qo_indptr.iter().skip(1) {
            batch.qo_indptr.push(token_base + boundary);
        }

        if deferred_geometry {
            for _ in 0..rows {
                batch
                    .kv_page_indptr
                    .push(batch.kv_page_indices.len() as u32);
                batch.kv_last_page_lens.push(0);
                batch.kv_len.push(0);
            }
        } else {
            assert_eq!(
                req.kv_page_indptr.last().copied().unwrap_or(0) as usize,
                req.kv_page_indices.len(),
                "multi-row KV CSR must cover every page"
            );
            let page_base = batch.kv_page_indices.len() as u32;
            batch.kv_page_indices.extend(&req.kv_page_indices);
            for &boundary in req.kv_page_indptr.iter().skip(1) {
                batch.kv_page_indptr.push(page_base + boundary);
            }
            for row in 0..rows {
                let page_count = req.kv_page_indptr[row + 1] - req.kv_page_indptr[row];
                let last = req
                    .kv_last_page_lens
                    .get(row)
                    .copied()
                    .unwrap_or(fallback_last_page_len);
                batch.kv_last_page_lens.push(last);
                batch.kv_len.push(if page_count == 0 {
                    0
                } else {
                    (page_count - 1) * page_size + last
                });
            }
        }

        let sample_base = batch.sampling_indices.len() as u32;
        for row in 0..rows {
            let begin = req.sampling_indptr[row] as usize;
            let end = req.sampling_indptr[row + 1] as usize;
            let row_len = req.qo_indptr[row + 1] - req.qo_indptr[row];
            for &index in &req.sampling_indices[begin..end] {
                assert!(index < row_len, "multi-row sampling index exceeds its row");
                batch.sampling_indices.push(index);
            }
            batch
                .sampling_indptr
                .push(sample_base + batch.sampling_indices.len() as u32 - sample_base);
        }

        if req.mask_indptr.len() == rows + 1 {
            let mask_base = batch.masks.len() as u32;
            batch.masks.extend(req.masks.iter().cloned());
            for &boundary in req.mask_indptr.iter().skip(1) {
                batch.mask_indptr.push(mask_base + boundary);
            }
        } else {
            // Admission (`single_request_limit_error`) rejects multi-row
            // masks without a row CSR; by batch build it is an invariant —
            // a hard assert here would panic the scheduler thread on a
            // malformed FIRE instead of rejecting that one fire (RV-20).
            debug_assert!(req.masks.is_empty(), "multi-row masks require a row CSR");
            for _ in 0..rows {
                batch.mask_indptr.push(batch.masks.len() as u32);
            }
        }

        assert!(
            req.rs_slot_ids.is_empty() || req.rs_slot_ids.len() == rows,
            "multi-row RS slots must align with rows"
        );
        assert!(
            req.rs_slot_flags.is_empty() || req.rs_slot_flags.len() == rows,
            "multi-row RS flags must align with rows"
        );
        batch.rs_slot_ids.extend(&req.rs_slot_ids);
        batch.rs_slot_flags.extend(&req.rs_slot_flags);
        batch.rs_fold_lens.extend(&req.rs_fold_lens);
        if req.rs_buffer_slot_indptr.len() == rows + 1 {
            let buffer_base = batch.rs_buffer_slot_ids.len() as u32;
            batch.rs_buffer_slot_ids.extend(&req.rs_buffer_slot_ids);
            for &boundary in req.rs_buffer_slot_indptr.iter().skip(1) {
                batch.rs_buffer_slot_indptr.push(buffer_base + boundary);
            }
        } else {
            assert!(
                req.rs_buffer_slot_ids.is_empty(),
                "multi-row RS buffers require a row CSR"
            );
            for _ in 0..rows {
                batch
                    .rs_buffer_slot_indptr
                    .push(batch.rs_buffer_slot_ids.len() as u32);
            }
        }

        batch.context_ids.extend(&req.context_ids);
        assert!(
            req.image_grids.is_empty() && req.audio_features.is_empty(),
            "multi-row multimodal merge is not supported"
        );
        assert!(
            req.embed_rows.is_empty(),
            "multi-row embedding merge is not supported"
        );
        for _ in 0..rows {
            batch
                .image_indptr
                .push((batch.image_grids.len() / 3) as u32);
            batch
                .audio_indptr
                .push(batch.audio_anchor_rows.len() as u32);
            batch
                .embed_block_indptr
                .push(batch.embed_dtypes.len() as u32);
        }
        batch.single_token_mode = false;
        batch.has_user_mask |= req.has_user_mask;
    }

    // Sampling indices.
    batch.sampling_indices.extend(&req.sampling_indices);
    batch
        .sampling_indptr
        .push(batch.sampling_indices.len() as u32);

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
    batch
        .image_mrope_positions
        .extend(&req.image_mrope_positions);
    for &off in req.image_mrope_indptr.iter().skip(1) {
        batch.image_mrope_indptr.push(mrope_base + off);
    }

    batch
        .image_indptr
        .push((batch.image_grids.len() / 3) as u32);

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

    let embed_byte_base = batch.embed_rows.len() as u32;
    batch.embed_rows.extend(&req.embed_rows);
    for &offset in req.embed_indptr.iter().skip(1) {
        batch.embed_indptr.push(embed_byte_base + offset);
    }
    batch.embed_shapes.extend(&req.embed_shapes);
    batch.embed_dtypes.extend(&req.embed_dtypes);
    for &row in &req.embed_anchor_rows {
        batch.embed_anchor_rows.push(row_base + row);
    }
    batch
        .embed_block_indptr
        .push(batch.embed_dtypes.len() as u32);

    // Inference hint: prefill kernel when ANY request needs `custom_mask`.
    if req.token_ids.len() > 1 || req.has_user_mask {
        batch.single_token_mode = false;
    }
    if req.has_user_mask {
        batch.has_user_mask = true;
    }
}
