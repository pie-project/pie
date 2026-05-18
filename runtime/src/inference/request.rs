//! Per-request and batched forward-pass helpers built on the canonical
//! schema types.
//!
//! The wire schema lives in `pie_bridge` (rkyv-derived Rust structs).
//! Pie reuses [`pie_bridge::ForwardRequest`] and [`pie_bridge::ForwardResponse`]
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
use crate::context::ContextId;
use crate::context::pagestore::PhysicalPageId;
use crate::driver::DriverId;
use pie_bridge::Brle;

/// Build a per-request [`pie_bridge::ForwardRequest`].
///
/// `kv_page_indices` and `kv_last_page_lens` are left empty — the
/// scheduler fills them in during batching from the physical page list
/// it resolves out-of-band. `qo_indptr` and the other indptrs encode
/// the single-element shape (`[0, N]`) so the value is a valid
/// `ForwardRequest` on its own should anyone want to send it that way.
#[allow(clippy::too_many_arguments)]
pub fn new_per_request(
    context_id: ContextId,
    tokens: Vec<u32>,
    positions: Vec<u32>,
    masks: Vec<Brle>,
    has_user_mask: bool,
    logit_mask: Option<Brle>,
    sampling_indices: Vec<u32>,
    samplers: Vec<pie_bridge::Sampler>,
    speculative_tokens: Vec<u32>,
    speculative_positions: Vec<u32>,
    output_speculative_tokens: bool,
    adapter_id: Option<AdapterId>,
    adapter_seed: Option<i64>,
) -> pie_bridge::ForwardRequest {
    let n_tokens = tokens.len() as u32;
    let n_masks = masks.len() as u32;
    let n_sampling = sampling_indices.len() as u32;
    let n_samplers = samplers.len() as u32;
    let n_spec = speculative_tokens.len() as u32;
    let logit_masks: Vec<Brle> = logit_mask.into_iter().collect();
    let n_logit = logit_masks.len() as u32;
    let single_token_mode = !has_user_mask && n_tokens <= 1;

    pie_bridge::ForwardRequest {
        token_ids: tokens,
        position_ids: positions,
        kv_page_indices: Vec::new(),
        kv_page_indptr: vec![0],
        kv_last_page_lens: Vec::new(),
        qo_indptr: vec![0, n_tokens],
        masks,
        mask_indptr: vec![0, n_masks],
        logit_masks,
        logit_mask_indptr: vec![0, n_logit],
        sampling_indices,
        sampling_indptr: vec![0, n_sampling],
        samplers,
        sampler_indptr: vec![0, n_samplers],
        adapter_bindings: vec![pie_bridge::AdapterBinding {
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
    }
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
        pie_bridge::brle::set_bits(&mut eligible, 0, first_writeable_page);

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
// Batched-request accumulator (free functions on pie_bridge::ForwardRequest)
// =============================================================================

/// Initialize a `pie_bridge::ForwardRequest` for the empty-batch state:
/// indptrs seeded with `[0]` so subsequent `append_request` calls can
/// push the rolling totals. `single_token_mode` starts at `true`; the
/// first per-request append that needs `custom_mask` flips it to false.
pub fn new_batched_forward_request() -> pie_bridge::ForwardRequest {
    pie_bridge::ForwardRequest {
        kv_page_indptr: vec![0],
        qo_indptr: vec![0],
        mask_indptr: vec![0],
        logit_mask_indptr: vec![0],
        sampling_indptr: vec![0],
        sampler_indptr: vec![0],
        spec_indptr: vec![0],
        single_token_mode: true,
        ..Default::default()
    }
}

/// Wrap a batched [`pie_bridge::ForwardRequest`] in a routable Frame.
pub fn forward_frame(driver_id: DriverId, req: pie_bridge::ForwardRequest) -> pie_bridge::Frame {
    pie_bridge::Frame {
        driver_id: driver_id as u32,
        payload: pie_bridge::RequestPayload::Forward(req),
    }
}

/// Append the request's physical page IDs to `kv_page_indices`,
/// honoring the trim plan if present.
fn emit_kv_pages(
    batch: &mut pie_bridge::ForwardRequest,
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
    batch: &mut pie_bridge::ForwardRequest,
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

/// Append a per-request [`pie_bridge::ForwardRequest`] into the batched form.
/// `req` is the single-element shape produced by [`new_per_request`]
/// (indptrs `[0, N]`, empty kv pages). The scheduler resolved
/// `physical_page_ids` and `last_page_len` out-of-band; this call
/// folds them in along with the page-trim plan derived from
/// `req.masks`. See the file-level docs for trim criteria.
pub fn append_request(
    batch: &mut pie_bridge::ForwardRequest,
    req: &pie_bridge::ForwardRequest,
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

/// Append a per-request [`pie_bridge::ForwardRequest`] with caller-selected
/// decode mask elision. `elide_decode_masks` is only valid when the entire
/// batch is pure single-token decode; mixed prefill/decode batches need one
/// flattened mask row per query row for the bridge's custom-mask view.
pub fn append_request_with_options(
    batch: &mut pie_bridge::ForwardRequest,
    req: &pie_bridge::ForwardRequest,
    physical_page_ids: &[PhysicalPageId],
    last_page_len: u32,
    page_size: u32,
    elide_decode_masks: bool,
) {
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
    batch
        .kv_page_indptr
        .push(batch.kv_page_indices.len() as u32);
    batch.kv_last_page_lens.push(last_page_len);
    batch.qo_indptr.push(batch.token_ids.len() as u32);

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

    // Samplers: variants flow through directly.
    batch.samplers.extend(req.samplers.iter().cloned());
    batch.sampler_indptr.push(batch.samplers.len() as u32);

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

/// Extract request `r`'s slice from a batched `pie_bridge::ForwardResponse`
/// into a single-request `ForwardResponse` (with `num_requests = 1` and
/// indptrs offset to zero). This is the "scatter" step that lets each
/// inferlet's response future see only its own request's data.
pub fn extract_per_request(
    fr: &pie_bridge::ForwardResponse,
    r: usize,
) -> pie_bridge::ForwardResponse {
    let mut out = pie_bridge::ForwardResponse {
        num_requests: 1,
        ..Default::default()
    };

    // Tokens: one indptr range per request.
    let (tok_lo, tok_hi) = indptr_range(&fr.tokens_indptr, r);

    // Hot path for normal generation: token samples only, no probe payloads.
    // Avoid allocating several empty indptr vectors for every request in
    // every decode batch.
    let token_payload_only = fr.dists_ids.is_empty()
        && fr.dists_probs.is_empty()
        && fr.logits_bytes.is_empty()
        && fr.logprobs_values.is_empty()
        && fr.entropies.is_empty();
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

    // -- Page-trim integration tests -----------------------------------------

    fn make_request(
        tokens: Vec<u32>,
        positions: Vec<u32>,
        masks: Vec<Brle>,
    ) -> pie_bridge::ForwardRequest {
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
        // Mask buffer is the original BRLE (no rewrite).
        assert_eq!(batch.masks, vec![causal.clone()]);
        assert_eq!(batch.mask_indptr, vec![0, 1]);
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
}
