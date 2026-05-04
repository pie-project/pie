//! Per-request work: decode the BPIQ payload, fabricate per-slot
//! outputs based on the sampler type, and emit a msgpack-mode BPIS
//! response. Token-producing slots (Argmax/TopP/TopK/MinP/TopKTopP/
//! Multinomial) get a random token, drawn from the per-request
//! constraint mask when one is set so grammar/JSON-schema examples
//! produce in-grammar output. Probe slots (RawLogits, Logprob,
//! Logprobs, Distribution, Entropy) still emit zero-shaped placeholders
//! — those values are not derivable from the dummy's no-compute path.

use rand::{RngCore, SeedableRng};
use rand::rngs::SmallRng;

use crate::schema::{
    self, BatchedForwardPassResponse, DecodedRequest, ForwardPassResponse, sampler_type,
};

pub struct Handler {
    rng: SmallRng,
    vocab_size: u32,
    /// Pre-sized zero buffer for RawLogits slots — `vocab_size * 4`
    /// bytes of native-endian f32 zeros. Cloned per slot rather than
    /// rebuilt; argmax of zero-filled logits is deterministic, which
    /// is fine for plumbing tests.
    zero_logits: Vec<u8>,
}

impl Handler {
    pub fn new(seed: u64, vocab_size: u32) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
            vocab_size,
            zero_logits: vec![0u8; (vocab_size as usize) * 4],
        }
    }

    /// Decode the request and write a msgpack response. Returns bytes
    /// written; 0 on any error (the runtime treats empty as failure
    /// and the caller logs to stderr).
    pub fn handle_fire_batch(&mut self, request: &[u8], response_buf: &mut [u8]) -> usize {
        let decoded: DecodedRequest = match schema::decode_request(request) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("[pie-driver-dummy] decode failed: {e}");
                return 0;
            }
        };

        let mut batched = BatchedForwardPassResponse {
            results: Vec::with_capacity(decoded.request_num_samplers.len()),
        };
        let mut slot_cursor: usize = 0;
        for (req_idx, &n_slots) in decoded.request_num_samplers.iter().enumerate() {
            let mut resp = ForwardPassResponse::default();
            let allowed = request_brle(decoded.logit_masks, decoded.logit_mask_indptr, req_idx)
                .and_then(|brle| AllowedRuns::parse(brle, self.vocab_size));
            for _ in 0..n_slots {
                if let Some(&ty) = decoded.sampler_types.get(slot_cursor) {
                    self.fill_slot(
                        ty,
                        slot_cursor,
                        decoded.sampler_label_indptr,
                        allowed.as_ref(),
                        &mut resp,
                    );
                } else {
                    eprintln!(
                        "[pie-driver-dummy] sampler_types short: slot {slot_cursor} \
                         missing (have {})",
                        decoded.sampler_types.len()
                    );
                }
                slot_cursor += 1;
            }
            batched.results.push(resp);
        }

        match schema::encode_msgpack_response(response_buf, &batched) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("[pie-driver-dummy] encode failed: {e}");
                0
            }
        }
    }

    fn fill_slot(
        &mut self,
        ty: u32,
        slot: usize,
        label_indptr: &[u32],
        allowed: Option<&AllowedRuns>,
        resp: &mut ForwardPassResponse,
    ) {
        match ty {
            sampler_type::TOP_P
            | sampler_type::TOP_K
            | sampler_type::TOP_K_TOP_P
            | sampler_type::MIN_P
            | sampler_type::MULTINOMIAL => {
                let token = match allowed {
                    Some(a) => self.random_masked_token(a),
                    None => self.random_token(),
                };
                resp.tokens.push(token);
            }
            sampler_type::DISTRIBUTION => {
                // The runtime contract is "top-K (id, prob) pairs"
                // and many inferlets bind `dist.first()` directly
                // (sampler-suite) or sample from the full list with a
                // re-weighting (watermarking). Empty would technically
                // satisfy the type but force every consumer to
                // dummy-guard the call site, and a single deterministic
                // entry would make watermarking-style demos produce
                // the same token every step (`!!!!...`). Emit 8
                // random ids with a geometric-decay probability
                // distribution that sums to ≈1 instead — dimensionally
                // valid, downstream cross-checks (sorted, sum≤1) hold,
                // and there's enough variety for a sample-from-dist
                // loop to produce varied output.
                let mut ids = Vec::with_capacity(8);
                for _ in 0..8 {
                    ids.push(self.random_token());
                }
                let probs = vec![
                    0.5_f32,
                    0.25,
                    0.125,
                    0.0625,
                    0.03125,
                    0.015625,
                    0.0078125,
                    0.0078125,
                ];
                resp.dists.push((ids, probs));
            }
            sampler_type::RAW_LOGITS => {
                resp.logits.push(self.zero_logits.clone());
            }
            sampler_type::LOGPROB => {
                resp.logprobs.push(vec![0.0]);
            }
            sampler_type::LOGPROBS => {
                let k = label_count(label_indptr, slot);
                resp.logprobs.push(vec![0.0; k]);
            }
            sampler_type::ENTROPY => {
                resp.entropies.push(0.0);
            }
            sampler_type::EMBEDDING => {
                // Reserved type id 6. The runtime currently filters
                // Embedding out of the response slot stream
                // (`scheduler.rs::Sampler::Embedding => None`), so
                // pushing nothing here keeps the dummy's response
                // aligned with what the host decoder expects.
            }
            other => {
                eprintln!("[pie-driver-dummy] unknown sampler_type={other} at slot {slot}");
            }
        }
    }

    fn random_token(&mut self) -> u32 {
        let mut buf = [0u8; 4];
        self.rng.fill_bytes(&mut buf);
        u32::from_ne_bytes(buf) % self.vocab_size
    }

    /// Pick a uniform random allowed token id by drawing an index in
    /// `[0, allowed.total)` and walking the run list to map it back
    /// to a token id. Falls back to uniform vocab sampling (with a
    /// stderr note) when the mask blocks the entire vocabulary —
    /// that's a degenerate request the dummy can't satisfy
    /// faithfully, but silently returning 0 would mask a genuine bug
    /// upstream.
    ///
    /// The within-mask distribution is uniform-by-token, not
    /// uniform-by-run: a single-token allowed run gets probability
    /// `1/allowed.total`, the same as one token in a 100k-token run.
    /// Earlier prototypes biased toward "the shortest run" or "the
    /// lowest-id token" as a way to nudge grammars toward closing
    /// (closing literals tend to be short or low-id). Both biases
    /// produced visible artifacts — repeated `!` filler, `im` filler,
    /// long whitespace runs — without actually fixing the
    /// convergence story for the larger grammars (`json-schema-
    /// validation` etc.). The honest answer is: long-tail grammars
    /// may truncate at `max_tokens` on the dummy, and inferlets that
    /// post-process the output should handle that gracefully (see
    /// `inferlets/json-schema-validation/src/lib.rs`).
    fn random_masked_token(&mut self, allowed: &AllowedRuns) -> u32 {
        if allowed.total == 0 {
            eprintln!("[pie-driver-dummy] logit_mask allows zero tokens; falling back to uniform");
            return self.random_token();
        }
        let mut buf = [0u8; 4];
        self.rng.fill_bytes(&mut buf);
        let mut idx = u32::from_ne_bytes(buf) % allowed.total;
        for &(start, len) in &allowed.runs {
            if idx < len {
                return start + idx;
            }
            idx -= len;
        }
        unreachable!("allowed.total is the sum of run lengths")
    }
}

/// Slice the per-request BRLE buffer out of the flat `logit_masks`
/// array. Returns `None` when the indptr is shorter than expected, the
/// request's slice is empty, or the bounds are inconsistent — all of
/// which mean "no constraint, sample uniform from the whole vocab".
fn request_brle<'a>(masks: &'a [u32], indptr: &[u32], req_idx: usize) -> Option<&'a [u32]> {
    let lo = indptr.get(req_idx).copied()? as usize;
    let hi = indptr.get(req_idx + 1).copied()? as usize;
    if hi <= lo || hi > masks.len() {
        return None;
    }
    Some(&masks[lo..hi])
}

/// The allowed-token spans of a per-request BRLE constraint mask,
/// clipped to the configured vocab size.
struct AllowedRuns {
    /// Each entry is `(start_token_id, run_length)` for a contiguous
    /// span of allowed token ids. Half-open: covers
    /// `[start_token_id, start_token_id + run_length)`.
    runs: Vec<(u32, u32)>,
    /// Sum of `run_length` across `runs` — the size of the allowed
    /// alphabet, used as the modulus when picking a uniform random
    /// token from the mask.
    total: u32,
}

impl AllowedRuns {
    /// Parse a BRLE buffer into the list of allowed-token spans.
    /// Returns `None` when the buffer is empty, which the BPIQ schema
    /// uses to mean "no constraint".
    fn parse(brle: &[u32], vocab_size: u32) -> Option<Self> {
        if brle.is_empty() {
            return None;
        }
        let mut runs = Vec::new();
        let mut total: u32 = 0;
        let mut pos: u32 = 0;
        for (i, &len) in brle.iter().enumerate() {
            if pos >= vocab_size {
                break;
            }
            let clipped = len.min(vocab_size - pos);
            // BRLE convention: even index = false run, odd index = true.
            let is_true = i % 2 != 0;
            if is_true && clipped > 0 {
                runs.push((pos, clipped));
                total += clipped;
            }
            pos = pos.saturating_add(clipped);
        }
        Some(Self { runs, total })
    }
}

/// Read `label_indptr[slot+1] - label_indptr[slot]`, defaulting to 0 if
/// the indptr is shorter than expected (the runtime can omit it for
/// requests with no Logprob/Logprobs slots).
fn label_count(indptr: &[u32], slot: usize) -> usize {
    let lo = indptr.get(slot).copied().unwrap_or(0);
    let hi = indptr.get(slot + 1).copied().unwrap_or(lo);
    hi.saturating_sub(lo) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{NUM_ARRAYS, REQ_HEADER_SIZE, REQ_MAGIC, REQ_SCHEMA_VERSION};

    const A_LOGIT_MASKS: usize = 8;
    const A_LOGIT_MASK_INDPTR: usize = 9;
    const A_SAMPLER_TYPES: usize = 16;
    const A_REQUEST_NUM_SAMPLERS: usize = 18;
    const A_SAMPLER_LABEL_INDPTR: usize = 20;
    const FIXED_HEADER: usize = 32;

    fn align_up(n: usize, a: usize) -> usize { (n + a - 1) & !(a - 1) }

    /// Build a BPIQ buffer with all five arrays the dummy decodes.
    /// Pass empty slices for anything the test doesn't care about.
    fn build_request(
        nums: &[u32],
        types: &[u32],
        label_indptr: &[u32],
        logit_masks: &[u32],
        logit_mask_indptr: &[u32],
    ) -> Vec<u8> {
        let mut bodies: Vec<(usize, &[u32])> = Vec::new();
        let mut cursor = REQ_HEADER_SIZE;
        for arr in [nums, types, label_indptr, logit_masks, logit_mask_indptr] {
            cursor = align_up(cursor, 8);
            bodies.push((cursor, arr));
            cursor += arr.len() * 4;
        }
        let mut buf = vec![0u8; cursor];
        buf[0..4].copy_from_slice(&REQ_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&REQ_SCHEMA_VERSION.to_le_bytes());
        buf[16..20].copy_from_slice(&(NUM_ARRAYS as u32).to_le_bytes());
        for (idx, (off, arr)) in [
            A_REQUEST_NUM_SAMPLERS,
            A_SAMPLER_TYPES,
            A_SAMPLER_LABEL_INDPTR,
            A_LOGIT_MASKS,
            A_LOGIT_MASK_INDPTR,
        ]
        .iter()
        .zip(bodies.iter())
        {
            let entry = FIXED_HEADER + idx * 8;
            buf[entry..entry + 4].copy_from_slice(&(*off as u32).to_le_bytes());
            buf[entry + 4..entry + 8].copy_from_slice(&(arr.len() as u32).to_le_bytes());
            for (i, &v) in arr.iter().enumerate() {
                let o = off + i * 4;
                buf[o..o + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }

    fn decode_response(bytes: &[u8]) -> BatchedForwardPassResponseDecoded {
        rmp_serde::from_slice(&bytes[crate::schema::RESP_HEADER_SIZE..]).unwrap()
    }

    #[derive(Debug, serde::Deserialize)]
    struct ForwardPassResponseDecoded {
        tokens: Vec<u32>,
        #[serde(default)]
        dists: Vec<(Vec<u32>, Vec<f32>)>,
        #[serde(default)]
        logits: Vec<Vec<u8>>,
        #[serde(default)]
        logprobs: Vec<Vec<f32>>,
        #[serde(default)]
        entropies: Vec<f32>,
    }
    #[derive(Debug, serde::Deserialize)]
    struct BatchedForwardPassResponseDecoded {
        results: Vec<ForwardPassResponseDecoded>,
    }

    #[test]
    fn token_only_request_produces_one_token_per_slot() {
        let req = build_request(
            &[1, 2, 1],
            &[
                sampler_type::TOP_P,
                sampler_type::TOP_P,
                sampler_type::TOP_P,
                sampler_type::TOP_P,
            ],
            &[],
            &[],
            &[],
        );
        let mut h = Handler::new(42, 1000);
        let mut resp = vec![0u8; 4096];
        let n = h.handle_fire_batch(&req, &mut resp);
        assert!(n > 0);
        let decoded = decode_response(&resp[..n]);
        assert_eq!(decoded.results.len(), 3);
        assert_eq!(decoded.results[0].tokens.len(), 1);
        assert_eq!(decoded.results[1].tokens.len(), 2);
        assert_eq!(decoded.results[2].tokens.len(), 1);
        assert!(decoded.results[0].logits.is_empty());
    }

    #[test]
    fn raw_logits_slot_returns_vocab_sized_buffer() {
        let req = build_request(
            &[1],
            &[sampler_type::RAW_LOGITS],
            &[],
            &[],
            &[],
        );
        let mut h = Handler::new(42, 250);
        let mut resp = vec![0u8; 16 * 1024];
        let n = h.handle_fire_batch(&req, &mut resp);
        assert!(n > 0);
        let decoded = decode_response(&resp[..n]);
        assert_eq!(decoded.results[0].tokens.len(), 0);
        assert_eq!(decoded.results[0].logits.len(), 1);
        assert_eq!(decoded.results[0].logits[0].len(), 250 * 4);
    }

    #[test]
    fn mixed_token_and_probe_in_same_request() {
        let req = build_request(
            &[3],
            &[
                sampler_type::TOP_P,
                sampler_type::ENTROPY,
                sampler_type::LOGPROB,
            ],
            &[0, 0, 0, 1],
            &[],
            &[],
        );
        let mut h = Handler::new(42, 1000);
        let mut resp = vec![0u8; 4096];
        let n = h.handle_fire_batch(&req, &mut resp);
        let decoded = decode_response(&resp[..n]);
        assert_eq!(decoded.results[0].tokens.len(), 1);
        assert_eq!(decoded.results[0].entropies.len(), 1);
        assert_eq!(decoded.results[0].logprobs.len(), 1);
        assert_eq!(decoded.results[0].logprobs[0].len(), 1);
    }

    /// Mask allows only token ids in `[10, 14)`. Sampling many slots
    /// must keep every emitted token inside the allowed run.
    #[test]
    fn logit_mask_constrains_sampler_output() {
        // BRLE: false 10, true 4, false 86 → allow [10, 14) over vocab=100.
        let masks = [10u32, 4, 86];
        let indptr = [0u32, masks.len() as u32];
        let req = build_request(
            &[64], // 64 ARGMAX slots so the test catches even rare misses
            &vec![sampler_type::TOP_P; 64],
            &[],
            &masks,
            &indptr,
        );
        let mut h = Handler::new(42, 100);
        let mut resp = vec![0u8; 16 * 1024];
        let n = h.handle_fire_batch(&req, &mut resp);
        let decoded = decode_response(&resp[..n]);
        assert_eq!(decoded.results[0].tokens.len(), 64);
        for &t in &decoded.results[0].tokens {
            assert!(
                (10..14).contains(&t),
                "token {t} fell outside allowed range [10, 14) — mask was ignored"
            );
        }
    }

    /// Two requests with different masks must not bleed into each
    /// other: request 0 only sees its allowed run, request 1 only sees
    /// its own.
    #[test]
    fn per_request_masks_are_independent() {
        // Request 0: BRLE false 0, true 5 → allow [0, 5)
        // Request 1: BRLE false 95, true 5 → allow [95, 100)
        let masks: Vec<u32> = vec![0, 5, 95, 5];
        let indptr = [0u32, 2, 4];
        let req = build_request(
            &[8, 8],
            &vec![sampler_type::TOP_P; 16],
            &[],
            &masks,
            &indptr,
        );
        let mut h = Handler::new(7, 100);
        let mut resp = vec![0u8; 16 * 1024];
        let n = h.handle_fire_batch(&req, &mut resp);
        let decoded = decode_response(&resp[..n]);
        for &t in &decoded.results[0].tokens {
            assert!((0..5).contains(&t), "request 0 emitted {t} outside [0, 5)");
        }
        for &t in &decoded.results[1].tokens {
            assert!(
                (95..100).contains(&t),
                "request 1 emitted {t} outside [95, 100)"
            );
        }
    }

    /// Empty BRLE for a request means "no constraint" — sampler must
    /// fall back to uniform-over-vocab. Combined with a real mask in a
    /// sibling request to exercise the indptr boundary.
    #[test]
    fn empty_brle_means_no_constraint() {
        // Request 0: empty BRLE (indptr 0..0) → uniform sampling.
        // Request 1: BRLE false 0, true 1 → only token 0.
        let masks: Vec<u32> = vec![0, 1];
        let indptr = [0u32, 0, 2];
        let req = build_request(
            &[1, 4],
            &vec![sampler_type::TOP_P; 5],
            &[],
            &masks,
            &indptr,
        );
        let mut h = Handler::new(11, 50);
        let mut resp = vec![0u8; 16 * 1024];
        let n = h.handle_fire_batch(&req, &mut resp);
        let decoded = decode_response(&resp[..n]);
        // Request 0: just a shape check — uniform sampling can hit
        // anything in [0, 50).
        assert_eq!(decoded.results[0].tokens.len(), 1);
        assert!(decoded.results[0].tokens[0] < 50);
        // Request 1: every slot must be exactly token 0.
        for &t in &decoded.results[1].tokens {
            assert_eq!(t, 0, "constrained slot fell outside the singleton mask");
        }
    }

    /// BRLE with multiple true runs should distribute samples across
    /// every allowed span. We don't check the empirical distribution,
    /// just that no sample lands in a false run.
    #[test]
    fn multi_run_mask_keeps_all_samples_in_true_runs() {
        // BRLE: false 1, true 2, false 1, true 2, false 4 → allow
        // {1, 2, 4, 5} over vocab=10.
        let masks = [1u32, 2, 1, 2, 4];
        let indptr = [0u32, masks.len() as u32];
        let req = build_request(
            &[128],
            &vec![sampler_type::TOP_P; 128],
            &[],
            &masks,
            &indptr,
        );
        let mut h = Handler::new(123, 10);
        let mut resp = vec![0u8; 16 * 1024];
        let n = h.handle_fire_batch(&req, &mut resp);
        let decoded = decode_response(&resp[..n]);
        for &t in &decoded.results[0].tokens {
            assert!(
                matches!(t, 1 | 2 | 4 | 5),
                "token {t} fell into a false run — mask runs were not respected"
            );
        }
    }

    /// BRLE that runs past the vocab boundary should be clipped — no
    /// out-of-vocab token may be emitted.
    #[test]
    fn mask_clips_to_vocab_size() {
        // BRLE: false 0, true 1000 → would allow [0, 1000) but vocab is 50.
        let masks = [0u32, 1000];
        let indptr = [0u32, masks.len() as u32];
        let req = build_request(
            &[64],
            &vec![sampler_type::TOP_P; 64],
            &[],
            &masks,
            &indptr,
        );
        let mut h = Handler::new(99, 50);
        let mut resp = vec![0u8; 16 * 1024];
        let n = h.handle_fire_batch(&req, &mut resp);
        let decoded = decode_response(&resp[..n]);
        for &t in &decoded.results[0].tokens {
            assert!(t < 50, "emitted out-of-vocab token {t}");
        }
    }
}
