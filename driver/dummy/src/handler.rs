//! Per-request work: decode an rkyv-archived `ForwardRequest`, fabricate
//! per-slot outputs based on the `Sampler` variant, and build a
//! `ForwardResponse` to send back. Token-producing samplers
//! (Multinomial/TopK/TopP/MinP/TopKTopP) draw a random token, optionally
//! constrained by a BRLE logit mask. Probe samplers (RawLogits, Logprob,
//! Logprobs, Dist, Entropy) emit zero-shaped placeholders — the dummy's
//! no-compute path can't synthesize real values.

use rand::rngs::SmallRng;
use rand::{RngCore, SeedableRng};

use pie_ipc::wire::encode_response;
use pie_driver_abi::{
    ArchivedForwardRequest, ForwardResponse, ResponseFrame, ResponsePayload, PIE_SAMPLER_DIST,
    PIE_SAMPLER_EMBEDDING, PIE_SAMPLER_ENTROPY, PIE_SAMPLER_LOGPROB, PIE_SAMPLER_LOGPROBS,
    PIE_SAMPLER_RAW_LOGITS,
};

pub struct Handler {
    rng: SmallRng,
    vocab_size: u32,
    /// Pre-sized native-endian zero buffer for `RawLogits` slots.
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

    /// Decode an archived `ForwardRequest`, fabricate outputs, return a
    /// fully-encoded `ResponseFrame` ready to commit on the shmem lease.
    pub fn handle_forward(
        &mut self,
        driver_id: u32,
        fr: &ArchivedForwardRequest,
    ) -> anyhow::Result<Vec<u8>> {
        // Per-request output accumulators (wire SoA shape).
        let mut tokens_indptr: Vec<u32> = vec![0];
        let mut tokens: Vec<u32> = Vec::new();
        let mut dists_req_indptr: Vec<u32> = vec![0];
        let mut dists_kv_indptr: Vec<u32> = vec![0];
        let mut dists_ids: Vec<u32> = Vec::new();
        let mut dists_probs: Vec<f32> = Vec::new();
        let mut logits_req_indptr: Vec<u32> = vec![0];
        let mut logits_byte_indptr: Vec<u32> = vec![0];
        let mut logits_bytes: Vec<u8> = Vec::new();
        let mut logprobs_req_indptr: Vec<u32> = vec![0];
        let mut logprobs_val_indptr: Vec<u32> = vec![0];
        let mut logprobs_values: Vec<f32> = Vec::new();
        let mut entropies_indptr: Vec<u32> = vec![0];
        let mut entropies: Vec<f32> = Vec::new();

        // num_requests = sampler_indptr.len() - 1.
        let n: usize = fr.sampler_indptr.len().saturating_sub(1);

        for req_idx in 0..n {
            let s_lo: u32 = fr.sampler_indptr[req_idx].into();
            let s_hi: u32 = fr.sampler_indptr[req_idx + 1].into();

            let allowed = brle_for_request(fr, req_idx)
                .and_then(|brle| AllowedRuns::parse(&brle, self.vocab_size));

            for slot in s_lo as usize..s_hi as usize {
                let kind: u8 = fr.sampler_kinds[slot];
                // Logprobs labels live in the sampler_token_ids CSR; the slot's
                // run length is the count this slot emits.
                let lo: u32 = fr.sampler_token_ids_indptr[slot].into();
                let hi: u32 = fr.sampler_token_ids_indptr[slot + 1].into();
                let logprobs_k = (hi - lo) as usize;
                self.fill_slot(
                    kind,
                    logprobs_k,
                    allowed.as_ref(),
                    &mut tokens,
                    &mut dists_ids,
                    &mut dists_probs,
                    &mut dists_kv_indptr,
                    &mut logits_bytes,
                    &mut logits_byte_indptr,
                    &mut logprobs_values,
                    &mut logprobs_val_indptr,
                    &mut entropies,
                );
            }

            tokens_indptr.push(tokens.len() as u32);
            dists_req_indptr.push((dists_kv_indptr.len() - 1) as u32);
            logits_req_indptr.push((logits_byte_indptr.len() - 1) as u32);
            logprobs_req_indptr.push((logprobs_val_indptr.len() - 1) as u32);
            entropies_indptr.push(entropies.len() as u32);
        }

        let resp = ForwardResponse {
            num_requests: n as u32,
            tokens_indptr,
            tokens,
            dists_req_indptr,
            dists_kv_indptr,
            dists_ids,
            dists_probs,
            logits_req_indptr,
            logits_byte_indptr,
            logits_bytes,
            logprobs_req_indptr,
            logprobs_val_indptr,
            logprobs_values,
            entropies_indptr,
            entropies,
            ..Default::default()
        };
        let frame = ResponseFrame {
            driver_id,
            aborted: false,
            payload: ResponsePayload::Forward(resp),
        };
        encode_response(&frame).map_err(|e| anyhow::anyhow!("encode: {e}"))
    }

    #[allow(clippy::too_many_arguments)]
    fn fill_slot(
        &mut self,
        kind: u8,
        logprobs_k: usize,
        allowed: Option<&AllowedRuns>,
        tokens: &mut Vec<u32>,
        dists_ids: &mut Vec<u32>,
        dists_probs: &mut Vec<f32>,
        dists_kv_indptr: &mut Vec<u32>,
        logits_bytes: &mut Vec<u8>,
        logits_byte_indptr: &mut Vec<u32>,
        logprobs_values: &mut Vec<f32>,
        logprobs_val_indptr: &mut Vec<u32>,
        entropies: &mut Vec<f32>,
    ) {
        match kind {
            PIE_SAMPLER_DIST => {
                for _ in 0..8 {
                    dists_ids.push(self.random_token());
                }
                dists_probs.extend_from_slice(&[
                    0.5_f32, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.0078125,
                ]);
                dists_kv_indptr.push(dists_ids.len() as u32);
            }
            PIE_SAMPLER_RAW_LOGITS => {
                logits_bytes.extend_from_slice(&self.zero_logits);
                logits_byte_indptr.push(logits_bytes.len() as u32);
            }
            PIE_SAMPLER_LOGPROB => {
                logprobs_values.push(0.0);
                logprobs_val_indptr.push(logprobs_values.len() as u32);
            }
            PIE_SAMPLER_LOGPROBS => {
                for _ in 0..logprobs_k {
                    logprobs_values.push(0.0);
                }
                logprobs_val_indptr.push(logprobs_values.len() as u32);
            }
            PIE_SAMPLER_ENTROPY => {
                entropies.push(0.0);
            }
            PIE_SAMPLER_EMBEDDING => {
                // Reserved variant; runtime filters Embedding out of the slot
                // stream, so emitting nothing keeps the response aligned.
            }
            // Token-producing samplers (Multinomial/TopK/TopP/MinP/TopKTopP).
            _ => {
                let token = match allowed {
                    Some(a) => self.random_masked_token(a),
                    None => self.random_token(),
                };
                tokens.push(token);
            }
        }
    }

    fn random_token(&mut self) -> u32 {
        let mut buf = [0u8; 4];
        self.rng.fill_bytes(&mut buf);
        u32::from_ne_bytes(buf) % self.vocab_size
    }

    fn random_masked_token(&mut self, allowed: &AllowedRuns) -> u32 {
        if allowed.total == 0 {
            eprintln!("[pie-driver-dummy] logit_mask allows zero tokens; falling back");
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
        unreachable!()
    }
}

/// Pull the per-request BRLE constraint mask buffer out of the archived
/// `logit_masks: Vec<Brle>`. Returns `None` if the request has no logit
/// mask (i.e. the `logit_mask_indptr[r..r+1]` range is empty).
fn brle_for_request(fr: &ArchivedForwardRequest, req_idx: usize) -> Option<Vec<u32>> {
    if req_idx + 1 >= fr.logit_mask_indptr.len() {
        return None;
    }
    let lo: u32 = fr.logit_mask_indptr[req_idx].into();
    let hi: u32 = fr.logit_mask_indptr[req_idx + 1].into();
    if hi <= lo || (hi as usize) > fr.logit_masks.len() {
        return None;
    }
    // Each request contributes 0 or 1 Brle entries; take the first
    // present one and return its buffer as an owned Vec<u32>.
    let brle = &fr.logit_masks[lo as usize];
    Some(brle.buffer.iter().map(|x| u32::from(*x)).collect())
}

/// Allowed-token spans of a per-request BRLE constraint mask, clipped
/// to the configured vocab size.
struct AllowedRuns {
    runs: Vec<(u32, u32)>,
    total: u32,
}

impl AllowedRuns {
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
