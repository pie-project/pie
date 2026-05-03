//! Per-request work: decode the BPIQ payload, fabricate per-slot
//! outputs based on the sampler type, and emit a msgpack-mode BPIS
//! response. Token-producing slots (Argmax/TopP/TopK/MinP/TopKTopP/
//! Multinomial) get a random token; probe slots get a zero-shaped
//! placeholder (RawLogits → vocab×4 zeros, Logprob → `[0.0]`,
//! Logprobs → `[0.0; K]`, Distribution → empty pair, Entropy → `0.0`).
//!
//! Probe outputs are dimensionally correct but numerically meaningless;
//! the dummy is for plumbing tests, not feature validation.

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
        for &n_slots in decoded.request_num_samplers {
            let mut resp = ForwardPassResponse::default();
            for _ in 0..n_slots {
                if let Some(&ty) = decoded.sampler_types.get(slot_cursor) {
                    self.fill_slot(ty, slot_cursor, decoded.sampler_label_indptr, &mut resp);
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
        resp: &mut ForwardPassResponse,
    ) {
        match ty {
            sampler_type::ARGMAX
            | sampler_type::TOP_P
            | sampler_type::TOP_K
            | sampler_type::TOP_K_TOP_P
            | sampler_type::MIN_P
            | sampler_type::MULTINOMIAL => {
                resp.tokens.push(self.random_token());
            }
            sampler_type::DISTRIBUTION => {
                // Empty top-K — the runtime decoder accepts this; users
                // who want non-empty distributions need a real driver.
                resp.dists.push((Vec::new(), Vec::new()));
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

    const A_SAMPLER_TYPES: usize = 16;
    const A_REQUEST_NUM_SAMPLERS: usize = 18;
    const A_SAMPLER_LABEL_INDPTR: usize = 20;
    const FIXED_HEADER: usize = 32;

    fn align_up(n: usize, a: usize) -> usize { (n + a - 1) & !(a - 1) }

    fn build_request(nums: &[u32], types: &[u32], label_indptr: &[u32]) -> Vec<u8> {
        let mut bodies: Vec<(usize, &[u32])> = Vec::new();
        let mut cursor = REQ_HEADER_SIZE;
        for arr in [nums, types, label_indptr] {
            cursor = align_up(cursor, 8);
            bodies.push((cursor, arr));
            cursor += arr.len() * 4;
        }
        let mut buf = vec![0u8; cursor];
        buf[0..4].copy_from_slice(&REQ_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&REQ_SCHEMA_VERSION.to_le_bytes());
        buf[16..20].copy_from_slice(&(NUM_ARRAYS as u32).to_le_bytes());
        for (idx, (off, arr)) in [A_REQUEST_NUM_SAMPLERS, A_SAMPLER_TYPES, A_SAMPLER_LABEL_INDPTR]
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
                sampler_type::ARGMAX,
                sampler_type::ARGMAX,
                sampler_type::TOP_P,
                sampler_type::ARGMAX,
            ],
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
                sampler_type::ARGMAX,
                sampler_type::ENTROPY,
                sampler_type::LOGPROB,
            ],
            &[0, 0, 0, 1],
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
}
