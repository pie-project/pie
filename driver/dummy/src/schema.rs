//! BPIQ request decoder + BPIS response encoder.
//!
//! Layout mirrors `runtime/src/shmem_schema.rs`. The dummy decodes the
//! per-request slot count (`request_num_samplers`), per-slot type
//! (`sampler_types`), and the Logprob/Logprobs label-count CSR
//! (`sampler_label_indptr`). Sampler types are enumerated below — see
//! `runtime/src/inference/request.rs::Sampler::type_id`.

use anyhow::{Result, anyhow, ensure};
use serde::Serialize;

pub const REQ_MAGIC: u32 = 0x42504951; // 'BPIQ'
pub const REQ_SCHEMA_VERSION: u32 = 1;
pub const REQ_HEADER_SIZE: usize = 256;
pub const NUM_ARRAYS: usize = 28;

const FIXED_HEADER: usize = 32;

const A_SAMPLER_TYPES: usize = 16;
const A_REQUEST_NUM_SAMPLERS: usize = 18;
const A_SAMPLER_LABEL_INDPTR: usize = 20;

pub const RESP_MAGIC: u32 = 0x42504953; // 'BPIS'
pub const RESP_HEADER_SIZE: usize = 16;
pub const RESP_MODE_MSGPACK: u32 = 1;

/// Sampler / probe type ids — must match `Sampler::type_id` in
/// `runtime/src/inference/request.rs`. Types 0..=5 are token-producing
/// (one `tokens.push`); 6..=10 are probes with their own output shape.
pub mod sampler_type {
    pub const ARGMAX: u32 = 0;
    pub const TOP_P: u32 = 1;
    pub const TOP_K: u32 = 2;
    pub const TOP_K_TOP_P: u32 = 3;
    pub const MIN_P: u32 = 4;
    pub const MULTINOMIAL: u32 = 5;
    pub const DISTRIBUTION: u32 = 6;
    pub const RAW_LOGITS: u32 = 7;
    pub const LOGPROB: u32 = 8;
    pub const LOGPROBS: u32 = 9;
    pub const ENTROPY: u32 = 10;
}

/// Per-request and per-slot views into the BPIQ payload. All borrows
/// share the request buffer's lifetime.
#[derive(Debug)]
pub struct DecodedRequest<'a> {
    /// One u32 per request: total slot count (samplers + probes).
    pub request_num_samplers: &'a [u32],
    /// One u32 per slot, flat across the batch. Slot `s` for request
    /// `i` is at index `request_num_samplers[..i].iter().sum() + s`.
    pub sampler_types: &'a [u32],
    /// CSR of label counts for Logprob/Logprobs slots: per-slot K is
    /// `sampler_label_indptr[s+1] - sampler_label_indptr[s]`. Empty if
    /// the request has no Logprob-type slots.
    pub sampler_label_indptr: &'a [u32],
}

pub fn decode_request(buf: &[u8]) -> Result<DecodedRequest<'_>> {
    ensure!(
        buf.len() >= REQ_HEADER_SIZE,
        "request buffer too small: {} < {REQ_HEADER_SIZE}",
        buf.len()
    );

    let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    ensure!(
        magic == REQ_MAGIC,
        "request magic mismatch: 0x{magic:08x} != 0x{REQ_MAGIC:08x}"
    );
    let schema = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    ensure!(
        schema == REQ_SCHEMA_VERSION,
        "request schema mismatch: {schema} != {REQ_SCHEMA_VERSION}"
    );
    let num_arrays = u32::from_le_bytes(buf[16..20].try_into().unwrap()) as usize;
    ensure!(
        num_arrays == NUM_ARRAYS,
        "request num_arrays mismatch: {num_arrays} != {NUM_ARRAYS}"
    );

    Ok(DecodedRequest {
        request_num_samplers: read_u32_array(buf, A_REQUEST_NUM_SAMPLERS)?,
        sampler_types: read_u32_array(buf, A_SAMPLER_TYPES)?,
        sampler_label_indptr: read_u32_array(buf, A_SAMPLER_LABEL_INDPTR)?,
    })
}

fn read_u32_array(buf: &[u8], idx: usize) -> Result<&[u32]> {
    let entry_off = FIXED_HEADER + idx * 8;
    if entry_off + 8 > REQ_HEADER_SIZE {
        return Err(anyhow!("array {idx} table entry past header"));
    }
    let offset = u32::from_le_bytes(buf[entry_off..entry_off + 4].try_into().unwrap()) as usize;
    let len = u32::from_le_bytes(buf[entry_off + 4..entry_off + 8].try_into().unwrap()) as usize;
    let nbytes = len * 4;
    if offset + nbytes > buf.len() {
        return Err(anyhow!(
            "array {idx} body past buffer: offset={offset} len={len} buf={}",
            buf.len()
        ));
    }
    // The runtime writer aligns each array to 8 bytes (see
    // `runtime/src/shmem_schema.rs::write_request`), satisfying u32's
    // 4-byte alignment. `try_cast_slice` validates at runtime.
    bytemuck::try_cast_slice(&buf[offset..offset + nbytes])
        .map_err(|e| anyhow!("array {idx} cast: {e}"))
}

// =============================================================================
// Response — mirror of `runtime/src/inference/request::ForwardPassResponse`
// =============================================================================
//
// Field order, names, and serde defaults must match the runtime's struct
// exactly so rmp-serde decodes our msgpack on the host side. See
// `runtime/src/inference/request.rs::ForwardPassResponse`. Vec<Vec<u8>>
// for logits is intentionally not `serde_bytes` — runtime uses default
// (msgpack array) too; mismatching encodings break the round-trip.

#[derive(Debug, Clone, Default, Serialize)]
pub struct ForwardPassResponse {
    pub tokens: Vec<u32>,
    pub dists: Vec<(Vec<u32>, Vec<f32>)>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub logits: Vec<Vec<u8>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub logprobs: Vec<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entropies: Vec<f32>,
    pub spec_tokens: Vec<u32>,
    pub spec_positions: Vec<u32>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct BatchedForwardPassResponse {
    pub results: Vec<ForwardPassResponse>,
}

/// Encode a `BatchedForwardPassResponse` into the BPIS msgpack frame.
/// Returns bytes written. The runtime side accepts either flat-mode
/// (token-only) or msgpack-mode bodies; the dummy always uses msgpack
/// because probe slots can't be expressed in the flat shape.
pub fn encode_msgpack_response(buf: &mut [u8], resp: &BatchedForwardPassResponse) -> Result<usize> {
    let body = rmp_serde::to_vec_named(resp)
        .map_err(|e| anyhow!("encode response: msgpack: {e}"))?;
    let n_req = resp.results.len();
    let total_tokens: usize = resp.results.iter().map(|r| r.tokens.len()).sum();
    let needed = RESP_HEADER_SIZE + body.len();
    ensure!(
        needed <= buf.len(),
        "encode_response: response buffer too small: need {needed}, have {}",
        buf.len()
    );

    buf[0..4].copy_from_slice(&RESP_MAGIC.to_le_bytes());
    buf[4..8].copy_from_slice(&RESP_MODE_MSGPACK.to_le_bytes());
    buf[8..12].copy_from_slice(&(n_req as u32).to_le_bytes());
    buf[12..16].copy_from_slice(&(total_tokens as u32).to_le_bytes());
    buf[RESP_HEADER_SIZE..needed].copy_from_slice(&body);
    Ok(needed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn align_up(n: usize, a: usize) -> usize {
        (n + a - 1) & !(a - 1)
    }

    /// Build a BPIQ buffer with the request_num_samplers,
    /// sampler_types, and sampler_label_indptr arrays populated.
    fn build_request(
        nums: &[u32],
        types: &[u32],
        label_indptr: &[u32],
    ) -> Vec<u8> {
        // Layout each requested array sequentially after the header,
        // 8-byte aligned.
        let mut bodies: Vec<(usize, &[u32])> = Vec::new();
        let mut cursor = REQ_HEADER_SIZE;
        for arr in [nums, types, label_indptr] {
            cursor = align_up(cursor, 8);
            bodies.push((cursor, arr));
            cursor += arr.len() * 4;
        }
        let total = cursor;

        let mut buf = vec![0u8; total];
        buf[0..4].copy_from_slice(&REQ_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&REQ_SCHEMA_VERSION.to_le_bytes());
        buf[16..20].copy_from_slice(&(NUM_ARRAYS as u32).to_le_bytes());

        for (idx, (body_off, arr)) in
            [A_REQUEST_NUM_SAMPLERS, A_SAMPLER_TYPES, A_SAMPLER_LABEL_INDPTR]
                .iter()
                .zip(bodies.iter())
        {
            let entry = FIXED_HEADER + idx * 8;
            buf[entry..entry + 4].copy_from_slice(&(*body_off as u32).to_le_bytes());
            buf[entry + 4..entry + 8].copy_from_slice(&(arr.len() as u32).to_le_bytes());
            for (i, &v) in arr.iter().enumerate() {
                let off = body_off + i * 4;
                buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }

    #[test]
    fn decode_round_trips_all_arrays() {
        let nums = [2u32, 1u32];
        let types = [
            sampler_type::ARGMAX,
            sampler_type::RAW_LOGITS,
            sampler_type::LOGPROBS,
        ];
        let label_indptr = [0u32, 0, 0, 3];
        let buf = build_request(&nums, &types, &label_indptr);

        let d = decode_request(&buf).unwrap();
        assert_eq!(d.request_num_samplers, &nums);
        assert_eq!(d.sampler_types, &types);
        assert_eq!(d.sampler_label_indptr, &label_indptr);
    }

    #[test]
    fn encode_msgpack_round_trips() {
        let mut buf = vec![0u8; 1024];
        let resp = BatchedForwardPassResponse {
            results: vec![ForwardPassResponse {
                tokens: vec![1, 2, 3],
                ..Default::default()
            }],
        };
        let n = encode_msgpack_response(&mut buf, &resp).unwrap();
        assert!(n > RESP_HEADER_SIZE);
        assert_eq!(u32::from_le_bytes(buf[0..4].try_into().unwrap()), RESP_MAGIC);
        assert_eq!(u32::from_le_bytes(buf[4..8].try_into().unwrap()), RESP_MODE_MSGPACK);
        assert_eq!(u32::from_le_bytes(buf[8..12].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(buf[12..16].try_into().unwrap()), 3);
    }
}
