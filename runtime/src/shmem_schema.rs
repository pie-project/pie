//! Zero-copy flat schema for `BatchedForwardPassRequest` over shmem.
//!
//! ## Layout
//!
//! ```text
//! [512-byte header]
//!   0:  u32 magic = 0x42504951 ("BPIQ")
//!   4:  u32 schema_version = 2
//!   8:  u32 device_id
//!   12: u32 flags  (bit 0 = single_token_mode)
//!   16: u32 num_arrays = 29
//!   20: u32 reserved
//!   24: u64 reserved
//!   32: 29 × (u32 offset, u32 len_in_elements) = 232 bytes
//!   264..512: reserved (room for future array slots without resizing)
//!   512: array data (concatenated, each array 8-byte aligned)
//! ```
//!
//! Byte order: little-endian, matches host. We assume both processes are
//! on x86_64 LE.
//!
//! v2 added `A_PREDICT_FLAGS` (one u8 per request) for pass-level
//! speculative execution and bumped HEADER_SIZE 256 → 512 so the
//! offset/len table has room (32 + 29*8 = 264 > 256). See
//! SPECULATIVE_EXECUTION_DESIGN.md.

use crate::inference::request::BatchedForwardPassRequest;

pub const MAGIC: u32 = 0x42504951; // 'BPIQ'
pub const SCHEMA_VERSION: u32 = 2;
pub const HEADER_SIZE: usize = 512;
pub const NUM_ARRAYS: usize = 29;

const FIXED_HEADER: usize = 32;

// Array indices (must match Python side).
pub const A_TOKEN_IDS: usize = 0;
pub const A_POSITION_IDS: usize = 1;
pub const A_KV_PAGE_INDICES: usize = 2;
pub const A_KV_PAGE_INDPTR: usize = 3;
pub const A_KV_LAST_PAGE_LENS: usize = 4;
pub const A_QO_INDPTR: usize = 5;
pub const A_FLATTENED_MASKS: usize = 6;
pub const A_MASK_INDPTR: usize = 7;
pub const A_LOGIT_MASKS: usize = 8;
pub const A_LOGIT_MASK_INDPTR: usize = 9;
pub const A_SAMPLING_INDICES: usize = 10;
pub const A_SAMPLING_INDPTR: usize = 11;
pub const A_SAMPLER_TEMPERATURES: usize = 12; // f32
pub const A_SAMPLER_TOP_K: usize = 13;
pub const A_SAMPLER_TOP_P: usize = 14; // f32
pub const A_SAMPLER_MIN_P: usize = 15; // f32
pub const A_SAMPLER_TYPES: usize = 16;
pub const A_SAMPLER_SEEDS: usize = 17;
pub const A_REQUEST_NUM_SAMPLERS: usize = 18;
pub const A_SAMPLER_LABEL_IDS: usize = 19;
pub const A_SAMPLER_LABEL_INDPTR: usize = 20;
pub const A_ADAPTER_INDICES: usize = 21; // i64, -1 = None
pub const A_ADAPTER_SEEDS: usize = 22; // i64, i64::MIN = None
pub const A_SPEC_TOKEN_IDS: usize = 23;
pub const A_SPEC_POSITION_IDS: usize = 24;
pub const A_SPEC_INDPTR: usize = 25;
pub const A_OUTPUT_SPEC_FLAGS: usize = 26; // u8
pub const A_CONTEXT_IDS: usize = 27; // u64
pub const A_PREDICT_FLAGS: usize = 28; // u8 (v2)

const ELEM_SIZE: [usize; NUM_ARRAYS] = [
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 4, 4, 4, 1, 8, 1,
];

#[inline]
fn align_up(n: usize, a: usize) -> usize {
    (n + a - 1) & !(a - 1)
}

/// Serialize a `BatchedForwardPassRequest` into `buf`. Returns the number
/// of bytes written. Returns `Err` if the buffer is too small.
pub fn write_request(req: &BatchedForwardPassRequest, buf: &mut [u8]) -> anyhow::Result<usize> {
    if buf.len() < HEADER_SIZE {
        anyhow::bail!("shmem buffer {} < header size {}", buf.len(), HEADER_SIZE);
    }

    // Compute lengths per array.
    let lens: [usize; NUM_ARRAYS] = [
        req.token_ids.0.len(),
        req.position_ids.0.len(),
        req.kv_page_indices.0.len(),
        req.kv_page_indptr.0.len(),
        req.kv_last_page_lens.0.len(),
        req.qo_indptr.0.len(),
        req.flattened_masks.0.len(),
        req.mask_indptr.0.len(),
        req.logit_masks.0.len(),
        req.logit_mask_indptr.0.len(),
        req.sampling_indices.0.len(),
        req.sampling_indptr.0.len(),
        req.sampler_temperatures.0.len(),
        req.sampler_top_k.0.len(),
        req.sampler_top_p.0.len(),
        req.sampler_min_p.0.len(),
        req.sampler_types.0.len(),
        req.sampler_seeds.0.len(),
        req.request_num_samplers.0.len(),
        req.sampler_label_ids.0.len(),
        req.sampler_label_indptr.0.len(),
        req.adapter_indices.len(),
        req.adapter_seeds.len(),
        req.spec_token_ids.0.len(),
        req.spec_position_ids.0.len(),
        req.spec_indptr.0.len(),
        req.output_spec_flags.len(),
        req.context_ids.len(),
        req.predict_flags.len(),
    ];

    // Compute offsets.
    let mut offsets = [0u32; NUM_ARRAYS];
    let mut cursor = HEADER_SIZE;
    for i in 0..NUM_ARRAYS {
        cursor = align_up(cursor, 8);
        if cursor > u32::MAX as usize {
            anyhow::bail!("array offset overflow");
        }
        offsets[i] = cursor as u32;
        cursor += lens[i] * ELEM_SIZE[i];
    }
    let total = cursor;
    if total > buf.len() {
        anyhow::bail!("shmem buffer {} < required {}", buf.len(), total);
    }

    // Write header.
    let h = &mut buf[..HEADER_SIZE];
    h[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    h[4..8].copy_from_slice(&SCHEMA_VERSION.to_le_bytes());
    h[8..12].copy_from_slice(&(req.device_id as u32).to_le_bytes());
    let flags: u32 = if req.single_token_mode { 1 } else { 0 };
    h[12..16].copy_from_slice(&flags.to_le_bytes());
    h[16..20].copy_from_slice(&(NUM_ARRAYS as u32).to_le_bytes());
    h[20..24].copy_from_slice(&0u32.to_le_bytes()); // reserved
    h[24..32].copy_from_slice(&0u64.to_le_bytes()); // reserved

    // Write the offset/len table.
    for i in 0..NUM_ARRAYS {
        let base = FIXED_HEADER + i * 8;
        h[base..base + 4].copy_from_slice(&offsets[i].to_le_bytes());
        h[base + 4..base + 8].copy_from_slice(&(lens[i] as u32).to_le_bytes());
    }

    // Write array bodies. Use bytemuck to splat slices.
    // Fixed mapping arms: each closure writes the appropriate type.
    macro_rules! write_u32 {
        ($idx:expr, $slice:expr) => {{
            let off = offsets[$idx] as usize;
            let nbytes = lens[$idx] * 4;
            let bytes: &[u8] = bytemuck::cast_slice($slice);
            buf[off..off + nbytes].copy_from_slice(bytes);
        }};
    }
    macro_rules! write_f32 {
        ($idx:expr, $slice:expr) => {{
            let off = offsets[$idx] as usize;
            let nbytes = lens[$idx] * 4;
            let bytes: &[u8] = bytemuck::cast_slice($slice);
            buf[off..off + nbytes].copy_from_slice(bytes);
        }};
    }

    write_u32!(A_TOKEN_IDS, &req.token_ids.0);
    write_u32!(A_POSITION_IDS, &req.position_ids.0);
    write_u32!(A_KV_PAGE_INDICES, &req.kv_page_indices.0);
    write_u32!(A_KV_PAGE_INDPTR, &req.kv_page_indptr.0);
    write_u32!(A_KV_LAST_PAGE_LENS, &req.kv_last_page_lens.0);
    write_u32!(A_QO_INDPTR, &req.qo_indptr.0);
    write_u32!(A_FLATTENED_MASKS, &req.flattened_masks.0);
    write_u32!(A_MASK_INDPTR, &req.mask_indptr.0);
    write_u32!(A_LOGIT_MASKS, &req.logit_masks.0);
    write_u32!(A_LOGIT_MASK_INDPTR, &req.logit_mask_indptr.0);
    write_u32!(A_SAMPLING_INDICES, &req.sampling_indices.0);
    write_u32!(A_SAMPLING_INDPTR, &req.sampling_indptr.0);
    write_f32!(A_SAMPLER_TEMPERATURES, &req.sampler_temperatures.0);
    write_u32!(A_SAMPLER_TOP_K, &req.sampler_top_k.0);
    write_f32!(A_SAMPLER_TOP_P, &req.sampler_top_p.0);
    write_f32!(A_SAMPLER_MIN_P, &req.sampler_min_p.0);
    write_u32!(A_SAMPLER_TYPES, &req.sampler_types.0);
    write_u32!(A_SAMPLER_SEEDS, &req.sampler_seeds.0);
    write_u32!(A_REQUEST_NUM_SAMPLERS, &req.request_num_samplers.0);
    write_u32!(A_SAMPLER_LABEL_IDS, &req.sampler_label_ids.0);
    write_u32!(A_SAMPLER_LABEL_INDPTR, &req.sampler_label_indptr.0);

    // Adapter indices (i64, -1 = None). AdapterId is u64; we serialize as
    // i64 with -1 sentinel — preserves the valid range used in practice.
    {
        let off = offsets[A_ADAPTER_INDICES] as usize;
        for (k, opt) in req.adapter_indices.iter().enumerate() {
            let v: i64 = match opt {
                Some(id) => *id as i64,
                None => -1,
            };
            buf[off + k * 8..off + (k + 1) * 8].copy_from_slice(&v.to_le_bytes());
        }
    }
    // Adapter seeds (i64, i64::MIN = None).
    {
        let off = offsets[A_ADAPTER_SEEDS] as usize;
        for (k, opt) in req.adapter_seeds.iter().enumerate() {
            let v: i64 = opt.unwrap_or(i64::MIN);
            buf[off + k * 8..off + (k + 1) * 8].copy_from_slice(&v.to_le_bytes());
        }
    }

    write_u32!(A_SPEC_TOKEN_IDS, &req.spec_token_ids.0);
    write_u32!(A_SPEC_POSITION_IDS, &req.spec_position_ids.0);
    write_u32!(A_SPEC_INDPTR, &req.spec_indptr.0);

    // output_spec_flags as u8.
    {
        let off = offsets[A_OUTPUT_SPEC_FLAGS] as usize;
        for (k, b) in req.output_spec_flags.iter().enumerate() {
            buf[off + k] = if *b { 1 } else { 0 };
        }
    }

    // context_ids as u64.
    {
        let off = offsets[A_CONTEXT_IDS] as usize;
        let bytes: &[u8] = bytemuck::cast_slice(&req.context_ids);
        let nbytes = req.context_ids.len() * 8;
        buf[off..off + nbytes].copy_from_slice(bytes);
    }

    // predict_flags as u8 (v2).
    {
        let off = offsets[A_PREDICT_FLAGS] as usize;
        for (k, b) in req.predict_flags.iter().enumerate() {
            buf[off + k] = if *b { 1 } else { 0 };
        }
    }

    Ok(total)
}


// =============================================================================
// Response schema
// =============================================================================

pub const RESP_MAGIC: u32 = 0x42504953; // 'BPIS'
pub const RESP_HEADER_SIZE: usize = 16;
pub const RESP_MODE_FLAT: u32 = 0;
pub const RESP_MODE_MSGPACK: u32 = 1;

use crate::inference::request::{BatchedForwardPassResponse, ForwardPassResponse};

/// Parse a `BatchedForwardPassResponse` from a shmem buffer.
///
/// Recognizes either the flat token-only schema (fast path) or a
/// msgpack-encoded fallback. Returns the parsed struct.
pub fn read_response(buf: &[u8]) -> anyhow::Result<BatchedForwardPassResponse> {
    if buf.len() < RESP_HEADER_SIZE {
        anyhow::bail!("response buffer too small ({} < {})", buf.len(), RESP_HEADER_SIZE);
    }
    let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    if magic != RESP_MAGIC {
        anyhow::bail!(
            "response magic mismatch: 0x{:08x} != 0x{:08x}",
            magic, RESP_MAGIC
        );
    }
    let mode = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    let n_req = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
    let total_tokens = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;

    if mode == RESP_MODE_MSGPACK {
        // Fallback path — body is msgpack-encoded.
        let body = &buf[RESP_HEADER_SIZE..];
        let resp: BatchedForwardPassResponse = rmp_serde::from_slice(body)
            .map_err(|e| anyhow::anyhow!("failed to deserialize msgpack response: {e}"))?;
        return Ok(resp);
    }
    if mode != RESP_MODE_FLAT {
        anyhow::bail!("unknown response mode {}", mode);
    }

    // Flat path: counts table, then tokens table.
    let counts_off = RESP_HEADER_SIZE;
    let tokens_off = counts_off + n_req * 4;
    let needed = tokens_off + total_tokens * 4;
    if buf.len() < needed {
        anyhow::bail!("response body truncated: have {}, need {}", buf.len(), needed);
    }

    let counts: &[u32] = bytemuck::cast_slice(&buf[counts_off..counts_off + n_req * 4]);
    let tokens_all: &[u32] = bytemuck::cast_slice(&buf[tokens_off..tokens_off + total_tokens * 4]);

    let mut results: Vec<ForwardPassResponse> = Vec::with_capacity(n_req);
    let mut cursor = 0usize;
    for i in 0..n_req {
        let n = counts[i] as usize;
        let toks = tokens_all[cursor..cursor + n].to_vec();
        cursor += n;
        results.push(ForwardPassResponse {
            tokens: toks,
            dists: Vec::new(),
            logits: Vec::new(),
            logprobs: Vec::new(),
            entropies: Vec::new(),
            spec_tokens: Vec::new(),
            spec_positions: Vec::new(),
        });
    }

    Ok(BatchedForwardPassResponse { results })
}
