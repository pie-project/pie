//! `DriverChannel` over POSIX shared memory.
//!
//! Transport mechanics (mmap, ring, busy-spin polling, RAII Lease,
//! schema_hash handshake) live in `pie_bridge::ipc::ShmemClient`. This
//! module wraps each `DriverRequest` in a `pie_bridge::Frame`, encodes
//! and ships the bytes, then unwraps the `pie_bridge::ResponseFrame`
//! on receive — no pie-internal mirror types.

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, anyhow};
use async_trait::async_trait;

use pie_bridge::ipc::ShmemClient;
use pie_bridge::wire::{deserialize_response, encode_request, parse_response};
use pie_bridge::{Frame, SCHEMA_HASH};

use super::{DriverChannel, DriverRequest, DriverResponse};

pub struct ShmemChannel {
    client: ShmemClient,
    next_req_id: AtomicU64,
}

impl ShmemChannel {
    /// Open the channel. `spin_budget_us` is the busy-spin window
    /// before parking on the slot's `resp_wake` atomic via the
    /// cross-process kernel primitive — see [`ShmemClient::open`].
    pub fn open(name: &str, spin_budget_us: u64) -> Result<Self> {
        Ok(Self {
            client: ShmemClient::open(name, spin_budget_us, SCHEMA_HASH)?,
            next_req_id: AtomicU64::new(1),
        })
    }

    /// One blocking roundtrip. Used by both `submit` (off the tokio
    /// worker via `block_in_place`) and `notify` (inline on the caller's
    /// thread).
    fn roundtrip_sync(&self, req: DriverRequest) -> Result<DriverResponse> {
        let req_id = self.next_req_id.fetch_add(1, Ordering::Relaxed) as u32;
        let frame = Frame {
            driver_id: req.driver_id as u32,
            payload: req.payload,
        };
        let bytes = encode_request(&frame).map_err(|e| anyhow!("encode: {e}"))?;
        let resp_bytes = self.client.roundtrip(req_id, &bytes)?;
        decode_driver_response(&resp_bytes)
    }
}

fn decode_driver_response(bytes: &[u8]) -> Result<DriverResponse> {
    if let Some(resp) = decode_fast_token_response(bytes) {
        return resp;
    }

    // Validate at the archived layer first so we get cheap zero-copy
    // error reporting on malformed buffers.
    let archived = parse_response(bytes).map_err(|e| anyhow!("parse_response: {e}"))?;
    // Then materialize an owned ResponseFrame so the caller can walk
    // plain `Vec<T>` fields. One extra alloc per response.
    let owned = deserialize_response(archived).map_err(|e| anyhow!("deserialize_response: {e}"))?;
    Ok(DriverResponse {
        aborted: owned.aborted,
        payload: owned.payload,
    })
}

const FAST_TOKEN_RESPONSE_MAGIC: &[u8; 8] = b"PIETFWD1";
const FAST_TOKEN_RESPONSE_HEADER_LEN: usize = 20;

fn decode_fast_token_response(bytes: &[u8]) -> Option<Result<DriverResponse>> {
    if bytes.len() < FAST_TOKEN_RESPONSE_MAGIC.len()
        || &bytes[..FAST_TOKEN_RESPONSE_MAGIC.len()] != FAST_TOKEN_RESPONSE_MAGIC
    {
        return None;
    }
    Some(decode_fast_token_response_inner(bytes))
}

fn read_le_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let raw = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| anyhow!("fast token response truncated at offset {offset}"))?;
    Ok(u32::from_le_bytes(
        raw.try_into().expect("slice length checked"),
    ))
}

fn decode_fast_token_response_inner(bytes: &[u8]) -> Result<DriverResponse> {
    if bytes.len() < FAST_TOKEN_RESPONSE_HEADER_LEN {
        return Err(anyhow!(
            "fast token response truncated: {} < {}",
            bytes.len(),
            FAST_TOKEN_RESPONSE_HEADER_LEN
        ));
    }
    let _driver_id = read_le_u32(bytes, 8)?;
    let num_requests = read_le_u32(bytes, 12)?;
    let token_count = read_le_u32(bytes, 16)? as usize;
    if token_count != num_requests as usize {
        return Err(anyhow!(
            "fast token response expected one token per request, got token_count={} num_requests={}",
            token_count,
            num_requests
        ));
    }
    let expected_len = FAST_TOKEN_RESPONSE_HEADER_LEN + token_count * std::mem::size_of::<u32>();
    if bytes.len() != expected_len {
        return Err(anyhow!(
            "fast token response length mismatch: got {}, want {}",
            bytes.len(),
            expected_len
        ));
    }

    let mut tokens = Vec::with_capacity(token_count);
    for chunk in bytes[FAST_TOKEN_RESPONSE_HEADER_LEN..].chunks_exact(4) {
        tokens.push(u32::from_le_bytes(
            chunk.try_into().expect("chunks_exact(4)"),
        ));
    }

    let payload = pie_bridge::ResponsePayload::Forward(pie_bridge::ForwardResponse {
        num_requests,
        tokens_indptr: (0..=num_requests).collect(),
        tokens,
        ..Default::default()
    });
    Ok(DriverResponse {
        aborted: false,
        payload,
    })
}

#[async_trait]
impl DriverChannel for ShmemChannel {
    async fn submit(&self, req: DriverRequest) -> Result<DriverResponse> {
        // busy-spin in `ShmemClient::roundtrip` — offload to blocking pool.
        tokio::task::block_in_place(|| self.roundtrip_sync(req))
    }

    fn notify(&self, req: DriverRequest) -> Result<()> {
        let _ = self.roundtrip_sync(req)?;
        Ok(())
    }

    fn abort(&self) {
        self.client.abort();
    }
}
