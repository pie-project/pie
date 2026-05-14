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
    // Validate at the archived layer first so we get cheap zero-copy
    // error reporting on malformed buffers.
    let _ = parse_response(bytes).map_err(|e| anyhow!("parse_response: {e}"))?;
    // Then materialize an owned ResponseFrame so the caller can walk
    // plain `Vec<T>` fields. One extra alloc per response.
    let archived = parse_response(bytes).map_err(|e| anyhow!("{e}"))?;
    let owned = deserialize_response(archived).map_err(|e| anyhow!("deserialize_response: {e}"))?;
    Ok(DriverResponse {
        aborted: owned.aborted,
        payload: owned.payload,
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
