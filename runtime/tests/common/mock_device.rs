//! Mock driver backend for integration tests.
//!
//! Implements [`DriverChannel`] directly and pre-registers one channel
//! per device through [`register_driver`] + [`install_channel`]. The
//! unified driver surface means there's no RPC server, no mqueue, and
//! no shmem ring in the test path — just a typed channel that dispatches
//! `DriverRequest::Forward` to a [`Behavior`] and treats every cold
//! method as a no-op `Status(0)`.

use std::sync::atomic::AtomicU32;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use async_trait::async_trait;

use pie::driver::{
    DriverChannel, DriverRequest, DriverResponse, DriverSpec, install_channel, register_driver,
};
use pie::inference::request::{
    BatchedForwardPassRequest, BatchedForwardPassResponse, ForwardPassResponse,
};

// =============================================================================
// Behavior Trait
// =============================================================================

/// How the mock device handles a forward batch.
pub trait Behavior: Send + Sync + 'static {
    fn handle_fire_batch(&self, req: &BatchedForwardPassRequest) -> BatchedForwardPassResponse;
}

// =============================================================================
// Built-in Behaviors
// =============================================================================

/// Always returns the same token for every request.
pub struct EchoBehavior(pub u32);

impl Behavior for EchoBehavior {
    fn handle_fire_batch(&self, req: &BatchedForwardPassRequest) -> BatchedForwardPassResponse {
        let num = req.num_requests();
        BatchedForwardPassResponse {
            results: (0..num)
                .map(|_| ForwardPassResponse {
                    tokens: vec![self.0],
                    dists: vec![],
                    logits: vec![],
                    logprobs: vec![],
                    entropies: vec![],
                    spec_tokens: vec![],
                    spec_positions: vec![],
                })
                .collect(),
        }
    }
}

/// Returns sequential tokens starting from `start`.
pub struct CounterBehavior {
    next: AtomicU32,
}

impl CounterBehavior {
    pub fn new(start: u32) -> Self {
        Self {
            next: AtomicU32::new(start),
        }
    }
}

impl Behavior for CounterBehavior {
    fn handle_fire_batch(&self, req: &BatchedForwardPassRequest) -> BatchedForwardPassResponse {
        let num = req.num_requests();
        let mut results = Vec::with_capacity(num);
        for _ in 0..num {
            let t = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            results.push(ForwardPassResponse {
                tokens: vec![t],
                dists: vec![],
                logits: vec![],
                logprobs: vec![],
                entropies: vec![],
                spec_tokens: vec![],
                spec_positions: vec![],
            });
        }
        BatchedForwardPassResponse { results }
    }
}

// =============================================================================
// CallRecorder
// =============================================================================

#[derive(Debug, Clone)]
pub struct RecordedCall {
    pub device_idx: usize,
    pub method: &'static str,
    pub num_requests: usize,
    pub total_tokens: usize,
    pub timestamp: Instant,
}

#[derive(Debug, Default)]
pub struct CallRecorder {
    calls: Mutex<Vec<RecordedCall>>,
}

impl CallRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    fn record(&self, call: RecordedCall) {
        self.calls.lock().unwrap().push(call);
    }

    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }

    pub fn calls(&self) -> Vec<RecordedCall> {
        self.calls.lock().unwrap().clone()
    }

    pub fn wait_for_calls(&self, n: usize, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            if self.call_count() >= n {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

// =============================================================================
// MockChannel — DriverChannel impl per device.
// =============================================================================

struct MockChannel {
    device_idx: usize,
    behavior: Arc<dyn Behavior>,
    recorder: Arc<CallRecorder>,
    aborted: Mutex<bool>,
}

#[async_trait]
impl DriverChannel for MockChannel {
    async fn submit(&self, req: DriverRequest) -> Result<DriverResponse> {
        if *self.aborted.lock().unwrap() {
            anyhow::bail!("mock channel {}: aborted", self.device_idx);
        }
        match req {
            DriverRequest::Forward(batch) => {
                self.recorder.record(RecordedCall {
                    device_idx: self.device_idx,
                    method: "fire_batch",
                    num_requests: batch.num_requests(),
                    total_tokens: batch.total_tokens(),
                    timestamp: Instant::now(),
                });
                let resp = self.behavior.handle_fire_batch(&batch);
                Ok(DriverResponse::Forward(resp))
            }
            DriverRequest::CopyD2H { .. }
            | DriverRequest::CopyH2D { .. }
            | DriverRequest::CopyD2D { .. }
            | DriverRequest::CopyH2H { .. }
            | DriverRequest::LoadAdapter { .. }
            | DriverRequest::SaveAdapter { .. }
            | DriverRequest::ZoInitializeAdapter { .. }
            | DriverRequest::ZoUpdateAdapter { .. } => Ok(DriverResponse::Status(0)),
        }
    }

    fn notify(&self, _req: DriverRequest) -> Result<()> {
        // Cold ops are no-ops in the mock — they always "succeed".
        Ok(())
    }

    fn abort(&self) {
        *self.aborted.lock().unwrap() = true;
    }
}

// =============================================================================
// MockBackend — owns the per-device channels.
// =============================================================================

pub struct MockBackend {
    driver_ids: Vec<usize>,
    recorder: Arc<CallRecorder>,
}

impl MockBackend {
    /// Pre-register `num_devices` mock channels with the runtime. Each
    /// channel gets a fresh [`DriverId`] from the runtime's allocator;
    /// callers thread those IDs through their bootstrap config so the
    /// scheduler dispatches forward batches into the mocks.
    pub fn new(num_devices: usize, behavior: Arc<dyn Behavior>) -> Self {
        let recorder = Arc::new(CallRecorder::new());
        let mut driver_ids = Vec::with_capacity(num_devices);
        for device_idx in 0..num_devices {
            let channel = Arc::new(MockChannel {
                device_idx,
                behavior: Arc::clone(&behavior),
                recorder: Arc::clone(&recorder),
                aborted: Mutex::new(false),
            });
            let id = register_driver(DriverSpec {
                num_kv_pages: 64,
                max_batch_size: 32,
                max_batch_tokens: 4096,
            });
            install_channel(id, channel);
            driver_ids.push(id);
        }
        Self {
            driver_ids,
            recorder,
        }
    }

    /// Driver IDs allocated by [`Self::new`], one per device. Hand to
    /// the bootstrap config builder.
    pub fn driver_ids(&self) -> &[usize] {
        &self.driver_ids
    }

    pub fn recorder(&self) -> &CallRecorder {
        &self.recorder
    }
}
