//! Mock device backend for integration tests.
//!
//! Provides a trait-based mock that speaks the real IPC protocol
//! via `RpcServer`, allowing integration tests to exercise the full
//! RPC stack without a Python backend.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use pie::device::RpcServer;
use pie::inference::request::{
    BatchedForwardPassRequest, BatchedForwardPassResponse, ForwardPassResponse,
};

// =============================================================================
// Behavior Trait
// =============================================================================

/// Trait for defining mock device behavior.
///
/// Implementors define how the mock device responds to `fire_batch` RPC calls.
pub trait Behavior: Send + Sync + 'static {
    /// Handle a batched forward pass request and return a response.
    fn handle_fire_batch(&self, req: &BatchedForwardPassRequest) -> BatchedForwardPassResponse;
}

// =============================================================================
// Built-in Behaviors
// =============================================================================

/// Always returns the same token for every request in the batch.
pub struct EchoBehavior(pub u32);

impl Behavior for EchoBehavior {
    fn handle_fire_batch(&self, req: &BatchedForwardPassRequest) -> BatchedForwardPassResponse {
        let num_requests = req.num_requests();
        BatchedForwardPassResponse {
            results: (0..num_requests)
                .map(|_| ForwardPassResponse {
                    tokens: vec![self.0],
                    dists: vec![],
                })
                .collect(),
        }
    }
}

/// Returns sequential tokens starting from the given value.
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
        let num_requests = req.num_requests();
        BatchedForwardPassResponse {
            results: (0..num_requests)
                .map(|_| {
                    let token = self.next.fetch_add(1, Ordering::Relaxed);
                    ForwardPassResponse {
                        tokens: vec![token],
                        dists: vec![],
                    }
                })
                .collect(),
        }
    }
}

/// Wraps another behavior and adds simulated latency before responding.
pub struct DelayedBehavior<B: Behavior> {
    pub inner: B,
    pub latency: Duration,
}

impl<B: Behavior> Behavior for DelayedBehavior<B> {
    fn handle_fire_batch(&self, req: &BatchedForwardPassRequest) -> BatchedForwardPassResponse {
        std::thread::sleep(self.latency);
        self.inner.handle_fire_batch(req)
    }
}

/// Wraps another behavior and fails after N successful calls.
pub struct FailAfterBehavior<B: Behavior> {
    pub inner: B,
    remaining: AtomicU32,
}

impl<B: Behavior> FailAfterBehavior<B> {
    pub fn new(inner: B, success_count: u32) -> Self {
        Self {
            inner,
            remaining: AtomicU32::new(success_count),
        }
    }
}

impl<B: Behavior> Behavior for FailAfterBehavior<B> {
    fn handle_fire_batch(&self, req: &BatchedForwardPassRequest) -> BatchedForwardPassResponse {
        if self.remaining.fetch_sub(1, Ordering::Relaxed) == 0 {
            // Return empty results to simulate failure
            BatchedForwardPassResponse { results: vec![] }
        } else {
            self.inner.handle_fire_batch(req)
        }
    }
}

// =============================================================================
// Call Recorder
// =============================================================================

/// A recorded RPC call for test assertions.
#[derive(Debug, Clone)]
pub struct RecordedCall {
    pub device_idx: usize,
    pub method: String,
    pub num_requests: usize,
    pub total_tokens: usize,
    pub timestamp: Instant,
}

/// Records all RPC calls made to mock devices for later assertion.
pub struct CallRecorder {
    calls: Mutex<Vec<RecordedCall>>,
}

impl CallRecorder {
    fn new() -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
        }
    }

    fn record(&self, call: RecordedCall) {
        self.calls.lock().unwrap().push(call);
    }

    /// Returns the total number of recorded calls.
    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }

    /// Returns a snapshot of all recorded calls.
    pub fn calls(&self) -> Vec<RecordedCall> {
        self.calls.lock().unwrap().clone()
    }

    /// Blocks until at least `n` calls have been recorded, or until `timeout`.
    /// Returns `true` if the condition was met, `false` on timeout.
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
// Mock Backend
// =============================================================================

/// A mock device backend that runs `RpcServer` instances in background threads.
///
/// Each device gets its own `RpcServer` and poll/respond thread.
/// Drop closes all servers and joins threads.
pub struct MockBackend {
    servers: Vec<Arc<RpcServer>>,
    handles: Vec<JoinHandle<()>>,
    server_names: Vec<String>,
    recorder: Arc<CallRecorder>,
}

impl MockBackend {
    /// Create a new mock backend with `num_devices` devices, all using the same behavior.
    pub fn new(num_devices: usize, behavior: Arc<dyn Behavior>) -> Self {
        let recorder = Arc::new(CallRecorder::new());
        let mut servers = Vec::with_capacity(num_devices);
        let mut handles = Vec::with_capacity(num_devices);
        let mut server_names = Vec::with_capacity(num_devices);

        for device_idx in 0..num_devices {
            let server = Arc::new(
                RpcServer::create().expect("Failed to create mock RpcServer"),
            );
            let name = server.server_name().to_string();

            let server_clone = Arc::clone(&server);
            let behavior_clone = Arc::clone(&behavior);
            let recorder_clone = Arc::clone(&recorder);

            let handle = std::thread::Builder::new()
                .name(format!("mock-device-{device_idx}"))
                .spawn(move || {
                    Self::poll_loop(device_idx, server_clone, behavior_clone, recorder_clone);
                })
                .expect("Failed to spawn mock device thread");

            servers.push(server);
            handles.push(handle);
            server_names.push(name);
        }

        Self {
            servers,
            handles,
            server_names,
            recorder,
        }
    }

    /// Returns the IPC server names, one per device.
    /// Use these as `DeviceConfig.hostname`.
    pub fn server_names(&self) -> &[String] {
        &self.server_names
    }

    /// Access the shared call recorder for assertions.
    pub fn recorder(&self) -> &CallRecorder {
        &self.recorder
    }

    fn poll_loop(
        device_idx: usize,
        server: Arc<RpcServer>,
        behavior: Arc<dyn Behavior>,
        recorder: Arc<CallRecorder>,
    ) {
        let poll_timeout = Duration::from_millis(100);

        loop {
            match server.poll(poll_timeout) {
                Ok(Some(request)) => {
                    let method = request.method.clone();

                    let response_payload = if method == "fire_batch" {
                        // Deserialize the batched request
                        let batch_req: BatchedForwardPassRequest =
                            rmp_serde::from_slice(&request.payload)
                                .expect("Failed to deserialize BatchedForwardPassRequest");

                        // Record the call
                        recorder.record(RecordedCall {
                            device_idx,
                            method: method.clone(),
                            num_requests: batch_req.num_requests(),
                            total_tokens: batch_req.total_tokens(),
                            timestamp: Instant::now(),
                        });

                        // Dispatch to behavior
                        let response = behavior.handle_fire_batch(&batch_req);
                        rmp_serde::to_vec(&response)
                            .expect("Failed to serialize BatchedForwardPassResponse")
                    } else {
                        // Record unknown methods too
                        recorder.record(RecordedCall {
                            device_idx,
                            method: method.clone(),
                            num_requests: 0,
                            total_tokens: 0,
                            timestamp: Instant::now(),
                        });
                        // Return empty response for unknown methods
                        vec![]
                    };

                    if let Err(e) = server.respond(request.request_id, response_payload) {
                        tracing::warn!(
                            "Mock device {device_idx}: failed to send response: {e}"
                        );
                    }
                }
                Ok(None) => {
                    // Timeout, check if closed
                    if server.is_closed() {
                        break;
                    }
                }
                Err(_) => {
                    // Server closed or channel error
                    break;
                }
            }
        }
    }
}

impl Drop for MockBackend {
    fn drop(&mut self) {
        // Close all servers first
        for server in &self.servers {
            server.close();
        }
        // Then join all threads
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}
