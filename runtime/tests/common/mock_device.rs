//! Mock device backend for integration tests.
//!
//! A real pie device exposes two planes, and the mock must serve both:
//!
//!   * **Control plane** — a named Mach `RpcServer`. The runtime's
//!     `device::spawn` does `RpcClient::connect(hostname)` at bootstrap, so the
//!     mock must answer on the hostname it advertises. Control calls are
//!     answered with empty payloads (the integration tests never drive a
//!     control method that needs a structured reply; bootstrap only needs the
//!     connection to be live).
//!   * **Forward-pass plane** — a POSIX shared-memory region. The runtime's
//!     `device::fire_batch` only speaks the shmem fast path (`shmem_ipc`), so
//!     the mock hosts the dummy driver's shmem server (`pie_driver_dummy_lib`)
//!     at the exact region name the client attaches to (`shmem::region_name`).
//!     The dummy `Handler` decodes each batched request and fabricates valid
//!     per-slot token outputs.
//!
//! This is a faithful device double, not a stub: forward passes traverse the
//! real shmem request/response wire format through the real decode/encode path.
//! Tests assert that forward passes complete (and how many tokens they yield),
//! not specific token values, so the dummy handler's fabricated tokens are
//! sufficient and honest.

use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use pie::device::RpcServer;
use pie::shmem::region_name;
use pie_driver_dummy_lib::handler::Handler;
use pie_driver_dummy_lib::shmem::{ShmemServer, METHOD_TAG_FIRE_BATCH};

/// Shmem region geometry. The runtime client reads slot/buffer sizes from the
/// region header, so only the *name* must agree; these mirror the dummy
/// driver's defaults and are generous for test batches.
const NUM_SLOTS: usize = 8;
const REQ_BUF: usize = 4 * 1024 * 1024;
const RESP_BUF: usize = 4 * 1024 * 1024;
/// Server poll interval — small enough to answer well within the client's
/// busy-spin window without pinning a core per device thread.
const SERVER_SPIN_US: u64 = 50;
/// Deterministic token source for the fabricated forward-pass outputs.
const HANDLER_SEED: u64 = 42;
const HANDLER_VOCAB: u32 = 32_000;

// =============================================================================
// Mock Backend
// =============================================================================

/// A mock device backend hosting, per device, a control RPC server plus a
/// shmem forward-pass server. Drop closes/stops both and joins their threads.
pub struct MockBackend {
    rpc_servers: Vec<Arc<RpcServer>>,
    shmem_servers: Vec<Arc<ShmemServer>>,
    handles: Vec<JoinHandle<()>>,
    server_names: Vec<String>,
}

impl MockBackend {
    /// Create a backend for `num_devices` devices.
    pub fn new(num_devices: usize) -> Self {
        let mut rpc_servers = Vec::with_capacity(num_devices);
        let mut shmem_servers = Vec::with_capacity(num_devices);
        let mut handles = Vec::with_capacity(num_devices * 2);
        let mut server_names = Vec::with_capacity(num_devices);

        for device_idx in 0..num_devices {
            // Control plane: Mach RPC server. The advertised name becomes the
            // device hostname the runtime connects to.
            let rpc = Arc::new(RpcServer::create().expect("Failed to create mock RpcServer"));
            let name = rpc.server_name().to_string();
            let rpc_clone = Arc::clone(&rpc);
            let rpc_handle = std::thread::Builder::new()
                .name(format!("mock-rpc-{device_idx}"))
                .spawn(move || control_loop(rpc_clone))
                .expect("Failed to spawn mock control thread");

            // Forward-pass plane: shmem server at the region the client attaches
            // to for this device index.
            let shmem = Arc::new(
                ShmemServer::create(
                    &region_name(device_idx),
                    NUM_SLOTS,
                    REQ_BUF,
                    RESP_BUF,
                    SERVER_SPIN_US,
                )
                .expect("Failed to create mock shmem device server"),
            );
            let shmem_clone = Arc::clone(&shmem);
            let shmem_handle = std::thread::Builder::new()
                .name(format!("mock-shmem-{device_idx}"))
                .spawn(move || {
                    let mut handler = Handler::new(HANDLER_SEED, HANDLER_VOCAB);
                    shmem_clone.serve_forever(|req, resp| {
                        if req.method_tag != METHOD_TAG_FIRE_BATCH {
                            return 0;
                        }
                        handler.handle_fire_batch(req.payload, resp)
                    });
                })
                .expect("Failed to spawn mock shmem thread");

            rpc_servers.push(rpc);
            shmem_servers.push(shmem);
            handles.push(rpc_handle);
            handles.push(shmem_handle);
            server_names.push(name);
        }

        Self {
            rpc_servers,
            shmem_servers,
            handles,
            server_names,
        }
    }

    /// Returns one control-plane RPC name per device, for `DeviceConfig.hostname`.
    pub fn server_names(&self) -> &[String] {
        &self.server_names
    }
}

/// Control-plane poll loop: answer every request with an empty payload until
/// the server closes. Forward passes never arrive here (they go via shmem).
fn control_loop(server: Arc<RpcServer>) {
    let poll_timeout = Duration::from_millis(100);
    loop {
        match server.poll(poll_timeout) {
            Ok(Some(request)) => {
                let _ = server.respond(request.request_id, Vec::new());
            }
            Ok(None) => {
                if server.is_closed() {
                    break;
                }
            }
            Err(_) => break,
        }
    }
}

impl Drop for MockBackend {
    fn drop(&mut self) {
        for server in &self.rpc_servers {
            server.close();
        }
        for server in &self.shmem_servers {
            server.stop();
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}
