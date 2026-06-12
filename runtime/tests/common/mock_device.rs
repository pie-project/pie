//! Mock device backend for integration tests.
//!
//! Provides a trait-based mock that speaks the real device protocols:
//! the control-plane `RpcServer` plus the current shared-memory `fire_batch`
//! fast path. This lets integration tests exercise the runtime stack without
//! a Python or native driver backend.

use std::ffi::CString;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use pie::device::RpcServer;
use pie::inference::request::{
    BatchedForwardPassRequest, BatchedForwardPassResponse, ByteVec, ByteVecF32, ForwardPassResponse,
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

fn token_response(token: u32) -> ForwardPassResponse {
    ForwardPassResponse {
        tokens: vec![token],
        dists: vec![],
        logits: vec![],
        logprobs: vec![],
        entropies: vec![],
        spec_tokens: vec![],
        spec_positions: vec![],
    }
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
            results: (0..num_requests).map(|_| token_response(self.0)).collect(),
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
                    token_response(token)
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
    shmem_servers: Vec<Arc<ShmemServer>>,
    handles: Vec<JoinHandle<()>>,
    server_names: Vec<String>,
    recorder: Arc<CallRecorder>,
}

impl MockBackend {
    /// Create a new mock backend with `num_devices` devices, all using the same behavior.
    pub fn new(num_devices: usize, behavior: Arc<dyn Behavior>) -> Self {
        let recorder = Arc::new(CallRecorder::new());
        let mut servers = Vec::with_capacity(num_devices);
        let mut shmem_servers = Vec::with_capacity(num_devices);
        let mut handles = Vec::with_capacity(num_devices);
        let mut server_names = Vec::with_capacity(num_devices);

        for device_idx in 0..num_devices {
            let server = Arc::new(RpcServer::create().expect("Failed to create mock RpcServer"));
            let shmem_server = Arc::new(
                ShmemServer::create(
                    &pie::shmem::region_name(device_idx),
                    4,
                    4 * 1024 * 1024,
                    4 * 1024 * 1024,
                )
                .expect("Failed to create mock shmem server"),
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

            let shmem_server_clone = Arc::clone(&shmem_server);
            let shmem_behavior_clone = Arc::clone(&behavior);
            let shmem_recorder_clone = Arc::clone(&recorder);
            let shmem_handle = std::thread::Builder::new()
                .name(format!("mock-device-shmem-{device_idx}"))
                .spawn(move || {
                    Self::shmem_loop(
                        device_idx,
                        shmem_server_clone,
                        shmem_behavior_clone,
                        shmem_recorder_clone,
                    );
                })
                .expect("Failed to spawn mock shmem device thread");

            servers.push(server);
            shmem_servers.push(shmem_server);
            handles.push(handle);
            handles.push(shmem_handle);
            server_names.push(name);
        }

        Self {
            servers,
            shmem_servers,
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
                        tracing::warn!("Mock device {device_idx}: failed to send response: {e}");
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

    fn shmem_loop(
        device_idx: usize,
        server: Arc<ShmemServer>,
        behavior: Arc<dyn Behavior>,
        recorder: Arc<CallRecorder>,
    ) {
        server.serve_forever(|req, resp_buf| {
            if req.method_tag != pie::shmem_ipc::METHOD_TAG_FIRE_BATCH {
                return 0;
            }

            let batch_req = decode_shmem_request(req.payload)
                .expect("Failed to decode shmem BatchedForwardPassRequest");
            recorder.record(RecordedCall {
                device_idx,
                method: "fire_batch".to_string(),
                num_requests: batch_req.num_requests(),
                total_tokens: batch_req.total_tokens(),
                timestamp: Instant::now(),
            });

            let response = behavior.handle_fire_batch(&batch_req);
            encode_shmem_response(resp_buf, &response)
                .expect("Failed to encode shmem BatchedForwardPassResponse")
        });
    }
}

impl Drop for MockBackend {
    fn drop(&mut self) {
        // Stop all servers first so polling threads can exit promptly.
        for server in &self.shmem_servers {
            server.stop();
        }
        for server in &self.servers {
            server.close();
        }
        // Then join all threads.
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

// =============================================================================
// Shared-memory fire_batch server
// =============================================================================

#[derive(Debug)]
struct SlotRequest<'a> {
    method_tag: u32,
    payload: &'a [u8],
}

struct ShmemServer {
    name: CString,
    num_slots: usize,
    req_buf_size: usize,
    resp_buf_size: usize,
    slot_stride: usize,
    total_size: usize,
    fd: libc::c_int,
    base: *mut u8,
    stop: AtomicBool,
}

unsafe impl Send for ShmemServer {}
unsafe impl Sync for ShmemServer {}

impl ShmemServer {
    fn create(
        name: &str,
        num_slots: usize,
        req_buf: usize,
        resp_buf: usize,
    ) -> anyhow::Result<Self> {
        let cname = CString::new(name)?;
        let slot_stride = pie::shmem_ipc::SLOT_HEADER_SIZE + req_buf + resp_buf;
        let total_size = pie::shmem_ipc::HEADER_SIZE + num_slots * slot_stride;

        unsafe {
            libc::shm_unlink(cname.as_ptr());
        }
        let fd = unsafe { libc::shm_open(cname.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o600) };
        if fd < 0 {
            anyhow::bail!(
                "shm_open({name}) failed: {}",
                std::io::Error::last_os_error()
            );
        }
        if unsafe { libc::ftruncate(fd, total_size as libc::off_t) } != 0 {
            let err = std::io::Error::last_os_error();
            unsafe {
                libc::close(fd);
                libc::shm_unlink(cname.as_ptr());
            }
            anyhow::bail!("ftruncate({total_size}) failed: {err}");
        }

        let mapped = unsafe {
            libc::mmap(
                ptr::null_mut(),
                total_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if mapped == libc::MAP_FAILED {
            let err = std::io::Error::last_os_error();
            unsafe {
                libc::close(fd);
                libc::shm_unlink(cname.as_ptr());
            }
            anyhow::bail!("mmap({name}) failed: {err}");
        }

        let base = mapped.cast::<u8>();
        unsafe {
            ptr::write_bytes(base, 0, total_size);
        }
        write_u32(base, 0, pie::shmem_ipc::MAGIC);
        write_u32(base, 4, pie::shmem_ipc::SCHEMA_VERSION);
        write_u32(base, 8, num_slots as u32);
        write_u32(base, 12, slot_stride as u32);
        write_u32(base, 16, req_buf as u32);
        write_u32(base, 20, resp_buf as u32);

        Ok(Self {
            name: cname,
            num_slots,
            req_buf_size: req_buf,
            resp_buf_size: resp_buf,
            slot_stride,
            total_size,
            fd,
            base,
            stop: AtomicBool::new(false),
        })
    }

    fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    fn serve_forever<F>(&self, mut handler: F)
    where
        F: FnMut(&SlotRequest<'_>, &mut [u8]) -> usize,
    {
        let mut last_seen = vec![0u64; self.num_slots];

        while !self.stop.load(Ordering::Relaxed) {
            let mut did_work = false;

            for (slot_idx, last_seen_seq) in last_seen.iter_mut().enumerate() {
                let slot = unsafe {
                    self.base
                        .add(pie::shmem_ipc::HEADER_SIZE + slot_idx * self.slot_stride)
                };
                let req_seq = atomic_load_u64(slot, 0);
                if req_seq == *last_seen_seq {
                    continue;
                }

                let method_tag = read_u32(slot, 20);
                let req_len = read_u32(slot, 24) as usize;
                let req_payload = unsafe {
                    std::slice::from_raw_parts(slot.add(pie::shmem_ipc::SLOT_HEADER_SIZE), req_len)
                };
                let resp_payload = unsafe {
                    std::slice::from_raw_parts_mut(
                        slot.add(pie::shmem_ipc::SLOT_HEADER_SIZE + self.req_buf_size),
                        self.resp_buf_size,
                    )
                };

                let resp_len = handler(
                    &SlotRequest {
                        method_tag,
                        payload: req_payload,
                    },
                    resp_payload,
                );

                write_u32(slot, 28, resp_len as u32);
                atomic_store_u64(slot, 40, now_us());
                atomic_store_u64(slot, 8, req_seq);
                *last_seen_seq = req_seq;
                did_work = true;
            }

            if !did_work {
                std::thread::yield_now();
            }
        }
    }
}

impl Drop for ShmemServer {
    fn drop(&mut self) {
        unsafe {
            if !self.base.is_null() {
                libc::munmap(self.base.cast::<libc::c_void>(), self.total_size);
            }
            if self.fd >= 0 {
                libc::close(self.fd);
            }
            libc::shm_unlink(self.name.as_ptr());
        }
    }
}

fn write_u32(base: *mut u8, off: usize, v: u32) {
    unsafe {
        ptr::write_unaligned(base.add(off).cast::<u32>(), v);
    }
}

fn read_u32(base: *const u8, off: usize) -> u32 {
    unsafe { ptr::read_unaligned(base.add(off).cast::<u32>()) }
}

fn atomic_load_u64(base: *const u8, off: usize) -> u64 {
    unsafe { (*(base.add(off).cast::<AtomicU64>())).load(Ordering::Acquire) }
}

fn atomic_store_u64(base: *mut u8, off: usize, v: u64) {
    unsafe { (*(base.add(off).cast::<AtomicU64>())).store(v, Ordering::Release) }
}

fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

// =============================================================================
// BPIQ/BPIS helpers for tests
// =============================================================================

fn decode_shmem_request(buf: &[u8]) -> anyhow::Result<BatchedForwardPassRequest> {
    if buf.len() < pie::shmem_schema::HEADER_SIZE {
        anyhow::bail!(
            "request buffer too small: {} < {}",
            buf.len(),
            pie::shmem_schema::HEADER_SIZE
        );
    }
    let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    anyhow::ensure!(magic == pie::shmem_schema::MAGIC, "request magic mismatch");
    let schema = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    anyhow::ensure!(
        schema == pie::shmem_schema::SCHEMA_VERSION,
        "request schema mismatch"
    );

    let device_id = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
    let flags = u32::from_le_bytes(buf[12..16].try_into().unwrap());

    Ok(BatchedForwardPassRequest {
        token_ids: ByteVec(read_u32_array(buf, pie::shmem_schema::A_TOKEN_IDS)?),
        position_ids: ByteVec(read_u32_array(buf, pie::shmem_schema::A_POSITION_IDS)?),
        kv_page_indices: ByteVec(read_u32_array(buf, pie::shmem_schema::A_KV_PAGE_INDICES)?),
        kv_page_indptr: ByteVec(read_u32_array(buf, pie::shmem_schema::A_KV_PAGE_INDPTR)?),
        kv_last_page_lens: ByteVec(read_u32_array(buf, pie::shmem_schema::A_KV_LAST_PAGE_LENS)?),
        qo_indptr: ByteVec(read_u32_array(buf, pie::shmem_schema::A_QO_INDPTR)?),
        flattened_masks: ByteVec(read_u32_array(buf, pie::shmem_schema::A_FLATTENED_MASKS)?),
        mask_indptr: ByteVec(read_u32_array(buf, pie::shmem_schema::A_MASK_INDPTR)?),
        logit_masks: ByteVec(read_u32_array(buf, pie::shmem_schema::A_LOGIT_MASKS)?),
        logit_mask_indptr: ByteVec(read_u32_array(buf, pie::shmem_schema::A_LOGIT_MASK_INDPTR)?),
        sampling_indices: ByteVec(read_u32_array(buf, pie::shmem_schema::A_SAMPLING_INDICES)?),
        sampling_indptr: ByteVec(read_u32_array(buf, pie::shmem_schema::A_SAMPLING_INDPTR)?),
        sampler_temperatures: ByteVecF32(read_f32_array(
            buf,
            pie::shmem_schema::A_SAMPLER_TEMPERATURES,
        )?),
        sampler_top_k: ByteVec(read_u32_array(buf, pie::shmem_schema::A_SAMPLER_TOP_K)?),
        sampler_top_p: ByteVecF32(read_f32_array(buf, pie::shmem_schema::A_SAMPLER_TOP_P)?),
        sampler_min_p: ByteVecF32(read_f32_array(buf, pie::shmem_schema::A_SAMPLER_MIN_P)?),
        sampler_types: ByteVec(read_u32_array(buf, pie::shmem_schema::A_SAMPLER_TYPES)?),
        sampler_seeds: ByteVec(read_u32_array(buf, pie::shmem_schema::A_SAMPLER_SEEDS)?),
        request_num_samplers: ByteVec(read_u32_array(
            buf,
            pie::shmem_schema::A_REQUEST_NUM_SAMPLERS,
        )?),
        sampler_label_ids: ByteVec(read_u32_array(buf, pie::shmem_schema::A_SAMPLER_LABEL_IDS)?),
        sampler_label_indptr: ByteVec(read_u32_array(
            buf,
            pie::shmem_schema::A_SAMPLER_LABEL_INDPTR,
        )?),
        adapter_indices: read_i64_option_array(buf, pie::shmem_schema::A_ADAPTER_INDICES)?
            .into_iter()
            .map(|v| v.map(|x| x as u64))
            .collect(),
        adapter_seeds: read_i64_option_array(buf, pie::shmem_schema::A_ADAPTER_SEEDS)?,
        spec_token_ids: ByteVec(read_u32_array(buf, pie::shmem_schema::A_SPEC_TOKEN_IDS)?),
        spec_position_ids: ByteVec(read_u32_array(buf, pie::shmem_schema::A_SPEC_POSITION_IDS)?),
        spec_indptr: ByteVec(read_u32_array(buf, pie::shmem_schema::A_SPEC_INDPTR)?),
        output_spec_flags: read_u8_array(buf, pie::shmem_schema::A_OUTPUT_SPEC_FLAGS)?
            .into_iter()
            .map(|b| b != 0)
            .collect(),
        context_ids: read_u64_array(buf, pie::shmem_schema::A_CONTEXT_IDS)?,
        single_token_mode: flags & 1 != 0,
        has_user_mask: false,
        device_id,
    })
}

fn read_array_bytes(buf: &[u8], idx: usize, elem_size: usize) -> anyhow::Result<&[u8]> {
    let entry_off = 32 + idx * 8;
    if entry_off + 8 > pie::shmem_schema::HEADER_SIZE {
        anyhow::bail!("array {idx} table entry past header");
    }
    let offset = u32::from_le_bytes(buf[entry_off..entry_off + 4].try_into().unwrap()) as usize;
    let len = u32::from_le_bytes(buf[entry_off + 4..entry_off + 8].try_into().unwrap()) as usize;
    let nbytes = len * elem_size;
    if offset + nbytes > buf.len() {
        anyhow::bail!(
            "array {idx} body past buffer: offset={offset} len={len} elem_size={elem_size} buf={}",
            buf.len()
        );
    }
    Ok(&buf[offset..offset + nbytes])
}

fn read_u32_array(buf: &[u8], idx: usize) -> anyhow::Result<Vec<u32>> {
    let bytes = read_array_bytes(buf, idx, 4)?;
    Ok(bytemuck::try_cast_slice::<u8, u32>(bytes)
        .map_err(|e| anyhow::anyhow!("array {idx} cast: {e:?}"))?
        .to_vec())
}

fn read_f32_array(buf: &[u8], idx: usize) -> anyhow::Result<Vec<f32>> {
    let bytes = read_array_bytes(buf, idx, 4)?;
    Ok(bytemuck::try_cast_slice::<u8, f32>(bytes)
        .map_err(|e| anyhow::anyhow!("array {idx} cast: {e:?}"))?
        .to_vec())
}

fn read_u64_array(buf: &[u8], idx: usize) -> anyhow::Result<Vec<u64>> {
    let bytes = read_array_bytes(buf, idx, 8)?;
    Ok(bytemuck::try_cast_slice::<u8, u64>(bytes)
        .map_err(|e| anyhow::anyhow!("array {idx} cast: {e:?}"))?
        .to_vec())
}

fn read_u8_array(buf: &[u8], idx: usize) -> anyhow::Result<Vec<u8>> {
    Ok(read_array_bytes(buf, idx, 1)?.to_vec())
}

fn read_i64_option_array(buf: &[u8], idx: usize) -> anyhow::Result<Vec<Option<i64>>> {
    let bytes = read_array_bytes(buf, idx, 8)?;
    Ok(bytemuck::try_cast_slice::<u8, i64>(bytes)
        .map_err(|e| anyhow::anyhow!("array {idx} cast: {e:?}"))?
        .iter()
        .copied()
        .map(|v| {
            if v == -1 || v == i64::MIN {
                None
            } else {
                Some(v)
            }
        })
        .collect())
}

fn encode_shmem_response(
    buf: &mut [u8],
    resp: &BatchedForwardPassResponse,
) -> anyhow::Result<usize> {
    let body = rmp_serde::to_vec_named(resp)?;
    let needed = pie::shmem_schema::RESP_HEADER_SIZE + body.len();
    if needed > buf.len() {
        anyhow::bail!(
            "response buffer too small: need {needed}, have {}",
            buf.len()
        );
    }

    let total_tokens: usize = resp.results.iter().map(|r| r.tokens.len()).sum();
    buf[0..4].copy_from_slice(&pie::shmem_schema::RESP_MAGIC.to_le_bytes());
    buf[4..8].copy_from_slice(&pie::shmem_schema::RESP_MODE_MSGPACK.to_le_bytes());
    buf[8..12].copy_from_slice(&(resp.results.len() as u32).to_le_bytes());
    buf[12..16].copy_from_slice(&(total_tokens as u32).to_le_bytes());
    buf[pie::shmem_schema::RESP_HEADER_SIZE..needed].copy_from_slice(&body);
    Ok(needed)
}
