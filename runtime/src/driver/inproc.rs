//! `DriverChannel` over an in-process FFI vtable.
//!
//! Direct-FFI handoff with the C++ driver: `recv` returns a
//! `*const PieFrameDesc` aliasing the runtime's own heap; `send_response`
//! takes a `*const PieResponseFrameDesc` filled by the C++ driver. No
//! rkyv encode/decode on this path — the shmem transport's bytes
//! pathway exists in `super::shmem` for the cross-process case.
//!
//! Concurrency model: `submit` runs on a tokio worker; `recv` /
//! `send_response` run on the C++ driver's own thread. The two ends
//! coordinate through a small `InProcState` shared via the vtable's
//! `ctx` pointer (registered in `CTX_REGISTRY` by `ffi_vtable`).

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use dashmap::DashMap;
use once_cell::sync::Lazy;
use tokio::runtime::RuntimeFlavor;

use super::{DriverChannel, DriverRequest, DriverResponse};
use pie_bridge::schema::{
    __pie_response_frame_from_desc, PieFrameDesc, PieFrameView, PieResponseFrameDesc,
    pie_frame_view,
};

pub use pie_bridge::ffi::InProcVTable;

// ---------------------------------------------------------------------------
// Pending entry
// ---------------------------------------------------------------------------

/// Self-referential pair: a `Box<Frame>` whose heap is stable, plus a
/// `PieFrameView<'static>` that borrows from it. The 'static lifetime
/// is a manual liveness promise — the view is constructed in `recv`
/// and dropped here in `send_response`, always before the `Box<Frame>`.
struct PendingEntry {
    // Drop order: view first, then frame. Rust drops struct fields in
    // declaration order, so the view (which has raw pointers into
    // `frame`'s heap) is dropped before the frame whose heap they
    // point into. Reordering these fields is a soundness break.
    view: Option<PieFrameView<'static>>,
    frame: Box<pie_bridge::Frame>,
    slot: Option<Arc<ResponseSlot>>,
}

// SAFETY: `PieFrameView` carries raw pointers (no lifetime) into a
// heap allocation we own via `frame: Box<Frame>`. The state mutex
// serializes access; pointers are only dereferenced while the pending
// entry exists (between `recv` and `send_response`). Sending the entry
// across threads is therefore sound — both writer and reader threads
// see the same heap allocation, and the lifetime is enforced manually
// by the recv → send_response handshake.
unsafe impl Send for PendingEntry {}
unsafe impl Sync for PendingEntry {}

struct ResponseSlot {
    ready: AtomicBool,
    result: Mutex<Option<Result<DriverResponse>>>,
    cv: Condvar,
}

impl ResponseSlot {
    fn new() -> Self {
        Self {
            ready: AtomicBool::new(false),
            result: Mutex::new(None),
            cv: Condvar::new(),
        }
    }

    fn complete(&self, result: Result<DriverResponse>) {
        let mut guard = self.result.lock().unwrap_or_else(|e| e.into_inner());
        *guard = Some(result);
        self.ready.store(true, Ordering::Release);
        self.cv.notify_one();
    }

    fn take_result(&self) -> Result<DriverResponse> {
        let mut guard = self.result.lock().unwrap_or_else(|e| e.into_inner());
        guard
            .take()
            .unwrap_or_else(|| Err(anyhow!("InProcChannel response slot empty")))
    }

    fn wait(&self, spin_budget_us: u64) -> Result<DriverResponse> {
        if spin_budget_us > 0 {
            let started = Instant::now();
            let deadline = (spin_budget_us != u64::MAX)
                .then(|| started + Duration::from_micros(spin_budget_us));
            let mut iters: u32 = 0;
            loop {
                if self.ready.load(Ordering::Acquire) {
                    return self.take_result();
                }
                iters = iters.wrapping_add(1);
                if iters & 0xFF == 0 && deadline.is_some_and(|deadline| Instant::now() >= deadline)
                {
                    break;
                }
                std::hint::spin_loop();
            }
        }

        let mut guard = self.result.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(result) = guard.take() {
                return result;
            }
            guard = self.cv.wait(guard).unwrap_or_else(|e| e.into_inner());
        }
    }
}

struct InProcState {
    aborted: AtomicBool,
    next_id: AtomicU32,

    /// Driver-side spin budget (µs) before `recv` falls back to
    /// `Condvar::wait`. Zero = park immediately (lowest CPU, ~7 µs wake).
    /// Non-zero burns one core for up to that long after each completed
    /// request, in exchange for ~ns wake on back-to-back requests. Pie's
    /// production workload fires every ~10-100 µs during a generation,
    /// so 50-200 µs covers the common case without measurable CPU cost.
    spin_budget_us: u64,

    /// Inbox: req_ids waiting to be recv'd by the driver thread.
    inbox: Mutex<VecDeque<u32>>,
    inbox_cv: Condvar,

    /// Pending: req_id → entry.
    pending: Mutex<HashMap<u32, PendingEntry>>,
}

impl InProcState {
    fn new(spin_budget_us: u64) -> Arc<Self> {
        Arc::new(Self {
            aborted: AtomicBool::new(false),
            next_id: AtomicU32::new(1),
            spin_budget_us,
            inbox: Mutex::new(VecDeque::new()),
            inbox_cv: Condvar::new(),
            pending: Mutex::new(HashMap::new()),
        })
    }

    /// Look up the channel state for a vtable callback.
    fn from_ctx(ctx: *mut c_void) -> Option<Arc<InProcState>> {
        CTX_REGISTRY
            .get(&(ctx as usize))
            .map(|entry| (**entry.value()).clone())
    }

    fn signal_abort(&self) {
        self.aborted.store(true, Ordering::Release);
        if let Ok(_guard) = self.inbox.lock() {
            self.inbox_cv.notify_all();
        }
    }
}

/// Registry of live `InProcState` instances, keyed by the vtable's
/// ctx pointer.
static CTX_REGISTRY: Lazy<DashMap<usize, Box<Arc<InProcState>>>> = Lazy::new(DashMap::new);

// ---------------------------------------------------------------------------
// Public InProcChannel
// ---------------------------------------------------------------------------

pub struct InProcChannel {
    state: Arc<InProcState>,
}

impl Default for InProcChannel {
    fn default() -> Self {
        Self::new()
    }
}

/// Default spin budget for the driver-side `recv` callback before it
/// falls back to `Condvar::wait`. See [`InProcState::spin_budget_us`].
pub const DEFAULT_SPIN_BUDGET_US: u64 = 1_000;

impl InProcChannel {
    pub fn new() -> Self {
        Self::with_spin_budget(DEFAULT_SPIN_BUDGET_US)
    }

    /// Build a channel with a custom driver-side spin budget. Set to `0`
    /// to disable spinning (driver thread parks immediately on
    /// `Condvar::wait`); higher values trade CPU for wake latency.
    pub fn with_spin_budget(spin_budget_us: u64) -> Self {
        Self {
            state: InProcState::new(spin_budget_us),
        }
    }

    pub fn ffi_vtable(&self) -> InProcVTable {
        let boxed: Box<Arc<InProcState>> = Box::new(self.state.clone());
        let ctx_ptr = Box::as_ref(&boxed) as *const Arc<InProcState> as *mut c_void;
        CTX_REGISTRY.insert(ctx_ptr as usize, boxed);
        InProcVTable {
            recv: vt_recv,
            send_response: vt_send_response,
            ctx: ctx_ptr,
        }
    }

    /// # Safety
    /// `ctx` must have been produced by [`Self::ffi_vtable`] and not
    /// already released.
    pub unsafe fn release(ctx: *mut c_void) {
        CTX_REGISTRY.remove(&(ctx as usize));
    }

    fn submit_sync_for_state(
        state: &Arc<InProcState>,
        req: DriverRequest,
    ) -> Result<DriverResponse> {
        if state.aborted.load(Ordering::Acquire) {
            return Err(anyhow!("InProcChannel aborted"));
        }
        let req_id = state.next_id.fetch_add(1, Ordering::Relaxed);
        let slot = Arc::new(ResponseSlot::new());
        let frame = Box::new(pie_bridge::Frame {
            driver_id: req.driver_id as u32,
            payload: req.payload,
        });

        {
            let mut pending = state
                .pending
                .lock()
                .map_err(|_| anyhow!("pending lock poisoned"))?;
            pending.insert(
                req_id,
                PendingEntry {
                    view: None,
                    frame,
                    slot: Some(slot.clone()),
                },
            );
            if state.aborted.load(Ordering::Acquire) {
                pending.remove(&req_id);
                return Err(anyhow!("InProcChannel aborted"));
            }
        }
        {
            let mut inbox = state
                .inbox
                .lock()
                .map_err(|_| anyhow!("inbox lock poisoned"))?;
            if state.aborted.load(Ordering::Acquire) {
                drop(inbox);
                if let Ok(mut pending) = state.pending.lock() {
                    pending.remove(&req_id);
                }
                return Err(anyhow!("InProcChannel aborted"));
            }
            inbox.push_back(req_id);
            state.inbox_cv.notify_one();
        }

        slot.wait(state.spin_budget_us)
    }
}

#[async_trait]
impl DriverChannel for InProcChannel {
    async fn submit(&self, req: DriverRequest) -> Result<DriverResponse> {
        let state = self.state.clone();
        match tokio::runtime::Handle::try_current() {
            Ok(handle) if handle.runtime_flavor() == RuntimeFlavor::MultiThread => {
                tokio::task::block_in_place(|| Self::submit_sync_for_state(&state, req))
            }
            Ok(_) => tokio::task::spawn_blocking(move || Self::submit_sync_for_state(&state, req))
                .await
                .map_err(|e| anyhow!("InProcChannel blocking submit task failed: {e}"))?,
            Err(_) => Self::submit_sync_for_state(&state, req),
        }
    }

    fn submit_sync(&self, req: DriverRequest) -> Result<DriverResponse> {
        Self::submit_sync_for_state(&self.state, req)
    }

    fn notify(&self, req: DriverRequest) -> Result<()> {
        if self.state.aborted.load(Ordering::Acquire) {
            return Err(anyhow!("InProcChannel aborted"));
        }
        let req_id = self.state.next_id.fetch_add(1, Ordering::Relaxed);
        let frame = Box::new(pie_bridge::Frame {
            driver_id: req.driver_id as u32,
            payload: req.payload,
        });

        {
            let mut pending = self
                .state
                .pending
                .lock()
                .map_err(|_| anyhow!("pending lock poisoned"))?;
            pending.insert(
                req_id,
                PendingEntry {
                    view: None,
                    frame,
                    slot: None,
                },
            );
            if self.state.aborted.load(Ordering::Acquire) {
                pending.remove(&req_id);
                return Err(anyhow!("InProcChannel aborted"));
            }
        }
        {
            let mut inbox = self
                .state
                .inbox
                .lock()
                .map_err(|_| anyhow!("inbox lock poisoned"))?;
            if self.state.aborted.load(Ordering::Acquire) {
                drop(inbox);
                if let Ok(mut pending) = self.state.pending.lock() {
                    pending.remove(&req_id);
                }
                return Err(anyhow!("InProcChannel aborted"));
            }
            inbox.push_back(req_id);
            self.state.inbox_cv.notify_one();
        }
        Ok(())
    }

    fn abort(&self) {
        self.state.signal_abort();
        let slots = if let Ok(mut p) = self.state.pending.lock() {
            let mut slots = Vec::new();
            p.retain(|_, entry| {
                if let Some(slot) = entry.slot.as_ref() {
                    slots.push(slot.clone());
                }
                // Keep entries already handed to the driver alive until
                // send_response returns. Their views contain raw pointers
                // into the frame allocation owned by the pending entry.
                entry.view.is_some()
            });
            slots
        } else {
            Vec::new()
        };
        if let Ok(mut q) = self.state.inbox.lock() {
            q.clear();
        }
        for slot in slots {
            slot.complete(Err(anyhow!("InProcChannel aborted")));
        }
    }
}

// ---------------------------------------------------------------------------
// Vtable callbacks
// ---------------------------------------------------------------------------

/// Try to pop the next req_id off the inbox, spinning up to
/// `state.spin_budget_us` before falling back to `Condvar::wait`.
/// Returns `None` if the channel aborted while waiting.
///
/// The spin phase uses `try_lock` + a periodic `Instant::now()` deadline
/// check (every 256 iters to keep the cost off the hot path). The park
/// phase re-checks the inbox under the mutex before waiting so it
/// closes the race where the producer pushes between the spin's timeout
/// and the consumer's `lock()`.
fn recv_with_spin(state: &InProcState) -> Option<u32> {
    if state.spin_budget_us > 0 {
        let started = Instant::now();
        let deadline = (state.spin_budget_us != u64::MAX)
            .then(|| started + Duration::from_micros(state.spin_budget_us));
        let mut iters: u32 = 0;
        loop {
            if state.aborted.load(Ordering::Acquire) {
                return None;
            }
            if let Ok(mut inbox) = state.inbox.try_lock() {
                if let Some(id) = inbox.pop_front() {
                    return Some(id);
                }
            }
            iters = iters.wrapping_add(1);
            // `Instant::now()` is a vDSO call but still ~10 ns per
            // invocation; amortize by sampling every 256 spin iters.
            if iters & 0xFF == 0 && deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }
            std::hint::spin_loop();
        }
    }
    // Park phase: take the lock and Condvar-wait. Re-checking under the
    // mutex on every wake closes the lost-wakeup gap (producer pushed
    // between our spin timeout and our lock acquisition).
    let mut inbox = state.inbox.lock().ok()?;
    loop {
        if state.aborted.load(Ordering::Acquire) {
            return None;
        }
        if let Some(id) = inbox.pop_front() {
            return Some(id);
        }
        inbox = state.inbox_cv.wait(inbox).ok()?;
    }
}

/// recv: block until a request is queued or the channel is aborted.
unsafe extern "C" fn vt_recv(
    ctx: *mut c_void,
    out_request: *mut *const PieFrameDesc,
    out_req_id: *mut u32,
) -> c_int {
    let Some(state) = InProcState::from_ctx(ctx) else {
        return -1;
    };

    let req_id = match recv_with_spin(&state) {
        Some(id) => id,
        None => return -1,
    };

    // Build view; park under same id for `send_response` to find.
    let desc_ptr: *const PieFrameDesc = {
        let mut pending = match state.pending.lock() {
            Ok(g) => g,
            Err(_) => return -1,
        };
        let Some(entry) = pending.get_mut(&req_id) else {
            return -1;
        };
        // SAFETY: `entry.frame` is a Box; its heap is stable. The
        // returned view holds raw pointers into that heap.
        let view = pie_frame_view(&*entry.frame);
        // SAFETY: extend lifetime — view dies before frame (see
        // `PendingEntry` field order doc).
        let view_static: PieFrameView<'static> = unsafe { std::mem::transmute(view) };
        entry.view = Some(view_static);
        &entry.view.as_ref().unwrap().desc as *const PieFrameDesc
    };

    unsafe {
        if !out_request.is_null() {
            *out_request = desc_ptr;
        }
        if !out_req_id.is_null() {
            *out_req_id = req_id;
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use pie_bridge::schema::{
        PIE_REQUEST_PAYLOAD_ADAPTER, PIE_RESPONSE_PAYLOAD_STATUS, PieResponsePayloadDesc,
        PieStatusResponseDesc,
    };

    fn save_adapter_req(driver_id: usize, adapter_id: u64) -> DriverRequest {
        DriverRequest {
            driver_id,
            payload: pie_bridge::RequestPayload::Adapter(pie_bridge::AdapterRequest {
                op: pie_bridge::AdapterOp::Save,
                adapter_id,
                path: String::new(),
            }),
        }
    }

    fn zo_init_req(driver_id: usize, adapter_id: u64) -> DriverRequest {
        DriverRequest {
            driver_id,
            payload: pie_bridge::RequestPayload::Adapter(pie_bridge::AdapterRequest {
                op: pie_bridge::AdapterOp::ZoInit,
                adapter_id,
                path: String::new(),
            }),
        }
    }

    fn status_code(resp: &DriverResponse) -> Option<i32> {
        match &resp.payload {
            pie_bridge::ResponsePayload::Status(s) => Some(s.status),
            _ => None,
        }
    }

    /// Round-trip: submit a Health request from one task, simulate a
    /// C++ driver thread that calls recv → builds a StatusResponse →
    /// calls send_response. Verify the awaiter wakes with the right
    /// response.
    #[tokio::test]
    async fn health_round_trip_through_inproc_vtable() {
        let channel = InProcChannel::new();
        let vt = channel.ffi_vtable();
        // Smuggle the raw fn-ptrs + ctx across thread boundaries as a
        // usize. `*mut c_void` and `unsafe extern "C" fn` types are
        // !Send by default; this is harmless here because the C ABI
        // doesn't care which thread invokes them.
        let vt_ctx_us = vt.ctx as usize;
        let recv_us = vt.recv as usize;
        let send_us = vt.send_response as usize;

        // "Driver thread" — pulls one request, replies with status 0.
        let driver = std::thread::spawn(move || {
            type RecvFn = unsafe extern "C" fn(
                ctx: *mut c_void,
                out_request: *mut *const PieFrameDesc,
                out_req_id: *mut u32,
            ) -> c_int;
            type SendFn = unsafe extern "C" fn(
                ctx: *mut c_void,
                req_id: u32,
                response: *const PieResponseFrameDesc,
            );
            let vt_ctx = vt_ctx_us as *mut c_void;
            let recv: RecvFn = unsafe { std::mem::transmute(recv_us) };
            let send: SendFn = unsafe { std::mem::transmute(send_us) };

            let mut request_ptr: *const PieFrameDesc = ::core::ptr::null();
            let mut req_id: u32 = 0;
            let rc = unsafe { recv(vt_ctx, &mut request_ptr, &mut req_id) };
            assert_eq!(rc, 0);
            assert!(!request_ptr.is_null());
            let frame: &PieFrameDesc = unsafe { &*request_ptr };
            assert_eq!(frame.driver_id, 42);
            // SaveAdapter → REQUEST_PAYLOAD_ADAPTER.
            assert_eq!(frame.payload.kind, PIE_REQUEST_PAYLOAD_ADAPTER);

            let mut resp = PieResponseFrameDesc::default();
            resp.driver_id = 42;
            resp.aborted = 0;
            resp.payload = PieResponsePayloadDesc {
                kind: PIE_RESPONSE_PAYLOAD_STATUS,
                forward: Default::default(),
                status: PieStatusResponseDesc { status: 0 },
            };
            unsafe { send(vt_ctx, req_id, &resp as *const _) };
        });

        let resp = channel.submit(save_adapter_req(42, 0)).await;
        driver.join().unwrap();
        let r = resp.expect("submit failed");
        assert_eq!(
            status_code(&r),
            Some(0),
            "expected Status response, got {:?}",
            r
        );

        unsafe { InProcChannel::release(vt_ctx_us as *mut c_void) };
    }

    /// Abort path: an in-flight submit should fail when the channel
    /// aborts, and a fresh submit on the aborted channel returns
    /// immediately with an error.
    #[tokio::test]
    async fn abort_wakes_pending_submitters() {
        let channel = InProcChannel::new();
        let vt = channel.ffi_vtable();
        let vt_ctx = vt.ctx;

        // Submit but never drain — driver isn't started. Then abort.
        let abort_handle = {
            let arc = Arc::new(channel);
            let arc2 = arc.clone();
            let fut = tokio::spawn(async move { arc.submit(save_adapter_req(0, 0)).await });
            // Give the submit a tick to enqueue.
            tokio::task::yield_now().await;
            arc2.abort();
            fut
        };
        let outcome = abort_handle.await.expect("task panicked");
        assert!(
            outcome.is_err(),
            "expected error after abort, got {outcome:?}"
        );

        unsafe { InProcChannel::release(vt_ctx) };
    }

    // Helper: spawn a driver thread that pulls `n` requests and replies
    // with `Status(req_id as i32)` to each. Returns the join handle so
    // the test can wait on completion.
    fn spawn_status_driver(vt: &InProcVTable, expected: usize) -> std::thread::JoinHandle<usize> {
        let vt_ctx_us = vt.ctx as usize;
        let recv_us = vt.recv as usize;
        let send_us = vt.send_response as usize;
        std::thread::spawn(move || {
            type RecvFn = unsafe extern "C" fn(
                ctx: *mut c_void,
                out_request: *mut *const PieFrameDesc,
                out_req_id: *mut u32,
            ) -> c_int;
            type SendFn = unsafe extern "C" fn(
                ctx: *mut c_void,
                req_id: u32,
                response: *const PieResponseFrameDesc,
            );
            let vt_ctx = vt_ctx_us as *mut c_void;
            let recv: RecvFn = unsafe { std::mem::transmute(recv_us) };
            let send: SendFn = unsafe { std::mem::transmute(send_us) };

            let mut handled = 0usize;
            while handled < expected {
                let mut request_ptr: *const PieFrameDesc = ::core::ptr::null();
                let mut req_id: u32 = 0;
                let rc = unsafe { recv(vt_ctx, &mut request_ptr, &mut req_id) };
                if rc != 0 {
                    break;
                }
                let mut resp = PieResponseFrameDesc::default();
                resp.payload = PieResponsePayloadDesc {
                    kind: PIE_RESPONSE_PAYLOAD_STATUS,
                    forward: Default::default(),
                    status: PieStatusResponseDesc {
                        status: req_id as i32,
                    },
                };
                unsafe { send(vt_ctx, req_id, &resp as *const _) };
                handled += 1;
            }
            handled
        })
    }

    /// 16 submitters race against one driver thread. Verify every
    /// submit completes with the right Status — req_id is echoed back
    /// via the driver, so each await gets the response intended for it.
    #[tokio::test]
    async fn many_concurrent_submitters() {
        const N: usize = 16;
        let channel = Arc::new(InProcChannel::new());
        let vt = channel.ffi_vtable();
        let vt_ctx = vt.ctx;

        let driver = spawn_status_driver(&vt, N);

        let tasks: Vec<_> = (0..N)
            .map(|_| {
                let ch = channel.clone();
                tokio::spawn(async move { ch.submit(save_adapter_req(0, 0)).await })
            })
            .collect();

        let mut seen = std::collections::HashSet::new();
        for t in tasks {
            let r = t.await.expect("task panicked").expect("submit failed");
            let s = status_code(&r).expect("expected Status response");
            assert!(seen.insert(s), "duplicate Status: {s}");
        }
        assert_eq!(seen.len(), N);

        let handled = driver.join().expect("driver thread panicked");
        assert_eq!(handled, N);

        unsafe { InProcChannel::release(vt_ctx) };
    }

    /// notify() should enqueue and reach the driver, but the runtime
    /// side doesn't await — send_response must not panic when tx is None.
    #[tokio::test]
    async fn notify_with_no_awaiter_does_not_panic() {
        let channel = InProcChannel::new();
        let vt = channel.ffi_vtable();
        let vt_ctx = vt.ctx;
        let driver = spawn_status_driver(&vt, 1);

        channel.notify(zo_init_req(0, 0)).expect("notify");

        // Wait for the driver to process.
        let handled = driver.join().expect("driver");
        assert_eq!(handled, 1);

        unsafe { InProcChannel::release(vt_ctx) };
    }

    /// Recv on an aborted channel returns -1 (not blocked indefinitely).
    #[tokio::test]
    async fn recv_returns_minus_one_on_aborted_channel() {
        type RecvFn = unsafe extern "C" fn(
            ctx: *mut c_void,
            out_request: *mut *const PieFrameDesc,
            out_req_id: *mut u32,
        ) -> c_int;

        let channel = InProcChannel::new();
        let vt = channel.ffi_vtable();
        let vt_ctx_us = vt.ctx as usize;
        let recv_us = vt.recv as usize;

        channel.abort();

        let h = std::thread::spawn(move || {
            let recv: RecvFn = unsafe { std::mem::transmute(recv_us) };
            let mut request_ptr: *const PieFrameDesc = ::core::ptr::null();
            let mut req_id: u32 = 0;
            unsafe { recv(vt_ctx_us as *mut c_void, &mut request_ptr, &mut req_id) }
        });
        let rc = h.join().expect("driver thread");
        assert_eq!(rc, -1);

        unsafe { InProcChannel::release(vt_ctx_us as *mut c_void) };
    }

    /// Abort racing with the driver thread waiting in recv. The driver
    /// must wake within a few ms.
    #[tokio::test]
    async fn abort_wakes_blocked_recv() {
        type RecvFn = unsafe extern "C" fn(
            ctx: *mut c_void,
            out_request: *mut *const PieFrameDesc,
            out_req_id: *mut u32,
        ) -> c_int;

        let channel = Arc::new(InProcChannel::new());
        let vt = channel.ffi_vtable();
        let vt_ctx_us = vt.ctx as usize;
        let recv_us = vt.recv as usize;

        let h = std::thread::spawn(move || {
            let recv: RecvFn = unsafe { std::mem::transmute(recv_us) };
            let mut request_ptr: *const PieFrameDesc = ::core::ptr::null();
            let mut req_id: u32 = 0;
            unsafe { recv(vt_ctx_us as *mut c_void, &mut request_ptr, &mut req_id) }
        });

        // Give the driver a moment to enter the wait.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        channel.abort();

        let rc = h.join().expect("driver thread");
        assert_eq!(rc, -1);

        unsafe { InProcChannel::release(vt_ctx_us as *mut c_void) };
    }

    /// Submit/abort race repeated 32 times. Each iteration: spawn submit
    /// + abort in parallel; either the submit completes (driver ran it)
    /// or it errors (abort got there first). No panics, no leaks.
    #[tokio::test]
    async fn submit_abort_race_repeated() {
        for _ in 0..32 {
            let channel = Arc::new(InProcChannel::new());
            let vt = channel.ffi_vtable();
            let vt_ctx = vt.ctx;
            let driver = spawn_status_driver(&vt, 1);

            let arc = channel.clone();
            let submit = tokio::spawn(async move { arc.submit(save_adapter_req(0, 0)).await });
            // Yield to let submit enqueue, then race abort.
            tokio::task::yield_now().await;
            channel.abort();

            let _ = submit.await.expect("task");
            let _ = driver.join();
            unsafe { InProcChannel::release(vt_ctx) };
        }
    }
}

/// send_response: take the parked entry, decode the response desc,
/// deliver to the awaiter.
unsafe extern "C" fn vt_send_response(
    ctx: *mut c_void,
    req_id: u32,
    response: *const PieResponseFrameDesc,
) {
    let Some(state) = InProcState::from_ctx(ctx) else {
        return;
    };
    let entry = {
        let mut pending = match state.pending.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        pending.remove(&req_id)
    };
    let Some(entry) = entry else { return };

    let driver_response: Result<DriverResponse> = if response.is_null() {
        Err(anyhow!("InProcChannel: null response desc"))
    } else {
        let desc_ref: &PieResponseFrameDesc = unsafe { &*response };
        let owned = __pie_response_frame_from_desc(desc_ref);
        Ok(DriverResponse {
            aborted: owned.aborted,
            payload: owned.payload,
        })
    };

    let PendingEntry { slot, .. } = entry; // view + frame drop in field order
    if let Some(slot) = slot {
        slot.complete(driver_response);
    }
}
