//! Experimental in-process channel backed by preallocated slots.
//!
//! This keeps the same C ABI vtable shape as [`super::InProcChannel`]:
//! the driver calls `recv(ctx, &request, &req_id)`, reads a borrowed
//! `PieFrameDesc`, then calls `send_response(ctx, req_id, response)`.
//! The difference is internal bookkeeping: requests live in fixed slots
//! and the driver receives slot ids through a bounded queue instead of
//! `HashMap + VecDeque + per-request ResponseSlot`.

use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use crossbeam::queue::ArrayQueue;
use dashmap::DashMap;
use once_cell::sync::Lazy;
use tokio::runtime::RuntimeFlavor;

use super::{DriverChannel, DriverRequest, DriverResponse};
use pie_bridge::ffi::InProcVTable;
use pie_bridge::schema::{
    __pie_response_frame_from_desc, PieFrameDesc, PieFrameView, PieResponseFrameDesc,
    pie_frame_view,
};

const SLOT_BITS: u32 = 16;
const SLOT_MASK: u32 = (1 << SLOT_BITS) - 1;
pub const DEFAULT_RING_CAPACITY: usize = 1024;
pub const DEFAULT_SPIN_BUDGET_US: u64 = 100;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotState {
    Free,
    Queued,
    InDriver,
    Responded,
}

struct RingSlot {
    ready_generation: AtomicU32,
    inner: Mutex<RingSlotInner>,
    cv: Condvar,
}

struct RingSlotInner {
    state: SlotState,
    generation: u32,
    view: Option<PieFrameView<'static>>,
    frame: Option<Box<pie_bridge::Frame>>,
    result: Option<Result<DriverResponse>>,
    wants_response: bool,
}

// SAFETY: `view` contains raw pointers into `frame`. The slot mutex
// serializes mutation, and `view` is always cleared before `frame`.
unsafe impl Send for RingSlotInner {}

impl RingSlot {
    fn new() -> Self {
        Self {
            ready_generation: AtomicU32::new(0),
            inner: Mutex::new(RingSlotInner {
                state: SlotState::Free,
                generation: 0,
                view: None,
                frame: None,
                result: None,
                wants_response: false,
            }),
            cv: Condvar::new(),
        }
    }
}

struct InProcRingState {
    aborted: AtomicBool,
    spin_budget_us: u64,
    slots: Vec<RingSlot>,
    free: ArrayQueue<usize>,
    ready: ArrayQueue<usize>,
    free_lock: Mutex<()>,
    free_cv: Condvar,
    ready_lock: Mutex<()>,
    ready_cv: Condvar,
}

impl InProcRingState {
    fn new(capacity: usize, spin_budget_us: u64) -> Result<Arc<Self>> {
        if capacity == 0 || capacity > SLOT_MASK as usize {
            return Err(anyhow!(
                "InProcRingChannel capacity must be in 1..={}",
                SLOT_MASK
            ));
        }
        let free = ArrayQueue::new(capacity);
        for i in 0..capacity {
            free.push(i).expect("initial free queue has capacity");
        }
        Ok(Arc::new(Self {
            aborted: AtomicBool::new(false),
            spin_budget_us,
            slots: (0..capacity).map(|_| RingSlot::new()).collect(),
            free,
            ready: ArrayQueue::new(capacity),
            free_lock: Mutex::new(()),
            free_cv: Condvar::new(),
            ready_lock: Mutex::new(()),
            ready_cv: Condvar::new(),
        }))
    }

    fn make_req_id(&self, slot: usize, generation: u32) -> u32 {
        debug_assert!(slot <= SLOT_MASK as usize);
        (generation << SLOT_BITS) | slot as u32
    }

    fn decode_req_id(&self, req_id: u32) -> Option<(usize, u32)> {
        let slot = (req_id & SLOT_MASK) as usize;
        let generation = req_id >> SLOT_BITS;
        (slot < self.slots.len() && generation != 0).then_some((slot, generation))
    }

    fn pop_free(&self) -> Result<usize> {
        if self.aborted.load(Ordering::Acquire) {
            return Err(anyhow!("InProcRingChannel aborted"));
        }
        if let Some(slot) = self.free.pop() {
            return Ok(slot);
        }
        let mut guard = self.free_lock.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if self.aborted.load(Ordering::Acquire) {
                return Err(anyhow!("InProcRingChannel aborted"));
            }
            if let Some(slot) = self.free.pop() {
                return Ok(slot);
            }
            guard = self.free_cv.wait(guard).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn push_free(&self, slot: usize) {
        let _guard = self.free_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.free.push(slot).expect("free queue overflow");
        self.free_cv.notify_one();
    }

    fn push_ready(&self, slot: usize) {
        let _guard = self.ready_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.ready.push(slot).expect("ready queue overflow");
        self.ready_cv.notify_one();
    }

    fn pop_ready(&self) -> Option<usize> {
        if self.spin_budget_us > 0 {
            let deadline = Instant::now() + Duration::from_micros(self.spin_budget_us);
            let mut iters: u32 = 0;
            loop {
                if self.aborted.load(Ordering::Acquire) {
                    return None;
                }
                if let Some(slot) = self.ready.pop() {
                    return Some(slot);
                }
                iters = iters.wrapping_add(1);
                if iters & 0xFF == 0 && Instant::now() >= deadline {
                    break;
                }
                std::hint::spin_loop();
            }
        }

        let mut guard = self.ready_lock.lock().ok()?;
        loop {
            if self.aborted.load(Ordering::Acquire) {
                return None;
            }
            if let Some(slot) = self.ready.pop() {
                return Some(slot);
            }
            guard = self.ready_cv.wait(guard).ok()?;
        }
    }

    fn enqueue(&self, req: DriverRequest, wants_response: bool) -> Result<(usize, u32)> {
        let slot_idx = self.pop_free()?;
        let slot = &self.slots[slot_idx];
        let req_id = {
            let mut inner = slot.inner.lock().unwrap_or_else(|e| e.into_inner());
            debug_assert_eq!(inner.state, SlotState::Free);
            let mut generation = inner.generation.wrapping_add(1);
            if generation == 0 {
                generation = 1;
            }
            inner.generation = generation;
            inner.ready_for_reuse();
            inner.state = SlotState::Queued;
            inner.wants_response = wants_response;
            inner.frame = Some(Box::new(pie_bridge::Frame {
                driver_id: req.driver_id as u32,
                payload: req.payload,
            }));
            slot.ready_generation.store(0, Ordering::Release);
            self.make_req_id(slot_idx, generation)
        };
        self.push_ready(slot_idx);
        Ok((slot_idx, req_id))
    }

    fn wait_response(&self, slot_idx: usize, generation: u32) -> Result<DriverResponse> {
        let slot = &self.slots[slot_idx];
        if self.spin_budget_us > 0 {
            let deadline = Instant::now() + Duration::from_micros(self.spin_budget_us);
            let mut iters: u32 = 0;
            loop {
                if slot.ready_generation.load(Ordering::Acquire) == generation {
                    break;
                }
                iters = iters.wrapping_add(1);
                if iters & 0xFF == 0 && Instant::now() >= deadline {
                    break;
                }
                std::hint::spin_loop();
            }
        }

        let result = {
            let mut inner = slot.inner.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if inner.generation == generation {
                    if let Some(result) = inner.result.take() {
                        inner.ready_for_reuse();
                        inner.state = SlotState::Free;
                        break result;
                    }
                }
                inner = slot.cv.wait(inner).unwrap_or_else(|e| e.into_inner());
            }
        };
        self.push_free(slot_idx);
        result
    }

    fn abort(&self) {
        if self.aborted.swap(true, Ordering::AcqRel) {
            return;
        }
        self.ready_cv.notify_all();
        self.free_cv.notify_all();
        for slot in &self.slots {
            let mut inner = slot.inner.lock().unwrap_or_else(|e| e.into_inner());
            if inner.state != SlotState::Free && inner.wants_response && inner.result.is_none() {
                let generation = inner.generation;
                inner.result = Some(Err(anyhow!("InProcRingChannel aborted")));
                inner.view = None;
                inner.frame = None;
                inner.state = SlotState::Responded;
                slot.ready_generation.store(generation, Ordering::Release);
                slot.cv.notify_all();
            }
        }
    }
}

impl RingSlotInner {
    fn ready_for_reuse(&mut self) {
        self.view = None;
        self.frame = None;
        self.result = None;
        self.wants_response = false;
    }
}

static RING_CTX_REGISTRY: Lazy<DashMap<usize, Box<Arc<InProcRingState>>>> = Lazy::new(DashMap::new);

pub struct InProcRingChannel {
    state: Arc<InProcRingState>,
}

impl Default for InProcRingChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl InProcRingChannel {
    pub fn new() -> Self {
        Self::with_capacity_and_spin_budget(DEFAULT_RING_CAPACITY, DEFAULT_SPIN_BUDGET_US)
            .expect("default InProcRingChannel capacity is valid")
    }

    pub fn with_capacity(capacity: usize) -> Result<Self> {
        Self::with_capacity_and_spin_budget(capacity, DEFAULT_SPIN_BUDGET_US)
    }

    pub fn with_capacity_and_spin_budget(capacity: usize, spin_budget_us: u64) -> Result<Self> {
        Ok(Self {
            state: InProcRingState::new(capacity, spin_budget_us)?,
        })
    }

    pub fn ffi_vtable(&self) -> InProcVTable {
        let boxed: Box<Arc<InProcRingState>> = Box::new(self.state.clone());
        let ctx_ptr = Box::as_ref(&boxed) as *const Arc<InProcRingState> as *mut c_void;
        RING_CTX_REGISTRY.insert(ctx_ptr as usize, boxed);
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
        RING_CTX_REGISTRY.remove(&(ctx as usize));
    }

    fn state_from_ctx(ctx: *mut c_void) -> Option<Arc<InProcRingState>> {
        RING_CTX_REGISTRY
            .get(&(ctx as usize))
            .map(|entry| (**entry.value()).clone())
    }

    fn submit_sync_for_state(
        state: &Arc<InProcRingState>,
        req: DriverRequest,
    ) -> Result<DriverResponse> {
        let (slot, req_id) = state.enqueue(req, true)?;
        let generation = req_id >> SLOT_BITS;
        state.wait_response(slot, generation)
    }
}

#[async_trait]
impl DriverChannel for InProcRingChannel {
    async fn submit(&self, req: DriverRequest) -> Result<DriverResponse> {
        let state = self.state.clone();
        match tokio::runtime::Handle::try_current() {
            Ok(handle) if handle.runtime_flavor() == RuntimeFlavor::MultiThread => {
                tokio::task::block_in_place(|| Self::submit_sync_for_state(&state, req))
            }
            Ok(_) => tokio::task::spawn_blocking(move || Self::submit_sync_for_state(&state, req))
                .await
                .map_err(|e| anyhow!("InProcRingChannel blocking submit task failed: {e}"))?,
            Err(_) => Self::submit_sync_for_state(&state, req),
        }
    }

    fn notify(&self, req: DriverRequest) -> Result<()> {
        self.state.enqueue(req, false).map(|_| ())
    }

    fn abort(&self) {
        self.state.abort();
    }
}

unsafe extern "C" fn vt_recv(
    ctx: *mut c_void,
    out_request: *mut *const PieFrameDesc,
    out_req_id: *mut u32,
) -> c_int {
    let Some(state) = InProcRingChannel::state_from_ctx(ctx) else {
        return -1;
    };

    loop {
        let Some(slot_idx) = state.pop_ready() else {
            return -1;
        };
        let slot = &state.slots[slot_idx];
        let (req_id, desc_ptr) = {
            let mut inner = match slot.inner.lock() {
                Ok(g) => g,
                Err(_) => return -1,
            };
            if inner.state != SlotState::Queued {
                continue;
            }
            let Some(frame) = inner.frame.as_ref() else {
                continue;
            };
            let view = pie_frame_view(frame);
            // SAFETY: the boxed frame is stored in the same slot and
            // kept alive until send_response clears the view first.
            let view_static: PieFrameView<'static> = unsafe { std::mem::transmute(view) };
            inner.view = Some(view_static);
            inner.state = SlotState::InDriver;
            let req_id = state.make_req_id(slot_idx, inner.generation);
            let desc_ptr = &inner.view.as_ref().unwrap().desc as *const PieFrameDesc;
            (req_id, desc_ptr)
        };

        unsafe {
            if !out_request.is_null() {
                *out_request = desc_ptr;
            }
            if !out_req_id.is_null() {
                *out_req_id = req_id;
            }
        }
        return 0;
    }
}

unsafe extern "C" fn vt_send_response(
    ctx: *mut c_void,
    req_id: u32,
    response: *const PieResponseFrameDesc,
) {
    let Some(state) = InProcRingChannel::state_from_ctx(ctx) else {
        return;
    };
    let Some((slot_idx, generation)) = state.decode_req_id(req_id) else {
        return;
    };
    let slot = &state.slots[slot_idx];

    let release_without_waiter = {
        let mut inner = match slot.inner.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        if inner.generation != generation || inner.state != SlotState::InDriver {
            return;
        }

        let wants_response = inner.wants_response;
        let result = if wants_response {
            if response.is_null() {
                Err(anyhow!("InProcRingChannel: null response desc"))
            } else {
                let desc_ref: &PieResponseFrameDesc = unsafe { &*response };
                let owned = __pie_response_frame_from_desc(desc_ref);
                Ok(DriverResponse {
                    aborted: owned.aborted,
                    payload: owned.payload,
                })
            }
        } else {
            Ok(DriverResponse {
                aborted: false,
                payload: pie_bridge::ResponsePayload::Status(pie_bridge::StatusResponse {
                    status: 0,
                }),
            })
        };

        inner.view = None;
        inner.frame = None;
        if wants_response {
            inner.result = Some(result);
            inner.state = SlotState::Responded;
            slot.ready_generation.store(generation, Ordering::Release);
            slot.cv.notify_one();
            false
        } else {
            inner.ready_for_reuse();
            inner.state = SlotState::Free;
            true
        }
    };

    if release_without_waiter {
        state.push_free(slot_idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pie_bridge::schema::{
        PIE_REQUEST_PAYLOAD_HEALTH, PIE_RESPONSE_PAYLOAD_STATUS, PieResponsePayloadDesc,
        PieStatusResponseDesc,
    };

    #[tokio::test]
    async fn health_round_trip_through_ring_vtable() {
        let channel = InProcRingChannel::with_capacity_and_spin_budget(4, 100).unwrap();
        let vt = channel.ffi_vtable();
        let ctx = vt.ctx as usize;
        let recv = vt.recv;
        let send_response = vt.send_response;

        let driver = std::thread::spawn(move || {
            let ctx = ctx as *mut c_void;
            let mut request_ptr: *const PieFrameDesc = ::core::ptr::null();
            let mut req_id = 0u32;
            let rc = unsafe { recv(ctx, &mut request_ptr, &mut req_id) };
            assert_eq!(rc, 0);
            assert!(!request_ptr.is_null());
            let frame = unsafe { &*request_ptr };
            assert_eq!(frame.driver_id, 7);
            assert_eq!(frame.payload.kind, PIE_REQUEST_PAYLOAD_HEALTH);

            let resp = PieResponseFrameDesc {
                driver_id: 7,
                aborted: 0,
                payload: PieResponsePayloadDesc {
                    kind: PIE_RESPONSE_PAYLOAD_STATUS,
                    forward: Default::default(),
                    status: PieStatusResponseDesc { status: 0 },
                },
            };
            unsafe { send_response(ctx, req_id, &resp) };
        });

        let response = channel
            .submit(DriverRequest {
                driver_id: 7,
                payload: pie_bridge::RequestPayload::Health,
            })
            .await
            .expect("submit");
        match response.payload {
            pie_bridge::ResponsePayload::Status(s) => assert_eq!(s.status, 0),
            _ => panic!("expected status response"),
        }
        driver.join().unwrap();
        unsafe { InProcRingChannel::release(vt.ctx) };
    }
}
