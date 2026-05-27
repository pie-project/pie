//! Experimental in-process channel backed by preallocated slots.
//!
//! This keeps the same C ABI vtable shape as [`super::InProcChannel`]:
//! the driver calls `recv(ctx, &request, &req_id)`, reads a borrowed
//! `PieFrameDesc`, then calls `send_response(ctx, req_id, response)`.
//! The difference is internal bookkeeping: requests live in fixed slots
//! and the driver receives slot ids through a bounded lock-free queue.
//!
//! Contract: many runtime submitters may enqueue concurrently, but the
//! intended driver side is a single consumer thread calling `recv`.

use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use crossbeam::queue::ArrayQueue;
use crossbeam::utils::CachePadded;
use tokio::runtime::RuntimeFlavor;

use super::{DriverChannel, DriverRequest, DriverResponse};
use pie_bridge::ffi::InProcVTable;
use pie_bridge::schema::{
    __pie_response_frame_from_desc, PieFrameDesc, PieFrameView, PieResponseFrameDesc,
    pie_frame_view,
};

const SLOT_BITS: u32 = 16;
const SLOT_MASK: u32 = (1 << SLOT_BITS) - 1;
const GENERATION_MASK: u32 = u32::MAX >> SLOT_BITS;

const SLOT_FREE: u8 = 0;
const SLOT_FILLING: u8 = 1;
const SLOT_QUEUED: u8 = 2;
const SLOT_IN_DRIVER: u8 = 3;
const SLOT_RESPONDING: u8 = 4;
const SLOT_RESPONDED: u8 = 5;
const SLOT_ABORTED_IN_DRIVER: u8 = 6;
const SLOT_ABORTED_DRIVER_DONE: u8 = 7;
const SLOT_ABORTED_WAITER_DONE: u8 = 8;

pub const DEFAULT_POLLING_CAPACITY: usize = 1024;
pub const DEFAULT_SPIN_BUDGET_US: u64 = 1_000;

struct PollingSlot {
    state: AtomicU8,
    generation: AtomicU32,
    response_generation: AtomicU32,
    wants_response: AtomicBool,
    // Drop order matters: view must disappear before the frame whose
    // buffers it points into.
    view: UnsafeCell<Option<PieFrameView<'static>>>,
    frame: UnsafeCell<Option<pie_bridge::Frame>>,
    result: UnsafeCell<Option<Result<DriverResponse>>>,
}

// SAFETY: slot ownership is governed by `state`:
// - submitter owns fields while FILLING
// - driver owns view creation while transitioning QUEUED -> IN_DRIVER
// - responder owns result publication after IN_DRIVER -> RESPONDING
// - waiter owns cleanup after observing response_generation
// During abort of an in-driver slot, the waiter may consume `result`
// while the driver still owns `view`/`frame`; the state machine keeps
// those field groups disjoint and recycles only after both sides finish.
// The view field is cleared before frame.
unsafe impl Send for PollingSlot {}
unsafe impl Sync for PollingSlot {}

impl PollingSlot {
    fn new() -> Self {
        Self {
            state: AtomicU8::new(SLOT_FREE),
            generation: AtomicU32::new(0),
            response_generation: AtomicU32::new(0),
            wants_response: AtomicBool::new(false),
            view: UnsafeCell::new(None),
            frame: UnsafeCell::new(None),
            result: UnsafeCell::new(None),
        }
    }

    #[inline(always)]
    unsafe fn clear_frame_view(&self) {
        unsafe {
            (*self.view.get()).take();
            (*self.frame.get()).take();
        }
    }

    #[inline(always)]
    unsafe fn clear_response(&self) {
        unsafe {
            (*self.result.get()).take();
        }
        self.clear_response_metadata();
    }

    #[inline(always)]
    fn clear_response_metadata(&self) {
        self.wants_response.store(false, Ordering::Release);
        self.response_generation.store(0, Ordering::Release);
    }

    #[inline(always)]
    unsafe fn reset_for_reuse(&self) {
        unsafe {
            self.clear_frame_view();
            self.clear_response();
        }
    }

    #[inline(always)]
    unsafe fn reset_after_result_consumed(&self) {
        unsafe {
            self.clear_frame_view();
        }
        self.clear_response_metadata();
    }

    #[inline(always)]
    unsafe fn take_result(&self) -> Result<DriverResponse> {
        unsafe { (*self.result.get()).take() }
            .unwrap_or_else(|| Err(anyhow!("InProcPollingChannel response slot empty")))
    }
}

struct InProcPollingState {
    aborted: AtomicBool,
    spin_budget_us: u64,
    slots: Vec<CachePadded<PollingSlot>>,
    free: ArrayQueue<usize>,
    ready: ArrayQueue<usize>,
}

impl InProcPollingState {
    fn new(capacity: usize, spin_budget_us: u64) -> Result<Arc<Self>> {
        if capacity == 0 || capacity > SLOT_MASK as usize {
            return Err(anyhow!(
                "InProcPollingChannel capacity must be in 1..={}",
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
            slots: (0..capacity)
                .map(|_| CachePadded::new(PollingSlot::new()))
                .collect(),
            free,
            ready: ArrayQueue::new(capacity),
        }))
    }

    fn make_req_id(&self, slot: usize, generation: u32) -> u32 {
        debug_assert!(slot <= SLOT_MASK as usize);
        ((generation & GENERATION_MASK) << SLOT_BITS) | slot as u32
    }

    fn decode_req_id(&self, req_id: u32) -> Option<(usize, u32)> {
        let slot = (req_id & SLOT_MASK) as usize;
        let generation = req_id >> SLOT_BITS;
        (slot < self.slots.len() && generation != 0).then_some((slot, generation))
    }

    #[inline(always)]
    fn spin_budget_expired(&self, deadline: &mut Option<Instant>) -> bool {
        match (self.spin_budget_us, *deadline) {
            (u64::MAX, _) => false,
            (0, _) => true,
            (_, Some(deadline)) => Instant::now() >= deadline,
            (us, None) => {
                *deadline = Some(Instant::now() + Duration::from_micros(us));
                false
            }
        }
    }

    #[inline(always)]
    fn spin_or_yield(&self, iters: &mut u32, deadline: &mut Option<Instant>) {
        if self.spin_budget_us == 0 {
            std::thread::yield_now();
            return;
        }
        *iters = iters.wrapping_add(1);
        if (*iters & 0xff) == 0 && self.spin_budget_expired(deadline) {
            std::thread::yield_now();
        } else {
            std::hint::spin_loop();
        }
    }

    fn pop_free(&self) -> Result<usize> {
        let mut deadline = None;
        let mut iters = 0u32;
        loop {
            if self.aborted.load(Ordering::Acquire) {
                return Err(anyhow!("InProcPollingChannel aborted"));
            }
            if let Some(slot) = self.free.pop() {
                return Ok(slot);
            }
            self.spin_or_yield(&mut iters, &mut deadline);
        }
    }

    fn push_free(&self, slot: usize) {
        self.free.push(slot).expect("free queue overflow");
    }

    fn push_ready(&self, slot: usize) {
        self.ready.push(slot).expect("ready queue overflow");
    }

    fn pop_ready(&self) -> Option<usize> {
        let mut deadline = None;
        let mut iters = 0u32;
        loop {
            if self.aborted.load(Ordering::Acquire) {
                return None;
            }
            if let Some(slot) = self.ready.pop() {
                return Some(slot);
            }
            self.spin_or_yield(&mut iters, &mut deadline);
        }
    }

    fn next_generation(slot: &PollingSlot) -> u32 {
        let mut generation = slot.generation.load(Ordering::Relaxed).wrapping_add(1);
        generation &= GENERATION_MASK;
        if generation == 0 {
            generation = 1;
        }
        slot.generation.store(generation, Ordering::Relaxed);
        generation
    }

    fn enqueue(&self, req: DriverRequest, wants_response: bool) -> Result<(usize, u32)> {
        let slot_idx = self.pop_free()?;
        let slot = &self.slots[slot_idx];
        let prev = slot.state.compare_exchange(
            SLOT_FREE,
            SLOT_FILLING,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        debug_assert!(prev.is_ok(), "free queue yielded non-free slot");
        if prev.is_err() {
            return Err(anyhow!(
                "InProcPollingChannel free queue returned busy slot"
            ));
        }

        let generation = Self::next_generation(slot);
        unsafe { slot.reset_for_reuse() };
        slot.wants_response.store(wants_response, Ordering::Release);
        unsafe {
            *slot.frame.get() = Some(pie_bridge::Frame {
                driver_id: req.driver_id as u32,
                payload: req.payload,
            });
        }

        slot.state.store(SLOT_QUEUED, Ordering::Release);
        let req_id = self.make_req_id(slot_idx, generation);

        if self.aborted.load(Ordering::Acquire) {
            if wants_response {
                if slot
                    .state
                    .compare_exchange(
                        SLOT_QUEUED,
                        SLOT_RESPONDING,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    unsafe {
                        *slot.result.get() = Some(Err(anyhow!("InProcPollingChannel aborted")));
                    }
                    slot.state.store(SLOT_RESPONDED, Ordering::Release);
                    slot.response_generation
                        .store(generation, Ordering::Release);
                    return Ok((slot_idx, req_id));
                }
                let observed = slot.state.load(Ordering::Acquire);
                if observed == SLOT_RESPONDING || observed == SLOT_RESPONDED {
                    return Ok((slot_idx, req_id));
                }
            } else {
                unsafe { slot.reset_for_reuse() };
                slot.state.store(SLOT_FREE, Ordering::Release);
                self.push_free(slot_idx);
            }
            return Err(anyhow!("InProcPollingChannel aborted"));
        }

        self.push_ready(slot_idx);
        Ok((slot_idx, req_id))
    }

    fn wait_response(&self, slot_idx: usize, generation: u32) -> Result<DriverResponse> {
        let slot = &self.slots[slot_idx];
        let mut deadline = None;
        let mut iters = 0u32;
        while slot.response_generation.load(Ordering::Acquire) != generation {
            self.spin_or_yield(&mut iters, &mut deadline);
        }

        let state = slot.state.load(Ordering::Acquire);
        let result = unsafe { slot.take_result() };
        if state == SLOT_ABORTED_IN_DRIVER || state == SLOT_ABORTED_DRIVER_DONE {
            return self.finish_waiter_after_abort(slot_idx, state, result);
        }

        unsafe { slot.reset_after_result_consumed() };
        slot.state.store(SLOT_FREE, Ordering::Release);
        self.push_free(slot_idx);
        result
    }

    #[cold]
    #[inline(never)]
    fn finish_waiter_after_abort(
        &self,
        slot_idx: usize,
        observed: u8,
        result: Result<DriverResponse>,
    ) -> Result<DriverResponse> {
        let slot = &self.slots[slot_idx];
        if observed == SLOT_ABORTED_IN_DRIVER {
            match slot.state.compare_exchange(
                SLOT_ABORTED_IN_DRIVER,
                SLOT_ABORTED_WAITER_DONE,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return result,
                Err(SLOT_ABORTED_DRIVER_DONE) => {
                    self.recycle_after_aborted_driver_and_waiter_done(slot_idx);
                    return result;
                }
                Err(_) => return result,
            }
        }

        self.recycle_after_aborted_driver_and_waiter_done(slot_idx);
        result
    }

    #[cold]
    #[inline(never)]
    fn recycle_after_aborted_driver_and_waiter_done(&self, slot_idx: usize) {
        let slot = &self.slots[slot_idx];
        slot.clear_response_metadata();
        slot.state.store(SLOT_FREE, Ordering::Release);
        self.push_free(slot_idx);
    }

    #[cold]
    #[inline(never)]
    fn finish_driver_after_abort(&self, slot_idx: usize) {
        let slot = &self.slots[slot_idx];
        unsafe { slot.clear_frame_view() };
        match slot.state.compare_exchange(
            SLOT_ABORTED_IN_DRIVER,
            SLOT_ABORTED_DRIVER_DONE,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {}
            Err(SLOT_ABORTED_WAITER_DONE) => {
                self.recycle_after_aborted_driver_and_waiter_done(slot_idx);
            }
            Err(_) => {}
        }
    }

    fn abort(&self) {
        if self.aborted.swap(true, Ordering::AcqRel) {
            return;
        }
        for slot in &self.slots {
            if !slot.wants_response.load(Ordering::Acquire) {
                continue;
            }
            let generation = slot.generation.load(Ordering::Acquire);
            if generation == 0 {
                continue;
            }

            if slot
                .state
                .compare_exchange(
                    SLOT_QUEUED,
                    SLOT_RESPONDING,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                unsafe {
                    *slot.result.get() = Some(Err(anyhow!("InProcPollingChannel aborted")));
                }
                slot.state.store(SLOT_RESPONDED, Ordering::Release);
                slot.response_generation
                    .store(generation, Ordering::Release);
                continue;
            }

            if slot
                .state
                .compare_exchange(
                    SLOT_IN_DRIVER,
                    SLOT_ABORTED_IN_DRIVER,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                unsafe {
                    *slot.result.get() = Some(Err(anyhow!("InProcPollingChannel aborted")));
                }
                slot.response_generation
                    .store(generation, Ordering::Release);
            }
        }
    }
}

pub struct InProcPollingChannel {
    state: Arc<InProcPollingState>,
}

impl Default for InProcPollingChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl InProcPollingChannel {
    pub fn new() -> Self {
        Self::with_capacity_and_spin_budget(DEFAULT_POLLING_CAPACITY, DEFAULT_SPIN_BUDGET_US)
            .expect("default InProcPollingChannel capacity is valid")
    }

    pub fn with_capacity(capacity: usize) -> Result<Self> {
        Self::with_capacity_and_spin_budget(capacity, DEFAULT_SPIN_BUDGET_US)
    }

    pub fn with_spin_budget(spin_budget_us: u64) -> Result<Self> {
        Self::with_capacity_and_spin_budget(DEFAULT_POLLING_CAPACITY, spin_budget_us)
    }

    pub fn with_capacity_and_spin_budget(capacity: usize, spin_budget_us: u64) -> Result<Self> {
        Ok(Self {
            state: InProcPollingState::new(capacity, spin_budget_us)?,
        })
    }

    pub fn ffi_vtable(&self) -> InProcVTable {
        let boxed: Box<Arc<InProcPollingState>> = Box::new(self.state.clone());
        let ctx_ptr = Box::into_raw(boxed) as *mut c_void;
        InProcVTable {
            recv: vt_recv,
            send_response: vt_send_response,
            ctx: ctx_ptr,
        }
    }

    /// # Safety
    /// `ctx` must have been produced by [`Self::ffi_vtable`] and not
    /// already released. No callback using this `ctx` may be running,
    /// and no callback may start after this function is called.
    pub unsafe fn release(ctx: *mut c_void) {
        if !ctx.is_null() {
            unsafe {
                drop(Box::from_raw(ctx as *mut Arc<InProcPollingState>));
            }
        }
    }

    unsafe fn state_from_ctx<'a>(ctx: *mut c_void) -> Option<&'a InProcPollingState> {
        if ctx.is_null() {
            None
        } else {
            unsafe { Some((&*(ctx as *const Arc<InProcPollingState>)).as_ref()) }
        }
    }

    pub fn submit_blocking(&self, req: DriverRequest) -> Result<DriverResponse> {
        Self::submit_sync_for_state(self.state.as_ref(), req)
    }

    fn submit_sync_for_state(
        state: &InProcPollingState,
        req: DriverRequest,
    ) -> Result<DriverResponse> {
        let (slot, req_id) = state.enqueue(req, true)?;
        let generation = req_id >> SLOT_BITS;
        state.wait_response(slot, generation)
    }
}

#[async_trait]
impl DriverChannel for InProcPollingChannel {
    async fn submit(&self, req: DriverRequest) -> Result<DriverResponse> {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) if handle.runtime_flavor() == RuntimeFlavor::MultiThread => {
                tokio::task::block_in_place(|| self.submit_blocking(req))
            }
            Ok(_) => {
                let state = self.state.clone();
                tokio::task::spawn_blocking(move || {
                    Self::submit_sync_for_state(state.as_ref(), req)
                })
                .await
                .map_err(|e| anyhow!("InProcPollingChannel blocking submit task failed: {e}"))?
            }
            Err(_) => self.submit_blocking(req),
        }
    }

    fn submit_sync(&self, req: DriverRequest) -> Result<DriverResponse> {
        self.submit_blocking(req)
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
    let Some(state) = (unsafe { InProcPollingChannel::state_from_ctx(ctx) }) else {
        return -1;
    };

    loop {
        let Some(slot_idx) = state.pop_ready() else {
            return -1;
        };
        let slot = &state.slots[slot_idx];
        if slot
            .state
            .compare_exchange(
                SLOT_QUEUED,
                SLOT_IN_DRIVER,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            continue;
        }

        let generation = slot.generation.load(Ordering::Acquire);
        let req_id = state.make_req_id(slot_idx, generation);
        let desc_ptr = unsafe {
            let Some(frame) = (*slot.frame.get()).as_ref() else {
                slot.state.store(SLOT_FREE, Ordering::Release);
                state.push_free(slot_idx);
                continue;
            };
            let view = pie_frame_view(frame);
            // SAFETY: the frame is stored in the same slot and
            // kept alive until send_response clears the view first.
            *slot.view.get() =
                Some(std::mem::transmute::<PieFrameView<'_>, PieFrameView<'static>>(view));
            &(*slot.view.get()).as_ref().unwrap().desc as *const PieFrameDesc
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
    let Some(state) = (unsafe { InProcPollingChannel::state_from_ctx(ctx) }) else {
        return;
    };
    let Some((slot_idx, generation)) = state.decode_req_id(req_id) else {
        return;
    };
    let slot = &state.slots[slot_idx];
    if slot.generation.load(Ordering::Acquire) != generation {
        return;
    }

    let observed = slot
        .state
        .compare_exchange(
            SLOT_IN_DRIVER,
            SLOT_RESPONDING,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .unwrap_or_else(|state| state);

    if observed != SLOT_IN_DRIVER {
        finish_unexpected_driver_response_state(state, slot_idx, observed);
        return;
    }

    let wants_response = slot.wants_response.load(Ordering::Acquire);
    if wants_response {
        let result = if response.is_null() {
            Err(anyhow!("InProcPollingChannel: null response desc"))
        } else {
            let desc_ref: &PieResponseFrameDesc = unsafe { &*response };
            let owned = __pie_response_frame_from_desc(desc_ref);
            Ok(DriverResponse {
                aborted: owned.aborted,
                payload: owned.payload,
            })
        };
        unsafe {
            *slot.result.get() = Some(result);
        }
        slot.state.store(SLOT_RESPONDED, Ordering::Release);
        slot.response_generation
            .store(generation, Ordering::Release);
    } else {
        unsafe { slot.reset_after_result_consumed() };
        slot.state.store(SLOT_FREE, Ordering::Release);
        state.push_free(slot_idx);
    }
}

#[cold]
#[inline(never)]
fn finish_unexpected_driver_response_state(
    state: &InProcPollingState,
    slot_idx: usize,
    observed: u8,
) {
    let slot = &state.slots[slot_idx];
    match observed {
        SLOT_ABORTED_IN_DRIVER => state.finish_driver_after_abort(slot_idx),
        SLOT_ABORTED_WAITER_DONE => {
            unsafe { slot.clear_frame_view() };
            state.recycle_after_aborted_driver_and_waiter_done(slot_idx);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::mpsc;

    use super::*;
    use pie_bridge::schema::{
        PIE_REQUEST_PAYLOAD_HEALTH, PIE_RESPONSE_PAYLOAD_STATUS, PieResponsePayloadDesc,
        PieStatusResponseDesc,
    };

    fn health_request(driver_id: usize) -> DriverRequest {
        DriverRequest {
            driver_id,
            payload: pie_bridge::RequestPayload::Health,
        }
    }

    fn status_code(resp: &DriverResponse) -> i32 {
        match &resp.payload {
            pie_bridge::ResponsePayload::Status(s) => s.status,
            _ => panic!("expected status response"),
        }
    }

    fn status_response(driver_id: u32, status: i32) -> PieResponseFrameDesc {
        PieResponseFrameDesc {
            driver_id,
            aborted: 0,
            payload: PieResponsePayloadDesc {
                kind: PIE_RESPONSE_PAYLOAD_STATUS,
                forward: Default::default(),
                status: PieStatusResponseDesc { status },
            },
        }
    }

    fn spawn_status_driver(vt: InProcVTable) -> std::thread::JoinHandle<usize> {
        let ctx = vt.ctx as usize;
        let recv = vt.recv;
        let send_response = vt.send_response;
        std::thread::spawn(move || {
            let ctx = ctx as *mut c_void;
            let mut handled = 0usize;
            loop {
                let mut request_ptr: *const PieFrameDesc = ::core::ptr::null();
                let mut req_id = 0u32;
                let rc = unsafe { recv(ctx, &mut request_ptr, &mut req_id) };
                if rc != 0 || request_ptr.is_null() {
                    break;
                }
                let frame = unsafe { &*request_ptr };
                let resp = PieResponseFrameDesc {
                    driver_id: frame.driver_id,
                    aborted: 0,
                    payload: PieResponsePayloadDesc {
                        kind: PIE_RESPONSE_PAYLOAD_STATUS,
                        forward: Default::default(),
                        status: PieStatusResponseDesc { status: 0 },
                    },
                };
                unsafe { send_response(ctx, req_id, &resp) };
                handled += 1;
            }
            handled
        })
    }

    fn spawn_echo_status_driver_for(
        vt: InProcVTable,
        expected: usize,
        yield_every: usize,
    ) -> std::thread::JoinHandle<usize> {
        let ctx = vt.ctx as usize;
        let recv = vt.recv;
        let send_response = vt.send_response;
        std::thread::spawn(move || {
            let ctx = ctx as *mut c_void;
            for handled in 0..expected {
                let mut request_ptr: *const PieFrameDesc = ::core::ptr::null();
                let mut req_id = 0u32;
                let rc = unsafe { recv(ctx, &mut request_ptr, &mut req_id) };
                assert_eq!(rc, 0, "driver recv failed at request {handled}");
                assert!(
                    !request_ptr.is_null(),
                    "driver recv returned null request at request {handled}"
                );
                let frame = unsafe { &*request_ptr };
                let resp = status_response(frame.driver_id, frame.driver_id as i32);
                unsafe { send_response(ctx, req_id, &resp) };
                if yield_every != 0 && handled % yield_every == 0 {
                    std::thread::yield_now();
                }
            }
            expected
        })
    }

    fn spawn_delayed_echo_status_driver(
        vt: InProcVTable,
        delay: Duration,
        handled: Arc<AtomicUsize>,
    ) -> std::thread::JoinHandle<()> {
        let ctx = vt.ctx as usize;
        let recv = vt.recv;
        let send_response = vt.send_response;
        std::thread::spawn(move || {
            let ctx = ctx as *mut c_void;
            loop {
                let mut request_ptr: *const PieFrameDesc = ::core::ptr::null();
                let mut req_id = 0u32;
                let rc = unsafe { recv(ctx, &mut request_ptr, &mut req_id) };
                if rc != 0 || request_ptr.is_null() {
                    break;
                }
                let frame = unsafe { &*request_ptr };
                if !delay.is_zero() {
                    std::thread::sleep(delay);
                }
                let resp = status_response(frame.driver_id, frame.driver_id as i32);
                unsafe { send_response(ctx, req_id, &resp) };
                handled.fetch_add(1, Ordering::Relaxed);
            }
        })
    }

    fn put_single_slot_in_driver(state: &InProcPollingState) -> (usize, u32) {
        let (slot_idx, req_id) = state.enqueue(health_request(1), true).unwrap();
        assert_eq!(state.pop_ready(), Some(slot_idx));
        let slot = &state.slots[slot_idx];
        slot.state
            .compare_exchange(
                SLOT_QUEUED,
                SLOT_IN_DRIVER,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .unwrap();
        (slot_idx, req_id >> SLOT_BITS)
    }

    #[tokio::test]
    async fn health_round_trip_through_polling_vtable() {
        let channel = InProcPollingChannel::with_capacity_and_spin_budget(4, 100).unwrap();
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

        let response = channel.submit(health_request(7)).await.expect("submit");
        match response.payload {
            pie_bridge::ResponsePayload::Status(s) => assert_eq!(s.status, 0),
            _ => panic!("expected status response"),
        }
        driver.join().unwrap();
        unsafe { InProcPollingChannel::release(ctx as *mut c_void) };
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_submitters_reuse_slots() {
        let channel =
            Arc::new(InProcPollingChannel::with_capacity_and_spin_budget(8, 100).unwrap());
        let vt = channel.ffi_vtable();
        let ctx = vt.ctx as usize;
        let driver = spawn_status_driver(vt);

        let completed = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();
        for i in 0..64usize {
            let channel = channel.clone();
            let completed = completed.clone();
            tasks.push(tokio::spawn(async move {
                let response = channel.submit(health_request(i)).await.expect("submit");
                match response.payload {
                    pie_bridge::ResponsePayload::Status(s) => assert_eq!(s.status, 0),
                    _ => panic!("expected status response"),
                }
                completed.fetch_add(1, Ordering::Relaxed);
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        channel.abort();
        assert_eq!(driver.join().unwrap(), 64);
        assert_eq!(completed.load(Ordering::Relaxed), 64);
        unsafe { InProcPollingChannel::release(ctx as *mut c_void) };
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn stress_concurrent_submitters_reuse_tiny_polling_channel() {
        const WORKERS: usize = 8;
        const ITERS: usize = 128;
        const TOTAL: usize = WORKERS * ITERS;

        let channel =
            Arc::new(InProcPollingChannel::with_capacity_and_spin_budget(4, 100).unwrap());
        let vt = channel.ffi_vtable();
        let ctx = vt.ctx as usize;
        let driver = spawn_echo_status_driver_for(vt, TOTAL, 17);

        let completed = Arc::new(AtomicUsize::new(0));
        let run = async {
            let mut tasks = Vec::new();
            for worker in 0..WORKERS {
                let channel = channel.clone();
                let completed = completed.clone();
                tasks.push(tokio::spawn(async move {
                    for iter in 0..ITERS {
                        let driver_id = worker * 10_000 + iter;
                        let response = channel
                            .submit(health_request(driver_id))
                            .await
                            .expect("submit");
                        assert_eq!(status_code(&response), driver_id as i32);
                        completed.fetch_add(1, Ordering::Relaxed);
                        if iter % 19 == 0 {
                            tokio::task::yield_now().await;
                        }
                    }
                }));
            }
            for task in tasks {
                task.await.unwrap();
            }
        };
        tokio::time::timeout(Duration::from_secs(10), run)
            .await
            .expect("stress submitters timed out");

        assert_eq!(driver.join().unwrap(), TOTAL);
        assert_eq!(completed.load(Ordering::Relaxed), TOTAL);
        unsafe { InProcPollingChannel::release(ctx as *mut c_void) };
    }

    #[test]
    fn stress_notify_reuses_slots_without_waiters() {
        const TOTAL: usize = 512;

        let channel = InProcPollingChannel::with_capacity_and_spin_budget(4, 100).unwrap();
        let vt = channel.ffi_vtable();
        let ctx = vt.ctx as usize;
        let driver = spawn_echo_status_driver_for(vt, TOTAL, 11);

        for i in 0..TOTAL {
            channel.notify(health_request(i)).expect("notify");
            if i % 23 == 0 {
                std::thread::yield_now();
            }
        }

        assert_eq!(driver.join().unwrap(), TOTAL);
        unsafe { InProcPollingChannel::release(ctx as *mut c_void) };
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn abort_wakes_pending_submit() {
        let channel =
            Arc::new(InProcPollingChannel::with_capacity_and_spin_budget(4, 100).unwrap());
        let pending = {
            let channel = channel.clone();
            tokio::spawn(async move { channel.submit(health_request(1)).await })
        };

        tokio::time::sleep(Duration::from_millis(10)).await;
        channel.abort();

        let result = tokio::time::timeout(Duration::from_secs(1), pending)
            .await
            .expect("pending submit timed out")
            .expect("submit task panicked");
        assert!(result.is_err());
    }

    #[test]
    fn stress_abort_while_driver_holds_descriptor_vtable() {
        const ITERS: usize = 64;

        for iter in 0..ITERS {
            let channel =
                Arc::new(InProcPollingChannel::with_capacity_and_spin_budget(2, 100).unwrap());
            let vt = channel.ffi_vtable();
            let ctx = vt.ctx as usize;
            let recv = vt.recv;
            let send_response = vt.send_response;
            let (ready_tx, ready_rx) = mpsc::channel();
            let (release_tx, release_rx) = mpsc::channel();

            let driver = std::thread::spawn(move || {
                let ctx = ctx as *mut c_void;
                let mut request_ptr: *const PieFrameDesc = ::core::ptr::null();
                let mut req_id = 0u32;
                let rc = unsafe { recv(ctx, &mut request_ptr, &mut req_id) };
                assert_eq!(rc, 0);
                assert!(!request_ptr.is_null());
                let frame = unsafe { &*request_ptr };
                let driver_id = frame.driver_id;
                ready_tx.send(driver_id).unwrap();
                release_rx.recv().unwrap();
                let resp = status_response(driver_id, driver_id as i32);
                unsafe { send_response(ctx, req_id, &resp) };
            });

            let submitter_channel = channel.clone();
            let submitter =
                std::thread::spawn(move || submitter_channel.submit_blocking(health_request(iter)));

            assert_eq!(
                ready_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
                iter as u32
            );
            channel.abort();

            if iter % 2 == 0 {
                let result = submitter.join().unwrap();
                assert!(result.is_err(), "submit unexpectedly succeeded after abort");
                release_tx.send(()).unwrap();
                driver.join().unwrap();
            } else {
                release_tx.send(()).unwrap();
                driver.join().unwrap();
                let result = submitter.join().unwrap();
                assert!(result.is_err(), "submit unexpectedly succeeded after abort");
            }

            unsafe { InProcPollingChannel::release(ctx as *mut c_void) };
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn stress_abort_settles_concurrent_submitters() {
        const WORKERS: usize = 8;
        const ITERS: usize = 128;

        let channel =
            Arc::new(InProcPollingChannel::with_capacity_and_spin_budget(8, 100).unwrap());
        let vt = channel.ffi_vtable();
        let ctx = vt.ctx as usize;
        let handled = Arc::new(AtomicUsize::new(0));
        let driver =
            spawn_delayed_echo_status_driver(vt, Duration::from_micros(50), handled.clone());
        let start = Arc::new(tokio::sync::Barrier::new(WORKERS + 1));
        let settled = Arc::new(AtomicUsize::new(0));

        let mut tasks = Vec::new();
        for worker in 0..WORKERS {
            let channel = channel.clone();
            let start = start.clone();
            let settled = settled.clone();
            tasks.push(tokio::spawn(async move {
                start.wait().await;
                for iter in 0..ITERS {
                    let driver_id = worker * 10_000 + iter;
                    match channel.submit(health_request(driver_id)).await {
                        Ok(response) => assert_eq!(status_code(&response), driver_id as i32),
                        Err(_) => break,
                    }
                    if iter % 7 == 0 {
                        tokio::task::yield_now().await;
                    }
                }
                settled.fetch_add(1, Ordering::Relaxed);
            }));
        }

        start.wait().await;
        tokio::time::sleep(Duration::from_millis(5)).await;
        channel.abort();

        let wait_all = async {
            for task in tasks {
                task.await.unwrap();
            }
        };
        tokio::time::timeout(Duration::from_secs(10), wait_all)
            .await
            .expect("abort stress submitters timed out");
        driver.join().unwrap();

        assert_eq!(settled.load(Ordering::Relaxed), WORKERS);
        assert!(handled.load(Ordering::Relaxed) > 0);
        unsafe { InProcPollingChannel::release(ctx as *mut c_void) };
    }

    #[test]
    fn aborted_in_driver_waiter_first_recycles_after_driver_done() {
        let state = InProcPollingState::new(1, 100).unwrap();
        let (slot_idx, generation) = put_single_slot_in_driver(&state);
        let slot = &state.slots[slot_idx];

        state.abort();
        let result = state.wait_response(slot_idx, generation);
        assert!(result.is_err());
        assert_eq!(slot.state.load(Ordering::Acquire), SLOT_ABORTED_WAITER_DONE);

        state.finish_driver_after_abort(slot_idx);
        assert_eq!(slot.state.load(Ordering::Acquire), SLOT_FREE);
        assert_eq!(state.free.pop(), Some(slot_idx));
    }

    #[test]
    fn aborted_in_driver_driver_first_recycles_after_waiter_done() {
        let state = InProcPollingState::new(1, 100).unwrap();
        let (slot_idx, generation) = put_single_slot_in_driver(&state);
        let slot = &state.slots[slot_idx];

        state.abort();
        state.finish_driver_after_abort(slot_idx);
        assert_eq!(slot.state.load(Ordering::Acquire), SLOT_ABORTED_DRIVER_DONE);

        let result = state.wait_response(slot_idx, generation);
        assert!(result.is_err());
        assert_eq!(slot.state.load(Ordering::Acquire), SLOT_FREE);
        assert_eq!(state.free.pop(), Some(slot_idx));
    }
}
