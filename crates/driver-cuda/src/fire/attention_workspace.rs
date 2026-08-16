//! Device/pinned scratch for FlashInfer's plan + dispatch path. Kernels see only
//! an [`AttentionWorkspaceView`]; the rest is scheduling sized by run-ahead
//! depth. One staging slot is claimed per step, reusable only after its upload
//! event retires — a pending upload is synced before hand-out, the fence that
//! makes reuse safe. Release is explicit ([`AttentionWorkspace::release`])
//! because `Drop` has no `&mut O` for [`StagingOps`]; it only asserts it ran.

use std::ffi::c_void;

use crate::bind::abi::AttentionWorkspaceView;

/// What the workspace asks of CUDA: pinned host memory, events, device buffers.
/// The event type is associated so the live side uses `cudaEvent_t` while tests
/// use ordinals.
pub trait StagingOps {
    /// The upload-fence event handle.
    type Event;

    /// `cudaMallocHost`, or `None` on failure.
    fn malloc_host(&mut self, bytes: usize) -> Option<*mut c_void>;
    /// `cudaFreeHost`.
    fn free_host(&mut self, ptr: *mut c_void);
    /// `cudaEventCreateWithFlags(cudaEventDisableTiming)`, or `None`.
    fn event_create(&mut self) -> Option<Self::Event>;
    /// `cudaEventDestroy`.
    fn event_destroy(&mut self, event: Self::Event);
    /// Wait on a fence; `false` is a CUDA failure. Fallible, not a panic: the
    /// calibration sweep probes fire shapes, so an unbindable shape is the
    /// probe's answer, and a probe that aborts the process cannot probe.
    #[must_use]
    fn event_synchronize(&mut self, event: &Self::Event) -> bool;
    /// Record a fence on `stream`; `false` is a CUDA failure. See
    /// [`Self::event_synchronize`] for why neither panics.
    #[must_use]
    fn event_record(&mut self, event: &Self::Event, stream: *mut c_void) -> bool;
    /// The device allocation behind each scratch buffer, or `None` on failure.
    fn alloc_device(&mut self, bytes: usize) -> Option<*mut c_void>;
    /// Release a device buffer.
    fn free_device(&mut self, ptr: *mut c_void);
}

/// The live [`StagingOps`]: raw `cudaMalloc`/`cudaMallocHost` and the event
/// quartet through cudarc's runtime. Raw rather than [`crate::device::Allocator`]
/// because the workspace allocates at boot and frees at teardown, never inside a
/// capture. The fence pair reports rather than panicking: a silent failure would
/// hand a slot to the host while the GPU still reads it — see
/// [`StagingOps::event_synchronize`].
#[derive(Debug, Default, Clone, Copy)]
pub struct LiveStagingOps;

impl StagingOps for LiveStagingOps {
    type Event = cudarc::runtime::sys::cudaEvent_t;

    fn malloc_host(&mut self, bytes: usize) -> Option<*mut c_void> {
        use cudarc::runtime::sys::{cudaError, cudaMallocHost};
        let mut p: *mut c_void = std::ptr::null_mut();
        let ok = unsafe { cudaMallocHost(&mut p, bytes) } == cudaError::cudaSuccess;
        (ok && !p.is_null()).then_some(p)
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)] // seam method; recorders share it
    fn free_host(&mut self, ptr: *mut c_void) {
        let _ = unsafe { cudarc::runtime::sys::cudaFreeHost(ptr) };
    }

    fn event_create(&mut self) -> Option<Self::Event> {
        use cudarc::runtime::sys::{cudaError, cudaEventCreateWithFlags, cudaEventDisableTiming};
        let mut e: Self::Event = std::ptr::null_mut();
        let ok = unsafe { cudaEventCreateWithFlags(&mut e, cudaEventDisableTiming) }
            == cudaError::cudaSuccess;
        ok.then_some(e)
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)] // seam method; recorders share it
    fn event_destroy(&mut self, event: Self::Event) {
        let _ = unsafe { cudarc::runtime::sys::cudaEventDestroy(event) };
    }

    fn event_synchronize(&mut self, event: &Self::Event) -> bool {
        use cudarc::runtime::sys::{cudaError, cudaEventSynchronize};
        let code = unsafe { cudaEventSynchronize(*event) };
        code == cudaError::cudaSuccess
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)] // seam method; recorders share it
    fn event_record(&mut self, event: &Self::Event, stream: *mut c_void) -> bool {
        use cudarc::runtime::sys::{cudaError, cudaEventRecord};
        let code = unsafe { cudaEventRecord(*event, stream.cast()) };
        code == cudaError::cudaSuccess
    }

    fn alloc_device(&mut self, bytes: usize) -> Option<*mut c_void> {
        use cudarc::runtime::sys::{cudaError, cudaMalloc};
        let mut p: *mut c_void = std::ptr::null_mut();
        let ok = unsafe { cudaMalloc(&mut p, bytes) } == cudaError::cudaSuccess;
        (ok && !p.is_null()).then_some(p)
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)] // seam method; recorders share it
    fn free_device(&mut self, ptr: *mut c_void) {
        let _ = unsafe { cudarc::runtime::sys::cudaFree(ptr) };
    }
}

/// Why an allocation or a plan-slot claim failed. Named separately because
/// recovery differs: a device alloc is a sizing problem, a pin is host memory
/// pressure, an event create is handle exhaustion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StagingError {
    /// A device scratch buffer could not be allocated.
    DeviceAllocFailed,
    /// `cudaMallocHost` refused a staging slot's pin.
    PinFailed,
    /// `cudaEventCreateWithFlags` refused a slot's fence event.
    EventCreateFailed,
    /// A fence would not record or retire. Its own variant, not a panic, so the
    /// calibration sweep can treat an unbindable shape as an answer.
    FenceFailed,
}

/// One staging slot: a pinned host block and the event fencing its upload.
struct PlanStaging<E> {
    host: *mut c_void,
    upload_done: Option<E>,
    upload_pending: bool,
}

impl<E> Default for PlanStaging<E> {
    fn default() -> Self {
        Self { host: std::ptr::null_mut(), upload_done: None, upload_pending: false }
    }
}

/// The owner of FlashInfer's plan/dispatch scratch. See the module docs.
pub struct AttentionWorkspace<E> {
    float_buf: *mut c_void,
    float_bytes: usize,
    int_buf: *mut c_void,
    int_bytes: usize,
    staging_bytes: usize,
    /// Slot 0 is pinned at [`allocate`](Self::allocate); the rest pin lazily on
    /// first rotation, so a non-rotating workspace holds only one pin.
    plan_staging: Vec<PlanStaging<E>>,
    active_plan_slot: usize,
    next_plan_slot: usize,
    released: bool,
}

impl<E> AttentionWorkspace<E> {
    /// Allocate the two device scratch buffers and the staging pool.
    /// `plan_staging_slots` is the run-ahead depth in steps; `0` clamps to one
    /// slot, since slot 0 is pinned here and the rotation takes a modulus. On
    /// failure, everything created before the failing call is released.
    pub fn allocate<O: StagingOps<Event = E>>(
        ops: &mut O,
        float_workspace_bytes: usize,
        int_workspace_bytes: usize,
        plan_staging_slots: usize,
    ) -> Result<Self, StagingError> {
        let slots = if plan_staging_slots == 0 { 1 } else { plan_staging_slots };
        let mut ws = Self {
            float_buf: std::ptr::null_mut(),
            float_bytes: 0,
            int_buf: std::ptr::null_mut(),
            int_bytes: 0,
            staging_bytes: 0,
            plan_staging: Vec::new(),
            active_plan_slot: 0,
            next_plan_slot: 0,
            released: false,
        };
        ws.plan_staging.resize_with(slots, PlanStaging::default);
        ws.float_buf = match ops.alloc_device(float_workspace_bytes) {
            Some(p) => p,
            None => {
                ws.released = true;
                return Err(StagingError::DeviceAllocFailed);
            }
        };
        ws.float_bytes = float_workspace_bytes;
        ws.int_buf = match ops.alloc_device(int_workspace_bytes) {
            Some(p) => p,
            None => {
                ops.free_device(ws.float_buf);
                ws.released = true;
                return Err(StagingError::DeviceAllocFailed);
            }
        };
        ws.int_bytes = int_workspace_bytes;
        ws.staging_bytes = int_workspace_bytes;
        if let Err(e) = ensure_plan_slot(ops, ws.staging_bytes, &mut ws.plan_staging[0]) {
            for staging in &mut ws.plan_staging {
                if let Some(ev) = staging.upload_done.take() {
                    ops.event_destroy(ev);
                }
                if !staging.host.is_null() {
                    ops.free_host(staging.host);
                    staging.host = std::ptr::null_mut();
                }
            }
            ops.free_device(ws.float_buf);
            ops.free_device(ws.int_buf);
            ws.released = true;
            return Err(e);
        }
        Ok(ws)
    }

    /// What a kernel is handed — named, not a conversion, so the crate boundary
    /// is legible at the call site.
    #[must_use]
    pub fn view(&self) -> AttentionWorkspaceView {
        AttentionWorkspaceView {
            float_buffer: self.float_buf,
            float_bytes: self.float_bytes,
            int_buffer: self.int_buf,
            int_bytes: self.int_bytes,
            page_locked_int: self.plan_staging[self.active_plan_slot].host,
        }
    }

    /// The split-KV accumulation scratch (device).
    #[must_use]
    pub fn float_buffer(&self) -> *mut c_void {
        self.float_buf
    }

    /// The scheduling-metadata scratch (device).
    #[must_use]
    pub fn int_buffer(&self) -> *mut c_void {
        self.int_buf
    }

    /// The active slot's pinned host block.
    #[must_use]
    pub fn page_locked_int(&self) -> *mut c_void {
        self.plan_staging[self.active_plan_slot].host
    }

    /// Bytes in [`Self::float_buffer`].
    #[must_use]
    pub fn float_bytes(&self) -> usize {
        self.float_bytes
    }

    /// Bytes in [`Self::int_buffer`].
    #[must_use]
    pub fn int_bytes(&self) -> usize {
        self.int_bytes
    }

    /// Claim the next staging slot for this step's plan writes. Rotates first,
    /// so on failure the rotation stays advanced; then lazily pins the slot and
    /// syncs its pending upload so reuse is fenced.
    pub fn begin_plan_update<O: StagingOps<Event = E>>(
        &mut self,
        ops: &mut O,
    ) -> Result<(), StagingError> {
        self.active_plan_slot = self.next_plan_slot;
        self.next_plan_slot = (self.next_plan_slot + 1) % self.plan_staging.len();
        let staging = &mut self.plan_staging[self.active_plan_slot];
        ensure_plan_slot(ops, self.staging_bytes, staging)?;
        if staging.upload_pending {
            let ev =
                staging.upload_done.as_ref().expect("a pending upload always has its fence event");
            if !ops.event_synchronize(ev) {
                return Err(StagingError::FenceFailed);
            }
            staging.upload_pending = false;
        }
        Ok(())
    }

    /// Record the active slot's upload fence on `stream` and mark it pending —
    /// the slot is the GPU's until the event retires.
    ///
    /// # Errors
    ///
    /// [`StagingError::FenceFailed`] when the fence will not record; the slot is
    /// left unpending, since nothing is owed to a fence that does not exist.
    pub fn end_plan_update<O: StagingOps<Event = E>>(
        &mut self,
        ops: &mut O,
        stream: *mut c_void,
    ) -> Result<(), StagingError> {
        let staging = &mut self.plan_staging[self.active_plan_slot];
        let ev = staging
            .upload_done
            .as_ref()
            .expect("end_plan_update on a slot begin_plan_update never staged");
        if !ops.event_record(ev, stream) {
            return Err(StagingError::FenceFailed);
        }
        staging.upload_pending = true;
        Ok(())
    }

    /// The destructor: per slot, sync a pending upload, destroy the event, free
    /// the pin; then release the device buffers.
    pub fn release<O: StagingOps<Event = E>>(&mut self, ops: &mut O) {
        if self.released {
            return;
        }
        for staging in &mut self.plan_staging {
            if staging.upload_pending {
                let ev = staging
                    .upload_done
                    .as_ref()
                    .expect("a pending upload always has its fence event");
                // Release cannot report, so a failed fence is ignored: leaking
                // the pin and event is worse than freeing after an unconfirmed fence.
                let _ = ops.event_synchronize(ev);
                staging.upload_pending = false;
            }
            if let Some(ev) = staging.upload_done.take() {
                ops.event_destroy(ev);
            }
            if !staging.host.is_null() {
                ops.free_host(staging.host);
                staging.host = std::ptr::null_mut();
            }
        }
        if !self.float_buf.is_null() {
            ops.free_device(self.float_buf);
            self.float_buf = std::ptr::null_mut();
        }
        if !self.int_buf.is_null() {
            ops.free_device(self.int_buf);
            self.int_buf = std::ptr::null_mut();
        }
        self.released = true;
    }
}

/// Pin the slot's host block and create its fence event if either is missing.
/// Host first, then event — the order decides what a failure leaves behind.
fn ensure_plan_slot<O: StagingOps>(
    ops: &mut O,
    staging_bytes: usize,
    slot: &mut PlanStaging<O::Event>,
) -> Result<(), StagingError> {
    if slot.host.is_null() && staging_bytes > 0 {
        slot.host = ops.malloc_host(staging_bytes).ok_or(StagingError::PinFailed)?;
    }
    if slot.upload_done.is_none() {
        slot.upload_done = Some(ops.event_create().ok_or(StagingError::EventCreateFailed)?);
    }
    Ok(())
}

impl<E> Drop for AttentionWorkspace<E> {
    /// A leaked workspace does not leak safely: a pending upload's slot would be
    /// reusable while the GPU still reads it. The obligation is [`Self::release`];
    /// this only checks it was met, and not while already panicking — a panic in
    /// a `Drop` running because of an earlier panic aborts the process.
    fn drop(&mut self) {
        debug_assert!(
            self.released || std::thread::panicking(),
            "AttentionWorkspace dropped without release(ops)"
        );
    }
}
