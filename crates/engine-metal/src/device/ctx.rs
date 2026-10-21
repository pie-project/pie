#[cfg(target_vendor = "apple")]
use std::cell::RefCell;

use crate::error::{Fault, Result};

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2::runtime::ProtocolObject;
#[cfg(target_vendor = "apple")]
use objc2_foundation::NSString;
#[cfg(target_vendor = "apple")]
use objc2_metal::{
    MTLBarrierScope, MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLDispatchType, MTLGPUFamily, MTLResourceOptions,
    MTLSize,
};

static RESERVATIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[must_use]
pub fn reservations() -> u64 {
    RESERVATIONS.load(std::sync::atomic::Ordering::Relaxed)
}

#[must_use]
pub fn present() -> bool {
    #[cfg(target_vendor = "apple")]
    {
        MTLCreateSystemDefaultDevice().is_some()
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        false
    }
}

#[cfg(target_vendor = "apple")]
fn family(device: &ProtocolObject<dyn MTLDevice>) -> u32 {
    const NEWEST_FIRST: [(MTLGPUFamily, u32); 4] = [
        (MTLGPUFamily::Apple10, 10),
        (MTLGPUFamily::Apple9, 9),
        (MTLGPUFamily::Apple8, 8),
        (MTLGPUFamily::Apple7, 7),
    ];
    for (family, number) in NEWEST_FIRST {
        if device.supportsFamily(family) {
            return number;
        }
    }
    0
}

#[cfg(target_vendor = "apple")]
type Device = Retained<ProtocolObject<dyn MTLDevice>>;
#[cfg(not(target_vendor = "apple"))]
type Device = ();

#[cfg(target_vendor = "apple")]
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
#[cfg(not(target_vendor = "apple"))]
type Queue = ();

/// Where a GPU core count came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoresFrom {
    /// The IORegistry's `gpu-core-count`: macOS.
    Registry,
    /// The checked-in model table, by this model identifier.
    Model(&'static str),
}

impl CoresFrom {
    /// Its value of the `cores_from` fact.
    #[must_use]
    pub fn word(self) -> &'static str {
        match self {
            CoresFrom::Registry => "registry",
            CoresFrom::Model(_) => "model",
        }
    }
}

impl std::fmt::Display for CoresFrom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoresFrom::Registry => write!(f, "from the IORegistry"),
            CoresFrom::Model(id) => write!(f, "from the model table, as {id}"),
        }
    }
}

pub struct Context {
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    device: Device,
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    queue: Queue,
    name: String,
    working_set: u64,
    max_buffer: u64,
    /// The GPU's cores, where the IORegistry or the model table says.
    cores: Option<(u32, CoresFrom)>,
    /// The newest `MTLGPUFamily::AppleN` the device supports.
    family: u32,
    /// The Metal 4 mapping side, opened on first use: most loads never map a
    /// tile, and a second command queue is a scheduler context in the
    /// driver. `None` inside once opened means the device is not Metal 4.
    #[cfg(target_vendor = "apple")]
    sparse: std::cell::OnceCell<Option<std::sync::Arc<super::sparse::Sparse>>>,
}

// SAFETY: `MTLDevice` and `MTLCommandQueue` are documented thread-safe.
// `Send` only lets the boot thread hand the bound context to the lane thread.
unsafe impl Send for Context {}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("name", &self.name)
            .field("working_set", &self.working_set)
            .finish()
    }
}

impl Context {
    pub fn bind() -> Result<Context> {
        #[cfg(target_vendor = "apple")]
        {
            let device = MTLCreateSystemDefaultDevice().ok_or(Fault::Device {
                call: "MTLCreateSystemDefaultDevice",
                why: "this machine publishes no Metal device".to_string(),
            })?;
            if !device.hasUnifiedMemory() {
                return Err(Fault::Device {
                    call: "hasUnifiedMemory",
                    why: format!(
                        "`{}` does not share memory with the host, and this shell writes \
                         its buffers through `contents()`",
                        device.name()
                    ),
                });
            }
            let queue = device.newCommandQueue().ok_or(Fault::Device {
                call: "newCommandQueue",
                why: "the device would not open a command queue".to_string(),
            })?;
            let name = device.name().to_string();
            if !device.supportsFamily(MTLGPUFamily::Metal4) {
                return Err(Fault::Device {
                    call: "supportsFamily(Metal4)",
                    why: format!(
                        "`{name}` does not run Metal 4, and pie's Metal engine needs it: \
                         Apple Silicon on macOS 26 or newer"
                    ),
                });
            }
            let family = family(&device);
            let cores = super::iokit::gpu_cores().map(|n| (n, CoresFrom::Registry));
            #[cfg(target_os = "ios")]
            let cores = cores.or_else(|| {
                let (n, id) = super::models::gpu_cores(&super::models::machine()?, family)?;
                Some((n, CoresFrom::Model(id)))
            });
            let working_set = device.recommendedMaxWorkingSetSize();
            let max_buffer = device.maxBufferLength() as u64;
            Ok(Context {
                device,
                queue,
                name,
                working_set,
                max_buffer,
                cores,
                family,
                sparse: std::cell::OnceCell::new(),
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(Fault::Deviceless)
        }
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub fn working_set(&self) -> u64 {
        self.working_set
    }

    /// The working set other processes hold now: what of it the system
    /// could not hand this one (app memory, wired and compressed pages; the
    /// file cache is counted as handed). Measured before a load wires anything,
    /// it is what `device_memory_share` yields to, as on CUDA. 0
    /// where the system does not say.
    #[must_use]
    pub fn held_elsewhere(&self) -> u64 {
        available_memory().map_or(0, |available| {
            self.working_set
                .saturating_sub(available.min(self.working_set))
        })
    }

    #[must_use]
    pub fn physical_memory() -> u64 {
        #[cfg(target_vendor = "apple")]
        {
            let mut size: u64 = 0;
            let mut len = std::mem::size_of::<u64>();
            // SAFETY: `hw.memsize` is a u64 sysctl; `len` states the out
            // buffer's size and the call writes at most that many bytes.
            let rc = unsafe {
                libc::sysctlbyname(
                    c"hw.memsize".as_ptr(),
                    (&raw mut size).cast(),
                    &raw mut len,
                    std::ptr::null_mut(),
                    0,
                )
            };
            if rc == 0 { size } else { 0 }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            0
        }
    }

    #[must_use]
    pub fn max_buffer(&self) -> u64 {
        self.max_buffer
    }

    #[must_use]
    pub fn cores(&self) -> Option<u32> {
        self.cores.map(|(n, _)| n)
    }

    #[must_use]
    pub fn cores_from(&self) -> Option<CoresFrom> {
        self.cores.map(|(_, from)| from)
    }

    #[must_use]
    pub fn family(&self) -> u32 {
        self.family
    }

    pub fn bind_thread(&self) -> Result<()> {
        Ok(())
    }

    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    pub(crate) fn reserve(&self, bytes: u64) -> Result<super::alloc::Slab> {
        RESERVATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        #[cfg(target_vendor = "apple")]
        {
            if bytes > self.max_buffer {
                return Err(Fault::Ceiling {
                    what: "bytes in one buffer",
                    need: bytes,
                    have: self.max_buffer,
                });
            }
            let len = usize::try_from(bytes).map_err(|_| Fault::Ceiling {
                what: "bytes in one buffer",
                need: bytes,
                have: self.max_buffer,
            })?;
            self.device
                .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
                .ok_or(Fault::Device {
                    call: "newBufferWithLength:options:",
                    why: format!("the device declined {bytes} bytes of shared storage"),
                })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = bytes;
            Err(Fault::Deviceless)
        }
    }

    #[cfg(target_vendor = "apple")]
    pub(crate) unsafe fn no_copy(
        &self,
        at: std::ptr::NonNull<u8>,
        span: usize,
    ) -> Result<super::alloc::Slab> {
        RESERVATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let bytes = span as u64;
        if bytes > self.max_buffer {
            return Err(Fault::Ceiling {
                what: "bytes in one buffer",
                need: bytes,
                have: self.max_buffer,
            });
        }
        // SAFETY: the caller's contract is exactly this call's — an aligned
        // live mapping of `span` readable bytes that outlives the buffer —
        // and a nil deallocator is what leaves the pages theirs.
        unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    at.cast::<std::ffi::c_void>(),
                    span,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
                .ok_or(Fault::Device {
                    call: "newBufferWithBytesNoCopy:length:options:deallocator:",
                    why: format!("the device declined a zero-copy wrap of {bytes} bytes"),
                })
        }
    }

    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    pub(crate) fn empty(&self) -> super::alloc::Slab {
        RESERVATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        #[cfg(target_vendor = "apple")]
        {
            self.device
                .newBufferWithLength_options(1, MTLResourceOptions::StorageModeShared)
                .expect("one byte of shared storage")
        }
        #[cfg(not(target_vendor = "apple"))]
        {}
    }

    #[cfg(target_vendor = "apple")]
    pub(crate) fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Whether this device can map sparse buffer tiles: `MTLGPUFamilyMetal4`.
    #[must_use]
    pub fn supports_elastic(&self) -> bool {
        #[cfg(target_vendor = "apple")]
        {
            super::sparse::Sparse::supported(&self.device)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            false
        }
    }

    /// The mapping side, opened on first ask; `None` when the device is not
    /// Metal 4. Its residency set is attached to this context's frame queue.
    #[cfg(target_vendor = "apple")]
    pub fn sparse(&self) -> Result<Option<std::sync::Arc<super::sparse::Sparse>>> {
        if let Some(opened) = self.sparse.get() {
            return Ok(opened.clone());
        }
        let opened = super::sparse::Sparse::open(&self.device, &self.queue)?;
        let _ = self.sparse.set(opened.clone());
        Ok(opened)
    }

    pub fn frame(&self) -> Result<Frame> {
        self.open_frame(false)
    }

    /// A frame whose launches encoded through [`Frame::tracked`] run in a
    /// concurrent pass, each waiting only on the launches whose bytes it
    /// shares (see [`crate::hazard`]); what is encoded through
    /// [`Frame::encoder`] runs in a serial pass of its own, after all of
    /// them and before any that follow. The `serial` diagnostic hands out a
    /// serial frame instead.
    pub fn concurrent_frame(&self) -> Result<Frame> {
        self.open_frame(!crate::diag::on().serial)
    }

    #[cfg_attr(not(target_vendor = "apple"), allow(unused_variables))]
    fn open_frame(&self, concurrent: bool) -> Result<Frame> {
        #[cfg(target_vendor = "apple")]
        {
            let buffer = self.queue.commandBuffer().ok_or(Fault::Device {
                call: "commandBuffer",
                why: "the queue would not open a command buffer".to_string(),
            })?;
            let encoder = compute_pass(&buffer, concurrent).ok_or(Fault::Device {
                call: "computeCommandEncoder",
                why: "the command buffer would not open a compute pass".to_string(),
            })?;
            Ok(Frame {
                buffer: RefCell::new(buffer),
                encoder: RefCell::new(Some(encoder)),
                blit: None,
                committed: RefCell::new(Vec::new()),
                flushed: RefCell::new(Vec::new()),
                concurrent,
                parallel: std::cell::Cell::new(concurrent),
                hazards: RefCell::new(crate::hazard::Hazards::new()),
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(Fault::Deviceless)
        }
    }
}

/// A compute pass on `buffer`, its dispatches concurrent or serial.
#[cfg(target_vendor = "apple")]
fn compute_pass(
    buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    concurrent: bool,
) -> Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
    if concurrent {
        buffer.computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
    } else {
        buffer.computeCommandEncoder()
    }
}

static LAUNCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BARRIERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SWITCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Since the process began: launches encoded into concurrent passes, the
/// barriers placed ahead of them, and the passes a concurrent frame opened
/// to switch between its concurrent and serial kinds.
#[must_use]
pub fn barrier_census() -> (u64, u64, u64) {
    (
        LAUNCHES.load(std::sync::atomic::Ordering::Relaxed),
        BARRIERS.load(std::sync::atomic::Ordering::Relaxed),
        SWITCHES.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// One command buffer and the compute pass open on it. A fence (see
/// [`Frame::fence`]) closes the pass, encodes an event signal or wait on
/// the buffer, and opens the next pass. The cells let a fence happen
/// through a shared `&Frame`, which is all a live encode holds.
pub struct Frame {
    #[cfg(target_vendor = "apple")]
    buffer: RefCell<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    #[cfg(not(target_vendor = "apple"))]
    #[allow(dead_code)]
    buffer: (),
    #[cfg(target_vendor = "apple")]
    encoder: RefCell<Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>>,
    #[cfg(not(target_vendor = "apple"))]
    encoder: Option<()>,
    #[cfg(target_vendor = "apple")]
    blit: Option<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>>,
    #[cfg(not(target_vendor = "apple"))]
    #[allow(dead_code)]
    blit: Option<()>,
    /// What runs once the buffer is committed, in the order it was added.
    #[cfg(target_vendor = "apple")]
    committed: RefCell<Vec<Box<dyn FnOnce() + Send>>>,
    /// The buffers [`Frame::flush`] committed ahead of the one open now, in
    /// the order the queue runs them.
    #[cfg(target_vendor = "apple")]
    flushed: RefCell<Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
    /// Whether this frame runs tracked launches in concurrent passes.
    #[cfg(target_vendor = "apple")]
    concurrent: bool,
    /// Whether the open pass is a concurrent one, ordered only by the
    /// barriers [`Frame::tracked`] places.
    #[cfg(target_vendor = "apple")]
    parallel: std::cell::Cell<bool>,
    /// What the launches since the open pass's last barrier touched.
    #[cfg(target_vendor = "apple")]
    hazards: RefCell<crate::hazard::Hazards>,
}

// SAFETY: a `Frame` is created, encoded into and committed on one thread —
// the lane thread. `Send` is what lets a `Shell` holding one move.
unsafe impl Send for Frame {}

impl Frame {
    /// The open compute pass, retained: a fence may replace it while the
    /// caller still holds this one, so nothing borrows the cell for long.
    ///
    /// The pass is a serial one: what the caller encodes on it touches
    /// bytes nobody listed, as many launches as it likes, so on a concurrent
    /// frame an open concurrent pass ends here and a serial one opens. A
    /// pass boundary orders everything on its two sides.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn encoder(&self) -> Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> {
        if self.parallel.get() {
            self.switch(false);
        }
        self.pass()
    }

    /// The open compute pass, for one launch touching exactly `accesses`.
    /// On a concurrent frame the pass is a concurrent one, and a barrier
    /// goes ahead of the launch when one of them meets bytes a launch since
    /// the last barrier wrote, or writes bytes one read.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn tracked(
        &self,
        accesses: &[crate::hazard::Access],
    ) -> Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> {
        if !self.concurrent {
            return self.pass();
        }
        if !self.parallel.get() {
            self.switch(true);
        }
        let encoder = self.pass();
        let owed = self.hazards.borrow_mut().admit(accesses);
        LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if owed {
            BARRIERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
        encoder
    }

    /// The open compute pass, concurrent on a concurrent frame, for one
    /// indirect command buffer whose commands carry their own barriers;
    /// the launch after it waits for all of them.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn indirect(&self) -> Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> {
        if !self.concurrent {
            return self.pass();
        }
        if !self.parallel.get() {
            self.switch(true);
        }
        let encoder = self.pass();
        if self.hazards.borrow_mut().opaque() {
            encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
        encoder
    }

    /// Whether this frame's passes run their dispatches concurrently.
    #[cfg(target_vendor = "apple")]
    #[must_use]
    pub fn concurrent(&self) -> bool {
        self.concurrent
    }

    /// Launches this frame admitted to its concurrent passes, and the
    /// barriers it placed ahead of them.
    #[cfg(target_vendor = "apple")]
    #[must_use]
    pub fn census(&self) -> (u64, u64) {
        self.hazards.borrow().census()
    }

    #[cfg(target_vendor = "apple")]
    fn pass(&self) -> Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> {
        self.encoder
            .borrow()
            .clone()
            .expect("the pass is open until `commit` closes it")
    }

    /// Ends the open compute pass and opens one of the other kind on the
    /// same buffer.
    #[cfg(target_vendor = "apple")]
    fn switch(&self, parallel: bool) {
        let ended = self
            .encoder
            .borrow_mut()
            .take()
            .expect("the pass is open until `commit` closes it");
        ended.endEncoding();
        SWITCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let opened = self
            .open_pass(&self.buffer.borrow(), parallel)
            .expect("the command buffer opens a compute pass");
        *self.encoder.borrow_mut() = Some(opened);
    }

    /// A fresh compute pass on `buffer`, concurrent or serial; nothing in a
    /// pass before it overlaps it, so the hazard sets start empty.
    #[cfg(target_vendor = "apple")]
    fn open_pass(
        &self,
        buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        parallel: bool,
    ) -> Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
        self.hazards.borrow_mut().reset();
        let parallel = parallel && self.concurrent;
        self.parallel.set(parallel);
        compute_pass(buffer, parallel)
    }

    /// Has `then` run once this frame's buffer is committed; a frame dropped
    /// uncommitted never runs it.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn on_commit(&self, then: Box<dyn FnOnce() + Send>) {
        self.committed.borrow_mut().push(then);
    }

    /// Commits what is encoded so far and opens a fresh buffer on the same
    /// queue for the rest, so the device starts on the head of a long fire
    /// while its tail is still being encoded. The queue runs the two in
    /// order and the buffers' resources are tracked, so nothing reads ahead
    /// of what it waits for. Refused (`false`) while a blit pass is open.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn flush(&self) -> Result<bool> {
        if self.blit.is_some() {
            return Ok(false);
        }
        let head = self.buffer.borrow().clone();
        let tail = head.commandQueue().commandBuffer().ok_or(Fault::Device {
            call: "commandBuffer",
            why: "the queue would not open the buffer a flushed frame goes on in".to_string(),
        })?;
        if let Some(encoder) = self.encoder.borrow_mut().take() {
            encoder.endEncoding();
        }
        let encoder = self
            .open_pass(&tail, self.parallel.get())
            .ok_or(Fault::Device {
                call: "computeCommandEncoder",
                why: "the command buffer would not open a compute pass".to_string(),
            })?;
        head.commit();
        for then in self.committed.borrow_mut().drain(..) {
            then();
        }
        *self.buffer.borrow_mut() = tail;
        *self.encoder.borrow_mut() = Some(encoder);
        self.flushed.borrow_mut().push(head);
        Ok(true)
    }

    #[cfg(target_vendor = "apple")]
    fn commit_buffer(&mut self) -> Retained<ProtocolObject<dyn MTLCommandBuffer>> {
        self.end_pass();
        let buffer = self.buffer.get_mut().clone();
        buffer.commit();
        for then in self.committed.get_mut().drain(..) {
            then();
        }
        buffer
    }

    #[cfg(target_vendor = "apple")]
    fn end_pass(&mut self) {
        if let Some(encoder) = self.encoder.get_mut().take() {
            encoder.endEncoding();
        }
        if let Some(blit) = self.blit.take() {
            blit.endEncoding();
        }
    }

    #[cfg(target_vendor = "apple")]
    pub(crate) fn next_pass(&mut self) -> Result<&ProtocolObject<dyn MTLComputeCommandEncoder>> {
        self.end_pass();
        let buffer = self.buffer.get_mut().clone();
        // A serial pass: the caller may encode on the pass it is handed.
        let encoder = self.open_pass(&buffer, false).ok_or(Fault::Device {
            call: "computeCommandEncoder",
            why: "the command buffer would not open a second compute pass".to_string(),
        })?;
        *self.encoder.get_mut() = Some(encoder);
        Ok(self.encoder.get_mut().as_deref().expect("just opened"))
    }

    /// Ends the pass here and opens the next, with an event between them
    /// in the same command buffer. With `signal`, the GPU signals `event`
    /// to `value` once the work so far is done; without, it waits for
    /// `event` to reach `value` before the work after. Either way the GPU
    /// work on both sides stays ordered, and the other engine runs between.
    ///
    /// The buffer stays one: closing it at each fence and chaining the next
    /// on the queue was seen to cost ~200 ms at a boundary every so often,
    /// the GPU starting the scheduled buffer late, which a long prefill
    /// paid a dozen times over.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn fence(
        &self,
        event: &ProtocolObject<dyn objc2_metal::MTLEvent>,
        value: u64,
        signal: bool,
    ) -> Result<()> {
        if self.blit.is_some() {
            return Err(Fault::Device {
                call: "fence",
                why: "a blit pass is open where a compute pass was expected".to_string(),
            });
        }
        if let Some(encoder) = self.encoder.borrow_mut().take() {
            encoder.endEncoding();
        }
        let current = self.buffer.borrow().clone();
        if signal {
            current.encodeSignalEvent_value(event, value);
        } else {
            current.encodeWaitForEvent_value(event, value);
        }
        let reopened = self
            .open_pass(&self.buffer.borrow(), self.parallel.get())
            .ok_or(Fault::Device {
                call: "computeCommandEncoder",
                why: "the command buffer would not reopen a compute pass after a fence".to_string(),
            })?;
        *self.encoder.borrow_mut() = Some(reopened);
        Ok(())
    }

    #[cfg_attr(not(target_vendor = "apple"), allow(unused_variables))]
    #[cfg_attr(not(target_vendor = "apple"), allow(unused_variables))]
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    pub(crate) fn fill(&mut self, slab: &super::alloc::Slab, at: u64, len: u64) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            if len == 0 {
                return Ok(());
            }
            if self.blit.is_none() {
                if let Some(encoder) = self.encoder.get_mut().take() {
                    encoder.endEncoding();
                }
                self.blit = Some(self.buffer.get_mut().blitCommandEncoder().ok_or(
                    Fault::Device {
                        call: "blitCommandEncoder",
                        why: "the command buffer would not open a blit pass".to_string(),
                    },
                )?);
            }
            let blit = self.blit.as_deref().expect("just opened");
            blit.fillBuffer_range_value(
                slab,
                objc2_foundation::NSRange {
                    location: at as usize,
                    length: len as usize,
                },
                0,
            );
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(Fault::Deviceless)
        }
    }

    #[cfg_attr(not(target_vendor = "apple"), allow(unused_variables))]
    pub(crate) fn fill_zero(
        &mut self,
        into: &super::alloc::Slab,
        into_at: u64,
        len: u64,
    ) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            if len == 0 {
                return Ok(());
            }
            if self.blit.is_none() {
                if let Some(encoder) = self.encoder.get_mut().take() {
                    encoder.endEncoding();
                }
                self.blit = Some(self.buffer.get_mut().blitCommandEncoder().ok_or(
                    Fault::Device {
                        call: "blitCommandEncoder",
                        why: "the command buffer would not open a blit pass".to_string(),
                    },
                )?);
            }
            let blit = self.blit.as_deref().expect("just opened");
            blit.fillBuffer_range_value(
                into,
                objc2_foundation::NSRange::new(into_at as usize, len as usize),
                0,
            );
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(Fault::Deviceless)
        }
    }

    /// Blit `len` bytes from `source_at` of `source` to `into_at` of
    /// `into`, both spans checked against their buffers first.
    pub fn copy_span(
        &mut self,
        source: &super::Buffer,
        source_at: u64,
        into: &super::Buffer,
        into_at: u64,
        len: u64,
    ) -> Result<()> {
        source.span(source_at, len)?;
        into.span(into_at, len)?;
        self.copy(source.slab(), source_at, into.slab(), into_at, len)
    }

    #[cfg_attr(not(target_vendor = "apple"), allow(unused_variables))]
    pub(crate) fn copy(
        &mut self,
        source: &super::alloc::Slab,
        source_at: u64,
        into: &super::alloc::Slab,
        into_at: u64,
        len: u64,
    ) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            if len == 0 {
                return Ok(());
            }
            if self.blit.is_none() {
                if let Some(encoder) = self.encoder.get_mut().take() {
                    encoder.endEncoding();
                }
                self.blit = Some(self.buffer.get_mut().blitCommandEncoder().ok_or(
                    Fault::Device {
                        call: "blitCommandEncoder",
                        why: "the command buffer would not open a blit pass".to_string(),
                    },
                )?);
            }
            let blit = self.blit.as_deref().expect("just opened");
            // SAFETY: both spans were bounds-checked by `Buffer::span`, and
            // both buffers outlive the command buffer (owned by the shell
            // for the life of the load, or retained via the handle row).
            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    source,
                    source_at as usize,
                    into,
                    into_at as usize,
                    len as usize,
                );
            }
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(Fault::Deviceless)
        }
    }

    pub fn commit(mut self) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            let buffer = self.commit_buffer();
            buffer.waitUntilCompleted();
            for buffer in self.flushed.get_mut().iter().chain([&buffer]) {
                if let Some(error) = buffer.error() {
                    return Err(Fault::Device {
                        call: "waitUntilCompleted",
                        why: error.localizedDescription().to_string(),
                    });
                }
            }
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = self.encoder.take();
            Err(Fault::Deviceless)
        }
    }

    pub fn commit_timed(mut self) -> Result<f64> {
        #[cfg(target_vendor = "apple")]
        {
            let buffer = self.commit_buffer();
            buffer.waitUntilCompleted();
            for buffer in self.flushed.get_mut().iter().chain([&buffer]) {
                if let Some(error) = buffer.error() {
                    return Err(Fault::Device {
                        call: "waitUntilCompleted",
                        why: error.localizedDescription().to_string(),
                    });
                }
            }
            let start = self
                .flushed
                .get_mut()
                .first()
                .map_or_else(|| buffer.GPUStartTime(), |head| head.GPUStartTime());
            Ok(buffer.GPUEndTime() - start)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = self.encoder.take();
            Err(Fault::Deviceless)
        }
    }

    #[cfg_attr(not(target_vendor = "apple"), allow(unused_variables))]
    pub fn commit_async(
        mut self,
        on_done: Option<Box<dyn Fn(Option<String>) + Send + 'static>>,
    ) -> Result<Pending> {
        #[cfg(target_vendor = "apple")]
        {
            let flushed = std::mem::take(self.flushed.get_mut());
            if let Some(on_done) = on_done {
                let heads = Flushed(flushed.clone());
                let handler = block2::RcBlock::new(
                    move |buffer: core::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                        // SAFETY: Metal hands the handler a live reference to
                        // the command buffer it is about to retire, valid for
                        // the length of this call.
                        let buffer = unsafe { buffer.as_ref() };
                        // The queue retired every flushed head before this
                        // tail, so their verdicts are in.
                        let refused = heads
                            .0
                            .iter()
                            .find_map(|head| head.error())
                            .or_else(|| buffer.error());
                        on_done(refused.map(|error| error.localizedDescription().to_string()));
                    },
                );
                // SAFETY: `addCompletedHandler:` copies the block, so the
                // `RcBlock` may be dropped at the end of this scope; the
                // closure owns everything it touches and is `Send`.
                unsafe {
                    self.buffer
                        .get_mut()
                        .addCompletedHandler(block2::RcBlock::as_ptr(&handler));
                }
            }
            Ok(Pending {
                buffer: self.commit_buffer(),
                flushed,
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = self.encoder.take();
            Err(Fault::Deviceless)
        }
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        #[cfg(target_vendor = "apple")]
        self.end_pass();
    }
}

/// Flushed heads a completion handler reads the verdicts of.
#[cfg(target_vendor = "apple")]
struct Flushed(Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>);

// SAFETY: the handler only reads `error` off buffers already retired, which
// is documented safe from any thread.
#[cfg(target_vendor = "apple")]
unsafe impl Send for Flushed {}

// SAFETY: as for `Send`; nothing but `error` is read through a shared one.
#[cfg(target_vendor = "apple")]
unsafe impl Sync for Flushed {}

pub struct Pending {
    #[cfg(target_vendor = "apple")]
    buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    /// The heads a flushed frame committed ahead of `buffer`.
    #[cfg(target_vendor = "apple")]
    flushed: Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    #[cfg(not(target_vendor = "apple"))]
    #[allow(dead_code)]
    buffer: (),
}

// SAFETY: what a `Pending` does to its command buffer is `status`, `error`
// and `waitUntilCompleted`, all documented safe from any thread; encoding —
// the part that is not — is over before one exists.
unsafe impl Send for Pending {}

impl std::fmt::Debug for Pending {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pending").finish()
    }
}

impl Pending {
    #[must_use]
    pub fn landed(&self) -> bool {
        #[cfg(target_vendor = "apple")]
        {
            matches!(
                self.buffer.status(),
                MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
            )
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            true
        }
    }

    #[must_use]
    pub fn gpu_span_us(&self) -> (u64, u64) {
        #[cfg(target_vendor = "apple")]
        {
            // SAFETY: documented safe to read after completion; both are plain
            // `CFTimeInterval` getters.
            let start = self
                .flushed
                .first()
                .map_or_else(|| self.buffer.GPUStartTime(), |head| head.GPUStartTime());
            let end = self.buffer.GPUEndTime();
            ((start * 1e6) as u64, (end * 1e6) as u64)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            (0, 0)
        }
    }

    pub fn wait(&self) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            self.buffer.waitUntilCompleted();
            for buffer in self.flushed.iter().chain([&self.buffer]) {
                if let Some(error) = buffer.error() {
                    return Err(Fault::Device {
                        call: "waitUntilCompleted",
                        why: error.localizedDescription().to_string(),
                    });
                }
            }
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(Fault::Deviceless)
        }
    }
}

#[cfg(target_vendor = "apple")]
pub(crate) fn threadgroup(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    lanes: [u32; 3],
) -> MTLSize {
    let width = pipeline.threadExecutionWidth().max(1);
    let total = pipeline.maxTotalThreadsPerThreadgroup().max(1);
    let x = width.min(lanes[0].max(1) as usize).max(1);
    let y = (total / x).min(lanes[1].max(1) as usize).max(1);
    let z = (total / (x * y)).min(lanes[2].max(1) as usize).max(1);
    MTLSize {
        width: x,
        height: y,
        depth: z,
    }
}

/// Without a stated version a library compiles as the default of the SDK the
/// app linked against, which can predate the MSL 4.0 some kernels need, so
/// name 4.0 wherever the OS has it (26 on both macOS and iOS).
#[cfg(target_vendor = "apple")]
pub(crate) fn compile_options() -> Retained<objc2_metal::MTLCompileOptions> {
    use objc2_foundation::{NSOperatingSystemVersion, NSProcessInfo};

    let options = objc2_metal::MTLCompileOptions::new();
    let msl4 =
        NSProcessInfo::processInfo().isOperatingSystemAtLeastVersion(NSOperatingSystemVersion {
            majorVersion: 26,
            minorVersion: 0,
            patchVersion: 0,
        });
    if msl4 {
        options.setLanguageVersion(objc2_metal::MTLLanguageVersion::Version4_0);
    }
    options
}

#[cfg(target_vendor = "apple")]
pub(crate) fn nsstring(text: &str) -> Retained<NSString> {
    NSString::from_str(text)
}

/// Unified memory the system could hand a process now, where it says: the
/// host VM statistics are a macOS interface.
fn available_memory() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let mut stats: libc::vm_statistics64 = unsafe { core::mem::zeroed() };
        let mut count = libc::HOST_VM_INFO64_COUNT;
        // SAFETY: `stats` is a live vm_statistics64 and `count` states its
        // size in words, as HOST_VM_INFO64 asks; the call writes at most that.
        #[allow(deprecated, reason = "libc's mach bindings are what this crate links")]
        let rc = unsafe {
            libc::host_statistics64(
                libc::mach_host_self(),
                libc::HOST_VM_INFO64,
                (&raw mut stats).cast(),
                &raw mut count,
            )
        };
        // SAFETY: a plain query.
        let page = u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).ok()?;
        // What the system cannot hand out is what Activity Monitor calls
        // used: app memory (anonymous pages a process may not give back),
        // wired and compressed. The file cache, active or not, is handed out
        // on demand, and a model's own weights sit in it after a read.
        let used = (u64::from(stats.internal_page_count)
            .saturating_sub(u64::from(stats.purgeable_count))
            + u64::from(stats.wire_count)
            + u64::from(stats.compressor_page_count))
            * page;
        (rc == libc::KERN_SUCCESS).then(|| Context::physical_memory().saturating_sub(used))
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    #[test]
    fn the_core_count_is_the_registrys_not_a_guess() {
        let ctx = super::Context::bind().expect("a Metal 4 device");
        let cores = super::super::iokit::gpu_cores().expect("the registry says");
        assert_eq!(ctx.cores(), Some(cores));
        assert_eq!(ctx.cores_from(), Some(super::CoresFrom::Registry));
        assert!(
            cores >= 7,
            "no Apple Silicon GPU has fewer than 7 cores: {cores}"
        );
    }
}
