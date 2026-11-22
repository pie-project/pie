#![allow(unsafe_code)]
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
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLEvent, MTLGPUFamily, MTLResourceOptions, MTLSize,
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
    sparse: std::cell::OnceCell<std::sync::Arc<super::sparse::Sparse>>,
    /// What [`Frame::barrier`] signals and waits on, and the value it last set.
    #[cfg(target_vendor = "apple")]
    barrier: (Retained<ProtocolObject<dyn MTLEvent>>, std::cell::Cell<u64>),
    /// Buffers kernels reach through an address in another (the pool's
    /// buffers past the first): every compute pass declares them
    /// (`useResource`), so Metal orders their writes and reads as it does a
    /// bound buffer's.
    #[cfg(target_vendor = "apple")]
    indirect: RefCell<Vec<super::alloc::Slab>>,
    /// The shared empty binding ([`Context::empty`]).
    #[cfg(target_vendor = "apple")]
    empty: std::cell::OnceCell<super::alloc::Slab>,
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
            let barrier = device.newEvent().ok_or(Fault::Device {
                call: "newEvent",
                why: "the device would not open an event".to_string(),
            })?;
            Ok(Context {
                device,
                queue,
                name,
                working_set,
                max_buffer,
                cores,
                family,
                sparse: std::cell::OnceCell::new(),
                barrier: (barrier, std::cell::Cell::new(0)),
                indirect: RefCell::new(Vec::new()),
                empty: std::cell::OnceCell::new(),
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

    /// What this process may place on the GPU now: the GPU's wired-memory
    /// limit where the operator set one (`iogpu.wired_limit_mb`), else the
    /// working set Metal recommends less what the process holds on the host
    /// beside its GPU memory ([`host_resident`]: its guests, staging, the
    /// weights' host side). On unified memory the device and the host are
    /// one pool, so a load sized to the recommended working set alone,
    /// beside a large host side, outgrows what the GPU can wire (the M1's
    /// "Insufficient Memory" command buffers under gemma-4-E4B at c256).
    /// Read now, not at bind: the host side grows as guests bind.
    #[must_use]
    pub fn working_set(&self) -> u64 {
        derived_working_set(self.working_set, self.allocated())
    }

    /// The working set Metal recommends for this process, as the device
    /// states it.
    #[must_use]
    pub fn recommended_working_set(&self) -> u64 {
        self.working_set
    }

    /// The GPU memory every other process holds now: the system memory the
    /// GPU driver has allocated for all of them together (IORegistry,
    /// `Alloc system memory`) less what this process holds
    /// ([`Self::allocated`]). The working set Metal recommends is shared
    /// with it: a load that takes the whole of it beside them is refused for
    /// memory at the edge. 0 where the registry does not say.
    #[must_use]
    pub fn others_allocated(&self) -> u64 {
        #[cfg(target_vendor = "apple")]
        {
            super::iokit::gpu_system_allocated()
                .map_or(0, |system| system.saturating_sub(self.allocated()))
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            0
        }
    }

    /// The bytes this device holds for the process now
    /// (`currentAllocatedSize`): every buffer, heap and the driver's own
    /// allocations on our behalf.
    #[must_use]
    pub fn allocated(&self) -> u64 {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLDevice as _;
            self.device().currentAllocatedSize() as u64
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            0
        }
    }

    /// What the host can still give this process's GPU memory now (the
    /// governor's `host_room`, read on its tick): [`Context::working_set`]'s
    /// rule over a fresh read of the device, less what every process's GPU
    /// memory holds of it and the driver's floor. Negative where they
    /// already hold more: the pool gives the difference back.
    pub fn host_room() -> std::result::Result<i64, String> {
        #[cfg(target_vendor = "apple")]
        {
            let device = MTLCreateSystemDefaultDevice()
                .ok_or_else(|| "no Metal device to read the room of".to_string())?;
            let mine = device.currentAllocatedSize() as u64;
            let all = super::iokit::gpu_system_allocated().unwrap_or(mine).max(mine);
            let usable = derived_working_set(device.recommendedMaxWorkingSetSize(), mine);
            let held = all.saturating_add(crate::store::accounting::DRIVER_FLOOR);
            Ok(i64::try_from(usable).unwrap_or(i64::MAX) - i64::try_from(held).unwrap_or(i64::MAX))
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Err("no Metal device off Apple platforms".to_string())
        }
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
            crate::diag::made("buffer", bytes);
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
        crate::diag::made("no-copy buffer", bytes);
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

    /// The one empty binding every zero-byte buffer shares, made on first
    /// ask: an empty bind allocates nothing after that.
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    pub(crate) fn empty(&self) -> super::alloc::Slab {
        #[cfg(target_vendor = "apple")]
        {
            // One byte every empty binding shares: one of the three tiny
            // fixed buffers outside the pool (this, the keepalive sink, the
            // warm-load bind placeholder), left there by ruling.
            self.empty
                .get_or_init(|| {
                    RESERVATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    crate::diag::made("empty buffer", 1);
                    self.device
                        .newBufferWithLength_options(1, MTLResourceOptions::StorageModeShared)
                        .expect("one byte of shared storage")
                })
                .clone()
        }
        #[cfg(not(target_vendor = "apple"))]
        {}
    }

    #[cfg(target_vendor = "apple")]
    pub(crate) fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// The mapping side, opened on first ask: every pool and span is a
    /// placement sparse buffer over it. Its residency set is attached to
    /// this context's frame queue.
    #[cfg(target_vendor = "apple")]
    pub fn sparse(&self) -> Result<std::sync::Arc<super::sparse::Sparse>> {
        if let Some(opened) = self.sparse.get() {
            return Ok(opened.clone());
        }
        let opened = super::sparse::Sparse::open(&self.device, &self.queue)?;
        let _ = self.sparse.set(opened.clone());
        Ok(opened)
    }

    /// Keeps `slabs` resident for every frame on this context's queue (or
    /// lets them go, `keep` false), through the Metal 4 residency set the
    /// queue carries: answers false where there is none.
    pub fn hold_resident<'a>(
        &self,
        slabs: impl IntoIterator<Item = &'a super::alloc::Slab>,
        keep: bool,
    ) -> Result<bool> {
        #[cfg(target_vendor = "apple")]
        {
            self.sparse()?.hold(slabs, keep);
            Ok(true)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (slabs.into_iter().count(), keep);
            Ok(false)
        }
    }

    /// Declares `slabs` to every compute pass from now on: buffers kernels
    /// reach through an address in another (the pool's past the first).
    #[cfg(target_vendor = "apple")]
    pub(crate) fn reach_indirectly(&self, slabs: Vec<super::alloc::Slab>) {
        *self.indirect.borrow_mut() = slabs;
    }

    /// No device off Apple: no pass to declare anything to.
    #[cfg(not(target_vendor = "apple"))]
    pub(crate) fn reach_indirectly(&self, _slabs: Vec<super::alloc::Slab>) {}

    pub fn frame(&self) -> Result<Frame> {
        #[cfg(target_vendor = "apple")]
        {
            let buffer = self.queue.commandBuffer().ok_or(Fault::Device {
                call: "commandBuffer",
                why: "the queue would not open a command buffer".to_string(),
            })?;
            let encoder = buffer.computeCommandEncoder().ok_or(Fault::Device {
                call: "computeCommandEncoder",
                why: "the command buffer would not open a compute pass".to_string(),
            })?;
            let indirect = self.indirect.borrow().clone();
            declare(&encoder, &indirect);
            Ok(Frame {
                indirect,
                buffer: RefCell::new(buffer),
                encoder: RefCell::new(Some(encoder)),
                blit: None,
                committed: RefCell::new(Vec::new()),
                flushed: RefCell::new(Vec::new()),
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(Fault::Deviceless)
        }
    }
}

/// One command buffer and the compute pass open on it. A fence (see
/// [`Frame::fence`]) closes the pass, encodes an event signal or wait on
/// the buffer, and opens the next pass. The cells let a fence happen
/// through a shared `&Frame`, which is all a live encode holds.
pub struct Frame {
    /// [`Context`]'s indirect buffers, declared in every compute pass.
    #[cfg(target_vendor = "apple")]
    indirect: Vec<super::alloc::Slab>,
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
}

// SAFETY: a `Frame` is created, encoded into and committed on one thread —
// the lane thread. `Send` is what lets a `Shell` holding one move.
unsafe impl Send for Frame {}

impl Frame {
    /// The open compute pass, retained: a fence may replace it while the
    /// caller still holds this one, so nothing borrows the cell for long.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn encoder(&self) -> Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> {
        self.encoder
            .borrow()
            .clone()
            .expect("the pass is open until `commit` closes it")
    }

    /// Has `then` run once this frame's buffer is committed; a frame dropped
    /// uncommitted never runs it.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn on_commit(&self, then: Box<dyn FnOnce() + Send>) {
        self.committed.borrow_mut().push(then);
    }

    /// Runs `body`'s encoding in a command buffer of its own (the frame's work
    /// before it flushed ahead, the frame's after it in the next buffer) and
    /// waits for it: its device seconds, or `None` where a blit pass is open
    /// and the frame can't flush. The kernel profile's guest regions.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn timed<R>(&self, body: impl FnOnce() -> Result<R>) -> Result<(R, Option<f64>)> {
        if !self.flush()? {
            return Ok((body()?, None));
        }
        let out = body()?;
        if !self.flush()? {
            return Ok((out, None));
        }
        let buffer = self.flushed.borrow().last().cloned().expect("a flush keeps its buffer");
        buffer.waitUntilCompleted();
        Ok((out, Some(buffer.GPUEndTime() - buffer.GPUStartTime())))
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
        let encoder = tail.computeCommandEncoder().ok_or(Fault::Device {
            call: "computeCommandEncoder",
            why: "the command buffer would not open a compute pass".to_string(),
        })?;
        declare(&encoder, &self.indirect);
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

    /// No device off Apple: no pass to open.
    #[cfg(not(target_vendor = "apple"))]
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    pub(crate) fn next_pass(&mut self) -> Result<()> {
        Err(Fault::Deviceless)
    }

    #[cfg(target_vendor = "apple")]
    pub(crate) fn next_pass(&mut self) -> Result<&ProtocolObject<dyn MTLComputeCommandEncoder>> {
        self.end_pass();
        let encoder = self
            .buffer
            .get_mut()
            .computeCommandEncoder()
            .ok_or(Fault::Device {
                call: "computeCommandEncoder",
                why: "the command buffer would not open a second compute pass".to_string(),
            })?;
        declare(&encoder, &self.indirect);
        *self.encoder.get_mut() = Some(encoder);
        Ok(self.encoder.get_mut().as_deref().expect("just opened"))
    }

    /// Ends the pass here and opens a concurrent one: its dispatches may
    /// run together, ordered only by [`Frame::level`] barriers between them.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn concurrent_pass(&mut self) -> Result<()> {
        use objc2_metal::MTLDispatchType;
        self.end_pass();
        let encoder = self
            .buffer
            .get_mut()
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .ok_or(Fault::Device {
                call: "computeCommandEncoderWithDispatchType",
                why: "the command buffer would not open a concurrent compute pass".to_string(),
            })?;
        declare(&encoder, &self.indirect);
        *self.encoder.get_mut() = Some(encoder);
        Ok(())
    }

    /// In a concurrent pass: every dispatch before this one is done with the
    /// pass's buffers before any after it starts.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn level(&self) {
        use objc2_metal::{MTLBarrierScope, MTLComputeCommandEncoder};
        self.encoder().memoryBarrierWithScope(MTLBarrierScope::Buffers);
    }

    /// Ends the pass here and opens the next once the GPU has done all the
    /// work before it: a pass that reads through an address, which Metal
    /// tracks no hazard by, sees what the passes before it wrote.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn barrier(&mut self, context: &Context) -> Result<()> {
        self.end_pass();
        let (event, last) = &context.barrier;
        let value = last.get() + 1;
        last.set(value);
        let buffer = self.buffer.get_mut();
        buffer.encodeSignalEvent_value(event, value);
        buffer.encodeWaitForEvent_value(event, value);
        let encoder = buffer.computeCommandEncoder().ok_or(Fault::Device {
            call: "computeCommandEncoder",
            why: "the command buffer would not open a compute pass after a barrier".to_string(),
        })?;
        declare(&encoder, &self.indirect);
        *self.encoder.get_mut() = Some(encoder);
        Ok(())
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
            .buffer
            .borrow()
            .computeCommandEncoder()
            .ok_or(Fault::Device {
                call: "computeCommandEncoder",
                why: "the command buffer would not reopen a compute pass after a fence".to_string(),
            })?;
        declare(&reopened, &self.indirect);
        *self.encoder.borrow_mut() = Some(reopened);
        Ok(())
    }

    /// `len` bytes of `slab` from `at` set to `value`, in a blit pass
    /// (test builds: the guest scratch's poison).
    #[cfg(feature = "test-slot-fill")]
    #[cfg_attr(not(target_vendor = "apple"), allow(unused_variables))]
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    pub(crate) fn fill_with(
        &mut self,
        slab: &super::alloc::Slab,
        at: u64,
        len: u64,
        value: u8,
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
                slab,
                objc2_foundation::NSRange {
                    location: at as usize,
                    length: len as usize,
                },
                value,
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
            #[cfg(feature = "fault-injection")]
            let injected = FAIL_NEXT.with(|next| next.replace(false));
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
                        let refused = refused.map(|error| error.localizedDescription().to_string());
                        #[cfg(feature = "fault-injection")]
                        let refused = if injected { Some(INJECTED.to_string()) } else { refused };
                        on_done(refused);
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
                #[cfg(feature = "fault-injection")]
                injected,
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
    /// Answers as failed whatever the device did ([`fail_next_frame`]).
    #[cfg(all(target_vendor = "apple", feature = "fault-injection"))]
    injected: bool,
    #[cfg(not(target_vendor = "apple"))]
    #[allow(dead_code)]
    buffer: (),
}

#[cfg(feature = "fault-injection")]
thread_local! {
    static FAIL_NEXT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[cfg(all(target_vendor = "apple", feature = "fault-injection"))]
const INJECTED: &str = "a failure injected by `fail_next_frame`";

/// Answers the next frame this thread commits with a completion handler
/// ([`Frame::commit_async`]) as failed, to its handler and to its
/// [`Pending::wait`], whatever the device did: tests of what a failed
/// command buffer leaves, which a real device fault can't be made to give.
///
/// A test seam only, compiled under the `fault-injection` feature (off by
/// default; engine-metal's tests turn it on): a served binary has no such
/// path, and nothing in serving, no config, env var or protocol, reaches
/// it. Its flag is per thread and inert unless this is called there.
#[cfg(feature = "fault-injection")]
pub fn fail_next_frame() {
    FAIL_NEXT.with(|next| next.set(true));
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
            #[cfg(feature = "fault-injection")]
            if self.injected {
                return Err(Fault::Device {
                    call: "waitUntilCompleted",
                    why: INJECTED.to_string(),
                });
            }
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

/// The threadgroup `fire` launches in: the one it states, else the widest
/// `pipeline` runs over its lanes. A stated group wider than the pipeline
/// runs is refused: Metal drops such a dispatch without a word, and the
/// kernel's writes never land.
#[cfg(target_vendor = "apple")]
pub(crate) fn group_of(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    fire: kernels_metal::Fire,
) -> Result<MTLSize> {
    if fire.group == [0, 0, 0] {
        return Ok(threadgroup(pipeline, fire.lanes));
    }
    let group = MTLSize {
        width: fire.group[0].max(1) as usize,
        height: fire.group[1].max(1) as usize,
        depth: fire.group[2].max(1) as usize,
    };
    let threads = group.width * group.height * group.depth;
    let most = pipeline.maxTotalThreadsPerThreadgroup();
    if threads > most {
        return Err(Fault::Shader {
            file: fire.file,
            entrypoint: fire.entrypoint,
            why: format!(
                "the fire asks for {threads} threads a group and this device runs the pipeline at most {most}"
            ),
        });
    }
    Ok(group)
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

/// Declares `slabs` read and written through addresses in `encoder`'s
/// dispatches, so Metal tracks their hazards as a bound buffer's.
#[cfg(target_vendor = "apple")]
fn declare(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, slabs: &[super::alloc::Slab]) {
    for slab in slabs {
        encoder.useResource_usage(
            ProtocolObject::from_ref(&**slab),
            objc2_metal::MTLResourceUsage::Read | objc2_metal::MTLResourceUsage::Write,
        );
    }
}

/// [`Context::working_set`]'s rule: the operator's GPU wired limit where
/// set, else `recommended` less this process's host-resident memory beside
/// the `gpu` bytes Metal holds for it.
#[must_use]
pub fn derived_working_set(recommended: u64, gpu: u64) -> u64 {
    match wired_limit() {
        Some(limit) => limit,
        None => recommended.saturating_sub(host_resident(gpu)),
    }
}

/// The GPU's wired-memory limit the operator set (`sysctl
/// iogpu.wired_limit_mb`), in bytes; `None` at the system's default (0).
#[must_use]
pub fn wired_limit() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let mut mb: i32 = 0;
        let mut len = std::mem::size_of::<i32>();
        // SAFETY: `iogpu.wired_limit_mb` is an int sysctl; `len` states the
        // out buffer's size and the call writes at most that many bytes.
        let rc = unsafe {
            libc::sysctlbyname(
                c"iogpu.wired_limit_mb".as_ptr(),
                (&raw mut mb).cast(),
                &raw mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (rc == 0 && mb > 0).then(|| u64::try_from(mb).unwrap_or(0) << 20)
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// What this process holds on the host beside the `gpu` bytes Metal holds
/// for it: its physical footprint (which counts its GPU memory on unified
/// memory) less those. 0 where the footprint cannot be read.
#[must_use]
pub fn host_resident(gpu: u64) -> u64 {
    #[cfg(target_vendor = "apple")]
    {
        let mut info: libc::rusage_info_v2 = unsafe { std::mem::zeroed() };
        // SAFETY: `info` is a `rusage_info_v2`, the struct the
        // RUSAGE_INFO_V2 flavor writes, owned here for the call.
        let rc = unsafe {
            libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V2,
                (&raw mut info).cast(),
            )
        };
        if rc == 0 {
            info.ri_phys_footprint.saturating_sub(gpu)
        } else {
            0
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = gpu;
        0
    }
}
