//! The bound device: an `MTLDevice`, the queue every fire is committed on,
//! and the facts a shell reads once.

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
    MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily, MTLResourceOptions, MTLSize,
};

/// Every device reservation this process has asked for, counted. Bumped on
/// attempt (not success) at the top of the three doors that hand back an
/// `MTLBuffer`, so a gate can assert a refusal happened before any
/// allocation was taken.
static RESERVATIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// [`RESERVATIONS`]. A gate reads it before and after the call it's making a
/// claim about; the absolute number is meaningless (shared across tests).
#[must_use]
pub fn reservations() -> u64 {
    RESERVATIONS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Is there a Metal device on this machine? A headless Apple target or a VM
/// with no GPU exposed answers `nil` to `MTLCreateSystemDefaultDevice`.
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

/// Which `MTLGPUFamilyApple<N>` this device answers to, or 0 for none.
/// Probed newest-first since families are cumulative (a newer device answers
/// `supportsFamily:` for Apple7 as well as Apple9). The name-match fallback
/// catches a family newer than this list.
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
    kernels_metal::DeviceInfo::of_name(&device.name().to_string()).apple_family
}

#[cfg(target_vendor = "apple")]
type Device = Retained<ProtocolObject<dyn MTLDevice>>;
#[cfg(not(target_vendor = "apple"))]
type Device = ();

#[cfg(target_vendor = "apple")]
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
#[cfg(not(target_vendor = "apple"))]
type Queue = ();

/// The bound device and the queue its fires are committed on.
///
/// The two Metal objects are read only under `cfg(target_vendor = "apple")`;
/// off Apple the type aliases are `()` and the struct is never constructed.
pub struct Context {
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    device: Device,
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    queue: Queue,
    name: String,
    /// `recommendedMaxWorkingSetSize` — what the device holds without
    /// paging. The budget check's ceiling, not a hard limit.
    working_set: u64,
    /// `maxBufferLength` — one reservation's ceiling.
    max_buffer: u64,
    /// GPU core count, as the CUDA sibling's SM-count stand-in. Metal
    /// publishes none, so this is a stated default feeding only the cost
    /// model — no kernel argument reads it.
    cores: u32,
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
    /// Bind the system default device and open one command queue on it.
    ///
    /// # Errors
    ///
    /// [`Fault::Deviceless`] off Apple or with no device present,
    /// [`Fault::Device`] when the queue would not open or the device does
    /// not share memory with the host.
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
            // Stated once, here: `kernels_metal::tuning` holds one
            // process-wide `OnceLock` cell read by every crossover in that crate.
            kernels_metal::tuning::describe(kernels_metal::DeviceInfo {
                apple_family: family(&device),
                // Metal publishes no core count; nothing in `tuning` reads this.
                gpu_core_count: 0,
            });
            let working_set = device.recommendedMaxWorkingSetSize();
            let max_buffer = device.maxBufferLength() as u64;
            Ok(Context {
                device,
                queue,
                name,
                working_set,
                max_buffer,
                cores: 32,
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(Fault::Deviceless)
        }
    }

    /// The device's own name, for a footprint line or a refusal.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// What the device says it will hold resident.
    #[must_use]
    pub fn working_set(&self) -> u64 {
        self.working_set
    }

    /// One reservation's ceiling.
    #[must_use]
    pub fn max_buffer(&self) -> u64 {
        self.max_buffer
    }

    /// The core count the cost model is handed. Stated, not probed — see the field.
    #[must_use]
    pub fn cores(&self) -> u32 {
        self.cores
    }

    /// The contract's thread-binding verb, answered by explaining that Metal
    /// has no per-thread device state to bind.
    ///
    /// # Errors
    ///
    /// Never. The signature matches the CUDA sibling's so the runtime's call
    /// order is one shape.
    pub fn bind_thread(&self) -> Result<()> {
        Ok(())
    }

    /// Reserve `bytes` of shared storage.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] past `maxBufferLength`, [`Fault::Device`] when the
    /// device declined.
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

    /// Wrap `span` bytes of memory this process already holds, without
    /// copying (nil deallocator, so Metal does not own the pages). Only
    /// reachable through [`Buffer::mapped`](super::alloc::Buffer::mapped),
    /// which holds an `Arc<Mapping>` for the reservation's whole life.
    ///
    /// # Safety
    ///
    /// `at` must be page-aligned and `span` a multiple of the page size
    /// (required by `newBufferWithBytesNoCopy` on macOS, not checked by it),
    /// `[at, at + span)` must be one live readable mapping, and the caller
    /// must keep it mapped until every reference to the returned buffer is gone.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] past `maxBufferLength`, [`Fault::Device`] when the
    /// device declined the span.
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

    /// The stand-in a zero-length reservation holds. Metal refuses a
    /// zero-length buffer, so an empty pool row gets one byte nobody may
    /// mint a handle into (`Buffer::bytes` stays 0, every `span` refuses).
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
        {
        }
    }

    /// The `MTLDevice`, for the pipeline cache's compiler.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Open one command buffer and one compute pass for a fire. One encoder
    /// for the whole fire: a compute pass is `MTLDispatchTypeSerial`, so
    /// every dispatch observes the writes of every dispatch before it —
    /// matching what `model_exec::fire::walk` assumes of a stream, so it
    /// needs no barrier vocabulary.
    ///
    /// # Errors
    ///
    /// [`Fault::Deviceless`] off Apple, [`Fault::Device`] when the queue or
    /// the pass would not open.
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
            Ok(Frame {
                buffer,
                encoder: Some(encoder),
                blit: None,
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(Fault::Deviceless)
        }
    }
}

/// One fire's command buffer and the pass that is open on it. The shell
/// encodes into it through [`Frame::encoder`], copies rows through
/// [`Frame::copy`], and closes it with [`Frame::commit`] (waits) or
/// [`Frame::commit_async`] (doesn't — the fire path's choice). A command
/// buffer holds at most one open encoder, so compute and blit are two
/// `Option`s of which at most one is `Some`.
pub struct Frame {
    #[cfg(target_vendor = "apple")]
    buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    #[cfg(not(target_vendor = "apple"))]
    #[allow(dead_code)]
    buffer: (),
    #[cfg(target_vendor = "apple")]
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    #[cfg(not(target_vendor = "apple"))]
    encoder: Option<()>,
    #[cfg(target_vendor = "apple")]
    blit: Option<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>>,
    #[cfg(not(target_vendor = "apple"))]
    #[allow(dead_code)]
    blit: Option<()>,
}

// SAFETY: a `Frame` is created, encoded into and committed on one thread —
// the lane thread. `Send` is what lets a `Shell` holding one move.
unsafe impl Send for Frame {}

impl Frame {
    /// The open pass, for the encode sink.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn encoder(&self) -> &ProtocolObject<dyn MTLComputeCommandEncoder> {
        self.encoder
            .as_deref()
            .expect("the pass is open until `commit` closes it")
    }

    /// End whichever encoder is open. Idempotent.
    #[cfg(target_vendor = "apple")]
    fn end_pass(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }
        if let Some(blit) = self.blit.take() {
            blit.endEncoding();
        }
    }

    /// Close the open pass and open another in the same command buffer. The
    /// one place a fire needs two passes: `executeCommandsInBuffer:` is not
    /// a dispatch, so a compute pass's serial-between-dispatches guarantee
    /// doesn't order it after the rebind shader's write — two passes in one
    /// command buffer does.
    ///
    /// # Errors
    ///
    /// [`Fault::Deviceless`] off Apple, [`Fault::Device`] when the command
    /// buffer would not open another pass.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn next_pass(&mut self) -> Result<&ProtocolObject<dyn MTLComputeCommandEncoder>> {
        self.end_pass();
        let encoder = self.buffer.computeCommandEncoder().ok_or(Fault::Device {
            call: "computeCommandEncoder",
            why: "the command buffer would not open a second compute pass".to_string(),
        })?;
        self.encoder = Some(encoder);
        Ok(self.encoder.as_deref().expect("just opened"))
    }

    /// Copy `len` bytes device-side, inside this fire's own command buffer.
    /// With two frames in flight, this copies the rows a reader wants out
    /// while this fire still owns them, into a seat the next fire won't
    /// touch — a blit pass ordered after every prior dispatch. Also used by
    /// kv grafts to move cells within one pool reservation; the caller
    /// (`crate::store::Move::plan`) refuses overlapping regions, since an
    /// overlapping blit is undefined rather than a shift.
    ///
    /// # Errors
    ///
    /// [`Fault::Deviceless`] off Apple, [`Fault::Device`] when the command
    /// buffer would not open a blit pass.
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
                if let Some(encoder) = self.encoder.take() {
                    encoder.endEncoding();
                }
                self.blit = Some(self.buffer.blitCommandEncoder().ok_or(Fault::Device {
                    call: "blitCommandEncoder",
                    why: "the command buffer would not open a blit pass".to_string(),
                })?);
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

    /// Close the pass, commit the buffer, and wait for the device. Not the
    /// fire path's spelling any more; still used by the indirect plane's
    /// `executeCommandsInBuffer:` and the native surface's eager door, both
    /// of which have a caller waiting for the answer.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] carrying the command buffer's own error when the
    /// GPU refused the work.
    pub fn commit(mut self) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            self.end_pass();
            self.buffer.commit();
            self.buffer.waitUntilCompleted();
            if let Some(error) = self.buffer.error() {
                return Err(Fault::Device {
                    call: "waitUntilCompleted",
                    why: error.localizedDescription().to_string(),
                });
            }
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = self.encoder.take();
            Err(Fault::Deviceless)
        }
    }

    /// [`Frame::commit`], answering how long the device spent on this
    /// command buffer — `GPUEndTime - GPUStartTime`, seconds — for the
    /// kernel profile (`crate::encode::kernel_profile`).
    ///
    /// # Errors
    ///
    /// As [`Frame::commit`].
    pub fn commit_timed(mut self) -> Result<f64> {
        #[cfg(target_vendor = "apple")]
        {
            self.end_pass();
            self.buffer.commit();
            self.buffer.waitUntilCompleted();
            if let Some(error) = self.buffer.error() {
                return Err(Fault::Device {
                    call: "waitUntilCompleted",
                    why: error.localizedDescription().to_string(),
                });
            }
            Ok(self.buffer.GPUEndTime() - self.buffer.GPUStartTime())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = self.encoder.take();
            Err(Fault::Deviceless)
        }
    }

    /// Close the pass, arm the completion handler, commit — and return
    /// without waiting, so the next frame can be encoded while this one runs.
    /// `on_done` runs on Metal's own completion thread, not this one — it
    /// must not call back into the shell (belongs to the lane thread), only
    /// bump an atomic and call into the runtime's sink. Armed before
    /// `commit`, per Metal's rule (arming after races a buffer that may have
    /// finished).
    ///
    /// # Errors
    ///
    /// [`Fault::Deviceless`] off Apple.
    #[cfg_attr(not(target_vendor = "apple"), allow(unused_variables))]
    pub fn commit_async(
        mut self,
        on_done: Option<Box<dyn Fn(Option<String>) + Send + 'static>>,
    ) -> Result<Pending> {
        #[cfg(target_vendor = "apple")]
        {
            self.end_pass();
            if let Some(on_done) = on_done {
                let handler = block2::RcBlock::new(
                    move |buffer: core::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                        // SAFETY: Metal hands the handler a live reference to
                        // the command buffer it is about to retire, valid for
                        // the length of this call.
                        let buffer = unsafe { buffer.as_ref() };
                        on_done(
                            buffer
                                .error()
                                .map(|error| error.localizedDescription().to_string()),
                        );
                    },
                );
                // SAFETY: `addCompletedHandler:` copies the block, so the
                // `RcBlock` may be dropped at the end of this scope; the
                // closure owns everything it touches and is `Send`.
                unsafe {
                    self.buffer
                        .addCompletedHandler(block2::RcBlock::as_ptr(&handler));
                }
            }
            self.buffer.commit();
            Ok(Pending {
                buffer: self.buffer.clone(),
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = self.encoder.take();
            Err(Fault::Deviceless)
        }
    }
}

/// A frame that is dropped instead of committed still ends its pass: Metal
/// expects every opened encoder to be ended, and a walk that refuses
/// mid-dispatch returns through a `?` with the compute pass still open.
impl Drop for Frame {
    fn drop(&mut self) {
        #[cfg(target_vendor = "apple")]
        self.end_pass();
    }
}

/// One committed command buffer the host has not yet caught up with.
///
/// The receipt [`Frame::commit_async`] hands back. [`Pending::landed`] asks
/// without blocking and [`Pending::wait`] blocks — the fire path calls the
/// first to know whether a settle is free and the second only when it has
/// run out of seats.
pub struct Pending {
    #[cfg(target_vendor = "apple")]
    buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
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
    /// Has the device finished with this one? Asked, never waited on.
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

    /// Wait for the device to finish this one, and report what it said. The
    /// one synchronization left in the shell — the settle phase's, never
    /// the enqueue phase's.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] carrying the command buffer's own sentence.
    pub fn wait(&self) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            self.buffer.waitUntilCompleted();
            if let Some(error) = self.buffer.error() {
                return Err(Fault::Device {
                    call: "waitUntilCompleted",
                    why: error.localizedDescription().to_string(),
                });
            }
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(Fault::Deviceless)
        }
    }
}

/// The threadgroup a `Fire` that stated none gets. Picked from the compiled
/// pipeline rather than a constant, since `threadExecutionWidth` and
/// `maxTotalThreadsPerThreadgroup` are properties of the compiled shader.
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

/// An `NSString` for a `&str`, for the two Metal calls that take one.
#[cfg(target_vendor = "apple")]
pub(crate) fn nsstring(text: &str) -> Retained<NSString> {
    NSString::from_str(text)
}
