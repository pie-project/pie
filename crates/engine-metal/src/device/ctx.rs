//! The bound device: an `MTLDevice`, the queue every fire is committed on,
//! and the facts a shell reads once.
//!
//! **ONE DEVICE PER SHELL, AND NO THREAD RULE.** The CUDA sibling's
//! `bind_thread` exists because `cudaSetDevice` is per-thread state and a
//! context bound on the boot thread strands every later call. Metal has no
//! such state: an `MTLDevice` and an `MTLCommandQueue` are objects, they are
//! documented thread-safe, and moving a loaded shell onto a lane thread
//! costs nothing and needs no call. The contract's `bind_thread` is answered
//! `Ok(())` for that reason and not because it was forgotten.
//!
//! **UNIFIED MEMORY IS ASSERTED, NOT ASSUMED.** [`Buffer`](super::Buffer)
//! writes and reads through `contents()`, which is only the device's own
//! bytes on a machine where the CPU and the GPU share them. A device that
//! says otherwise is refused at bind with a sentence, rather than silently
//! reading a stale mapping every fire.

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

/// Is there a Metal device on this machine?
///
/// **A BUILD THAT NAMES METAL IS NOT A MACHINE THAT HAS IT** — a headless
/// Apple target, or a VM with no GPU exposed, answers `nil` to
/// `MTLCreateSystemDefaultDevice`. This is the door a GPU test knocks on
/// before it asks for anything.
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

/// Which `MTLGPUFamilyApple<N>` this device answers to, or 0 for one that
/// answers to none.
///
/// **PROBED NEWEST-FIRST, AND THE ORDER IS THE WHOLE CORRECTNESS.** The
/// families are cumulative: an M4 answers `supportsFamily:` for Apple7 as
/// well as for Apple9, so an oldest-first walk reports every Apple silicon GPU
/// ever made as an Apple7 and hands all of them the M1 constants — a bug that
/// looks exactly like the tuning table not existing.
///
/// The probe rather than `DeviceInfo::of_name`, and the difference is not
/// cosmetic: a name match answers 0 for any silicon minted after that table
/// was written, and the fallback below is what catches a family newer than
/// this list. Both roads end at the same measured defaults when nothing
/// answers, which is `DeviceTuning::of`'s rule.
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
/// off Apple the type aliases are `()` and the struct is never constructed,
/// which is the same `allow` the CUDA sibling carries for its runtime-less
/// build and for the same reason.
pub struct Context {
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    device: Device,
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    queue: Queue,
    name: String,
    /// `recommendedMaxWorkingSetSize` — what the device says it will hold
    /// without paging. The budget check's ceiling, not a hard limit.
    working_set: u64,
    /// `maxBufferLength` — one reservation's ceiling.
    max_buffer: u64,
    /// The number of GPU cores, as the shell's stand-in for the CUDA
    /// sibling's SM count. Metal publishes no such count, so this is a
    /// stated default rather than a probe, and it feeds only the profile's
    /// cost model — nothing a kernel argument reads.
    cores: u32,
}

// SAFETY: `MTLDevice` and `MTLCommandQueue` are documented thread-safe.
// What `Send` buys is the move from the thread that booted the shell onto
// the lane thread that fires it; nothing here is shared BETWEEN threads.
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
            // **SAY WHAT MACHINE THIS IS, ONCE, HERE.** `kernels_metal::
            // tuning` holds one process-wide cell and every crossover in that
            // crate reads it; until something fills it, an M4 is served the
            // M1 Max's measurements. This is the shell that binds the device,
            // so this is where the answer exists — and it is a `OnceLock`
            // set, so a boot document that already stated a family keeps it
            // (`crate::boot::tuning` runs at the door, before any load).
            kernels_metal::tuning::describe(kernels_metal::DeviceInfo {
                apple_family: family(&device),
                // Metal publishes no core count; IOKit's `gpu-core-count` is
                // the only place it lives, and nothing in `tuning` branches
                // on it. 0 is what the probe honestly has.
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

    /// The core count the cost model is handed. Stated, not probed — see the
    /// field.
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

    /// The stand-in a zero-length reservation holds.
    ///
    /// Metal refuses a zero-length buffer and a plan may state an empty
    /// pool row, so the empty reservation is one byte nobody may mint a
    /// handle into (`Buffer::bytes` stays 0, and every `span` refuses).
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    pub(crate) fn empty(&self) -> super::alloc::Slab {
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

    /// Open one command buffer and one compute pass for a fire.
    ///
    /// **ONE ENCODER FOR THE WHOLE FIRE, AND THAT IS THE ORDERING
    /// ARGUMENT.** A compute pass opened with `computeCommandEncoder` is
    /// `MTLDispatchTypeSerial`: every dispatch in it observes the writes of
    /// every dispatch before it, with the barriers Metal inserts. That is
    /// exactly the semantics `model_exec::fire::walk` assumes of a stream, so
    /// the walk needs no barrier vocabulary and the shell needs no fence
    /// bookkeeping. A concurrent pass — Metal's answer to §6's fork/join
    /// streams — would need the walk's `Sink` events and is not this wave's.
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

/// One fire's command buffer and the pass that is open on it.
///
/// The shell encodes into it through [`Frame::encoder`], copies the rows a
/// reader will want through [`Frame::copy`], and closes it with either
/// [`Frame::commit`] — which waits — or [`Frame::commit_async`], which does
/// not and is the one the fire path takes.
///
/// **TWO ENCODER KINDS, ONE AT A TIME.** A command buffer holds at most one
/// open encoder, so the compute pass and the blit pass are two `Option`s of
/// which at most one is `Some`; opening either ends whatever was open. The
/// ORDER that matters is the one the fire path uses — every dispatch, then
/// the readout copy — and it is the command buffer's own, which is why the
/// copy needs no fence.
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

    /// Close the open pass and open another in the same command buffer.
    ///
    /// **THE ONE PLACE A FIRE NEEDS TWO PASSES, AND IT IS WHY.** The rebind
    /// shader WRITES the indirect command buffer that the call after it
    /// EXECUTES, and an `executeCommandsInBuffer:` in the same pass as the
    /// dispatch that wrote the commands would be reading them concurrently:
    /// a compute pass is serial between dispatches, and
    /// `executeCommandsInBuffer:` is not a dispatch. Two passes in one
    /// command buffer is the ordering Metal states for that — the second
    /// encoder observes everything the first one wrote — and it costs one
    /// encoder open rather than a second commit.
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

    /// **Copy `len` bytes device-side, inside this fire's own command
    /// buffer** — the readout's capture, and the reason it is here rather
    /// than in a host `memcpy` after the wait.
    ///
    /// With one frame in flight the host could read the arena the instant
    /// the fire was done, because nothing else was running. With two, the
    /// frame BEHIND this one is already writing the same rectangles by the
    /// time the host gets round to settling this one: the out seam is one
    /// arena slot that every fire carves over. So the rows a reader will
    /// want are copied out WHILE this fire owns them, into a seat the next
    /// fire does not touch, and the host reads that seat instead.
    ///
    /// A blit pass, not a dispatch: there is no kernel here, and the copy is
    /// ordered after every dispatch encoded before it by the command
    /// buffer's own encoder order.
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
            // SAFETY: both spans were bounds-checked by `Buffer::span` at the
            // handle that named them, and both buffers outlive the command
            // buffer — the readout seat is owned by the shell for the life of
            // the load, and the source is retained by the handle row the
            // caller resolved it through.
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

    /// Close the pass, commit the buffer, and wait for the device.
    ///
    /// **THE SYNCHRONOUS SPELLING, AND IT IS NOT THE FIRE PATH'S ANY MORE.**
    /// What still takes it is the indirect plane's `executeCommandsInBuffer:`
    /// (`crate::icb`) and the native surface's eager door, both of which have
    /// a caller standing there for the answer.
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

    /// **Close the pass, arm the completion handler, commit — and return.**
    ///
    /// The fire path's commit (alto article 1). Nothing here waits: the
    /// command buffer goes to the queue and the host walks away with a
    /// [`Pending`] receipt, which is what lets the next frame be encoded
    /// while this one runs.
    ///
    /// `on_done` runs on **Metal's own completion thread**, not this one, so
    /// what it is handed is already a value: `None` for a command buffer that
    /// completed and `Some(sentence)` for one the device refused. It must not
    /// call back into the shell — the shell is the lane thread's — and the
    /// two things it does do are one atomic bump and one call into the
    /// runtime's sink.
    ///
    /// **THE HANDLER IS ARMED BEFORE `commit`, WHICH IS METAL'S RULE**, not a
    /// preference: a handler added to an already-committed buffer is a race
    /// against a buffer that may have finished.
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

/// **A frame that is dropped instead of committed still ends its pass.**
///
/// Metal expects every encoder it opens to be ended, and a walk that refuses
/// mid-dispatch returns through a `?` with the compute pass still open — so
/// the close belongs in the destructor rather than on the happy path. Both
/// commit spellings end the pass themselves and leave nothing here to do,
/// which is what makes this a belt and not a second policy.
impl Drop for Frame {
    fn drop(&mut self) {
        #[cfg(target_vendor = "apple")]
        self.end_pass();
    }
}

/// **One committed command buffer the host has not yet caught up with.**
///
/// The receipt [`Frame::commit_async`] hands back: the work is on the queue,
/// the completion handler is armed, and this is what the host holds until it
/// comes for the numbers. [`Pending::landed`] asks without blocking and
/// [`Pending::wait`] blocks — the fire path calls the first to know whether a
/// settle is free and the second only when it has run out of seats.
pub struct Pending {
    #[cfg(target_vendor = "apple")]
    buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    #[cfg(not(target_vendor = "apple"))]
    #[allow(dead_code)]
    buffer: (),
}

// SAFETY: what a `Pending` does to its command buffer is `status`, `error`
// and `waitUntilCompleted`, all of which Apple documents as safe from any
// thread; encoding — the part that is not — is over before one exists.
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

    /// Wait for the device to finish this one, and report what it said.
    ///
    /// **THE ONE SYNCHRONIZATION LEFT IN THE SHELL**, and it is the settle
    /// phase's — never the enqueue phase's. A step whose completion handler
    /// has already run returns from here without entering the kernel.
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

/// The threadgroup a `Fire` that stated none gets.
///
/// A [`Fire`](kernels_metal::Fire) carries `lanes` always and `group`
/// sometimes — `Fire::apply` over a bare `[u32; 3]` sets the first and
/// leaves the second at zero. Metal has no default: `dispatchThreads:`
/// takes both. So the shell picks one, and picks it from the PIPELINE
/// rather than from a constant, because the two numbers that bound it —
/// `threadExecutionWidth` and `maxTotalThreadsPerThreadgroup` — are
/// properties of the compiled shader and not of the device.
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
