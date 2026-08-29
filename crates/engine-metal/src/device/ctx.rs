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
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions, MTLSize,
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
    /// exactly the semantics `engine::fire::walk` assumes of a stream, so
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
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(Fault::Deviceless)
        }
    }
}

/// One fire's command buffer and its open compute pass.
///
/// The shell encodes into it through [`Frame::encoder`] and closes it with
/// [`Frame::commit`], which is the ONE synchronization point of a fire —
/// everything before it is enqueue-only, exactly as decision #15 says.
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
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }
        let encoder = self.buffer.computeCommandEncoder().ok_or(Fault::Device {
            call: "computeCommandEncoder",
            why: "the command buffer would not open a second compute pass".to_string(),
        })?;
        self.encoder = Some(encoder);
        Ok(self.encoder.as_deref().expect("just opened"))
    }

    /// Close the pass, commit the buffer, and wait for the device.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] carrying the command buffer's own error when the
    /// GPU refused the work — the one place a Metal fire reports a fault at
    /// all, since every encode before it is enqueue-only.
    pub fn commit(mut self) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(encoder) = self.encoder.take() {
                encoder.endEncoding();
            }
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
