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

pub struct Context {
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    device: Device,
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    queue: Queue,
    name: String,
    working_set: u64,
    max_buffer: u64,
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
            kernels_metal::tuning::describe(kernels_metal::DeviceInfo {
                apple_family: family(&device),
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

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub fn working_set(&self) -> u64 {
        self.working_set
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
    pub fn cores(&self) -> u32 {
        self.cores
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
    #[cfg(target_vendor = "apple")]
    pub(crate) fn encoder(&self) -> &ProtocolObject<dyn MTLComputeCommandEncoder> {
        self.encoder
            .as_deref()
            .expect("the pass is open until `commit` closes it")
    }

    #[cfg(target_vendor = "apple")]
    fn end_pass(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }
        if let Some(blit) = self.blit.take() {
            blit.endEncoding();
        }
    }

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
                if let Some(encoder) = self.encoder.take() {
                    encoder.endEncoding();
                }
                self.blit = Some(self.buffer.blitCommandEncoder().ok_or(Fault::Device {
                    call: "blitCommandEncoder",
                    why: "the command buffer would not open a blit pass".to_string(),
                })?);
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
                if let Some(encoder) = self.encoder.take() {
                    encoder.endEncoding();
                }
                self.blit = Some(self.buffer.blitCommandEncoder().ok_or(Fault::Device {
                    call: "blitCommandEncoder",
                    why: "the command buffer would not open a blit pass".to_string(),
                })?);
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

impl Drop for Frame {
    fn drop(&mut self) {
        #[cfg(target_vendor = "apple")]
        self.end_pass();
    }
}

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
            let (start, end) = (self.buffer.GPUStartTime(), self.buffer.GPUEndTime());
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

#[cfg(target_vendor = "apple")]
pub(crate) fn nsstring(text: &str) -> Retained<NSString> {
    NSString::from_str(text)
}
