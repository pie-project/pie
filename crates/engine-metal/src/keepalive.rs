//! **THE KEEP-ALIVE QUEUE** (on by default; `PIE_METAL_KEEPALIVE=0` turns it
//! off): a streamed decode
//! fire cuts its command buffer after every router and waits while the host
//! copies expert seats — forty-odd idle gaps of a few milliseconds per token,
//! and the GPU's clocks fall into them. Measured on dsv4: a trivial spinner
//! in another process took the fire's device time from 149 to 86 ms and the
//! token from 232 to 180 ms. This is that spinner inside the shell: one
//! simdgroup of dependent FMAs on its own command queue, dispatched back to
//! back while a fire has been enqueued recently, asleep otherwise.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::device::Context;
use crate::error::Result;

pub struct KeepAlive {
    /// `Instant`-free clock: milliseconds since start of the last enqueue.
    last: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    epoch: std::time::Instant,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// How long after the last enqueue the spinner keeps going: the gap between
/// one token's fire and the next is host turnaround (well under this).
const LINGER_MS: u64 = 250;

impl KeepAlive {
    /// On unless the environment turns it off. Measured per token on the
    /// same prompt: dsv4 237 → 163 ms, qwen38 75 → 57 ms, GLM 320 → 337 ms
    /// (its host copies dominate and the spinner competes for bandwidth).
    #[must_use]
    pub fn wanted() -> bool {
        std::env::var_os("PIE_METAL_KEEPALIVE").is_none_or(|v| v != "0")
    }

    /// Note that a fire is being enqueued now.
    pub fn touch(&self) {
        self.last
            .store(self.epoch.elapsed().as_millis() as u64, Ordering::Relaxed);
    }

    #[cfg(target_vendor = "apple")]
    pub fn start(device: &Context) -> Result<KeepAlive> {
        use kernels_metal::encode::{Fire, Grid};
        use objc2_metal::{MTLCommandBuffer as _, MTLCommandEncoder as _, MTLCommandQueue as _, MTLComputeCommandEncoder as _, MTLDevice as _};

        let epoch = std::time::Instant::now();
        let last = Arc::new(AtomicU64::new(u64::MAX / 2));
        let stop = Arc::new(AtomicBool::new(false));
        let queue = device.device().newCommandQueue().ok_or(crate::error::Fault::Device {
            call: "newCommandQueue",
            why: "the device would not open the keep-alive queue".to_string(),
        })?;
        let pipelines = crate::device::library::Pipelines::new();
        let fire = Fire::at("layout/keepalive.metal", "keepalive_spin").apply(Grid::of([32, 1, 1], [32, 1, 1]));
        let pipeline = pipelines.at(device.device(), fire)?;
        let sink = crate::device::Buffer::zeroed(device, 256)?;
        // Tuned so one dispatch is a few hundred microseconds: short enough
        // that the real fire's next command buffer is never far behind it.
        let iters: u32 = std::env::var("PIE_METAL_KEEPALIVE_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(20_000);
        let (last_t, stop_t) = (Arc::clone(&last), Arc::clone(&stop));
        let carry = Carry { queue, pipeline, sink };
        let thread = std::thread::Builder::new()
            .name("pie-metal-keepalive".to_string())
            .spawn(move || {
                let carry = carry;
                loop {
                    if stop_t.load(Ordering::Relaxed) {
                        break;
                    }
                    let now = epoch.elapsed().as_millis() as u64;
                    let last = last_t.load(Ordering::Relaxed);
                    if now.saturating_sub(last) > LINGER_MS {
                        std::thread::sleep(std::time::Duration::from_millis(2));
                        continue;
                    }
                    let Some(buffer) = carry.queue.commandBuffer() else { break };
                    let Some(encoder) = buffer.computeCommandEncoder() else { break };
                    encoder.setComputePipelineState(&carry.pipeline);
                    // SAFETY: the sink buffer outlives the command buffer (it is
                    // owned by this thread's `carry`), and `iters` is copied out
                    // by `setBytes:` before the call returns.
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(&**carry.sink.slab()), 0, 0);
                        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&iters).cast(), size_of::<u32>(), 1);
                    }
                    let one = objc2_metal::MTLSize { width: 1, height: 1, depth: 1 };
                    let tg = objc2_metal::MTLSize { width: 32, height: 1, depth: 1 };
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(one, tg);
                    encoder.endEncoding();
                    buffer.commit();
                    buffer.waitUntilCompleted();
                }
            })
            .map_err(|err| crate::error::Fault::Device {
                call: "spawn",
                why: format!("the keep-alive thread would not start: {err}"),
            })?;
        Ok(KeepAlive {
            last,
            stop,
            epoch,
            thread: Some(thread),
        })
    }

    #[cfg(not(target_vendor = "apple"))]
    pub fn start(_device: &Context) -> Result<KeepAlive> {
        Err(crate::error::Fault::Deviceless)
    }
}

#[cfg(target_vendor = "apple")]
struct Carry {
    queue: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
    pipeline: crate::device::library::Pipeline,
    sink: crate::device::Buffer,
}

// SAFETY: Metal command queues, pipeline states and buffers are documented
// thread-safe; the thread is the only user of these three once started.
#[cfg(target_vendor = "apple")]
unsafe impl Send for Carry {}

impl Drop for KeepAlive {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
