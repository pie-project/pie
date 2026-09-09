use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::device::Context;
use crate::error::Result;

pub struct KeepAlive {
    last: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    epoch: std::time::Instant,
    thread: Option<std::thread::JoinHandle<()>>,
}

#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
const LINGER_MS: u64 = 250;

#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
const DEFAULT_ITERS: u32 = 20_000;

impl KeepAlive {
    #[must_use]
    pub fn wanted() -> bool {
        crate::diag::on().keepalive
    }

    pub fn touch(&self) {
        self.last
            .store(self.epoch.elapsed().as_millis() as u64, Ordering::Relaxed);
    }

    #[cfg(target_vendor = "apple")]
    pub fn start(device: &Context) -> Result<KeepAlive> {
        use kernels_metal::encode::{Fire, Grid};
        use objc2_metal::{
            MTLCommandBuffer as _, MTLCommandEncoder as _, MTLCommandQueue as _,
            MTLComputeCommandEncoder as _, MTLDevice as _,
        };

        let epoch = std::time::Instant::now();
        let last = Arc::new(AtomicU64::new(u64::MAX / 2));
        let stop = Arc::new(AtomicBool::new(false));
        let queue = device
            .device()
            .newCommandQueue()
            .ok_or(crate::error::Fault::Device {
                call: "newCommandQueue",
                why: "the device would not open the keep-alive queue".to_string(),
            })?;
        let pipelines = crate::device::library::Pipelines::new();
        let fire = Fire::at("layout/keepalive.metal", "keepalive_spin")
            .apply(Grid::of([32, 1, 1], [32, 1, 1]));
        let pipeline = pipelines.at(device.device(), fire)?;
        let sink = crate::device::Buffer::zeroed(device, 256)?;
        let iters: u32 = crate::diag::on().keepalive_iters.unwrap_or(DEFAULT_ITERS);
        let (last_t, stop_t) = (Arc::clone(&last), Arc::clone(&stop));
        let carry = Carry {
            queue,
            pipeline,
            sink,
        };
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
                    let spun = objc2::rc::autoreleasepool(|_| {
                        let Some(buffer) = carry.queue.commandBuffer() else {
                            return false;
                        };
                        let Some(encoder) = buffer.computeCommandEncoder() else {
                            return false;
                        };
                        encoder.setComputePipelineState(&carry.pipeline);
                        // SAFETY: the sink buffer outlives the command buffer (it is
                        // owned by this thread's `carry`), and `iters` is copied out
                        // by `setBytes:` before the call returns.
                        unsafe {
                            encoder.setBuffer_offset_atIndex(Some(&**carry.sink.slab()), 0, 0);
                            encoder.setBytes_length_atIndex(
                                std::ptr::NonNull::from(&iters).cast(),
                                size_of::<u32>(),
                                1,
                            );
                        }
                        let one = objc2_metal::MTLSize {
                            width: 1,
                            height: 1,
                            depth: 1,
                        };
                        let tg = objc2_metal::MTLSize {
                            width: 32,
                            height: 1,
                            depth: 1,
                        };
                        encoder.dispatchThreadgroups_threadsPerThreadgroup(one, tg);
                        encoder.endEncoding();
                        buffer.commit();
                        buffer.waitUntilCompleted();
                        true
                    });
                    if !spun {
                        break;
                    }
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
