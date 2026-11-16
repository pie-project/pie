#![allow(unsafe_code)]
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::device::Context;
use crate::error::Result;

pub struct KeepAlive {
    last: Arc<AtomicU64>,
    busy: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    epoch: std::time::Instant,
    thread: Option<std::thread::JoinHandle<()>>,
    /// The spin kernel, for [`KeepAlive::wake`]'s dispatch on the frames'
    /// own queue.
    #[cfg(target_vendor = "apple")]
    wake: crate::device::library::Pipeline,
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

    /// A fire the device runs from its encoding to its harvest: the spin
    /// stands aside meanwhile, so it fills the device's idle gaps only.
    pub fn begin(&self) {
        self.busy.fetch_add(1, Ordering::Relaxed);
    }

    pub fn end(&self) {
        let _ = self
            .busy
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1));
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
        let busy = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let queue = device
            .device()
            .newCommandQueue()
            .ok_or(crate::error::Fault::Device {
                call: "newCommandQueue",
                why: "the device would not open the keep-alive queue".to_string(),
            })?;
        let pipelines = crate::device::library::Pipelines::new();
        let fire = Fire::at("runtime/keepalive.metal", "keepalive_spin")
            .apply(Grid::of([32, 1, 1], [32, 1, 1]));
        let pipeline = pipelines.at(device.device(), fire)?;
        let wake = pipeline.clone();
        // One of the three tiny fixed buffers outside the pool (this, the
        // warm-load bind placeholder, the shared empty buffer), left there by
        // ruling: what the keepalive kernel writes nothing anyone reads into.
        let sink = crate::device::Buffer::zeroed(device, 256)?;
        let iters: u32 = crate::diag::on().keepalive_iters.unwrap_or(DEFAULT_ITERS);
        let (last_t, stop_t, busy_t) = (Arc::clone(&last), Arc::clone(&stop), Arc::clone(&busy));
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
                    if busy_t.load(Ordering::Relaxed) > 0 {
                        std::thread::sleep(std::time::Duration::from_micros(20));
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
            busy,
            stop,
            epoch,
            thread: Some(thread),
            wake,
        })
    }

    /// Work is coming (`Engine::expect_work`): one spin dispatch on the
    /// frames' own queue, in a frame that declares every slab a fire runs
    /// on, committed without a wait; and the spin thread lingers from now.
    /// After 1 s or more of quiet, the frames' queue's first commit waits
    /// before the GPU starts, by what it names (15 ms over a fixture's
    /// pool, 40-60 ms over Qwen3.5-0.8B's on an M1 Max); a spin on the
    /// keepalive's own queue does not take it. Committed 20 ms or more
    /// ahead of the next fire, this one takes all of it.
    #[cfg(target_vendor = "apple")]
    pub fn wake(&self, device: &Context) -> Result<()> {
        use objc2_metal::MTLComputeCommandEncoder as _;
        self.touch();
        let frame = device.frame()?;
        {
            let encoder = frame.encoder();
            encoder.setComputePipelineState(&self.wake);
            let sink = device.empty();
            let iters: u32 = 1;
            // SAFETY: the shared empty slab outlives the frame (the context
            // holds it), and the kernel writes it only on a value its one
            // iteration never reaches; `iters` is copied by `setBytes:`
            // before the call returns.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&*sink), 0, 0);
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
            let lanes = objc2_metal::MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(one, lanes);
        }
        drop(frame.commit_async(None)?);
        Ok(())
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
