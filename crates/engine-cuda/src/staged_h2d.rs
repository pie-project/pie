//! Pinned double-buffered H2D pump: host->device copies partitioned
//! round-robin across worker lanes, each double-buffering a pinned staging
//! buffer so the next memcpy overlaps the DMA already in flight. GDS is not
//! used: a single lane already outruns NVMe O_DIRECT throughput.

use core::ffi::c_void;

use crate::device::alloc::Pinned;
use crate::device::graph::Event;
use crate::error::{Fault, Result};

/// How many staging lanes a bulk move opens. Comfortably past every NVMe
/// reader measured; costs `LANES * 2 * CHUNK` (16 MiB) pinned memory per load.
pub const LANES: usize = 4;

/// How much each staging buffer holds. Small on purpose: large enough to
/// overlap memcpy/DMA, small enough not to stall the lane at the seams.
pub const CHUNK: usize = 2 << 20;

/// One host->device copy: a device address, host bytes, a length.
///
/// `src` is a raw pointer, not a slice: sources are mmap'd files whose pages
/// are faulted in by the memcpy itself. Caller must keep the mapping alive
/// for the whole call.
#[derive(Debug, Clone, Copy)]
pub struct Transfer {
    /// The device address the bytes land at. Bounds are checked by the
    /// caller before reaching here.
    pub dst: u64,
    /// The first host byte. Must stay mapped and unwritten for the call.
    pub src: *const u8,
    /// How many bytes.
    pub len: u64,
}

/// The chunk list, shared read-only across worker threads.
///
/// SAFETY: built before the scope opens and never mutated inside it.
struct Cargo(Vec<Transfer>);

// SAFETY: read-only list of addresses for the scope's lifetime.
unsafe impl Sync for Cargo {}
// SAFETY: same as Sync.
unsafe impl Send for Cargo {}

/// One lane: a stream, and the pinned double buffer it feeds.
#[derive(Debug)]
struct Lane {
    stream: *mut c_void,
    pinned: [Pinned; 2],
    /// Recorded after the H2D that reads `pinned[i]`. A lane waits on it
    /// before overwriting that buffer.
    done: [Event; 2],
}

// SAFETY: sound via exclusive ownership — `pump` hands each worker thread a
// `&mut Lane` nobody else holds, and no lane touches another's stream, buffer,
// or event.
unsafe impl Send for Lane {}

/// A pool of staging lanes, opened once and pumped as many times as the
/// caller likes. Buffers, streams and events are made in [`Lanes::open`] and
/// freed in `Drop`.
#[derive(Debug)]
pub struct Lanes {
    lanes: Vec<Lane>,
    buf_bytes: usize,
    /// The ordinal every worker thread binds: `cudaSetDevice` is per-thread
    /// and does not travel with a spawn.
    device: i32,
}

impl Lanes {
    /// Open `lanes` staging lanes of `buf_bytes` each. Zero lanes opens one.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] for a build with no CUDA runtime,
    /// [`Fault::Device`] for a stream, allocation, or event the runtime
    /// refused.
    pub fn open(lanes: usize, buf_bytes: usize) -> Result<Lanes> {
        #[cfg(feature = "cuda")]
        {
            let buf_bytes = buf_bytes.max(1);
            let device = current_device()?;
            let mut open = Vec::with_capacity(lanes.max(1));
            for _ in 0..lanes.max(1) {
                open.push(Lane {
                    stream: new_stream()?,
                    pinned: [Pinned::mapped(buf_bytes)?, Pinned::mapped(buf_bytes)?],
                    done: [Event::new()?, Event::new()?],
                });
            }
            Ok(Lanes {
                lanes: open,
                buf_bytes,
                device,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (lanes, buf_bytes);
            Err(Fault::Runtimeless)
        }
    }

    /// The pool [`LANES`] and [`CHUNK`] describe — what a bulk move opens
    /// unless it has a measured reason to open something else.
    ///
    /// # Errors
    ///
    /// As [`Lanes::open`].
    pub fn standard() -> Result<Lanes> {
        Lanes::open(LANES, CHUNK)
    }

    /// How many lanes are open.
    #[must_use]
    pub fn width(&self) -> usize {
        self.lanes.len()
    }

    /// Copy every transfer, and block until all DMAs have landed. Sliced into
    /// sub-chunks of at most one staging buffer, one contiguous run per lane
    /// so each lane's host reads stay sequential for read-ahead.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`], or the first [`Fault::Device`] any lane met.
    /// Every lane drains its stream before returning.
    pub fn pump(&mut self, copies: &[Transfer]) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            if copies.is_empty() {
                return Ok(());
            }
            let cap = self.buf_bytes as u64;
            let mut chunks: Vec<Transfer> = Vec::new();
            for copy in copies {
                let mut at = 0u64;
                while at < copy.len {
                    let take = cap.min(copy.len - at);
                    chunks.push(Transfer {
                        dst: copy.dst + at,
                        // SAFETY: `at < copy.len`, and the caller's contract is
                        // that `[src, src + len)` is one live mapping.
                        src: unsafe { copy.src.add(usize::try_from(at).unwrap_or(usize::MAX)) },
                        len: take,
                    });
                    at += take;
                }
            }
            if chunks.is_empty() {
                return Ok(());
            }
            let width = self.lanes.len();
            let per_lane = chunks.len().div_ceil(width);
            let cargo = Cargo(chunks);
            let device = self.device;

            let outcomes: Vec<Result<()>> = std::thread::scope(|scope| {
                let mut running = Vec::with_capacity(width);
                for (at, lane) in self.lanes.iter_mut().enumerate() {
                    let cargo = &cargo;
                    running.push(scope.spawn(move || {
                        let begin = (at * per_lane).min(cargo.0.len());
                        let end = (begin + per_lane).min(cargo.0.len());
                        run_lane(device, lane, &cargo.0[begin..end])
                    }));
                }
                running
                    .into_iter()
                    // a panicked lane is a device fault, not resumed: unwinding
                    // through the scope would free the buffers under a live DMA
                    .map(|handle| {
                        handle.join().unwrap_or(Err(Fault::Device {
                            call: "staged_h2d lane",
                            code: -1,
                        }))
                    })
                    .collect()
            });
            // every lane has drained by now
            outcomes.into_iter().collect::<Result<Vec<()>>>()?;
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = copies;
            Err(Fault::Runtimeless)
        }
    }
}

/// One lane's whole life: bind, stream the run through the double buffer,
/// drain.
#[cfg(feature = "cuda")]
fn run_lane(device: i32, lane: &mut Lane, run: &[Transfer]) -> Result<()> {
    use cudarc::runtime::sys as rt;

    if run.is_empty() {
        return Ok(());
    }
    // SAFETY: an ordinal the calling thread was already bound to.
    unsafe { crate::device::ctx::check("cudaSetDevice", rt::cudaSetDevice(device))? };

    let streamed = (|| -> Result<()> {
        let mut buf = 0usize;
        for chunk in run {
            // this buffer may still be the source of the H2D two iterations
            // ago; an event never recorded returns at once
            lane.done[buf].settle()?;
            let take = usize::try_from(chunk.len).unwrap_or(usize::MAX);
            // SAFETY: `take <= buf_bytes` by the chunking above, so the
            // destination holds it; the source is the caller's live mapping.
            unsafe {
                core::ptr::copy_nonoverlapping(chunk.src, lane.pinned[buf].host(), take);
            }
            // SAFETY: pinned host memory, a device address the caller bounded,
            // and a stream this lane owns.
            unsafe {
                crate::device::ctx::check(
                    "cudaMemcpyAsync",
                    rt::cudaMemcpyAsync(
                        chunk.dst as *mut c_void,
                        lane.pinned[buf].host().cast(),
                        take,
                        rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                        lane.stream.cast(),
                    ),
                )?;
            }
            lane.done[buf].record(lane.stream)?;
            buf ^= 1;
        }
        Ok(())
    })();

    // drains regardless of outcome: pinned buffers are freed by Drop, and a
    // free under a live DMA must not happen
    let drained = crate::device::ctx::sync(lane.stream);
    streamed.and(drained)
}

/// The ordinal this thread is bound to, so the workers can bind the same one.
#[cfg(feature = "cuda")]
fn current_device() -> Result<i32> {
    use cudarc::runtime::sys as rt;

    let mut ordinal: i32 = 0;
    // SAFETY: a live local out-parameter.
    unsafe { crate::device::ctx::check("cudaGetDevice", rt::cudaGetDevice(&raw mut ordinal))? };
    Ok(ordinal)
}

/// A stream for one lane.
#[cfg(feature = "cuda")]
fn new_stream() -> Result<*mut c_void> {
    use cudarc::runtime::sys as rt;

    let mut stream: rt::cudaStream_t = core::ptr::null_mut();
    // SAFETY: a live local out-parameter; the stream is this pool's and is
    // destroyed exactly once in `Drop`.
    unsafe { crate::device::ctx::check("cudaStreamCreate", rt::cudaStreamCreate(&raw mut stream))? };
    Ok(stream.cast())
}

impl Drop for Lanes {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        for lane in &self.lanes {
            if !lane.stream.is_null() {
                // SAFETY: handle from this module's own cudaStreamCreate,
                // destroyed exactly once; every pump drained before returning.
                unsafe {
                    let _ = cudarc::runtime::sys::cudaStreamDestroy(lane.stream.cast());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    /// No CUDA runtime: every device call returns Fault::Runtimeless.
    #[test]
    #[cfg(not(feature = "cuda"))]
    fn a_runtimeless_build_opens_no_lanes() {
        assert!(matches!(Lanes::standard(), Err(Fault::Runtimeless)));
    }
}
