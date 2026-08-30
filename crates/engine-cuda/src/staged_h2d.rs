//! **The pinned double-buffered H2D pump** — alto streaming §1's `staged_h2d`,
//! ported from `origin/dev`'s `driver/cuda/src/loader/staged_h2d.hpp`.
//!
//! # What it is
//!
//! A set of host→device copies partitioned round-robin across `lanes` worker
//! threads, each lane double-buffering a small pinned staging buffer so the
//! next host memcpy overlaps the DMA already in flight. It is the one path
//! bulk weight bytes take, so that every bulk mover behaves identically: the
//! warm-boot restore uses it today ([`crate::weight_cache`]), and the dense
//! prefetch schedule of streaming §2 is meant to use the same object.
//!
//! # Why there is no GPUDirect Storage path
//!
//! dev's measurement, carried over verbatim because it is the whole argument
//! for this file's shape. GDS exists to delete the bounce buffer this module
//! is built around, so it is the obvious thing to reach for. Measured on dev's
//! box (RTX 4090, PCIe 4 x16, NVMe ext4) at the 2 MiB chunk these lanes use:
//!
//! ```text
//!     NVMe, O_DIRECT sequential read        4.63 GB/s
//!     memcpy -> pinned -> H2D, ONE lane     7.34 GB/s
//!     pinned -> H2D alone (PCIe ceiling)   19.30 GB/s
//! ```
//!
//! **The bounce is not the bottleneck.** A single staging lane already outruns
//! the disk by 1.6×, and this engine runs four of them, so the bytes arrive as
//! fast as storage can produce them; the copy GDS removes is hidden behind I/O
//! that has to happen anyway. The headroom GDS competes for is the gap between
//! 7.34 and 19.30, and nothing reaches it while the source is a file on this
//! device. Skipping the page cache would also cost the second load of the same
//! checkpoint, which is free today. Independently: GDS needs the `nvidia_fs`
//! kernel module, and without it cuFile silently falls back to a POSIX read
//! into its own bounce buffer — which is this path with an extra layer.
//!
//! So **the number to watch is the disk's, not the copy engine's.** Revisit if
//! checkpoints ever come off storage faster than ~7 GB/s per reader (a RAID of
//! Gen5 NVMe, or a warm page cache serving most of the file), because that is
//! the point where the staging memcpy starts to be the thing in the way.
//!
//! # What is adapted, and why
//!
//! * **No environment, no knobs** (article 9): [`LANES`] and [`CHUNK`] are
//!   constants with a measurement attached, not a configuration surface. They
//!   are statutes — they move with a new measurement, not with a deployment.
//! * **Events are the crate's** ([`Event`]), so the completion gate a lane
//!   reuses a buffer behind is the same handle every other fork/join in this
//!   shell uses. dev's `cudaEventDisableTiming` is [`Event::new`]'s flag.
//! * **Pinned memory is the crate's** ([`Pinned`]), so the staging buffers are
//!   freed by the same `Drop` that frees every other host allocation here.
//! * **Failures are typed and returned**, not left to surface at the caller's
//!   next synchronize. dev's lane loop used raw calls specifically to avoid
//!   throwing across a thread boundary; a `Result` crosses a `JoinHandle`
//!   without that problem, so the fault is answered where it happened.
//! * **A lane always drains before it returns**, error or not, because the
//!   pinned buffers a failed run left in flight are about to be dropped.
//!
//! The streams are made here rather than by [`Context`](crate::device::ctx::Context)
//! on purpose: these are LOAD-TIME lanes with no fire on them and no cuBLAS
//! handle to attach, opened and destroyed inside one call. Nothing captured
//! ever runs on them.

use core::ffi::c_void;

use crate::device::alloc::Pinned;
use crate::device::graph::Event;
use crate::error::{Fault, Result};

/// **How many staging lanes a bulk move opens.**
///
/// dev's four, and dev's reason: one lane measures 7.34 GB/s against an NVMe
/// that produces 4.63, so four is comfortably past every reader this path has
/// ever been pointed at while still costing only `LANES * 2 * CHUNK` of pinned
/// memory (16 MiB) for the duration of a load.
pub const LANES: usize = 4;

/// **How much each staging buffer holds** — dev's `kChunkBytes`.
///
/// Small on purpose: the double buffer's whole job is to keep a memcpy and a
/// DMA in flight together, and a chunk large enough to be worth measuring
/// separately is a chunk large enough to stall its lane at the seams.
pub const CHUNK: usize = 2 << 20;

/// **One host→device copy**: a device address, host bytes, a length.
///
/// The host side is a raw pointer rather than a slice because the sources this
/// pump exists for are mmap'd files whose pages are faulted in BY the memcpy —
/// the copy is the read. The caller owns the mapping for the whole call.
#[derive(Debug, Clone, Copy)]
pub struct Transfer {
    /// The device address the bytes land at. Bounds are the caller's: by the
    /// time an offset reaches here it has already met its length at
    /// [`Buffer::at`](crate::device::alloc::Buffer::at)'s door.
    pub dst: u64,
    /// The first host byte. Must stay mapped and unwritten for the call.
    pub src: *const u8,
    /// How many bytes.
    pub len: u64,
}

/// The chunk list, as something the worker threads may share.
///
/// SAFETY: a [`Transfer`] is two addresses and a length, and what makes them
/// sound to read from four threads is that this list is built before the scope
/// opens and never touched again inside it. The raw pointer is the reason the
/// derive does not happen on its own.
struct Cargo(Vec<Transfer>);

// SAFETY: as above — an immutable list of addresses, read-only for the scope's
// lifetime, pointing into a mapping the caller holds open across the call.
unsafe impl Sync for Cargo {}
// SAFETY: as `Sync`.
unsafe impl Send for Cargo {}

/// One lane: a stream, and the pinned double buffer it feeds.
#[derive(Debug)]
struct Lane {
    stream: *mut c_void,
    pinned: [Pinned; 2],
    /// Recorded after the H2D that reads `pinned[i]`. A lane waits on it
    /// before it overwrites that buffer, and that wait is the whole
    /// synchronization this module has.
    done: [Event; 2],
}

// SAFETY: a `Lane` is a stream handle, two pinned allocations (already `Send`)
// and two events. Every one of them is a plain handle, and the discipline that
// makes the concurrency sound is exclusive ownership: `pump` hands each worker
// thread a `&mut Lane` nobody else holds for the length of one scope, and no
// lane touches another's stream, buffer or event.
unsafe impl Send for Lane {}

/// **A pool of staging lanes**, opened once and pumped as many times as the
/// caller likes.
///
/// dev's `PinnedLanePool`: the executor holds one for an entire load, and the
/// artifact restore makes a local one. Buffers, streams and events are made in
/// [`Lanes::open`] on the calling thread — where a failure is still a
/// `Result` — and freed in `Drop`.
#[derive(Debug)]
pub struct Lanes {
    lanes: Vec<Lane>,
    buf_bytes: usize,
    /// The ordinal every worker thread binds. `cudaSetDevice` is per-thread and
    /// does not travel with a spawn (`Context::bind_thread`'s own lesson).
    device: i32,
}

impl Lanes {
    /// **Open `lanes` staging lanes of `buf_bytes` each.**
    ///
    /// A request for zero lanes is one lane: a pump with nowhere to put bytes
    /// is a bug, not a configuration.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] for a build with no CUDA runtime,
    /// [`Fault::Device`] for a stream, a pinned allocation or an event the
    /// runtime refused. A failure part way through frees what was already
    /// opened, because the partially-built pool is dropped here.
    pub fn open(lanes: usize, buf_bytes: usize) -> Result<Lanes> {
        #[cfg(feature = "_cuda")]
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
        #[cfg(not(feature = "_cuda"))]
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

    /// **Copy every transfer, and block until all of the DMAs have landed.**
    ///
    /// Copies are sliced into sub-chunks of at most one staging buffer and
    /// split into one CONTIGUOUS run per lane, so a single large source (a
    /// whole weight artifact) saturates every lane rather than stalling one,
    /// and each lane's host reads stay sequential for the kernel's read-ahead.
    /// Interleaving the chunks instead would give the page cache four strided
    /// readers, which is dev's stated reason for the contiguous split.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`], or the first [`Fault::Device`] any lane met.
    /// Every lane still drains its stream before the call returns, so no DMA
    /// is left reading a staging buffer this pool is about to free.
    pub fn pump(&mut self, copies: &[Transfer]) -> Result<()> {
        #[cfg(feature = "_cuda")]
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
                    // A panicked lane is reported as a device fault rather
                    // than resumed: the pool's buffers are about to be freed
                    // and unwinding through the scope would free them with a
                    // DMA still reading.
                    .map(|handle| {
                        handle.join().unwrap_or(Err(Fault::Device {
                            call: "staged_h2d lane",
                            code: -1,
                        }))
                    })
                    .collect()
            });
            // Every lane has drained by now, so the first fault can be
            // returned without leaving anything in flight.
            outcomes.into_iter().collect::<Result<Vec<()>>>()?;
            Ok(())
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = copies;
            Err(Fault::Runtimeless)
        }
    }
}

/// One lane's whole life: bind, stream the run through the double buffer,
/// drain.
#[cfg(feature = "_cuda")]
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
            // The buffer this iteration is about to overwrite may still be the
            // source of the H2D two iterations ago. An event that was never
            // recorded returns at once, which is what makes the first two
            // chunks of every lane free.
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

    // **DRAINED WHATEVER HAPPENED.** The pinned buffers above are freed by
    // this pool's `Drop`, and a `cudaFreeHost` under a live DMA is the one
    // failure mode this module could cause that would not look like its own.
    let drained = crate::device::ctx::sync(lane.stream);
    streamed.and(drained)
}

/// The ordinal this thread is bound to, so the workers can bind the same one.
#[cfg(feature = "_cuda")]
fn current_device() -> Result<i32> {
    use cudarc::runtime::sys as rt;

    let mut ordinal: i32 = 0;
    // SAFETY: a live local out-parameter.
    unsafe { crate::device::ctx::check("cudaGetDevice", rt::cudaGetDevice(&raw mut ordinal))? };
    Ok(ordinal)
}

/// A stream for one lane.
#[cfg(feature = "_cuda")]
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
        #[cfg(feature = "_cuda")]
        for lane in &self.lanes {
            if !lane.stream.is_null() {
                // SAFETY: the handle came from this module's own
                // `cudaStreamCreate` and is destroyed exactly once. Every
                // `pump` drained before it returned, so nothing is in flight.
                unsafe {
                    let _ = cudarc::runtime::sys::cudaStreamDestroy(lane.stream.cast());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two statutes, stated as a test so that a change to either one is a
    /// change somebody made on purpose. Sixteen mebibytes of pinned memory for
    /// the length of a load is the whole cost of the mechanism.
    #[test]
    fn four_lanes_of_two_mebibytes_is_sixteen_mebibytes_of_pinned_memory() {
        assert_eq!(LANES, 4);
        assert_eq!(CHUNK, 2 << 20);
        assert_eq!(LANES * 2 * CHUNK, 16 << 20);
    }

    /// **A build with no runtime refuses rather than pretends.** This crate's
    /// standing property — every device call answers [`Fault::Runtimeless`] —
    /// and what lets a plain workspace check carry the host tests below.
    #[test]
    #[cfg(not(feature = "_cuda"))]
    fn a_runtimeless_build_opens_no_lanes() {
        assert!(matches!(Lanes::standard(), Err(Fault::Runtimeless)));
    }
}
