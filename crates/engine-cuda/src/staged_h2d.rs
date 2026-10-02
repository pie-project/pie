use core::ffi::c_void;

use crate::device::alloc::Pinned;
use crate::device::graph::Event;
use crate::error::{Fault, Result};

pub const LANES: usize = 4;

pub const CHUNK: usize = 2 << 20;

#[derive(Debug, Clone, Copy)]
pub struct Transfer {
    pub dst: u64,
    pub src: *const u8,
    pub len: u64,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct Cargo(Vec<Transfer>);

// SAFETY: read-only list of addresses for the scope's lifetime.
unsafe impl Sync for Cargo {}
// SAFETY: same as Sync.
unsafe impl Send for Cargo {}

#[derive(Debug)]
struct Lane {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    stream: *mut c_void,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pinned: [Pinned; 2],
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    done: [Event; 2],
}

// SAFETY: sound via exclusive ownership — `pump` hands each worker thread a
// `&mut Lane` nobody else holds, and no lane touches another's stream, buffer,
// or event.
unsafe impl Send for Lane {}

#[derive(Debug)]
pub struct Lanes {
    lanes: Vec<Lane>,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    buf_bytes: usize,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    device: i32,
}

impl Lanes {
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

    pub fn standard() -> Result<Lanes> {
        Lanes::open(LANES, CHUNK)
    }

    #[must_use]
    pub fn width(&self) -> usize {
        self.lanes.len()
    }

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
                    .map(|handle| {
                        handle.join().unwrap_or(Err(Fault::Device {
                            call: "staged_h2d lane",
                            code: -1,
                        }))
                    })
                    .collect()
            });
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

    let drained = crate::device::ctx::sync(lane.stream);
    streamed.and(drained)
}

#[cfg(feature = "cuda")]
fn current_device() -> Result<i32> {
    use cudarc::runtime::sys as rt;

    let mut ordinal: i32 = 0;
    // SAFETY: a live local out-parameter.
    unsafe { crate::device::ctx::check("cudaGetDevice", rt::cudaGetDevice(&raw mut ordinal))? };
    Ok(ordinal)
}

#[cfg(feature = "cuda")]
fn new_stream() -> Result<*mut c_void> {
    use cudarc::runtime::sys as rt;

    let mut stream: rt::cudaStream_t = core::ptr::null_mut();
    // SAFETY: a live local out-parameter; the stream is this pool's and is
    // destroyed exactly once in `Drop`.
    unsafe {
        crate::device::ctx::check("cudaStreamCreate", rt::cudaStreamCreate(&raw mut stream))?
    };
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

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn a_runtimeless_build_opens_no_lanes() {
        assert!(matches!(Lanes::standard(), Err(Fault::Runtimeless)));
    }
}
