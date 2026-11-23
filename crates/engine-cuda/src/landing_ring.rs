#![allow(unsafe_code)]
//! The pinned staging ring a load's weights cross to the device by: each
//! tensor the walk publishes is copied into a pinned slot and enqueued on
//! the ring's own stream, and the walk reads the next while it crosses.
//!
//! The ring is sized from the device's copy rate, measured when it opens:
//! a slot holds what the link moves in a hundred times one copy's fixed
//! cost (the latency of a page-sized copy), so that cost is at most a
//! hundredth of a slot's crossing; two slots, one filled while the other
//! crosses. A slot is never larger than the largest tensor landed.

use core::ffi::c_void;
use std::time::Instant;

use crate::device::alloc::{Buffer, Pinned};
use crate::device::graph::Event;
#[allow(unused_imports)]
use crate::error::{Fault, Result};

/// The share of a slot's crossing one copy's fixed cost may take.
const OVERHEAD_SHARE: f64 = 0.01;

/// The device's copy rate as the ring measured it.
#[derive(Clone, Copy, Debug)]
pub struct CopyRate {
    /// Seconds a page-sized copy takes, start to landed.
    pub latency: f64,
    /// Bytes a second a large copy moves.
    pub bytes_per_second: f64,
}

pub struct Ring {
    stream: *mut c_void,
    slots: Vec<(Pinned, Event)>,
    slot_bytes: usize,
    next: usize,
    pub rate: CopyRate,
}

impl Ring {
    /// A ring for tensors of at most `largest` bytes, sized by the copy rate
    /// it measures as it opens.
    pub fn open(largest: usize) -> Result<Ring> {
        let stream = crate::staged_h2d::new_stream()?;
        let opened = (|| {
            let rate = measure(stream, largest)?;
            let wanted = rate.bytes_per_second * rate.latency / OVERHEAD_SHARE;
            let page = 4096usize;
            let slot_bytes = (wanted as usize)
                .min(largest.max(page))
                .max(page)
                .next_multiple_of(page);
            let slots = (0..2)
                .map(|_| Ok((Pinned::mapped_uninit(slot_bytes)?, Event::new()?)))
                .collect::<Result<Vec<_>>>()?;
            Ok((rate, slot_bytes, slots))
        })();
        match opened {
            Ok((rate, slot_bytes, slots)) => Ok(Ring {
                stream,
                slots,
                slot_bytes,
                next: 0,
                rate,
            }),
            Err(fault) => {
                destroy(stream);
                Err(fault)
            }
        }
    }

    #[must_use]
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// Enqueues `bytes` to land at device address `dst`: each slot's worth
    /// copied into the next free slot and sent on the ring's stream. It
    /// returns once the last piece is enqueued, its copy still crossing.
    pub fn send(&mut self, dst: u64, bytes: &[u8]) -> Result<()> {
        for (at, piece) in bytes.chunks(self.slot_bytes).enumerate() {
            let (slot, done) = &self.slots[self.next];
            // The slot's last copy has left it.
            done.settle()?;
            // SAFETY: `piece` fits the slot (chunked by its size), and no
            // copy reads the slot any more.
            unsafe { core::ptr::copy_nonoverlapping(piece.as_ptr(), slot.host(), piece.len()) };
            let to = dst + (at * self.slot_bytes) as u64;
            enqueue(self.stream, to, slot.host(), piece.len())?;
            done.record(self.stream)?;
            self.next = (self.next + 1) % self.slots.len();
        }
        Ok(())
    }

    /// Every copy sent has landed.
    pub fn drain(&self) -> Result<()> {
        crate::device::ctx::sync(self.stream)
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        let _ = self.drain();
        destroy(self.stream);
    }
}

/// The copy rate of this device: a page-sized copy's latency, then copies
/// doubling in size from it until one takes a hundred latencies (its rate
/// the link's, the latency a hundredth of it), or reaches `largest`.
fn measure(stream: *mut c_void, largest: usize) -> Result<CopyRate> {
    let page = 4096usize;
    let timed = |src: &Pinned, dst: &Buffer, bytes: usize| -> Result<f64> {
        let started = Instant::now();
        enqueue(stream, dst.ptr(), src.host(), bytes)?;
        crate::device::ctx::sync(stream)?;
        Ok(started.elapsed().as_secs_f64())
    };
    let small = Pinned::mapped_uninit(page)?;
    let small_dst = Buffer::zeroed_apart(page)?;
    // The first copy pays the stream's start: the second is the latency.
    timed(&small, &small_dst, page)?;
    let latency = timed(&small, &small_dst, page)?.max(1e-7);
    drop((small, small_dst));
    let mut bytes = page;
    loop {
        let next = (bytes * 2).min(largest.max(page));
        let src = Pinned::mapped_uninit(next)?;
        let dst = Buffer::zeroed_apart(next)?;
        let took = timed(&src, &dst, next)?;
        bytes = next;
        if took >= latency / OVERHEAD_SHARE || bytes >= largest {
            return Ok(CopyRate {
                latency,
                bytes_per_second: bytes as f64 / took.max(1e-9),
            });
        }
    }
}

fn enqueue(stream: *mut c_void, dst: u64, src: *const u8, bytes: usize) -> Result<()> {
    #[cfg(feature = "device")]
    {
        use crate::device::rt;
        // SAFETY: `src` is pinned host memory of at least `bytes`, `dst` a
        // device span the caller bounded, and the stream this ring owns.
        unsafe {
            crate::device::ctx::check(
                "cudaMemcpyAsync",
                rt::cudaMemcpyAsync(
                    dst as *mut c_void,
                    src.cast(),
                    bytes,
                    rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                    stream.cast(),
                ),
            )
        }
    }
    #[cfg(not(feature = "device"))]
    {
        let _ = (stream, dst, src, bytes);
        Err(Fault::Runtimeless)
    }
}

fn destroy(stream: *mut c_void) {
    #[cfg(feature = "device")]
    if !stream.is_null() {
        // SAFETY: a stream this module created, destroyed once.
        unsafe {
            let _ = crate::device::rt::cudaStreamDestroy(stream.cast());
        }
    }
    #[cfg(not(feature = "device"))]
    let _ = stream;
}
