#![allow(unsafe_code)]
//! The pinned staging ring a load's weights cross to the device by: each
//! tensor the walk publishes is copied into a pinned slot and enqueued on
//! the ring's own stream, and the walk reads the next while it crosses.
//!
//! Two slots: one filled while the other crosses. The ring is sized from
//! the rates it measures as it opens: the link's copy rate and the size at
//! which a copy first reaches it (smaller copies pay their fixed cost on
//! every slot), and the rate a slot is filled at (a host copy into pinned
//! memory). A slot is at least the size the link reaches its rate at. Where
//! filling is slower than crossing, a slot also holds enough that its
//! crossing (`latency + s / copy`) is no longer than its filling
//! (`s / fill`), so the filler never waits on the link: at least
//! `latency / (1 / fill - 1 / copy)` bytes. Never more than the largest
//! tensor landed.

use core::ffi::c_void;
use std::time::Instant;

use crate::device::alloc::{Buffer, Pinned};
use crate::device::graph::Event;
#[allow(unused_imports)]
use crate::error::{Fault, Result};

/// The rates the ring was sized by, as it measured them.
#[derive(Clone, Copy, Debug)]
pub struct Rates {
    /// Seconds a page-sized copy to the device takes, start to landed.
    pub latency: f64,
    /// Bytes a second the link moves, at the smallest copy that reaches it.
    pub copy: f64,
    /// The smallest copy the link moves at that rate.
    pub saturated: usize,
    /// Bytes a second a host copy into pinned memory fills a slot at.
    pub fill: f64,
}

pub struct Ring {
    stream: *mut c_void,
    slots: Vec<(Pinned, Event)>,
    slot_bytes: usize,
    next: usize,
    pub rates: Rates,
}

impl Ring {
    /// A ring for tensors of at most `largest` bytes, sized by the rates it
    /// measures as it opens.
    pub fn open(largest: usize) -> Result<Ring> {
        let stream = crate::staged_h2d::new_stream()?;
        let opened = (|| {
            let page = page_bytes();
            let rates = measure(stream, largest.max(page))?;
            let overlap = 1.0 / rates.fill - 1.0 / rates.copy;
            let covered = if overlap > 0.0 {
                (rates.latency / overlap) as usize
            } else {
                0
            };
            let wanted = rates.saturated.max(covered);
            let slot_bytes = wanted
                .min(largest.max(page))
                .max(page)
                .next_multiple_of(page);
            let slots = (0..2)
                .map(|_| Ok((Pinned::mapped_uninit(slot_bytes)?, Event::new()?)))
                .collect::<Result<Vec<_>>>()?;
            Ok((rates, slot_bytes, slots))
        })();
        match opened {
            Ok((rates, slot_bytes, slots)) => Ok(Ring {
                stream,
                slots,
                slot_bytes,
                next: 0,
                rates,
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

/// The host's page.
fn page_bytes() -> usize {
    // SAFETY: a query of a constant of the system.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    usize::try_from(page).ok().filter(|&page| page > 0).unwrap_or(4096)
}

/// The link's fixed cost a copy and its rate, and the fill rate: a page
/// copy (after one that pays the stream's start), then copies doubling in
/// size, each timed twice, while a doubling moves bytes faster than the
/// last size did even at its best of the two (so the rate found is past the
/// measurement's own spread), up to the largest tensor; each size also
/// filled from pageable memory.
fn measure(stream: *mut c_void, largest: usize) -> Result<Rates> {
    let page = page_bytes();
    let crossing = |src: &Pinned, dst: &Buffer, bytes: usize| -> Result<f64> {
        let started = Instant::now();
        enqueue(stream, dst.ptr(), src.host(), bytes)?;
        crate::device::ctx::sync(stream)?;
        Ok(started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE))
    };
    let twice = |src: &Pinned, dst: &Buffer, bytes: usize| -> Result<(f64, f64)> {
        let (a, b) = (crossing(src, dst, bytes)?, crossing(src, dst, bytes)?);
        // Bytes a second at the slower and at the faster of the two.
        Ok((bytes as f64 / a.max(b), bytes as f64 / a.min(b)))
    };
    let filling = |from: &[u8], into: &Pinned| -> f64 {
        let started = Instant::now();
        // SAFETY: `into` holds `from.len()` bytes and nothing reads it.
        unsafe { core::ptr::copy_nonoverlapping(from.as_ptr(), into.host(), from.len()) };
        from.len() as f64 / started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE)
    };
    let (src, dst) = (Pinned::mapped_uninit(page)?, Buffer::zeroed_apart(page)?);
    crossing(&src, &dst, page)?;
    let latency = crossing(&src, &dst, page)?;
    let (mut at, mut best) = (page, twice(&src, &dst, page)?);
    let mut fill = 0.0f64;
    let mut bytes = page;
    while bytes < largest {
        bytes = (bytes * 2).min(largest);
        let (src, dst) = (Pinned::mapped_uninit(bytes)?, Buffer::zeroed_apart(bytes)?);
        // A pageable source, its pages made before it is timed.
        let from = vec![1u8; bytes];
        fill = fill.max(filling(&from, &src));
        let rates = twice(&src, &dst, bytes)?;
        if rates.0 <= best.1 {
            break;
        }
        (at, best) = (bytes, rates);
    }
    Ok(Rates {
        latency,
        copy: best.1,
        saturated: at,
        fill: if fill > 0.0 { fill } else { best.1 },
    })
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
