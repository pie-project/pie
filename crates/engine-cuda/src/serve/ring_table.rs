//! The table of the row runs a windowed fire moves around its body (its
//! page fills, its snapshots, its fold): laid at load, at the runs one fold
//! of every lane takes. A fire of more runs goes out in several launches,
//! each over the same table in the stream's order, so nothing is allocated
//! once the load serves.

use core::ffi::c_void;
use std::collections::VecDeque;
use std::ops::Range;

use crate::device::alloc::Pinned;
use crate::device::graph::Event;
use crate::error::Result;

/// A run as the kernel reads it (`kernels_cuda::runtime::channel::CopyRun`).
pub const RUN_BYTES: usize = size_of::<kernels_cuda::runtime::channel::CopyRun>();

/// The copies from the host table that can stand unlanded at once: past
/// them the next waits out the oldest. A fire makes three at most where
/// its runs fit the table; the events are made at load.
const COPIES: usize = 16;

/// The runs one fold (or refill) of `lanes` lanes takes at most: each lane's
/// rows span `pages` ring pages and one more where they start inside one, a
/// run a plane each.
#[must_use]
pub fn fold_runs(lanes: u32, pages: u32, planes: usize) -> usize {
    lanes as usize * (pages as usize + 1) * planes
}

/// A fire's run table: `runs` runs on the device at the place it is laid,
/// staged from a pinned host arena laid with it. A copy's bytes stand in the
/// arena until its event says the device read them.
#[derive(Debug)]
pub struct RingTable {
    /// Where the device's table lies; 0 until it is laid.
    at: u64,
    runs: usize,
    host: Pinned,
    next: usize,
    copies: VecDeque<(Range<usize>, Event)>,
    spare: Vec<Event>,
}

impl RingTable {
    /// The table of `runs` runs, its host arena holding one for each of
    /// `frames` frames in flight and the fire being staged.
    pub fn new(runs: usize, frames: usize) -> Result<RingTable> {
        Ok(RingTable {
            at: 0,
            runs,
            host: Pinned::mapped(runs * RUN_BYTES * (frames + 1))?,
            next: 0,
            copies: VecDeque::new(),
            spare: (0..COPIES).map(|_| Event::new()).collect::<Result<_>>()?,
        })
    }

    /// The device table's bytes.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        (self.runs * RUN_BYTES) as u64
    }

    /// The host arena's bytes.
    #[must_use]
    pub fn host_bytes(&self) -> usize {
        self.host.bytes()
    }

    /// The table lies at `at`, [`Self::bytes`] of device memory.
    pub fn lay(&mut self, at: u64) {
        self.at = at;
    }

    /// Stages `runs` (to, from, bytes) on `stream` and has `launch` copy
    /// them (the table and its runs), a table's worth at a time, each launch
    /// and its staging after the one before it in the stream's order.
    pub fn copy(
        &mut self,
        stream: *mut c_void,
        runs: &[(u64, u64, u64)],
        mut launch: impl FnMut(u64, u32) -> Result<()>,
    ) -> Result<()> {
        for part in runs.chunks(self.runs.max(1)) {
            let bytes = part.len() * RUN_BYTES;
            let (place, event) = self.place(bytes)?;
            let words: Vec<u8> = part
                .iter()
                .flat_map(|&(dst, src, bytes)| [dst, src, bytes])
                .flat_map(u64::to_le_bytes)
                .collect();
            let written = self.host.write(place.start, &words);
            debug_assert!(written, "the arena holds a table");
            crate::device::alloc::copy_any(
                stream,
                self.at,
                self.host.host() as u64 + place.start as u64,
                bytes,
            )?;
            event.record(stream)?;
            self.copies.push_back((place, event));
            launch(self.at, u32::try_from(part.len()).unwrap_or(u32::MAX))?;
        }
        Ok(())
    }

    /// `bytes` of the arena no unlanded copy reads, and an event: the oldest
    /// copies are waited out where they hold either.
    fn place(&mut self, bytes: usize) -> Result<(Range<usize>, Event)> {
        if self.next + bytes > self.host.bytes() {
            self.next = 0;
        }
        let place = self.next..self.next + bytes;
        loop {
            let clear = |held: &Range<usize>| held.end <= place.start || place.end <= held.start;
            if self.copies.iter().all(|(held, _)| clear(held))
                && let Some(event) = self.spare.pop()
            {
                self.next = place.end;
                return Ok((place, event));
            }
            let Some((_, event)) = self.copies.pop_front() else {
                // Unreachable: with no copy standing every event is spare.
                return Ok((place, Event::new()?));
            };
            event.settle()?;
            self.spare.push(event);
        }
    }
}
