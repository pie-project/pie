//! The host staging ring a load lays for every guest pass (ruling (v)(2)):
//! one host-mapped buffer, laid at load from the fire's bound
//! ([`crate::program::GuestIo`]), a lane's room for every lane of every
//! fire in flight. A pass takes a run of lane rooms ([`Ring::take`]) for
//! the lanes it runs, stages its roots in and lands its puts and fault word
//! there, and gives them back when it lands ([`Region`] drops). Nothing is
//! allocated for a pass: a pass that finds no room waits for one in flight
//! to land.
//!
//! The ring is host memory the device writes and the host maps to read
//! (`Memory::Readback`): WebGPU maps one range of a buffer at a time, so the
//! host's reads of it go one after another ([`Ring::read`]). Where the API
//! stages host bytes through its queue (WebGPU's writes), a pass's staged
//! side stands unused.

use std::sync::{Arc, Condvar, Mutex};

use crate::backend::{Backend, Buffer as _};
use crate::error::{Fault, Result};
use crate::program::GuestIo;

pub struct Ring<B: Backend> {
    buffer: B::Buffer,
    io: GuestIo,
    /// Which lane rooms a pass holds, and where the next search starts.
    rooms: Mutex<(Vec<bool>, usize)>,
    given: Condvar,
    /// Held across a host read of the buffer.
    reading: Mutex<()>,
}

impl<B: Backend> std::fmt::Debug for Ring<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ring")
            .field("bytes", &self.buffer.bytes())
            .field("io", &self.io)
            .finish()
    }
}

impl<B: Backend> Ring<B> {
    /// The ring for `io`, host-mapped, laid now (at load).
    pub fn lay(device: &B::Context, io: GuestIo) -> Result<Arc<Ring<B>>> {
        let rooms = io.lanes.max(1) as usize * io.depth.max(1) as usize;
        Ok(Arc::new(Ring {
            buffer: B::Buffer::with(
                device,
                io.ring_bytes(),
                crate::backend::Memory::Readback,
                crate::books::owners::GUEST,
            )?,
            io,
            rooms: Mutex::new((vec![false; rooms], 0)),
            given: Condvar::new(),
            reading: Mutex::new(()),
        }))
    }

    #[must_use]
    pub fn io(&self) -> GuestIo {
        self.io
    }

    #[must_use]
    pub fn buffer(&self) -> &B::Buffer {
        &self.buffer
    }

    /// A run of `lanes` adjacent lane rooms for one pass, waiting for a pass
    /// in flight to give some back where none is free. Refused where the
    /// ring holds fewer rooms than `lanes`.
    pub fn take(self: &Arc<Self>, lanes: u32) -> Result<Region<B>> {
        let want = lanes.max(1) as usize;
        let mut rooms = self.rooms.lock().expect("ring lock poisoned");
        let total = rooms.0.len();
        if want > total {
            return Err(Fault::Ceiling {
                what: "lane rooms of the guest staging ring",
                need: want as u64,
                have: total as u64,
            });
        }
        loop {
            let (held, next) = &mut *rooms;
            let start = *next % total;
            let found = (start..=total - want)
                .chain(0..start.min(total - want + 1))
                .find(|&first| held[first..first + want].iter().all(|taken| !taken));
            if let Some(first) = found {
                held[first..first + want].iter_mut().for_each(|taken| *taken = true);
                *next = first + want;
                return Ok(Region {
                    ring: Arc::clone(self),
                    first,
                    lanes: want,
                });
            }
            rooms = self.given.wait(rooms).expect("ring lock poisoned");
        }
    }

    /// Reads `into` from ring byte `at`, one host read of the ring at a
    /// time.
    pub fn read(&self, at: u64, into: &mut [u8]) -> Result<()> {
        let _held = self.reading.lock().expect("ring read lock poisoned");
        self.buffer.read(at, into)
    }

    fn give(&self, first: usize, lanes: usize) {
        let mut rooms = self.rooms.lock().expect("ring lock poisoned");
        rooms.0[first..first + lanes]
            .iter_mut()
            .for_each(|taken| *taken = false);
        self.given.notify_all();
    }
}

/// A pass's lane rooms: its staged roots, then its landed puts, then its
/// fault word, each from an aligned byte. Given back as it drops.
pub struct Region<B: Backend> {
    ring: Arc<Ring<B>>,
    first: usize,
    lanes: usize,
}

impl<B: Backend> std::fmt::Debug for Region<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Region")
            .field("first", &self.first)
            .field("lanes", &self.lanes)
            .finish()
    }
}

impl<B: Backend> Region<B> {
    fn base(&self) -> u64 {
        self.first as u64 * self.ring.io.lane_room()
    }

    fn side(&self) -> u64 {
        self.lanes as u64
            * self
                .ring
                .io
                .lane_bytes
                .next_multiple_of(crate::guest::session::RING_ALIGN)
    }

    /// The ring's buffer.
    #[must_use]
    pub fn buffer(&self) -> &B::Buffer {
        &self.ring.buffer
    }

    /// The bytes the pass stages or lands, each way.
    #[must_use]
    pub fn bound(&self) -> u64 {
        self.lanes as u64 * self.ring.io.lane_bytes
    }

    /// The ring byte the pass's staged roots start at.
    #[must_use]
    pub fn staged(&self) -> u64 {
        self.base()
    }

    /// The ring byte the pass's landed puts start at.
    #[must_use]
    pub fn landed(&self) -> u64 {
        self.base() + self.side()
    }

    /// The ring byte the pass's fault word lands at.
    #[must_use]
    pub fn status(&self) -> u64 {
        self.base() + 2 * self.side()
    }

    /// Reads `into` from ring byte `at` ([`Ring::read`]).
    pub fn read(&self, at: u64, into: &mut [u8]) -> Result<()> {
        self.ring.read(at, into)
    }

    /// The fault word the pass landed: `None` where it ran clean.
    pub fn fault(&self) -> Result<Option<u32>> {
        let mut word = [0u8; 4];
        self.ring.read(self.status(), &mut word)?;
        let code = u32::from_le_bytes(word);
        Ok((code != 0).then_some(code))
    }
}

impl<B: Backend> Drop for Region<B> {
    fn drop(&mut self) {
        self.ring.give(self.first, self.lanes);
    }
}
