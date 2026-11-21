//! The host-visible room a backend's guest passes stage their host-held
//! roots through and read their status and puts back into, laid once at the
//! load (by the load's shelf, before the pool is sized, so the ledger holds
//! it) and handed out a pass at a time. A pass takes a berth of exactly what
//! it stages and reads back when it is recorded, and gives it back when it
//! lands; nothing is reserved while serving. A pass larger than the whole
//! ring is refused by name; one that does not fit beside the passes still
//! in flight waits for one of them to land (each was submitted, so each
//! lands).

use std::sync::{Arc, Condvar, Mutex};

use crate::error::{Fault, Result};

/// The alignment of a berth in the ring: the widest word a pass copies.
pub const BERTH_ALIGN: u64 = 16;

/// The ring: `buffer` of `bytes`, and the berths taken in it, by first byte.
#[derive(Debug)]
pub struct Ring<T> {
    buffer: T,
    bytes: u64,
    taken: Mutex<Vec<(u64, u64)>>,
    given: Condvar,
}

/// A span of the ring a pass holds until it drops.
#[derive(Debug)]
pub struct Berth<T> {
    ring: Arc<Ring<T>>,
    at: u64,
    bytes: u64,
}

impl<T> Ring<T> {
    /// A ring over the whole of `buffer`, `bytes` long.
    pub fn new(buffer: T, bytes: u64) -> Arc<Ring<T>> {
        Arc::new(Ring {
            buffer,
            bytes,
            taken: Mutex::new(Vec::new()),
            given: Condvar::new(),
        })
    }

    #[must_use]
    pub fn buffer(&self) -> &T {
        &self.buffer
    }

    /// The ring's bytes: the most one pass takes.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The berths held now.
    #[must_use]
    pub fn held(&self) -> usize {
        self.taken.lock().expect("ring lock poisoned").len()
    }

    /// A berth of `bytes`: refused by name past the ring, and, where the
    /// passes in flight leave no room for it, taken once enough of them land.
    pub fn take(self: &Arc<Self>, bytes: u64) -> Result<Berth<T>> {
        let want = bytes.max(1).next_multiple_of(BERTH_ALIGN);
        if want > self.bytes {
            return Err(Fault::Ceiling {
                what: "bytes one guest pass stages and reads back (the guest ring the load laid)",
                need: want,
                have: self.bytes,
            });
        }
        let mut taken = self.taken.lock().expect("ring lock poisoned");
        loop {
            if let Some((index, at)) = room(&taken, want, self.bytes) {
                taken.insert(index, (at, want));
                return Ok(Berth {
                    ring: Arc::clone(self),
                    at,
                    bytes: want,
                });
            }
            taken = self.given.wait(taken).expect("ring lock poisoned");
        }
    }

    fn give(&self, at: u64) {
        let mut taken = self.taken.lock().expect("ring lock poisoned");
        if let Some(index) = taken.iter().position(|&(start, _)| start == at) {
            taken.remove(index);
        }
        self.given.notify_all();
    }
}

/// The first gap of `want` bytes among `taken` (sorted by first byte) in a
/// ring of `bytes`: where it goes in the list, and its first byte.
fn room(taken: &[(u64, u64)], want: u64, bytes: u64) -> Option<(usize, u64)> {
    let mut start = 0u64;
    for (index, &(at, len)) in taken.iter().enumerate() {
        if at - start >= want {
            return Some((index, start));
        }
        start = at + len;
    }
    (bytes - start >= want).then_some((taken.len(), start))
}

impl<T> Berth<T> {
    /// Its first byte in the ring's buffer.
    #[must_use]
    pub fn at(&self) -> u64 {
        self.at
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    #[must_use]
    pub fn buffer(&self) -> &T {
        &self.ring.buffer
    }
}

impl<T> Drop for Berth<T> {
    fn drop(&mut self) {
        self.ring.give(self.at);
    }
}

/// A pass's layout in its berth: what it stages, from the berth's first
/// byte, then its status, then each value it reads back, each at a word.
#[derive(Debug, Default, Clone)]
pub struct PassLayout {
    /// (value, byte in the berth, bytes) a root staged from the host.
    pub staged: Vec<(u32, u64, u64)>,
    /// The status word's byte in the berth.
    pub status: Option<u64>,
    /// (value, byte in the berth, bytes) a value read back.
    pub read: Vec<(u32, u64, u64)>,
    end: u64,
}

impl PassLayout {
    /// The next `bytes` of the berth, refused by name past `room`.
    pub fn next(&mut self, bytes: u64, room: u64) -> Result<u64> {
        let at = self.end.next_multiple_of(4);
        let end = at.saturating_add(bytes);
        if end > room {
            return Err(Fault::Ceiling {
                what: "bytes of a guest pass's berth in the ring",
                need: end,
                have: room,
            });
        }
        self.end = end;
        Ok(at)
    }

    /// Where value `value` was read back to: its byte in the berth and bytes.
    #[must_use]
    pub fn read_at(&self, value: u32) -> Option<(u64, u64)> {
        self.read
            .iter()
            .find(|&&(read, _, _)| read == value)
            .map(|&(_, at, bytes)| (at, bytes))
    }
}

/// The berth a pass of these spans takes: each staged root's bytes, the
/// status word, and each value read back's, each from a word.
#[must_use]
pub fn pass_bytes(
    staged: impl IntoIterator<Item = u64>,
    status: u64,
    read: impl IntoIterator<Item = u64>,
) -> u64 {
    staged
        .into_iter()
        .chain(std::iter::once(status))
        .chain(read)
        .map(|bytes| bytes.next_multiple_of(4))
        .sum()
}

/// The ring a load lays for the guest passes of `seats` frames in flight
/// (each arm's readout and pixel seat) and the one a settle holds beside
/// them, a pool chunk each: the frames' passes take that much before one
/// waits on the frames ahead. A stage whose pass could take more than the
/// whole ring ([`stage_need`]) binds no session.
#[must_use]
pub fn ring_bytes(chunk_bytes: u64, seats: usize) -> u64 {
    chunk_bytes.saturating_mul(seats as u64 + 1)
}

/// The most a pass of stage `at` of `package` takes of the ring
/// ([`pass_bytes`]): every root of the stage the host may stage (a
/// constant, a channel's take or read), its status of `status` bytes, and
/// every value it puts, each of the bytes `span` states for the stage's
/// value. Known at the bind, so a stage whose pass could outgrow the ring
/// binds no session.
pub fn stage_need(
    package: &eta_compiler::codegen::launch::LaunchPackage,
    at: usize,
    status: u64,
    span: impl Fn(u32) -> Option<u64>,
) -> Result<u64> {
    use eta_compiler::codegen::launch::ValueOrigin;
    let local = |global: u32| match package.plan_local(global) {
        Some((stage, local)) if stage == at => local,
        _ => None,
    };
    let bytes = |value: u32| {
        span(value).ok_or_else(|| Fault::Program {
            at: "guest::ring",
            why: format!("value {value} is not stage {at}'s"),
        })
    };
    let staged = package
        .values
        .iter()
        .enumerate()
        .filter(|(_, value)| {
            matches!(
                value.source,
                ValueOrigin::Const | ValueOrigin::ChannelTake | ValueOrigin::ChannelRead
            )
        })
        .filter_map(|(global, _)| local(u32::try_from(global).ok()?))
        .map(bytes)
        .collect::<Result<Vec<u64>>>()?;
    let mut put: Vec<u32> = Vec::new();
    for value in package
        .stages
        .get(at)
        .into_iter()
        .flat_map(|stage| &stage.puts)
        .filter_map(|put| local(put.value))
    {
        if !put.contains(&value) {
            put.push(value);
        }
    }
    let read = put.into_iter().map(bytes).collect::<Result<Vec<u64>>>()?;
    Ok(pass_bytes(staged, status, read).next_multiple_of(BERTH_ALIGN))
}

#[cfg(test)]
mod tests {
    use super::{PassLayout, Ring};

    /// Berths lie first-fit, aligned, and a berth given back is room again.
    #[test]
    fn berths_lie_first_fit_and_come_back() {
        let ring = Ring::new((), 256);
        let a = ring.take(40).unwrap();
        let b = ring.take(64).unwrap();
        assert_eq!((a.at(), a.bytes()), (0, 48));
        assert_eq!((b.at(), b.bytes()), (48, 64));
        drop(a);
        let c = ring.take(16).unwrap();
        assert_eq!(c.at(), 0, "the gap a landed pass left");
        let d = ring.take(128).unwrap();
        assert_eq!(d.at(), 112);
        assert_eq!(ring.held(), 3);
        drop((b, c, d));
        assert_eq!(ring.held(), 0);
    }

    /// A pass past the whole ring is refused by name, never waited for.
    #[test]
    fn a_pass_past_the_ring_is_refused_by_name() {
        let ring = Ring::new((), 256);
        let fault = ring.take(257).expect_err("past the ring").to_string();
        assert!(fault.contains("the guest ring the load laid"), "{fault}");
    }

    /// A pass the passes in flight leave no room for takes its berth once
    /// one of them lands.
    #[test]
    fn a_pass_waits_for_one_in_flight_to_land() {
        let ring = Ring::new((), 256);
        let held = ring.take(200).unwrap();
        let waiter = {
            let ring = std::sync::Arc::clone(&ring);
            std::thread::spawn(move || ring.take(100).map(|berth| berth.at()))
        };
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!waiter.is_finished(), "no room until the first lands");
        drop(held);
        assert_eq!(waiter.join().unwrap().unwrap(), 0);
    }

    /// A pass lays its spans one after another and is refused past its berth.
    #[test]
    fn a_pass_lays_its_spans_inside_its_berth() {
        let mut layout = PassLayout::default();
        assert_eq!(layout.next(6, 32).unwrap(), 0);
        assert_eq!(layout.next(16, 32).unwrap(), 8);
        assert!(layout.next(9, 32).is_err());
    }
}
