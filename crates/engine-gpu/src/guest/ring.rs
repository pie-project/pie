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
        let want = berth_bytes(bytes);
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

/// The berth a pass of `bytes` takes of the ring: whole aligned words.
#[must_use]
pub fn berth_bytes(bytes: u64) -> u64 {
    bytes.max(1).next_multiple_of(BERTH_ALIGN)
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

/// A pass's layout in its berth, laid before the berth is taken: each value
/// it may stage from the host, from the berth's first byte, then its
/// status, then each value it may read back, each from a word and each
/// once, however many of a fire's values the stage's compile folded into
/// it (two equal constants, a put of a value another put names). The berth
/// a pass takes is [`PassLayout::bytes`] of this layout, and every copy the
/// pass records lands at the byte this layout gave its value, so a pass
/// never asks for more of its berth than it took.
#[derive(Debug, Default, Clone)]
pub struct PassLayout {
    /// (value, byte in the berth, bytes) a root the host may stage.
    stage: Vec<(u32, u64, u64)>,
    status_at: u64,
    /// (value, byte in the berth, bytes) a value the host may read back.
    back: Vec<(u32, u64, u64)>,
    end: u64,
    /// Which of `stage` the pass staged, which of `back` it read back, and
    /// whether it copied its status.
    staged: Vec<bool>,
    read: Vec<bool>,
    status: bool,
}

impl PassLayout {
    /// The layout of a pass that may stage the roots `staged` and read back
    /// `read` (each a value and its bytes; a value named twice is laid
    /// once), its status of `status` bytes between them.
    pub fn plan(
        staged: impl IntoIterator<Item = (u32, u64)>,
        status: u64,
        read: impl IntoIterator<Item = (u32, u64)>,
    ) -> PassLayout {
        let mut end = 0u64;
        let mut lay = |bytes: u64| {
            let at = end;
            end = (at + bytes).next_multiple_of(4);
            at
        };
        let mut stage: Vec<(u32, u64, u64)> = Vec::new();
        for (value, bytes) in staged {
            if !stage.iter().any(|&(laid, _, _)| laid == value) {
                stage.push((value, lay(bytes), bytes));
            }
        }
        let status_at = lay(status);
        let mut back: Vec<(u32, u64, u64)> = Vec::new();
        for (value, bytes) in read {
            if !back.iter().any(|&(laid, _, _)| laid == value) {
                back.push((value, lay(bytes), bytes));
            }
        }
        PassLayout {
            staged: vec![false; stage.len()],
            read: vec![false; back.len()],
            stage,
            status_at,
            back,
            end,
            status: false,
        }
    }

    /// The berth the pass takes: every span it laid.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.end
    }

    /// Where root `value` stages to: its byte in the berth and bytes, marked
    /// staged. A value the layout did not lay is the caller's error, by name.
    pub fn stage(&mut self, value: u32) -> Result<(u64, u64)> {
        let index = self
            .stage
            .iter()
            .position(|&(laid, _, _)| laid == value)
            .ok_or_else(|| Fault::Program {
                at: "guest::ring",
                why: format!("value {value} was staged and its pass laid no span for it"),
            })?;
        self.staged[index] = true;
        let (_, at, bytes) = self.stage[index];
        Ok((at, bytes))
    }

    /// Where the status word lands, marked copied.
    pub fn status(&mut self) -> u64 {
        self.status = true;
        self.status_at
    }

    /// Where value `value` reads back to: its byte in the berth and bytes,
    /// marked read. A value the layout did not lay is the caller's error.
    pub fn read_back(&mut self, value: u32) -> Result<(u64, u64)> {
        let index = self
            .back
            .iter()
            .position(|&(laid, _, _)| laid == value)
            .ok_or_else(|| Fault::Program {
                at: "guest::ring",
                why: format!("value {value} was read back and its pass laid no span for it"),
            })?;
        self.read[index] = true;
        let (_, at, bytes) = self.back[index];
        Ok((at, bytes))
    }

    /// (value, byte in the berth, bytes) of each root the pass staged.
    pub fn staged(&self) -> impl Iterator<Item = (u32, u64, u64)> + '_ {
        self.stage
            .iter()
            .zip(&self.staged)
            .filter(|&(_, &staged)| staged)
            .map(|(&span, _)| span)
    }

    /// The status word's byte in the berth, once the pass copied it there.
    #[must_use]
    pub fn status_at(&self) -> Option<u64> {
        self.status.then_some(self.status_at)
    }

    /// Where value `value` was read back to: its byte in the berth and bytes.
    #[must_use]
    pub fn read_at(&self, value: u32) -> Option<(u64, u64)> {
        self.back
            .iter()
            .zip(&self.read)
            .find(|&(&(laid, _, _), &read)| read && laid == value)
            .map(|(&(_, at, bytes), _)| (at, bytes))
    }
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

/// The layout of a pass of stage `at` of `package` over the stage's roots
/// `roots` and the values `wanted` the host may read back (global ids):
/// each root the host may stage (all but an output's, which the device
/// binds), its status of `status` bytes, and each wanted value, each of the
/// bytes `span` states for the stage's value and each once. A fire lays its
/// pass by this, and the bind its most ([`stage_need`]), so the two are one
/// computation.
pub fn pass_layout(
    package: &eta_compiler::codegen::launch::LaunchPackage,
    at: usize,
    roots: &[u32],
    wanted: &[u32],
    status: u64,
    span: impl Fn(u32) -> Option<u64>,
) -> Result<PassLayout> {
    use eta_compiler::codegen::launch::ValueOrigin;
    let sized = |local: u32| {
        span(local)
            .map(|bytes| (local, bytes))
            .ok_or_else(|| Fault::Program {
                at: "guest::ring",
                why: format!("value {local} is not stage {at}'s"),
            })
    };
    let mut staged = Vec::new();
    for &global in roots {
        let origin = package
            .values
            .get(global as usize)
            .map(|value| value.source);
        if origin != Some(ValueOrigin::Output)
            && let Some(local) = super::run::local_of(package, at, global)?
        {
            staged.push(sized(local)?);
        }
    }
    let mut read = Vec::new();
    for &global in wanted {
        if let Some(local) = super::run::local_of(package, at, global)? {
            read.push(sized(local)?);
        }
    }
    Ok(PassLayout::plan(staged, status, read))
}

/// The most a pass of stage `at` of `package` takes of the ring: the
/// [`pass_layout`] over every root of the stage the host may stage (a
/// constant, a channel's take or read) and every value it puts, as a berth
/// of it is taken. Known at the bind, so a stage whose pass could outgrow
/// the ring binds no session; every fire's pass lays a part of it.
pub fn stage_need(
    package: &eta_compiler::codegen::launch::LaunchPackage,
    at: usize,
    status: u64,
    span: impl Fn(u32) -> Option<u64>,
) -> Result<u64> {
    use eta_compiler::codegen::launch::ValueOrigin;
    let ours = |global: u32| matches!(package.plan_local(global), Some((stage, _)) if stage == at);
    let roots: Vec<u32> = (0..package.values.len())
        .filter_map(|global| u32::try_from(global).ok())
        .filter(|&global| {
            ours(global)
                && matches!(
                    package.values[global as usize].source,
                    ValueOrigin::Const | ValueOrigin::ChannelTake | ValueOrigin::ChannelRead
                )
        })
        .collect();
    let puts: Vec<u32> = package
        .stages
        .get(at)
        .into_iter()
        .flat_map(|stage| &stage.puts)
        .map(|put| put.value)
        .filter(|&global| ours(global))
        .collect();
    let layout = pass_layout(package, at, &roots, &puts, status, span)?;
    Ok(berth_bytes(layout.bytes()))
}

#[cfg(test)]
mod tests {
    use super::Ring;

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
}
