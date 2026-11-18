//! The shared pool's host tier (CONTRACT-BC 22): pageable host memory, and
//! a page-locked staging ring every move to or from the device goes
//! through.
//!
//! The tier is one anonymous mapping of the budget's bytes, committed by the
//! kernel page by page as moves first write it: the process holds what the
//! tier holds, never the budget. A move to the host copies the device's
//! slots into the ring on the swap stream and returns; a thread of the
//! tier's own waits for each copy to land and moves its bytes from the ring
//! into their slots, so the copy costs the engine thread what enqueueing it
//! does. A move to the device first waits for any landing into its slots,
//! copies them into the ring, and copies the ring to the device on the swap
//! stream. Nothing is page-locked on a move's path or after ready: the
//! load pins the ring before it measures the device ([`Ring::pin`]), so the
//! memory the ring takes is gone from what the device is sized by, and a
//! ring that cannot be pinned refuses the load by name.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};

use crate::device::Pinned;
use crate::device::alloc;
use crate::device::graph::Event;
use crate::error::{Fault, Result};

/// The page-locked staging ring's bytes for a tier of at most `budget`
/// bytes whose largest move is `sequence` bytes and whose largest slot is
/// `slot` bytes: two sequences, so a move enqueues while the one before it
/// lands. The planner moves at most one sequence's state in a step (a
/// process parked or brought back, a run promoted), and the engine thread
/// waits only where a move finds the ring full. Within the tier's budget,
/// but never under two slots: a slot is copied through the ring whole
/// ([`HostTier::take`]), and two let one land while the next is copied.
#[must_use]
pub fn staging_ring(sequence: u64, slot: u64, budget: u64) -> u64 {
    sequence
        .saturating_mul(2)
        .min(budget)
        .max(slot.saturating_mul(2))
}

/// The staging ring, page-locked on the load's path ([`Ring::pin`]): the
/// tier copies every move through it.
#[derive(Debug)]
pub struct Ring(Pinned);

impl Ring {
    /// Pins `bytes` of staging ring, or refuses the load naming them: a
    /// ring pinned later would take page-locked memory after ready, under
    /// whatever the device was sized to leave.
    pub fn pin(bytes: u64) -> Result<Ring> {
        let len = usize::try_from(bytes).unwrap_or(usize::MAX);
        pin_ring(len).map(Ring).map_err(|why| {
            Fault::Residency(format!(
                "the host tier's staging ring of {} MiB could not be pinned: {why}; state a \
                 smaller `[engine.memory] budget_host` or a shorter `[engine] max_context`",
                bytes >> 20
            ))
        })
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.0.bytes() as u64
    }
}

/// An event a landing thread waits on: the runtime's events are safe to
/// wait on from any thread, and only the landing thread touches this one.
struct Landing(Event);

// SAFETY: a CUDA event may be waited on and destroyed from any thread of the
// process; the job that carries it is the only holder once it is sent.
unsafe impl Send for Landing {}

/// One batch of moves: the copies `landed` follows on the swap stream, and
/// each `(ring offset, tier offset, bytes)` to move from the ring into the
/// tier once it has (none for a move to the device, whose ring space comes
/// back then).
struct Job {
    id: u64,
    landed: Landing,
    moves: Vec<(usize, usize, usize)>,
    /// The ring's allocation counter past this batch: its space comes back.
    end: u64,
    /// The ring's host address.
    ring: usize,
    /// The runtime's copy this batch ends, resolved once its bytes are in
    /// their slots (CONTRACT-BC 23); a dropped job fails it.
    copy: Option<engine::completion::CopyLanding>,
}

/// What the landing thread reports back.
#[derive(Default)]
struct Landed {
    /// Ring bytes free up to (the allocation counter's value).
    freed: u64,
    /// The last batch landed.
    done: u64,
    fault: Option<String>,
}

struct Shared {
    landed: Mutex<Landed>,
    moved: Condvar,
}

/// The host tier: pageable chunks, the staging ring, and the landing
/// thread.
pub struct HostTier {
    chunk_bytes: u64,
    chunks: u32,
    base: *mut u8,
    len: usize,
    /// The staging ring, pinned at the load. Dropped after the landing
    /// thread is joined (fields drop after [`Drop::drop`]).
    ring: Ring,
    /// Ring bytes handed out (a counter; the offset is it modulo the ring).
    head: u64,
    /// The landing thread's `freed` as last read: it only grows.
    freed: u64,
    next: u64,
    /// The last batch writing into each chunk: one a move reads waits for.
    writing: Vec<u64>,
    /// Whether each chunk's pages are committed ([`alloc::commit_anon`]).
    committed: Vec<AtomicBool>,
    shared: Arc<Shared>,
    send: Option<Sender<Job>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl std::fmt::Debug for HostTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostTier")
            .field("chunk_bytes", &self.chunk_bytes)
            .field("chunks", &self.chunks)
            .field("ring", &self.ring.bytes())
            .finish_non_exhaustive()
    }
}

// SAFETY: `base` is this tier's own mapping, written by the engine thread
// and by the landing thread only into slots no unlanded batch is writing
// (`writing` orders a read after the batch that writes it).
unsafe impl Send for HostTier {}

impl HostTier {
    /// A tier of `chunks` chunks of `chunk_bytes`, reserved, each committed
    /// when a span first names it, behind the pinned staging `ring`
    /// ([`staging_ring`]).
    pub fn new(chunk_bytes: u64, chunks: u32, ring: Ring) -> Result<HostTier> {
        let len =
            usize::try_from(chunk_bytes.saturating_mul(u64::from(chunks))).unwrap_or(usize::MAX);
        let Some(base) = alloc::reserve_anon(len) else {
            return Err(Fault::Ceiling {
                what: "bytes of address space for the host tier",
                need: len as u64,
                have: 0,
            });
        };
        // Committed a huge page at a time where the kernel has them: a move
        // landing in fresh memory faults a page in per 2 MiB, not per 4 KiB.
        // SAFETY: advice on this tier's own mapping; a kernel without
        // transparent huge pages ignores it.
        #[cfg(target_os = "linux")]
        unsafe {
            libc::madvise(base.cast(), len.max(1), libc::MADV_HUGEPAGE);
        }
        let shared = Arc::new(Shared {
            landed: Mutex::new(Landed::default()),
            moved: Condvar::new(),
        });
        let (send, jobs) = std::sync::mpsc::channel::<Job>();
        let thread = {
            let shared = shared.clone();
            let ordinal = crate::device::ctx::current()?;
            let base_at = base as usize;
            std::thread::Builder::new()
                .name("pie-host-tier-land".to_string())
                .spawn(move || land(&jobs, &shared, ordinal, base_at))
                .map_err(|why| Fault::Unbound {
                    what: format!("the host tier's landing thread: {why}"),
                })?
        };
        Ok(HostTier {
            chunk_bytes,
            chunks,
            base: base.cast(),
            len,
            ring,
            head: 0,
            freed: 0,
            next: 0,
            writing: vec![0; chunks as usize],
            committed: (0..chunks).map(|_| AtomicBool::new(false)).collect(),
            shared,
            send: Some(send),
            thread: Some(thread),
        })
    }

    #[must_use]
    pub fn chunks(&self) -> u32 {
        self.chunks
    }

    #[must_use]
    pub fn chunk_bytes(&self) -> u64 {
        self.chunk_bytes
    }

    /// The ring's host and device addresses.
    fn ring(&self) -> (usize, u64) {
        (self.ring.0.host() as usize, self.ring.0.device())
    }

    /// Bytes of ring `len` bytes may take, waiting for earlier batches to
    /// land where it is full; `flush` sends what the caller has gathered
    /// first, since what it waits for may be that.
    fn take(
        &mut self,
        len: usize,
        flush: &mut impl FnMut(&mut Self) -> Result<()>,
    ) -> Result<usize> {
        let size = self.ring.bytes();
        let len = len as u64;
        if len > size {
            return Err(Fault::Ceiling {
                what: "bytes of the host tier's staging ring one slot takes",
                need: len,
                have: size,
            });
        }
        let mut at = self.head % size;
        let mut head = self.head;
        if at + len > size {
            head += size - at;
            at = 0;
        }
        if head + len - self.freed > size {
            self.freed = self.shared.landed.lock().expect("the landing state").freed;
        }
        if head + len - self.freed > size {
            flush(self)?;
            let mut landed = self.shared.landed.lock().expect("the landing state");
            while head + len - landed.freed > size {
                if let Some(fault) = landed.fault.take() {
                    return Err(Fault::Unbound { what: fault });
                }
                landed = self.shared.moved.wait(landed).expect("the landing state");
            }
            self.freed = landed.freed;
        }
        self.head = head + len;
        Ok(at as usize)
    }

    /// Waits until every batch writing into `chunks` has landed.
    fn wait_for(&self, chunks: impl IntoIterator<Item = u32>) -> Result<()> {
        let need = chunks
            .into_iter()
            .map(|chunk| self.writing.get(chunk as usize).copied().unwrap_or(0))
            .max()
            .unwrap_or(0);
        if need == 0 {
            return Ok(());
        }
        let mut landed = self.shared.landed.lock().expect("the landing state");
        while landed.done < need {
            if let Some(fault) = landed.fault.take() {
                return Err(Fault::Unbound { what: fault });
            }
            landed = self.shared.moved.wait(landed).expect("the landing state");
        }
        Ok(())
    }

    fn check(&self, offset: u64, len: usize) -> Result<usize> {
        let end = offset.saturating_add(len as u64);
        if end > self.len as u64 {
            return Err(Fault::Ceiling {
                what: "bytes of the host tier",
                need: end,
                have: self.len as u64,
            });
        }
        let chunk_bytes = self.chunk_bytes.max(1);
        for chunk in offset / chunk_bytes..end.div_ceil(chunk_bytes) {
            let committed = &self.committed[chunk as usize];
            if committed.load(Ordering::Relaxed) {
                continue;
            }
            let at = chunk * chunk_bytes;
            let bytes = chunk_bytes.min(self.len as u64 - at);
            // SAFETY: the chunk lies inside this tier's own reservation.
            if !alloc::commit_anon(unsafe { self.base.add(at as usize) }, bytes as usize) {
                return Err(Fault::Ceiling {
                    what: "bytes of host memory the OS would commit for a host-tier chunk",
                    need: bytes,
                    have: 0,
                });
            }
            committed.store(true, Ordering::Relaxed);
        }
        Ok(offset as usize)
    }

    /// Sends one batch: `spans` copied on `swap`, then `moves` from the ring
    /// into the tier once they land. Returns the event a fire waits on.
    fn send_batch(
        &mut self,
        swap: *mut core::ffi::c_void,
        to_host: bool,
        spans: &mut Vec<(u64, u64, usize)>,
        moves: &mut Vec<(usize, usize, usize)>,
        landed: &mut Option<Event>,
        copy: Option<engine::completion::CopyLanding>,
    ) -> Result<()> {
        if spans.is_empty() {
            if let Some(copy) = copy {
                copy.landed();
            }
            return Ok(());
        }
        // A move to the device reads its slots into the ring on the swap
        // stream's own host thread, ahead of the copy: not on this one.
        let (ring, _) = self.ring();
        if !to_host {
            let base = self.base as usize;
            let reads = moves.clone();
            crate::device::ctx::host_fn_on(
                swap,
                Box::new(move || {
                    for (at, offset, len) in reads {
                        // SAFETY: both spans lie inside their buffers
                        // (checked when the batch was made); the ring span
                        // is this batch's, and no batch writes the tier
                        // span before this one is done (`writing`).
                        unsafe {
                            core::ptr::copy_nonoverlapping(
                                (base + offset) as *const u8,
                                (ring + at) as *mut u8,
                                len,
                            );
                        }
                    }
                }),
            )?;
        }
        crate::device::copy_batch(swap, to_host, spans)?;
        let ours = Event::new()?;
        ours.record(swap)?;
        let fence = Event::new()?;
        fence.record(swap)?;
        self.next += 1;
        let id = self.next;
        for &(_, offset, _) in moves.iter() {
            let chunk = (offset as u64 / self.chunk_bytes.max(1)) as usize;
            if let Some(writing) = self.writing.get_mut(chunk) {
                *writing = id;
            }
        }
        // A move to the device leaves the landing thread nothing to copy:
        // its batch only hands the ring space back once it is done.
        let moves = std::mem::take(moves);
        let job = Job {
            id,
            landed: Landing(ours),
            moves: if to_host { moves } else { Vec::new() },
            end: self.head,
            ring,
            copy,
        };
        spans.clear();
        self.send
            .as_ref()
            .and_then(|send| send.send(job).ok())
            .ok_or_else(|| Fault::Unbound {
                what: "the host tier's landing thread is gone".to_string(),
            })?;
        *landed = Some(fence);
        Ok(())
    }

    /// Copies each `(device address, tier offset, bytes)` into the tier on
    /// `swap`, behind what it has enqueued. Returns the event the last copy
    /// lands at on the stream (a fire naming one of the device slots waits
    /// on it); the bytes reach the tier on the landing thread, which then
    /// resolves `copy`.
    pub fn to_host(
        &mut self,
        swap: *mut core::ffi::c_void,
        slots: &[(u64, u64, usize)],
        copy: engine::completion::CopyLanding,
    ) -> Result<Option<Event>> {
        let (_, ring) = self.ring();
        let (mut spans, mut moves, mut landed) = (Vec::new(), Vec::new(), None);
        for &(device, offset, len) in slots {
            let offset = self.check(offset, len)?;
            let at = {
                let (spans, moves, landed) = (&mut spans, &mut moves, &mut landed);
                self.take(len, &mut |tier: &mut Self| {
                    tier.send_batch(swap, true, spans, moves, landed, None)
                })?
            };
            spans.push((ring + at as u64, device, len));
            moves.push((at, offset, len));
        }
        self.send_batch(swap, true, &mut spans, &mut moves, &mut landed, Some(copy))?;
        Ok(landed)
    }

    /// Copies each `(tier offset, device address, bytes)` to the device on
    /// `swap`, once every batch writing those slots has landed. Returns the
    /// event the last copy lands at on the stream; the landing thread
    /// resolves `copy` once it has.
    pub fn to_device(
        &mut self,
        swap: *mut core::ffi::c_void,
        slots: &[(u64, u64, usize)],
        copy: engine::completion::CopyLanding,
    ) -> Result<Option<Event>> {
        let chunk_bytes = self.chunk_bytes.max(1);
        self.wait_for(
            slots
                .iter()
                .map(|&(offset, ..)| (offset / chunk_bytes) as u32),
        )?;
        let (_, ring) = self.ring();
        let (mut spans, mut moves, mut landed) = (Vec::new(), Vec::new(), None);
        for &(offset, device, len) in slots {
            let offset = self.check(offset, len)?;
            let at = {
                let (spans, moves, landed) = (&mut spans, &mut moves, &mut landed);
                self.take(len, &mut |tier: &mut Self| {
                    tier.send_batch(swap, false, spans, moves, landed, None)
                })?
            };
            spans.push((device, ring + at as u64, len));
            moves.push((at, offset, len));
        }
        self.send_batch(swap, false, &mut spans, &mut moves, &mut landed, Some(copy))?;
        Ok(landed)
    }

    /// The tier's bytes at `offset`, once every batch writing them landed.
    pub fn slice(&self, offset: u64, len: usize) -> Result<&mut [u8]> {
        let at = self.check(offset, len)?;
        self.wait_for([(offset / self.chunk_bytes.max(1)) as u32])?;
        // SAFETY: inside the mapping (checked), and no batch writes it now.
        Ok(unsafe { core::slice::from_raw_parts_mut(self.base.add(at), len) })
    }
}

impl Drop for HostTier {
    fn drop(&mut self) {
        drop(self.send.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        // This tier's own mapping, no longer read or written.
        alloc::unmap_anon(self.base, self.len);
    }
}

/// The landing thread: each batch, once its copies landed, moved from the
/// ring into the tier, and its ring space handed back.
fn land(jobs: &Receiver<Job>, shared: &Shared, ordinal: i32, base: usize) {
    if let Err(why) = crate::device::ctx::bind_thread(ordinal) {
        let what = format!("the host tier's landing thread bound no device: {why}");
        eprintln!("engine-cuda: {what}");
        let mut landed = shared.landed.lock().expect("the landing state");
        landed.fault = Some(what);
        shared.moved.notify_all();
        return;
    }
    while let Ok(mut job) = jobs.recv() {
        let settled = job.landed.0.settle();
        if let Err(why) = &settled {
            let what = format!("a host tier move did not land: {why}");
            if let Some(copy) = job.copy.take() {
                copy.failed(&what);
            }
            let mut landed = shared.landed.lock().expect("the landing state");
            landed.fault = Some(what);
        } else {
            land_moves(&job.moves, job.ring, base);
            if let Some(copy) = job.copy.take() {
                copy.landed();
            }
        }
        drop(job.landed);
        let mut landed = shared.landed.lock().expect("the landing state");
        landed.freed = job.end;
        landed.done = job.id;
        shared.moved.notify_all();
    }
}

/// The threads that fault the ring's pages in before it is locked.
#[cfg(target_os = "linux")]
const RING_FAULTERS: usize = 8;

/// The staging ring, page-locked: faulted in first, in huge pages where the
/// kernel has them, so the driver only pins what is there (a fraction of
/// `cudaHostAlloc`'s time under its lock, and of its unpinning at exit).
fn pin_ring(bytes: usize) -> Result<Pinned> {
    #[cfg(target_os = "linux")]
    if let Some(faulted) = crate::device::Pinning::faulted(bytes, RING_FAULTERS)
        && let Ok(ring) = faulted.lock()
    {
        return Ok(ring);
    }
    Pinned::mapped_uninit(bytes)
}

/// The threads one batch's moves are split over: copying into fresh pages
/// is bound by the kernel committing them, about 5 GB/s a thread.
const LANDERS: usize = 4;

/// Copies a landed batch's moves from the ring into the tier, split by
/// bytes over [`LANDERS`] threads where it is large enough to pay for them.
fn land_moves(moves: &[(usize, usize, usize)], ring: usize, base: usize) {
    let copy = |moves: &[(usize, usize, usize)]| {
        for &(at, offset, len) in moves {
            // SAFETY: the ring span is this batch's, landed; the tier span is
            // inside the mapping (checked when the batch was made), and no
            // other batch writes it before this one lands; the spans of one
            // batch do not overlap.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    (ring + at) as *const u8,
                    (base + offset) as *mut u8,
                    len,
                );
            }
        }
    };
    let total: usize = moves.iter().map(|m| m.2).sum();
    if total < 8 << 20 || moves.len() < 2 {
        copy(moves);
        return;
    }
    let share = total.div_ceil(LANDERS);
    let mut parts: Vec<&[(usize, usize, usize)]> = Vec::with_capacity(LANDERS);
    let (mut from, mut held) = (0, 0);
    for (i, m) in moves.iter().enumerate() {
        held += m.2;
        if held >= share {
            parts.push(&moves[from..=i]);
            (from, held) = (i + 1, 0);
        }
    }
    if from < moves.len() {
        parts.push(&moves[from..]);
    }
    std::thread::scope(|scope| {
        for part in &parts[1..] {
            scope.spawn(move || copy(part));
        }
        copy(parts[0]);
    });
}

#[cfg(test)]
mod tests {
    use super::staging_ring;

    const GIB: u64 = 1 << 30;

    #[test]
    fn the_ring_holds_two_sequences_within_the_tier() {
        const MIB: u64 = 1 << 20;
        // A 5 GiB sequence over a 28 GiB tier: two of it.
        assert_eq!(staging_ring(5 * GIB, 18 * MIB, 28 * GIB), 10 * GIB);
        // No more than the tier holds.
        assert_eq!(staging_ring(5 * GIB, 18 * MIB, 8 * GIB), 8 * GIB);
        // Never under two slots, whatever the tier's budget: a slot is
        // copied through the ring whole.
        assert_eq!(staging_ring(5 * GIB, 18 * MIB, 20 * MIB), 36 * MIB);
    }
}
