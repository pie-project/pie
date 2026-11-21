//! The shared pool's host tier (CONTRACT-BC 22): pageable host memory, and
//! a page-locked staging ring every move to or from the device goes
//! through, a piece at a time.
//!
//! The tier is one anonymous mapping of the budget's bytes, committed by the
//! kernel page by page as moves first write it: the process holds what the
//! tier holds, never the budget. A move of any size goes through the ring
//! in pieces of whole slots: the device copies one piece on the swap stream
//! while a thread of the tier's own (the landing thread) moves the one
//! before it between the ring and the tier, and a piece is reused once both
//! are done with it. The ring holds a piece for each of those [`STAGES`]
//! for each frame the engine keeps in flight, since moves ride between
//! frames, one frame's ahead of the next ([`staging_ring`]): it is sized by
//! the copies in flight, not by what one move carries.
//!
//! A move to the host copies a piece of device slots into the ring on the
//! swap stream; the landing thread waits for it to land and moves it into
//! the tier. A move to the device has the landing thread move a piece from
//! the tier into the ring, and the swap stream copies it to the device once
//! that is done. Every wait for a piece sits on the swap stream, never on
//! the engine thread: the stream waits on the device for a word past the
//! ring's pieces, the last job the landing thread finished, which that
//! thread writes into the mapped ring ([`HostTier::gate`]); no host function
//! blocks and no driver thread waits. A move costs the engine thread what
//! enqueueing it does, as it did when the ring held whole moves. Each
//! landing-thread job waits only for copies enqueued on the swap stream
//! before every stream wait for it, so the two never wait on each other.
//!
//! Nothing is page-locked on a move's path or after ready: the load pins the
//! ring before it measures the device ([`Ring::pin`]), so the memory the
//! ring takes is gone from what the device is sized by, and a ring that
//! cannot be pinned refuses the load by name.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};

use crate::device::Pinned;
use crate::device::alloc;
use crate::device::graph::Event;
use crate::error::{Fault, Result};

/// The stages a piece passes through, each holding one piece while the
/// other works on the next: the device's copy (on the swap stream) and the
/// host's move between the ring and the tier (on the landing thread).
pub const STAGES: u64 = 2;

/// The page a piece is a whole number of.
const PIECE_GRAIN: u64 = 4096;

/// The staging ring's shape: `pieces` pieces of `piece` bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingSize {
    pub piece: u64,
    pub pieces: u64,
}

impl RingSize {
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.piece.saturating_mul(self.pieces)
    }
}

/// The page-locked staging ring for a tier of at most `budget` bytes whose
/// moves go through it in pieces of at most `piece` bytes, behind an
/// engine that keeps `frames` frames in flight: a piece for each of the
/// [`STAGES`] for each frame, since the runtime hands the engine moves
/// between frames (its pre-launch copies), so the moves handed before each
/// frame still in flight may be on their way at once. Within the tier's
/// budget (a tier never moves more than it holds), a piece never under a
/// page.
#[must_use]
pub fn staging_ring(piece: u64, frames: u8, budget: u64) -> RingSize {
    let pieces = STAGES * u64::from(frames.max(1));
    let piece = piece.min(budget / pieces) / PIECE_GRAIN * PIECE_GRAIN;
    RingSize {
        piece: piece.max(PIECE_GRAIN),
        pieces,
    }
}

/// How a move's slots, `(device address, tier offset, bytes)` each, go
/// through pieces of `piece` bytes: each piece its `(piece offset, device
/// address, tier offset, bytes)` spans, whole slots back to back, a slot
/// larger than a piece split across pieces of its own.
fn pieces(slots: &[(u64, usize, usize)], piece: usize) -> Vec<Vec<(usize, u64, usize, usize)>> {
    let piece = piece.max(1);
    let mut out: Vec<Vec<(usize, u64, usize, usize)>> = Vec::new();
    let mut held = piece;
    for &(device, tier, len) in slots {
        if len == 0 {
            continue;
        }
        if held + len > piece {
            held = 0;
            out.push(Vec::new());
        }
        let mut done = 0;
        while done < len {
            if held == piece {
                held = 0;
                out.push(Vec::new());
            }
            let take = (len - done).min(piece - held);
            out.last_mut().expect("a piece was opened").push((
                held,
                device + done as u64,
                tier + done,
                take,
            ));
            held += take;
            done += take;
        }
    }
    out
}

/// The staging ring, page-locked on the load's path ([`Ring::pin`]): the
/// tier copies every move through it.
#[derive(Debug)]
pub struct Ring(Pinned, RingSize);

impl Ring {
    /// Pins a staging ring of `size`, or refuses the load naming it: a
    /// ring pinned later would take page-locked memory after ready, under
    /// whatever the device was sized to leave.
    pub fn pin(size: RingSize) -> Result<Ring> {
        // The pieces, and a page past them for the landing thread's word.
        let bytes = size.bytes() + PIECE_GRAIN;
        let len = usize::try_from(bytes).unwrap_or(usize::MAX);
        pin_ring(len)
            .map(|pinned| Ring(pinned, size))
            .map_err(|why| {
                Fault::Residency(format!(
                    "the host tier's staging ring of {} MiB could not be pinned: {why}; state a \
                 smaller `[engine.memory] budget_host`",
                    bytes >> 20
                ))
            })
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.0.bytes() as u64
    }

    #[must_use]
    pub fn size(&self) -> RingSize {
        self.1
    }
}

/// An event a landing thread waits on: the runtime's events are safe to
/// wait on from any thread, and only the landing thread touches this one.
struct Landing(Event);

// SAFETY: a CUDA event may be waited on and destroyed from any thread of the
// process; the job that carries it is the only holder once it is sent.
unsafe impl Send for Landing {}

/// One piece for the landing thread: once the copy `after` follows has
/// landed (if any), each `(ring address, tier address, bytes)` moved into
/// the tier (`to_tier`) or out of it into the ring.
struct Job {
    id: u64,
    after: Option<Landing>,
    to_tier: bool,
    moves: Vec<(usize, usize, usize)>,
}

/// What the landing thread reports back.
#[derive(Default)]
struct Landed {
    /// The last job done: every job up to it is (jobs run in order).
    done: u64,
    fault: Option<String>,
}

struct Shared {
    landed: Mutex<Landed>,
    moved: Condvar,
    /// The host address of the ring's word the swap stream waits on: the
    /// low 32 bits of the last job done.
    word: usize,
}

impl Shared {
    /// Waits until job `id` is done, or the landing thread is gone.
    fn wait(&self, id: u64) -> std::sync::MutexGuard<'_, Landed> {
        let mut landed = self.landed.lock().expect("the landing state");
        while landed.done < id {
            landed = self.moved.wait(landed).expect("the landing state");
        }
        landed
    }

    fn finish(&self, id: u64, fault: Option<String>) {
        let mut landed = self.landed.lock().expect("the landing state");
        landed.done = landed.done.max(id);
        if fault.is_some() && landed.fault.is_none() {
            landed.fault = fault;
        }
        // Released after the job's bytes: a stream past the word reads them.
        // SAFETY: the word is the ring's own, aligned, and lives as long as
        // the tier (the ring drops after the landing thread is joined).
        unsafe { &*(self.word as *const AtomicU32) }.store(landed.done as u32, Ordering::Release);
        self.moved.notify_all();
    }
}

/// Who holds a piece of the ring until it may be written again.
enum Piece {
    Free,
    /// A move to the host: the landing thread's job that moves it out.
    Landing(u64),
    /// A move to the device: the copy out of it on the swap stream.
    Read(Event),
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
    /// A piece's bytes, and who holds each of the ring's pieces.
    piece: usize,
    held: Vec<Piece>,
    /// The piece the next one taken is.
    turn: usize,
    /// The last job id handed out.
    next: u64,
    /// The last job writing into each chunk: one a read waits for.
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
            .field("piece", &self.piece)
            .finish_non_exhaustive()
    }
}

// SAFETY: `base` is this tier's own mapping, written by the engine thread
// and by the landing thread only into slots no unfinished job is writing
// (`writing` orders a read after the job that writes it).
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
        let RingSize { piece, pieces } = ring.size();
        let word = ring.0.host() as usize + (piece * pieces) as usize;
        // SAFETY: the word lies in the ring's page past its pieces, aligned.
        unsafe { &*(word as *const AtomicU32) }.store(0, Ordering::Release);
        let shared = Arc::new(Shared {
            landed: Mutex::new(Landed::default()),
            moved: Condvar::new(),
            word,
        });
        let (send, jobs) = std::sync::mpsc::channel::<Job>();
        let thread = {
            let shared = shared.clone();
            let ordinal = crate::device::ctx::current()?;
            std::thread::Builder::new()
                .name("pie-host-tier-land".to_string())
                .spawn(move || land(&jobs, &shared, ordinal))
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
            piece: usize::try_from(piece).unwrap_or(usize::MAX),
            held: (0..pieces).map(|_| Piece::Free).collect(),
            turn: 0,
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

    /// A fault the landing thread met, once met.
    fn fault(&self) -> Result<()> {
        match &self.shared.landed.lock().expect("the landing state").fault {
            Some(fault) => Err(Fault::Unbound {
                what: fault.clone(),
            }),
            None => Ok(()),
        }
    }

    /// Holds `swap` until landing-thread job `id` is done: the stream waits
    /// on the device for the landing thread's word to reach it (compared
    /// wrapping, so the word's 32 bits never run out), enqueued only where
    /// the job is not done already.
    fn gate(&self, swap: *mut core::ffi::c_void, id: u64) -> Result<()> {
        if self.shared.landed.lock().expect("the landing state").done >= id {
            return Ok(());
        }
        let RingSize { piece, pieces } = self.ring.size();
        let word = self.ring.0.device() + piece * pieces;
        crate::device::ctx::wait_value(swap, word, id as u32)
    }

    fn job(
        &mut self,
        after: Option<Event>,
        to_tier: bool,
        moves: Vec<(usize, usize, usize)>,
    ) -> Result<u64> {
        self.next += 1;
        let id = self.next;
        let job = Job {
            id,
            after: after.map(Landing),
            to_tier,
            moves,
        };
        self.send
            .as_ref()
            .and_then(|send| send.send(job).ok())
            .ok_or_else(|| Fault::Unbound {
                what: "the host tier's landing thread is gone".to_string(),
            })?;
        Ok(id)
    }

    /// Waits until every job writing into `chunks` is done.
    fn wait_for(&self, chunks: impl IntoIterator<Item = u32>) -> Result<()> {
        let need = chunks
            .into_iter()
            .map(|chunk| self.writing.get(chunk as usize).copied().unwrap_or(0))
            .max()
            .unwrap_or(0);
        if need > 0 {
            drop(self.shared.wait(need));
        }
        self.fault()
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

    /// Moves `slots`, `(device address, tier offset, bytes)` each, through
    /// the ring a piece at a time, on `swap` behind what it has enqueued.
    /// Returns the event the last device copy is done at on the stream.
    fn through(
        &mut self,
        swap: *mut core::ffi::c_void,
        to_host: bool,
        slots: Vec<(u64, usize, usize)>,
    ) -> Result<Option<Event>> {
        self.fault()?;
        if slots.iter().all(|&(.., len)| len == 0) {
            return Ok(None);
        }
        let (ring, ring_device) = self.ring();
        let base = self.base as usize;
        for spans in pieces(&slots, self.piece) {
            let at = self.turn * self.piece;
            let held = std::mem::replace(&mut self.held[self.turn], Piece::Free);
            let moves: Vec<(usize, usize, usize)> = spans
                .iter()
                .map(|&(offset, _, tier, len)| (ring + at + offset, base + tier, len))
                .collect();
            let copies: Vec<(u64, u64, usize)> = spans
                .iter()
                .map(|&(offset, device, _, len)| {
                    let staged = ring_device + (at + offset) as u64;
                    if to_host {
                        (staged, device, len)
                    } else {
                        (device, staged, len)
                    }
                })
                .collect();
            self.held[self.turn] = if to_host {
                // The piece is written once the landing thread moved what
                // it held out (a copy out of it to the device came before
                // on this stream).
                if let Piece::Landing(id) = held {
                    self.gate(swap, id)?;
                }
                crate::device::copy_batch(swap, true, &copies)?;
                let landed = Event::new()?;
                landed.record(swap)?;
                let id = self.job(Some(landed), true, moves)?;
                for &(_, _, tier, len) in &spans {
                    let chunk_bytes = self.chunk_bytes.max(1);
                    for chunk in tier as u64 / chunk_bytes..(tier + len) as u64 / chunk_bytes + 1 {
                        if let Some(writing) = self.writing.get_mut(chunk as usize) {
                            *writing = id;
                        }
                    }
                }
                Piece::Landing(id)
            } else {
                // The landing thread fills the piece once the device read
                // what it held (a move out of it to the tier is a job
                // before this one), and the stream copies it on from there.
                let after = match held {
                    Piece::Read(read) => Some(read),
                    _ => None,
                };
                let id = self.job(after, false, moves)?;
                self.gate(swap, id)?;
                crate::device::copy_batch(swap, false, &copies)?;
                let read = Event::new()?;
                read.record(swap)?;
                Piece::Read(read)
            };
            self.turn = (self.turn + 1) % self.held.len();
        }
        let fence = Event::new()?;
        fence.record(swap)?;
        Ok(Some(fence))
    }

    /// Copies each `(device address, tier offset, bytes)` into the tier on
    /// `swap`, behind what it has enqueued. Returns the event the last copy
    /// lands at on the stream (a fire naming one of the device slots waits
    /// on it); the bytes reach the tier on the landing thread.
    pub fn to_host(
        &mut self,
        swap: *mut core::ffi::c_void,
        slots: &[(u64, u64, usize)],
    ) -> Result<Option<Event>> {
        let mut spans = Vec::with_capacity(slots.len());
        for &(device, offset, len) in slots {
            spans.push((device, self.check(offset, len)?, len));
        }
        self.through(swap, true, spans)
    }

    /// Copies each `(tier offset, device address, bytes)` to the device on
    /// `swap`, behind every move into those slots (the landing thread's
    /// order). Returns the event the last copy lands at on the stream.
    pub fn to_device(
        &mut self,
        swap: *mut core::ffi::c_void,
        slots: &[(u64, u64, usize)],
    ) -> Result<Option<Event>> {
        let mut spans = Vec::with_capacity(slots.len());
        for &(offset, device, len) in slots {
            spans.push((device, self.check(offset, len)?, len));
        }
        self.through(swap, false, spans)
    }

    /// The tier's bytes at `offset`, once every job writing them is done.
    pub fn slice(&self, offset: u64, len: usize) -> Result<&mut [u8]> {
        let at = self.check(offset, len)?;
        let chunk_bytes = self.chunk_bytes.max(1);
        self.wait_for(
            (offset / chunk_bytes) as u32..=((offset + len as u64) / chunk_bytes) as u32,
        )?;
        // SAFETY: inside the mapping (checked), and no job writes it now.
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

/// The landing thread: each job, once the copy it follows landed, moved
/// between the ring and the tier, in order.
fn land(jobs: &Receiver<Job>, shared: &Shared, ordinal: i32) {
    if let Err(why) = crate::device::ctx::bind_thread(ordinal) {
        let what = format!("the host tier's landing thread bound no device: {why}");
        eprintln!("engine-cuda: {what}");
        // Every job finishes, moving nothing, so no stream waits forever.
        while let Ok(job) = jobs.recv() {
            shared.finish(job.id, Some(what.clone()));
        }
        return;
    }
    while let Ok(job) = jobs.recv() {
        let settled = job.after.as_ref().map_or(Ok(()), |after| after.0.settle());
        let fault = match settled {
            Err(why) => Some(format!("a host tier move did not land: {why}")),
            Ok(()) => {
                land_moves(&job.moves, job.to_tier);
                None
            }
        };
        drop(job.after);
        shared.finish(job.id, fault);
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

/// The threads one piece's moves are split over: copying into fresh pages
/// is bound by the kernel committing them, about 5 GB/s a thread.
const LANDERS: usize = 4;

/// Moves a piece's `(ring address, tier address, bytes)` spans into the
/// tier (`to_tier`) or out of it, split by bytes over [`LANDERS`] threads
/// where it is large enough to pay for them.
fn land_moves(moves: &[(usize, usize, usize)], to_tier: bool) {
    let copy = |moves: &[(usize, usize, usize)]| {
        for &(ring, tier, len) in moves {
            let (src, dst) = if to_tier { (ring, tier) } else { (tier, ring) };
            // SAFETY: the ring span is this job's piece, its copy landed or
            // its last read done; the tier span is inside the mapping
            // (checked when the job was made), and no other job touches it
            // before this one is done; the spans of one job do not overlap.
            unsafe {
                core::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len);
            }
        }
    };
    let total: usize = moves.iter().map(|m| m.2).sum();
    if total < 8 << 20 {
        copy(moves);
        return;
    }
    let parts = split(moves, total.div_ceil(LANDERS));
    std::thread::scope(|scope| {
        for part in &parts[1..] {
            scope.spawn(move || copy(part));
        }
        copy(&parts[0]);
    });
}

/// `moves` cut into parts of about `share` bytes each, a span cut where a
/// part ends.
fn split(moves: &[(usize, usize, usize)], share: usize) -> Vec<Vec<(usize, usize, usize)>> {
    let share = share.max(1);
    let mut parts = vec![Vec::new()];
    let mut held = 0;
    for &(ring, tier, len) in moves {
        let mut done = 0;
        while done < len {
            if held == share {
                parts.push(Vec::new());
                held = 0;
            }
            let take = (len - done).min(share - held);
            parts
                .last_mut()
                .expect("a part is open")
                .push((ring + done, tier + done, take));
            held += take;
            done += take;
        }
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::{RingSize, pieces, split, staging_ring};

    const MIB: u64 = 1 << 20;

    #[test]
    fn the_ring_holds_a_piece_a_stage_a_frame_within_the_tier() {
        let ring = |piece, pieces| RingSize { piece, pieces };
        // 32 MiB pieces over a 28 GiB tier, two frames in flight: a piece
        // for each of the two stages of each frame, whatever the context.
        assert_eq!(staging_ring(32 * MIB, 2, 28 << 30), ring(32 * MIB, 4));
        assert_eq!(staging_ring(32 * MIB, 1, 28 << 30), ring(32 * MIB, 2));
        // No more than the tier holds, never under a page a piece.
        assert_eq!(staging_ring(32 * MIB, 2, 20 * MIB), ring(5 * MIB, 4));
        assert_eq!(staging_ring(32 * MIB, 2, 1), ring(4096, 4));
    }

    #[test]
    fn a_move_goes_through_pieces_of_whole_slots() {
        // Three 3-byte slots through 8-byte pieces: two whole a piece.
        let got = pieces(&[(100, 0, 3), (200, 10, 3), (300, 20, 3)], 8);
        assert_eq!(
            got,
            vec![vec![(0, 100, 0, 3), (3, 200, 10, 3)], vec![(0, 300, 20, 3)],]
        );
        // A slot past a piece is split over pieces of its own, and the
        // next slot starts a fresh one.
        let got = pieces(&[(100, 0, 2), (200, 10, 19), (300, 40, 1)], 8);
        assert_eq!(
            got,
            vec![
                vec![(0, 100, 0, 2)],
                vec![(0, 200, 10, 8)],
                vec![(0, 208, 18, 8)],
                vec![(0, 216, 26, 3), (3, 300, 40, 1)],
            ]
        );
        // Every byte once.
        let total: usize = got.iter().flatten().map(|s| s.3).sum();
        assert_eq!(total, 22);
    }

    #[test]
    fn a_piece_splits_by_bytes_over_its_landers() {
        let got = split(&[(0, 1000, 10), (50, 2000, 2)], 4);
        assert_eq!(
            got,
            vec![
                vec![(0, 1000, 4)],
                vec![(4, 1004, 4)],
                vec![(8, 1008, 2), (50, 2000, 2)],
            ]
        );
    }
}
