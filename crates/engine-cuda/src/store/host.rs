//! The shared pool's host tier: pageable host memory, and
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

use engine::Landing;

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
struct Waiting(Event);

// SAFETY: a CUDA event may be waited on and destroyed from any thread of the
// process; the job that carries it is the only holder once it is sent.
unsafe impl Send for Waiting {}

/// One piece for the landing thread: once the copy `after` follows has
/// landed (if any), each `(ring address, tier address, bytes)` moved into
/// the tier (`to_tier`) or out of it into the ring.
struct Job {
    id: u64,
    after: Option<Waiting>,
    to_tier: bool,
    moves: Vec<(usize, usize, usize)>,
    /// A move between the tier and the slot file, done on the landing
    /// thread in the jobs' order, in place of `moves`.
    disk: Option<(usize, DiskMove)>,
}

/// A move between the host tier and the disk tier's slot file: each
/// `(tier offset, slot file byte, bytes)` read from the file into the tier,
/// or written from the tier into the file (`to_disk`), on the tier's landing
/// thread; told how it ended where `told` is given.
pub struct DiskMove {
    pub to_disk: bool,
    pub file: Arc<engine::disk::SlotFile>,
    /// Held across a slot's cover's read-modify-write: a write keeps the
    /// bytes of the slots beside it, which another thread may be writing.
    pub io: Arc<Mutex<()>>,
    pub slots: Vec<(u64, u64, usize)>,
    pub told: Option<crate::serve::landing::Told>,
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
    /// The last host<->disk job on each aligned block of the slot file: a
    /// move between the device and the file waits for it.
    disk_writing: std::collections::HashMap<u64, u64>,
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
            disk_writing: std::collections::HashMap::new(),
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
            after: after.map(Waiting),
            disk: None,
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

    /// Hands a move between the tier and the slot file to the landing
    /// thread, behind every move it holds: the engine thread only enqueues
    /// it. Returns its job id ([`HostTier::wait_job`]).
    pub fn disk(&mut self, mv: DiskMove) -> Result<u64> {
        self.fault()?;
        for &(offset, _, len) in &mv.slots {
            self.check(offset, len)?;
        }
        self.next += 1;
        let id = self.next;
        let align = engine::disk::ALIGN as u64;
        let chunk_bytes = self.chunk_bytes.max(1);
        for &(offset, place, len) in &mv.slots {
            let mut block = place / align * align;
            while block < place + len as u64 {
                self.disk_writing.insert(block, id);
                block += align;
            }
            if !mv.to_disk {
                for chunk in offset / chunk_bytes..(offset + len as u64).div_ceil(chunk_bytes) {
                    if let Some(writing) = self.writing.get_mut(chunk as usize) {
                        *writing = id;
                    }
                }
            }
        }
        let job = Job {
            id,
            after: None,
            to_tier: !mv.to_disk,
            moves: Vec::new(),
            disk: Some((self.base as usize, mv)),
        };
        self.send
            .as_ref()
            .and_then(|send| send.send(job).ok())
            .ok_or_else(|| Fault::Unbound {
                what: "the host tier's landing thread is gone".to_string(),
            })?;
        Ok(id)
    }

    /// Waits, on the calling thread, until landing-thread job `id` is done.
    pub fn wait_job(&self, id: u64) -> Result<()> {
        drop(self.shared.wait(id));
        self.fault()
    }

    /// Waits until every host<->disk job on the slot file's bytes `places`
    /// (`(byte, bytes)` each) is done: a move between the device and the
    /// file reads or writes them after.
    pub fn wait_disk(&self, places: impl IntoIterator<Item = (u64, u64)>) -> Result<()> {
        let align = engine::disk::ALIGN as u64;
        let mut need = 0;
        for (place, len) in places {
            let mut block = place / align * align;
            while block < place + len {
                need = need.max(self.disk_writing.get(&block).copied().unwrap_or(0));
                block += align;
            }
        }
        if need > 0 {
            drop(self.shared.wait(need));
        }
        self.fault()
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
    #[allow(clippy::wrong_self_convention)]
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
        tracing::warn!("{what}");
        // Every job finishes, moving nothing, so no stream waits forever.
        while let Ok(job) = jobs.recv() {
            shared.finish(job.id, Some(what.clone()));
        }
        return;
    }
    while let Ok(job) = jobs.recv() {
        if let Some((base, mv)) = job.disk {
            // A slot file that faults fails this move alone, told by name:
            // the tier and the other moves go on.
            let moved = disk_io(base, &mv);
            if let Some(told) = mv.told {
                told(match moved {
                    Ok(()) => Landing::Landed,
                    Err(why) => Landing::Failed(format!(
                        "a move between the host tier and the slot file failed: {why}"
                    )),
                });
            } else if let Err(why) = moved {
                shared.finish(
                    job.id,
                    Some(format!("a move between the host tier and the slot file failed: {why}")),
                );
                continue;
            }
            shared.finish(job.id, None);
            continue;
        }
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

/// Debug and tests: a delay before each host<->disk move's I/O, in
/// milliseconds (a slow disk).
pub static DISK_DELAY_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `mv`'s I/O, on the landing thread: each slot's cover of aligned blocks
/// read whole, and written keeping its neighbours' bytes.
fn disk_io(base: usize, mv: &DiskMove) -> std::io::Result<()> {
    let delay = DISK_DELAY_MS.load(Ordering::Relaxed);
    if delay > 0 {
        std::thread::sleep(std::time::Duration::from_millis(delay));
    }
    let align = engine::disk::ALIGN;
    let mut edge = engine::disk::Aligned::zeroed(align);
    for &(offset, place, len) in &mv.slots {
        let start = place / align as u64 * align as u64;
        let end = (place + len as u64).next_multiple_of(align as u64);
        let lead = (place - start) as usize;
        let mut staged = engine::disk::Aligned::zeroed((end - start) as usize);
        let span = staged.as_mut_slice();
        // SAFETY: the span lies in the tier's mapping (checked and committed
        // when the move was enqueued), and only this thread moves its bytes
        // now: every job before it touching them is done, none after runs.
        let tier = unsafe { std::slice::from_raw_parts_mut((base + offset as usize) as *mut u8, len) };
        let _io = mv.io.lock().expect("the slot file's I/O");
        if mv.to_disk {
            span[lead..lead + len].copy_from_slice(tier);
            let block = edge.as_mut_slice();
            if lead > 0 {
                mv.file.read_span(start, block)?;
                span[..lead].copy_from_slice(&block[..lead]);
            }
            if lead + len < span.len() {
                let last = span.len() - align;
                mv.file.read_span(start + last as u64, block)?;
                span[lead + len..].copy_from_slice(&block[lead + len - last..]);
            }
            mv.file.write_span(start, span)?;
        } else {
            mv.file.read_span(start, span)?;
            tier.copy_from_slice(&span[lead..lead + len]);
        }
    }
    Ok(())
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

    /// A tier of 4 chunks of 64 KiB behind a small ring, and a slot file of
    /// 16 slots of 64 KiB, on the device this thread binds.
    #[cfg(feature = "device")]
    fn tier_and_file(name: &str) -> (crate::device::Context, super::HostTier, std::sync::Arc<engine::disk::SlotFile>, std::path::PathBuf) {
        let device = crate::device::Context::bind(0, None).expect("a device");
        let ring = super::Ring::pin(staging_ring(64 << 10, 1, 256 << 10)).expect("a ring");
        let tier = super::HostTier::new(64 << 10, 4, ring).expect("a tier");
        let dir = std::env::temp_dir().join(format!("pie-host-disk-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = engine::disk::SlotFile::open(&dir, "t.state", 64 << 10, 16 * (64 << 10))
            .expect("the slot file opens")
            .expect("it holds slots");
        (device, tier, std::sync::Arc::new(file), dir)
    }

    /// A slow disk write holds no engine-thread call: enqueueing the move
    /// and the next one return at once, and the move is told landed after
    /// the disk took its time; its bytes come back whole.
    #[cfg(feature = "device")]
    #[test]
    fn a_slow_disk_move_never_holds_the_engine_thread() {
        use std::sync::atomic::Ordering;
        use std::time::{Duration, Instant};
        let (_device, mut tier, file, dir) = tier_and_file("slow");
        let io = std::sync::Arc::default();
        tier.slice(0, 4096).unwrap().fill(0x5a);
        super::DISK_DELAY_MS.store(1500, Ordering::Relaxed);
        let (tx, rx) = std::sync::mpsc::channel();
        let began = Instant::now();
        let told = tx.clone();
        tier.disk(super::DiskMove {
            to_disk: true,
            file: std::sync::Arc::clone(&file),
            io: std::sync::Arc::clone(&io),
            slots: vec![(0, 3 * (64 << 10), 4096)],
            told: Some(Box::new(move |landing| told.send(landing).unwrap())),
        })
        .unwrap();
        let enqueued = began.elapsed();
        let next = Instant::now();
        tier.disk(super::DiskMove {
            to_disk: true,
            file: std::sync::Arc::clone(&file),
            io: std::sync::Arc::clone(&io),
            slots: vec![(64 << 10, 5 * (64 << 10), 4096)],
            told: Some(Box::new(move |landing| tx.send(landing).unwrap())),
        })
        .unwrap();
        assert!(enqueued < Duration::from_millis(200), "enqueueing took {enqueued:?}");
        assert!(next.elapsed() < Duration::from_millis(200), "the next call waited on the disk");
        assert_eq!(rx.recv_timeout(Duration::from_secs(10)).unwrap(), engine::Landing::Landed);
        assert!(began.elapsed() >= Duration::from_millis(1400), "told before the disk was done");
        assert_eq!(rx.recv_timeout(Duration::from_secs(10)).unwrap(), engine::Landing::Landed);
        super::DISK_DELAY_MS.store(0, Ordering::Relaxed);
        // Back from the file into another chunk, waited out by the caller.
        let id = tier
            .disk(super::DiskMove {
                to_disk: false,
                file,
                io,
                slots: vec![(2 * (64 << 10), 3 * (64 << 10), 4096)],
                told: None,
            })
            .unwrap();
        tier.wait_job(id).unwrap();
        assert!(tier.slice(2 * (64 << 10), 4096).unwrap().iter().all(|&b| b == 0x5a));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A move the slot file refuses is told failed with the fault, never
    /// landed, and the tier goes on.
    #[cfg(feature = "device")]
    #[test]
    fn a_disk_move_that_faults_is_told_failed() {
        let (_device, mut tier, file, dir) = tier_and_file("fault");
        let (tx, rx) = std::sync::mpsc::channel();
        tier.disk(super::DiskMove {
            to_disk: true,
            file,
            io: std::sync::Arc::default(),
            // Past the file's 16 slots.
            slots: vec![(0, 40 * (64 << 10), 4096)],
            told: Some(Box::new(move |landing| tx.send(landing).unwrap())),
        })
        .unwrap();
        match rx.recv_timeout(std::time::Duration::from_secs(10)).unwrap() {
            engine::Landing::Failed(why) => assert!(why.contains("slot file"), "{why}"),
            engine::Landing::Landed => panic!("a refused move was told landed"),
        }
        assert!(tier.slice(0, 4096).is_ok(), "the tier goes on");
        let _ = std::fs::remove_dir_all(dir);
    }
}
