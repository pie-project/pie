//! The shared pool's host tier (CONTRACT-BC 22): pageable host memory, and
//! a small page-locked staging ring every move to or from the device goes
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
//! stream. Nothing is page-locked on a move's path: the ring was at seating.

use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};

use crate::device::Pinned;
use crate::device::graph::Event;
use crate::error::{Fault, Result};

/// The page-locked staging ring's bytes, at least two chunks: a burst of
/// moves larger than it waits for the earliest to land (one 512 MiB move
/// does not).
const RING: u64 = 512 << 20;

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
    ring: Pinned,
    /// Ring bytes handed out (a counter; the offset is it modulo the ring).
    head: u64,
    /// The landing thread's `freed` as last read: it only grows.
    freed: u64,
    next: u64,
    /// The last batch writing into each chunk: one a move reads waits for.
    writing: Vec<u64>,
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
    /// A tier of `chunks` chunks of `chunk_bytes`, mapped, not committed.
    pub fn new(chunk_bytes: u64, chunks: u32) -> Result<HostTier> {
        let len =
            usize::try_from(chunk_bytes.saturating_mul(u64::from(chunks))).unwrap_or(usize::MAX);
        // SAFETY: an anonymous private mapping no other code knows of; the
        // kernel commits its pages as they are first written.
        let base = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                len.max(1),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return Err(Fault::Ceiling {
                what: "bytes of address space for the host tier",
                need: len as u64,
                have: 0,
            });
        }
        // Committed a huge page at a time where the kernel has them: a move
        // landing in fresh memory faults a page in per 2 MiB, not per 4 KiB.
        // SAFETY: advice on this tier's own mapping; a kernel without
        // transparent huge pages ignores it.
        unsafe {
            libc::madvise(base, len.max(1), libc::MADV_HUGEPAGE);
        }
        let ring_bytes = RING.max(2 * chunk_bytes);
        let ring = Pinned::mapped_uninit(usize::try_from(ring_bytes).unwrap_or(usize::MAX))?;
        let shared = Arc::new(Shared {
            landed: Mutex::new(Landed::default()),
            moved: Condvar::new(),
        });
        let (send, jobs) = std::sync::mpsc::channel::<Job>();
        let thread = {
            let shared = shared.clone();
            let ordinal = crate::device::ctx::current()?;
            let ring_at = ring.host() as usize;
            let base_at = base as usize;
            std::thread::Builder::new()
                .name("pie-host-tier-land".to_string())
                .spawn(move || land(&jobs, &shared, ordinal, ring_at, base_at))
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
            shared,
            send: Some(send),
            thread: Some(thread),
        })
    }

    #[must_use]
    pub fn chunks(&self) -> u32 {
        self.chunks
    }

    /// Bytes of ring `len` bytes may take, waiting for earlier batches to
    /// land where it is full; `flush` sends what the caller has gathered
    /// first, since what it waits for may be that.
    fn take(
        &mut self,
        len: usize,
        flush: &mut impl FnMut(&mut Self) -> Result<()>,
    ) -> Result<usize> {
        let size = self.ring.bytes() as u64;
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
    ) -> Result<()> {
        if spans.is_empty() {
            return Ok(());
        }
        // A move to the device reads its slots into the ring on the swap
        // stream's own host thread, ahead of the copy: not on this one.
        if !to_host {
            let (ring, base) = (self.ring.host() as usize, self.base as usize);
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
    /// on it); the bytes reach the tier on the landing thread.
    pub fn to_host(
        &mut self,
        swap: *mut core::ffi::c_void,
        slots: &[(u64, u64, usize)],
    ) -> Result<Option<Event>> {
        let (mut spans, mut moves, mut landed) = (Vec::new(), Vec::new(), None);
        for &(device, offset, len) in slots {
            let offset = self.check(offset, len)?;
            let at = {
                let (spans, moves, landed) = (&mut spans, &mut moves, &mut landed);
                self.take(len, &mut |tier: &mut Self| {
                    tier.send_batch(swap, true, spans, moves, landed)
                })?
            };
            spans.push((self.ring.device() + at as u64, device, len));
            moves.push((at, offset, len));
        }
        self.send_batch(swap, true, &mut spans, &mut moves, &mut landed)?;
        Ok(landed)
    }

    /// Copies each `(tier offset, device address, bytes)` to the device on
    /// `swap`, once every batch writing those slots has landed. Returns the
    /// event the last copy lands at on the stream.
    pub fn to_device(
        &mut self,
        swap: *mut core::ffi::c_void,
        slots: &[(u64, u64, usize)],
    ) -> Result<Option<Event>> {
        let chunk_bytes = self.chunk_bytes.max(1);
        self.wait_for(
            slots
                .iter()
                .map(|&(offset, ..)| (offset / chunk_bytes) as u32),
        )?;
        let (mut spans, mut moves, mut landed) = (Vec::new(), Vec::new(), None);
        for &(offset, device, len) in slots {
            let offset = self.check(offset, len)?;
            let at = {
                let (spans, moves, landed) = (&mut spans, &mut moves, &mut landed);
                self.take(len, &mut |tier: &mut Self| {
                    tier.send_batch(swap, false, spans, moves, landed)
                })?
            };
            spans.push((device, self.ring.device() + at as u64, len));
            moves.push((at, offset, len));
        }
        self.send_batch(swap, false, &mut spans, &mut moves, &mut landed)?;
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
        // SAFETY: this tier's own mapping, no longer read or written.
        unsafe {
            libc::munmap(self.base.cast(), self.len.max(1));
        }
    }
}

/// The landing thread: each batch, once its copies landed, moved from the
/// ring into the tier, and its ring space handed back.
fn land(jobs: &Receiver<Job>, shared: &Shared, ordinal: i32, ring: usize, base: usize) {
    if let Err(why) = crate::device::ctx::bind_thread(ordinal) {
        let mut landed = shared.landed.lock().expect("the landing state");
        landed.fault = Some(format!(
            "the host tier's landing thread bound no device: {why}"
        ));
        shared.moved.notify_all();
        return;
    }
    while let Ok(job) = jobs.recv() {
        let settled = job.landed.0.settle();
        if let Err(why) = &settled {
            let mut landed = shared.landed.lock().expect("the landing state");
            landed.fault = Some(format!("a host tier move did not land: {why}"));
        } else {
            land_moves(&job.moves, ring, base);
        }
        drop(job.landed);
        let mut landed = shared.landed.lock().expect("the landing state");
        landed.freed = job.end;
        landed.done = job.id;
        shared.moved.notify_all();
    }
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
