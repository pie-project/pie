//! Shared-memory IPC fast path for `fire_batch`.
//!
//! Layout matches `pie_driver/shmem_ipc.py`. Both sides agree on:
//!
//! - 64-byte global header
//! - N slots, each `slot_stride` bytes
//! - Per-slot 64-byte header (req_seq, resp_seq, ids, lengths, timestamps)
//! - Then `req_buf` bytes for request payload, then `resp_buf` for response
//!
//! Sync is via the two atomics in the slot header. Rust bumps `req_seq`
//! after writing payload+lengths; Python bumps `resp_seq` after writing
//! the response. Both sides busy-spin (configurable via env vars).

use std::ffi::CString;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use anyhow::{anyhow, Context, Result};

pub const MAGIC: u32 = 0x50494533; // 'PIE3'
pub const SCHEMA_VERSION: u32 = 1;
pub const HEADER_SIZE: usize = 64;
pub const SLOT_HEADER_SIZE: usize = 64;

pub const METHOD_TAG_FIRE_BATCH: u32 = 0;
/// Routes `copy_d2d` (working-page fork copy) through the shmem fast
/// path so it shares a thread with `fire_batch` on the Python worker.
/// Without this, cold-path `copy_d2d` notify can be dispatched on a
/// different Python thread and reordered against fire_batch's
/// forward-pass kernels — see `project_pie_kv_bleed_d2d_fork_race.md`
/// and pie-project/pie#339.
pub const METHOD_TAG_COPY_D2D: u32 = 1;
/// Routes `copy_h2d` (suspend→restore working-page bring-back, plus
/// snapshot-restore committed-suffix) through the shmem fast path so
/// it shares a thread with `fire_batch`. Same race surface as
/// METHOD_TAG_COPY_D2D but on the eviction/restore code path —
/// surfaces under sustained pool overflow at axis-L L=6K c=16,
/// pin-on. See `project_pie_second_bleed_path_h2d_race.md`.
pub const METHOD_TAG_COPY_H2D: u32 = 2;
/// Routes `copy_d2h` (suspend's stash-to-CPU) through the shmem fast
/// path. Companion to METHOD_TAG_COPY_H2D — same single-threaded
/// dispatch ordering against `fire_batch`.
pub const METHOD_TAG_COPY_D2H: u32 = 3;
/// Notify Python that a `ctx.fork()` happened on a hybrid
/// Transformer+Mamba model: copy parent's mamba recurrent state into
/// the child's per-request state slot before any later `fire_batch`
/// touches the child. Same ordering guarantee as METHOD_TAG_COPY_D2D
/// (sibling shmem op on the same Python thread that runs fire_batch).
/// Payload is two little-endian u64 ContextIds. Engine ignores the op
/// when it has no mamba layers (pure-attention vllm, helloworld
/// tests, etc.) — see ticket pie-agents#108 phase 5b.
pub const METHOD_TAG_MAMBA_FORK: u32 = 4;

/// Slot header offsets relative to the start of the slot.
const OFF_REQ_SEQ: usize = 0;
const OFF_RESP_SEQ: usize = 8;
const OFF_REQ_ID: usize = 16;
const OFF_METHOD_TAG: usize = 20;
const OFF_REQ_LEN: usize = 24;
const OFF_RESP_LEN: usize = 28;
const OFF_SEND_WT: usize = 32;
const OFF_RESPOND_WT: usize = 40;

#[link(name = "rt")]
unsafe extern "C" {
    fn shm_open(name: *const libc::c_char, oflag: libc::c_int, mode: libc::mode_t) -> libc::c_int;
}

/// Rust-side client for the shmem fast path.
///
/// Owned by the Device actor. Each call grabs a free slot, writes the request,
/// busy-spins on the response, and releases the slot.
pub struct ShmemClient {
    /// Mapped region base (aliased — read/write by both processes).
    base: *mut u8,
    total_size: usize,
    num_slots: usize,
    slot_stride: usize,
    req_buf_size: usize,
    /// Round-robin slot allocator.
    next_slot: AtomicUsize,
    /// Per-slot busy lock so only one in-flight request per slot at a time.
    slot_locks: Vec<Mutex<()>>,
    /// Spin parameters.
    spin_us: u64,
}

unsafe impl Send for ShmemClient {}
unsafe impl Sync for ShmemClient {}

impl ShmemClient {
    /// Open an existing shmem region created by the Python server.
    pub fn open(
        name: &str,
        num_slots: usize,
        req_buf_size: usize,
        resp_buf_size: usize,
        spin_us: u64,
    ) -> Result<Self> {
        let slot_stride = SLOT_HEADER_SIZE + req_buf_size + resp_buf_size;
        let total_size = HEADER_SIZE + num_slots * slot_stride;

        let cname = CString::new(name)?;
        let fd = unsafe { shm_open(cname.as_ptr(), libc::O_RDWR, 0o600) };
        if fd < 0 {
            return Err(anyhow!(
                "shm_open({name}) failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                total_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            unsafe { libc::close(fd) };
            return Err(anyhow!("mmap failed"));
        }
        // We can close the fd now; the mapping persists.
        unsafe { libc::close(fd) };

        let base = base as *mut u8;

        // Validate header.
        let magic = unsafe { (base as *const u32).read_volatile() };
        let schema = unsafe { (base.add(4) as *const u32).read_volatile() };
        let n_slots = unsafe { (base.add(8) as *const u32).read_volatile() } as usize;
        let stride = unsafe { (base.add(12) as *const u32).read_volatile() } as usize;
        if magic != MAGIC {
            return Err(anyhow!("shmem magic mismatch: got 0x{:08x}, want 0x{:08x}", magic, MAGIC));
        }
        if schema != SCHEMA_VERSION {
            return Err(anyhow!("shmem schema version mismatch: got {}, want {}", schema, SCHEMA_VERSION));
        }
        if n_slots != num_slots || stride != slot_stride {
            return Err(anyhow!(
                "shmem geometry mismatch: server has slots={n_slots} stride={stride}, client wants slots={num_slots} stride={slot_stride}"
            ));
        }

        let slot_locks = (0..num_slots).map(|_| Mutex::new(())).collect();

        Ok(Self {
            base,
            total_size,
            num_slots,
            slot_stride,
            req_buf_size,
            next_slot: AtomicUsize::new(0),
            slot_locks,
            spin_us,
        })
    }

    fn slot_addr(&self, i: usize) -> *mut u8 {
        unsafe { self.base.add(HEADER_SIZE + i * self.slot_stride) }
    }
    fn req_payload_addr(&self, i: usize) -> *mut u8 {
        unsafe { self.slot_addr(i).add(SLOT_HEADER_SIZE) }
    }
    fn resp_payload_addr(&self, i: usize) -> *const u8 {
        unsafe { self.slot_addr(i).add(SLOT_HEADER_SIZE + self.req_buf_size) }
    }

    fn req_seq_atomic(&self, i: usize) -> &AtomicU64 {
        unsafe { &*(self.slot_addr(i).add(OFF_REQ_SEQ) as *const AtomicU64) }
    }
    fn resp_seq_atomic(&self, i: usize) -> &AtomicU64 {
        unsafe { &*(self.slot_addr(i).add(OFF_RESP_SEQ) as *const AtomicU64) }
    }

    /// Issue a request and return the response bytes.
    ///
    /// `writer` is called with the request payload buffer; it returns the
    /// number of bytes it wrote.
    pub fn call_with<F>(&self, request_id: u32, method_tag: u32, send_walltime_us: u64, writer: F) -> Result<(Vec<u8>, u64)>
    where
        F: FnOnce(&mut [u8]) -> Result<usize>,
    {
        // Pick a free slot via round-robin; serialize per-slot via Mutex.
        let n_attempts = self.num_slots.max(1);
        let start = self.next_slot.fetch_add(1, Ordering::Relaxed);
        let mut slot_idx = start % self.num_slots;
        let mut guard_opt = None;
        for k in 0..n_attempts {
            let candidate = (start + k) % self.num_slots;
            if let Ok(g) = self.slot_locks[candidate].try_lock() {
                slot_idx = candidate;
                guard_opt = Some(g);
                break;
            }
        }
        // If all slots busy, fall back to blocking on the round-robin
        // candidate. CRITICAL: lock the same slot we will use; previously
        // this returned a guard for a different slot than `slot_idx`,
        // allowing concurrent use of the same slot's seq counters.
        let _guard = match guard_opt {
            Some(g) => g,
            None => self.slot_locks[slot_idx]
                .lock()
                .map_err(|_| anyhow!("slot mutex poisoned"))?,
        };
        let i = slot_idx;

        // Write request payload.
        let req_buf = unsafe { std::slice::from_raw_parts_mut(self.req_payload_addr(i), self.req_buf_size) };
        let written = writer(req_buf).context("request writer failed")?;
        if written > self.req_buf_size {
            return Err(anyhow!("request payload {} exceeds buffer {}", written, self.req_buf_size));
        }

        // Write slot-header fields (non-atomic): req_id, method_tag, req_len, send_walltime.
        let slot_base = self.slot_addr(i);
        unsafe {
            (slot_base.add(OFF_REQ_ID) as *mut u32).write_volatile(request_id);
            (slot_base.add(OFF_METHOD_TAG) as *mut u32).write_volatile(method_tag);
            (slot_base.add(OFF_REQ_LEN) as *mut u32).write_volatile(written as u32);
            (slot_base.add(OFF_SEND_WT) as *mut u64).write_volatile(send_walltime_us);
        }

        // Bump req_seq with Release ordering (publishes payload + lengths).
        let new_seq = self.req_seq_atomic(i).load(Ordering::Relaxed) + 1;
        self.req_seq_atomic(i).store(new_seq, Ordering::Release);

        // Busy-spin on resp_seq with periodic yield, plus a hard deadline
        // so a stuck/dead worker doesn't hang the caller forever.
        let started = std::time::Instant::now();
        let yield_after = std::time::Duration::from_micros(self.spin_us);
        let hard_timeout = std::time::Duration::from_secs(60);
        loop {
            if self.resp_seq_atomic(i).load(Ordering::Acquire) >= new_seq {
                break;
            }
            let elapsed = started.elapsed();
            if elapsed >= hard_timeout {
                return Err(anyhow!(
                    "shmem call timed out after {:?} (slot {}, request_id {})",
                    hard_timeout, i, request_id
                ));
            }
            if elapsed >= yield_after {
                std::thread::yield_now();
            }
            std::hint::spin_loop();
        }

        // Read response.
        let resp_len = unsafe { (slot_base.add(OFF_RESP_LEN) as *const u32).read_volatile() } as usize;
        let respond_walltime_us =
            unsafe { (slot_base.add(OFF_RESPOND_WT) as *const u64).read_volatile() };
        let resp_bytes = unsafe {
            std::slice::from_raw_parts(self.resp_payload_addr(i), resp_len)
        };
        let owned = resp_bytes.to_vec();
        Ok((owned, respond_walltime_us))
    }
}

impl Drop for ShmemClient {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.base as *mut _, self.total_size) };
    }
}
