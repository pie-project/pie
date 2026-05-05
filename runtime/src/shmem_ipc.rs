//! Shared-memory IPC fast path for `fire_batch`.
//!
//! Layout matches `pie_driver_dev/shmem_ipc.py` and the C++ servers under
//! `driver/{cuda,portable}/src/shmem_ipc.{cpp,hpp}`. Both sides agree on:
//!
//! - 64-byte global header (magic, schema, num_slots, slot_stride,
//!   req_buf_size, resp_buf_size — server is the source of truth)
//! - N slots, each `slot_stride` bytes
//! - Per-slot 64-byte header (req_seq, resp_seq, ids, lengths, timestamps)
//! - Then `req_buf` bytes for request payload, then `resp_buf` for response
//!
//! Sync is via the two atomics in the slot header. Rust bumps `req_seq`
//! after writing payload+lengths; the server bumps `resp_seq` after writing
//! the response. Both sides busy-spin (configurable via env vars).
//!
//! Geometry (`num_slots`, `req_buf_size`, `resp_buf_size`) is owned by
//! whichever process creates the region (the driver) — only it knows how
//! big its responses can get (e.g. full-vocab distribution probes). The
//! Rust client reads those values out of the header at attach time rather
//! than asserting them against compile-time constants, so the driver can
//! resize without touching the runtime.

use std::ffi::CString;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};

use anyhow::{Context, Result, anyhow};
#[cfg(windows)]
use windows_sys::Win32::Foundation::CloseHandle;
#[cfg(windows)]
use windows_sys::Win32::System::Memory::{
    FILE_MAP_ALL_ACCESS, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, OpenFileMappingA,
    UnmapViewOfFile,
};

pub const MAGIC: u32 = 0x50494533; // 'PIE3'
/// Bump in lockstep with `pie_driver_dev/shmem_ipc.py::SCHEMA_VERSION` and the
/// `SCHEMA_VERSION` constant in `driver/{cuda,portable}/src/shmem_ipc.hpp`.
/// v2 added `req_buf_size` / `resp_buf_size` to the global header so the
/// client no longer hardcodes geometry.
pub const SCHEMA_VERSION: u32 = 2;
pub const HEADER_SIZE: usize = 64;
pub const SLOT_HEADER_SIZE: usize = 64;

pub const METHOD_TAG_FIRE_BATCH: u32 = 0;

/// Global-header field offsets (relative to base).
const HDR_OFF_MAGIC: usize = 0;
const HDR_OFF_SCHEMA: usize = 4;
const HDR_OFF_NUM_SLOTS: usize = 8;
const HDR_OFF_SLOT_STRIDE: usize = 12;
const HDR_OFF_REQ_BUF: usize = 16;
const HDR_OFF_RESP_BUF: usize = 20;

/// Slot header offsets relative to the start of the slot.
const OFF_REQ_SEQ: usize = 0;
const OFF_RESP_SEQ: usize = 8;
const OFF_REQ_ID: usize = 16;
const OFF_METHOD_TAG: usize = 20;
const OFF_REQ_LEN: usize = 24;
const OFF_RESP_LEN: usize = 28;
const OFF_SEND_WT: usize = 32;
const OFF_RESPOND_WT: usize = 40;

#[cfg(unix)]
#[cfg_attr(target_os = "linux", link(name = "rt"))]
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
    /// Set by the supervisor when the backing driver process has exited
    /// (or is otherwise known dead). Checked inside `call_with`'s
    /// busy-spin so an in-flight request bails immediately instead of
    /// waiting `hard_timeout` for a response that's never coming. The
    /// watchdog flips this from `serve::lifecycle` via
    /// `pie::device::abort_all_shmem_clients`.
    aborted: AtomicBool,
}

unsafe impl Send for ShmemClient {}
unsafe impl Sync for ShmemClient {}

impl ShmemClient {
    /// Attach to an existing shmem region created by the driver. Geometry
    /// (`num_slots`, `slot_stride`, `req_buf_size`, `resp_buf_size`) is read
    /// out of the global header — the driver is the source of truth.
    pub fn open(name: &str, spin_us: u64) -> Result<Self> {
        let mapping = map_shmem(name)?;
        let base = mapping.base;
        let total_size = mapping.total_size;
        let read_u32 =
            |off: usize| -> u32 { unsafe { (base.add(off) as *const u32).read_volatile() } };

        let magic = read_u32(HDR_OFF_MAGIC);
        if magic != MAGIC {
            unsafe { unmap_shmem(base, total_size) };
            return Err(anyhow!(
                "shmem magic mismatch: got 0x{:08x}, want 0x{:08x}",
                magic,
                MAGIC
            ));
        }
        let schema = read_u32(HDR_OFF_SCHEMA);
        if schema != SCHEMA_VERSION {
            unsafe { unmap_shmem(base, total_size) };
            return Err(anyhow!(
                "shmem schema version mismatch: got {}, want {}. \
                 Rebuild the driver and runtime against the same tree.",
                schema,
                SCHEMA_VERSION
            ));
        }
        let num_slots = read_u32(HDR_OFF_NUM_SLOTS) as usize;
        let slot_stride = read_u32(HDR_OFF_SLOT_STRIDE) as usize;
        let req_buf_size = read_u32(HDR_OFF_REQ_BUF) as usize;
        let resp_buf_size = read_u32(HDR_OFF_RESP_BUF) as usize;

        // Cross-check: stride must be slot header + req + resp, and the
        // file must be exactly header + N·stride. Anything else means the
        // driver wrote an inconsistent header — bail loudly rather than
        // silently overrunning a buffer.
        let expected_stride = SLOT_HEADER_SIZE + req_buf_size + resp_buf_size;
        if slot_stride != expected_stride {
            unsafe { unmap_shmem(base, total_size) };
            return Err(anyhow!(
                "shmem header inconsistent: slot_stride={slot_stride} \
                 != SLOT_HEADER_SIZE({SLOT_HEADER_SIZE}) + req_buf({req_buf_size}) + resp_buf({resp_buf_size}) = {expected_stride}"
            ));
        }
        // The header fields are the canonical bound; macOS rounds the shm
        // region up to a page (16 KiB on Apple Silicon), so tolerate a
        // larger file but reject one that's actually short.
        let expected_total = HEADER_SIZE + num_slots * slot_stride;
        if total_size < expected_total {
            unsafe { unmap_shmem(base, total_size) };
            return Err(anyhow!(
                "shmem size too small: file is {total_size} bytes, header implies {expected_total} (slots={num_slots} stride={slot_stride})"
            ));
        }
        if num_slots == 0 {
            unsafe { unmap_shmem(base, total_size) };
            return Err(anyhow!("shmem header reports num_slots=0"));
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
            aborted: AtomicBool::new(false),
        })
    }

    /// Mark the client as aborted so any in-flight `call_with` returns
    /// promptly with a clear error. Idempotent. Called by the supervisor
    /// when it observes the driver subprocess has exited.
    pub fn abort(&self) {
        self.aborted.store(true, Ordering::Release);
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
    pub fn call_with<F>(
        &self,
        request_id: u32,
        method_tag: u32,
        send_walltime_us: u64,
        writer: F,
    ) -> Result<(Vec<u8>, u64)>
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
        let req_buf =
            unsafe { std::slice::from_raw_parts_mut(self.req_payload_addr(i), self.req_buf_size) };
        let written = writer(req_buf).context("request writer failed")?;
        if written > self.req_buf_size {
            return Err(anyhow!(
                "request payload {} exceeds buffer {}",
                written,
                self.req_buf_size
            ));
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

        // Busy-spin on resp_seq with periodic yield, an abort check
        // (driver-death signal from the watchdog) and a hard deadline as
        // a backstop. The supervisor's watchdog normally flips
        // `self.aborted` within ~1s of the driver dying, so the hard
        // timeout exists only for the no-supervisor or wedged-supervisor
        // case — kept at 60s by default because a healthy driver's first
        // fire_batch can spend tens of seconds JIT-compiling triton /
        // flashinfer kernels on a cold cache. `PIE_SHMEM_TIMEOUT_S` lets
        // CI runs override (lower for fast failure, higher if a benchmark
        // legitimately blocks longer than 60s).
        let started = std::time::Instant::now();
        let yield_after = std::time::Duration::from_micros(self.spin_us);
        let hard_timeout = *HARD_TIMEOUT;
        loop {
            if self.resp_seq_atomic(i).load(Ordering::Acquire) >= new_seq {
                break;
            }
            if self.aborted.load(Ordering::Acquire) {
                return Err(anyhow!(
                    "shmem call aborted: driver exited (slot {}, request_id {})",
                    i,
                    request_id
                ));
            }
            let elapsed = started.elapsed();
            if elapsed >= hard_timeout {
                return Err(anyhow!(
                    "shmem call timed out after {:?} (slot {}, request_id {})",
                    hard_timeout,
                    i,
                    request_id
                ));
            }
            if elapsed >= yield_after {
                std::thread::yield_now();
            }
            std::hint::spin_loop();
        }

        // Read response.
        let resp_len =
            unsafe { (slot_base.add(OFF_RESP_LEN) as *const u32).read_volatile() } as usize;
        let respond_walltime_us =
            unsafe { (slot_base.add(OFF_RESPOND_WT) as *const u64).read_volatile() };
        let resp_bytes = unsafe { std::slice::from_raw_parts(self.resp_payload_addr(i), resp_len) };
        let owned = resp_bytes.to_vec();
        Ok((owned, respond_walltime_us))
    }
}

impl Drop for ShmemClient {
    fn drop(&mut self) {
        unsafe { unmap_shmem(self.base, self.total_size) };
    }
}

struct Mapping {
    base: *mut u8,
    total_size: usize,
}

#[cfg(unix)]
fn map_shmem(name: &str) -> Result<Mapping> {
    let cname = CString::new(name)?;
    let fd = unsafe { shm_open(cname.as_ptr(), libc::O_RDWR, 0o600) };
    if fd < 0 {
        return Err(anyhow!(
            "shm_open({name}) failed: {}",
            std::io::Error::last_os_error()
        ));
    }

    // The driver `ftruncate`d the region to its full size before
    // publishing it, so `fstat` is enough to learn how much to mmap.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut st as *mut _) } != 0 {
        let err = std::io::Error::last_os_error();
        unsafe { libc::close(fd) };
        return Err(anyhow!("fstat({name}) failed: {err}"));
    }
    let total_size = st.st_size as usize;
    if total_size < HEADER_SIZE {
        unsafe { libc::close(fd) };
        return Err(anyhow!(
            "shmem region {name} too small: {total_size} < {HEADER_SIZE}"
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
    unsafe { libc::close(fd) };

    Ok(Mapping {
        base: base as *mut u8,
        total_size,
    })
}

#[cfg(unix)]
unsafe fn unmap_shmem(base: *mut u8, total_size: usize) {
    unsafe {
        libc::munmap(base as *mut _, total_size);
    }
}

#[cfg(windows)]
fn map_shmem(name: &str) -> Result<Mapping> {
    let cname = windows_mapping_name(name)?;
    let handle = unsafe { OpenFileMappingA(FILE_MAP_ALL_ACCESS, 0, cname.as_ptr().cast()) };
    if handle.is_null() {
        return Err(anyhow!(
            "OpenFileMappingA({name}) failed: {}",
            std::io::Error::last_os_error()
        ));
    }

    let header_view = unsafe { MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, HEADER_SIZE) };
    if header_view.Value.is_null() {
        let err = std::io::Error::last_os_error();
        unsafe { CloseHandle(handle) };
        return Err(anyhow!("MapViewOfFile({name}, header) failed: {err}"));
    }

    let header = header_view.Value as *mut u8;
    let read_u32 =
        |off: usize| -> u32 { unsafe { (header.add(off) as *const u32).read_volatile() } };
    let num_slots = read_u32(HDR_OFF_NUM_SLOTS) as usize;
    let slot_stride = read_u32(HDR_OFF_SLOT_STRIDE) as usize;
    let total_size = HEADER_SIZE + num_slots * slot_stride;
    unsafe { UnmapViewOfFile(header_view) };

    if total_size < HEADER_SIZE {
        unsafe { CloseHandle(handle) };
        return Err(anyhow!(
            "shmem region {name} too small: {total_size} < {HEADER_SIZE}"
        ));
    }

    let base = unsafe { MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, total_size) };
    let map_err = std::io::Error::last_os_error();
    unsafe { CloseHandle(handle) };
    if base.Value.is_null() {
        return Err(anyhow!("MapViewOfFile({name}) failed: {map_err}"));
    }

    Ok(Mapping {
        base: base.Value as *mut u8,
        total_size,
    })
}

#[cfg(windows)]
unsafe fn unmap_shmem(base: *mut u8, _total_size: usize) {
    unsafe {
        UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: base.cast() });
    }
}

#[cfg(windows)]
fn windows_mapping_name(name: &str) -> Result<CString> {
    let trimmed = name.trim_start_matches(['/', '\\']);
    if trimmed.is_empty() {
        return Err(anyhow!("shmem name {name:?} is empty after normalization"));
    }
    let normalized = trimmed.replace(['/', '\\'], "_");
    CString::new(format!("Local\\{normalized}")).context("Windows shmem name contains NUL")
}

/// Backstop timeout for `call_with`'s busy-spin. The watchdog `aborted`
/// flag is the primary fast-detection path; this fires only when the
/// supervisor itself is wedged (or absent — direct callers of
/// `runtime::device::fire_batch` outside `pie-server`). Resolved once at
/// startup from `PIE_SHMEM_TIMEOUT_S` (float seconds), defaulting to
/// 60s so a healthy driver's first fire_batch — which can spend tens of
/// seconds JIT-compiling triton / flashinfer kernels — doesn't trip it.
static HARD_TIMEOUT: LazyLock<std::time::Duration> = LazyLock::new(|| {
    const DEFAULT_SECS: f64 = 60.0;
    let secs = std::env::var("PIE_SHMEM_TIMEOUT_S")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(DEFAULT_SECS);
    std::time::Duration::from_secs_f64(secs)
});
