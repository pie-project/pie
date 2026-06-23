//! Inter-process transport — POSIX shared-memory ring. Both server
//! and client live in this module. Gated behind the `ipc` feature.
//!
//! Each region carries a fixed-geometry handshake header plus N slots,
//! each holding a request payload buffer and a response payload buffer.
//! [`ShmemServer`] creates the region and acts as the responder;
//! [`ShmemClient`] attaches as the requester.
//!
//! ```text
//! [64-byte global header]
//!   0:  u32 magic = 0x50494534 ('PIE4')
//!   4:  u32 schema_version = 2
//!   8:  u32 num_slots
//!   12: u32 slot_stride (including tail padding)
//!   16: u32 req_buf_size
//!   20: u32 resp_buf_size
//!   24: u64 schema_hash (xxh3 of all schema types; see §5.4 handshake)
//!   32: u32 req_wake     (bumped by client on any send; server parks here)
//!   ...padding...
//!
//! [num_slots × slot_stride bytes]
//!   slot[i]:
//!     0:  u64 req_seq    (atomic; client bumps on send)
//!     8:  u64 resp_seq   (atomic; server bumps on respond)
//!     16: u32 req_id
//!     20: u32 _reserved  (was method_tag in old bridge)
//!     24: u32 req_payload_len
//!     28: u32 resp_payload_len
//!     32: u64 send_walltime_us
//!     40: u64 respond_walltime_us
//!     48: u32 resp_wake  (bumped by server on commit; client parks here)
//!     ...padding to 64...
//!     64: request payload (req_buf_size bytes) — opaque rkyv archive
//!     ...: response payload (resp_buf_size bytes) — opaque rkyv archive
//!     ...: padding to the next cache-line-aligned slot
//! ```
//!
//! Wait strategy. Both server and client are hybrid spin-then-park:
//!   1. Busy-spin (`std::hint::spin_loop`) for `spin_budget_us` —
//!      catches back-to-back fires with sub-µs wake.
//!   2. Park on the corresponding wake atomic via a cross-process
//!      kernel primitive (Linux `futex(2)`, Windows `WaitOnAddress`,
//!      macOS `__ulock_wait` with `UL_COMPARE_AND_WAIT_SHARED`). See
//!      [`platform::park`] / [`platform::wake_all`].
//!
//! Platform-specific code lives in `ipc/{linux,macos,windows,fallback}.rs`.
//! POSIX shmem region setup is shared by Linux + macOS via
//! `ipc/posix.rs`. Each platform module exports the same surface
//! (`map_shmem_*`, `unmap_shmem_*`, `ServerMapping`, `ClientMapping`,
//! `park`, `wake_all`), so this file has zero `#[cfg]` blocks past
//! the initial dispatch.

use std::ffi::CString;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};

// ============================================================================
// Platform dispatch — exactly one of these is compiled in.
// ============================================================================

#[cfg(unix)]
mod posix;

#[cfg(any(target_os = "linux", target_os = "android"))]
mod linux;
#[cfg(any(target_os = "linux", target_os = "android"))]
use linux as platform;

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
use macos as platform;

#[cfg(windows)]
mod windows;
#[cfg(windows)]
use windows as platform;

#[cfg(all(
    unix,
    not(any(target_os = "linux", target_os = "android", target_os = "macos"))
))]
mod fallback;
#[cfg(all(
    unix,
    not(any(target_os = "linux", target_os = "android", target_os = "macos"))
))]
use fallback as platform;

// ============================================================================
// Wire-format constants. Bumping any of these is a wire-incompatible
// change — clients are required to read `MAGIC` + `SCHEMA_VERSION` +
// `schema_hash` from the region header before attaching, and reject
// any mismatch.
// ============================================================================

pub const MAGIC: u32 = 0x50494534; // 'PIE4'

/// Bumped from 1 to 2 when the slot header grew a `resp_wake` u32
/// counter and the global header grew a `req_wake` u32 counter. Old
/// clients connecting to new servers (or vice versa) will be rejected
/// by the schema-version check in `ShmemClient::open`.
pub const SCHEMA_VERSION: u32 = 2;
pub const HEADER_SIZE: usize = 64;
pub const SLOT_HEADER_SIZE: usize = 64;
const SLOT_ALIGN: usize = 64;

const HDR_OFF_NUM_SLOTS: usize = 8;
const HDR_OFF_SLOT_STRIDE: usize = 12;
const HDR_OFF_REQ_BUF: usize = 16;
const HDR_OFF_RESP_BUF: usize = 20;
const HDR_OFF_SCHEMA_HASH: usize = 24;
/// Global wake counter — bumped by `ShmemClient` after writing to ANY
/// slot's request seq, then `platform::wake_all`'d. Server's
/// `poll_blocking` parks on this u32, then scans slots after wake.
const HDR_OFF_REQ_WAKE: usize = 32;

const OFF_REQ_SEQ: usize = 0;
const OFF_RESP_SEQ: usize = 8;
const OFF_REQ_ID: usize = 16;
// 20: reserved (was method_tag in the legacy bridge)
const OFF_REQ_LEN: usize = 24;
const OFF_RESP_LEN: usize = 28;
const OFF_SEND_WT: usize = 32;
const OFF_RESPOND_WT: usize = 40;
/// Per-slot response wake counter — bumped by the server when it
/// commits, then `platform::wake_all`'d. Client's `roundtrip` parks
/// on this slot's u32, then re-checks `resp_seq` after wake.
const OFF_RESP_WAKE: usize = 48;

#[inline]
fn checked_u32_field(name: &str, value: usize) -> Result<u32> {
    if value > u32::MAX as usize {
        Err(anyhow!("{name} {value} exceeds u32::MAX"))
    } else {
        Ok(value as u32)
    }
}

#[inline]
fn align_up(value: usize, align: usize) -> Result<usize> {
    debug_assert!(align.is_power_of_two());
    value
        .checked_add(align - 1)
        .map(|v| v & !(align - 1))
        .ok_or_else(|| anyhow!("slot stride overflow"))
}

#[inline]
fn checked_slot_stride(req_buf: usize, resp_buf: usize) -> Result<usize> {
    let raw = SLOT_HEADER_SIZE
        .checked_add(req_buf)
        .and_then(|v| v.checked_add(resp_buf))
        .ok_or_else(|| anyhow!("slot stride overflow"))?;
    align_up(raw, SLOT_ALIGN)
}

#[inline]
fn checked_region_size(num_slots: usize, slot_stride: usize) -> Result<usize> {
    if num_slots == 0 {
        return Err(anyhow!("num_slots must be greater than zero"));
    }
    let slots = num_slots
        .checked_mul(slot_stride)
        .ok_or_else(|| anyhow!("shmem size overflow"))?;
    HEADER_SIZE
        .checked_add(slots)
        .ok_or_else(|| anyhow!("shmem size overflow"))
}

/// Hard upper bound on a single `ShmemClient::roundtrip` call. The
/// server is expected to respond well within this; any longer and we
/// treat the call as failed (typically the driver crashed).
static HARD_TIMEOUT: LazyLock<Duration> = LazyLock::new(|| {
    std::env::var("PIE_SHMEM_HARD_TIMEOUT_S")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(Duration::from_secs_f64)
        .unwrap_or_else(|| Duration::from_secs(60))
});

// ============================================================================
// Small helpers for unaligned reads/writes against the mmap'd header.
// ============================================================================

fn write_u32(base: *mut u8, off: usize, v: u32) {
    unsafe { (base.add(off) as *mut u32).write_unaligned(v) };
}
fn read_u32(base: *const u8, off: usize) -> u32 {
    unsafe { (base.add(off) as *const u32).read_unaligned() }
}
fn write_bytes(base: *mut u8, off: usize, bytes: &[u8]) {
    unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), base.add(off), bytes.len()) };
}
fn atomic_load_u64(base: *const u8, off: usize) -> u64 {
    unsafe { (*(base.add(off) as *const AtomicU64)).load(Ordering::Acquire) }
}
fn atomic_store_u64(base: *mut u8, off: usize, v: u64) {
    unsafe { (*(base.add(off) as *const AtomicU64)).store(v, Ordering::Release) };
}
fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

// ============================================================================
// ShmemServerInner — the mmap'd state, shared via Arc with Lease.
// ============================================================================

struct ShmemServerInner {
    name: CString,
    num_slots: usize,
    req_buf_size: usize,
    resp_buf_size: usize,
    slot_stride: usize,
    /// Owned mapping (platform-specific bits — fd on POSIX, HANDLE on
    /// Windows). Released by [`Drop`] via `platform::unmap_shmem_server`.
    mapping: platform::ServerMapping,
    /// Busy-spin window in µs after a poll comes up empty; once it
    /// elapses, the server parks on the global `req_wake` atomic via
    /// `platform::park`. Set to `u64::MAX` for unbounded busy-spin
    /// (HFT-style, one core 100%); set to `0` to always park.
    spin_budget_us: u64,
    stop: AtomicBool,
    /// Last seen req_seq per slot — edge-triggered polling state.
    last_seen: Vec<AtomicU64>,
    /// Next slot to poll (round-robin starting hint).
    poll_cursor: AtomicUsize,
}

// SAFETY: All mutating access goes through mmap'd memory synchronized by
// per-slot atomics; the raw pointer is never freed while Arc references remain.
unsafe impl Send for ShmemServerInner {}
unsafe impl Sync for ShmemServerInner {}

impl Drop for ShmemServerInner {
    fn drop(&mut self) {
        // SAFETY: `mapping` came from `platform::map_shmem_server` and
        // hasn't been freed yet.
        unsafe { platform::unmap_shmem_server(&self.mapping, &self.name) };
    }
}

impl ShmemServerInner {
    fn base(&self) -> *mut u8 {
        self.mapping.base
    }

    fn slot_base(&self, slot: usize) -> *mut u8 {
        unsafe { self.base().add(HEADER_SIZE + slot * self.slot_stride) }
    }

    /// Global request-wake atomic in the region header. Bumped by
    /// `ShmemClient` after writing any slot; server parks on this.
    fn req_wake_atomic(&self) -> &AtomicU32 {
        unsafe { &*(self.base().add(HDR_OFF_REQ_WAKE) as *const AtomicU32) }
    }

    /// Per-slot response-wake atomic. Bumped by `commit`; client
    /// `roundtrip` parks on this for its slot.
    fn resp_wake_atomic(&self, slot: usize) -> &AtomicU32 {
        unsafe { &*(self.slot_base(slot).add(OFF_RESP_WAKE) as *const AtomicU32) }
    }

    fn req_payload_ptr(&self, slot: usize) -> *mut u8 {
        unsafe { self.slot_base(slot).add(SLOT_HEADER_SIZE) }
    }

    fn resp_payload_ptr(&self, slot: usize) -> *mut u8 {
        unsafe {
            self.slot_base(slot)
                .add(SLOT_HEADER_SIZE + self.req_buf_size)
        }
    }

    /// Poll one slot; return its `PolledSlot` if the request seq advanced.
    fn poll_one(&self, slot: usize) -> Option<PolledSlot> {
        let slot_ptr = self.slot_base(slot);
        let req_seq = atomic_load_u64(slot_ptr, OFF_REQ_SEQ);
        let prev = self.last_seen[slot].load(Ordering::Relaxed);
        if req_seq == prev {
            return None;
        }
        self.last_seen[slot].store(req_seq, Ordering::Relaxed);

        let req_id = read_u32(slot_ptr, OFF_REQ_ID);
        let payload_len = read_u32(slot_ptr, OFF_REQ_LEN) as usize;
        let send_walltime_us = atomic_load_u64(slot_ptr, OFF_SEND_WT);
        Some(PolledSlot {
            slot,
            req_seq,
            req_id,
            payload_len,
            send_walltime_us,
        })
    }

    /// Commit a response payload for `slot`. Caller has already filled the
    /// response buffer up to `resp_len` bytes.
    fn commit(&self, slot: usize, req_seq: u64, resp_len: usize) -> Result<()> {
        debug_assert!(resp_len <= self.resp_buf_size);
        let slot_ptr = self.slot_base(slot);
        let current_req_seq = atomic_load_u64(slot_ptr, OFF_REQ_SEQ);
        if current_req_seq != req_seq {
            return Err(anyhow!(
                "stale shmem lease for slot {slot}: request seq advanced from {req_seq} to {current_req_seq}"
            ));
        }
        write_u32(slot_ptr, OFF_RESP_LEN, resp_len as u32);
        atomic_store_u64(slot_ptr, OFF_RESPOND_WT, now_us());
        atomic_store_u64(slot_ptr, OFF_RESP_SEQ, req_seq);
        // Bump the per-slot response-wake counter, then wake any
        // client parked on it. `fetch_add` is Release so the seq
        // write above is visible before the futex wake delivers.
        let wake = self.resp_wake_atomic(slot);
        wake.fetch_add(1, Ordering::Release);
        // SAFETY: `wake` points into the shmem region this server owns.
        unsafe { platform::wake_all(wake as *const AtomicU32) };
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct PolledSlot {
    slot: usize,
    req_seq: u64,
    req_id: u32,
    payload_len: usize,
    #[allow(dead_code)]
    send_walltime_us: u64,
}

// ============================================================================
// ShmemServer — public facade.
// ============================================================================

pub struct ShmemServer {
    inner: Arc<ShmemServerInner>,
}

impl ShmemServer {
    /// Create a new shmem region.
    ///
    /// `spin_budget_us` controls how long `poll_blocking` busy-polls
    /// after a poll comes up empty before parking on the global
    /// `req_wake` atomic via the cross-process kernel primitive
    /// ([`platform::park`]). Set to `0` to always park (lowest CPU,
    /// ~3-5 µs wake on Linux); `u64::MAX` for unbounded busy-spin
    /// (one core 100%, ~ns wake).
    ///
    /// Replaces any stale region with the same name (best-effort
    /// `shm_unlink` first on POSIX). The `schema_hash` is written
    /// into the global header and compared on the client side.
    pub fn create(
        name: &str,
        num_slots: usize,
        req_buf: usize,
        resp_buf: usize,
        spin_budget_us: u64,
        schema_hash: [u8; 8],
    ) -> Result<Self> {
        let cname = CString::new(name).map_err(|e| anyhow!("shmem name {name:?}: {e}"))?;
        let num_slots_u32 = checked_u32_field("num_slots", num_slots)?;
        let req_buf_u32 = checked_u32_field("req_buf", req_buf)?;
        let resp_buf_u32 = checked_u32_field("resp_buf", resp_buf)?;
        let slot_stride = checked_slot_stride(req_buf, resp_buf)?;
        let slot_stride_u32 = checked_u32_field("slot_stride", slot_stride)?;
        let total_size = checked_region_size(num_slots, slot_stride)?;

        let mapping = platform::map_shmem_server(&cname, total_size)
            .map_err(|e| anyhow!("map_shmem_server({name}): {e}"))?;
        let base = mapping.base;

        // Initialize: zero + write header.
        unsafe { ptr::write_bytes(base, 0, total_size) };
        write_u32(base, 0, MAGIC);
        write_u32(base, 4, SCHEMA_VERSION);
        write_u32(base, HDR_OFF_NUM_SLOTS, num_slots_u32);
        write_u32(base, HDR_OFF_SLOT_STRIDE, slot_stride_u32);
        write_u32(base, HDR_OFF_REQ_BUF, req_buf_u32);
        write_u32(base, HDR_OFF_RESP_BUF, resp_buf_u32);
        write_bytes(base, HDR_OFF_SCHEMA_HASH, &schema_hash);

        let last_seen = (0..num_slots).map(|_| AtomicU64::new(0)).collect();

        Ok(Self {
            inner: Arc::new(ShmemServerInner {
                name: cname,
                num_slots,
                req_buf_size: req_buf,
                resp_buf_size: resp_buf,
                slot_stride,
                mapping,
                spin_budget_us,
                stop: AtomicBool::new(false),
                last_seen,
                poll_cursor: AtomicUsize::new(0),
            }),
        })
    }

    pub fn name(&self) -> &str {
        self.inner.name.to_str().unwrap_or("<invalid utf-8>")
    }
    pub fn num_slots(&self) -> usize {
        self.inner.num_slots
    }
    pub fn req_buf_size(&self) -> usize {
        self.inner.req_buf_size
    }
    pub fn resp_buf_size(&self) -> usize {
        self.inner.resp_buf_size
    }

    /// Signal the polling loop to exit. Idempotent. Wakes any thread
    /// currently parked in `poll_blocking` so it re-checks the flag.
    pub fn stop(&self) {
        self.inner.stop.store(true, Ordering::Release);
        // Bump the wake counter so parked waiters see a value change
        // and return immediately from `platform::park`.
        let wake = self.inner.req_wake_atomic();
        wake.fetch_add(1, Ordering::Release);
        unsafe { platform::wake_all(wake as *const AtomicU32) };
    }
    pub fn stopped(&self) -> bool {
        self.inner.stop.load(Ordering::Relaxed)
    }

    /// Non-blocking poll: returns the next pending request as a
    /// [`Lease`], or `None` if no slot has a new request right now.
    pub fn poll(&self) -> Option<Lease> {
        let n = self.inner.num_slots;
        let start = self.inner.poll_cursor.fetch_add(1, Ordering::Relaxed);
        for k in 0..n {
            let slot = (start + k) % n;
            if let Some(p) = self.inner.poll_one(slot) {
                return Some(Lease::new(self.inner.clone(), p));
            }
        }
        None
    }

    /// Block until a request lands or `timeout` elapses. Respects the
    /// `stop` flag (returns `None` immediately if stopped).
    ///
    /// Wait strategy is hybrid spin-then-park (see module docs):
    ///   * **Phase 1 (spin):** busy-poll with `spin_loop()` for up to
    ///     `spin_budget_us`. Catches back-to-back fires with ~ns wake.
    ///   * **Phase 2 (park):** park on the global `req_wake` atomic
    ///     via the cross-process kernel primitive — zero CPU during
    ///     idle, ~3-5 µs wake on Linux.
    pub fn poll_blocking(&self, timeout: Duration) -> Option<Lease> {
        let started = Instant::now();
        let deadline = started + timeout;
        let spin_deadline = (self.inner.spin_budget_us != u64::MAX)
            .then(|| started + Duration::from_micros(self.inner.spin_budget_us));

        // Phase 1: busy-spin. `Instant::now()` is a vDSO call (~10 ns)
        // but still worth amortizing — sample the deadline every 256
        // iters so the spin loop body stays tight.
        if self.inner.spin_budget_us > 0 {
            let mut iters: u32 = 0;
            loop {
                if self.inner.stop.load(Ordering::Relaxed) {
                    return None;
                }
                if let Some(l) = self.poll() {
                    return Some(l);
                }
                iters = iters.wrapping_add(1);
                if iters & 0xFF == 0 {
                    let now = Instant::now();
                    if now >= deadline {
                        return None;
                    }
                    if spin_deadline.is_some_and(|deadline| now >= deadline) {
                        break;
                    }
                }
                std::hint::spin_loop();
            }
        }

        // Phase 2: park on the global req_wake atomic.
        let wake = self.inner.req_wake_atomic();
        loop {
            if self.inner.stop.load(Ordering::Relaxed) {
                return None;
            }
            // Snapshot the wake counter *before* polling so any
            // concurrent producer's bump-after-our-snapshot returns
            // from `park` immediately via value mismatch.
            let snapshot = wake.load(Ordering::Acquire);
            if let Some(l) = self.poll() {
                return Some(l);
            }
            let now = Instant::now();
            if now >= deadline {
                return None;
            }
            let remaining = deadline.saturating_duration_since(now);
            // SAFETY: `wake` lives in the shmem region we own; the
            // pointer is valid for the duration of the syscall.
            unsafe {
                platform::park(wake as *const AtomicU32, snapshot, Some(remaining));
            }
        }
    }
}

// ============================================================================
// Lease — owned handle to a single in-flight request.
// ============================================================================

/// RAII handle to a single in-flight request.
///
/// On drop without `commit`, the lease writes a `ResponseFrame {
/// aborted: true, status: -1 }` to the slot's response buffer.
/// The lease is `Send`.
pub struct Lease {
    inner: Arc<ShmemServerInner>,
    slot: usize,
    req_seq: u64,
    #[allow(dead_code)]
    req_id: u32,
    payload_len: usize,
    committed: AtomicBool,
}

impl Lease {
    fn new(inner: Arc<ShmemServerInner>, polled: PolledSlot) -> Self {
        Self {
            inner,
            slot: polled.slot,
            req_seq: polled.req_seq,
            req_id: polled.req_id,
            payload_len: polled.payload_len,
            committed: AtomicBool::new(false),
        }
    }

    /// Request payload bytes for this lease. The slice is borrowed
    /// from the shmem slot — valid until `commit` / drop releases the
    /// slot.
    pub fn payload(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.inner.req_payload_ptr(self.slot), self.payload_len)
        }
    }

    /// Copy `bytes` into the slot's response buffer, then publish to
    /// the client. Consumes the lease.
    pub fn commit(self, bytes: &[u8]) -> Result<()> {
        if bytes.len() > self.inner.resp_buf_size {
            return Err(anyhow!(
                "response payload {} exceeds buffer {}",
                bytes.len(),
                self.inner.resp_buf_size
            ));
        }
        let dst = self.inner.resp_payload_ptr(self.slot);
        unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len()) };
        match self.inner.commit(self.slot, self.req_seq, bytes.len()) {
            Ok(()) => {
                self.committed.store(true, Ordering::Release);
                Ok(())
            }
            Err(e) => {
                self.committed.store(true, Ordering::Release);
                Err(e)
            }
        }
    }

    /// Convenience: build a minimal `StatusResponse { status }` frame
    /// and commit it.
    pub fn commit_status(self, status: i32) -> Result<()> {
        let buf = build_status_frame(status, false);
        self.commit(&buf)
    }

    /// Explicit abort. Same as dropping without commit. Consumes self.
    pub fn abort(self) {
        drop(self);
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        if !self.committed.load(Ordering::Acquire) {
            // Best-effort abort write. Failures here are unrecoverable
            // (the slot is stuck), so we swallow errors and let the
            // client hit its hard timeout.
            let buf = build_status_frame(-1, true);
            let dst = self.inner.resp_payload_ptr(self.slot);
            let len = buf.len().min(self.inner.resp_buf_size);
            unsafe { ptr::copy_nonoverlapping(buf.as_ptr(), dst, len) };
            let _ = self.inner.commit(self.slot, self.req_seq, len);
        }
    }
}

/// Build a `ResponseFrame { aborted, payload: StatusResponse { status } }`
/// rkyv-archived buffer.
fn build_status_frame(status: i32, aborted: bool) -> Vec<u8> {
    use crate::schema::{ResponseFrame, ResponsePayload, StatusResponse};
    crate::wire::encode_response(&ResponseFrame {
        driver_id: 0,
        aborted,
        payload: ResponsePayload::Status(StatusResponse { status }),
    })
    .unwrap_or_default()
}

// ============================================================================
// ShmemClient — attaches to an existing region as the requester side.
// ============================================================================

pub struct ShmemClient {
    mapping: platform::ClientMapping,
    num_slots: usize,
    slot_stride: usize,
    req_buf_size: usize,
    next_slot: AtomicUsize,
    slot_locks: Vec<Mutex<()>>,
    /// Pure busy-spin window before falling back to a cross-process
    /// park on the slot's `resp_wake` atomic.
    spin_budget_us: u64,
    aborted: AtomicBool,
}

unsafe impl Send for ShmemClient {}
unsafe impl Sync for ShmemClient {}

impl ShmemClient {
    /// Open an existing shmem region.
    ///
    /// `spin_budget_us` controls how long the client busy-polls the
    /// slot's `resp_wake` atomic after a roundtrip before parking via
    /// the cross-process kernel primitive — see
    /// [`ShmemServer::create`] for the full semantics.
    ///
    /// Validates magic + schema_version + `expected_schema_hash`.
    /// Returns an error if the server's schema hash differs — that
    /// indicates producer and consumer were compiled against
    /// different schemas.
    pub fn open(name: &str, spin_budget_us: u64, expected_schema_hash: [u8; 8]) -> Result<Self> {
        let cname = CString::new(name).map_err(|e| anyhow!("shmem name {name:?}: {e}"))?;
        let mapping = platform::map_shmem_client(&cname)
            .map_err(|e| anyhow!("map_shmem_client({name}): {e}"))?;
        let base = mapping.base;
        let total_size = mapping.total_size;
        let read_u32_at =
            |off: usize| -> u32 { unsafe { (base.add(off) as *const u32).read_volatile() } };

        // Validate handshake. On any failure, drop the mapping
        // ourselves via `unmap_shmem_client` since we haven't yet
        // wrapped it in `Self`.
        macro_rules! fail {
            ($msg:expr) => {{
                unsafe { platform::unmap_shmem_client(&mapping) };
                return Err($msg);
            }};
        }

        let magic = read_u32_at(0);
        if magic != MAGIC {
            fail!(anyhow!(
                "shmem magic mismatch: got 0x{magic:08x}, want 0x{MAGIC:08x}"
            ));
        }
        let schema_v = read_u32_at(4);
        if schema_v != SCHEMA_VERSION {
            fail!(anyhow!(
                "shmem schema version mismatch: got {schema_v}, want {SCHEMA_VERSION}"
            ));
        }
        let mut server_hash = [0u8; 8];
        unsafe {
            ptr::copy_nonoverlapping(base.add(HDR_OFF_SCHEMA_HASH), server_hash.as_mut_ptr(), 8)
        };
        if server_hash != expected_schema_hash {
            fail!(anyhow!(
                "shmem schema_hash mismatch: server {server_hash:?}, client {expected_schema_hash:?}"
            ));
        }
        let num_slots = read_u32_at(HDR_OFF_NUM_SLOTS) as usize;
        let slot_stride = read_u32_at(HDR_OFF_SLOT_STRIDE) as usize;
        let req_buf_size = read_u32_at(HDR_OFF_REQ_BUF) as usize;
        let resp_buf_size = read_u32_at(HDR_OFF_RESP_BUF) as usize;

        if num_slots == 0 {
            fail!(anyhow!("shmem header reports num_slots=0"));
        }
        let expected_stride = match checked_slot_stride(req_buf_size, resp_buf_size) {
            Ok(v) => v,
            Err(e) => fail!(anyhow!("shmem header invalid slot geometry: {e}")),
        };
        if slot_stride != expected_stride {
            fail!(anyhow!(
                "shmem header inconsistent: slot_stride={slot_stride} != {expected_stride}"
            ));
        }
        let expected_total = match checked_region_size(num_slots, slot_stride) {
            Ok(v) => v,
            Err(e) => fail!(anyhow!("shmem header invalid region geometry: {e}")),
        };
        if total_size < expected_total {
            fail!(anyhow!(
                "shmem size too small: {total_size} < {expected_total}"
            ));
        }
        let slot_locks = (0..num_slots).map(|_| Mutex::new(())).collect();
        Ok(Self {
            mapping,
            num_slots,
            slot_stride,
            req_buf_size,
            next_slot: AtomicUsize::new(0),
            slot_locks,
            spin_budget_us,
            aborted: AtomicBool::new(false),
        })
    }

    pub fn abort(&self) {
        self.aborted.store(true, Ordering::Release);
        // Wake every parked client. Iterate slots and bump each
        // `resp_wake` — a parked `roundtrip` will see the value
        // change and return via the aborted-flag check.
        for i in 0..self.num_slots {
            let wake = self.resp_wake_atomic(i);
            wake.fetch_add(1, Ordering::Release);
            unsafe { platform::wake_all(wake as *const AtomicU32) };
        }
    }

    fn base(&self) -> *mut u8 {
        self.mapping.base
    }
    fn slot_addr(&self, i: usize) -> *mut u8 {
        unsafe { self.base().add(HEADER_SIZE + i * self.slot_stride) }
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
    /// Global request-wake atomic in the region header. Client bumps
    /// after writing a slot's req_seq; server's `poll_blocking` parks
    /// on this u32 cross-process.
    fn req_wake_atomic(&self) -> &AtomicU32 {
        unsafe { &*(self.base().add(HDR_OFF_REQ_WAKE) as *const AtomicU32) }
    }
    /// Per-slot response-wake atomic. Server bumps after `commit`;
    /// client `roundtrip` parks on this slot's u32 cross-process.
    fn resp_wake_atomic(&self, i: usize) -> &AtomicU32 {
        unsafe { &*(self.slot_addr(i).add(OFF_RESP_WAKE) as *const AtomicU32) }
    }

    /// Send one request and wait for its response. `payload` is the
    /// rkyv-archived request bytes; the response is returned as a
    /// freshly-allocated `Vec<u8>` (also rkyv-archived).
    ///
    /// Blocks via hybrid spin-then-park until the response arrives,
    /// the hard timeout fires, or [`Self::abort`] is called.
    pub fn roundtrip(&self, request_id: u32, payload: &[u8]) -> Result<Vec<u8>> {
        if payload.len() > self.req_buf_size {
            return Err(anyhow!(
                "request payload {} exceeds buffer {}",
                payload.len(),
                self.req_buf_size
            ));
        }

        // Pick a slot. Try-lock all candidates round-robin; fall back
        // to blocking on the first one if all are busy.
        let n = self.num_slots.max(1);
        let start = self.next_slot.fetch_add(1, Ordering::Relaxed);
        let mut slot_idx = start % n;
        let mut guard_opt = None;
        for k in 0..n {
            let candidate = (start + k) % n;
            if let Ok(g) = self.slot_locks[candidate].try_lock() {
                slot_idx = candidate;
                guard_opt = Some(g);
                break;
            }
        }
        let _guard = match guard_opt {
            Some(g) => g,
            None => self.slot_locks[slot_idx]
                .lock()
                .map_err(|_| anyhow!("slot mutex poisoned"))?,
        };
        let i = slot_idx;

        // Write payload + slot header.
        unsafe {
            ptr::copy_nonoverlapping(payload.as_ptr(), self.req_payload_addr(i), payload.len())
        };
        let slot_base = self.slot_addr(i);
        unsafe {
            (slot_base.add(OFF_REQ_ID) as *mut u32).write_volatile(request_id);
            (slot_base.add(OFF_REQ_LEN) as *mut u32).write_volatile(payload.len() as u32);
            (slot_base.add(OFF_SEND_WT) as *mut u64).write_volatile(now_us());
        }
        let new_seq = self.req_seq_atomic(i).load(Ordering::Relaxed) + 1;
        self.req_seq_atomic(i).store(new_seq, Ordering::Release);

        // Notify any server parked in `poll_blocking`. Bump the global
        // `req_wake` counter, then cross-process wake-all on it. The
        // counter increment is Release so the seq write above is
        // visible to whoever wakes.
        let req_wake = self.req_wake_atomic();
        req_wake.fetch_add(1, Ordering::Release);
        unsafe { platform::wake_all(req_wake as *const AtomicU32) };

        // Wait for response. Hybrid spin-then-park:
        //   * Phase 1 (busy-spin): for up to `spin_budget_us`, poll
        //     `resp_seq` with `spin_loop()` — no scheduler involvement.
        //   * Phase 2 (park): cross-process futex / WaitOnAddress /
        //     __ulock_wait on the slot's `resp_wake` atomic.
        let started = Instant::now();
        let hard_timeout = *HARD_TIMEOUT;
        let spin_deadline = (self.spin_budget_us != u64::MAX)
            .then(|| started + Duration::from_micros(self.spin_budget_us));

        // Phase 1: pure busy-spin. Hot loop, no syscalls.
        if self.spin_budget_us > 0 {
            let mut iters: u32 = 0;
            loop {
                if self.resp_seq_atomic(i).load(Ordering::Acquire) >= new_seq {
                    let resp_len =
                        unsafe { (slot_base.add(OFF_RESP_LEN) as *const u32).read_volatile() }
                            as usize;
                    let resp_bytes =
                        unsafe { std::slice::from_raw_parts(self.resp_payload_addr(i), resp_len) };
                    return Ok(resp_bytes.to_vec());
                }
                if self.aborted.load(Ordering::Acquire) {
                    return Err(anyhow!(
                        "shmem call aborted (slot {i}, request_id {request_id})"
                    ));
                }
                iters = iters.wrapping_add(1);
                if iters & 0xFF == 0 {
                    let now = Instant::now();
                    if now - started >= hard_timeout {
                        return Err(anyhow!(
                            "shmem call timed out after {hard_timeout:?} (slot {i}, request_id {request_id})"
                        ));
                    }
                    if spin_deadline.is_some_and(|deadline| now >= deadline) {
                        break;
                    }
                }
                std::hint::spin_loop();
            }
        }

        // Phase 2: park on the slot's resp_wake atomic.
        let resp_wake = self.resp_wake_atomic(i);
        loop {
            // Snapshot the wake counter *before* the seq check so any
            // race between our load and our park returns immediately
            // via value mismatch.
            let snapshot = resp_wake.load(Ordering::Acquire);
            if self.resp_seq_atomic(i).load(Ordering::Acquire) >= new_seq {
                break;
            }
            if self.aborted.load(Ordering::Acquire) {
                return Err(anyhow!(
                    "shmem call aborted (slot {i}, request_id {request_id})"
                ));
            }
            let now = Instant::now();
            if now - started >= hard_timeout {
                return Err(anyhow!(
                    "shmem call timed out after {hard_timeout:?} (slot {i}, request_id {request_id})"
                ));
            }
            let remaining = hard_timeout - (now - started);
            // SAFETY: `resp_wake` lives in the shmem region we have
            // mapped; the pointer stays valid for the syscall.
            unsafe {
                platform::park(resp_wake as *const AtomicU32, snapshot, Some(remaining));
            }
        }

        let resp_len =
            unsafe { (slot_base.add(OFF_RESP_LEN) as *const u32).read_volatile() } as usize;
        let resp_bytes = unsafe { std::slice::from_raw_parts(self.resp_payload_addr(i), resp_len) };
        Ok(resp_bytes.to_vec())
    }
}

impl Drop for ShmemClient {
    fn drop(&mut self) {
        // SAFETY: `mapping` came from `platform::map_shmem_client`.
        unsafe { platform::unmap_shmem_client(&self.mapping) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn unique_name(tag: &str) -> String {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("/pie_ipc_ipc_unit_{tag}_{}_{n}", std::process::id())
    }

    #[test]
    fn slot_stride_aligns_odd_sized_buffers() {
        let stride = checked_slot_stride(4097, 4099).unwrap();
        assert_eq!(stride % SLOT_ALIGN, 0);
        assert!(stride >= SLOT_HEADER_SIZE + 4097 + 4099);
        assert_eq!(
            checked_slot_stride(4096, 4096).unwrap(),
            SLOT_HEADER_SIZE + 4096 + 4096
        );
    }

    #[test]
    fn server_rejects_zero_slots() {
        let name = unique_name("zero_slots");
        let err = ShmemServer::create(&name, 0, 4096, 4096, 0, [0u8; 8])
            .err()
            .expect("zero-slot region should fail");
        assert!(format!("{err}").contains("num_slots"));
    }

    #[test]
    fn unbounded_spin_poll_respects_timeout() {
        let name = unique_name("unbounded_spin");
        let server = ShmemServer::create(&name, 1, 128, 128, u64::MAX, [0u8; 8]).unwrap();

        assert!(server.poll_blocking(Duration::ZERO).is_none());
    }

    #[test]
    fn stale_lease_commit_does_not_publish_new_sequence() {
        let name = unique_name("stale_seq");
        let server = ShmemServer::create(&name, 1, 128, 128, 0, [1u8; 8]).unwrap();
        let slot_ptr = server.inner.slot_base(0);
        write_u32(slot_ptr, OFF_REQ_ID, 7);
        write_u32(slot_ptr, OFF_REQ_LEN, 0);
        atomic_store_u64(slot_ptr, OFF_REQ_SEQ, 1);

        let lease = server.poll().expect("lease for seq=1");
        atomic_store_u64(slot_ptr, OFF_REQ_SEQ, 2);

        let err = lease
            .commit(b"late response")
            .expect_err("stale commit should fail");
        assert!(format!("{err}").contains("stale shmem lease"));
        assert_eq!(atomic_load_u64(slot_ptr, OFF_RESP_SEQ), 0);
    }
}
