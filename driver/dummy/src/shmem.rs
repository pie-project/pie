//! POSIX shared-memory server for the BPIQ fast path.
//!
//! Port of `driver/portable/src/shmem_ipc.cpp`. Layout matches
//! `runtime/src/shmem_ipc.rs::SCHEMA_VERSION = 2`:
//!
//! ```text
//! [64-byte global header]
//!   0:  u32 magic = 0x50494533 ('PIE3')
//!   4:  u32 schema_version = 2
//!   8:  u32 num_slots
//!   12: u32 slot_stride
//!   16: u32 req_buf_size
//!   20: u32 resp_buf_size
//!   ...padding...
//!
//! [num_slots × slot_stride bytes]
//!   slot[i]:
//!     0:  u64 req_seq    (atomic; runtime bumps on send)
//!     8:  u64 resp_seq   (atomic; we bump on respond)
//!     16: u32 req_id
//!     20: u32 method_tag
//!     24: u32 req_payload_len
//!     28: u32 resp_payload_len
//!     32: u64 send_walltime_us
//!     40: u64 respond_walltime_us
//!     ...padding to 64...
//!     64: request payload (req_buf_size bytes)
//!     ...: response payload (resp_buf_size bytes)
//! ```

use std::ffi::CString;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};

pub const MAGIC: u32 = 0x50494533;
pub const SCHEMA_VERSION: u32 = 2;
pub const HEADER_SIZE: usize = 64;
pub const SLOT_HEADER_SIZE: usize = 64;

pub const METHOD_TAG_FIRE_BATCH: u32 = 0;

const OFF_REQ_SEQ: usize = 0;
const OFF_RESP_SEQ: usize = 8;
const OFF_REQ_ID: usize = 16;
const OFF_METHOD_TAG: usize = 20;
const OFF_REQ_LEN: usize = 24;
const OFF_RESP_LEN: usize = 28;
const OFF_RESPOND_WT: usize = 40;

#[derive(Debug)]
pub struct SlotRequest<'a> {
    pub req_id: u32,
    pub method_tag: u32,
    pub payload: &'a [u8],
}

pub struct ShmemServer {
    name: CString,
    num_slots: usize,
    req_buf_size: usize,
    resp_buf_size: usize,
    slot_stride: usize,
    total_size: usize,
    spin_us: u64,
    fd: libc::c_int,
    base: *mut u8,
    stop: AtomicBool,
}

// Safe to share across threads — all mutating access goes through `mmap`'d
// memory which is synchronized by the per-slot atomics.
unsafe impl Send for ShmemServer {}
unsafe impl Sync for ShmemServer {}

impl ShmemServer {
    /// Create a new shmem region. Replaces any stale region with the same
    /// name (best-effort `shm_unlink` first).
    pub fn create(name: &str, num_slots: usize, req_buf: usize, resp_buf: usize, spin_us: u64) -> Result<Self> {
        let cname = CString::new(name)
            .map_err(|e| anyhow!("shmem name {name:?} contains NUL: {e}"))?;
        let slot_stride = SLOT_HEADER_SIZE + req_buf + resp_buf;
        let total_size = HEADER_SIZE + num_slots * slot_stride;

        // Best-effort cleanup of stale region.
        unsafe { libc::shm_unlink(cname.as_ptr()) };

        let fd =
            unsafe { libc::shm_open(cname.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o600) };
        if fd < 0 {
            return Err(anyhow!(
                "shm_open({name}) failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        if unsafe { libc::ftruncate(fd, total_size as libc::off_t) } != 0 {
            unsafe {
                libc::close(fd);
                libc::shm_unlink(cname.as_ptr());
            }
            return Err(anyhow!(
                "ftruncate({total_size}) failed: {}",
                std::io::Error::last_os_error()
            ));
        }

        let p = unsafe {
            libc::mmap(
                ptr::null_mut(),
                total_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            unsafe {
                libc::close(fd);
                libc::shm_unlink(cname.as_ptr());
            }
            return Err(anyhow!(
                "mmap failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        let base = p as *mut u8;

        // Zero + write the header.
        unsafe {
            ptr::write_bytes(base, 0, total_size);
        }
        write_u32(base, 0, MAGIC);
        write_u32(base, 4, SCHEMA_VERSION);
        write_u32(base, 8, num_slots as u32);
        write_u32(base, 12, slot_stride as u32);
        write_u32(base, 16, req_buf as u32);
        write_u32(base, 20, resp_buf as u32);

        Ok(Self {
            name: cname,
            num_slots,
            req_buf_size: req_buf,
            resp_buf_size: resp_buf,
            slot_stride,
            total_size,
            spin_us,
            fd,
            base,
            stop: AtomicBool::new(false),
        })
    }

    pub fn name(&self) -> &str {
        self.name.to_str().unwrap_or("<invalid utf-8>")
    }
    pub fn num_slots(&self) -> usize { self.num_slots }
    pub fn req_buf_size(&self) -> usize { self.req_buf_size }
    pub fn resp_buf_size(&self) -> usize { self.resp_buf_size }

    /// Set the stop flag. Idempotent. The serve loop checks this between
    /// slot polls and exits cleanly.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Block forever, dispatching one request per ready slot to `handler`.
    /// `handler` returns the number of bytes written into the response
    /// buffer; 0 is a valid empty response.
    pub fn serve_forever<F>(&self, mut handler: F)
    where
        F: FnMut(&SlotRequest<'_>, &mut [u8]) -> usize,
    {
        let mut last_seen = vec![0u64; self.num_slots];

        while !self.stop.load(Ordering::Relaxed) {
            let mut did_work = false;

            for i in 0..self.num_slots {
                let slot = unsafe { self.base.add(HEADER_SIZE + i * self.slot_stride) };
                let req_seq = atomic_load_u64(slot, OFF_REQ_SEQ);
                if req_seq == last_seen[i] {
                    continue;
                }

                let req_id = read_u32(slot, OFF_REQ_ID);
                let method_tag = read_u32(slot, OFF_METHOD_TAG);
                let req_len = read_u32(slot, OFF_REQ_LEN) as usize;

                let req_payload = unsafe {
                    std::slice::from_raw_parts(slot.add(SLOT_HEADER_SIZE), req_len)
                };
                let resp_payload = unsafe {
                    std::slice::from_raw_parts_mut(
                        slot.add(SLOT_HEADER_SIZE + self.req_buf_size),
                        self.resp_buf_size,
                    )
                };

                let resp_len = handler(
                    &SlotRequest { req_id, method_tag, payload: req_payload },
                    resp_payload,
                );

                write_u32(slot, OFF_RESP_LEN, resp_len as u32);
                atomic_store_u64(slot, OFF_RESPOND_WT, now_us());
                // Publish: bump resp_seq to match req_seq.
                atomic_store_u64(slot, OFF_RESP_SEQ, req_seq);

                last_seen[i] = req_seq;
                did_work = true;
            }

            if !did_work {
                if self.spin_us > 0 {
                    std::thread::sleep(Duration::from_micros(self.spin_us));
                } else {
                    std::thread::yield_now();
                }
            }
        }
    }
}

impl Drop for ShmemServer {
    fn drop(&mut self) {
        unsafe {
            if !self.base.is_null() {
                libc::munmap(self.base as *mut libc::c_void, self.total_size);
            }
            if self.fd >= 0 {
                libc::close(self.fd);
            }
            libc::shm_unlink(self.name.as_ptr());
        }
    }
}

// ---------------------------------------------------------------------------
// Memory access helpers
// ---------------------------------------------------------------------------

fn write_u32(base: *mut u8, off: usize, v: u32) {
    unsafe {
        ptr::write_unaligned(base.add(off) as *mut u32, v);
    }
}
fn read_u32(base: *const u8, off: usize) -> u32 {
    unsafe { ptr::read_unaligned(base.add(off) as *const u32) }
}

// 8-byte atomics on the slot header. We use AtomicU64 references via
// pointer cast — the slot region is mmap'd shared and aligned by mmap,
// the offsets are 8-byte multiples per the layout.
fn atomic_load_u64(base: *const u8, off: usize) -> u64 {
    unsafe { (*(base.add(off) as *const AtomicU64)).load(Ordering::Acquire) }
}
fn atomic_store_u64(base: *mut u8, off: usize, v: u64) {
    unsafe { (*(base.add(off) as *const AtomicU64)).store(v, Ordering::Release) }
}

/// Wall-clock microseconds since the Unix epoch. Stamped into the slot's
/// `respond_walltime_us` so the runtime client can pair it with its own
/// `send_walltime_us` for request-latency telemetry.
fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_drop_unique_region() {
        // Use a unique name per test run so concurrent test invocations
        // don't collide.
        let name = format!("/pie_test_dummy_{}", std::process::id());
        let server = ShmemServer::create(&name, 4, 4096, 4096, 0).unwrap();
        assert_eq!(server.num_slots(), 4);
        assert_eq!(server.req_buf_size(), 4096);
        drop(server);
        // Region should be unlinked; re-creating with the same name works.
        let _ = ShmemServer::create(&name, 4, 4096, 4096, 0).unwrap();
    }
}
