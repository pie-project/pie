//! The read-back room guest passes collect their status and puts into: a
//! pool of readback buffers made at load, a frame's passes all reading back
//! into the one it takes. A mapped buffer takes no part in a submission, so
//! a frame's buffer is its own until every pass it carried has been read,
//! and goes back to the pool then. A frame that finds none free waits for a
//! frame ahead of it to land (each was submitted, so each lands).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use crate::device::Buffer;
use engine_gpu::error::{Fault, Result};

/// The pool: buffers of `bytes` each, those free now.
#[derive(Debug)]
pub struct Pool {
    free: Mutex<Vec<Buffer>>,
    given: Condvar,
    bytes: u64,
}

impl Pool {
    /// A pool of `buffers`, each at least `bytes`.
    #[must_use]
    pub fn new(buffers: Vec<Buffer>) -> Arc<Pool> {
        let bytes = buffers.iter().map(Buffer::bytes).min().unwrap_or(0);
        Arc::new(Pool {
            free: Mutex::new(buffers),
            given: Condvar::new(),
            bytes,
        })
    }

    /// The bytes one frame's passes read back at most.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// A buffer for a frame's passes, once one is free.
    pub fn take(self: &Arc<Self>) -> Arc<Outbox> {
        let mut free = self.free.lock().expect("outbox pool lock poisoned");
        loop {
            if let Some(buffer) = free.pop() {
                return Arc::new(Outbox {
                    buffer: Some(buffer),
                    pool: Arc::clone(self),
                    used: AtomicU64::new(0),
                    collected: OnceLock::new(),
                });
            }
            free = self.given.wait(free).expect("outbox pool lock poisoned");
        }
    }

    fn give(&self, buffer: Buffer) {
        self.free
            .lock()
            .expect("outbox pool lock poisoned")
            .push(buffer);
        self.given.notify_all();
    }
}

/// A frame's read-back buffer: each pass it carries reads back into its own
/// span of it, and once the frame lands every span is read in one map.
#[derive(Debug)]
pub struct Outbox {
    buffer: Option<Buffer>,
    pool: Arc<Pool>,
    used: AtomicU64,
    collected: OnceLock<Vec<u8>>,
}

impl Outbox {
    #[must_use]
    pub fn buffer(&self) -> &Buffer {
        self.buffer
            .as_ref()
            .expect("an outbox holds its buffer until it drops")
    }

    /// A span of `len` bytes for a pass, from a word: refused by name past
    /// the buffer.
    pub fn reserve(&self, len: u64) -> Result<u64> {
        let len = len.next_multiple_of(4);
        let at = self.used.fetch_add(len, Ordering::Relaxed);
        let need = at + len;
        if need > self.buffer().bytes() {
            return Err(Fault::Ceiling {
                what: "bytes one frame's guest passes read back (a readback buffer made at load)",
                need,
                have: self.buffer().bytes(),
            });
        }
        Ok(at)
    }

    /// Reads what the frame's passes read back, the frame having landed.
    pub fn collect(&self) -> Result<&[u8]> {
        if let Some(collected) = self.collected.get() {
            return Ok(collected);
        }
        let mut out = vec![0u8; self.used.load(Ordering::Relaxed) as usize];
        crate::device::host::tab_wait("collects", "a guest outbox");
        self.buffer().read(0, &mut out)?;
        crate::device::host::tab_wait("collected", "a guest outbox");
        Ok(self.collected.get_or_init(|| out))
    }

    /// Reads what the frame's passes read back without blocking the thread
    /// it is called on, then runs `then`: `Some(why)` where it was not read.
    pub fn collect_async(self: &Arc<Self>, then: impl FnOnce(Option<String>) + Send + 'static) {
        let used = self.used.load(Ordering::Relaxed);
        if used == 0 || self.collected.get().is_some() {
            then(None);
            return;
        }
        let outbox = Arc::clone(self);
        crate::device::host::tab_wait("collects async", "a guest outbox");
        self.buffer().read_async(0, used, move |raw| match raw {
            Ok(raw) => {
                let _ = outbox.collected.set(raw);
                then(None);
            }
            Err(fault) => then(Some(fault.to_string())),
        });
    }

    /// The bytes of `len` at `at`, collected.
    #[must_use]
    pub fn collected(&self, at: u64, len: u64) -> Option<&[u8]> {
        let collected = self.collected.get()?;
        collected.get(at as usize..(at + len) as usize)
    }
}

impl Drop for Outbox {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.give(buffer);
        }
    }
}
