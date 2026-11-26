//! Programs held in host memory past what the host holds, dropped and
//! loaded again. A parked program a fire loads (`Shell::load_parked`) is
//! held for its next fire: ~320 MiB of host memory each on v6e Qwen3.5, so
//! a server meeting most of its 747 shapes held more than its cgroup's 150
//! GiB. Before a load, while the host's free memory does not hold the
//! program beside what the guest stages' compiles take, the least recently
//! fired program with a copy in the disk cache is parked again: its next
//! fire loads it from there (~0.7-0.9 s, never compiled). Dropping one
//! takes ~130 ms of the host's time, so a dropper thread drops them, off
//! the fire's path; a load waits only on the drops it needs the room of.

use std::sync::{Arc, Condvar, Mutex, mpsc};

use crate::device::Program;

/// Drops evicted programs on a thread of its own and counts the bytes they
/// hold until they are gone.
pub(crate) struct Dropper {
    send: Option<mpsc::Sender<(Vec<Arc<Program>>, u64)>>,
    pending: Arc<(Mutex<u64>, Condvar)>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Default for Dropper {
    #[allow(unsafe_code)]
    fn default() -> Dropper {
        let pending: Arc<(Mutex<u64>, Condvar)> = Arc::default();
        let (send, recv) = mpsc::channel::<(Vec<Arc<Program>>, u64)>();
        let left = Arc::clone(&pending);
        let handle = std::thread::Builder::new()
            .name("xla-program-dropper".into())
            .spawn(move || {
                for (programs, bytes) in recv {
                    drop(programs);
                    #[cfg(target_os = "linux")]
                    // SAFETY: `malloc_trim` only releases free heap pages.
                    unsafe {
                        libc::malloc_trim(0);
                    }
                    let (lock, ended) = &*left;
                    if let Ok(mut held) = lock.lock() {
                        *held = held.saturating_sub(bytes);
                    }
                    ended.notify_all();
                }
            })
            .ok();
        Dropper {
            send: Some(send),
            pending,
            handle,
        }
    }
}

impl Dropper {
    /// Hands `programs`, holding `bytes` of the host, to the dropper.
    pub(crate) fn drop_later(&self, programs: Vec<Arc<Program>>, bytes: u64) {
        let (lock, _) = &*self.pending;
        if let Ok(mut held) = lock.lock() {
            *held += bytes;
        }
        match &self.send {
            Some(send) if send.send((programs, bytes)).is_ok() => {}
            // no thread: dropped here
            _ => {
                if let Ok(mut held) = lock.lock() {
                    *held = held.saturating_sub(bytes);
                }
            }
        }
    }

    /// The bytes handed over and not yet dropped.
    pub(crate) fn pending(&self) -> u64 {
        self.pending.0.lock().map_or(0, |held| *held)
    }

    /// Waits until a drop under way ends (returns at once with none).
    pub(crate) fn wait(&self) {
        let (lock, ended) = &*self.pending;
        if let Ok(held) = lock.lock()
            && *held > 0
        {
            drop(ended.wait(held));
        }
    }
}

impl Drop for Dropper {
    fn drop(&mut self) {
        self.send.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
