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

/// A program held in host memory that may be dropped: when a fire last
/// took it (0: the loader holds it and no fire took it yet) and the host
/// memory it holds.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Droppable {
    pub(crate) key: [u8; 32],
    pub(crate) used: u64,
    pub(crate) bytes: u64,
}

/// Which of `held` to drop so that `fits` holds of the bytes dropped
/// (a program load or a host-tier kv landing asks it): least recently
/// fired first, the loader's untaken ones before any, `keep` never; and
/// whether that makes room. With no droppable program left short of room,
/// the caller goes ahead past the room: a fire's program loads, a kv slot
/// lands (neither is refused for programs held).
pub(crate) fn choose(
    mut held: Vec<Droppable>,
    keep: Option<[u8; 32]>,
    fits: impl Fn(u64) -> bool,
) -> (Vec<[u8; 32]>, bool) {
    held.retain(|d| Some(d.key) != keep);
    held.sort_by_key(|d| (d.used, d.key));
    let (mut dropped, mut freed) = (Vec::new(), 0u64);
    for d in held {
        if fits(freed) {
            break;
        }
        dropped.push(d.key);
        freed = freed.saturating_add(d.bytes);
    }
    (dropped, fits(freed))
}

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

#[cfg(test)]
mod tests {
    use super::{Droppable, choose};

    fn key(n: u8) -> [u8; 32] {
        [n; 32]
    }

    fn held(of: &[(u8, u64, u64)]) -> Vec<Droppable> {
        of.iter()
            .map(|&(k, used, bytes)| Droppable {
                key: key(k),
                used,
                bytes,
            })
            .collect()
    }

    /// A host bound of 100 with 95 held: a kv slot of 10 landing.
    fn landing(freed: u64) -> bool {
        95 - freed.min(95) + 10 <= 100
    }

    #[test]
    fn a_kv_landing_past_the_bound_drops_the_least_recently_fired_first() {
        // the loader's untaken program, then the one fired longest ago
        let (dropped, fits) = choose(held(&[(1, 9, 3), (2, 0, 2), (3, 4, 3), (4, 7, 3)]), None, landing);
        assert!(fits, "the landing finds room while droppable programs remain");
        assert_eq!(dropped, vec![key(2), key(3)], "5 bytes freed, the two oldest");
        // room already: nothing dropped
        let (dropped, fits) = choose(held(&[(1, 9, 3)]), None, |_| true);
        assert!(fits && dropped.is_empty());
    }

    #[test]
    fn the_program_about_to_load_is_never_dropped_for_itself() {
        let (dropped, _) = choose(held(&[(1, 0, 4), (2, 5, 4)]), Some(key(1)), landing);
        assert_eq!(dropped, vec![key(2)]);
    }

    #[test]
    fn with_none_left_to_drop_the_caller_goes_ahead_past_the_room() {
        // every program dropped and still short: the landing is not refused
        let (dropped, fits) = choose(held(&[(1, 3, 1), (2, 1, 1)]), None, landing);
        assert_eq!(dropped, vec![key(2), key(1)]);
        assert!(!fits);
        let (dropped, fits) = choose(Vec::new(), None, landing);
        assert!(dropped.is_empty() && !fits);
    }
}
