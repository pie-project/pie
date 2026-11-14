use std::sync::{Arc, Weak};

use super::{ProcessId, ResidencyPlanner};

/// A process's hold on its place in the planner past its exit: the process
/// keeps one, its context one, and each of its KV working sets one until
/// that set is released. A process leaves the planner (`unregister`) at
/// once, but the pages it held come back only as its sets are released,
/// which waits for any fire still leasing one. Until the last hold drops,
/// the planner counts those pages as on their way back, as it does an
/// eviction's; the drop of the last settles them and pokes.
pub struct Departure {
    planner: Weak<ResidencyPlanner>,
    pid: ProcessId,
}

impl Departure {
    pub(super) fn new(planner: &Arc<ResidencyPlanner>, pid: ProcessId) -> Self {
        Self {
            planner: Arc::downgrade(planner),
            pid,
        }
    }
}

impl std::fmt::Debug for Departure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Departure").field("pid", &self.pid).finish()
    }
}

impl Drop for Departure {
    fn drop(&mut self) {
        if let Some(planner) = self.planner.upgrade() {
            planner.departed(self.pid);
        }
        let told = {
            let mut watched = WATCHED.lock().unwrap();
            let mut told = false;
            for watch in watched.values_mut() {
                // Only the watched process's newest run: a restart's old
                // run departing is not its departure.
                if watch.current == self.pid {
                    watch.gone = true;
                    told = true;
                }
            }
            told
        };
        if told {
            GONE.notify_waiters();
        }
    }
}

/// One watched process: the id of its newest run (a restart runs it again
/// under a new id) and whether that run has fully departed.
struct Watched {
    current: ProcessId,
    gone: bool,
}

/// Processes whose full departure (the last [`Departure`] hold dropped: the
/// process ended and every working set it held released) someone waits on,
/// by the id the watch was taken on.
static WATCHED: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<ProcessId, Watched>>,
> = std::sync::LazyLock::new(Default::default);

/// Told when a watched process has fully departed.
static GONE: tokio::sync::Notify = tokio::sync::Notify::const_new();

/// Process `old` is run again as `new` (a restart): a watch on it follows
/// the new run, and the old one's departure is not the process's.
pub fn follow_restart(old: ProcessId, new: ProcessId) {
    for watch in WATCHED.lock().unwrap().values_mut() {
        if watch.current == old {
            watch.current = new;
            watch.gone = false;
        }
    }
}

/// A watch on process `pid`'s full departure, taken as the process is
/// spawned, before it holds anything, so its departure cannot be missed
/// (`inferlet::process::spawn_watched`). Dropped, the watch goes.
pub struct DepartureWatch(ProcessId);

impl DepartureWatch {
    /// Watches `pid` from now. Taken by the spawn itself, before the
    /// process can run.
    #[must_use]
    pub fn new(pid: ProcessId) -> Self {
        WATCHED.lock().unwrap().insert(
            pid,
            Watched {
                current: pid,
                gone: false,
            },
        );
        Self(pid)
    }

    /// Whether the process (its newest run) has fully departed.
    #[must_use]
    pub fn gone(&self) -> bool {
        WATCHED
            .lock()
            .unwrap()
            .get(&self.0)
            .is_none_or(|watch| watch.gone)
    }

    /// The next departure of a watched process: arm it before looking.
    #[must_use]
    pub fn next() -> tokio::sync::futures::Notified<'static> {
        GONE.notified()
    }
}

impl Drop for DepartureWatch {
    fn drop(&mut self) {
        WATCHED.lock().unwrap().remove(&self.0);
    }
}
