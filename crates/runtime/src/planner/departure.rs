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
        let mut watched = WATCHED.lock().unwrap();
        if let Some(gone) = watched.get_mut(&self.pid) {
            *gone = true;
            drop(watched);
            GONE.notify_waiters();
        }
    }
}

/// Processes whose full departure (the last [`Departure`] hold dropped: the
/// process ended and every working set it held released) someone waits on,
/// and whether it has come.
static WATCHED: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<ProcessId, bool>>> =
    std::sync::LazyLock::new(Default::default);

/// Told when a watched process has fully departed.
static GONE: tokio::sync::Notify = tokio::sync::Notify::const_new();

/// A watch on process `pid`'s full departure, from now: taken while it
/// still runs, so its departure is not missed. Dropped, the watch goes.
pub struct DepartureWatch(ProcessId);

impl DepartureWatch {
    #[must_use]
    pub fn new(pid: ProcessId) -> Self {
        WATCHED.lock().unwrap().insert(pid, false);
        Self(pid)
    }

    /// Whether the process has fully departed.
    #[must_use]
    pub fn gone(&self) -> bool {
        WATCHED
            .lock()
            .unwrap()
            .get(&self.0)
            .copied()
            .unwrap_or(true)
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
