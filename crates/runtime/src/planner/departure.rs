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
    }
}
