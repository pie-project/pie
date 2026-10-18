//! Where a process's passes may run, as the worker plane's policy installs it
//! (`install-offload`). No table leaves it to the worker's configuration.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::inferlet::process::ProcessId;

use super::schedule::plex_id;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Never,
    /// Where the host's estimate says the pass finishes soonest.
    Cost,
    Always,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rule {
    pub mode: Mode,
    /// A pass with fewer uncached tokens stays home.
    pub min_tokens: u32,
    /// Executors, best first; empty leaves them to the host.
    pub executors: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Table {
    pub entries: HashMap<ProcessId, Rule>,
    pub default: Option<Rule>,
}

static CURRENT: RwLock<Option<Arc<Table>>> = RwLock::new(None);

pub(crate) fn install(table: Option<Table>) {
    *CURRENT.write() = table.map(Arc::new);
}

/// `None` while no policy installed a table: the configuration decides.
/// `Some(None)`: a table stands and names no rule for `pid`, which then stays
/// home.
pub fn rule(pid: ProcessId) -> Option<Option<Rule>> {
    let table = CURRENT.read().clone()?;
    Some(
        table
            .entries
            .get(&plex_id(pid))
            .or(table.default.as_ref())
            .cloned(),
    )
}
