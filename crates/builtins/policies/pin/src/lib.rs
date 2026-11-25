//! Sends a launch first to the worker its client names, over what
//! `default-cluster` does.
//!
//! - Hint `worker` (int or text): the worker's number. The other workers
//!   that take new processes come after it. One that is gone, draining or
//!   no decode worker is named for nothing, and with every worker full the
//!   launch waits as any other.

use std::collections::BTreeMap;

use cluster_defaults::{Defaults, named, wake};
use plex::prelude::*;

#[derive(Default)]
struct Pin {
    defaults: Defaults,
}

impl Policy for Pin {
    fn plan(&mut self, now: Mark) {
        wake([self.defaults.plan(now, named, BTreeMap::new())]);
    }
}

plex::export!(Pin);
