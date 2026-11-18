//! Traps on every call: for the hosts' fault tests. Its writes are
//! discarded, and once its lease runs out its rows are withdrawn.

use plex::prelude::*;

#[derive(Default)]
struct Trap;

impl Policy for Trap {
    fn plan(&mut self, _now: Mark) {
        schedule::priority::set_default(1).ok();
        panic!("test-trap traps on every call");
    }
}

plex::export!(Trap);
