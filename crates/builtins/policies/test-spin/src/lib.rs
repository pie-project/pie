//! Stages a row, then never returns: the host must cut it off at its
//! deadline and discard the row.

use plex::prelude::*;

#[derive(Default)]
struct Spin;

impl Policy for Spin {
    fn plan(&mut self, _now: Mark) {
        schedule::priority::set_default(1).ok();
        let mut n: u64 = 0;
        loop {
            n = std::hint::black_box(n.wrapping_add(1));
        }
    }
}

plex::export!(Spin);
