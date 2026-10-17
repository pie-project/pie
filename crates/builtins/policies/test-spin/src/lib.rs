//! Stages a table, then never returns: the host must cut it off at its
//! deadline and discard the table.

use policy::engine;

#[derive(Default)]
struct Spin;

impl engine::Policy for Spin {
    fn plan(&mut self, _now_ms: u64) {
        let _ = engine::install_schedule(&[("spun".to_string(), 1)], None);
        let mut n: u64 = 0;
        loop {
            n = std::hint::black_box(n.wrapping_add(1));
        }
    }
}

policy::engine!(Spin);
