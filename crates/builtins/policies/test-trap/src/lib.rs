//! Stages a table, then traps: the host must discard it.

use policy::engine;

#[derive(Default)]
struct Trap;

impl engine::Policy for Trap {
    fn plan(&mut self, _now_ms: u64) {
        engine::install_schedule(&[("trapped".to_string(), 1)], None);
        panic!("trapping on purpose");
    }
}

policy::engine!(Trap);
