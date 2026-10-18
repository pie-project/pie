//! Stages a table, then traps: the host must discard it.

use policy::worker;

#[derive(Default)]
struct Trap;

impl worker::Policy for Trap {
    fn plan(&mut self, _now_ms: u64) {
        worker::install_schedule(&[("trapped".to_string(), 1)], None);
        panic!("trapping on purpose");
    }
}

policy::worker!(Trap);
