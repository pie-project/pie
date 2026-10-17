//! Stages a table and a scratch write, then traps: the host must discard
//! both.

use policy::{Value, engine, put};

#[derive(Default)]
struct Trap;

impl engine::Policy for Trap {
    fn plan(&mut self, _now_ms: u64) {
        let _ = engine::install_schedule(&[("trapped".to_string(), 1)], None);
        put("trapped", Some(&Value::Boolean(true)));
        panic!("trapping on purpose");
    }
}

policy::engine!(Trap);
