use wasmtime::component::Resource;
use wasmtime_wasi_io::poll::{DynPollable, subscribe};

use super::bindings::clocks::{monotonic_clock, wall_clock};
use crate::WasiWebCtxView;
use crate::clock::{self, Deadline};

impl monotonic_clock::Host for WasiWebCtxView<'_> {
    fn now(&mut self) -> wasmtime::Result<monotonic_clock::Instant> {
        Ok(clock::now())
    }

    fn resolution(&mut self) -> wasmtime::Result<monotonic_clock::Duration> {
        Ok(1)
    }

    fn subscribe_instant(
        &mut self,
        when: monotonic_clock::Instant,
    ) -> wasmtime::Result<Resource<DynPollable>> {
        let deadline = self.table.push(Deadline(Some(when)))?;
        subscribe(self.table, deadline)
    }

    fn subscribe_duration(
        &mut self,
        when: monotonic_clock::Duration,
    ) -> wasmtime::Result<Resource<DynPollable>> {
        // A deadline past u64::MAX is "never"; waiting forever beats trapping.
        let deadline = self.table.push(Deadline(clock::now().checked_add(when)))?;
        subscribe(self.table, deadline)
    }
}

impl wall_clock::Host for WasiWebCtxView<'_> {
    fn now(&mut self) -> wasmtime::Result<wall_clock::Datetime> {
        let ns = clock::wall_now();
        Ok(wall_clock::Datetime {
            seconds: ns / 1_000_000_000,
            nanoseconds: (ns % 1_000_000_000) as u32,
        })
    }

    fn resolution(&mut self) -> wasmtime::Result<wall_clock::Datetime> {
        Ok(wall_clock::Datetime {
            seconds: 0,
            nanoseconds: 1_000_000,
        })
    }
}
