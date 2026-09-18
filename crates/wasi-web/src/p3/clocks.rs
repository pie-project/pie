use wasmtime::component::Accessor;

use super::bindings::clocks::{monotonic_clock, system_clock, types};
use crate::clock::{self, Wait};
use crate::{WasiWeb, WasiWebCtxView};

impl types::Host for WasiWebCtxView<'_> {}

impl monotonic_clock::Host for WasiWebCtxView<'_> {
    fn now(&mut self) -> wasmtime::Result<monotonic_clock::Mark> {
        Ok(clock::now())
    }

    fn get_resolution(&mut self) -> wasmtime::Result<types::Duration> {
        Ok(1)
    }
}

impl<U> monotonic_clock::HostWithStore<U> for WasiWeb {
    async fn wait_until(
        _store: &Accessor<U, Self>,
        when: monotonic_clock::Mark,
    ) -> wasmtime::Result<()> {
        Wait::until(when).await;
        Ok(())
    }

    async fn wait_for(
        _store: &Accessor<U, Self>,
        how_long: types::Duration,
    ) -> wasmtime::Result<()> {
        if how_long > 0 {
            Wait::duration(how_long).await;
        }
        Ok(())
    }
}

impl system_clock::Host for WasiWebCtxView<'_> {
    fn now(&mut self) -> wasmtime::Result<system_clock::Instant> {
        let ns = clock::wall_now();
        Ok(system_clock::Instant {
            seconds: i64::try_from(ns / 1_000_000_000)?,
            nanoseconds: (ns % 1_000_000_000) as u32,
        })
    }

    fn get_resolution(&mut self) -> wasmtime::Result<types::Duration> {
        Ok(1_000_000)
    }
}
