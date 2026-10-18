use super::bindings::random::{insecure, insecure_seed, random};
use crate::wasi::random as rng;
use crate::wasi::{WasiWeb, WasiWebCtxView, WasiWebView};

impl random::Host for WasiWebCtxView<'_> {
    fn get_random_bytes(&mut self, max_len: u64) -> wasmtime::Result<Vec<u8>> {
        rng::bytes(max_len)
    }

    fn get_random_u64(&mut self) -> wasmtime::Result<u64> {
        rng::u64()
    }
}

impl insecure::Host for WasiWebCtxView<'_> {
    fn get_insecure_random_bytes(&mut self, max_len: u64) -> wasmtime::Result<Vec<u8>> {
        rng::bytes(max_len)
    }

    fn get_insecure_random_u64(&mut self) -> wasmtime::Result<u64> {
        rng::u64()
    }
}

impl insecure_seed::Host for WasiWebCtxView<'_> {
    fn get_insecure_seed(&mut self) -> wasmtime::Result<(u64, u64)> {
        Ok(self.ctx.insecure_seed)
    }
}

pub(super) fn add_to_linker<T: WasiWebView + 'static>(
    linker: &mut wasmtime::component::Linker<T>,
) -> wasmtime::Result<()> {
    use super::bindings::random;

    random::random::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;
    random::insecure::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;
    random::insecure_seed::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;
    Ok(())
}
