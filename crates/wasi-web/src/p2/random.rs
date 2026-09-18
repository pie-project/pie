use super::bindings::random::{insecure, insecure_seed, random};
use crate::WasiWebCtxView;
use crate::random as rng;

impl random::Host for WasiWebCtxView<'_> {
    fn get_random_bytes(&mut self, len: u64) -> wasmtime::Result<Vec<u8>> {
        rng::bytes(len)
    }

    fn get_random_u64(&mut self) -> wasmtime::Result<u64> {
        rng::u64()
    }
}

/// "Insecure" is allowed to be the same CSPRNG; there is no cheaper source here.
impl insecure::Host for WasiWebCtxView<'_> {
    fn get_insecure_random_bytes(&mut self, len: u64) -> wasmtime::Result<Vec<u8>> {
        rng::bytes(len)
    }

    fn get_insecure_random_u64(&mut self) -> wasmtime::Result<u64> {
        rng::u64()
    }
}

impl insecure_seed::Host for WasiWebCtxView<'_> {
    fn insecure_seed(&mut self) -> wasmtime::Result<(u64, u64)> {
        Ok(self.ctx.insecure_seed)
    }
}
