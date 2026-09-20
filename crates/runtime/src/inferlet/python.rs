pub(crate) mod runtime;

#[cfg(not(target_arch = "wasm32"))]
pub(super) mod snapshot;

#[cfg(target_arch = "wasm32")]
pub(super) mod snapshot {
    use anyhow::{Result, bail};
    use wasmtime::Engine;
    use wasmtime::component::Component;

    pub(crate) async fn snapshot_from_bytes(
        _engine: &Engine,
        _raw_bytes: &[u8],
        _dep_components: Vec<Component>,
    ) -> Result<Vec<u8>> {
        bail!("python snapshots are not available in a browser host")
    }

    pub(crate) fn strip_module_data(_module_bytes: &[u8]) -> Result<Vec<u8>> {
        bail!("python module stripping is not available in a browser host")
    }
}
