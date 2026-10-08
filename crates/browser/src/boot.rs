use std::path::PathBuf;

#[cfg(target_arch = "wasm32")]
use anyhow::Context;
use anyhow::Result;

pub use runtime::embed::{BootConfig, BootSummary};

pub const MODELS_DIR: &str = "/models";

pub struct Mount {
    pub len: u64,
    pub fetch: Box<dyn ztensor::memfs::Fetch>,
}

pub async fn boot(
    config: BootConfig,
    model_name: &str,
    mount: Mount,
    device: Option<crate::engine::Device>,
) -> Result<BootSummary> {
    let artifact = PathBuf::from(MODELS_DIR).join(model_name);
    let Mount { len, fetch } = mount;
    ztensor::memfs::mount_lazy(&artifact, len, fetch);
    tracing::info!(
        path = %artifact.display(),
        len,
        window = ztensor::memfs::LAZY_WINDOW,
        "artifact mounted lazily"
    );

    let loaded = {
        let config = config.clone();
        let artifact = artifact.clone();
        let model_name = model_name.to_string();
        on_green_thread("load", move || {
            let engine = match device {
                Some(device) => Some((crate::engine::open(device)?, poem_ir::Platform::Wgpu)),
                None => None,
            };
            runtime::embed::load(&config, &artifact, &model_name, engine)
        })
        .await?
    };
    if let Some(stats) = ztensor::memfs::lazy_stats(&artifact) {
        tracing::info!(
            requests = stats.requests,
            mib = stats.bytes >> 20,
            "artifact ranges fetched for the boot"
        );
    }

    // Freed before the runtime spawns its fibers: wasm-bindgen's glue writes
    // import results through signed i32 offsets, so a fiber stack above 2 GiB
    // breaks every string-returning import.
    if let Some(chunks) = ztensor::memfs::unmount(&artifact) {
        let held: Vec<usize> = chunks
            .parts()
            .iter()
            .map(|part| std::sync::Arc::strong_count(part) - 1)
            .collect();
        tracing::info!(
            ?held,
            lazy = chunks.is_lazy(),
            "artifact unmounted (references left per chunk)"
        );
    }
    let probe = vec![0u8; 1 << 20];
    tracing::info!(
        probe_mib = (probe.as_ptr() as usize) >> 20,
        "a fresh allocation lands here after the unmount"
    );
    drop(probe);

    let host = runtime::embed::Host {
        name: "browser".into(),
        home: PathBuf::from("/"),
        worker_threads: 1,
    };
    let embedded = runtime::embed::start(&config, &host, artifact, loaded, Vec::new()).await?;
    let summary = embedded.summary.clone();
    std::mem::forget(embedded);
    Ok(summary)
}

#[cfg(target_arch = "wasm32")]
const LOAD_STACK: usize = 8 << 20;

#[cfg(target_arch = "wasm32")]
async fn on_green_thread<T: 'static>(
    name: &str,
    f: impl FnOnce() -> Result<T> + 'static,
) -> Result<T> {
    web_std::thread::Builder::new()
        .name(name.to_string())
        .stack_size(LOAD_STACK)
        .spawn(f)
        .with_context(|| format!("spawn the {name} thread"))?
        .await
}

#[cfg(not(target_arch = "wasm32"))]
async fn on_green_thread<T: 'static>(
    _name: &str,
    f: impl FnOnce() -> Result<T> + 'static,
) -> Result<T> {
    f()
}
