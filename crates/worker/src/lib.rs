#[cfg(feature = "net")]
#[global_allocator]
static GLOBAL_ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// mimalloc returning to the system the pages its freed blocks still hold:
/// it purges them only from later allocator calls, so the planes a load
/// lands stay committed through the reservations made right after
/// ([`engine::memory::return_freed_heap`]).
#[cfg(feature = "net")]
#[allow(unsafe_code)]
fn return_freed_heap() {
    // SAFETY: `mi_collect` takes no pointers; with `force` it purges this
    // thread's heap and the segments other threads abandoned.
    unsafe { libmimalloc_sys::mi_collect(true) }
}

pub mod backend;
mod boot;
pub mod config;
pub mod disk;
pub mod embedded;
#[cfg(feature = "net")]
pub mod serve;
#[cfg(feature = "standalone")]
pub mod standalone;
mod translate;
pub mod weights;

#[cfg(not(target_arch = "wasm32"))]
mod server;
#[cfg(not(target_arch = "wasm32"))]
pub use server::Server;

#[cfg(feature = "net")]
pub mod http;
#[cfg(feature = "net")]
mod link;

pub use config::Config;
pub use controller_api::Role;
#[cfg(feature = "net")]
pub use link::control::ControlLink;
#[cfg(feature = "net")]
pub use link::pool::{PoolStore, run as run_pool};
#[cfg(feature = "net")]
pub use serve::{WorkerHandle, run, run_with};

/// Swaps the worker policy for the one `config`'s `[runtime] worker_policy` names,
/// or for none (a SIGHUP's reload): it loads whole or the running one goes
/// on ([`runtime::policy::reload`]).
#[cfg(not(target_arch = "wasm32"))]
pub async fn reload_policy(config: &Config) -> anyhow::Result<()> {
    let next = match config.runtime.worker_policy.as_deref() {
        Some(spec) => Some((
            spec.to_string(),
            builtins::policy_component(spec)?.into_owned(),
        )),
        None => None,
    };
    tokio::task::spawn_blocking(move || {
        runtime::policy::reload(next.as_ref().map(|(name, c)| (name.as_str(), c.as_slice())))
    })
    .await?
    .map_err(|why| anyhow::anyhow!("reload the worker policy: {why}"))
}
