#[cfg(feature = "net")]
#[global_allocator]
static GLOBAL_ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod backend;
mod boot;
pub mod config;
pub mod disk;
pub mod embedded;
pub mod paths;
#[cfg(feature = "net")]
pub mod serve;
#[cfg(feature = "standalone")]
pub mod standalone;
pub mod translate;
pub mod weights;

#[cfg(not(target_arch = "wasm32"))]
mod server;
#[cfg(not(target_arch = "wasm32"))]
pub use server::Server;

mod executor;
#[cfg(feature = "net")]
mod link;

pub use config::Config;
pub use controller_api::Role;
#[cfg(feature = "net")]
pub use link::control::ControlLink;
#[cfg(feature = "net")]
pub use serve::{WorkerHandle, run, run_with};
