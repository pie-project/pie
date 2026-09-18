#[global_allocator]
static GLOBAL_ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod backend;
pub mod config;
pub mod disk;
pub mod serve;
pub mod translate;
pub mod weights;

mod executor;
mod link;

#[cfg(feature = "session-lifetime-diagnostic")]
pub mod session_lifetime_diagnostic {
    pub use crate::link::gateway::{
        GatewayLink, SessionObserver, SessionOwnerKeys, connect_gateway,
    };
}
pub use config::Config;
pub use controller_api::Role;
pub use link::control::ControlLink;
pub use serve::{WorkerHandle, run, run_with};
