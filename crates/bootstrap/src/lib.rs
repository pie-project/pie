pub mod paths;

#[cfg(feature = "cli")]
mod cli;
#[cfg(feature = "cli")]
mod config;
#[cfg(feature = "cli")]
mod lifecycle;
#[cfg(feature = "cli")]
mod observe;
#[cfg(feature = "cli")]
pub mod report;

#[cfg(feature = "cli")]
pub use cli::*;
#[cfg(feature = "cli")]
pub use config::{Origin, cli_config_path};
