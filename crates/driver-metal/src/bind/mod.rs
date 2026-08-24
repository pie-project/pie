//! Where a bound launch meets the device.
//!
//! [`crate::baker`] decides symbol, grid and addresses host-side — the walk,
//! and it runs on any host; compiling a pipeline, staging the fire's tables
//! ([`tables`]) and encoding the commands ([`encode`]) cannot be.

pub mod encode;
pub mod tables;
