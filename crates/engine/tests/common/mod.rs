//! Test infrastructure for integration tests.
//!
//! Provides mock device backends and environment helpers for testing
//! the runtime without a Python backend.

mod env;
pub mod inferlets;
pub mod mock_device;
pub mod sampler_assert;

// Compiled once per test binary; not every binary uses every re-export.
#[allow(unused_imports)]
pub use env::{MockEnv, create_mock_env};
