//! Test infrastructure for integration tests.
//!
//! Provides mock device backends and environment helpers for testing
//! the runtime without a Python backend.

pub mod mock_device;
pub mod inferlets;
mod env;

pub use env::{MockEnv, create_mock_env};
