//! Pie Runtime - Programmable Inference Engine
//!
//! This crate provides the core runtime for the Pie inference engine.
//! It exposes functionality via PyO3 bindings for integration with Python.

// Public modules (core engine logic)
pub mod api;
pub mod api_legacy;
pub mod auth;
pub mod dummy;
pub mod engine;
pub mod instance;
pub mod kvs;
pub mod messaging;
pub mod model;
pub mod runtime;
pub mod server;
pub mod service;
pub mod telemetry;
pub mod utils;

// FFI module for PyO3 bindings
mod ffi;

// Re-export the Python module entry point
pub use ffi::_pie;
