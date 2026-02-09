//! Pie Runtime - Programmable Inference Engine
//!
//! This crate provides the core runtime for the Pie inference engine.
//! It exposes functionality via PyO3 bindings for integration with Python.

// Public modules (core engine logic)
pub mod service;
pub mod adapter;
pub mod api;
pub mod auth;
pub mod context;
pub mod device;
pub mod bootstrap;
pub mod inference;
pub mod messaging;
pub mod model;
pub mod program;
pub mod runtime;
pub mod server;
pub mod linker;
pub mod process;
pub mod daemon;

// Re-export instance and output types from runtime
pub use runtime::instance;
pub use runtime::output;
pub mod telemetry;


// FFI module for PyO3 bindings, IPC, and format types
pub mod ffi;

// Re-export the Python module entry point
pub use ffi::_pie;
