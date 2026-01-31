//! Pie Runtime - Programmable Inference Engine
//!
//! This crate provides the core runtime for the Pie inference engine.
//! It exposes functionality via PyO3 bindings for integration with Python.

// Public modules (core engine logic)
pub mod actor;
pub mod adapter;
pub mod api;
pub mod auth;
pub mod brle;
pub mod context;
pub mod dummy;
pub mod engine;
pub mod inference;
pub mod instance;
pub mod kvcache;
pub mod messaging;
pub mod model;
pub mod runtime;
pub mod server;
pub mod telemetry;
pub mod utils;

// Legacy modules (still used by api, instance, engine, ffi)
pub mod legacy_api;
pub mod legacy_kvs;
pub mod legacy_messaging;
pub mod legacy_model;
pub mod legacy_runtime;
pub mod legacy_server;
pub mod legacy_service;

// FFI module for PyO3 bindings, IPC, and format types
pub mod ffi;

// Re-export the Python module entry point
pub use ffi::_pie;

