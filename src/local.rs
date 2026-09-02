//! What this machine has: the artifact store, the HF snapshot cache, and the
//! embedded Python-WASM runtime.
//!
//! These are not commands: `ops/` is the command tree, and anything a command
//! *uses* to reach this machine's disk lives here.
//!
//! The provisioning IO (`hf`, `py_runtime`) is here because the worker lib
//! links no download machinery, so the standalone root is what fetches.

pub mod hf;
pub mod py_runtime;
pub mod store;
