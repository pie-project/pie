//! What this machine has: the artifact store, the HF snapshot cache, and the
//! embedded Python-WASM runtime.
//!
//! These are not commands. They were under `ops/` beside `model` and `config`,
//! which put a namespace of subcommands and a helper module at the same level
//! and left no way to tell which was which by looking. `ops/` is the command
//! tree now; anything a command *uses* to reach this machine's disk lives here.
//!
//! The R3 provisioning IO (`hf`, `py_runtime`) is here for the same reason it
//! was in `ops/`: the worker lib links no download machinery, so the standalone
//! root is what fetches. Only the address changed.

pub mod hf;
pub mod py_runtime;
pub mod store;
