//! Inferlet program management, instantiation, and process execution.
//!
//! - [`process`]: spawn/list/attach/terminate a running guest instance.
//! - [`program`]: install/add a guest program (WASM component + manifest).
//! - `host`: the `pie:inferlet` WIT boundary (bindgen! + `Host*` impls) —
//!   internal wiring only, never named by external callers.
//! - `linker`/`python`/`sandbox`: component linking, Python guest support,
//!   filesystem/network policy — internal.

pub(crate) mod host;
pub(crate) mod linker;
pub mod process;
pub mod program;
pub(crate) mod python;
pub(crate) mod sandbox;

pub use process::ProcessId;
pub(crate) use process::{ProcessCtx, ProcessEvent};
pub(crate) use program::Manifest;
pub use program::ProgramName;
pub(crate) use sandbox::InstancePolicy;
