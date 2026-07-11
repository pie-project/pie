//! Inferlet program management, instantiation, and process execution.

pub(crate) mod linker;
pub mod process;
pub mod program;
pub(crate) mod python;
pub(crate) mod sandbox;

pub use process::{ProcessCtx, ProcessEvent, ProcessId};
pub use program::{Manifest, ProgramName};
pub use sandbox::InstancePolicy;
