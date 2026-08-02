//! The Metal execution shell, in Rust.
//!
//! This crate grows beside `driver-metal` rather than inside it. The C++
//! shell keeps running and keeps its tests; nothing here is on the serving
//! path until a module here has an equivalent that passes them. That is the
//! whole reason for the second crate: a rewrite that has to keep the old one
//! working is a rewrite that can be abandoned halfway without a revert.
//!
//! # What is here, and why it is shaped this way
//!
//! The C++ shell is ~42k lines, of which 13 files and ~8.6k lines name a
//! Metal or Objective-C type at all. The other 80% is scheduling, geometry,
//! pool arithmetic and plan interpretation -- logic that never touches the
//! GPU and is only in C++ because it was written next to the part that does.
//! So the split here is by that line rather than by subsystem:
//!
//! * [`bump`], [`region`], [`shader`] and [`tuning`] are portable. They compile and test on any
//!   host, including the Linux boxes the rest of the workspace is developed
//!   on, because their inputs are text and integers.
//! * [`metal`] is Apple-only and is where every `unsafe` message send lives.
//!
//! The portable half is not a convenience. It is the half that can be tested
//! without a GPU, and keeping it importable from a Linux `cargo test` is what
//! stops it from drifting back into the untestable half.
//!
//! # Ownership
//!
//! The C++ shell hands out `void*` for every Metal object, because its header
//! is included by plain C++ translation units that cannot name an `id<>`.
//! Nothing here does. `Retained<ProtocolObject<dyn MTLBuffer>>` is the same
//! pointer with the retain/release already correct, and the reason the port
//! is worth doing at all is that the lifetime bugs the `void*` boundary can
//! express stop being representable.

#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![deny(missing_docs)]
#![deny(
    clippy::todo,
    clippy::unimplemented,
    clippy::dbg_macro,
    clippy::mem_forget
)]
#![deny(clippy::print_stdout, clippy::print_stderr)]

pub mod bump;
mod error;
pub mod region;
pub mod shader;
pub mod tuning;

pub use error::{Error, Result};
pub use region::Region;

#[cfg(target_vendor = "apple")]
pub mod metal;

#[cfg(target_vendor = "apple")]
pub use metal::{
    ArgumentTable, Compiler, Context, DeviceInfo, External, Externals, Feedback, Feedbacks, Heap,
    MAX_BINDINGS, Mapped, Pool, PoolStats, Slot, StepEncoder, Stepper, Tables, Transient,
    Visibility,
};
