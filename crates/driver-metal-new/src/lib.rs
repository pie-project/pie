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

pub mod batch;
pub mod bump;
mod error;
pub mod facts;
pub mod loader;
pub mod model;
pub mod pipeline;
pub mod region;
pub mod shader;
pub mod store;
pub mod tuning;

pub use error::{Error, Result};
pub use facts::{ModelFacts, ModelFamily};
pub use region::Region;
pub use shader::{Batch, Request};

#[cfg(target_vendor = "apple")]
pub mod metal;

#[cfg(target_vendor = "apple")]
pub use metal::{
    Archived, Archives, Arena, ArgumentTable, Budget, CHUNK, Compiled, Compiler, Context,
    DeviceInfo, DeviceInputs, Elastic, Execution, External, Externals, Feedback, Feedbacks,
    FusedExecutable, Granularity, GroupStats, GroupedExecutable, Handle, Heap, Keepalive,
    LaneCandidate, M2Command, M3Group, MAX_BINDINGS, MAX_FUSED_CHANNELS, MAX_LANES,
    MAX_REGIONS_PER_PROGRAM, MAX_REGIONS_PER_STAGE, MIN_DEPTH, MIN_THREADGROUPS, Mapped, Math,
    Memory, Mode, Need, ORDINAL_BASE, PAGE, Pool, PoolStats, Prepare, PreparedFire, Pressure,
    ProgramExecutable, ProgramStage, Pso, RegionExecutable, Ring, Runtime, Slot, StageExecutable,
    StepEncoder, Stepper, THREADS_PER_THREADGROUP, TILE, Tables, Timestamps, Timing, Transient,
    Visibility, create_elastic, pages_for_bytes,
};
