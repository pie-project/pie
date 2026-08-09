//! User programs: compile, cache, run.
//!
//! This is the PTIR plane — programs the *user* supplies at run time, as
//! opposed to the model's own fire. Two words used to collide here:
//! `src/pipeline.rs` was the channel plane and `metal/pipeline.rs` was the
//! shader compiler, one word at two altitudes. They are
//! [`crate::channel`] and [`compile`] now.
//!
//! The channel plane itself is NOT here. It is
//! [`crate::channel`] — `pub use driver::*` and nothing else, so it is this
//! crate's naming of an external ABI rather than a layer of it. Under
//! `gpu/program/` it made `gpu/device/ring.rs` point up at the shader
//! compiler it is not related to; see `tests/layering.rs`.
//!
//! * [`compile`] — turning `.metal` text into a compute pipeline state.
//! * [`cache`] — the program compile and the pipeline cache around it. Was
//!   `metal/runtime.rs`, named for the C++ `M1Runtime` class.
//! * [`executable`] — what a compiled program is: the executables the three
//!   launch paths share.
//! * [`single`] — one single-lane fire, prepared and executed. The M1 path.
//! * [`fused`] — a fire's fused regions placed around someone else's
//!   forward. The M2 path.
//! * [`grouped`] — up to 64 fires dispatched as one group of lanes. The M3
//!   path.

pub mod cache;
pub mod compile;
pub mod executable;
pub mod fused;
pub mod grouped;
pub mod single;

pub use cache::{
    MAX_FUSED_CHANNELS, MAX_REGIONS_PER_PROGRAM, MAX_REGIONS_PER_STAGE, ORDINAL_BASE, Runtime,
};
pub use compile::{Archived, Compiled, Compiler, Math};
pub use executable::{
    FusedExecutable, GroupedExecutable, ProgramExecutable, ProgramStage, Pso, RegionExecutable,
    StageExecutable,
};
pub use fused::M2Command;
pub use grouped::{GroupStats, LaneCandidate, M3Group, MAX_LANES, REGION_THREADS};
pub use single::{DeviceInputs, Execution, Mode, Prepare, PreparedFire};
