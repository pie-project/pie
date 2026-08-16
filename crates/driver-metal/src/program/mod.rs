//! User programs: compile, cache, run.
//!
//! The PTIR plane — programs the *user* supplies at run time, as opposed to
//! the model's own fire. Distinct from [`crate::channel`]'s external ABI
//! naming by design: see `tests/layering.rs`.
//!
//! * [`compile`] — turning `.metal` text into a compute pipeline state.
//! * [`cache`] — the program compile and the pipeline cache around it.
//! * [`executable`] — the executables the three launch paths share.
//! * [`single`] — one single-lane fire, prepared and executed (M1).
//! * [`fused`] — a fire's fused regions placed around someone else's forward (M2).
//! * [`grouped`] — up to 64 fires dispatched as one group of lanes (M3).

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
