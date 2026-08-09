//! The one `#[cfg]`.
//!
//! Everything below this module needs a device to be correct; everything
//! above it does not. That is the whole of the boundary, and
//! `.wiki/driver/real-metal-north-star.md` §6 states why it is drawn here
//! rather than per subsystem:
//!
//! > Four gates inside one module is how `tables` and `resolve` came to sit
//! > ungated beside gated siblings and reach across. One gated subtree makes
//! > that unrepresentable rather than merely discouraged.
//!
//! The cut is *does answering this need a device*, not *is this about the
//! GPU*: [`crate::layout::tuning`] is entirely about the GPU and is above the
//! line, because its inputs are two integers.
//!
//! # The seven rooms
//!
//! * [`device`] — **the only place vendor vocabulary is allowed.** `queue`,
//!   `heap`, `residency`, `PSO`, `MTLBuffer` are the correct words here and
//!   nowhere else, because this is the layer read alongside Apple's
//!   documentation.
//! * [`weights`] — checkpoint to device. The plan half is in
//!   [`crate::layout`].
//! * [`pools`] — what `layout` planned, allocated.
//! * [`bind`] — a lowered launch becomes a kernel entry, its arguments and
//!   its grid.
//! * [`fire`] — what one fire keeps between steps: scratch, tables,
//!   recordings.
//! * [`program`] — user programs: compile, cache, channel, run.
//! * [`serve`] — the transfers the engine asks for.
//!
//! # `unsafe`
//!
//! Every objc2 message send is `unsafe`, so this half cannot carry the
//! workspace's `unsafe_code = "forbid"`. What it carries instead is the rule
//! that an `unsafe` block states the invariant it is relying on -- Metal's
//! own API contract does not stop being a contract because it is written in
//! Objective-C.

pub mod bind;
pub mod device;
pub mod fire;
pub mod pools;
pub mod program;
pub mod serve;
pub mod weights;

pub use device::{
    ALLOCATOR_COUNT, Allocation, Arena, ArgumentTable, Archives, Bind, Budget, CACHE_ENV, CHUNK,
    Command, Context, DEFAULT_CAPACITY, DeviceInfo, EXTENSION, Elastic, External, Externals,
    Feedback, Feedbacks, Granularity, Handle, Heap, Keepalive, MAX_AGE, MAX_BINDINGS, MIN_DEPTH,
    MIN_THREADGROUPS, Mapped, Memory, Need, PAGE, Pages, Pool, PoolStats, Pressure, Recording,
    Regions, Ring, SMALLEST_CLASS, Slot, StepEncoder, Stepper, TILE, THREADS_PER_THREADGROUP,
    Tables, Timestamps, Timing, Transient, Visibility, create_elastic, page_size, pages_for_bytes,
    reclaimable_pages, record,
};
pub use fire::{Lease, Recordings, Scratch, fingerprint};
pub use program::{
    Archived, Compiled, Compiler, DeviceInputs, Execution, FusedExecutable, GroupStats,
    GroupedExecutable, LaneCandidate, M2Command, M3Group, MAX_FUSED_CHANNELS, MAX_LANES,
    MAX_REGIONS_PER_PROGRAM, MAX_REGIONS_PER_STAGE, Math, Mode, ORDINAL_BASE, Prepare,
    PreparedFire, ProgramExecutable, ProgramStage, Pso, REGION_THREADS, RegionExecutable, Runtime,
    StageExecutable,
};
pub use weights::stage_plan_weights;
