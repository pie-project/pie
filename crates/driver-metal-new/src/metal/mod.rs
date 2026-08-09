//! The Apple half: every type here names a Metal or IOKit symbol.
//!
//! Gated on `cfg(target_vendor = "apple")` as a whole, which is what lets the
//! rest of the crate compile and test on a Linux host. The boundary is drawn
//! at "does this need a GPU to be correct", not at "is this about the GPU":
//! the tuning table is about the GPU and lives outside, because its inputs
//! are two integers.
//!
//! # `unsafe`
//!
//! Every objc2 message send is `unsafe`, so this half cannot carry the
//! workspace's `unsafe_code = "forbid"`. What it carries instead is the rule
//! that an `unsafe` block states the invariant it is relying on -- Metal's
//! own API contract does not stop being a contract because it is written in
//! Objective-C.
//!
//! # What is not here yet
//!
//! The device query is first because it is self-contained: it depends on no
//! other Metal object and it feeds [`crate::tuning`], which is already
//! complete and tested. [`context`] follows it -- the queue, the allocator
//! pair and the residency set, which every later object is created against.
//! [`heap`] places every long-lived buffer inside one resident range. The
//! [`pipeline`] compiles kernel text into pipeline states -- in batches, and
//! through [`archive`], which is what keeps a second start from paying for
//! the first one's compilation. [`encoder`] encodes a step against them and
//! waits for it with a bound. [`tables`] keeps the argument tables a step
//! binds, so encoding one allocates nothing. [`handle`] is the checked view
//! of a buffer sub-range that the launch path stores and binds.

mod archive;
mod context;
mod device;
mod elastic;
mod encoder;
mod external;
mod feedback;
mod fire;
mod fused;
mod grouped;
mod handle;
mod heap;
mod keepalive;
mod memory;
mod pipeline;
mod pool;
mod program;
mod ring;
mod runtime;
mod storage;
mod tables;
mod timestamp;
mod timing;

pub use archive::{Archives, CACHE_ENV, EXTENSION, MAX_AGE};
pub use context::{ALLOCATOR_COUNT, Context};
pub use device::DeviceInfo;
pub use elastic::{
    Arena, Budget, CHUNK, Elastic, Need, PAGE, Pressure, TILE, create as create_elastic,
    pages_for_bytes,
};
pub use encoder::{ArgumentTable, StepEncoder, Stepper, Visibility};
pub use external::{External, Externals, Mapped, page_size};
pub use feedback::{Feedback, Feedbacks};
pub use fire::{DeviceInputs, Execution, Mode, Prepare, PreparedFire};
pub use fused::M2Command;
pub use grouped::{GroupStats, LaneCandidate, M3Group, MAX_LANES, REGION_THREADS};
pub use handle::Handle;
pub use heap::{Heap, Slot};
pub use keepalive::{Keepalive, MIN_DEPTH, MIN_THREADGROUPS, THREADS_PER_THREADGROUP};
pub use memory::{Memory, Pages, reclaimable_pages};
pub use pipeline::{Archived, Compiled, Compiler, Math};
pub use pool::{DEFAULT_CAPACITY, Pool, PoolStats, SMALLEST_CLASS, Transient};
pub use program::{
    FusedExecutable, GroupedExecutable, ProgramExecutable, ProgramStage, Pso, RegionExecutable,
    StageExecutable,
};
pub use ring::{Ring, allocate};
pub use runtime::{
    MAX_FUSED_CHANNELS, MAX_REGIONS_PER_PROGRAM, MAX_REGIONS_PER_STAGE, ORDINAL_BASE, Runtime,
};
pub use storage::stage_plan_weights;
pub use tables::{MAX_BINDINGS, Tables};
pub use timestamp::{Granularity, Timestamps};
pub use timing::Timing;
