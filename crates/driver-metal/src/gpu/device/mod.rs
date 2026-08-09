//! The only place vendor vocabulary is allowed.
//!
//! `queue`, `heap`, `residency`, `PSO`, `MTLBuffer`, `MTL4CommandAllocator`
//! are the *correct* words in this module and are wrong above it. The reason
//! is stated in `.wiki/driver/real-metal-north-star.md` §5 rule 4: this is
//! the layer a reader opens beside Apple's documentation, so it should spell
//! things the way that documentation does. Every layer above it is read
//! alongside the CUDA shell, where one word per concept matters more.
//!
//! # What is here
//!
//! The device query is first because it is self-contained: it depends on no
//! other Metal object and it feeds [`crate::layout::tuning`], which is
//! already complete and tested. [`context`] follows it -- the queue, the
//! allocator pair and the residency set, which every later object is created
//! against. [`heap`] places every long-lived buffer inside one resident
//! range; [`allocation`] is one standalone buffer that leaves the residency
//! set when it drops; [`allocator`] recycles the short-lived ones;
//! [`elastic`] is the
//! buffer whose address never moves and whose memory comes and goes.
//! [`encoder`] encodes a step against a pipeline and waits for it with a
//! bound. [`argtable`] keeps the argument tables a step binds, so encoding
//! one allocates nothing. [`handle`] is the checked view of a buffer
//! sub-range that the launch path stores and binds. [`recording`] is one
//! recorded unit; the cache that keeps them is fire state and lives in
//! [`crate::gpu::fire::recordings`].

pub mod allocation;
pub mod allocator;
pub mod archive;
pub mod argtable;
pub mod context;
pub mod probe;
pub mod elastic;
pub mod encoder;
pub mod external;
pub mod feedback;
pub mod handle;
pub mod heap;
pub mod keepalive;
pub mod memory;
pub mod recording;
pub mod regions;
pub mod ring;
pub mod step_cost;
pub mod timestamp;

pub use allocator::{DEFAULT_CAPACITY, Pool, PoolStats, SMALLEST_CLASS, Transient};
pub use archive::{Archives, CACHE_ENV, EXTENSION, MAX_AGE};
pub use argtable::{MAX_BINDINGS, Tables};
pub use context::{ALLOCATOR_COUNT, Context};
pub use probe::DeviceInfo;
pub use elastic::{
    Arena, Budget, CHUNK, Elastic, Need, PAGE, Pressure, TILE, create as create_elastic,
    pages_for_bytes,
};
pub use encoder::{ArgumentTable, StepEncoder, Stepper, Visibility};
pub use external::{External, Externals, Mapped, page_size};
pub use feedback::{Feedback, Feedbacks};
pub use handle::Handle;
pub use heap::{Heap, Slot};
pub use keepalive::{Keepalive, MIN_DEPTH, MIN_THREADGROUPS, THREADS_PER_THREADGROUP};
pub use memory::{Memory, Pages, reclaimable_pages};
pub use recording::{Bind, Command, Recording, record};
pub use regions::Regions;
pub use allocation::Allocation;
pub use ring::Ring;
pub use step_cost::Timing;
pub use timestamp::{Granularity, Timestamps};
