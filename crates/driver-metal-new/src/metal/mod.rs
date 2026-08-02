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
//! [`pipeline`] compiles kernel text into pipeline states, and [`encoder`]
//! encodes a step against them and waits for it with a bound. [`tables`] keeps
//! the argument tables a step binds, so encoding one allocates nothing.

mod context;
mod device;
mod encoder;
mod external;
mod feedback;
mod heap;
mod pipeline;
mod pool;
mod tables;

pub use context::{ALLOCATOR_COUNT, Context};
pub use device::DeviceInfo;
pub use encoder::{ArgumentTable, StepEncoder, Stepper, Visibility};
pub use external::{External, Externals, Mapped, page_size};
pub use feedback::{Feedback, Feedbacks};
pub use heap::{Heap, Slot};
pub use pipeline::Compiler;
pub use pool::{DEFAULT_CAPACITY, Pool, PoolStats, SMALLEST_CLASS, Transient};
pub use tables::{MAX_BINDINGS, Tables};
