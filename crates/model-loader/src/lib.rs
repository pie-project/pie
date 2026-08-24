//! Runtime-owned planner for Pie model loading.
//!
//! `plan = compile(source_facts, program, target)`. None of the three inputs is
//! a model's name: the driver states what it needs as a contract over the
//! checkpoint's byte space, the loader reads the checkpoint's own metadata, and
//! the target carries the numbers a device measured. `tests/standalone.rs` pins
//! that as four properties rather than as prose.
//!
//! There is no C++ side. The driver's `load_plan_executor.hpp` and its
//! `WeightStore` were deleted with the rest of the C++ loader; CUDA's arena
//! backing is [`executor::cuda`] and the drivers link this crate as an rlib,
//! through Rust types. Reading a checkpoint is [`checkpoint`]'s alone, and the
//! compiler below it opens nothing — [`executor::walk`] does, which is exactly
//! why it is not the compiler.

pub mod checkpoint;
pub mod codec;
pub mod contract;
pub mod dump;
pub mod error;
pub mod executor;
pub mod extent;
pub mod group_slot;
pub mod plan;
#[cfg(feature = "testkit")]
pub mod testkit;
pub mod types;
pub mod verify;
