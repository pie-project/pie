//! Runtime-owned planner for turning a checkpoint into resident bytes.
//!
//! `plan = compile(source_facts, program, target)`. None of the three inputs is
//! a model's name: the engine states what it needs as a contract over the
//! checkpoint's byte space, this crate reads the checkpoint's own metadata, and
//! the target carries the numbers a device measured. `tests/standalone.rs` pins
//! that as four properties rather than as prose.
//!
//! # It was called model-loader
//!
//! And the name claimed something it does not do. A *model* loader would know
//! model semantics — which tensor is an attention projection, what a layer is,
//! how a family is put together. This crate knows none of that and refuses to:
//! the whole of `tests/standalone.rs` is the property that no input to
//! `compile` is a family name, and `contract::materialize` synthesizes what to
//! produce from the checkpoint alone. What it does know is a CHECKPOINT — its
//! containers, its dtypes, its byte extents — which is what it is now named
//! for. The rename waited on the device half leaving, because a crate that
//! launched CUDA kernels was doing something a checkpoint reader does not.
//!
//! There is no C++ side. The engine's `load_plan_executor.hpp` and its
//! `WeightStore` were deleted with the rest of the C++ loader; the engines
//! link this crate as an rlib and pass Rust types. Reading a checkpoint's own
//! bytes is [`file`](crate::file)'s alone — the module named for what it
//! touches, a checkpoint's files, rather than for the crate it sits in — and
//! the compiler below it opens nothing; [`executor::walk`] does, which is
//! exactly why it is not the compiler.
//!
//! **AND NO DEVICE SIDE EITHER.** Nothing here calls a GPU, links a runtime,
//! or wants a toolkit — the whole crate is `dtype`, `half`, `ztensor`,
//! `serde` and `thiserror`, in every configuration. A plan for a CUDA target
//! is a claim about what a device *will* do, and this crate exists to make
//! that claim on a machine that has no device to check it against; a CUDA
//! arena used to live in [`executor`] behind an optional feature and is
//! `engine-cuda`'s now. What is left of the device here is vocabulary:
//! [`executor::arena::ArenaBacking`] is the seam a consumer supplies, and
//! [`plan::passes::tile`]'s `CUDA_*` constants are the words a plan carries
//! to whoever launches them.

pub mod codec;
pub mod contract;
pub mod dump;
pub mod error;
pub mod executor;
pub mod extent;
pub mod file;
pub mod plan;
#[cfg(feature = "testkit")]
pub mod testkit;
pub mod types;
pub mod verify;
