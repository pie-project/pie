//! The Vulkan execution shell: what it takes to actually FIRE the modules
//! `kernels-vulkan` compiles.
//!
//! `kernels-vulkan` is a table and 665 SPIR-V modules. It knows what each
//! entrypoint's operands are, what its push block looks like, and which device
//! features it needs — and it deliberately knows nothing about instances,
//! queues, descriptor pools or command buffers. This crate is that half.
//!
//! # Why this is not a port of `driver-metal`
//!
//! It shares that crate's vocabulary — the same [`kernels::LaunchRule`], the
//! same `Dims` field names — and it should, because a disagreement about which
//! rule a row names would be a real defect rather than a backend difference.
//!
//! But the thing a rule ANSWERS is not the same. Metal's encoder takes a thread
//! count and a threadgroup and sizes the group at dispatch time.
//! `vkCmdDispatch` takes only a count of workgroups, and how wide one is was
//! decided when `glslc` ran. So the driver's arithmetic is a division by a
//! number it does not choose, against a divisor that varies per module, and
//! [`geometry`] is that division, written down with the reason each rounding
//! goes the way it does.
//!
//! # What the split is for, which is not what `driver-metal`'s is for
//!
//! `driver-metal` is split by what a COMPILER will accept: no Linux host can
//! build an `objc2` message send, so its portable half exists to be buildable
//! away from a Mac.
//!
//! Vulkan is a loader, not a platform. Every line here compiles on every host
//! in the tree. The `native` feature gates what needs a GPU to be PRESENT, so
//! the portable half is defined by what can be PROVED without one — and that is
//! a much better deal than the Metal side got, because the device half is
//! testable on the same machine this crate is written on, against a validation
//! layer that turns a silent misuse into a failed test.
//!
//! # What is here so far
//!
//! [`geometry`], and it is deliberately first. Every kernel in this tree that
//! was wrong after the Vulkan port was wrong in its LAUNCH SHAPE and not in its
//! arithmetic, because an undershot Vulkan grid writes nothing, leaves the
//! buffer's birth zeros in the gap, and returns success from every call
//! involved. Getting the division right, and being able to check it against
//! each module's own declared workgroup, is the part of a Vulkan shell that
//! carries the defects.

// The manifest deliberately does not take the workspace lint table, because it
// forbids `unsafe_code` and every `ash` entry point is unsafe. The rest of that
// table is worth having, so it is restated here without that one name -- and
// the portable half keeps its own guarantee a different way, by containing no
// `unsafe` at all, which `tests/portable.rs` asserts.
#![deny(missing_docs)]
#![deny(
    clippy::todo,
    clippy::unimplemented,
    clippy::dbg_macro,
    clippy::mem_forget
)]
#![deny(clippy::print_stdout)]

#[cfg(feature = "native")]
pub mod binding;
#[cfg(feature = "native")]
pub mod device;
#[cfg(feature = "native")]
pub mod dispatch;
pub mod geometry;
pub mod lowering;
#[cfg(feature = "native")]
pub mod resources;
pub mod spirv;

#[cfg(feature = "native")]
pub use binding::{Arena, Resolve, Unbindable, bind, resolve};
pub use geometry::{Dims, Local, Module, Rule, Tile, Ungeometric, groups, lanes};
pub use lowering::{Call, Mismatch, Value, pack};
pub use spirv::Declared;
