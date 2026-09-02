//! Owns the device, command queue, pipeline cache and allocations that
//! `kernels-metal`'s `dyn Encode` sink does not create itself.
//!
//! Metal binds a buffer + offset rather than a pointer, so
//! `kernels_metal::Tensor` carries a `u32` handle (resolved via
//! [`handles`]) instead of the `u64` device address CUDA uses.
//!
//! Compiles on non-Apple targets, returning
//! [`Fault::Deviceless`](crate::Fault::Deviceless) there.

pub mod alloc;
pub mod ctx;
pub mod handles;
pub mod library;

pub use alloc::Buffer;
pub use ctx::{Context, Pending, present, reservations};
pub use handles::{Binding, Handles};
pub use library::Pipelines;
