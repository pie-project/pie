//! The thin layer beneath `kernels-cuda`: the stream everything is enqueued
//! on, and the bytes no kernel entry allocates.

pub mod alloc;
pub mod conditional;
pub mod ctx;
/// A budgeted pool of physical pages, and virtual ranges whose backing
/// grows and trims under a fixed address. Maps memory rather than
/// allocating it.
pub mod elastic;
pub mod graph;
/// The coordinate system one capture publishes, and the diff between two
/// of them.
pub mod map;
/// Captured-graph introspection: the walk [`map`] is built on, and the
/// write-side probes, the only place this shell prices a rebind.
pub mod nodes;

pub use alloc::{
    Buffer, Pinned, Pinning, copy_any, copy_d2d, copy_d2h, free_bytes, write_raw, zero_span,
    zero_span_on,
};
pub use elastic::{Arena, PhysicalPool};
pub use ctx::{Context, present};
pub use graph::{Graph, GraphExec};
