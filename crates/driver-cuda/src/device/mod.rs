//! The CUDA substrate: streams, allocation, capture, graphs, and the elastic
//! virtual-memory arena. The only place that names a CUDA symbol, so the only
//! place with `unsafe`. Freeing is deferred because `Drop` can run on another
//! thread mid-capture, past a borrow's reach; [`Allocator`] queues it.
//! Allocating during a capture is unrepresentable too: `begin_capture` takes
//! `&mut Allocator`, forbidding further allocation for the capture's life.

mod alloc;
pub mod cublas;
mod device;
mod graph;
mod stream;

pub use alloc::{Allocator, CaptureScope, DeviceBuffer};
pub use device::{COMPILED_MAJOR, Device};
pub use graph::{ConditionalIf, Graph, GraphExec};
pub use stream::{Event, OwnedStream, PinnedBuf, StreamRef};
