//! The CUDA substrate: streams, allocation, capture, graphs, and the elastic
//! virtual-memory arena.
//!
//! Everything above this module -- the store, the loader, the batch machinery,
//! the dispatcher -- is host logic that happens to issue CUDA calls. This is
//! the only place that names a CUDA symbol, so it is the only place with
//! `unsafe` in it, and the layer's job is to leave the callers above it with
//! nothing to get wrong.
//!
//! # The capture discipline, demonstrated
//!
//! The claim in the crate docs is that allocating inside a graph capture stops
//! being *representable*, not merely discouraged. That is a claim about the
//! type system, so it is checked by the compiler, in a doctest that must fail
//! to build:
//!
//! ```compile_fail
//! use driver_cuda::device::{Allocator, StreamRef};
//!
//! let mut alloc = Allocator::new();
//! let scope = alloc.begin_capture(StreamRef::null()).unwrap();
//!
//! // `begin_capture` took `&mut alloc` and `scope` still holds it, so this
//! // line cannot borrow `alloc` again. In the C++ shell the equivalent line
//! // compiles, runs, corrupts the capture, and reports it somewhere else.
//! let buf = alloc.alloc(1024).unwrap();
//!
//! drop(scope);
//! ```
//!
//! The same program with the allocation moved before the capture is accepted,
//! which is what makes the failure above about the capture rather than about
//! the borrow being unsatisfiable in general:
//!
//! ```no_run
//! use driver_cuda::device::{Allocator, StreamRef};
//!
//! let mut alloc = Allocator::new();
//! let buf = alloc.alloc(1024).unwrap();          // before: fine
//! let scope = alloc.begin_capture(StreamRef::null()).unwrap();
//! // ... record work on `scope.stream()` ...
//! let graph = scope.end().unwrap();
//! let buf2 = alloc.alloc(1024).unwrap();         // after: fine again
//! ```
//!
//! Freeing is the half a borrow cannot reach, since `Drop` runs wherever a
//! value dies -- including on another thread, mid-capture. That one is handled
//! at runtime instead, by [`Allocator`]'s deferred-free queue; see
//! [`alloc`](self::alloc) for why the decision and the queue have to move
//! under one lock.

mod alloc;
pub mod cublas;
mod device;
mod graph;
mod stream;
mod vmm;

pub use alloc::{
    Allocator, CaptureScope, DeviceBuffer, copy_raw_span, fill_raw_span, read_raw_span,
    write_raw_span,
};
pub use device::{COMPILED_MAJOR, Device};
#[cfg(feature = "_cuda")]
pub use graph::{Cond, SupergraphBuilder};
pub use graph::{ConditionalIf, Graph, GraphExec};
pub use graph::{
    PRED_SLOTS, PeelWindowWord, PredicateWord, SLOT_HAS_CUSTOM_MASK, SLOT_HAS_LORA,
    SLOT_HAS_STAGE_HOOKS, SLOT_HAS_WRITE_DESC, SLOT_PEEL_ALL_FAST, SLOT_PEEL_ALL_HOOKED,
    SLOT_TOKENS_GT, SLOT_TOKENS_LE, SLOT_TOKENS_MULTIPLE, SLOT_WANTS_ATTN_SCORE,
    SLOT_WINDOW_ONE,
};
pub use stream::{Event, OwnedStream, PinnedBuf, StreamRef};
pub use vmm::{Arena, LOGICAL_PAGE_BYTES, PhysicalPool, PoolBudget, pages_for_bytes};
