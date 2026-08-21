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
    SLOT_TOKENS_GT, SLOT_TOKENS_LE, SLOT_TOKENS_MULTIPLE, SLOT_WANTS_ATTN_SCORE, SLOT_WINDOW_ONE,
};
pub use stream::{Event, OwnedStream, PinnedBuf, StreamRef};
pub use vmm::{Arena, LOGICAL_PAGE_BYTES, PhysicalPool, PoolBudget, pages_for_bytes};
