pub mod alloc;
pub mod conditional;
pub mod ctx;
pub mod elastic;
pub mod graph;
#[cfg(feature = "rocm")]
pub mod hip;
#[cfg(all(feature = "device", not(feature = "rocm")))]
pub mod recycle;
#[cfg(all(feature = "device", not(feature = "rocm")))]
pub mod slab;

/// Endpoint memory kept for the next channel of its shape.
#[cfg(feature = "rocm")]
pub use hip::held::pool;
#[cfg(all(feature = "device", not(feature = "rocm")))]
pub use recycle as pool;

/// The runtime, driver, NCCL and cuBLAS bindings the engine calls: CUDA's,
/// or on ROCm HIP's, RCCL's and rocBLAS's under the same names.
#[cfg(feature = "cuda")]
pub use cudarc::{cublas::sys as blas, driver::sys as dr, nccl::sys as nccl, runtime::sys as rt};
#[cfg(feature = "rocm")]
pub use hip::{blas, dr, nccl, rt};

pub use alloc::{
    Buffer, Pinned, Pinning, copy_any, copy_d2d, copy_d2h, free_bytes, stage_raw, write_raw,
    zero_span, zero_span_on,
};
pub use ctx::{Context, count, present};
pub use elastic::{Arena, PhysicalPool};
pub use graph::{Graph, GraphExec};
