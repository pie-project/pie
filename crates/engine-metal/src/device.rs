//! The thin layer beneath `kernels-metal`: the device everything is encoded
//! against, the command buffer a fire is encoded into, and the bytes no
//! kernel entry allocates.
//!
//! **WHY THIS EXISTS AT ALL, GIVEN `Ctx`.** `kernels_metal::Ctx` is
//! `dyn Encode` — a sink that takes a [`Fire`](kernels_metal::Fire) and a
//! flat argument list. It names a shader by path and an entrypoint by name
//! and stops there; it makes no device, opens no queue and allocates
//! nothing. So everything with a lifetime — the `MTLDevice`, the command
//! queue, the pipeline cache, the arena, the weight store, the pool pages,
//! the resident fire inputs — belongs to the shell, and this module is where
//! the shell touches Metal. Nothing above it names `objc2`.
//!
//! **The one structural divergence from the CUDA sibling, and it is the
//! handle.** A `kernels_cuda::Tensor` carries a device ADDRESS, so a shell
//! that owns one slab hands out `base + offset` and is done. A
//! `kernels_metal::Tensor` carries a `u32` the encode sink resolves, because
//! Metal binds a BUFFER and an OFFSET rather than a pointer — there is no
//! address to hand out. [`handles`] is that resolution: a table the shell
//! mints rows into and the sink reads, and the reason `Buffer::at` here
//! returns a handle where its CUDA twin returns a `u64`.
//!
//! **It compiles without Metal.** Every entry here is present on a
//! non-Apple target and answers [`Fault::Deviceless`](crate::Fault::Deviceless)
//! there. That is what lets the call-order code above it be type-checked by
//! a plain workspace sweep on Linux — the standing doctrine for this crate
//! since it was the dispatch layer alone.

pub mod alloc;
pub mod ctx;
pub mod handles;
pub mod library;

pub use alloc::Buffer;
pub use ctx::{Context, Pending, present};
pub use handles::{Binding, Handles};
pub use library::Pipelines;
