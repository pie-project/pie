//! Tier A's rows, under the names this crate's consumers already spell.
//!
//! The rows themselves are `kernels_cuda_new::device`'s. This module is one
//! `use`, and the whole of what it does is keep `kernels_cuda::norm_device::*`
//! resolving: `driver-cuda`'s build script generates its dispatch from
//! [`jit_dispatched`], its `bind::launch` and `bind::device` read [`ENTRIES`],
//! `bind::nvrtc` reads [`ELEMENTWISE`], and `examples/emit_device_typecheck`
//! spells all three. Not one of them was edited when the rows moved, which is
//! the only reason the move was one commit rather than a flag day across four
//! crates.
//!
//! # What a row here is, and why it is not a table row
//!
//! A row in [`crate::KERNELS`] describes a `pie_k_*` entry point: a C++ host
//! function holding a `<<<>>>`, taking a stream, compiled by nvcc months ago
//! into `libpie_kernels_cuda.a`. A row HERE describes a `__global__` template
//! and the type to instantiate it at — no entry point, no launcher, no `.cu`.
//! [`crate::abi::emit_device_typecheck`] turns one into a translation unit
//! that does not compile if the template path, the element type or the
//! operand list is wrong, which is how the ahead-of-time build checks rows it
//! does not itself launch.
//!
//! # Which direction the dependency runs, and when that changed
//!
//! It used to run the other way. These rows were authored in this file and
//! `kernels-cuda-new/src/device.rs` was a `pub use` of them, because while
//! the ahead-of-time path and the JIT path must both run, a symbol has to
//! have exactly one contract — and two copies of a table are two contracts
//! that agree until the day they do not.
//!
//! That argument never said which crate should hold the file, only that one
//! of them must. The JIT crate is where it belongs: 109 of the 198 table rows
//! still have no JIT twin, so this crate's archive is not going away — but an
//! archive is a CONSUMER of a contract, and a build needing CMake, nvcc and a
//! Linux target must not be what `model-compiler` depends on to read a
//! symbol's operand list. So the rows moved and the edge inverted, in one
//! change because a cycle admits no other order.
//!
//! One name moved with them. `ENTRIES` is [`kernels_cuda_new::device::ALTUP_AUX`]
//! there, because a unit's name is the file it compiles and `ENTRIES` says
//! nothing about which file that is. The alias is spelled once, below, rather
//! than at four call sites.

pub use kernels_cuda_new::device::{
    ALTUP_AUX as ENTRIES, DeviceKernel, ELEMENTWISE, JIT_DISPATCHED, jit_dispatched,
};
