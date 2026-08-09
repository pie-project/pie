//! The launch ABI emitters, under the name this crate's consumers already
//! spell.
//!
//! The emitters themselves are [`kernels_cuda_new::abi`]'s. This module is one
//! `use`, and the whole of what it does is keep `kernels_cuda::abi::*`
//! resolving: this crate's `build.rs` generates `shim.cpp` and `ffi.rs` from
//! [`emit_c_shim`] and [`emit_rust_bindings_portable`], `examples/emit_device_typecheck`
//! spells [`emit_device_typecheck`], and `driver-cuda`'s test suite reads
//! [`emit_c_shim`], [`emit_rust_bindings`], [`emit_rust_dispatch`],
//! [`emit_layout_assertions`], [`entry_name`] and [`Record`] across two files.
//! Not one of them was edited when the emitters moved, which is the same
//! property [`crate::norm_device`] exists for and the reason both moves were
//! one commit rather than a flag day.
//!
//! # Why they went, when the artefacts they emit are this crate's
//!
//! Because what a module IS to a build is decided by its INPUTS, and this
//! one's were already next door. `emit_c_shim` writes an `extern "C"`
//! forwarder into a launcher nvcc compiled; `emit_device_typecheck` writes a
//! translation unit the ahead-of-time build compiles. Both of those are the
//! archive's artefacts, and neither emitter opens a `.cu`, runs nvcc or links
//! anything — they read `kernels_cuda_new::table` and
//! `kernels_cuda_new::device` and return a `String`.
//!
//! Leaving them here meant a build script that wanted one had to depend on a
//! crate whose other content is CMake's. `driver-cuda`'s did, and that edge
//! was one of the three `new-horizon.md` §21.5 counted — the only one with no
//! archive behind it. It is gone: `driver-cuda/build.rs` names
//! `kernels_cuda_new::abi` directly and this crate is off its
//! `[build-dependencies]`.
//!
//! What is left of `kernels-cuda` after that is exactly the archive, which
//! was the point. The two remaining edges are `bridge = ["kernels-cuda/native"]`
//! and the dev-dependency, and both are the same nvcc build under two names.
//!
//! # Why this file still exists rather than a `pub use` in `lib.rs`
//!
//! `build.rs` reads it with `#[path = "src/abi.rs"]`, because a build script
//! cannot depend on the crate it builds. A glob re-export is what makes that
//! include resolve to the moved emitters in both compilations — the library's
//! and the script's — with no edit to either, since `kernels-cuda-new` is in
//! this crate's `[dependencies]` AND its `[build-dependencies]` and has to
//! stay in both for exactly this reason.

pub use kernels_cuda_new::abi::*;
