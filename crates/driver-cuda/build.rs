//! The link closure, behind the `abi` feature — and nothing else.
//!
//! **This script generated the launch bridge until north star §6.** With
//! `bridge` on it wrote `launch_bindings.rs` (`emit_rust_bindings` over every
//! family table, included by `bind::abi::ffi`), `rust_dispatch.rs`
//! (`emit_rust_dispatch`, included by `bind::dispatch_generated`),
//! `rust_dispatch_probe.rs` for the parity harness, located
//! `libpie_launch_shim.a` through `kernels-cuda`'s `links` key, checked that
//! every call the generated dispatch made resolved in it, and emitted the
//! `-l` line for it.
//!
//! # What died, and why each one died
//!
//! Every generated file had exactly one includer and all three are deleted
//! (`c9ab3a936`), so the writes had no reader. Underneath that, the tables
//! they were generated FROM are empty: `emit_rust_bindings` emits a
//! declaration per row `abi::stated()` keeps, `emit_rust_dispatch` emits an
//! arm per the same, `stated()` drops a row with no operands,
//! `table::ROW_TABLES` is `&[]` as of `5a789298b`, and every row
//! `table::KERNELS` still holds is a `Contract::sig`, which states none.
//!
//! **The three emitters themselves are NOT deleted, and deleting them would
//! be a serious mistake.** They live in `kernels_cuda_new::abi` as pure
//! `[KernelSig] -> String` functions, and `emit_c_shim` is what generates the
//! **device typecheck translation unit** that
//! `kernels-cuda-new/tests/device_typecheck_types.rs` compiles — the only
//! thing in the tree that catches a `unit!` declaration disagreeing with its
//! `.cuh` about a by-value aggregate. What died here is this script's CALL
//! SITES. A reader deleting "the emitters" as a group takes the typecheck
//! with them.
//!
//! The build-time checks went with the calls they guarded, for the same
//! reason and not as a loosening: `routed_rows_are_hosted`,
//! `every_call_resolves_in_the_shim`, `the_probe_is_the_arm_routing_emits`
//! and their helpers all asked questions about generated text, and there is
//! no generated text. `unit::unit_of` still answers the first of them from
//! the table alone, in `kernels-cuda-new`'s own tests.
//!
//! # What is KEPT, and the comment that was wrong about it
//!
//! The `-l` block below. It stood inside `mod bridge` under a comment calling
//! it *"the link closure the shim needs"*, and that sentence explains six
//! libraries by naming a seventh:
//!
//! * `nccl` is the custom all-reduce's — `comm::all_reduce_bf16`'s, one crate
//!   down, not the shim's.
//! * the `cuda` driver stub is `cuMem*`'s, for the arena's virtual-memory
//!   reservations.
//! * `cudart`, `cublas`, `cublasLt` are the CUDA runtime's, named because
//!   something in the final artifact calls them directly rather than through
//!   `cudarc`.
//! * `stdc++`, `pthread`, `m`, `dl`, `rt` are the C++ runtime's, which a
//!   host C++ object needs and a Rust one does not.
//!
//! Only the last group was ever the shim's, and the shim has **zero members**
//! at the cut: `emit_c_shim` emits one `pie_k_*` forwarder per stated row.
//!
//! # Why `abi`, and which build decides whether this can shrink
//!
//! `abi` is the only feature that produces an artifact linking a C++ closure
//! — the thirteen `pie_cuda_*` exports the engine binds — and it is where
//! these lines already were: `abi = ["bridge", …]` until this commit, so
//! **every `abi` build emitted exactly this block before and emits exactly
//! this block now.**
//!
//! What changes is the four `["_cuda", "bridge"]` test targets, which stop
//! getting it. That is argued and not assumed: `cudarc` is built with
//! `fallback-dynamic-loading` and a pinned version, so it emits no link line
//! and `dlopen`s every symbol it needs — the property this crate's
//! toolkit-free build is built on — and none of the four names an `extern`
//! the closure defines (`real_prefill`'s only `nccl`-shaped hit is a config
//! field, `all_reduce_p2p_max_rows: 0`).
//!
//! **It is not proven, because no parse can prove a link.** The build that
//! decides it is `cargo build -p driver-cuda --features abi,cuda-12` for the
//! block's necessity and `cargo test -p driver-cuda --features cuda-12
//! --test real_prefill` for its sufficiency without it. Both are the
//! integration pass's. Nothing here is deleted on a guess: the lines are
//! moved from a gate that is being removed to the narrowest surviving gate
//! that already implied it.
//!
//! `_cuda` would be WRONG and is worth saying so: that build is deliberately
//! toolkit-free, `cuda_home()` would be called on a machine with no CUDA, and
//! CI depends on it not being.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_ABI").is_none() {
        return;
    }
    link_closure();
}

fn cuda_home() -> PathBuf {
    std::env::var_os("CUDA_HOME")
        .or_else(|| std::env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"))
}

/// Dynamic cudart + cuBLAS + cuBLASLt, the driver-API stub for `cuMem*`,
/// NCCL for the custom all-reduce, and the C++ runtime.
///
/// `libpie_kernels_cuda.a` STOOD HERE and went first: `add_library` had no
/// sources of its own — its whole content was
/// `$<TARGET_OBJECTS:pie_flashinfer_cutlass_moe>` — so retiring the fused
/// CUTLASS MoE left the archive empty rather than small, and a `-l` for an
/// archive that is not produced is a link error rather than a no-op.
///
/// `libpie_launch_shim.a` stood beside it and goes with this commit. It was
/// a HOST C++ compile (`cc`, not `cmake`, not nvcc) of a `shim.cpp` that
/// `kernels-cuda/build.rs` generates from `emit_c_shim`, whose members were
/// `pie_k_*` forwarders into the `.cu` launchers, and it was what the
/// generated dispatch actually called. Its directory arrived through
/// `kernels-cuda`'s `links` key as `DEP_PIE_KERNELS_CUDA_LAUNCH_SHIM` —
/// **the last thread between this crate and `kernels-cuda`**, and it is cut
/// here. The archive still builds under that crate's `native` feature;
/// nothing in this one asks for it.
fn link_closure() {
    let cuda_lib = cuda_home().join("lib64");
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    for lib in ["cudart", "cublas", "cublasLt"] {
        println!("cargo:rustc-link-lib={lib}");
    }
    let stubs = cuda_lib.join("stubs");
    if stubs.is_dir() {
        println!("cargo:rustc-link-search=native={}", stubs.display());
    }
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=nccl");
    for lib in ["stdc++", "pthread", "m", "dl", "rt"] {
        println!("cargo:rustc-link-lib={lib}");
    }
}
