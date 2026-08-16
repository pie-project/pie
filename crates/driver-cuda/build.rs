//! The link closure, behind the `abi` feature — and nothing else.
//!
//! `abi` is the only feature producing an artifact that links a C++ closure,
//! and it is the narrowest gate that already implied these lines. The four
//! `["_cuda", "bridge"]` test targets stop getting them: `cudarc` is built with
//! `fallback-dynamic-loading`, so it `dlopen`s what it needs, and none of the
//! four names an `extern` the closure defines. `_cuda` would be WRONG — that
//! build is deliberately toolkit-free and CI depends on `cuda_home()` never
//! being called there. No parse can prove a link, so the deciding builds are
//! `cargo build -p driver-cuda --features abi,cuda-12` for necessity and
//! `cargo test -p driver-cuda --features cuda-12 --test real_prefill` for
//! sufficiency without it.

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

/// Dynamic cudart + cuBLAS + cuBLASLt, the driver-API stub for `cuMem*`, and
/// the C++ runtime.
///
/// The `cuda` stub is `cuMem*`'s, for the arena's virtual-memory reservations;
/// `cudart`/`cublas`/`cublasLt` are named because something in the final
/// artifact calls them directly rather than through `cudarc`; `stdc++`,
/// `pthread`, `m`, `dl`, `rt` are what a host C++ object needs and a Rust one
/// does not.
///
/// `-lnccl` is absent until an NCCL arm exists: `comm::all_reduce_bf16` is the
/// P2P arm and calls no NCCL entry point, `dist::all_reduce_bf16` refuses, and
/// `cudarc` is depended on without the `nccl` feature, so no binding exists to
/// call. Measured: `nm -u -D` finds no undefined `nccl*` and `readelf -d` no
/// `NEEDED` for `libnccl`. Two archives are absent for a harder reason —
/// `libpie_kernels_cuda.a` is no longer produced (a `-l` for that is a link
/// error, not a no-op) and `libpie_launch_shim.a` arrived through the deleted
/// `kernels-cuda` crate's `links` key, the last thread between the two.
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
    for lib in ["stdc++", "pthread", "m", "dl", "rt"] {
        println!("cargo:rustc-link-lib={lib}");
    }
}
