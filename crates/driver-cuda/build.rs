//! The launch bridge (retirement plan phase A), behind the `bridge` feature.
//!
//! Without `bridge` this script does nothing at all — the crate's
//! toolkit-free build is load-bearing for CI and must not learn a CUDA
//! dependency here. With it, the Rust half of the flat launch ABI is
//! generated from the kernel table at build time and never committed:
//!
//! * `launch_bindings.rs` — `emit_rust_bindings` over every family table,
//!   included by `bind::abi::ffi`. These are DECLARATIONS, which is why
//!   they live with their caller: they are spelled in this crate's own
//!   `#[repr(C)]` mirrors (`WeightView`, `DType`, the workspace views), and
//!   any number of crates may declare one symbol.
//! * `rust_dispatch.rs` — `emit_rust_dispatch` over the same tables, the
//!   statement-keyed `match` the binder includes.
//!
//! The C shim that DEFINES those symbols is `kernels-cuda`'s, generated and
//! compiled by its `native` feature — which `bridge` turns on. A definition
//! may exist once, so the crate that owns the launchers owns the entry points
//! forwarding into them; this crate was only ever the first caller.
//!
//! The link directives for the archive live HERE, and the order is
//! load-bearing: a static archive is scanned once, in place, so the shim that
//! references the launchers must precede `pie_kernels_cuda` on the link line.
//! `kernels-cuda` emits the shim's directive, and cargo puts a dependency's
//! ahead of its dependent's; ours follow.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_BRIDGE").is_none() {
        return;
    }
    bridge::build();
}

mod bridge {
    use std::path::{Path, PathBuf};

    /// Every family table, in the crate's own concatenation order.
    fn tables() -> Vec<&'static [kernels_cuda::KernelSig]> {
        vec![
            kernels_cuda::attn::KERNELS,
            kernels_cuda::rope::KERNELS,
            kernels_cuda::norm::KERNELS,
            kernels_cuda::mlp::KERNELS,
            kernels_cuda::gemm::KERNELS,
            kernels_cuda::moe::KERNELS,
            kernels_cuda::ssm::KERNELS,
            kernels_cuda::quant::KERNELS,
            kernels_cuda::layout::KERNELS,
            kernels_cuda::sample::KERNELS,
            kernels_cuda::adapter::KERNELS,
            // The second table: launchers the driver fires with no DSL
            // statement (envelope tier, QKV split, mask packers, cell
            // moves). Same rows, same proof, outside the DSL-surface
            // equality — see `kernels_cuda::driver_internal`.
            kernels_cuda::driver_internal::DRIVER_KERNELS,
        ]
    }

    fn cuda_home() -> PathBuf {
        std::env::var_os("CUDA_HOME")
            .or_else(|| std::env::var_os("CUDA_PATH"))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"))
    }

    pub fn build() {
        let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
        let tables = tables();

        let bindings = kernels_cuda::abi::emit_rust_bindings(&tables);
        std::fs::write(out_dir.join("launch_bindings.rs"), bindings).expect("write bindings");

        // The DISPATCH, from the same rows. `emit_rust_dispatch` is the
        // sibling of the C++ `emit_dispatch` the declared executor
        // includes — same table, same guards, different strings — and it
        // is generated here for the reason the bindings are: a second
        // hand-written switch over a table that already knows the answer
        // is the duplication this whole arc removes.
        let dispatch = kernels_cuda::abi::emit_rust_dispatch(&tables);
        std::fs::write(out_dir.join("rust_dispatch.rs"), dispatch).expect("write dispatch");

        println!("cargo:rerun-if-env-changed=CUDA_HOME");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        let cuda_include = cuda_home().join("include");
        if !cuda_include.join("cuda_runtime.h").is_file() {
            panic!(
                "the `bridge` feature needs the CUDA toolkit headers, and \
                 {cuda_include:?} has no cuda_runtime.h. Set $CUDA_HOME/$CUDA_PATH \
                 or install the toolkit — or drop `bridge` for a toolkit-free build."
            );
        }

        // The supergraph's set-cond kernel, which is the ONE thing in this
        // crate that has to be device code: `cudaGraphSetConditional` is a
        // `__device__` function, so arming a conditional handle from inside a
        // graph — the whole point, since it is what removes the host
        // round-trip — cannot be spelled in Rust or in `.cpp`.
        //
        // Its own archive rather than a `.file` on the one above, because
        // that build is `cpp(true)` and this needs nvcc. It goes in
        // `driver-cuda` rather than `kernels-cuda` because its argument
        // is a conditional handle — a shell object — not a tensor; see
        // `src/device/graph.rs`'s header for the same argument.
        let supergraph = Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc/supergraph.cu");
        println!("cargo:rerun-if-changed=csrc/supergraph.cu");
        cc::Build::new()
            .cuda(true)
            .include(&cuda_include)
            .file(&supergraph)
            .compile("pie_supergraph");

        // The kernels archive the shim forwards into. Search paths come from
        // `kernels-cuda`'s own build script (the `native` feature this
        // crate's `bridge` turns on); the `-l` is ours so it lands AFTER the
        // shim's.
        println!("cargo:rustc-link-lib=static=pie_kernels_cuda");

        // The archive's own closure, `driver-cuda/build.rs`'s list minus
        // nvrtc (the NVRTC JIT is pipeline code, which stayed C++): dynamic
        // cudart + cublas + cublasLt, the driver-API stub for `cuMem*`, NCCL
        // for the custom all-reduce, and the C++ runtime.
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
}
