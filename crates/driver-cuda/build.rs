//! The launch bridge (retirement plan phase A), behind the `bridge` feature.
//!
//! Without `bridge` this script does nothing at all — the crate's
//! toolkit-free build is load-bearing for CI and must not learn a CUDA
//! dependency here. With it, both halves of the flat launch ABI are
//! generated from the kernel table at build time and never committed:
//!
//! * `shim.cpp` — `kernels_cuda::abi::emit_c_shim` over every family table,
//!   compiled by `cc` against the real headers into `libpie_launch_shim.a`.
//!   Compiling it is the same proof `tests/launch_abi.rs` runs per family,
//!   now against the real `cuda_runtime.h` instead of the stub.
//! * `launch_bindings.rs` — `emit_rust_bindings` over the same tables,
//!   included by `launch::ffi`. Both halves come from one read of one table
//!   in one process, so they cannot disagree with each other; the C++
//!   compiler is what keeps them from disagreeing with the launchers.
//!
//! The link directives for the archive live HERE, not in `kernels-cuda`'s
//! build script, and the order is load-bearing: a static archive is scanned
//! once, in place, so the shim that references the launchers must precede
//! `pie_kernels_cuda` on the link line. `cc` emits the shim's directive when
//! it runs; ours follow. (`driver-cuda/build.rs` documents the same rule for
//! the C++ shell.)

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

    fn csrc_src() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels-cuda/csrc/src")
    }

    /// The headers the shim compiles against: every family directory's
    /// `.hpp`s, plus the vendored Marlin wrapper (`moe`'s one out-of-tree
    /// row). The union of exactly the per-family lists the launch_abi tests
    /// prove — each family's set compiles alone there, and the shim is where
    /// they have to compile TOGETHER.
    fn includes() -> Vec<String> {
        let mut out = Vec::new();
        // `comm` joined when the fused all-reduce landing got a row. It is
        // the one directory no per-family `launch_abi` case covers, so
        // nothing proved its headers alone and the shim was the first
        // thing to ask for them — which is the failure mode the doc above
        // describes: a family's set compiling alone is not the shim
        // compiling, and here there was no alone-case either.
        for dir in [
            "attn", "rope", "norm", "mlp", "gemm", "moe", "ssm", "quant", "layout", "sample",
            "vision", "comm",
        ] {
            let mut hs: Vec<String> = std::fs::read_dir(csrc_src().join(dir))
                .unwrap_or_else(|e| panic!("csrc/src/{dir}: {e}"))
                .filter_map(|e| {
                    let n = e.ok()?.file_name().into_string().ok()?;
                    n.ends_with(".hpp").then(|| format!("{dir}/{n}"))
                })
                .collect();
            hs.sort();
            out.extend(hs);
        }
        out.push("../third_party/marlin_moe/marlin_moe_wrapper.hpp".into());
        out
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
        let includes = includes();
        let include_refs: Vec<&str> = includes.iter().map(String::as_str).collect();

        let shim = kernels_cuda::abi::emit_c_shim(&tables, &include_refs)
            .expect("two rows may not claim one entry point");
        let shim_path = out_dir.join("shim.cpp");
        std::fs::write(&shim_path, shim).expect("write shim.cpp");

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

        // The shim's own directive (`-lpie_launch_shim`) is emitted by `cc`
        // here, ahead of everything below.
        cc::Build::new()
            .cpp(true)
            .std("c++20")
            .include(csrc_src())
            .include(&cuda_include)
            .file(&shim_path)
            .compile("pie_launch_shim");

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
        // `src/cuda/graph.rs`'s header for the same argument.
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
