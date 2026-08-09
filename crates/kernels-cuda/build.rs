//! Build `csrc/` into `libpie_kernels_cuda.a`, and publish what the shell
//! needs to compile against it.
//!
//! Two jobs, and the second is the interesting one. Building the archive is
//! ordinary. Publishing is not: FlashInfer is fetched and PATCHED by this
//! crate's CMake — three source patches over a CPM cache shared by every
//! build directory on the machine — so `driver-cuda` must not fetch it a
//! second time. Its `batch/workspace.cu` calls FlashInfer's scheduler
//! directly, so it needs those headers on its include path, and it gets them
//! from here through cargo's `links` metadata rather than by running the
//! fetch again.
//!
//! Everything published below comes out of `pie_kernels_cuda_paths.txt`,
//! which the CMake writes at generate time. Nothing here re-derives a path
//! CMake already resolved.
//!
//! Without the `native` feature this script does nothing at all, which is the
//! point of the feature: `model-compiler` depends on this crate for the
//! signature table and must never pay nvcc to read it.
//!
//! # The launch shim
//!
//! `native` also generates `shim.cpp` — one `extern "C" pie_k_*` per stated
//! row, forwarding to the real launcher with its header in scope — and
//! compiles it into `libpie_launch_shim.a`.
//!
//! It lives here rather than in a consumer because the shim is the only thing
//! that DEFINES those symbols, and a symbol may have one definition. While
//! `driver-cuda` was the only caller it could own the shim; it is not, and a
//! second generator would make the two mutually exclusive in a binary rather
//! than merely redundant. The Rust `unsafe extern "C"` bindings are a
//! different matter and stay with their callers: those are DECLARATIONS, any
//! number of crates may state them, and `driver-cuda`'s name its own
//! `#[repr(C)]` mirrors.
//!
//! Everything the shim compiles against is this crate's already — `csrc/src`'s
//! per-family headers and the vendored Marlin wrapper — which is the same
//! argument in the other direction.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Table only. Silently, and on purpose: this is the path `model-compiler`
    // takes on every build, and a build script that printed something here
    // would print it on every build.
    //
    // The gate is `#[cfg]` rather than a runtime `CARGO_FEATURE_NATIVE` check
    // because the `cmake` crate is an OPTIONAL build-dependency: without the
    // feature it is not in the graph at all, and a runtime check would still
    // leave `cmake::Config` in the token stream for rustc to fail on.
    #[cfg(feature = "native")]
    native::build();
}

// This crate's own tables and emitters, read by the build script that builds
// them. A build script cannot depend on its own crate, so the modules are
// included by path instead — which works only because every one of them
// imports `kernels` and nothing else. A module here that reached for a
// sibling, or for `crate::`, would have to be pulled in with it.
//
// `abi.rs`'s two `crate::` mentions are a `#[cfg(test)]` module and an
// intra-doc link, neither of which this build sees.
//
// `dead_code` is allowed on each because this script uses two of `abi`'s
// emitters and the `KERNELS` statics; the rest of what a family module offers
// is unused HERE and load-bearing in the library. Silencing it per module is
// what keeps a real unused-code warning in the library visible.
#[cfg(feature = "native")]
#[allow(dead_code)]
#[path = "src/abi.rs"]
mod abi;
#[cfg(feature = "native")]
#[allow(dead_code)]
#[path = "src/adapter.rs"]
mod adapter;
#[cfg(feature = "native")]
#[allow(dead_code)]
#[path = "src/attn.rs"]
mod attn;
#[cfg(feature = "native")]
#[allow(dead_code)]
#[path = "src/driver_internal.rs"]
mod driver_internal;
#[cfg(feature = "native")]
#[allow(dead_code)]
#[path = "src/gemm.rs"]
mod gemm;
#[cfg(feature = "native")]
#[allow(dead_code)]
#[path = "src/layout.rs"]
mod layout;
#[cfg(feature = "native")]
#[allow(dead_code)]
#[path = "src/mlp.rs"]
mod mlp;
#[cfg(feature = "native")]
#[allow(dead_code)]
#[path = "src/moe.rs"]
mod moe;
#[cfg(feature = "native")]
#[allow(dead_code)]
#[path = "src/norm.rs"]
mod norm;
#[cfg(feature = "native")]
#[allow(dead_code)]
#[path = "src/quant.rs"]
mod quant;
#[cfg(feature = "native")]
#[allow(dead_code)]
#[path = "src/rope.rs"]
mod rope;
#[cfg(feature = "native")]
#[allow(dead_code)]
#[path = "src/sample.rs"]
mod sample;
#[cfg(feature = "native")]
#[allow(dead_code)]
#[path = "src/ssm.rs"]
mod ssm;

#[cfg(feature = "native")]
mod native {
    use std::path::{Path, PathBuf};

    /// Every family table, in the crate's own concatenation order.
    ///
    /// `driver_internal` joins them: its rows are launchers the driver fires
    /// with no DSL statement, which changes which invariant they answer to
    /// and not whether they need an entry point.
    fn tables() -> Vec<&'static [kernels::KernelSig]> {
        vec![
            crate::attn::KERNELS,
            crate::rope::KERNELS,
            crate::norm::KERNELS,
            crate::mlp::KERNELS,
            crate::gemm::KERNELS,
            crate::moe::KERNELS,
            crate::ssm::KERNELS,
            crate::quant::KERNELS,
            crate::layout::KERNELS,
            crate::sample::KERNELS,
            crate::adapter::KERNELS,
            crate::driver_internal::DRIVER_KERNELS,
        ]
    }

    fn csrc_src() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("csrc/src")
    }

    /// The headers the shim compiles against: every family directory's
    /// `.hpp`s, plus the vendored Marlin wrapper (`moe`'s one out-of-tree
    /// row). The union of exactly the per-family lists the `launch_abi` tests
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

    /// Generate `shim.cpp` and compile it into `libpie_launch_shim.a`.
    ///
    /// Compiling it is the proof, not a translation to be trusted: each
    /// `extern "C"` body CALLS the launcher with the real header in scope, so
    /// C++ overload resolution decides whether the row is right — arity,
    /// order, constness and width, all at once, all as compile errors.
    ///
    /// Emitted FIRST so that `-lpie_launch_shim` precedes every other `-l`
    /// this crate or its dependents state. A static archive is scanned once,
    /// in place, and the shim references `pie_kernels_cuda`; cargo emits a
    /// dependency's directives ahead of its dependent's, so the consumer that
    /// names `-lpie_kernels_cuda` lands after this by construction.
    fn shim(out_dir: &Path) {
        println!("cargo:rerun-if-env-changed=CUDA_HOME");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        let cuda_include = cuda_home().join("include");
        if !cuda_include.join("cuda_runtime.h").is_file() {
            panic!(
                "kernels-cuda's `native` feature needs the CUDA toolkit headers, \
                 and {cuda_include:?} has no cuda_runtime.h. Set \
                 $CUDA_HOME/$CUDA_PATH or install the toolkit — or drop `native` \
                 for the signature table alone."
            );
        }

        let includes = includes();
        let include_refs: Vec<&str> = includes.iter().map(String::as_str).collect();
        let text = crate::abi::emit_c_shim(&tables(), &include_refs)
            .expect("two rows may not claim one entry point");
        let shim_path = out_dir.join("shim.cpp");
        std::fs::write(&shim_path, text).expect("write shim.cpp");

        // The portable half of the Rust side, included by `kernels_cuda::ffi`.
        //
        // Only rows whose operands all cross as primitives, because this
        // block is placed in THIS crate and this crate holds no `#[repr(C)]`
        // mirror — an attention-workspace row would name a type that is not
        // here. The shell generates the full set for itself, where the
        // mirrors are; these are the rows a caller needs no layout to make,
        // which is what the loader's quantize/cast/scale are.
        std::fs::write(
            out_dir.join("ffi.rs"),
            crate::abi::emit_rust_bindings_portable(&tables()),
        )
        .expect("write ffi.rs");

        cc::Build::new()
            .cpp(true)
            .std("c++20")
            .include(csrc_src())
            .include(&cuda_include)
            .file(&shim_path)
            .compile("pie_launch_shim");
    }

    pub fn build() {
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        if target_os != "linux" {
            panic!(
                "kernels-cuda's `native` feature is Linux-only (got target_os={target_os:?}). \
             The signature table builds everywhere; only the CUDA archive does not."
            );
        }

        let csrc = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("csrc");
        for dir in ["CMakeLists.txt", "cmake", "src", "third_party"] {
            println!("cargo:rerun-if-changed=csrc/{dir}");
        }

        let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
        shim(&out_dir);

        for var in [
            "CUDACXX",
            "CMAKE_CUDA_COMPILER",
            "CMAKE_CUDA_ARCHITECTURES",
            "PIE_COMPILER_LAUNCHER",
            "PIE_CUDA_BUILD_MARLIN",
            "PIE_CUDA_BUILD_MARLIN_MOE",
            "CPM_SOURCE_CACHE",
        ] {
            println!("cargo:rerun-if-env-changed={var}");
        }

        let mut cfg = cmake::Config::new(&csrc);
        cfg.out_dir(PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("kernels-cuda"));
        // `ptir/tier0.cuh` includes `rng_contract.generated.h`: the bit mapping
        // tier-0 must reproduce exactly, because tier-1 emits the same RNG
        // through NVRTC and the two are diffed. That is a CONTRACT, not driver
        // machinery -- the same shape as `kernels.def` being read by both C++
        // and CMake -- so it is declared here rather than reached for by a
        // relative path, exactly as `driver-cuda/build.rs` declares it.
        let ptir_include = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("tensor-compiler/include");
        cfg.build_target("pie_kernels_cuda")
            .define("BUILD_SHARED_LIBS", "OFF")
            .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
            .define("PIE_PTIR_INCLUDE_DIR", &ptir_include);

        let build_dir = cfg.build().join("build");
        emit_link_search_paths(&build_dir);

        // Deliberately NO `cargo:rustc-link-lib=pie_kernels_cuda` here. A static
        // archive must come after everything that references it on the link
        // line, and cargo emits a dependency's link directives BEFORE its
        // dependent's -- so declaring it here would put `-lpie_kernels_cuda`
        // ahead of `-lpie_driver_cuda_lib`, which is exactly backwards for a
        // shell that calls kernels. `driver-cuda`'s build.rs names both, in
        // order. The search paths above do propagate and are the half that
        // belongs here.
        //
        // `-lpie_launch_shim` IS emitted here, by `shim()` above, and the same
        // rule is why: the shim references the launchers, so it must precede
        // the archive, and "before every dependent" is exactly where a
        // dependency's directives land.

        // --- the handoff -------------------------------------------------------
        //
        // Read as DEP_PIE_KERNELS_CUDA_<KEY> by any crate with a direct dependency
        // on this one. `include` is the conventional key (`driver`, `driver`
        // and `model-loader-capi` all publish one); the rest are this crate's,
        // and each names a tree that only exists because the CMake here fetched
        // or vendored it.
        let paths = read_paths(&build_dir);
        for key in [
            "lib",
            "include",
            "cccl",
            "flashinfer",
            "marlin",
            "marlin_moe",
            "has_marlin",
            "has_marlin_moe",
            "mamba_sm90",
        ] {
            let value = paths
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.as_str())
                .unwrap_or_else(|| {
                    panic!(
                        "csrc/CMakeLists.txt did not write `{key}=` into \
                     pie_kernels_cuda_paths.txt; the export block and this \
                     list have to name the same keys"
                    )
                });
            println!("cargo:{key}={value}");
        }

        println!(
            "cargo:rustc-env=PIE_KERNELS_CUDA_BUILD_DIR={}",
            build_dir.display()
        );
    }

    /// The `key=value` lines CMake generated. Multi-path values are `:`-joined,
    /// which is what a CMake `;`-list has to become to survive a cargo metadata
    /// line (cargo splits on nothing, but `;` in a `cargo:` value is a trap for
    /// every shell that later echoes it).
    fn read_paths(build_dir: &Path) -> Vec<(String, String)> {
        let file = build_dir.join("pie_kernels_cuda_paths.txt");
        let text = std::fs::read_to_string(&file).unwrap_or_else(|e| {
            panic!(
                "kernels-cuda's CMake did not write {}: {e}. It is generated by \
             the file(GENERATE) block at the bottom of csrc/CMakeLists.txt.",
                file.display()
            )
        });
        text.lines()
            .filter_map(|line| line.split_once('='))
            .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
            .collect()
    }

    /// Every directory under `build_dir` holding at least one static archive.
    /// CMake places `pie_kernels_cuda` under `lib/`, but the vendored object
    /// libraries land wherever their `add_library` put them, and a build-type
    /// subdirectory appears on multi-config generators.
    fn emit_link_search_paths(build_dir: &Path) {
        let mut dirs = std::collections::HashSet::new();
        walk(build_dir, &mut dirs);
        for d in &dirs {
            println!("cargo:rustc-link-search=native={}", d.display());
        }
    }

    fn walk(dir: &Path, out: &mut std::collections::HashSet<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut has_archive = false;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "a") {
                has_archive = true;
            }
        }
        if has_archive {
            out.insert(dir.to_path_buf());
        }
    }
}
