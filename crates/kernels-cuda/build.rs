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

#[cfg(feature = "native")]
mod native {
    use std::path::{Path, PathBuf};

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

        // Deliberately NO `cargo:rustc-link-lib` here. A static archive must come
        // after everything that references it on the link line, and cargo emits a
        // dependency's link directives BEFORE its dependent's -- so declaring it
        // here would put `-lpie_kernels_cuda` ahead of
        // `-lpie_driver_cuda_lib`, which is exactly backwards for a shell that
        // calls kernels. `driver-cuda`'s build.rs names both, in order. The
        // search paths above do propagate and are the half that belongs here.

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
