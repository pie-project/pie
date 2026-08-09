//! Build `csrc/` into `libpie_driver_cuda_lib.a` and emit the link line.
//!
//! This is the Cargo↔native *link bridge* for the CUDA shell: it invokes the
//! CMake build under `csrc/`, forwards the include handoffs cargo publishes
//! (`driver-abi`, `model-loader-capi`, `model-compiler`, and this workspace's
//! `driver` substrate), and emits the `cargo:rustc-link-*` directives for the
//! final binary — which rustc, not CMake, links.
//!
//! Build-time *discovery* — nvcc location, CUDA arch, sccache/ccache, NCCL,
//! the Marlin toggles, the CPM cache — stays in `csrc/CMakeLists.txt`, the
//! native build system's proper home for it.
//!
//! This used to live in `worker`'s build.rs, which meant the worker knew
//! where the C++ trees were and how to build them. It does not need to: it
//! needs a driver, and a driver is now a crate that builds itself. Selection
//! is still by feature, but the feature turns on a DEPENDENCY rather than a
//! branch in someone else's build script.

use std::path::{Path, PathBuf};

fn main() {
    let target_os = target_os();
    if target_os != "linux" {
        panic!(
            "driver-cuda is Linux-only (got target_os={target_os:?}). \
             On macOS, use `--features driver-metal`; \
             on Windows, the cuda flavor is not supported."
        );
    }

    let csrc = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("csrc");
    println!("cargo:rerun-if-changed=csrc/CMakeLists.txt");
    println!("cargo:rerun-if-changed=csrc/cmake");
    println!("cargo:rerun-if-changed=csrc/include");
    println!("cargo:rerun-if-changed=csrc/src");

    // Which CMake target to ask for. The default builds the library and
    // nothing else, which is what a normal `cargo build` wants: the 59 test
    // and bench executables are minutes of nvcc that a `cargo run` does not
    // need.
    //
    // But "not by default" became "by nobody". Eleven kernel-family renames
    // ran `cargo build -p driver-cuda`, went green every time, and never
    // compiled a single harness -- so the harnesses accumulated a dead
    // `namespace ops` alias, call sites naming a `kernels` not in scope,
    // tier-0 constants left behind in `kernels::ptir`, a `DType` that
    // resolved to the WRONG enum, and CMake targets that were never given
    // kernels-cuda's include path at all. The first real build of them
    // produced 2 executables out of 59.
    //
    // `PIE_DRIVER_CUDA_CHECKS=1` asks for `pie_driver_cuda_checks`, which
    // depends on the library AND every harness. It belongs in CI, and in the
    // hands of anyone about to move a header. Building it by hand with
    // `cmake --build` does NOT work: the defines below -- FlashInfer's fetched
    // include dirs, `PIE_FORWARD_INCLUDE_DIR`, the loader's -- come from
    // cargo's `links` handoff and exist nowhere else. A by-hand configure
    // silently drops them and fails in the library, which is a confusing way
    // to learn this.
    let checks = std::env::var_os("PIE_DRIVER_CUDA_CHECKS")
        .is_some_and(|v| v != "0" && v != "");
    println!("cargo:rerun-if-env-changed=PIE_DRIVER_CUDA_CHECKS");

    let mut cfg = cmake::Config::new(&csrc);
    cfg.out_dir(PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("cuda"));
    cfg.build_target(if checks { "pie_driver_cuda_checks" } else { "pie_driver_cuda_lib" })
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .define("PIE_DRIVER_ABI_INCLUDE_DIR", dep_include("PIE_DRIVER_ABI", "driver-abi"))
        .define("PIE_LOADER_INCLUDE_DIR", dep_include("PIE_LOADER", "model-loader-capi"))
        .define("PIE_FORWARD_INCLUDE_DIR", dep_include("PIE_FORWARD", "model"))
        .define("PIE_DRIVER_INCLUDE_DIR", dep_include("PIE_DRIVER", "driver"))
        .define("PIE_PTIR_INCLUDE_DIR", sibling("tensor-compiler").join("include"))
        .define("PIE_PTIR_RUNTIME_DIR", sibling("tensor-compiler").join("runtime"))
        .define("PIE_REPO_ROOT", repo_root())
        // The kernel archive's source tree and every include dir it was built
        // with. Both come from `kernels-cuda`'s build script, because only it
        // knows where FlashInfer landed -- it is the crate that fetches and
        // patches it, into its own OUT_DIR. Deriving these here would mean a
        // second fetch of the same repository and a second patch pass over
        // the CPM cache the first one already dirtied.
        .define("PIE_KERNELS_CUDA_SRC", dep_include("PIE_KERNELS_CUDA", "kernels-cuda"))
        .define(
            "PIE_KERNELS_CUDA_LIB",
            std::env::var("DEP_PIE_KERNELS_CUDA_LIB")
                .expect("kernels-cuda publishes the archive path as cargo:lib"),
        )
        .define("PIE_KERNELS_CUDA_INCLUDE_DIRS", kernels_include_dirs())
        // Both Marlin header trees, and whether each was actually built.
        //
        // `kernels-cuda` has published all four keys since the split; nothing
        // here read them, so a note in csrc/CMakeLists.txt concluded the paths
        // were unpublished and that wiring `bench/marlin_moe_verify.cu` needed
        // a change to the kernels crate. It needed these four lines.
        //
        // The `.cu`/`.cpp` behind each header are `target_sources` of the
        // kernel archive itself, so a consumer needs the include dir and
        // nothing more -- the symbols arrive with PIE_KERNELS_CUDA_LIB, but
        // ONLY if that tree's toggle was on when the archive was built. Hence
        // the `has_` keys travelling alongside: a consumer has to be able to
        // ask, or it compiles fine and fails at link.
        .define("PIE_MARLIN_INCLUDE_DIR", dep_meta("MARLIN"))
        .define("PIE_MARLIN_MOE_INCLUDE_DIR", dep_meta("MARLIN_MOE"))
        .define("PIE_HAS_MARLIN", dep_meta("HAS_MARLIN"))
        .define("PIE_HAS_MARLIN_MOE", dep_meta("HAS_MARLIN_MOE"));

    // nvcc discovery, CUDA arch, the sccache/ccache launcher, the Marlin
    // toggle and the CPM source cache are all handled by csrc's CMakeLists
    // (via find_program / `$ENV{...}`); we only declare the env deps here so
    // Cargo reconfigures when they change.
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

    let build_dir = cfg.build().join("build");
    add_link_search_paths(&build_dir);

    // --- link directives for the final rustc binary (CMake can't emit these) ---
    //
    // Order is load-bearing and is why both names are here rather than one in
    // each crate: a static archive is scanned once, in place, so the shell
    // must precede the kernels it calls. `kernels-cuda`'s build script emits
    // the search path for the second one and deliberately not the `-l`.
    println!("cargo:rustc-link-lib=static=pie_driver_cuda_lib");
    println!("cargo:rustc-link-lib=static=pie_kernels_cuda");

    // CUDA toolkit: dynamic cudart + cublas + cublasLt + nvrtc (gemm.cpp
    // references cublasLt directly; the Sampling-IR JIT calls the NVRTC
    // runtime-compilation API), plus the driver-API stub (`-lcuda`). Runtime
    // contract: the host ships the CUDA toolkit `.so`s.
    link_cuda_toolkit_dynamic(&["cudart", "cublas", "cublasLt", "nvrtc"]);
    link_cuda_driver_stub();
    // Only system NCCL is supported: CMake's `find_library(nccl)` locates the
    // header + library at configure time, and rustc resolves `-lnccl` here.
    println!("cargo:rustc-link-lib=nccl");

    add_system_libs();

    println!(
        "cargo:rustc-env=PIE_DRIVER_CUDA_BUILD_DIR={}",
        build_dir.display()
    );
}

/// An include directory a dependency published with `cargo:include`.
///
/// The whole point of `links` is that this answer comes from cargo. A missing
/// one is a wiring bug in that crate's manifest, not something to paper over
/// with a relative path.
fn dep_include(prefix: &str, crate_name: &str) -> PathBuf {
    let var = format!("DEP_{prefix}_INCLUDE");
    let dir = std::env::var(&var).unwrap_or_else(|_| {
        panic!(
            "{crate_name}'s build.rs did not emit cargo:include -- check that \
             its manifest sets the matching `links` key (read as ${var})"
        )
    });
    PathBuf::from(dir)
}

/// One `key=value` `kernels-cuda` published, read back by cargo's name for it.
///
/// Every key in that crate's export block is mandatory on its own side -- its
/// build script panics if CMake omitted one -- so a missing variable here means
/// the handoff itself is broken, not that this key is optional.
fn dep_meta(key: &str) -> String {
    let var = format!("DEP_PIE_KERNELS_CUDA_{key}");
    std::env::var(&var).unwrap_or_else(|_| {
        panic!(
            "kernels-cuda's build.rs did not emit ${var} -- it publishes every \
             key of its CMake export block, so this is a break in that block \
             or in the list build.rs copies from it"
        )
    })
}

/// The include path the shell compiles against, in the order the kernel
/// archive was built with.
///
/// CCCL first and FlashInfer's own headers ahead of anything the toolkit
/// ships: FlashInfer v0.6.15 needs the CCCL it vendors, and a CUDA toolkit
/// that bundles an older one would shadow it. The ordering is decided once,
/// in `kernels-cuda`'s CMake, and carried across intact rather than
/// reconstructed from a guess about which came first.
///
/// The `:` separators are `kernels-cuda`'s doing: a cargo metadata value is
/// one line, and CMake's own `;` list separator does not survive the trip.
fn kernels_include_dirs() -> String {
    let mut dirs: Vec<String> = Vec::new();
    for key in ["CCCL", "FLASHINFER", "INCLUDE"] {
        let var = format!("DEP_PIE_KERNELS_CUDA_{key}");
        let value = std::env::var(&var).unwrap_or_else(|_| {
            panic!(
                "kernels-cuda's build.rs did not emit ${var} -- it publishes \
                 this out of its CMake's export block, and the `native` \
                 feature is what turns that block on"
            )
        });
        dirs.extend(value.split(':').filter(|s| !s.is_empty()).map(str::to_string));
    }
    dirs.join(";")
}

/// A sibling crate's directory. Used for assets no `links` handoff publishes:
/// `tensor-compiler`'s generated `ptir_abi.h` and its device runtime
/// templates, which the emitters own and the drivers only read.
fn sibling(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/driver-cuda has a parent")
        .join(name)
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/driver-cuda sits two levels below the repo root")
        .to_path_buf()
}

/// Dynamic-link CUDA toolkit `.so`s (`-lcudart -lcublas` etc.) from
/// `$CUDA_HOME/lib64`. We deliberately do NOT static-link: NVIDIA's static
/// archives ship multi-arch kernels (sm_70 through sm_120), `nvprune`-pruning
/// them only helps for `.a` files (the dynamic `.so`s aren't relocatable and
/// can't be pruned), and a 400+ MB static binary is worse user experience than
/// a small binary with a CUDA-toolkit runtime requirement.
///
/// Runtime contract: the host must have the CUDA toolkit installed such that
/// `libcudart.so.X` / `libcublas.so.X` are resolvable by the dynamic loader.
fn link_cuda_toolkit_dynamic(libs: &[&str]) {
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    let cuda_lib = Path::new(&cuda_home()).join("lib64");
    if !cuda_lib.is_dir() {
        panic!(
            "could not locate CUDA toolkit lib dir at {cuda_lib:?}. \
             Set $CUDA_HOME/$CUDA_PATH or install the CUDA toolkit."
        );
    }
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    for lib in libs {
        println!("cargo:rustc-link-lib={lib}");
    }
}

/// Emit a `-lcuda` link against the CUDA driver-API stub at
/// `$CUDA_HOME/lib64/stubs/libcuda.so`. At runtime `libcuda.so.1` comes from
/// the NVIDIA kernel driver install (not the toolkit) and is universally
/// present on any GPU host. Provides `cuMem*/cuCtx*` and friends used by
/// pie's custom all-reduce.
fn link_cuda_driver_stub() {
    let stubs = Path::new(&cuda_home()).join("lib64/stubs");
    if stubs.is_dir() {
        println!("cargo:rustc-link-search=native={}", stubs.display());
    }
    println!("cargo:rustc-link-lib=cuda");
}

fn cuda_home() -> String {
    std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string())
}

fn target_os() -> String {
    std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default()
}

fn add_system_libs() {
    for lib in ["stdc++", "pthread", "m", "dl", "rt"] {
        println!("cargo:rustc-link-lib={lib}");
    }
}

/// Walk `build_dir` looking for directories that contain at least one static
/// archive, and emit `cargo:rustc-link-search` for each.
fn add_link_search_paths(build_dir: &Path) {
    use std::collections::HashSet;
    let mut dirs: HashSet<PathBuf> = HashSet::new();
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
