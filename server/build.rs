//! Build the embedded native driver static libraries via CMake and link
//! them into the `pie` binary as a single deployable artifact.
//!
//! Selection is via Cargo features. Any non-empty subset is allowed —
//! the resulting binary contains every enabled driver and dispatches
//! at runtime via `[[model]].driver.type` in the config TOML. The
//! drivers expose distinctly-named C entry points
//! (`pie_driver_{portable,cuda,dummy}_run` / `_request_stop`) so their
//! static archives can coexist in one binary without symbol collisions.
//!
//!   - `driver-portable` (default): ggml-backed CPU/CUDA/Metal/Vulkan
//!     driver. Static libs come from llama.cpp's CPM-vendored ggml.
//!   - `driver-cuda`: native CUDA driver with flashinfer kernels.
//!     Statically links the per-toolkit CUDA libs from `$CUDA_HOME`
//!     (or `/usr/local/cuda`) so the resulting binary has zero CUDA
//!     shared deps. Verified by the M3 spike (see
//!     `driver/cuda/CMakeLists.txt`).
//!   - `driver-dummy`: Rust staticlib — random tokens, no model load.

use std::path::{Path, PathBuf};

fn main() {
    let portable = cfg!(feature = "driver-portable");
    let cuda = cfg!(feature = "driver-cuda");
    let dummy = cfg!(feature = "driver-dummy");

    if !(portable || cuda || dummy) {
        panic!(
            "no driver feature enabled — build with one or more of \
             `--features driver-portable`, `--features driver-cuda`, \
             `--features driver-dummy`"
        );
    }

    if portable {
        build_portable();
    }
    if cuda {
        build_cuda();
    }
    if dummy {
        build_dummy();
    }
}

// -----------------------------------------------------------------------------
// driver/dummy — Rust staticlib, no cmake step
// -----------------------------------------------------------------------------

fn build_dummy() {
    // pie-driver-dummy is a Cargo dep with crate-type = ["staticlib", "rlib"].
    // Cargo links the rlib automatically through the optional dependency;
    // nothing to build here. The rlib path is enough — the C ABI symbols
    // (`pie_driver_dummy_run` / `_request_stop`) come along.
}

// -----------------------------------------------------------------------------
// driver/portable
// -----------------------------------------------------------------------------

fn build_portable() {
    let driver_dir = PathBuf::from("../driver/portable");
    let cuda_enabled = std::env::var("PIE_PORTABLE_CUDA").is_ok();
    let vulkan_enabled = std::env::var("PIE_PORTABLE_VULKAN").is_ok();
    let metal_enabled = std::env::var("PIE_PORTABLE_METAL").is_ok();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    println!("cargo:rerun-if-env-changed=PIE_PORTABLE_CUDA");
    println!("cargo:rerun-if-env-changed=PIE_PORTABLE_VULKAN");
    println!("cargo:rerun-if-env-changed=PIE_PORTABLE_METAL");

    if metal_enabled && target_os != "macos" {
        panic!("PIE_PORTABLE_METAL is only valid on macOS (got target_os={target_os:?})");
    }
    if cuda_enabled && target_os == "macos" {
        panic!(
            "PIE_PORTABLE_CUDA is not supported on macOS — use \
             PIE_PORTABLE_METAL instead, or build the cuda flavor on Linux."
        );
    }

    let mut cfg = cmake::Config::new(&driver_dir);
    // Per-flavor `out_dir` so multi-driver builds (e.g.
    // --features driver-portable,driver-cuda) don't clobber each
    // other's CMake cache. Without this, the second cmake::Config
    // invocation overwrites the first's `build/` and reconfigures
    // it for the wrong project.
    cfg.out_dir(std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap())
        .join("portable"));
    cfg.build_target("pie_driver_portable_lib")
        .define("BUILD_SHARED_LIBS", "OFF");
    if cuda_enabled {
        cfg.define("GGML_CUDA", "ON")
            // ggml's CUDA-graph capture/replay is OFF by default at the ggml-
            // subproject level; llama.cpp's parent CMakeLists.txt flips it
            // ON. Embedding ggml directly (as we do here) misses that override
            // and ends up issuing ~30 kernel launches per decode step, which
            // costs ~1 ms vs ~0.3 ms with capture. Enable explicitly.
            .define("GGML_CUDA_GRAPHS", "ON")
            .define("GGML_STATIC", "ON");
        if let Ok(arch) = std::env::var("PIE_PORTABLE_CUDA_ARCH") {
            cfg.define("CMAKE_CUDA_ARCHITECTURES", arch);
        }
    }
    if vulkan_enabled {
        cfg.define("GGML_VULKAN", "ON")
            .define("GGML_STATIC", "ON");
    }
    if metal_enabled {
        // ggml-metal links against Apple's MetalKit / Foundation. The
        // C++ side handles those via xcrun; we just need to flip the
        // ggml flag and add the framework links below.
        cfg.define("GGML_METAL", "ON")
            .define("GGML_STATIC", "ON");
    }
    // Apple clang (`/usr/bin/cc` on macOS) doesn't ship with OpenMP
    // and the ggml-cpu CMake hard-errors when `find_package(OpenMP)`
    // misses. Metal is the relevant backend on this OS anyway, so
    // turn ggml's OpenMP requirement off rather than dragging in
    // libomp from Homebrew. Linux + Windows have OpenMP via gcc/MSVC
    // out of the box, so leave the default on there.
    if target_os == "macos" {
        cfg.define("GGML_OPENMP", "OFF");
    }
    let dst = cfg.build();
    let build_dir = dst.join("build");

    add_link_search_paths(&build_dir);

    println!("cargo:rustc-link-lib=static=pie_driver_portable_lib");
    println!("cargo:rustc-link-arg=-Wl,--start-group");
    println!("cargo:rustc-link-arg=-l:libggml.a");
    println!("cargo:rustc-link-arg=-l:libggml-cpu.a");
    if cuda_enabled {
        println!("cargo:rustc-link-arg=-l:libggml-cuda.a");
    }
    if vulkan_enabled {
        println!("cargo:rustc-link-arg=-l:libggml-vulkan.a");
    }
    println!("cargo:rustc-link-arg=-l:libggml-base.a");
    println!("cargo:rustc-link-arg=-Wl,--end-group");

    if cuda_enabled {
        // ggml-cuda calls `cublasGemmEx` (libcublas), nothing in
        // libcublasLt directly. Dynamic-link cudart + cublas; let
        // libcublas.so resolve its own libcublasLt.so transitively
        // at load time. Runtime contract: CUDA toolkit installed on
        // the host (which always ships libcublasLt next to libcublas).
        link_cuda_toolkit_dynamic(&["cudart", "cublas"]);
        link_cuda_driver_stub();
    }
    if vulkan_enabled {
        // libvulkan.so is the standard system name on Linux. If
        // VULKAN_SDK is set, prefer its lib dir (so a project-vendored
        // SDK works without sudo apt install libvulkan-dev).
        if let Ok(sdk) = std::env::var("VULKAN_SDK") {
            let sdk_lib = Path::new(&sdk).join("lib");
            if sdk_lib.is_dir() {
                println!("cargo:rustc-link-search=native={}", sdk_lib.display());
            }
        }
        println!("cargo:rustc-link-lib=vulkan");
    }

    add_system_libs(&["gomp"], metal_enabled);

    println!(
        "cargo:rustc-env=PIE_DRIVER_PORTABLE_BUILD_DIR={}",
        build_dir.display()
    );
    rerun_if_changed(&driver_dir);
}

// -----------------------------------------------------------------------------
// driver/cuda
// -----------------------------------------------------------------------------

fn build_cuda() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "linux" {
        panic!(
            "--features driver-cuda is Linux-only (got target_os={target_os:?}). \
             On macOS, use `--features driver-portable` with PIE_PORTABLE_METAL=1; \
             on Windows, the cuda flavor is not supported."
        );
    }

    let driver_dir = PathBuf::from("../driver/cuda");

    let mut cfg = cmake::Config::new(&driver_dir);
    // See `build_portable` — per-flavor out_dir keeps the two CMake
    // configurations from clobbering each other in multi-driver builds.
    cfg.out_dir(std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap())
        .join("cuda"));
    cfg.build_target("pie_driver_cuda_lib")
        .define("BUILD_SHARED_LIBS", "OFF");

    // NCCL discovery hint. The cuda driver's CMakeLists.txt does
    // `find_path(NCCL_INCLUDE_DIR nccl.h ...)` against `/usr/include`
    // and `/usr/local/include` by default. Sites that install NCCL
    // sideways (e.g. the `nvidia-nccl-cu*` Python wheel under
    // `site-packages/nvidia/nccl/`) need an escape hatch — set
    // `PIE_NCCL_HOME` to its root so this build picks it up too.
    println!("cargo:rerun-if-env-changed=PIE_NCCL_HOME");
    if let Ok(nccl_home) = std::env::var("PIE_NCCL_HOME") {
        let inc = Path::new(&nccl_home).join("include");
        let lib = Path::new(&nccl_home).join("lib");
        if !inc.is_dir() || !lib.is_dir() {
            panic!(
                "PIE_NCCL_HOME={nccl_home:?} must contain include/ and lib/ \
                 subdirectories (got include={inc:?}, lib={lib:?})"
            );
        }
        cfg.define("NCCL_INCLUDE_DIR", inc.display().to_string());
        // Pin the exact library path so find_library doesn't trip over
        // a versioned-only `libnccl.so.2` next to a missing `libnccl.so`.
        let candidates = ["libnccl.so", "libnccl.so.2"];
        let nccl_lib = candidates
            .iter()
            .map(|name| lib.join(name))
            .find(|p| p.is_file())
            .unwrap_or_else(|| {
                panic!(
                    "PIE_NCCL_HOME={nccl_home:?}: no libnccl.so / libnccl.so.2 \
                     under {lib:?}"
                )
            });
        cfg.define("NCCL_LIBRARY", nccl_lib.display().to_string());
    }

    let dst = cfg.build();
    let build_dir = dst.join("build");

    add_link_search_paths(&build_dir);

    // Driver lib + the static deps CMake/CPM produced for us under
    // build_dir (most importantly zstd, plus any tomlplusplus / nlohmann
    // archives — those two are header-only at the time of writing but
    // we still walk the tree in case that changes).
    println!("cargo:rustc-link-lib=static=pie_driver_cuda_lib");
    println!("cargo:rustc-link-arg=-Wl,--start-group");
    println!("cargo:rustc-link-arg=-l:libpie_driver_cuda_lib.a");
    println!("cargo:rustc-link-arg=-l:libzstd.a");
    println!("cargo:rustc-link-arg=-Wl,--end-group");

    // CUDA toolkit: dynamic-link cudart + cublas + cublasLt.
    // The cuda driver's `src/ops/gemm.cpp` directly references
    // `cublasLt*` symbols (the native FP8 W8A16 path on sm_89+),
    // so we must satisfy them at link time. Runtime contract: the
    // host has CUDA toolkit installed; all three .so files ship
    // together with the toolkit.
    link_cuda_toolkit_dynamic(&["cudart", "cublas", "cublasLt"]);
    link_cuda_driver_stub();

    let cuda_home = std::env::var("CUDA_HOME")
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    // NCCL: dynamic-linked. Two install shapes in the wild:
    //   * System `libnccl-dev`: `libnccl.so` -> `libnccl.so.2.X` symlink,
    //     so `-lnccl` resolves cleanly.
    //   * `nvidia-nccl-cu*` Python wheel: ships only `libnccl.so.2`
    //     (versioned), no unversioned symlink. rust-lld is strict and
    //     refuses `-lnccl` against that, so we link the exact file
    //     when `PIE_NCCL_HOME` points at a wheel install.
    if let Ok(nccl_home) = std::env::var("PIE_NCCL_HOME") {
        let nccl_lib_dir = Path::new(&nccl_home).join("lib");
        println!("cargo:rustc-link-search=native={}", nccl_lib_dir.display());
        // Prefer the unversioned name if it exists (some wheels do
        // ship the symlink); otherwise fall back to libnccl.so.2.
        let unversioned = nccl_lib_dir.join("libnccl.so");
        if unversioned.is_file() {
            println!("cargo:rustc-link-lib=nccl");
        } else {
            println!("cargo:rustc-link-arg=-l:libnccl.so.2");
        }
        // Embed the wheel's lib dir as an rpath so the binary loads
        // libnccl.so.2 without requiring `LD_LIBRARY_PATH` at runtime.
        // System-NCCL builds skip this — the loader finds the .so via
        // the standard ld.so search path.
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}",
            nccl_lib_dir.display()
        );
    } else {
        println!("cargo:rustc-link-lib=nccl");
    }

    add_system_libs(&[], /*metal=*/ false);

    println!(
        "cargo:rustc-env=PIE_DRIVER_CUDA_BUILD_DIR={}",
        build_dir.display()
    );
    rerun_if_changed(&driver_dir);
}

// -----------------------------------------------------------------------------
// Shared helpers
// -----------------------------------------------------------------------------

/// Dynamic-link CUDA toolkit `.so`s (`-lcudart -lcublas` etc.) from
/// `$CUDA_HOME/lib64`. We deliberately do NOT static-link: NVIDIA's
/// static archives ship multi-arch kernels (sm_70 through sm_120),
/// `nvprune`-pruning them only helps for `.a` files (the dynamic
/// `.so`s aren't relocatable and can't be pruned), and a 400+ MB
/// static binary is worse user experience than a small binary with
/// a CUDA-toolkit runtime requirement.
///
/// Runtime contract: the host must have the CUDA toolkit installed
/// such that `libcudart.so.X` / `libcublas.so.X` are resolvable by
/// the dynamic loader. `libcublasLt.so.X` is pulled transitively
/// by libcublas (we don't reference it directly).
fn link_cuda_toolkit_dynamic(libs: &[&str]) {
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    let cuda_home = std::env::var("CUDA_HOME")
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let cuda_lib = Path::new(&cuda_home).join("lib64");
    if !cuda_lib.is_dir() {
        panic!(
            "could not locate CUDA toolkit lib dir at {cuda_lib:?}. \
             Set $CUDA_HOME or install CUDA toolkit at /usr/local/cuda."
        );
    }
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    for lib in libs {
        println!("cargo:rustc-link-lib={lib}");
    }
}

/// Emit a `-lcuda` link against the CUDA driver-API stub at
/// `$CUDA_HOME/lib64/stubs/libcuda.so`. At runtime `libcuda.so.1`
/// comes from the NVIDIA kernel driver install (not the toolkit) and
/// is universally present on any GPU host. Provides `cuMem*/cuCtx*`
/// and friends used by both ggml-cuda and pie's custom-all-reduce.
fn link_cuda_driver_stub() {
    let cuda_home = std::env::var("CUDA_HOME")
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let stubs = Path::new(&cuda_home).join("lib64/stubs");
    if stubs.is_dir() {
        println!("cargo:rustc-link-search=native={}", stubs.display());
    }
    println!("cargo:rustc-link-lib=cuda");
}

/// Emit per-OS system library link directives. `extra` is appended on
/// linux only (typically `gomp` for OpenMP). On macOS, `metal=true`
/// links the Metal/MetalKit/Foundation frameworks ggml-metal needs +
/// resolves libomp from a typical brew location (override via
/// `OPENMP_DIR`).
fn add_system_libs(extra: &[&str], metal: bool) {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "linux" => {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=pthread");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=dl");
            println!("cargo:rustc-link-lib=rt");
            for lib in extra {
                println!("cargo:rustc-link-lib={lib}");
            }
        }
        "macos" => {
            println!("cargo:rustc-link-lib=c++");

            // OpenMP via Homebrew's `libomp` (Apple Silicon: `/opt/homebrew`,
            // Intel: `/usr/local`). Override via `OPENMP_DIR` for a custom
            // install. We always link omp on macOS since ggml's CPU backend
            // is OpenMP-enabled by default; missing it is a confusing
            // link-time error.
            println!("cargo:rerun-if-env-changed=OPENMP_DIR");
            let openmp_dir = std::env::var("OPENMP_DIR").ok().map(PathBuf::from);
            let candidates: Vec<PathBuf> = match openmp_dir {
                Some(d) => vec![d.join("lib")],
                None => vec![
                    PathBuf::from("/opt/homebrew/opt/libomp/lib"),
                    PathBuf::from("/usr/local/opt/libomp/lib"),
                ],
            };
            for p in candidates.iter().filter(|p| p.is_dir()) {
                println!("cargo:rustc-link-search=native={}", p.display());
            }
            println!("cargo:rustc-link-lib=omp");

            if metal {
                // ggml-metal pulls these three frameworks. -framework on
                // macOS is the moral equivalent of -l on linux.
                println!("cargo:rustc-link-lib=framework=Metal");
                println!("cargo:rustc-link-lib=framework=MetalKit");
                println!("cargo:rustc-link-lib=framework=Foundation");
            }
        }
        other => {
            panic!("pie-server: unsupported target OS {other:?}");
        }
    }
}

fn rerun_if_changed(driver_dir: &Path) {
    println!("cargo:rerun-if-changed={}", driver_dir.join("CMakeLists.txt").display());
    println!("cargo:rerun-if-changed={}", driver_dir.join("src").display());
}

/// Walk `build_dir` looking for directories that contain at least one
/// `.a` file, and emit `cargo:rustc-link-search` for each.
fn add_link_search_paths(build_dir: &Path) {
    use std::collections::HashSet;
    let mut dirs: HashSet<PathBuf> = HashSet::new();
    walk(build_dir, &mut dirs);
    for d in &dirs {
        println!("cargo:rustc-link-search=native={}", d.display());
    }
}

fn walk(dir: &Path, out: &mut std::collections::HashSet<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
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
