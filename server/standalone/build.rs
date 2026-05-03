//! Build the embedded native driver static library via CMake and link
//! it into the `pie` binary as a single deployable artifact.
//!
//! Selection is via Cargo features (mutually exclusive):
//!   - `driver-portable` (default): ggml-backed CPU/CUDA/Metal/Vulkan
//!     driver. Static libs come from llama.cpp's CPM-vendored ggml.
//!   - `driver-cuda`: native CUDA driver with flashinfer kernels.
//!     Statically links the per-toolkit CUDA libs from `$CUDA_HOME`
//!     (or `/usr/local/cuda`) so the resulting binary has zero CUDA
//!     shared deps. Verified by the M3 spike (see
//!     `driver/cuda/CMakeLists.txt`).

use std::path::{Path, PathBuf};

fn main() {
    let portable = cfg!(feature = "driver-portable");
    let cuda = cfg!(feature = "driver-cuda");
    let dummy = cfg!(feature = "driver-dummy");

    let n = [portable, cuda, dummy].iter().filter(|x| **x).count();
    if n > 1 {
        panic!(
            "driver-* features are mutually exclusive (got portable={portable}, \
             cuda={cuda}, dummy={dummy}) — pick exactly one"
        );
    }
    if n == 0 {
        panic!(
            "no driver feature enabled — build with `--features driver-portable`, \
             `--features driver-cuda`, or `--features driver-dummy`"
        );
    }

    if portable {
        build_portable();
    } else if cuda {
        build_cuda();
    } else {
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
    //
    // We deliberately don't set `PIE_DRIVER_BUILD_DIR` — `main.rs` reads
    // it via `option_env!` and falls back to `<in-process>`.
    println!("cargo:rustc-env=PIE_DRIVER_FLAVOR=dummy");
}

// -----------------------------------------------------------------------------
// driver/portable
// -----------------------------------------------------------------------------

fn build_portable() {
    let driver_dir = PathBuf::from("../../driver/portable");
    let cuda_enabled = std::env::var("PIE_PORTABLE_CUDA").is_ok();
    let vulkan_enabled = std::env::var("PIE_PORTABLE_VULKAN").is_ok();
    println!("cargo:rerun-if-env-changed=PIE_PORTABLE_CUDA");
    println!("cargo:rerun-if-env-changed=PIE_PORTABLE_VULKAN");

    let mut cfg = cmake::Config::new(&driver_dir);
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
        let cuda_home = std::env::var("CUDA_HOME")
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        let cuda_lib = Path::new(&cuda_home).join("lib64");
        if !cuda_lib.is_dir() {
            panic!(
                "PIE_PORTABLE_CUDA=1 but CUDA toolkit not found at {cuda_lib:?}. \
                 Set $CUDA_HOME."
            );
        }
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
        // Driver-API stub (libcuda.so) — provides cuMem*/cuCtx*/cuLaunch*
        // symbols. Linked dynamically; libcuda.so ships with the NVIDIA
        // driver, not the toolkit.
        let stubs = Path::new(&cuda_home).join("lib64/stubs");
        if stubs.is_dir() {
            println!("cargo:rustc-link-search=native={}", stubs.display());
        }
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-arg=-Wl,--start-group");
        println!("cargo:rustc-link-arg=-l:libcublas_static.a");
        println!("cargo:rustc-link-arg=-l:libcublasLt_static.a");
        println!("cargo:rustc-link-arg=-l:libcudart_static.a");
        println!("cargo:rustc-link-arg=-l:libculibos.a");
        println!("cargo:rustc-link-arg=-Wl,--end-group");
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

    add_system_libs_linux(&[/* extra: */ "gomp"]);

    println!("cargo:rustc-env=PIE_DRIVER_BUILD_DIR={}", build_dir.display());
    let flavor_str = if cuda_enabled { "portable+cuda" }
        else if vulkan_enabled { "portable+vulkan" }
        else { "portable" };
    println!("cargo:rustc-env=PIE_DRIVER_FLAVOR={}", flavor_str);
    rerun_if_changed(&driver_dir);
}

// -----------------------------------------------------------------------------
// driver/cuda
// -----------------------------------------------------------------------------

fn build_cuda() {
    let driver_dir = PathBuf::from("../../driver/cuda");
    let dst = cmake::Config::new(&driver_dir)
        .build_target("pie_driver_cuda_lib")
        .define("BUILD_SHARED_LIBS", "OFF")
        .build();
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

    // CUDA toolkit static libs. Resolved via $CUDA_HOME or
    // /usr/local/cuda. The M3 spike verified `cublas_static +
    // cublasLt_static + cudart_static + culibos` link cleanly and
    // produce a binary with no CUDA shared deps.
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
    println!("cargo:rustc-link-arg=-Wl,--start-group");
    println!("cargo:rustc-link-arg=-l:libcublas_static.a");
    println!("cargo:rustc-link-arg=-l:libcublasLt_static.a");
    println!("cargo:rustc-link-arg=-l:libcudart_static.a");
    println!("cargo:rustc-link-arg=-l:libculibos.a");
    println!("cargo:rustc-link-arg=-Wl,--end-group");

    add_system_libs_linux(&[]);

    println!("cargo:rustc-env=PIE_DRIVER_BUILD_DIR={}", build_dir.display());
    println!("cargo:rustc-env=PIE_DRIVER_FLAVOR=cuda");
    rerun_if_changed(&driver_dir);
}

// -----------------------------------------------------------------------------
// Shared helpers
// -----------------------------------------------------------------------------

fn add_system_libs_linux(extra: &[&str]) {
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
            // OpenMP/CUDA on macOS isn't supported here yet.
        }
        other => {
            panic!("pie-standalone: unsupported target OS {other:?}");
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
