//! Build the embedded native driver static libraries via CMake and link
//! them into the `pie` binary as a single deployable artifact.
//!
//! This script is the Cargo↔native *link bridge*: it invokes each driver's
//! CMake build, forwards the two include-dir handoffs, and emits the
//! `cargo:rustc-link-*` directives for the final binary (which rustc, not
//! CMake, links). Build-time *discovery* — nvcc location, CUDA arch,
//! sccache/ccache, NCCL, Marlin toggles, the CPM cache — lives in
//! `driver/*/CMakeLists.txt`, the native build system's proper home for it.
//!
//! Selection for native C++ drivers is via Cargo features. The Rust
//! dummy driver is always linked; the resulting binary dispatches at
//! runtime via `[model].driver.type` in the config TOML. The drivers
//! expose distinctly-named C entry points
//! (`pie_driver_{cuda,metal,dummy}_run` / `_request_stop`) so their
//! static archives can coexist in one binary without symbol collisions.
//!
//!   - `driver-cuda`: native CUDA driver with flashinfer kernels.
//!     Dynamic-links the CUDA toolkit `.so`s from `$CUDA_HOME`
//!     (or `/usr/local/cuda`); Linux-only.
//!   - dummy: Rust staticlib — random tokens, no model load; always linked.

use std::path::{Path, PathBuf};

fn main() {
    let cuda = cfg!(feature = "driver-cuda");
    let metal = cfg!(feature = "driver-metal");
    if cuda {
        println!("cargo:rerun-if-changed=../driver/cuda/CMakeLists.txt");
        println!("cargo:rerun-if-changed=../driver/cuda/cmake");
        println!("cargo:rerun-if-changed=../driver/cuda/include");
        println!("cargo:rerun-if-changed=../driver/cuda/src");
    }
    if metal {
        println!("cargo:rerun-if-changed=../driver/metal/CMakeLists.txt");
        println!("cargo:rerun-if-changed=../driver/metal/cmake");
        println!("cargo:rerun-if-changed=../driver/metal/src");
    }

    if cuda {
        build_cuda();
    }
    if metal {
        build_metal();
    }
    build_dummy();
}

/// Read `DEP_PIE_DRIVER_ABI_INCLUDE` — the directory where `pie-driver-abi`'s
/// build.rs publishes the driver-ABI C header (`pie_driver_abi.h` + the
/// `pie_driver_abi/` view/response_builder helpers). Pass-through to CMake via
/// `-DPIE_SCHEMA_INCLUDE_DIR=...` — the CMake var name is deliberately kept (it
/// is the contract with `driver/*/CMakeLists.txt`'s `if(NOT PIE_SCHEMA_INCLUDE_DIR)`
/// fallback; renaming it would have to happen on both sides at once for zero gain)
/// — so each C++ driver backend can pick the headers up with
/// `target_include_directories`.
fn pie_driver_abi_include_dir() -> PathBuf {
    let dir = std::env::var("DEP_PIE_DRIVER_ABI_INCLUDE").expect(
        "pie-driver-abi's build.rs did not emit cargo:include — \
                 check that `links = \"pie_driver_abi\"` is set in interface/driver/Cargo.toml",
    );
    PathBuf::from(dir)
}

/// Read `DEP_PIE_IPC_INCLUDE` — the directory where `pie-ipc`'s build.rs
/// publishes the in-proc mechanism C headers (`pie_ipc.h` vtable +
/// `pie_ipc/inproc_server.hpp`). Pass-through to CMake via
/// `-DPIE_IPC_INCLUDE_DIR=...`. These headers `#include <pie_driver_abi.h>`,
/// so both include dirs are added to the C++ targets.
fn pie_ipc_include_dir() -> PathBuf {
    let dir = std::env::var("DEP_PIE_IPC_INCLUDE").expect(
        "pie-ipc's build.rs did not emit cargo:include — \
                 check that `links = \"pie_ipc\"` is set in driver/ipc/Cargo.toml",
    );
    PathBuf::from(dir)
}

// -----------------------------------------------------------------------------
// driver/metal — MLX-free raw-Metal driver (Apple Silicon, macOS-only)
// -----------------------------------------------------------------------------

fn build_metal() {
    let target_os = target_os();
    if target_os != "macos" {
        panic!(
            "--features driver-metal is macOS-only (got target_os={target_os:?}). \
             On Linux, use `--features driver-cuda`; the metal flavor targets \
             Apple Silicon via MLX + native Metal shaders."
        );
    }

    let driver_dir = PathBuf::from("../driver/metal");
    let mut cfg = driver_cmake_config(&driver_dir, "metal", "pie_driver_metal_lib");

    // CPM cache is read by the CMakeLists via `$ENV{CPM_SOURCE_CACHE}`;
    // declare the dep so a change reconfigures.
    println!("cargo:rerun-if-env-changed=CPM_SOURCE_CACHE");

    // MLX is OFF by default (the raw-Metal driver is MLX-free); opt in with
    // PIE_METAL_WITH_MLX=1 for the legacy MLX executor path. The flag also
    // gates the link below, so read it here.
    let mlx_on = env_is_truthy("PIE_METAL_WITH_MLX");
    cfg.define("PIE_METAL_WITH_MLX", if mlx_on { "ON" } else { "OFF" });

    // MLX provider: "fetch" (FetchContent from source, default) or "system"
    // (a prebuilt MLX via find_package(MLX), e.g. `brew install mlx`).
    println!("cargo:rerun-if-env-changed=PIE_METAL_MLX_PROVIDER");
    if let Ok(provider) = std::env::var("PIE_METAL_MLX_PROVIDER") {
        let provider = provider.to_ascii_lowercase();
        if !matches!(provider.as_str(), "fetch" | "system") {
            panic!("PIE_METAL_MLX_PROVIDER must be \"fetch\" or \"system\" (got {provider:?})");
        }
        cfg.define("PIE_METAL_MLX_PROVIDER", provider);
    }

    // Source-fetch only: build MLX's Metal GPU backend (needs `xcrun metal`).
    // Only forwarded when explicitly set (otherwise the CMake default holds).
    let build_metal_gpu = env_is_truthy("PIE_METAL_MLX_BUILD_METAL");
    if std::env::var_os("PIE_METAL_MLX_BUILD_METAL").is_some() {
        cfg.define("PIE_METAL_MLX_BUILD_METAL", if build_metal_gpu { "ON" } else { "OFF" });
    }

    let build_dir = cfg.build().join("build");
    add_link_search_paths(&build_dir);

    // --- link directives for the final rustc binary (CMake can't emit these) ---
    println!("cargo:rustc-link-lib=static=pie_driver_metal_lib");
    if mlx_on {
        link_mlx();
    }
    // Apple frameworks the metal driver pulls. -framework is macOS's -l.
    println!("cargo:rustc-link-lib=framework=Accelerate");
    add_system_libs(/*metal=*/ true);

    println!(
        "cargo:rustc-env=PIE_DRIVER_METAL_BUILD_DIR={}",
        build_dir.display()
    );
    rerun_if_changed(&driver_dir);
}

// -----------------------------------------------------------------------------
// driver/dummy — Rust staticlib, no cmake step
// -----------------------------------------------------------------------------

fn build_dummy() {
    // pie-driver-dummy is a Cargo dep with crate-type = ["staticlib", "rlib"].
    // Cargo links the rlib automatically through the dependency; nothing
    // to build here. The rlib path is enough — the C ABI symbols
    // (`pie_driver_dummy_run` / `_request_stop`) come along.
}

// -----------------------------------------------------------------------------
// driver/cuda
// -----------------------------------------------------------------------------

fn build_cuda() {
    let target_os = target_os();
    if target_os != "linux" {
        panic!(
            "--features driver-cuda is Linux-only (got target_os={target_os:?}). \
             On macOS, use `--features driver-metal`; \
             on Windows, the cuda flavor is not supported."
        );
    }

    let driver_dir = PathBuf::from("../driver/cuda");
    let mut cfg = driver_cmake_config(&driver_dir, "cuda", "pie_driver_cuda_lib");

    // nvcc discovery, CUDA arch, the sccache/ccache launcher, the Marlin
    // toggles and the CPM source cache are all handled by driver/cuda's
    // CMakeLists (via find_program / `$ENV{...}`); we only declare the env
    // deps here so Cargo reconfigures when they change.
    for var in [
        "CUDACXX",
        "CMAKE_CUDA_COMPILER",
        "CMAKE_CUDA_ARCHITECTURES",
        "PIE_COMPILER_LAUNCHER",
        "PIE_CUDA_BUILD_MARLIN",
        "PIE_MARLIN_ALL_SHAPES",
        "CPM_SOURCE_CACHE",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    let build_dir = cfg.build().join("build");
    add_link_search_paths(&build_dir);

    // --- link directives for the final rustc binary (CMake can't emit these) ---
    println!("cargo:rustc-link-lib=static=pie_driver_cuda_lib");

    // CUDA toolkit: dynamic cudart + cublas + cublasLt + nvrtc (gemm.cpp
    // references cublasLt directly; the Sampling-IR JIT calls the NVRTC
    // runtime-compilation API), plus the driver-API stub (`-lcuda`). Runtime
    // contract: the host ships the CUDA toolkit `.so`s.
    link_cuda_toolkit_dynamic(&["cudart", "cublas", "cublasLt", "nvrtc"]);
    link_cuda_driver_stub();
    link_nccl();

    add_system_libs(/*metal=*/ false);

    println!(
        "cargo:rustc-env=PIE_DRIVER_CUDA_BUILD_DIR={}",
        build_dir.display()
    );
    rerun_if_changed(&driver_dir);
}

// -----------------------------------------------------------------------------
// Shared helpers
// -----------------------------------------------------------------------------

/// Shared `cmake::Config` for a native driver flavor. Per-flavor `out_dir`
/// keeps multi-driver builds from clobbering each other's CMake cache; the
/// archives are static + PIC (downstream may relink them into the pyo3
/// `pie-server` shared object), and the two include-dir handoffs published
/// by `pie-driver-abi` / `pie-ipc`'s build scripts are forwarded so the C++
/// targets can find the ABI + in-proc IPC headers.
fn driver_cmake_config(driver_dir: &Path, out_subdir: &str, build_target: &str) -> cmake::Config {
    let mut cfg = cmake::Config::new(driver_dir);
    cfg.out_dir(PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join(out_subdir));
    cfg.build_target(build_target)
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .define("PIE_SCHEMA_INCLUDE_DIR", pie_driver_abi_include_dir())
        .define("PIE_IPC_INCLUDE_DIR", pie_ipc_include_dir());
    cfg
}

/// True when `env` is set to a truthy value (`1`/`on`/`true`/`yes`). Also
/// registers a Cargo `rerun-if-env-changed` for it.
fn env_is_truthy(env: &str) -> bool {
    println!("cargo:rerun-if-env-changed={env}");
    std::env::var(env)
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "on" | "true" | "yes"))
        .unwrap_or(false)
}

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
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    let target_os = target_os();
    let cuda_home = cuda_home();
    let cuda_lib = if target_os == "windows" {
        Path::new(&cuda_home).join("lib").join("x64")
    } else {
        Path::new(&cuda_home).join("lib64")
    };
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
/// `$CUDA_HOME/lib64/stubs/libcuda.so`. At runtime `libcuda.so.1`
/// comes from the NVIDIA kernel driver install (not the toolkit) and
/// is universally present on any GPU host. Provides `cuMem*/cuCtx*`
/// and friends used by both ggml-cuda and pie's custom-all-reduce.
fn link_cuda_driver_stub() {
    let target_os = target_os();
    let cuda_home = cuda_home();
    if target_os == "windows" {
        let lib = Path::new(&cuda_home).join("lib").join("x64");
        if lib.is_dir() {
            println!("cargo:rustc-link-search=native={}", lib.display());
        }
    } else {
        let stubs = Path::new(&cuda_home).join("lib64/stubs");
        if stubs.is_dir() {
            println!("cargo:rustc-link-search=native={}", stubs.display());
        }
    }
    println!("cargo:rustc-link-lib=cuda");
}

/// Emit the NCCL link directive for the final binary. Only system NCCL is
/// supported: CMake's `find_library(nccl)` locates the header + library at
/// configure time, and rustc resolves `-lnccl` at link time.
fn link_nccl() {
    println!("cargo:rustc-link-lib=nccl");
}

/// Emit MLX link directives (opt-in legacy path). The `system` provider
/// dylib-links a brew/prefix MLX (+ rpath); `fetch` static-links the
/// FetchContent build.
fn link_mlx() {
    let provider = std::env::var("PIE_METAL_MLX_PROVIDER")
        .map(|p| p.to_ascii_lowercase())
        .unwrap_or_else(|_| "fetch".to_string());
    if provider == "system" {
        let prefix = std::env::var("PIE_MLX_PREFIX")
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(brew_mlx_prefix)
            .unwrap_or_else(|| "/opt/homebrew/opt/mlx".to_string());
        let libdir = format!("{prefix}/lib");
        println!("cargo:rustc-link-search=native={libdir}");
        println!("cargo:rustc-link-lib=dylib=mlx");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
    } else {
        println!("cargo:rustc-link-lib=static=mlx");
    }
    println!("cargo:rustc-link-lib=framework=QuartzCore");
}

/// `brew --prefix mlx`, if brew is present and MLX is installed.
fn brew_mlx_prefix() -> Option<String> {
    std::process::Command::new("brew")
        .args(["--prefix", "mlx"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

fn cuda_home() -> String {
    std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string())
}

fn target_os() -> String {
    if let Ok(os) = std::env::var("CARGO_CFG_TARGET_OS") {
        if !os.is_empty() {
            return os;
        }
    }
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("windows") {
        return "windows".to_string();
    }
    if target.contains("apple-darwin") {
        return "macos".to_string();
    }
    if target.contains("linux") {
        return "linux".to_string();
    }
    if cfg!(windows) {
        "windows".to_string()
    } else if cfg!(target_os = "macos") {
        "macos".to_string()
    } else if cfg!(target_os = "linux") {
        "linux".to_string()
    } else {
        target
    }
}

/// Emit per-OS system library link directives. On macOS, `metal=true`
/// also links the Metal/MetalKit/Foundation frameworks ggml-metal needs.
fn add_system_libs(metal: bool) {
    let target_os = target_os();
    match target_os.as_str() {
        "linux" => {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=pthread");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=dl");
            println!("cargo:rustc-link-lib=rt");
        }
        "macos" => {
            println!("cargo:rustc-link-lib=c++");
            if metal {
                // ggml-metal pulls these three frameworks. -framework on
                // macOS is the moral equivalent of -l on linux.
                println!("cargo:rustc-link-lib=framework=Metal");
                println!("cargo:rustc-link-lib=framework=MetalKit");
                println!("cargo:rustc-link-lib=framework=Foundation");
            }
        }
        "windows" => {}
        other => {
            panic!("pie-worker: unsupported target OS {other:?}");
        }
    }
}

fn rerun_if_changed(driver_dir: &Path) {
    println!(
        "cargo:rerun-if-changed={}",
        driver_dir.join("CMakeLists.txt").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        driver_dir.join("src").display()
    );
}

/// Walk `build_dir` looking for directories that contain at least one
/// static archive (`.a` on Unix, `.lib` on Windows), and emit
/// `cargo:rustc-link-search` for each.
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
        } else if path.extension().is_some_and(|e| e == "a" || e == "lib") {
            has_archive = true;
        }
    }
    if has_archive {
        out.insert(dir.to_path_buf());
    }
}
