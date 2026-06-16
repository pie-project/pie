//! Build the embedded native driver static libraries via CMake and link
//! them into the `pie` binary as a single deployable artifact.
//!
//! Selection for native C++ drivers is via Cargo features. The Rust
//! dummy driver is always linked; the resulting binary dispatches at
//! runtime via `[[model]].driver.type` in the config TOML. The drivers
//! expose distinctly-named C entry points
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
//!   - dummy: Rust staticlib — random tokens, no model load; always linked.

use std::path::{Path, PathBuf};

fn main() {
    let portable = cfg!(feature = "driver-portable");
    let cuda = cfg!(feature = "driver-cuda");
    println!("cargo:rerun-if-changed=../driver/common/include");
    println!("cargo:rerun-if-changed=../driver/common/src");
    if cuda {
        println!("cargo:rerun-if-changed=../driver/cuda/CMakeLists.txt");
        println!("cargo:rerun-if-changed=../driver/cuda/cmake");
        println!("cargo:rerun-if-changed=../driver/cuda/include");
        println!("cargo:rerun-if-changed=../driver/cuda/src");
    }

    if portable {
        build_portable();
    }
    if cuda {
        build_cuda();
    }
    build_dummy();
}

/// Read `DEP_PIE_BRIDGE_INCLUDE` — the directory where `pie-bridge`'s
/// build.rs writes the cbindgen-generated `pie_bridge.h`. Pass-through to
/// CMake via `-DPIE_BRIDGE_INCLUDE_DIR=...` so each C++ driver backend
/// can pick the header up with `target_include_directories`.
fn pie_bridge_include_dir() -> PathBuf {
    let dir = std::env::var("DEP_PIE_BRIDGE_INCLUDE").expect(
        "pie-bridge's build.rs did not emit cargo:include — \
                 check that `links = \"pie_bridge\"` is set in driver/bridge/Cargo.toml",
    );
    PathBuf::from(dir)
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
// driver/portable
// -----------------------------------------------------------------------------

fn build_portable() {
    let driver_dir = PathBuf::from("../driver/portable");
    let cuda_enabled = std::env::var("PIE_PORTABLE_CUDA").is_ok();
    let vulkan_enabled = std::env::var("PIE_PORTABLE_VULKAN").is_ok();
    let metal_enabled = std::env::var("PIE_PORTABLE_METAL").is_ok();
    let target_os = target_os();
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
    let portable_out_dir = if target_os == "windows" {
        // ggml-vulkan creates a nested shader-generator CMake project.
        // Keeping this out of Cargo's hashed OUT_DIR avoids MSBuild's legacy
        // 260-character path limit on Windows.
        let target_root = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(std::env::var_os("OUT_DIR").unwrap()));
        let mut flavor = String::from("portable");
        if cuda_enabled {
            flavor.push_str("-cuda");
        }
        if vulkan_enabled {
            flavor.push_str("-vulkan");
        }
        if metal_enabled {
            flavor.push_str("-metal");
        }
        target_root.join("cmake").join(flavor)
    } else {
        PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("portable")
    };
    cfg.out_dir(portable_out_dir);
    cfg.build_target("pie_driver_portable_lib")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("PIE_BRIDGE_INCLUDE_DIR", pie_bridge_include_dir());
    enable_position_independent_archives(&mut cfg);
    // CMake caches option values under OUT_DIR. Define every backend flag
    // explicitly so flipping PIE_PORTABLE_* between builds cannot leave stale
    // CUDA/Vulkan/Metal sources compiled into the portable archive.
    cfg.define("GGML_CUDA", if cuda_enabled { "ON" } else { "OFF" })
        .define("GGML_VULKAN", if vulkan_enabled { "ON" } else { "OFF" })
        .define("GGML_METAL", if metal_enabled { "ON" } else { "OFF" });
    if cuda_enabled {
        cfg
            // ggml's CUDA-graph capture/replay is OFF by default at the ggml-
            // subproject level; llama.cpp's parent CMakeLists.txt flips it
            // ON. Embedding ggml directly (as we do here) misses that override
            // and ends up issuing ~30 kernel launches per decode step, which
            // costs ~1 ms vs ~0.3 ms with capture. Enable explicitly.
            .define("GGML_CUDA_GRAPHS", "ON")
            .define("GGML_STATIC", "ON")
            // llama.cpp b8994 added a multi-GPU NCCL allreduce path inside
            // ggml-cuda (`GGML_CUDA_NCCL`, default ON when NCCL is present
            // on the host). When set, libggml-cuda.a references nccl
            // symbols that the portable-cuda flavor (no driver-cuda) can't
            // satisfy at link time. We have our own NCCL-backed
            // tensor-parallel path in driver-cuda, so the ggml-cuda one
            // is unused either way — disable to keep the libggml-cuda.a
            // self-contained.
            .define("GGML_CUDA_NCCL", "OFF");
        let arch = std::env::var("PIE_PORTABLE_CUDA_ARCH").unwrap_or_else(|_| "native".to_string());
        cfg.define("CMAKE_CUDA_ARCHITECTURES", arch);
    }
    if vulkan_enabled {
        println!("cargo:rerun-if-env-changed=VULKAN_SDK");
        define_vulkan_sdk_paths(&mut cfg, &target_os);
        cfg.define("GGML_STATIC", "ON");
    }
    if metal_enabled {
        // ggml-metal links against Apple's MetalKit / Foundation. The
        // C++ side handles those via xcrun; we just need to flip the
        // ggml flag and add the framework links below.
        cfg.define("GGML_STATIC", "ON")
            .define("GGML_METAL_EMBED_LIBRARY", "ON");
    }
    // Disable ggml-cpu's OpenMP unconditionally. macOS's Apple clang
    // doesn't ship libomp; on Linux, `-Wl,--as-needed -lgomp` is
    // ordered before `-Wl,--start-group … libggml-cpu.a` and the
    // linker discards libgomp before ggml-cpu.a's GOMP_* symbols
    // are seen, leading to `undefined reference to GOMP_barrier`.
    // Cost: multi-thread CPU inference on the portable backend; not
    // relevant for CI artifacts where the GPU backends (Metal / CUDA
    // / Vulkan) are the hot path. The portable driver no longer uses
    // OpenMP itself.
    cfg.define("GGML_OPENMP", "OFF");
    let dst = cfg.build();
    let build_dir = dst.join("build");

    add_link_search_paths(&build_dir);

    println!("cargo:rustc-link-lib=static=pie_driver_portable_lib");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    if cuda_enabled {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
    }
    if vulkan_enabled {
        println!("cargo:rustc-link-lib=static=ggml-vulkan");
    }
    if metal_enabled {
        println!("cargo:rustc-link-lib=static=ggml-metal");
    }
    // ggml-cpu's macOS build pulls in the BLAS backend (calls
    // `_ggml_backend_blas_reg`); ggml-cpu uses Apple's Accelerate
    // framework for vDSP. Both need explicit links on macOS.
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=static=ggml-blas");
        println!("cargo:rustc-link-arg=-framework");
        println!("cargo:rustc-link-arg=Accelerate");
        if metal_enabled {
            for fw in ["Foundation", "Metal", "MetalKit", "MetalPerformanceShaders"] {
                println!("cargo:rustc-link-arg=-framework");
                println!("cargo:rustc-link-arg={fw}");
            }
        }
    }
    println!("cargo:rustc-link-lib=static=ggml-base");

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
        // SDK works without sudo apt install libvulkan-dev). The
        // Windows SDK uses `Lib`, while Unix installs normally use `lib`.
        if let Ok(sdk) = std::env::var("VULKAN_SDK") {
            let lib_dirs: &[&str] = if target_os == "windows" {
                &["Lib", "lib"]
            } else {
                &["lib", "Lib"]
            };
            for lib_dir in lib_dirs {
                let sdk_lib = Path::new(&sdk).join(lib_dir);
                if sdk_lib.is_dir() {
                    println!("cargo:rustc-link-search=native={}", sdk_lib.display());
                }
            }
        }
        let vulkan_lib = if target_os == "windows" {
            "vulkan-1"
        } else {
            "vulkan"
        };
        println!("cargo:rustc-link-lib={vulkan_lib}");
    }

    add_system_libs(metal_enabled);

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
    let target_os = target_os();
    if target_os != "linux" {
        panic!(
            "--features driver-cuda is Linux-only (got target_os={target_os:?}). \
             On macOS, use `--features driver-portable` with PIE_PORTABLE_METAL=1; \
             on Windows, the cuda flavor is not supported."
        );
    }

    let driver_dir = PathBuf::from("../driver/cuda");
    println!("cargo:rerun-if-changed=../driver/cuda/CMakeLists.txt");
    println!("cargo:rerun-if-changed=../driver/cuda/src");

    let mut cfg = cmake::Config::new(&driver_dir);
    // See `build_portable` — per-flavor out_dir keeps the two CMake
    // configurations from clobbering each other in multi-driver builds.
    cfg.out_dir(std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("cuda"));
    cfg.build_target("pie_driver_cuda_lib")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("PIE_BRIDGE_INCLUDE_DIR", pie_bridge_include_dir());
    enable_position_independent_archives(&mut cfg);

    // (The CUDA driver build no longer invokes Python: FlashInfer MoE
    // launchers are vendored, so no kernel generator runs at build time.)

    println!("cargo:rerun-if-env-changed=CPM_SOURCE_CACHE");
    if let Ok(cache) = std::env::var("CPM_SOURCE_CACHE") {
        cfg.define("CPM_SOURCE_CACHE", cache);
    }

    // Optional Marlin W4A16 support. Keep it off by default because the
    // vendored template kernels add substantial build time, but let Cargo
    // builds opt into the same CMake path used by standalone driver builds.
    println!("cargo:rerun-if-env-changed=PIE_CUDA_BUILD_MARLIN");
    if let Ok(value) = std::env::var("PIE_CUDA_BUILD_MARLIN") {
        let lower = value.to_ascii_lowercase();
        let on = matches!(lower.as_str(), "1" | "on" | "true" | "yes");
        cfg.define("PIE_CUDA_BUILD_MARLIN", if on { "ON" } else { "OFF" });
    }
    println!("cargo:rerun-if-env-changed=PIE_MARLIN_ALL_SHAPES");
    if std::env::var("PIE_MARLIN_ALL_SHAPES")
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "on" | "true" | "yes"))
        .unwrap_or(false)
    {
        cfg.define("PIE_MARLIN_ALL_SHAPES", "ON");
    }

    // nvcc discovery. CMake reads `CMAKE_CUDA_COMPILER` / `CUDACXX` to
    // locate nvcc; some toolchains install CUDA under `/usr/local/cuda`
    // without adding `/usr/local/cuda/bin` to the build user's PATH.
    // Probe standard locations and hand CMake the explicit path so
    // workspace builds work without the user having to source CUDA's
    // env script first.
    println!("cargo:rerun-if-env-changed=CUDACXX");
    println!("cargo:rerun-if-env-changed=CMAKE_CUDA_COMPILER");
    if std::env::var_os("CUDACXX").is_none() && std::env::var_os("CMAKE_CUDA_COMPILER").is_none() {
        for candidate in [
            "/usr/local/cuda/bin/nvcc",
            "/usr/local/cuda-13/bin/nvcc",
            "/usr/local/cuda-13.0/bin/nvcc",
            "/usr/local/cuda-12/bin/nvcc",
            "/usr/local/cuda-12.8/bin/nvcc",
            "/opt/cuda/bin/nvcc",
        ] {
            if Path::new(candidate).exists() {
                cfg.define("CMAKE_CUDA_COMPILER", candidate);
                break;
            }
        }
    }

    // CUDA architecture. driver/cuda's `DetectCudaArchitecture.cmake`
    // shells out to `nvidia-smi` if `CMAKE_CUDA_ARCHITECTURES` isn't
    // pre-set, which fails on CI runners with no GPU. cmake-rs does
    // not auto-forward env vars, so honor the standard
    // `CMAKE_CUDA_ARCHITECTURES` env var explicitly.
    println!("cargo:rerun-if-env-changed=CMAKE_CUDA_ARCHITECTURES");
    if let Ok(arch) = std::env::var("CMAKE_CUDA_ARCHITECTURES") {
        cfg.define("CMAKE_CUDA_ARCHITECTURES", arch);
    }

    // Compiler cache for nvcc + host C++ compiles: prefer sccache (it has a
    // GitHub Actions cache backend handy in CI), fall back to ccache. Speeds
    // up clean rebuilds / branch switches; no effect on a cold cache. Wire
    // only when one is on PATH so runners without it are unaffected. Override
    // explicitly with PIE_COMPILER_LAUNCHER=<sccache|ccache|/path>.
    println!("cargo:rerun-if-env-changed=PIE_COMPILER_LAUNCHER");
    let launcher = std::env::var("PIE_COMPILER_LAUNCHER")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            ["sccache", "ccache"].into_iter().find(|c| {
                std::process::Command::new(c)
                    .arg("--version")
                    .output()
                    .map(|o| o.status.success())
                    .unwrap_or(false)
            }).map(String::from)
        });
    if let Some(launcher) = launcher {
        cfg.define("CMAKE_C_COMPILER_LAUNCHER", &launcher);
        cfg.define("CMAKE_CXX_COMPILER_LAUNCHER", &launcher);
        cfg.define("CMAKE_CUDA_COMPILER_LAUNCHER", &launcher);
        println!("cargo:warning=cuda driver compiler launcher: {launcher}");
    }

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

    // Driver lib. tomlplusplus / nlohmann_json / CLI11 are header-only
    // and produce no archive; the static lib is self-contained.
    println!("cargo:rustc-link-lib=static=pie_driver_cuda_lib");

    // CUDA toolkit: dynamic-link cudart + cublas + cublasLt.
    // The cuda driver's `src/ops/gemm.cpp` directly references
    // `cublasLt*` symbols (the native FP8 W8A16 path on sm_89+),
    // so we must satisfy them at link time. Runtime contract: the
    // host has CUDA toolkit installed; all three .so files ship
    // together with the toolkit.
    link_cuda_toolkit_dynamic(&["cudart", "cublas", "cublasLt"]);
    link_cuda_driver_stub();

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
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", nccl_lib_dir.display());
    } else {
        println!("cargo:rustc-link-lib=nccl");
    }

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

fn define_vulkan_sdk_paths(cfg: &mut cmake::Config, target_os: &str) {
    let Ok(sdk) = std::env::var("VULKAN_SDK") else {
        return;
    };
    let sdk = PathBuf::from(sdk);

    let include_dir = sdk.join("Include");
    if include_dir.join("vulkan").join("vulkan.h").is_file() {
        cfg.define("Vulkan_INCLUDE_DIR", include_dir.display().to_string());
    }

    let library_candidates: &[&str] = if target_os == "windows" {
        &["Lib/vulkan-1.lib", "lib/vulkan-1.lib"]
    } else if target_os == "macos" {
        &["lib/libvulkan.dylib", "Lib/libvulkan.dylib"]
    } else {
        &["lib/libvulkan.so", "Lib/libvulkan.so"]
    };
    for candidate in library_candidates {
        let library = sdk.join(candidate);
        if library.is_file() {
            cfg.define("Vulkan_LIBRARY", library.display().to_string());
            break;
        }
    }
}

/// The embedded drivers are static CMake archives, but downstream crates may
/// link them into a shared object (notably the pyo3 `pie-server` wheel). Keep
/// the archives PIC-compatible so the same embedded driver build works for both
/// the standalone CLI binary and Python extension module.
fn enable_position_independent_archives(cfg: &mut cmake::Config) {
    cfg.define("CMAKE_POSITION_INDEPENDENT_CODE", "ON");
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
            panic!("pie-server: unsupported target OS {other:?}");
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
