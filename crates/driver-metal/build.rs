//! Build `csrc/` into `libpie_driver_metal_lib.a` and emit the link line.
//!
//! This is the Cargo↔native *link bridge* for the Metal shell: it invokes the
//! CMake build under `csrc/`, forwards the include handoffs cargo publishes
//! (`driver-abi`, `model-loader-capi`, `model-compiler`, and this workspace's
//! `driver` substrate), and emits the `cargo:rustc-link-*` directives for the
//! final binary — which rustc, not CMake, links.
//!
//! Build-time *discovery* — the Metal toolchain, the MLX provider, the CPM
//! cache — stays in `csrc/CMakeLists.txt`, the native build system's proper
//! home for it.
//!
//! This used to live in `worker`'s build.rs, which meant the worker knew
//! where the C++ trees were and how to build them. It does not need to: it
//! needs a driver, and a driver is now a crate that builds itself. Selection
//! is still by feature, but the feature turns on a DEPENDENCY rather than a
//! branch in someone else's build script.

use std::path::{Path, PathBuf};

fn main() {
    let target_os = target_os();
    if target_os != "macos" {
        panic!(
            "driver-metal is macOS-only (got target_os={target_os:?}). \
             On Linux, use `--features driver-cuda`; the metal flavor targets \
             Apple Silicon via native Metal shaders (MLX is an opt-in legacy path)."
        );
    }

    let csrc = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("csrc");
    println!("cargo:rerun-if-changed=csrc/CMakeLists.txt");
    println!("cargo:rerun-if-changed=csrc/cmake");
    println!("cargo:rerun-if-changed=csrc/src");

    let mut cfg = cmake::Config::new(&csrc);
    cfg.out_dir(PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("metal"));
    cfg.build_target("pie_driver_metal_lib")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .define("PIE_DRIVER_ABI_INCLUDE_DIR", dep_include("PIE_DRIVER_ABI", "driver-abi"))
        .define("PIE_LOADER_INCLUDE_DIR", dep_include("PIE_LOADER", "model-loader-capi"))
        .define("PIE_FORWARD_INCLUDE_DIR", dep_include("PIE_FORWARD", "model"))
        .define("PIE_DRIVER_INCLUDE_DIR", dep_include("PIE_DRIVER", "driver"))
        .define("PIE_PTIR_INCLUDE_DIR", sibling("tensor-compiler").join("include"))
        .define("PIE_PTIR_RUNTIME_DIR", sibling("tensor-compiler").join("runtime"))
        .define("PIE_REPO_ROOT", repo_root())
        // The shader tree, from cargo rather than by walking out of this tree.
        // It is an include dir (for the `*_params.h` a shader and its host
        // caller must agree on) and PIE_METAL_KERNELS_DIR_DEFAULT, the path
        // baked into the binary for the runtime shader compiler.
        .define(
            "PIE_KERNELS_METAL_DIR",
            std::env::var("DEP_PIE_KERNELS_METAL_KERNELS_DIR").expect(
                "kernels-metal's build.rs publishes the shader directory as \
                 cargo:kernels_dir",
            ),
        )
        // The HOST library, which is a different tree: `include/pie/kernels/`
        // holds the launch shapes, the entrypoint name grammar and the `Grid`
        // PODs -- ordinary C++ this shell compiles against, with no Metal in
        // it. They were one directory until §4 of
        // .wiki/kernel-metal-refactor.md separated the two roles.
        .define(
            "PIE_KERNELS_METAL_INCLUDE_DIR",
            metal_include_dir(),
        );

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
        cfg.define(
            "PIE_METAL_MLX_BUILD_METAL",
            if build_metal_gpu { "ON" } else { "OFF" },
        );
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
    // IOKit, for the GPU core count `MTLDevice` does not publish
    // (`csrc/src/device_tuning_apple.mm`). Named here as well as in the
    // CMakeLists because CMake's link line does not reach the Rust test
    // target, which links the static archive itself.
    println!("cargo:rustc-link-lib=framework=IOKit");
    add_system_libs();

    println!(
        "cargo:rustc-env=PIE_DRIVER_METAL_BUILD_DIR={}",
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

/// A sibling crate's directory. Used for assets no `links` handoff publishes:
/// `tensor-compiler`'s generated `ptir_abi.h` and its device runtime
/// templates, which the emitters own and the drivers only read.
fn sibling(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/driver-metal has a parent")
        .join(name)
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/driver-metal sits two levels below the repo root")
        .to_path_buf()
}

/// True when `env` is set to a truthy value (`1`/`on`/`true`/`yes`). Also
/// registers a Cargo `rerun-if-env-changed` for it.
fn env_is_truthy(env: &str) -> bool {
    println!("cargo:rerun-if-env-changed={env}");
    std::env::var(env)
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "on" | "true" | "yes"))
        .unwrap_or(false)
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

fn target_os() -> String {
    std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default()
}

fn add_system_libs() {
    println!("cargo:rustc-link-lib=c++");
    // ggml-metal pulls these three frameworks. -framework on macOS is the
    // moral equivalent of -l on linux.
    for framework in ["Metal", "MetalKit", "Foundation"] {
        println!("cargo:rustc-link-lib=framework={framework}");
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

/// The host include root out of `DEP_PIE_KERNELS_METAL_INCLUDE`.
///
/// `kernels-metal` publishes `cargo:include` as a `;`-separated list — the host
/// library first, the shader tree second — because a C++ include path is
/// naturally plural and cargo gives a `links` crate one key per name. CMake
/// takes them as two variables, so this picks the first and the shader tree
/// arrives through `PIE_KERNELS_METAL_DIR` above.
fn metal_include_dir() -> String {
    let published = std::env::var("DEP_PIE_KERNELS_METAL_INCLUDE").expect(
        "kernels-metal's build.rs publishes the host include roots as cargo:include",
    );
    published
        .split(';')
        .next()
        .expect("cargo:include is not empty")
        .to_string()
}
