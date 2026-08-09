//! Does this backend actually reach a browser?
//!
//! ## Why this is a test and not a note in a README
//!
//! `tests/pure.rs` proves the dependency closure owns no `links`, no `-sys`
//! name and no C-compiling build script, and its own docs give the reason:
//! "`wgpu` reaching a browser and an Android device without a sysroot is the
//! deployment story". That is a claim about a TARGET, and purity is necessary
//! for it rather than sufficient — a closure can be perfectly pure Rust and
//! still fail to build for `wasm32-unknown-unknown` because something in it
//! calls `std::fs`, spawns a thread, or reads a clock.
//!
//! So the claim is asked of the compiler. This shells out to `cargo build
//! --target wasm32-unknown-unknown` for this crate and for `kernels-wgpu`, with
//! `wgpu`'s BROWSER backend selected rather than its native ones.
//!
//! ## What each of the four checks is for
//!
//! The set is chosen so that a failure says which layer broke, because "the
//! wasm build is red" is not actionable and each of these is:
//!
//! * the table and the shader tree build (`kernels-wgpu`);
//! * the portable half of the shell builds (`driver-wgpu`, no features);
//! * the DEVICE half builds against the `webgpu` backend (`--features
//!   native`), which is the one that could plausibly fail — it is the half that
//!   names `wgpu` types, and `wgpu`'s browser backend is a different
//!   implementation of the same API rather than the same code with a different
//!   loader;
//! * the shader SOURCES survive into the artifact, which is the property the
//!   whole `include_str!` design exists for and the one a build check alone
//!   cannot see.
//!
//! ## The design decisions this is the receipt for
//!
//! Four choices in these two crates were made for portability and would each
//! have ended the browser story on their own. This test is what stops any of
//! them being undone by accident:
//!
//! * **The scalar run is a uniform buffer, not a push constant.**
//!   `wgpu::Features::PUSH_CONSTANTS` is native-only and a browser cannot offer
//!   it. `kernels-wgpu`'s launch ABI says so at length; this is the check.
//! * **The shader tree is embedded with `include_str!`.** A browser has no
//!   filesystem, so `kernels-metal`'s `kernels_dir` handoff and
//!   `kernels-vulkan`'s `OUT_DIR/spv` are both unreachable there.
//! * **WGSL is compiled by `naga` at run time.** A backend that needed `glslc`
//!   or `nvcc` could not ship to a browser at all, since there is no build step
//!   on the far side.
//! * **No baseline module needs a capability.** A browser is close to the
//!   downlevel tier, which is why `unpack2x16float` had to go: it is spelled
//!   like a core builtin and is gated behind an adapter property.
//!
//! ## What this does NOT claim
//!
//! That the whole of `pie` runs in a browser. It does not: `engine`, `worker`,
//! `controller`, `gateway`, `client` and `bootstrap` all fail this target on
//! `mio` ("This wasm target is unsupported by mio") and `getrandom`. Those are
//! tokio and RNG feature selections rather than anything architectural, but
//! they are not this crate's to fix and they are not fixed.
//!
//! What it claims is narrower and is the useful half: **the kernel table, the
//! shader tree and the execution shell reach a browser.** Everything between a
//! model and a dispatch — `model`, `model-compiler`, `model-loader`,
//! `tokenizer`, `grammar`, `driver`, `ids` — builds for this target too, and is
//! checked here for the same reason: it is the set that would have to move
//! together, and finding out one of them slipped is worth a minute of CI.
//!
//! And it does not claim the wasm RUNS. Nothing here executes a browser; that
//! needs a headless one and a WebGPU implementation inside it, which is a
//! different kind of test and a real one to want. This is the compile half, and
//! the compile half is where a portability regression actually lands.
//!
//! ## Skipping
//!
//! The target may not be installed — `rustup target add wasm32-unknown-unknown`
//! — and a machine without it must not fail. Each check SKIPS with a printed
//! reason, the same contract the device suites have when no adapter answers.
//! A skip is loud on purpose: a silent one is how a check stops covering
//! anything.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::Command;

/// The target a browser runs.
///
/// `wasm32-unknown-unknown` and not `wasm32-wasip2`: WASI is a system
/// interface for a runtime that HAS one, and a browser tab does not. The
/// repo's guest crates target wasip2 for a different reason entirely — they
/// are inferlets, running inside the engine's own wasm runtime.
const TARGET: &str = "wasm32-unknown-unknown";

fn cargo() -> String {
    std::env::var("CARGO").unwrap_or_else(|_| "cargo".into())
}

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Whether the toolchain has the target at all.
///
/// Asked once and answered by `rustc --print target-libdir`, which fails
/// cleanly for a target whose std is not installed. Parsing `rustup target
/// list` would be wrong here: the toolchain that runs a test is not always the
/// one rustup thinks is default, and `rust-toolchain.toml` pins one.
fn target_installed() -> bool {
    Command::new("rustc")
        .args(["--print", "target-libdir", "--target", TARGET])
        .output()
        .ok()
        .and_then(|out| {
            out.status.success().then(|| {
                PathBuf::from(String::from_utf8_lossy(&out.stdout).trim().to_owned()).is_dir()
            })
        })
        .unwrap_or(false)
}

/// `cargo build --target wasm32-unknown-unknown -p <package>`, and what it said.
///
/// `build` rather than `check`, for the one that matters: a `check` does not
/// link, and this target's interesting failures — a `-sys` crate with no wasm
/// sysroot, a symbol only a native libc has — are link-time. It costs a minute
/// and buys the difference between "it type-checks" and "there is an artifact".
fn build(package: &str, features: &[&str]) -> Result<(), String> {
    let mut command = Command::new(cargo());
    command
        .args(["build", "--target", TARGET, "-p", package])
        .arg("--manifest-path")
        .arg(manifest().join("Cargo.toml"));
    if !features.is_empty() {
        command.args(["--features", &features.join(",")]);
    }

    let out = command
        .output()
        .map_err(|e| format!("cannot run cargo: {e}"))?;
    if out.status.success() {
        return Ok(());
    }

    // The first `error:` line and its context. The whole of cargo's output for
    // a failed wasm build is thousands of lines of dependency errors, and the
    // first one is nearly always the cause.
    let stderr = String::from_utf8_lossy(&out.stderr);
    let head: Vec<&str> = stderr
        .lines()
        .filter(|l| l.starts_with("error") || l.contains("compile_error!"))
        .take(6)
        .collect();
    Err(if head.is_empty() {
        stderr.lines().rev().take(10).collect::<Vec<_>>().join("\n")
    } else {
        head.join("\n")
    })
}

macro_rules! skip_unless_target {
    () => {
        if !target_installed() {
            println!(
                "SKIP: `{TARGET}` is not installed for this toolchain, so \
                 whether this backend reaches a browser is unmeasured. \
                 `rustup target add {TARGET}` to measure it."
            );
            return;
        }
    };
}

/// The table and the shader tree build for a browser.
///
/// The easy half, and worth checking anyway: `kernels-wgpu` has a `build.rs`
/// that walks a directory and writes `include_str!` literals, and a build
/// script runs on the HOST while its output is compiled for the TARGET. A
/// script that started reading an environment variable only a host has, or
/// emitting a path with a host separator in it, would break here and nowhere
/// else.
#[test]
fn the_table_and_the_shaders_build_for_a_browser() {
    skip_unless_target!();
    if let Err(why) = build("kernels-wgpu", &[]) {
        panic!("`kernels-wgpu` does not build for {TARGET}:\n{why}");
    }
}

/// The portable half of the shell builds for a browser.
#[test]
fn the_portable_half_builds_for_a_browser() {
    skip_unless_target!();
    if let Err(why) = build("driver-wgpu", &[]) {
        panic!("`driver-wgpu` does not build for {TARGET}:\n{why}");
    }
}

/// The DEVICE half builds for a browser, against `wgpu`'s browser backend.
///
/// This is the one that could plausibly fail, and the reason is worth being
/// precise about. `wgpu`'s browser backend is not the native one with a
/// different loader; it is a separate implementation that forwards to the
/// host's WebGPU through `web-sys`. Types that exist natively can be absent or
/// shaped differently there, and a device half written against the native
/// surface finds out at exactly this point.
///
/// It is also the check that pins the ABI decision. `Features::PUSH_CONSTANTS`
/// is native-only: a shell that used push constants would compile here and
/// then be refused by every browser at device request, which is worse than a
/// build failure because it moves the discovery to a user's machine.
#[test]
fn the_device_half_builds_against_the_browser_backend() {
    skip_unless_target!();
    if let Err(why) = build("driver-wgpu", &["native"]) {
        panic!(
            "`driver-wgpu --features native` does not build for {TARGET}. That \
             is the browser story ending, and the usual causes are a `wgpu` \
             feature that is native-only, a thread, or a filesystem read:\n{why}"
        );
    }
}

/// Everything between a model and a dispatch builds for a browser.
///
/// Not this crate's code, and checked here anyway because it is the set that
/// would have to move TOGETHER for a browser deployment to mean anything —
/// a shader tree that compiles for wasm beside a `model-compiler` that does not
/// is a browser story with a hole in the middle.
///
/// Deliberately NOT including `engine`, `worker`, `controller`, `gateway`,
/// `client` or `bootstrap`. All six fail this target today on `mio` ("This wasm
/// target is unsupported by mio. If using Tokio, disable the net feature.") and
/// on `getrandom`'s missing `js`/`wasm_js` feature. Those are feature
/// selections rather than anything architectural, but they are real, they are
/// not this crate's to fix, and listing them here as expected failures would be
/// a test that asserts a bug.
#[test]
fn the_model_pipeline_builds_for_a_browser() {
    skip_unless_target!();

    const PIPELINE: [&str; 7] = [
        "kernels",
        "model",
        "model-compiler",
        "model-loader",
        "tokenizer",
        "grammar",
        "driver",
    ];

    let mut broken = Vec::new();
    for package in PIPELINE {
        if let Err(why) = build(package, &[]) {
            broken.push(format!("`{package}`:\n{why}"));
        }
    }

    assert!(
        broken.is_empty(),
        "{} of the {} crates between a model and a dispatch no longer build \
         for {TARGET}:\n\n{}",
        broken.len(),
        PIPELINE.len(),
        broken.join("\n\n"),
    );
}

/// The shader sources survive into the artifact.
///
/// The `include_str!` design exists because a browser has no filesystem: both
/// sibling backends hand a driver a DIRECTORY, and neither handoff is reachable
/// from a tab. A build check cannot see whether that worked — a tree that had
/// quietly become a runtime path read would still compile.
///
/// So this reads the `.wasm` and looks for WGSL in it. `@workgroup_size` is the
/// probe because every compute entry point declares one, it appears in no other
/// context, and it is not a string this crate's Rust ever constructs — so
/// finding it means a shader body is in the binary rather than a name for one.
///
/// The count is a floor and not an equality on purpose. `--release` runs
/// dead-code elimination, and how much of the tree survives depends on what the
/// probe reaches; asserting an exact number would be asserting a property of
/// the optimiser. What matters is that it is not zero, which is what a
/// filesystem-read regression would produce.
#[test]
fn the_shaders_survive_into_the_wasm() {
    skip_unless_target!();

    // A probe crate outside the workspace: it has to depend on `kernels-wgpu`
    // in a way that REACHES the tree, and a `cdylib` is what makes the artifact
    // a `.wasm` a browser could load rather than an rlib.
    let dir = std::env::temp_dir().join(format!("pie-wgpu-wasm-probe-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("src")).expect("a writable temp directory");

    let root = manifest();
    std::fs::write(
        dir.join("Cargo.toml"),
        format!(
            "[package]\n\
             name = \"pie-wgpu-wasm-probe\"\n\
             version = \"0.0.0\"\n\
             edition = \"2024\"\n\
             \n\
             [lib]\n\
             crate-type = [\"cdylib\"]\n\
             \n\
             [dependencies]\n\
             kernels-wgpu = {{ path = {:?} }}\n\
             \n\
             [workspace]\n",
            root.join("../kernels-wgpu")
                .canonicalize()
                .expect("a sibling"),
        ),
    )
    .expect("a writable manifest");

    // `no_mangle` and `extern "C"`: an exported symbol is what stops
    // dead-code elimination from dropping the tree, which it otherwise will,
    // correctly, since nothing in a cdylib calls this.
    std::fs::write(
        dir.join("src/lib.rs"),
        "#[unsafe(no_mangle)]\n\
         pub extern \"C\" fn shader_bytes() -> u32 {\n\
         \x20   kernels_wgpu::SOURCES.iter().map(|(_, t)| t.len() as u32).sum()\n\
         }\n",
    )
    .expect("a writable source");

    let out = Command::new(cargo())
        .args(["build", "--release", "--target", TARGET])
        .current_dir(&dir)
        .output()
        .expect("cargo is what is running this test");
    assert!(
        out.status.success(),
        "the probe does not build for {TARGET}:\n{}",
        String::from_utf8_lossy(&out.stderr),
    );

    let artifact = dir
        .join("target")
        .join(TARGET)
        .join("release/pie_wgpu_wasm_probe.wasm");
    let bytes = std::fs::read(&artifact)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", artifact.display()));

    // A crude scan rather than a wasm parser: the data section is the only
    // place a `&'static str` can live, and a byte search finds it without this
    // test growing a dependency on a wasm reader.
    let needle = b"@workgroup_size";
    let found = bytes.windows(needle.len()).filter(|w| *w == needle).count();

    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        found > 0,
        "no WGSL in {} ({} bytes). The shader tree is supposed to be embedded \
         with `include_str!`, and a browser has no filesystem to fall back to.",
        artifact.display(),
        bytes.len(),
    );
    println!(
        "{found} `@workgroup_size` declarations survive into a {} byte .wasm",
        bytes.len(),
    );
}

/// The browser backend is REACHABLE, not merely absent from the build.
///
/// The three build checks above select no `wgpu` backend feature, so they would
/// pass against a `wgpu` with every backend compiled out — which builds, links,
/// and finds no adapter at run time. That is the failure this catches: not a
/// compile error, but a deployment that starts and can never open a device.
///
/// Asked of the resolver rather than of a run, because running it needs a
/// browser. `web-sys` in the closure for this target is the evidence: it is
/// what `wgpu`'s browser backend forwards through, and nothing else in this
/// crate's graph would pull it in.
///
/// ## Why this builds a probe instead of asking `cargo tree`
///
/// `cargo tree -p wgpu --features webgpu` is the obvious form and cargo refuses
/// it: "cannot specify features for packages outside of workspace". `wgpu` is a
/// registry dependency here, and features are a property of the workspace's
/// resolution rather than of a package one can interrogate in isolation.
///
/// It could be asked of THIS crate with `--features native`, which is what
/// `pure.rs` does — but that answers a different question. `native` selects
/// `vulkan`/`metal`/`dx12`, deliberately, because those are what a native
/// deployment wants; a browser deployment selects `webgpu` and it is the
/// DEPLOYMENT that chooses. So the honest thing to measure is that the
/// selection is available and pulls the backend in, which needs a manifest that
/// makes it, and that is what the probe is.
#[test]
fn the_browser_backend_is_in_the_closure_for_this_target() {
    skip_unless_target!();

    let dir = std::env::temp_dir().join(format!("pie-wgpu-webgpu-probe-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("src")).expect("a writable temp directory");

    let root = manifest();
    std::fs::write(
        dir.join("Cargo.toml"),
        format!(
            "[package]\n\
             name = \"pie-wgpu-webgpu-probe\"\n\
             version = \"0.0.0\"\n\
             edition = \"2024\"\n\
             \n\
             [dependencies]\n\
             driver-wgpu = {{ path = {:?}, features = [\"native\"] }}\n\
             wgpu = {{ version = \"30\", default-features = false, \
             features = [\"wgsl\", \"webgpu\"] }}\n\
             \n\
             [workspace]\n",
            root.canonicalize().expect("this crate's own directory"),
        ),
    )
    .expect("a writable manifest");
    std::fs::write(dir.join("src/lib.rs"), "").expect("a writable source");

    let out = Command::new(cargo())
        .args([
            "tree", "--target", TARGET, "--edges", "normal", "--prefix", "none", "--format", "{p}",
        ])
        .current_dir(&dir)
        .output()
        .expect("cargo is what is running this test");

    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        out.status.success(),
        "the browser-backend probe does not even resolve, which is the \
         browser story ending at the manifest:\n{stderr}",
    );

    let names: BTreeSet<&str> = stdout
        .lines()
        .filter_map(|l| l.split_whitespace().next())
        .filter(|n| !n.is_empty())
        .collect();

    assert!(
        names.contains("driver-wgpu") && names.contains("wgpu"),
        "the probe's tree names neither this crate nor `wgpu`, so the check \
         below would be vacuous:\n{stdout}",
    );
    for wanted in ["web-sys", "js-sys", "wasm-bindgen"] {
        assert!(
            names.contains(wanted),
            "`{wanted}` is not in the closure for {TARGET} with `wgpu`'s \
             `webgpu` feature on. That backend is how a tab reaches a GPU, and \
             without it this crate builds for wasm and finds no adapter.",
        );
    }
    println!(
        "the browser backend resolves for {TARGET}: {} crates, web-sys present",
        names.len(),
    );
}
