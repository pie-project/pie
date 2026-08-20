//! The runtime Slang compiler, exercised against the tree it carries.
//!
//! # What this is for
//!
//! `build.rs` used to compile every declared point, so "does this shader
//! compile" was answered by the build failing. Compiling on demand moves that
//! answer to the first fire — which is a better place for a point nothing
//! runs, and a worse one for a point something runs on a machine that is
//! already serving. This fixture is what keeps the answer early.
//!
//! It is `native`-only because compiling is what `native` means here.

#![cfg(feature = "runtime")]

use kernels::jit::Root;
use kernels_vulkan::Capability;
use kernels_vulkan::runtime::{self, native::Slang};

/// The architecture term is a constant, so a point compiles without a device.
///
/// This is the difference from CUDA that lets the fixtures above run in CI. If
/// `arch` ever became `None`, every `resolve` would take the `no_device` arm
/// and this crate would refuse to compile anything, everywhere.
#[test]
fn arch_is_a_constant() {
    use kernels::jit::Compiles;
    assert_eq!(<Slang as Compiles>::arch(), Some("spirv"));
}

/// The tree is carried, and every source in it is reachable by its path.
#[test]
fn sources_are_carried() {
    let sources = runtime::sources();
    assert!(
        sources.len() >= 30,
        "the shader tree has ~37 files and this build carried {}",
        sources.len()
    );
    for (path, text) in sources {
        assert!(
            path.ends_with(".slang"),
            "`{path}` is in the source table and is not a shader"
        );
        assert!(!text.is_empty(), "`{path}` was carried empty");
        assert_eq!(
            runtime::source(path),
            Some(*text),
            "`{path}` is in the table and does not resolve by its own name"
        );
    }
}

/// The table is sorted, because [`runtime::source`] binary-searches it.
#[test]
fn sources_are_sorted() {
    let sources = runtime::sources();
    assert!(
        sources.windows(2).all(|w| w[0].0 < w[1].0),
        "the source table is binary-searched and is not sorted"
    );
}

/// A point compiles, and what comes out is SPIR-V.
///
/// One point rather than the tree, because this asserts the PLANE works —
/// options reach the preprocessor, `#include` resolves, the entry point
/// survives, the words are a valid module. Compiling every point is
/// `every_point_compiles`, which is `--ignored` for the reason it says.
#[test]
fn one_point_compiles() {
    let root = point(
        "quant/qmm_t.slang",
        &["PIE_GROUP=128", "PIE_BITS=4", "PIE_BM=16", "PIE_BN=16"],
    );
    let words = compile(&root, "affine_qmm_t_bfloat16_gs_128_b_4_bm_16_bn_16");

    assert_eq!(
        words[0], 0x0723_0203,
        "the first word of a SPIR-V module is its magic number"
    );
    assert!(
        words.len() > 64,
        "a GEMM lowered to {} words is not a GEMM",
        words.len()
    );
}

/// Two points off the same root differ, which is what `options` is for.
///
/// The lattice's whole premise: one source, and the axes chosen per point. If
/// the defines did not reach the preprocessor this would still compile — it
/// would just compile the same shader twice, and the caches would agree with
/// each other about the wrong thing.
#[test]
fn options_reach_the_shader() {
    let small = point(
        "quant/qmm_t.slang",
        &["PIE_GROUP=128", "PIE_BITS=4", "PIE_BM=16", "PIE_BN=16"],
    );
    let large = point(
        "quant/qmm_t.slang",
        &["PIE_GROUP=32", "PIE_BITS=8", "PIE_BM=64", "PIE_BN=64"],
    );

    let a = compile(&small, "affine_qmm_t_bfloat16_gs_128_b_4_bm_16_bn_16");
    let b = compile(&large, "affine_qmm_t_bfloat16_gs_32_b_8_bm_64_bn_64");

    assert_ne!(
        a, b,
        "two points of the lattice compiled to identical SPIR-V, so the \
         defines did not reach the preprocessor"
    );
}

/// A `#include` resolves, which is what the materialised tree is for.
///
/// `quant/qmm_t.slang` opens with `#include "common/bf16.slang"`, so the test
/// above already depends on this — but it depends on it silently, and a
/// failure would read as "the GEMM did not compile". This names the reason.
#[test]
fn includes_resolve() {
    let tree = runtime::materialise().expect("the tree materialises");
    assert!(
        tree.join("common/bf16.slang").is_file(),
        "`common/bf16.slang` is `#include`d by the tree and is not at {}",
        tree.display()
    );
    assert!(
        tree.join("quant/qmm_t.slang").is_file(),
        "the tree is materialised and `quant/qmm_t.slang` is not in it"
    );
}

/// A point that does not exist refuses, rather than compiling something else.
///
/// `PIE_ENTRYPOINT` names a function the shader stamps. A misspelling used to
/// be a build error; it has to stay an error.
#[test]
fn an_unknown_point_refuses() {
    let root = point(
        "quant/qmm_t.slang",
        &["PIE_GROUP=128", "PIE_BITS=4", "PIE_BM=16", "PIE_BN=16"],
    );
    let asked =
        kernels::jit::resolve::<Slang>(&root, "affine_qmm_t_bfloat16_gs_999_b_9_bm_9_bn_9", nope());
    match asked {
        Err(_) => {}
        Ok(compiled) => panic!(
            "a point the shader does not stamp compiled anyway: {} words, mangled as `{}`",
            compiled.entry.len(),
            compiled.mangled
        ),
    }
}

/// It refuses even when an image for it is already in the cache.
///
/// # The bug this is here for
///
/// `jit::load` reads the disk cache first and calls `Compiles::compile` only
/// on a miss. So the census check, written first as a guard INSIDE `compile`,
/// ran exactly once per point per machine — and the run that preceded it had
/// already written a valid SPIR-V module for `..._gs_999_...` to the cache.
/// Every run after that read the image back and reported success, with the
/// check sitting right there in the source, never reached. It took deleting
/// `~/.cache/pie/*.image` to see the check work at all.
///
/// # Why this test runs the ask TWICE, in two processes
///
/// A cache hit is the failing condition, so a test that asks once in a clean
/// cache proves nothing: it takes the `compile` path, where the guard also
/// worked. Reproducing it needs the image to exist BEFORE the ask — and only
/// `jit` can write one, through the private `write_disk`, on a successful
/// compile.
///
/// So the first process asks with the census check disabled
/// (`PIE_VULKAN_SKIP_CENSUS`, which exists for this test and is read nowhere
/// else), which compiles the bogus point and caches it exactly as the original
/// accident did. The second asks normally. If the check is behind the cache,
/// the second ask succeeds — which is the regression, and is what this
/// asserts against.
#[test]
fn a_cached_unknown_point_still_refuses() {
    // A name no `// pie:instantiate` line declares, unique to this run so a
    // previous run's cache cannot decide the outcome either way.
    let fake = format!(
        "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_{}",
        std::process::id()
    );

    let exe = std::env::current_exe().expect("the test binary");
    let mut planted = std::process::Command::new(&exe);
    planted
        .arg("--exact")
        .arg("plant_a_bogus_image")
        .arg("--ignored")
        .arg("--nocapture")
        .env("PIE_VULKAN_SKIP_CENSUS", "1")
        .env("PIE_VULKAN_BOGUS_POINT", &fake);
    let out = planted.output().expect("the planting run starts");
    assert!(
        out.status.success(),
        "the planting run failed, so nothing was cached and this test would \
         pass for the wrong reason:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // The image is now in the cache under the bogus point's key. Ask again,
    // with the check on.
    let root = point(
        "quant/qmm_t.slang",
        &["PIE_GROUP=64", "PIE_BITS=4", "PIE_BM=32", "PIE_BN=32"],
    );
    let asked = kernels::jit::resolve::<Slang>(&root, &fake, nope());
    assert!(
        asked.is_err(),
        "`{fake}` is not in the census, is in the CACHE, and resolved anyway \
         — the check has moved back behind the cache"
    );
}

/// Compile and cache a point the census does not have. Not a test.
///
/// `--ignored` so a normal run skips it, and it refuses to do anything unless
/// `a_cached_unknown_point_still_refuses` has set both variables.
#[test]
#[ignore = "a fixture for `a_cached_unknown_point_still_refuses`, not a test"]
fn plant_a_bogus_image() {
    let Ok(fake) = std::env::var("PIE_VULKAN_BOGUS_POINT") else {
        panic!("PIE_VULKAN_BOGUS_POINT is unset, so there is nothing to plant");
    };
    assert!(
        std::env::var_os("PIE_VULKAN_SKIP_CENSUS").is_some(),
        "the census check is on, so this cannot plant anything"
    );
    let root = point(
        "quant/qmm_t.slang",
        &["PIE_GROUP=64", "PIE_BITS=4", "PIE_BM=32", "PIE_BN=32"],
    );
    let planted = kernels::jit::resolve::<Slang>(&root, &fake, nope());
    assert!(
        planted.is_ok(),
        "with the census check off, a bogus point compiles — that is the \
         whole premise, and if it stops being true this fixture is stale"
    );
}

/// A source the tree does not have is `None`, not a panic.
#[test]
fn an_unknown_source_is_none() {
    assert_eq!(runtime::source("quant/not_a_shader.slang"), None);
    assert!(runtime::native::root("nope.slang", Capability::Baseline, &[]).is_none());
}

/// The error `resolve` would return if there were no device.
///
/// Unreachable here, and asserted so by `arch_is_a_constant`: this backend
/// answers a constant architecture because SPIR-V is portable, so the
/// `no_device` arm cannot be taken.
fn nope() -> kernels_vulkan::runtime::native::Failed {
    kernels_vulkan::runtime::native::Failed("unreachable: SPIR-V needs no device".to_string())
}

/// A `Root` for one point of one source.
fn point(file: &'static str, options: &'static [&'static str]) -> Root<Slang> {
    runtime::native::root(file, Capability::Baseline, options)
        .unwrap_or_else(|| panic!("`{file}` is in the carried tree"))
}

/// Compile `entry` out of `root`, or fail with what Slang said.
fn compile(root: &Root<Slang>, entry: &str) -> Vec<u32> {
    match kernels::jit::resolve::<Slang>(root, entry, nope()) {
        Ok(compiled) => compiled.entry.clone(),
        Err(e) => panic!("`{entry}` did not compile: {e}"),
    }
}
