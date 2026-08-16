//! Is this crate reachable without a C toolchain?
//!
//! This is the headline of `driver-wgpu`, and unlike its siblings' version it
//! is not a hygiene check. `wgpu` being pure Rust is the ENTIRE reason this
//! backend exists beside CUDA and Metal: a shell you can build on any machine,
//! cross-compile without a sysroot, and ship without a vendor SDK. If this test
//! fails, the crate has lost the only thing it has that its siblings do not.
//!
//! So the question is asked with `native` ON. A purity claim about the half
//! that does not talk to a GPU would be a technicality; the claim worth making
//! is that the DEVICE half is pure too, which is the one `driver-vulkan` can
//! also make (`ash` loads the loader with `libloading`) and `driver-cuda` can
//! almost make and `driver-metal` cannot make at all.
//!
//! It is easy to hold and easy to lose. Adding a dependency is one line, its
//! own dependencies are not read, and nothing in a passing test suite would
//! notice -- the machine that added it already had the toolchain. So this asks
//! the resolver rather than trusting a reading of the manifests.
//!
//! # How the question is asked
//!
//! Three signals, because no single one is sufficient:
//!
//! * a `links` key, which is how a crate declares it owns a native library;
//! * a build script among the closure's members, which is where C gets compiled
//!   and is worth naming even when it is innocent;
//! * a `-sys` suffix, which is a convention rather than a rule but catches the
//!   ones that predate `links` being required.
//!
//! # What the Vulkan version of this test caught, and what it got wrong
//!
//! Both findings transfer, and both would fire here for the same reasons.
//!
//! It fired the first time that crate took a dependency outside its own three
//! -- `driver-api`, for the `DeviceFacts` the engine's seam is handed -- and
//! named `js-sys`, `wasm-bindgen-shared` and `windows-sys`. All three are real,
//! all three arrive under `tarpc`, and `driver-api` needs `tarpc` for exactly
//! one trait. This crate's edges to `driver-api` and to `driver` are both
//! `default-features = false` from the start because of it.
//!
//! The test was also WRONG in a way worth recording, because the fix could
//! easily have been to weaken it. `cargo metadata` resolves features for the
//! whole workspace at once, so `tarpc` was an edge out of `driver-api`'s node
//! whether or not the crate under test asked for it, and a walk of that graph
//! reported it either way. `cargo tree -p` resolves for one package; the two
//! are intersected below, and the control for that narrowing is that turning
//! the feature back on still fails this test.
//!
//! # What this test found here
//!
//! One crate, and it is instructive rather than alarming: `renderdoc-sys`.
//! `wgpu` enables `wgpu-core/renderdoc` unconditionally on every non-wasm
//! target, so it is in the closure and cannot be turned off from this manifest.
//!
//! It is a `-sys` crate by name and by nothing else. No `links` key, no build
//! script, no build dependency, no `extern` block that links anything: 682
//! lines of `#[repr(C)]` declarations for an API `wgpu-hal` reaches through
//! `libloading` at RUN time, exactly the way `ash` reaches the Vulkan loader.
//! The suffix check found it because the suffix is a convention, which this
//! test's own signal list says out loud.
//!
//! So it is allow-listed by name in [`SYS_IN_NAME_ONLY`] and then re-checked
//! structurally by [`sys_is_only_a_name`], which asks the three real questions
//! of it. Weakening the suffix check instead would have admitted every genuine
//! `-sys` crate along with this one, and the whole value of the check is that
//! it catches the ones nobody looked at.
//!
//! # There is no `unsafe` scan here, and that is the point
//!
//! `driver-vulkan/tests/pure.rs` carries a second test that reads its own
//! module list out of `lib.rs` and scans the ungated modules for the word
//! `unsafe`. It has to: that crate cannot take the workspace lint table,
//! because `unsafe_code = "forbid"` is in it and every `ash` entry point is
//! unsafe, so the guarantee is kept for the portable half only and kept by
//! reading the source.
//!
//! `wgpu` is a safe API and `src/lib.rs` says `#![forbid(unsafe_code)]` for the
//! WHOLE crate, device half included. The compiler now enforces what that scan
//! was standing in for, and it enforces it more strongly -- a forbid cannot be
//! overruled by an inner `allow`, where a scan can be defeated by a macro or a
//! line break. Keeping the scan would be keeping a test that cannot fail while
//! the attribute is there, and a test that cannot fail teaches the next reader
//! that the property is being watched by something weaker than it is.

use std::collections::{BTreeMap, BTreeSet};

/// The crate whose closure is under test.
const ROOT: &str = "driver-wgpu";

/// The crates a build script reaches for when it is going to compile or find a
/// native library.
///
/// Not exhaustive and does not need to be: a crate that compiles C without any
/// of these is doing it by hand, and it will still have declared `links` or be
/// named `-sys`, both of which are checked separately.
const COMPILES_C: &[&str] = &["cc", "cmake", "bindgen", "pkg-config", "autotools"];

/// The `-sys` crates in this closure that own no native library.
///
/// # Why there is an exception at all, and why it is not a weakening
///
/// The `-sys` suffix is a CONVENTION, as this test's own documentation says,
/// and `renderdoc-sys` is the case that proves it is only that. `wgpu` enables
/// `wgpu-core/renderdoc` unconditionally on every non-wasm target, so this
/// crate is in the closure and cannot be turned off from here.
///
/// It is also, structurally, not what the suffix usually means: no `links`
/// key, no build script, no build dependency, and no `extern` block that links
/// anything. It is 682 lines of `#[repr(C)]` structs and function-pointer
/// typedefs for an API that `wgpu-hal` reaches through `libloading` at RUN
/// time -- the same shape `ash`'s loader has, and the same reason
/// `driver-vulkan` can call itself pure. Nothing about it costs a build
/// machine a compiler.
///
/// So the exception is granted BY NAME and then EARNED: [`sys_is_only_a_name`]
/// re-asks the three structural questions of every crate listed here, and a
/// `renderdoc-sys` that grew a build script tomorrow would fail this suite
/// twice over. That is a stronger arrangement than the alternative of dropping
/// the suffix check, which would silently admit every real `-sys` crate as
/// well.
const SYS_IN_NAME_ONLY: &[&str] = &["renderdoc-sys"];

/// Crates that declare `links` without owning a native library to link.
///
/// EMPTY, and the entry that stood here was deleted by the test below rather
/// than by anyone deciding it was settled — which is the arrangement working.
///
/// `js-sys`, `windows-sys` and `wasm-bindgen-shared` all appeared in this
/// closure at once when `driver` gained a `tensor-compiler` edge that took
/// `driver-api`'s DEFAULT features: `rpc` reaches `tarpc`, and tokio's
/// platform closure reaches all three. They were allow-listed here first,
/// with structural re-checks, on the reasoning that a browser-targeting crate
/// cannot filter its graph to the host triple.
///
/// That reasoning was sound and the premise was wrong. The three browser
/// tests were RED at the same moment — `getrandom` refuses to build for
/// `wasm32-unknown-unknown` — and the one-line `default-features = false` that
/// fixed those took the three crates out of the closure entirely. The
/// exceptions then failed as stale, by name, which is how they came out.
///
/// Worth keeping the shape for the next time: an allow-list whose staleness
/// check fires is an allow-list that told you the workaround was hiding
/// something.
const LINKS_WITHOUT_A_LIBRARY: &[&str] = &[];

/// Every allow-listed `-sys` crate is in the closure and owns nothing.
///
/// Two claims, and the second is the one that makes [`SYS_IN_NAME_ONLY`] an
/// exception rather than a hole.
///
/// A stale entry is a failure too. An allow-list that outlives its reason is
/// how a check quietly stops covering the case it was written for, and this
/// one costs nothing to keep honest -- the crate is either in the closure or it
/// is not.
#[test]
fn sys_is_only_a_name() {
    let meta = metadata().expect("cargo resolved the workspace");
    let packages = packages(&meta);
    let closure = closure(&meta, &packages);

    for want in SYS_IN_NAME_ONLY {
        let found = closure
            .iter()
            .filter_map(|id| packages.get(id))
            .find(|p| p.name == *want);
        let Some(p) = found else {
            panic!(
                "`{want}` is allow-listed and is no longer in `{ROOT}`'s closure, \
                 so the exception outlived its reason and should be deleted"
            );
        };
        assert!(
            p.links.is_none(),
            "`{want}` now owns a native library, which is exactly what the \
             suffix is supposed to mean -- the exception does not hold"
        );
        assert!(
            !p.build_script,
            "`{want}` now has a build script, so it is no longer a crate of \
             bare declarations"
        );
        assert!(
            p.build_deps.is_empty(),
            "`{want}` now has build dependencies: {:?}",
            p.build_deps
        );
    }
}

/// A `links` exception owns no library and cannot compile one.
///
/// The mirror of [`sys_is_only_a_name`] for [`LINKS_WITHOUT_A_LIBRARY`]. The
/// claim being allowed is narrow — *"this `links` key is a version lock"* — so
/// what is re-checked is the part that would make it false: a build script
/// with something to build WITH. `cc` or `bindgen` appearing among its build
/// dependencies is the whole of the risk, and it is the same signal
/// `nothing_this_crate_needs_to_build_compiles_c` uses everywhere else.
#[test]
fn links_is_only_a_version_lock() {
    let meta = metadata().expect("cargo resolved the workspace");
    let packages = packages(&meta);
    let closure = closure(&meta, &packages);

    for want in LINKS_WITHOUT_A_LIBRARY {
        let found = closure
            .iter()
            .filter_map(|id| packages.get(id))
            .find(|p| p.name == *want);
        let Some(p) = found else {
            panic!(
                "`{want}` is allow-listed and is no longer in `{ROOT}`'s \
                 closure, so the exception outlived its reason and should be \
                 deleted"
            );
        };
        assert!(
            p.links.is_some(),
            "`{want}` no longer declares `links`, so it does not need this \
             exception and the entry is dead weight"
        );
        for driver in &p.build_deps {
            assert!(
                !COMPILES_C.contains(driver),
                "`{want}` builds with `{driver}`, so its `links` key is no \
                 longer just a version lock and the exception does not hold"
            );
        }
    }
}

/// Every crate `driver-wgpu` needs to build, and none of them compiles C.
///
/// Dev-dependencies are excluded on purpose. They are what this suite needs to
/// ASK its questions -- a JSON parser to read the resolver's answer -- and none
/// of them is linked into anything a user builds. The constraint is about what
/// shipping this crate costs.
#[test]
fn nothing_this_crate_needs_to_build_compiles_c() {
    let meta = metadata().expect("cargo resolved the workspace");
    let packages = packages(&meta);
    let closure = closure(&meta, &packages);

    let mut suspect = Vec::new();
    let mut declares_links = BTreeSet::new();
    for id in &closure {
        let Some(p) = packages.get(id) else { continue };
        // Workspace members are held to the second claim below, not this one.
        // Their build scripts are ours and are gated; a crates.io dependency's
        // is not, and is the thing a careless `cargo add` brings in.
        if p.workspace {
            if p.links.is_some() || p.build_script {
                declares_links.insert(p.name);
            }
            continue;
        }
        if let Some(links) = p.links
            && !LINKS_WITHOUT_A_LIBRARY.contains(&p.name)
        {
            suspect.push(format!("`{}` owns the native library `{links}`", p.name));
        }
        // NOT "has a build script". Several crates in this closure have one and
        // none of them compiles anything -- `serde` and `proc-macro2` and
        // friends use theirs to probe the compiler version. Treating a build
        // script as the signal made the Vulkan version of this test fail on a
        // closure that is entirely pure Rust, which would have taught the next
        // reader to delete it. What a build script needs in order to compile C
        // is one of these, so that is what is asked for instead.
        for driver in &p.build_deps {
            if COMPILES_C.contains(driver) {
                suspect.push(format!("`{}` builds with `{driver}`", p.name));
            }
        }
        if p.name.ends_with("-sys") && !SYS_IN_NAME_ONLY.contains(&p.name) {
            suspect.push(format!("`{}` is a -sys crate by its name", p.name));
        }
    }

    assert!(
        suspect.is_empty(),
        "{} of the {} crates `{ROOT}` needs would want a C toolchain:\n  {}",
        suspect.len(),
        closure.len(),
        suspect.join("\n  ")
    );

    // The second claim, and the reason the first one is not the whole test.
    //
    // Four workspace crates in this closure have build scripts, and two of them
    // declare a native library -- which is exactly what a driver depending on
    // `kernels-cuda` would need. This crate depends on both and enables the
    // feature on neither, and each script does nothing without it.
    //
    // The evidence is not this list. It is that this suite RUNS on a machine
    // with no CUDA and no Metal: if either of those scripts compiled anything,
    // none of it would build here at all. The list is here so that a fifth
    // appearing is a decision somebody makes rather than one that happens.
    let want: BTreeSet<&str> = [
        // Here for its build script, which writes the carried-header list into
        // `OUT_DIR` and prints nothing but `rerun-if-changed`. Reached through
        // `model-ir`, which names every backend's table.
        //
        // This entry carried a `-new` suffix while two CUDA kernel crates
        // stood side by side in `crates/`: the ahead-of-time archive held the
        // plain name and the JIT replacement wore the suffix, and it was the
        // suffixed one `model-ir` resolved to. The archive was deleted at
        // `85c6c674b` and the suffix came off after it. This list caught both
        // moves rather than being told about them, which is the whole reason
        // it is pinned as a set instead of asserted to be small.
        //
        // AND IT IS GONE, at `bac4fa327`: a launch now names its file and its
        // symbol, so `model-ir` no longer reaches every backend's table to
        // find them and `kernels-cuda` is not in this closure at all. Third
        // move this entry has recorded, and the first that was an improvement
        // — one fewer crate whose build script a wgpu build waits on.
        // The Metal shader build, behind `native`, and macOS-only besides.
        // Same edge.
        "kernels-metal",
        // Walks `kernels/` and writes `include_str!` literals for the WGSL
        // tree, plus the entrypoint census. No C, no toolchain, and no product
        // on disk -- which is why THIS crate has no build script where
        // `driver-vulkan` has one: there is no `DEP_..._SPV_DIR` to relay,
        // because the shaders are in the rlib.
        "kernels-wgpu",
        // `model-compiler` STOOD HERE, described as *"content-hashes its own
        // `.rs` files to fingerprint the tracer"*. That script is `model-dsl`'s
        // now, and `model-dsl` is the AUTHORING surface -- a driver lowers a
        // traced form and never writes one, so it is not in this closure at
        // all. The entry left because the toolchain split, not because a
        // script was deleted, and the list noticing is the point of it.
        //
        // NOT `kernels-vulkan`, which `driver-vulkan`'s version of this list
        // has and this one does not: nothing in this closure names it. That is
        // worth an assertion rather than an omission, because the crate that
        // costs `glslc` being absent is a property of this backend and not an
        // accident of resolution.
    ]
    .into_iter()
    .collect();
    assert_eq!(
        declares_links, want,
        "a different set of workspace crates in `{ROOT}`'s closure has a build \
         script or a native library"
    );

    // A closure that came out empty would satisfy all of the above for the
    // wrong reason, and so would one that had lost `wgpu` -- at which point the
    // crate would be pure Rust by not talking to a GPU at all. `wgpu` is the
    // sanity check here for the same reason `ash` is next door, and it is a
    // stronger one: `ash` opens a loader that has to be installed, while
    // everything `wgpu` needs to COMPILE a shader is in the closure this test
    // just walked.
    assert!(
        closure.iter().any(|id| packages[id].name == "wgpu"),
        "the closure holds {} crates and `wgpu` is not among them, so this is \
         not measuring a WebGPU driver",
        closure.len()
    );
    // And `naga`, which is the claim that costs the most to keep: a WGSL front
    // end, in the dependency graph, with no toolchain behind it. If this ever
    // arrives as a `-sys` shim over some other compiler the assertion above
    // catches it, and this says it should be there at all.
    assert!(
        closure.iter().any(|id| packages[id].name == "naga"),
        "the shader front end is not in the closure, so `src/reflect.rs` is \
         reading modules some other way"
    );
}

/// What the resolver said about one package.
struct Package<'a> {
    name: &'a str,
    /// The native library it declares ownership of, if any.
    links: Option<&'a str>,
    /// Does it have a `build.rs`?
    build_script: bool,
    /// What its build script depends on, by name.
    build_deps: BTreeSet<&'a str>,
    /// Is it one of this repository's own crates?
    workspace: bool,
}

/// Ask cargo to resolve the workspace.
///
/// Every failure below here is loud. A check that cannot be made is not a check
/// that passed, and the Vulkan version of this test spent its first run
/// reporting success because the command it ran was rejected.
fn metadata() -> Option<serde_json::Value> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let out = std::process::Command::new(cargo)
        // No `--filter-platform`: unfiltered is every target, so a `-sys` crate
        // that only appears on one is still reported here. Passing the flag
        // with an empty value looks like it means that and does not -- cargo
        // exits with "target was empty", which the first version of this test
        // turned into a silent skip.
        //
        // `native` on, because `wgpu` is optional behind it and the closure
        // without it is not the one this test is for. The features that stay
        // OFF are the ones that matter: this does not enable `native` on
        // `kernels-cuda`, `kernels-metal` or `kernels-vulkan`, which is what
        // keeps nvcc and glslc out of the answer. `--all-features` would turn
        // all of those on and measure a build nothing performs.
        .args([
            "metadata",
            "--format-version",
            "1",
            "--features",
            "native",
            "--manifest-path",
        ])
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"))
        .output()
        .expect("cargo is what is running this test");
    assert!(
        out.status.success(),
        "cargo metadata failed, so nothing below was checked: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    Some(serde_json::from_slice(&out.stdout).expect("cargo metadata emits json"))
}

/// Every package the resolver knows, by id.
fn packages(meta: &serde_json::Value) -> BTreeMap<&str, Package<'_>> {
    let members: BTreeSet<&str> = meta["workspace_members"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .filter_map(serde_json::Value::as_str)
        .collect();
    meta["packages"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .filter_map(|p| {
            Some((
                p["id"].as_str()?,
                Package {
                    name: p["name"].as_str()?,
                    links: p["links"].as_str(),
                    // `targets` names a `custom-build` kind when and only when
                    // the package has a build script.
                    build_script: p["targets"].as_array().is_some_and(|ts| {
                        ts.iter().any(|t| {
                            t["kind"]
                                .as_array()
                                .is_some_and(|ks| ks.iter().any(|k| k == "custom-build"))
                        })
                    }),
                    // From the package's own manifest rather than the resolve
                    // graph, because the graph collapses a build-dependency
                    // into the same node and the kind is what matters here.
                    build_deps: p["dependencies"]
                        .as_array()
                        .map(Vec::as_slice)
                        .unwrap_or_default()
                        .iter()
                        .filter(|d| d["kind"].as_str() == Some("build"))
                        .filter_map(|d| d["name"].as_str())
                        .collect(),
                    workspace: members.contains(p["id"].as_str()?),
                },
            ))
        })
        .collect()
}

/// The packages `cargo tree` says [`ROOT`] alone needs.
///
/// # Why the resolve graph is not enough by itself
///
/// `cargo metadata` resolves features for the WHOLE workspace at once. A
/// dependency of `driver-api` that only `engine` asks for is still an edge out
/// of `driver-api`'s node, so a walk of that graph reports it as something this
/// crate needs -- and on the Vulkan side that failed the test for three crates
/// (`js-sys`, `wasm-bindgen-shared`, `windows-sys`) that a `cargo build -p` of
/// it never touches. They arrive under `tarpc`, which both crates turn off.
///
/// Over-approximating is the safe direction for a test like this, but an
/// over-approximation that cannot be satisfied is a test that gets deleted.
/// `cargo tree -p` resolves for one package, which is the question actually
/// being asked, so its answer is intersected with the walk below.
///
/// `--target all` because the walk it is narrowing is unfiltered by platform,
/// and narrowing it to THIS platform as well would quietly drop the
/// cross-compile half of the claim -- which for this crate is most of the
/// claim, since `wgpu` reaching a browser and an Android device without a
/// sysroot is the deployment story.
fn resolved() -> BTreeSet<String> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let out = std::process::Command::new(cargo)
        .args([
            "tree",
            "--features",
            "native",
            "--target",
            "all",
            "--edges",
            "normal",
            "--prefix",
            "depth",
            "--format",
            "{p}",
            "--manifest-path",
        ])
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"))
        .output()
        .expect("cargo is what is running this test");
    assert!(
        out.status.success(),
        "cargo tree failed, so the closure below is the unified one: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    // Depth-prefixed and not flat, so that `loom`'s subtree can be dropped
    // whole. It is here for the same reason the walk above has to name it --
    // `--target all` reports a `cfg(loom)` edge as if it were unconditional --
    // and dropping the edge alone is not enough: everything BELOW it stays in
    // this name set, and the set is what narrows the over-approximating walk.
    // `tracing-subscriber` arriving under `loom` is what un-narrowed `tarpc`'s
    // `windows-sys`, which no build of this crate has ever compiled.
    let text = String::from_utf8_lossy(&out.stdout);
    let mut names = BTreeSet::new();
    let mut pruned: Option<usize> = None;
    for line in text.lines() {
        let digits = line.chars().take_while(char::is_ascii_digit).count();
        let Ok(depth) = line[..digits].parse::<usize>() else {
            continue;
        };
        let Some(name) = line[digits..].split_whitespace().next() else {
            continue;
        };
        match pruned {
            Some(at) if depth > at => continue,
            _ => pruned = None,
        }
        if name == "loom" {
            pruned = Some(depth);
            continue;
        }
        names.insert(name.to_string());
    }
    // A parse that produced nothing would make the intersection empty and
    // every claim above vacuous.
    assert!(
        names.contains(ROOT),
        "cargo tree's output does not name the crate it was asked about"
    );
    names
}

/// Every package [`ROOT`] needs in order to build, by id.
///
/// Walked from the resolver's graph rather than read off manifests, because a
/// manifest states its own dependencies and the question is about all of them,
/// then narrowed by [`resolved`] -- see there for what the walk gets wrong.
fn closure<'a>(
    meta: &'a serde_json::Value,
    packages: &BTreeMap<&'a str, Package<'a>>,
) -> BTreeSet<&'a str> {
    let nodes: BTreeMap<&str, &serde_json::Value> = meta["resolve"]["nodes"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .filter_map(|n| Some((n["id"].as_str()?, n)))
        .collect();

    let root = packages
        .iter()
        .find(|(_, p)| p.name == ROOT)
        .map(|(id, _)| *id)
        .expect("the crate under test is in its own workspace");

    let mut seen = BTreeSet::new();
    let mut queue = vec![root];
    while let Some(id) = queue.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some(node) = nodes.get(id) else { continue };
        for dep in node["deps"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or_default()
        {
            // Dev and build kinds are dropped. A dev-dependency is not part of
            // what shipping this crate costs, and a build-dependency of some
            // OTHER crate is that crate's business -- what matters here is
            // whether the crate itself runs a build script, which is checked
            // per package above.
            let normal = dep["dep_kinds"]
                .as_array()
                .is_some_and(|ks| ks.iter().any(|k| k["kind"].is_null()));
            if !normal {
                continue;
            }
            if let Some(next) = dep["pkg"].as_str() {
                queue.push(next);
            }
        }
    }
    let resolved = resolved();
    seen.retain(|id| packages.get(id).is_some_and(|p| resolved.contains(p.name)));
    seen
}
