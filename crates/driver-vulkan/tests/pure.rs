//! Is this crate reachable without a C toolchain?
//!
//! `driver-vulkan` is meant to be pure Rust, and that is a property of the
//! whole dependency closure rather than of this crate's own source. One
//! `-sys` crate anywhere below it brings back everything the constraint exists
//! to avoid: a C compiler on every build machine, a vendored library to keep
//! current, a cross-compile that needs a sysroot, and a build that fails for
//! reasons no Rust error message explains.
//!
//! It is easy to hold and easy to lose. Adding a dependency is one line, its
//! own dependencies are not read, and nothing in a passing test suite would
//! notice -- the machine that added it already had the toolchain. So this
//! asks the resolver rather than trusting a reading of the manifests.
//!
//! # How the question is asked
//!
//! Three signals, because no single one is sufficient:
//!
//! * a `links` key, which is how a crate declares it owns a native library;
//! * a build script among the closure's members, which is where C gets
//!   compiled and is worth naming even when it is innocent;
//! * a `-sys` suffix, which is a convention rather than a rule but catches
//!   the ones that predate `links` being required.
//!
//! # What this test caught, and what it got wrong
//!
//! It fired the first time this crate took a dependency outside its own three
//! -- `driver-api`, for the `DeviceFacts` the engine's seam is handed -- and
//! named three crates: `js-sys`, `wasm-bindgen-shared` and `windows-sys`. All
//! three are real, all three arrive under `tarpc`, and `driver-api` needs
//! `tarpc` for exactly one trait. So the dependency moved behind a default-on
//! `rpc` feature there and this crate takes the crate without it.
//!
//! The test was also WRONG, in a way worth recording because the fix could
//! easily have been to weaken it. `cargo metadata` resolves features for the
//! whole workspace at once, so `tarpc` was an edge out of `driver-api`'s node
//! whether or not this crate asked for it, and the walk below reported it
//! either way. `cargo tree -p` resolves for one package; the two are
//! intersected now, and the control for that narrowing is that turning the
//! feature back on still fails this test.
//!
//! # Why Vulkan does not need one
//!
//! It would be reasonable to assume a GPU API needs a C library, and Vulkan
//! is where that assumption is least true. `ash` is a pure-Rust binding whose
//! `loaded` feature opens the loader with `libloading` at run time rather than
//! linking it, so nothing is compiled and nothing is required at build time --
//! a machine with no Vulkan at all still builds this crate, and only running
//! it needs a driver. That is why `tests/device.rs` is behind a feature and
//! the rest of the suite is not.

use std::collections::{BTreeMap, BTreeSet};

/// The crate whose closure is under test.
const ROOT: &str = "driver-vulkan";

/// The crates a build script reaches for when it is going to compile or find
/// a native library.
///
/// Not exhaustive and does not need to be: a crate that compiles C without any
/// of these is doing it by hand, and it will still have declared `links` or be
/// named `-sys`, both of which are checked separately.
const COMPILES_C: &[&str] = &["cc", "cmake", "bindgen", "pkg-config", "autotools"];

/// Every crate `driver-vulkan` needs to build, and none of them compiles C.
///
/// Dev-dependencies are excluded on purpose. They are what this suite needs
/// to ASK its questions -- a real model to lower, a JSON parser to read the
/// resolver's answer -- and none of them is linked into anything a user
/// builds. The constraint is about what shipping this crate costs.
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
        if let Some(links) = p.links {
            suspect.push(format!("`{}` owns the native library `{links}`", p.name));
        }
        // NOT "has a build script". Six crates in this closure have one and
        // none of them compiles anything -- `serde` and `proc-macro2` and
        // friends use theirs to probe the compiler version. Treating a build
        // script as the signal made this test fail on a closure that is
        // entirely pure Rust, which would have taught the next reader to
        // delete it. What a build script needs in order to compile C is one of
        // these, so that is what is asked for instead.
        for driver in &p.build_deps {
            if COMPILES_C.contains(driver) {
                suspect.push(format!("`{}` builds with `{driver}`", p.name));
            }
        }
        if p.name.ends_with("-sys") {
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
    // Workspace crates do have build scripts, and some declare a native
    // library -- which is exactly what a driver depending on `kernels-cuda`
    // would need. What matters is that none of them compiles C in the DEFAULT
    // build: `kernels-vulkan`'s runs `slangc` only under `native`, which is
    // this crate's `native` and not its default.
    //
    // The evidence is not this list. It is that this suite RUNS on a machine
    // with no CUDA and no Metal and no `slangc` in the default builds: if any
    // of those scripts compiled anything, none of it would build here at all.
    // The list is here so that a fourth appearing is a decision somebody makes
    // rather than one that happens.
    let want: BTreeSet<&str> = [
        // Reads `DEP_PIE_KERNELS_VULKAN_SPV_DIR` and re-emits it as a rustc
        // env. No compiler, no C.
        //
        // This row was REMOVED when the modules moved into `kernels-vulkan`'s
        // rlib, on the reasoning that a script relaying a module DIRECTORY
        // has no directory left to relay. That is true of the crate's own
        // code and of `tests/arena.rs` and `tests/device.rs`, which read the
        // modules as data -- and it is not true of `tests/planbench.rs`,
        // which still resolves them through
        // `option_env!("PIE_KERNELS_VULKAN_SPV_DIR")` and can only see that
        // env because this script sets it. So `crates/driver-vulkan/build.rs`
        // was never deleted and the row is back: this list is the crates
        // whose builds could compile C, and a script that exists is on it
        // whether or not the reason it exists has shrunk to one test.
        "driver-vulkan",
        // The CUDA table crate, reached from here by way of `model-ir`, whose
        // script writes into `OUT_DIR` and declares no `links`.
        //
        // This row has now LEFT and COME BACK, and both moves are the list
        // working. It left when `crates/kernels-cuda/build.rs` was deleted at
        // `bac4fa327` -- a launch stopped being reached through a generated
        // `static ROOT` and started naming its file and its symbol outright,
        // so nothing was generated and nothing generated it. The edge stayed;
        // only the fact went, which is the distinction this list is for.
        //
        // It is back because that crate has a build script again, for an
        // unrelated reason and a good one. NVRTC does no path resolution: it
        // matches `includeNames[]` against the literal string in an `#include`
        // directive, so a compile needs a list of SPELLINGS rather than a list
        // of files, and 187 lines of `include_str!` were being maintained
        // under a rule a person had to remember. The script walks `kernels/`
        // and `shim/` for the files, scans their own directives for the
        // spellings, and emits the same three `const` slices into `OUT_DIR`.
        // It is a directory walk and a string scan -- no `cc`, no `cmake`, no
        // `nvcc`, nothing compiled -- which is why it satisfies the FIRST
        // claim above and only has to be named here, in the list of workspace
        // scripts somebody decided on.
        //
        // The evidence remains what the paragraph above says it is: this suite
        // runs on a machine with no CUDA toolchain at all. If that script
        // compiled anything, this test would not build, let alone pass.
        "kernels-cuda",
        // The Metal shader build, behind `native`, and macOS-only besides.
        "kernels-metal",
        // slangc over the shader tree, and the `include_bytes!` table it
        // embeds, both behind `native`. This crate's own `native` turns it on;
        // its default half takes the crate with `default-features = false` and
        // gets the signature table with no toolchain behind it, which is what
        // keeps THIS test runnable on a machine with no `slangc`.
        "kernels-vulkan",
        // The WGSL table crate, reached from here the way `kernels-cuda` is:
        // by way of `model-ir`, which is the ONE join over every plane's
        // `*_CLAIMS` and so must name every plane's table crate whether or not
        // this driver fires on that plane.
        //
        // ADDED, and the date it should have been added is `c7bad6cf4` -- the
        // commit that gave `model-ir` this edge, which is the same commit that
        // landed `driver-wgpu`'s baker while `driver-vulkan` was outside the
        // workspace. So the list did not go stale slowly; it went stale at the
        // exact moment nothing was running this suite, and it stayed wrong for
        // as long as the crate could not be resolved. That is the failure this
        // whole file is an argument against, arriving in the file that makes
        // the argument.
        //
        // It satisfies the first claim for the plainest possible reason: its
        // `build.rs` walks `kernels/` and emits a `&[(&str, &str)]` of WGSL
        // SOURCE. WGSL is not a build product -- there is no `native` on that
        // crate and no toolchain behind one -- so unlike `kernels-vulkan`
        // there is not even a feature under which it could reach a compiler.
        // It declares `links = "pie_kernels_wgpu"`, which is why it is on this
        // list rather than invisible to it.
        "kernels-wgpu",
        // `model-compiler` STOOD HERE, described as *"content-hashes its own
        // `.rs` files to fingerprint the tracer"*. That script is `model-dsl`'s
        // now, and `model-dsl` is the AUTHORING surface -- a driver lowers a
        // traced form and never writes one, so it is not in this closure at
        // all. The entry left because the toolchain split, not because a
        // script was deleted. `driver-wgpu`'s copy of this list lost the same
        // row for the same reason.
    ]
    .into_iter()
    .collect();
    assert_eq!(
        declares_links, want,
        "a different set of workspace crates in `{ROOT}`'s closure has a build \
         script or a native library"
    );

    // A closure that came out empty would satisfy all of the above for the
    // wrong reason, and so would one that had lost `ash` -- at which point the
    // crate would be pure Rust by not talking to Vulkan at all.
    assert!(
        closure.iter().any(|id| packages[id].name == "ash"),
        "the closure holds {} crates and `ash` is not among them, so this is \
         not measuring a Vulkan driver",
        closure.len()
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
/// Every failure below here is loud. A check that cannot be made is not a
/// check that passed, and this one spent its first run reporting success
/// because the command it ran was rejected.
fn metadata() -> Option<serde_json::Value> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let out = std::process::Command::new(cargo)
        // No `--filter-platform`: unfiltered is every target, so a `-sys`
        // crate that only appears on one is still reported here. Passing the
        // flag with an empty value looks like it means that and does not --
        // cargo exits with "target was empty", which the first version of this
        // test turned into a silent skip. Adding a real `-sys` dependency then
        // did not fail it, which is how the skip was found.
        // `native` on, because `ash` is optional behind it and the closure
        // without it is not the one anybody ships. The features that stay OFF
        // are the ones that matter: this does not enable `native` on
        // `kernels-vulkan`, `kernels-cuda` or `kernels-metal`, which is what
        // keeps slangc and nvcc out of the answer. `--all-features` would turn
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
/// of `driver-api`'s node, so a walk of that graph reports it as something
/// this crate needs -- and adding `driver-api` here failed this test for three
/// crates (`js-sys`, `wasm-bindgen-shared`, `windows-sys`) that `cargo build
/// -p driver-vulkan` never touches. They arrive under `tarpc`, which this
/// crate turns off.
///
/// Over-approximating is the safe direction for a test like this, but an
/// over-approximation that cannot be satisfied is a test that gets deleted.
/// `cargo tree -p` resolves for one package, which is the question actually
/// being asked, so its answer is intersected with the walk below.
///
/// `--target all` because the walk it is narrowing is unfiltered by platform,
/// and narrowing it to THIS platform as well would quietly drop the
/// cross-compile half of the claim.
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
            //
            // An edge that exists only under `--cfg loom` is dropped with
            // them, and for the same reason. `waker` takes `loom` that way so
            // its register/commit race can be model-checked, and the closure
            // walked here is unfiltered by platform -- `--target all` turns
            // OFF cfg evaluation rather than answering it, so a cfg no build
            // ever sets reads as one every build sets. Nothing a user builds
            // passes that flag, and the alternative to naming it is filtering
            // the walk to this host, which would drop the cross-compile half
            // of the claim this test exists to make.
            let normal = dep["dep_kinds"].as_array().is_some_and(|ks| {
                ks.iter().any(|k| {
                    k["kind"].is_null() && !k["target"].as_str().is_some_and(|t| t.contains("loom"))
                })
            });
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

/// The half of this crate that builds without `native` contains no `unsafe`.
///
/// `lib.rs` says so in the comment explaining why the workspace lint table is
/// restated by hand instead of taken whole: the table forbids `unsafe_code`,
/// every `ash` entry point is unsafe, so the forbid is dropped and the
/// portable half is said to keep the guarantee a different way. It pointed at
/// a `tests/portable.rs` that has never existed. A claim with a citation to
/// nothing is worse than an uncited one, because the citation is what stops
/// the next reader checking.
///
/// So the claim is checked here, where the crate's other purity question
/// already lives. Not by a lint -- `#![forbid(unsafe_code)]` is a crate-wide
/// attribute and this is a per-module property -- but by reading the source,
/// which is what a reader would do.
///
/// The gates are read out of `lib.rs` rather than listed here, so a new module
/// is covered the day it is declared instead of the day someone remembers to
/// add it.
#[test]
fn the_half_that_builds_without_a_driver_contains_no_unsafe() {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let lib = std::fs::read_to_string(src.join("lib.rs")).expect("lib.rs is readable");

    // A module is portable unless the line before its declaration gates it.
    // Any gate not named here is unknown and treated as PORTABLE, so that a
    // new one shows up as a failure rather than as a silent exemption.
    //
    // IT DID. `native` was the only name until the Vulkan shell came back and
    // `device` was split out of it: the half that opens a card, builds a
    // pipeline and dispatches needs `ash` and nothing else, while the half
    // that walks a plan needs surfaces R3 deleted and does not compile at all.
    // Splitting them is what let a kernel fire before the walk exists. Both
    // are gates and both carry `unsafe`; the portable half is what is left
    // when neither is on, and that is still the claim this test makes.
    const GATES: [&str; 2] = ["native", "device"];
    let mut portable = Vec::new();
    let mut gated = 0usize;
    let lines: Vec<&str> = lib.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let Some(name) = line
            .trim()
            .strip_prefix("pub mod ")
            .and_then(|r| r.strip_suffix(';'))
        else {
            continue;
        };
        let gate = i > 0
            && GATES
                .iter()
                .any(|g| lines[i - 1].trim() == format!("#[cfg(feature = \"{g}\")]"));
        if gate {
            gated += 1;
        } else {
            portable.push(name.to_string());
        }
    }
    assert!(
        portable.len() >= 4 && gated >= 4,
        "read {} portable and {gated} gated modules out of lib.rs, which is not what it looks \
         like -- the parse is wrong and this test is measuring nothing",
        portable.len()
    );

    // Counted rather than assumed. The guard above is about the PARSE; this is
    // about the SCAN, and they are not the same claim -- a control that
    // emptied the list between them passed, which is exactly the shape of a
    // check that reports zero findings because it looked at zero files.
    let mut scanned = 0usize;
    let mut unsafes = Vec::new();
    // A MODULE IS ONE FILE OR A DIRECTORY OF THEM, and reading only the first
    // shape is how this scan reported a clean portable half while never
    // opening `baker/`'s seven files. `walk` is `walk.rs` beside `walk/`;
    // `baker` is a directory with a `mod.rs`. Both are scanned whole.
    let mut files: Vec<(String, std::path::PathBuf)> = Vec::new();
    for name in &portable {
        let one = src.join(format!("{name}.rs"));
        if one.is_file() {
            files.push((format!("{name}.rs"), one));
        }
        let dir = src.join(name);
        if dir.is_dir() {
            let mut stack = vec![dir];
            while let Some(d) = stack.pop() {
                for e in std::fs::read_dir(&d)
                    .expect("a readable module directory")
                    .flatten()
                {
                    let path = e.path();
                    if path.is_dir() {
                        stack.push(path);
                    } else if path.extension().is_some_and(|x| x == "rs") {
                        let shown = path
                            .strip_prefix(&src)
                            .unwrap_or(&path)
                            .to_string_lossy()
                            .into_owned();
                        files.push((shown, path));
                    }
                }
            }
        }
    }
    assert!(
        files.len() >= portable.len(),
        "{} portable modules resolved to {} file(s) -- a module that names \
         neither a file nor a directory means this scan is reading the wrong \
         tree",
        portable.len(),
        files.len(),
    );
    for (name, path) in &files {
        scanned += 1;
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{name}: {e}"));
        // Word-bounded, so `unsafely` in prose is not a finding. Comments and
        // strings are NOT excluded: this is a claim about what the file says,
        // and a false positive costs a rename while a false negative costs the
        // guarantee.
        for (n, line) in text.lines().enumerate() {
            if line
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .any(|w| w == "unsafe")
            {
                unsafes.push(format!("{name}:{}: {}", n + 1, line.trim()));
            }
        }
    }
    // THE CONTROL, and it used to read `scanned == portable.len()` because a
    // module was a file. `baker` is a directory of seven and `walk` is a file
    // beside a directory of six, so the honest form of the same claim is that
    // every file the resolve above found is a file this loop opened — a scan
    // that quietly read fewer than it resolved is the shape this guards.
    assert_eq!(
        scanned,
        files.len(),
        "resolved {} file(s) from {} portable module(s) and scanned {scanned}",
        files.len(),
        portable.len(),
    );
    assert!(
        unsafes.is_empty(),
        "the portable half names `unsafe` in {} place(s): {:#?}",
        unsafes.len(),
        unsafes
    );
}
