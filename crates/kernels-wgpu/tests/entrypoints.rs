//! The table's product, against the shader tree — and against `kernels-metal`.
//!
//! Invariant (1), which this backend inherits from both siblings:
//!
//! > every entrypoint in `kernels/` resolves to exactly one (row, axis point),
//! > and every (row, axis point) to exactly one entrypoint
//!
//! ## Why this one test does what Vulkan needs two and a Python script for
//!
//! `kernels-vulkan` splits the comparison in two hops: a `--check` mode of
//! `scripts/vulkan-kernel-audit.py` reads the `.comp` tree and writes
//! `entrypoints.generated.txt`, and a Rust test then compares that file to the
//! table. The split is forced — proving a declared variant COMPILES means
//! running `glslc`, and a `cargo test` that shells out to a Vulkan toolchain is
//! a test that fails on every box without one.
//!
//! There is no toolchain here. The directive parser is `kernels_wgpu::preproc`,
//! a library function; the shader tree is `kernels_wgpu::SOURCES`, embedded in
//! the rlib; and expanding a variant is `kernels_wgpu::entrypoint_source`,
//! which is Rust. So the tree half and the table half are both reachable from
//! one `#[test]`, with nothing to install and nothing to keep in step.
//!
//! `entrypoints.generated.txt` is gone, along with the `write_census` example
//! that wrote it. Nothing needed it to run — it was the artifact a human diffed
//! against `kernels-metal`'s and `kernels-vulkan`'s copies, a set difference in
//! a review rather than a number in a test log — and that is exactly what its
//! removal costs. What it did NOT carry is this crate's own shader-vs-table
//! check, which reads the tree directly and is unaffected.
//!
//! ## What this file does NOT prove
//!
//! That a module compiles. `naga` is the WGSL front end and it lives in `wgpu`,
//! which is a DEV-dependency of this crate for the reason `Cargo.toml` gives —
//! so the parse check lives in `tests/gpu.rs` beside the dispatches, and this
//! file stays a comparison between two descriptions.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every entrypoint the SHADER TREE declares, at any tier.
///
/// A tier is an additional variant of an entrypoint that already exists, so the
/// tier variants collapse into the same set the baselines produce — which is
/// the invariant `every_tier_has_a_baseline_beneath_it` checks separately.
fn from_the_shaders() -> BTreeSet<String> {
    kernels_wgpu::source::declared()
        .into_iter()
        .map(|(_, v)| v.entrypoint)
        .collect()
}

fn from_the_table() -> BTreeSet<String> {
    kernels_wgpu::entrypoints().into_iter().collect()
}

/// Every entrypoint a ROUTINE body names, read out of this crate's own source.
///
/// Once a family crosses to `routine!` its `kernel!` rows come off, and with
/// them the table's claim on its entrypoints. The entrypoints are still there
/// and still have to be reachable — a body names one in the `Fire` it
/// returns — so the checks below ask "table OR routine", not "table".
///
/// This reads text rather than the routine table because a body picks its
/// entrypoint while RUNNING, from the widths it is handed; there is no static
/// field to read. The scan is narrow — the `entrypoint:` field of a `Fire`
/// literal, comments stripped first — because the last raw-text scan this
/// crate grew matched the comment that explained the fix rather than the
/// code, and the first version of THIS one matched a function parameter.
///
/// A body that computed its entrypoint instead of spelling it would be
/// unreadable this way, so [`resolve`] returns `None` and this PANICS rather
/// than skipping it — a scan that quietly reads less is a scan that agrees
/// with everything.
fn from_the_routines() -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    let tables = entrypoint_tables();
    for file in routine_sources() {
        let text = std::fs::read_to_string(&file)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", file.display()));
        for (line, value) in fire_entrypoints(&text) {
            match resolve(&value, &tables) {
                Some(names) => found.extend(names),
                None => panic!(
                    "{}:{line}: `{value}` is neither a literal nor a lookup \
                     into a table of literals, so no static reading of this \
                     crate can say which entrypoints it fires",
                    file.display(),
                ),
            }
        }
    }
    found
}

/// The `entrypoint:` fields of `Fire` literals in one module, with the line.
///
/// Scoped to `Fire { .. }` rather than matching `entrypoint:` anywhere,
/// because the first version of this scan flagged `fn spell(entrypoint: &str)`
/// — a parameter, not a field. Brace depth is counted from the `Fire {` so a
/// literal nested inside one still ends in the right place.
fn fire_entrypoints(text: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut depth: Option<i32> = None;
    for (n, line) in text.lines().enumerate() {
        let code = line.split_once("//").map_or(line, |(before, _)| before);
        if depth.is_none() && code.contains("Fire {") {
            depth = Some(0);
        }
        let Some(level) = depth.as_mut() else {
            continue;
        };
        if let Some((_, rest)) = code.split_once("entrypoint:") {
            out.push((n + 1, rest.trim().trim_end_matches(',').to_owned()));
        }
        *level += i32::try_from(code.matches('{').count()).expect("few braces");
        *level -= i32::try_from(code.matches('}').count()).expect("few braces");
        if *level <= 0 {
            depth = None;
        }
    }
    out
}

/// Every `routine!(name)` declared under `dir`.
///
/// `kernels-metal` is not a dependency of this crate — these parity checks
/// read its SOURCE, so that they run on a machine with no Metal toolchain —
/// so its crossings are read the same way its rows are.
fn routine_names(dir: &std::path::Path) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let entries =
        std::fs::read_dir(dir).unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("a readable entry").path();
        if path.extension().is_none_or(|e| e != "rs") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("a readable module");
        for line in text.lines() {
            let code = line.split_once("//").map_or(line, |(before, _)| before);
            let mut rest = code;
            while let Some((_, after)) = rest.split_once("routine!(") {
                if let Some((name, tail)) = after.split_once(')') {
                    let name = name.trim();
                    if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                        out.insert(name.to_owned());
                    }
                    rest = tail;
                } else {
                    break;
                }
            }
        }
    }
    out
}

/// This crate's Rust sources, where routine bodies live.
fn routine_sources() -> Vec<std::path::PathBuf> {
    let src = manifest().join("src");
    let mut out = Vec::new();
    let entries =
        std::fs::read_dir(&src).unwrap_or_else(|e| panic!("cannot read {}: {e}", src.display()));
    for entry in entries {
        let path = entry.expect("a readable entry").path();
        if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    assert!(
        out.len() > 5,
        "expected this crate's modules, found {out:?}"
    );
    out
}

/// The entrypoints one `Fire`'s `entrypoint:` expression can name, or `None`
/// if this reading cannot tell.
///
/// Two shapes, and only two. A literal names itself. `TABLE[point(..)]` names
/// every string in `TABLE`, because the index picks an affine or head point
/// and every point on the axis is a real entrypoint — which is the same claim
/// `every_stated_file_carries_the_rows_own_entrypoints` makes about rows.
fn resolve(value: &str, tables: &BTreeMap<String, Vec<String>>) -> Option<Vec<String>> {
    if let Some(rest) = value.strip_prefix('"') {
        let name = rest.split('"').next()?;
        return Some(vec![name.to_owned()]);
    }
    let (name, _) = value.split_once('[')?;
    tables.get(name.trim()).cloned()
}

/// Every `static`/`const` `NAME: [&str; N] = [..]` in this crate, by name.
///
/// These are the entrypoint axes the bodies index — `EMBED_GATHER`,
/// `PAGED_DECODE`, `AFFINE_QMM` and the rest.
fn entrypoint_tables() -> BTreeMap<String, Vec<String>> {
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for file in routine_sources() {
        let text = std::fs::read_to_string(&file).expect("a readable module");
        let mut open: Option<String> = None;
        for line in text.lines() {
            let code = line.split_once("//").map_or(line, |(before, _)| before);
            if open.is_none() {
                let trimmed = code.trim();
                let Some(rest) = trimmed
                    .strip_prefix("static ")
                    .or_else(|| trimmed.strip_prefix("const "))
                else {
                    continue;
                };
                let Some((name, tail)) = rest.split_once(':') else {
                    continue;
                };
                if !tail.contains("&str") || !tail.contains('[') {
                    continue;
                }
                open = Some(name.trim().to_owned());
                continue;
            }
            let name = open.clone().expect("an open table");
            if code.contains(']') && !code.contains('"') {
                open = None;
                continue;
            }
            for piece in code.split('"').skip(1).step_by(2) {
                out.entry(name.clone()).or_default().push(piece.to_owned());
            }
        }
    }
    assert!(
        out.len() > 10,
        "expected this crate's entrypoint axes, found {:?}",
        out.keys().collect::<Vec<_>>(),
    );
    out
}

/// A retired family's stated `ENTRYPOINTS` are exactly what its bodies FIRE.
///
/// `RETIRED` is hand-written, and it is what `entrypoints()` returns for a
/// family whose rows are gone — so every sweep keyed on `entrypoints()` covers
/// what this list says rather than what the shaders are. A row could not drift
/// that way: its `axes` GENERATED its entrypoints. A typo here, or a body
/// changed without the list, silently moves the sweeps off the real shader,
/// which is the same silence the retirement already caused once.
///
/// Read out of the family's own module: the `entrypoint:` field of each `Fire`
/// literal, resolved through `TABLE[point(..)]` where the body indexes an
/// axis. A body that COMPUTED its entrypoint would be unreadable this way and
/// [`resolve`] panics rather than skipping it.
#[test]
fn a_retired_familys_stated_entrypoints_are_what_its_bodies_fire() {
    // One line per retired family, mirroring `lib.rs`'s `RETIRED`.
    let families: &[(&str, &[&str])] = &[
        ("sample.rs", kernels_wgpu::sample::ENTRYPOINTS),
        ("ptir.rs", kernels_wgpu::ptir::ENTRYPOINTS),
    ];

    assert_eq!(
        families.len(),
        kernels_wgpu::retired_rows().len(),
        "a family was retired without being checked here, which is how the \
         stated list stops being compared to the bodies at all",
    );

    for (module, stated) in families {
        let path = manifest().join("src").join(module);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let tables = entrypoint_tables();
        let mut fired = BTreeSet::new();
        for (line, value) in fire_entrypoints(&text) {
            match resolve(&value, &tables) {
                Some(names) => fired.extend(names),
                None => panic!(
                    "{module}:{line}: `{value}` is neither a literal nor a \
                     lookup into a table of literals, so no static reading of \
                     this crate can say which entrypoints it fires",
                ),
            }
        }
        let want: BTreeSet<String> = stated.iter().map(|s| (*s).to_owned()).collect();
        assert_eq!(
            fired, want,
            "`{module}`'s bodies fire a different set than its `ENTRYPOINTS` \
             states, and `ENTRYPOINTS` is what every sweep keyed on \
             `entrypoints()` will walk",
        );
    }
}

#[test]
fn the_table_names_exactly_what_the_shaders_instantiate() {
    let shaders = from_the_shaders();
    let mut table = from_the_table();
    table.extend(from_the_routines());

    let undeclared: Vec<_> = shaders.difference(&table).collect();
    assert!(
        undeclared.is_empty(),
        "{} entrypoints exist in kernels/ that no row declares. A new \
         instantiation needs a row, or a point on an existing row's axis:\n{:#?}",
        undeclared.len(),
        undeclared,
    );

    let phantom: Vec<_> = table.difference(&shaders).collect();
    assert!(
        phantom.is_empty(),
        "{} entrypoints are declared that no shader instantiates. An axis whose \
         product over-generates is the usual cause — see `sdpa_paged_decode`, \
         which lists its tails for exactly this reason:\n{:#?}",
        phantom.len(),
        phantom,
    );
}

/// The census `entrypoints.generated.txt` recorded, and the `write_census`
/// example that produced it, are both gone.
///
/// It was only ever a reviewer's convenience — the test module header said as
/// much — and the two things it enabled have gone different ways. The table's
/// product against THIS crate's shaders is checked above and loses nothing: the
/// shader tree is embedded in the rlib, so `source::declared()` reads it
/// directly. The set difference against `kernels-metal` is what is actually
/// lost, and it has no in-process replacement, since that crate does not build
/// off macOS and its census needs a C preprocessor to expand at all.

/// Two rows claiming one entrypoint would make `sig_in` order-dependent, and
/// the set comparison above cannot see it: a duplicate is absorbed by the set.
///
/// The shader side of the same question is caught earlier and harder — a
/// duplicate `pie:instantiate` is two variants under one name, so `build.rs`
/// refuses it rather than let whichever the tree walk reached second win.
#[test]
fn no_two_rows_claim_the_same_entrypoint() {
    let mut seen: BTreeMap<String, &str> = BTreeMap::new();
    for row in kernels_wgpu::KERNELS {
        for name in row.entrypoints() {
            if let Some(other) = seen.insert(name.clone(), row.name) {
                panic!("`{name}` is claimed by both `{other}` and `{}`", row.name);
            }
        }
    }
}

/// The row count is `kernels-metal`'s, and that is the point rather than a
/// coincidence: this backend's coverage is DEFINED as its sibling's, so the two
/// tables are comparable row for row and a divergence is a statement somebody
/// made rather than a drift nobody noticed.
///
/// Change it here when a kernel is added, deliberately — and when you do, say
/// in the same diff whether Metal grew one too, because a number that moves on
/// one side alone is exactly the fact this assertion exists to surface.
#[test]
fn the_table_is_one_hundred_kernels_over_four_hundred_and_eighty_one_entrypoints() {
    // Rows PLUS retired rows: `refactor-bigplan.md` §7 empties the table
    // family by family, and coverage is what the two together name. The
    // hundred is the invariant; which side of the crossing a kernel sits on
    // is not.
    assert_eq!(
        kernels_wgpu::KERNELS.len() + kernels_wgpu::retired_rows().len(),
        100
    );
    assert_eq!(kernels_wgpu::entrypoints().len(), 481);
}

/// The parity with `kernels-metal` above was checked here, entrypoint for
/// entrypoint, by diffing the two crates' committed censuses.
///
/// Both are deleted, and this comparison has no in-process replacement:
/// `kernels-metal` does not build off macOS, so a dev-dependency is not
/// available, and its census cannot be parsed the way this crate's is — the
/// Metal axis product is written as `instantiate_*` macros that only a C
/// preprocessor expands. The count above is what remains of the claim, and it
/// is strictly weaker: two tables can agree on 481 while disagreeing about
/// which 481.

/// Every entrypoint resolves through the public lookup `model-ir` uses,
/// at every point of every axis.
///
/// `sig_in` tries exact matches first and axis matches second, so a base that
/// shadows a sibling's point surfaces here and nowhere else.
#[test]
fn every_entrypoint_resolves_through_sig_in() {
    let retired = kernels_wgpu::retired();
    for name in kernels_wgpu::entrypoints() {
        // A crossed family resolves through `driver-wgpu`'s `arm_for` instead.
        // What must hold is that the entrypoint has SOME owner, not that the
        // owner is a row.
        if retired.contains(&name.as_str()) {
            continue;
        }
        assert!(
            kernels::sig_in(kernels_wgpu::KERNELS, &name).is_some(),
            "`{name}` resolves to no row",
        );
    }
}

/// A row that names a FILE names one that exists.
///
/// Metal can leave this to the runtime shader compiler, which fails at model
/// load with the path in hand. So could this backend — `naga` is a runtime
/// compiler too. But a row pointing at a shader nobody wrote is a pipeline the
/// shell asks for and cannot create, one layer away from the row that named it,
/// and the file list is right here.
#[test]
fn every_stated_file_exists() {
    for row in kernels_wgpu::KERNELS {
        let Some(file) = row.file else { continue };
        assert!(
            kernels_wgpu::source(file).is_some(),
            "`{}` names `kernels/{file}`, which the embedded tree does not have",
            row.name,
        );
    }
}

/// A row's stated file CARRIES the entrypoints that row generates.
///
/// # One word short
///
/// `every_stated_file_exists` checks the path resolves and stops there, so a
/// row may name a real shader that has nothing to do with it. `moe`'s
/// `qmv_routed` and `qmv_routed_bias` named `quant/qmv.wgsl`, which exists and
/// contains the string `qmv_routed` exactly zero times — every one of their
/// entrypoints is instantiated in `moe/qmv_routed.wgsl`.
///
/// `kernels-metal` shipped the same mistake one plane up and had to fix it:
/// *"three attention shaders were named after their routines, not their
/// files"*. There the module name is what a `Fire` DISPATCHES, so it is a
/// pipeline that cannot be created; here `file` is documentation today,
/// because `source::entrypoint_source` resolves by ENTRYPOINT and never asks
/// a row where its shader lives.
///
/// **Which is exactly why it has to be checked now.** A routine states its
/// module, and the obvious way to write one is to copy the row's `file`. A
/// wrong `file` that costs nothing today is a wrong `module` at the moment
/// the family crosses.
///
/// Rows that state no file are skipped, as above: an unfilled row names no
/// shader and claims nothing.
#[test]
fn every_stated_file_carries_the_rows_own_entrypoints() {
    let mut checked = 0usize;
    let mut wrong: Vec<String> = Vec::new();
    for row in kernels_wgpu::KERNELS {
        let Some(file) = row.file else { continue };
        let Some(text) = kernels_wgpu::source(file) else {
            continue; // `every_stated_file_exists` is what reports this.
        };
        for name in row.entrypoints() {
            checked += 1;
            if !text.contains(&format!("pie:instantiate {name}")) {
                wrong.push(format!(
                    "`{}` states `{file}`, which does not instantiate `{name}`",
                    row.name
                ));
            }
        }
    }
    // ASSERTED, not bounded. The first version said `> 200` because I
    // guessed; the truth is 196, and a guessed bound is a number that stops
    // meaning anything the moment it is satisfied. A row gaining or losing a
    // `file` moves this, and moving it should be a deliberate edit.
    assert_eq!(
        checked, 196,
        "the rows that state a file generate {checked} entrypoints, not 196"
    );
    assert!(
        wrong.is_empty(),
        "a row names a shader that does not carry its entrypoints. Harmless \
         while `file` is documentation, and a wrong `module` the moment the \
         family crosses -- see this test's own docs. {} of {checked}:\n  {}",
        wrong.len(),
        wrong.join("\n  ")
    );
}

/// Every `@tier` directive names an entrypoint that also has a baseline.
///
/// This is the whole of the backward-compatibility guarantee, and it is a test
/// rather than a convention because the failure it prevents is invisible until
/// a specific device runs a specific model: a tiered variant with no baseline
/// is an entrypoint that resolves on the author's GPU and on no other.
///
/// `build.rs` asserts the same thing and fails the build. This runs anyway,
/// because a build-script assertion is only as good as the last time the build
/// script ran, and cargo caches it.
#[test]
fn every_tier_has_a_baseline_beneath_it() {
    let declared = kernels_wgpu::source::declared();

    let baseline: BTreeSet<&str> = declared
        .iter()
        .filter(|(_, v)| v.tier == kernels_wgpu::Capability::Baseline)
        .map(|(_, v)| v.entrypoint.as_str())
        .collect();

    let orphans: Vec<String> = declared
        .iter()
        .filter(|(_, v)| v.tier != kernels_wgpu::Capability::Baseline)
        .filter(|(_, v)| !baseline.contains(v.entrypoint.as_str()))
        .map(|(file, v)| {
            format!(
                "kernels/{file}:{} `{}` @{}",
                v.line,
                v.entrypoint,
                v.tier.tag()
            )
        })
        .collect();

    assert!(
        orphans.is_empty(),
        "a tier is an ADDITIONAL variant of an entrypoint that already exists, \
         never a new entrypoint and never a replacement:\n{}",
        orphans.join("\n"),
    );
}

/// A tier never invents an entrypoint the table does not name.
///
/// The set comparison at the top of this file already covers it, since it folds
/// tiers into the same set. Stated separately anyway, because the day somebody
/// changes that fold is the day the coverage silently widens.
#[test]
fn a_tier_never_widens_the_four_hundred_and_eighty() {
    let table = from_the_table();
    for (file, variant) in kernels_wgpu::source::declared() {
        if variant.tier == kernels_wgpu::Capability::Baseline {
            continue;
        }
        assert!(
            table.contains(&variant.entrypoint),
            "kernels/{file}:{}: `{}` @{} is an entrypoint no row names",
            variant.line,
            variant.entrypoint,
            variant.tier.tag(),
        );
    }
}

/// Baseline is unsuffixed, so a driver that has never heard of a tier finds the
/// right variant knowing only the entrypoint name.
#[test]
fn baseline_variants_are_unsuffixed() {
    use kernels_wgpu::Capability;
    assert_eq!(
        Capability::Baseline.variant("rms_single_row_bfloat16"),
        "rms_single_row_bfloat16",
    );
    assert_eq!(
        Capability::Fp16.variant("rms_single_row_bfloat16"),
        "rms_single_row_bfloat16.fp16",
    );
    assert_eq!(
        *Capability::PREFERENCE.last().expect("three tiers"),
        Capability::Baseline,
    );
    assert!(Capability::Baseline.requires().is_empty());
}

/// Every `// pie:instantiate` names an entrypoint the table declares.
///
/// The reverse of `the_table_names_exactly_what_the_shaders_instantiate`, per
/// FILE rather than per set, so the failure message names the shader and the
/// line rather than a name in a list of hundreds.
#[test]
fn every_instantiated_variant_is_one_the_table_declares() {
    let mut known = from_the_table();
    known.extend(from_the_routines());
    for (file, text) in kernels_wgpu::SOURCES {
        let variants = kernels_wgpu::instantiations(text)
            .unwrap_or_else(|why| panic!("kernels/{file}: {why}"));
        for variant in variants {
            assert!(
                known.contains(&variant.entrypoint),
                "kernels/{file}:{}: `{}` is instantiated but neither a row nor \
                 a routine body names it",
                variant.line,
                variant.entrypoint,
            );
        }
    }
}

/// A shader that calls a row UNSTATED is making a claim about the TABLE.
///
/// These files carry a lot of prose about binding order, and the reason is
/// always the same: an UNSTATED row (`kernels::KernelSig::operands` empty)
/// gives `bindings()` nothing to answer with, so the slot placement in the
/// shader is a convention agreed with the sibling backends rather than a
/// consequence of the row. That comment is load-bearing — it is the only place
/// the convention is written down — and it goes silently false the day someone
/// STATES the row, which is exactly what happened to `sdpa_paged_tiled`.
///
/// So the prose is held against the table. Every backticked identifier in a
/// sentence containing "UNSTATED" that names a row must name an unstated one.
/// Nothing here checks the converse: a shader is free not to mention that a row
/// is stated, because a stated row's layout is `bindings()`'s answer and the
/// tests that read it are elsewhere in this file.
#[test]
fn no_shader_calls_a_stated_row_unstated() {
    // Prose wraps, and the row name is usually on the line after the word.
    const WRAP: usize = 2;
    let mut wrong = Vec::new();
    for (file, text) in kernels_wgpu::SOURCES {
        let lines: Vec<&str> = text.lines().collect();
        for (n, line) in lines.iter().enumerate() {
            if !line.contains("UNSTATED") {
                continue;
            }
            let end = (n + WRAP + 1).min(lines.len());
            for named in lines[n..end].iter().flat_map(|l| backticked(l)) {
                let Some(sig) = kernels_wgpu::sig(&named) else {
                    continue;
                };
                if !sig.operands.is_empty() {
                    wrong.push(format!(
                        "kernels/{file}:{}: `{named}` is called UNSTATED, but the \
                         table states {} operands for it",
                        n + 1,
                        sig.operands.len(),
                    ));
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} shader comment(s) describe a row the table has since stated. The \
         binding order they explain is now `kernels_wgpu::bindings()`'s answer, \
         so the comment should say so — or, if the sentence names a stated row \
         only in passing, move that mention out of the UNSTATED sentence.\n{}",
        wrong.len(),
        wrong.join("\n"),
    );
}

/// A `pie_*` helper named in shader prose either exists or is one of three that
/// deliberately does not.
///
/// Three names in this tree describe an API that CANNOT be written: naga
/// allows a pointer parameter only in the `private` and `function` address
/// spaces, so `pie_load_bf16(&queries, i)` and its two siblings are the shape
/// the code would take in a language with pointers, named in prose precisely
/// so a reader understands why the real function takes a word and an index
/// instead. `tests/gpu.rs` guards the first two against reintroduction by
/// scanning for `fn pie_load_bf16(` — that guard is the reason the names have
/// to keep meaning what they mean.
///
/// Which makes them a trap. A reader who greps `pie_store_bf16`, finds nothing,
/// and helpfully "fixes" the eight call sites to say `pie_bf16_into` has
/// destroyed the point of the paragraph; a reader who adds a FOURTH such name
/// by typo has written a comment that points at nothing and looks exactly like
/// the other three. So the list is closed, here, with reasons.
///
/// Anything else must exist. `every_module_parses_and_validates` in
/// `tests/gpu.rs` proves that naga accepted all 481 expanded modules, so a
/// `pie_*` name appearing in the CODE of one is a name that is defined — which
/// covers `fn`s, `var<workgroup>`s and constants without having to parse a
/// declaration. (`pie_partials`, a `var<workgroup>` in
/// `common/reduce.inc.wgsl`, is why that matters: the first draft of this check
/// scanned for `fn pie_` and reported it as missing.)
///
/// The existence half therefore reads the EXPANDED modules and not the files.
/// A file is a template — `//#if defined(PIE_TILED)` and eleven other switches
/// — and a name defined only in a branch no declared variant selects is a name
/// naga never sees and no fire can call. Reading the files would call that
/// "defined". Reading what the preprocessor produced calls it what it is.
/// The prose half still reads the FILES, because a comment in an unselected
/// branch is one a reader still reaches.
#[test]
fn every_pie_helper_named_in_shader_prose_exists_or_is_listed_as_hypothetical() {
    /// Named in prose, absent from the tree, and meant to be.
    const HYPOTHETICAL: &[(&str, &str)] = &[
        (
            "pie_load_bf16",
            "the pointer-taking loader naga refuses; the real one is \
             `pie_bf16_at(word, i)` and takes a word already loaded",
        ),
        (
            "pie_store_bf16",
            "its store counterpart; the real one is `pie_bf16_into(word, i, x)` \
             and RETURNS the word, which is what makes the sharing question \
             visible at the call site",
        ),
        (
            "pie_affine_dequant",
            "the pointer-taking dequantiser, for the same reason; \
             `common/affine.inc.wgsl` takes words instead",
        ),
    ];

    let mut in_code = BTreeSet::new();
    for (_, variant) in kernels_wgpu::source::declared() {
        let Ok(source) = kernels_wgpu::entrypoint_source(&variant.entrypoint, variant.tier) else {
            continue; // `every_module_parses_and_validates` owns that failure.
        };
        for line in source.lines() {
            let code = line.split_once("//").map_or(line, |(before, _)| before);
            for name in pie_names(code) {
                in_code.insert(name);
            }
        }
    }

    let mut in_prose: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (file, text) in kernels_wgpu::SOURCES {
        for (n, line) in text.lines().enumerate() {
            let Some((_, prose)) = line.split_once("//") else {
                continue;
            };
            for name in backticked(prose) {
                if name.starts_with("pie_") {
                    in_prose
                        .entry(name)
                        .or_default()
                        .push(format!("kernels/{file}:{}", n + 1));
                }
            }
        }
    }

    let listed: BTreeSet<&str> = HYPOTHETICAL.iter().map(|(n, _)| *n).collect();
    let missing: Vec<String> = in_prose
        .iter()
        .filter(|(name, _)| !in_code.contains(*name) && !listed.contains(name.as_str()))
        .map(|(name, at)| format!("`{name}` — named at {}", at.join(", ")))
        .collect();
    assert!(
        missing.is_empty(),
        "{} name(s) in shader prose that nothing in the tree defines. Either \
         the helper was renamed and the comment was not, or the name is a \
         deliberate hypothetical like `pie_load_bf16` — in which case add it \
         to HYPOTHETICAL with the reason it cannot be written.\n{}",
        missing.len(),
        missing.join("\n"),
    );

    // And the list closes in both directions: a hypothetical that has since
    // been written is a comment claiming something is impossible next to the
    // thing itself.
    for (name, why) in HYPOTHETICAL {
        assert!(
            !in_code.contains(*name),
            "`{name}` is listed as impossible to write ({why}), and the tree \
             now has one. Drop the entry and rewrite the prose that cites it.",
        );
        assert!(
            in_prose.contains_key(*name),
            "`{name}` is listed as a hypothetical no shader mentions any more. \
             Drop the entry.",
        );
    }
}

/// Every `pie_*` identifier in a stretch of shader CODE.
fn pie_names(code: &str) -> Vec<String> {
    let mut found = Vec::new();
    let bytes = code.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_alphanumeric() && bytes[i] != b'_' {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
            i += 1;
        }
        let word = &code[start..i];
        if word.starts_with("pie_") && (start == 0 || !bytes[start - 1].is_ascii_digit()) {
            found.push(word.to_string());
        }
    }
    found
}

/// The leading identifier of every `` `span` `` in a line of shader prose.
///
/// The span, not the identifier, is what a comment writes: a citation like
/// `` `pie_bf16_at(x[i >> 1u], i)` `` is one span, and `pie_bf16_at` is the
/// name in it. Spans that do not START with an identifier — `` `...Weak` ``,
/// `` `&mut T` `` — yield nothing, which is right: they are not naming
/// anything to check.
fn backticked(line: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut rest = line;
    while let Some(open) = rest.find('`') {
        rest = &rest[open + 1..];
        let Some(close) = rest.find('`') else { break };
        let ident: String = rest[..close]
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if !ident.is_empty() {
            found.push(ident);
        }
        rest = &rest[close + 1..];
    }
    found
}

/// The embedded tree is the tree on disk.
///
/// `build.rs` walks `kernels/` and writes `include_str!` literals. A file it
/// missed — a new subdirectory, an extension typo — is a shader that exists,
/// reads correctly and is not in the binary, which is a "no such variant" at a
/// model load a long way from the file that was added.
#[test]
fn the_embedded_tree_is_every_file_on_disk() {
    let root = manifest().join("kernels");
    let mut on_disk: BTreeSet<String> = BTreeSet::new();
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("a readable kernels/ directory") {
            let path = entry.expect("a readable entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "wgsl") {
                on_disk.insert(
                    path.strip_prefix(&root)
                        .expect("under kernels/")
                        .to_string_lossy()
                        .replace('\\', "/"),
                );
            }
        }
    }

    let embedded: BTreeSet<String> = kernels_wgpu::SOURCES
        .iter()
        .map(|(path, _)| (*path).to_owned())
        .collect();

    assert_eq!(
        embedded, on_disk,
        "the embedded tree and the directory disagree; `build.rs` walks \
         `kernels/**/*.wgsl` and something is not being seen",
    );
}

/// Every row states the same FACTS about its kernel as the `kernels-metal` row
/// of the same name — the grid it runs under, and every other scalar the macro
/// can carry.
///
/// The entrypoint-set comparison above is about coverage — which kernels exist
/// — and it is blind to what a row SAYS about the ones it has. That gap is not
/// hypothetical. `kernels-metal` grew `KernelSig::heads_param` (and
/// `head_param` before it) to fix a gemma-4 miswrite; `kernels-vulkan`'s port
/// carried neither field across, so the two tables agreed on all 480 names
/// while disagreeing about the grid two of them run under. That is a WRONG
/// answer rather than a missing one: an undershot `PerHead` grid writes
/// nothing, the gap reads back as the zeros the pool was born with, and the
/// fire completes.
///
/// The equivalent check on the Vulkan side found a third gap —
/// `vnorm_single_row`'s `grid_param` — on its first run, which is the argument
/// for checking the CATEGORY rather than the fields that prompted it.
///
/// `kernels-metal` does not build off macOS, so this cannot compare the two
/// `KERNELS` arrays as values. It reads the sibling's SOURCE and scrapes the
/// field off the row — coarse, but the fields are written by a macro whose
/// shape all three crates share, and being coarse in the direction of "diff the
/// text" is the safe direction for a check whose whole job is to notice that
/// somebody edited one side only.
///
/// **No exception list.** One is how the next real divergence gets waved
/// through.
#[test]
fn every_row_states_the_same_facts_the_sibling_table_does() {
    // The one row the two tables disagree about, and it is in the fleet's
    // ledger already: `shader_backends_agree.rs`'s `DRIFTED["route_gather"]`
    // says `driver-vulkan` does not read `rows_param` and this backend does.
    // Named here rather than waived, so a SECOND disagreement still fails.
    const DIFFERS: &[&str] = &["route_gather"];

    let ours = scrape_fields(&manifest().join("src"));
    let theirs = scrape_fields(&manifest().join("../kernels-vulkan/src"));

    assert_eq!(
        ours.len(),
        kernels_wgpu::KERNELS.len(),
        "the scrape read {} rows out of this crate's own {} — a text-based \
         check that silently reads nothing is a check that passes",
        ours.len(),
        kernels_wgpu::KERNELS.len(),
    );
    // BOTH tables are shrinking now, on their own schedules: this crate has
    // retired `sample` and `kernels-metal` has retired `sample` and `ptir`. So
    // neither count can be derived from the other, and what is checked instead
    // is that the scrape read a whole table rather than nothing — the failure a
    // text-based check invites — with the hundred as the ceiling it started
    // from.
    assert!(
        theirs.len() > 50 && theirs.len() <= 100,
        "the scrape read {} rows out of `kernels-vulkan`, which has been \
         between 100 and the handful Stage 3 will leave it at. A number \
         outside that is a scrape that stopped reading, not a table that \
         moved",
        theirs.len(),
    );
    for name in kernels_wgpu::retired_rows() {
        assert!(
            !ours.contains_key(*name),
            "`{name}` is listed as retired here and this crate still has a \
             row for it, so the entry is wrong in the direction that matters: \
             `entrypoints()` would name it twice",
        );
    }

    let mut wrong = Vec::new();
    for (name, want) in &theirs {
        let Some(got) = ours.get(name) else {
            // A row we have retired and they have not. Not drift: the two
            // tables empty family by family and need not be in step.
            if !kernels_wgpu::retired_rows().contains(&name.as_str()) {
                wrong.push(format!(
                    "`{name}`: kernels-metal has this row and we do not"
                ));
            }
            continue;
        };
        if got != want && !DIFFERS.contains(&name.as_str()) {
            wrong.push(format!(
                "`{name}`:\n    vulkan: {want:?}\n    wgpu:   {got:?}"
            ));
        }
    }
    // And the other direction: a row we still have that metal has retired is
    // equally not drift. `kernels-metal` states its own retired families in a
    // private `RETIRED`, so the readable signal here is that the name is one
    // of the hundred and has simply crossed over there first.
    let crossed_there = routine_names(&manifest().join("../kernels-vulkan/src"));
    assert!(
        crossed_there.len() > 50,
        "the scan found {} `routine!` declarations in `kernels-vulkan`, which \
         has crossed nearly all hundred. A scan that reads nothing would \
         excuse every difference below",
        crossed_there.len(),
    );
    for name in ours.keys() {
        if !theirs.contains_key(name) && !crossed_there.contains(name.as_str()) {
            wrong.push(format!(
                "`{name}`: we have this row and kernels-metal has neither a \
                 row nor a routine for it"
            ));
        }
    }

    assert!(
        wrong.is_empty(),
        "the two tables state different facts about the same kernels:\n{}",
        wrong.join("\n"),
    );
}

/// Every row asks for the same OPERANDS as the `kernels-metal` row of the same
/// name, per index.
///
/// An operand list is positional and it fixes the launch ABI, so the likely
/// mistakes — an inserted slot, a dropped one, a swapped pair — keep the length
/// the same and change every binding after them. Comparing per index is what
/// catches those; comparing the length is not.
///
/// Operand NAMES are deliberately not compared: each crate should be free to
/// call a slot whatever reads best beside its own shader. TYPES and SOURCES
/// are, because those are what a launch is built from.
#[test]
fn every_row_asks_for_the_same_operands_the_sibling_table_does() {
    let ours = scrape_operands(&manifest().join("src"));
    let theirs = scrape_operands(&manifest().join("../kernels-vulkan/src"));

    let stated = ours.values().filter(|ops| !ops.is_empty()).count();
    let really = kernels_wgpu::KERNELS
        .iter()
        .filter(|k| !k.operands.is_empty())
        .count();
    assert_eq!(
        stated, really,
        "the scrape read {stated} stated rows out of the table's {really} — a \
         text-based check that silently reads nothing is a check that passes",
    );

    let mut wrong = Vec::new();
    let mut allowed_seen = Vec::new();
    for (name, want) in &theirs {
        let Some(got) = ours.get(name) else { continue };
        if got.len() != want.len() {
            wrong.push(format!(
                "`{name}`: the sibling states {} operands, we state {}",
                want.len(),
                got.len(),
            ));
            continue;
        }
        for (at, (want, got)) in want.iter().zip(got).enumerate() {
            if want == got {
                continue;
            }
            let line = format!("`{name}` operand {at}: sibling `{want}`, wgpu `{got}`");
            if DELIBERATE.contains(&(name.as_str(), at)) {
                allowed_seen.push((name.clone(), at));
            } else {
                wrong.push(line);
            }
        }
    }

    assert!(
        wrong.is_empty(),
        "the two tables ask for different operands:\n{}",
        wrong.join("\n"),
    );

    // A stale exception is the failure mode an exception list has, so the
    // list is checked in BOTH directions: an entry that no longer names a
    // difference is one the tables have since agreed on, and keeping it would
    // hide the next divergence at the same slot.
    let mut stale: Vec<String> = DELIBERATE
        .iter()
        .filter(|(name, at)| {
            !allowed_seen
                .iter()
                .any(|(seen, sat)| seen == name && sat == at)
        })
        .map(|(name, at)| format!("`{name}` operand {at}"))
        .collect();
    stale.sort();
    assert!(
        stale.is_empty(),
        "these rows agree with the sibling table now, so the exception is \
         stale:\n  {}",
        stale.join("\n  "),
    );
}

/// The operands this table deliberately asks for differently, and why.
///
/// Not a skip list: the test asserts each entry still names a REAL difference
/// (see the staleness check), so an exception that outlives its reason fails
/// rather than accumulates.
///
/// `sdpa_paged_decode{,_sink}` operand 13 is the mask PITCH. `kernels-metal`
/// takes it from `Source::Param(3)` -- a launch parameter, which means from
/// the model TEXT -- and no text can know it: the pitch is a property of the
/// fire the driver is building, not of the program being lowered, and every
/// text in the corpus states `0` there. Zero is the value the shader reads as
/// "forbid every key of an enabled row", so a mask bound through that slot
/// answers nothing rather than answering wrongly -- which is the safe
/// direction and still not the mask the guest asked for.
///
/// This table asks for `Source::AttentionMaskStride`, which the driver
/// answers from the fire it is assembling. `tart-masked` runs on this backend
/// because of it, and `driver-wgpu/tests/serving.rs` holds the numeric pair
/// that says so: a causal mask gives bit-identical logits to no mask, and
/// forbidding half the keys moves the answer.
/// The tiled pair joined the list when upstream STATED their operands in
/// `kernels-metal`: they carry the same seventeen the decode rows do, in the
/// same order, so they carry the same divergence at the same index. The row
/// count is an eighteenth operand and sits past it. The MMA pair joined the
/// same way one rebase later, and for the same reason — this backend's
/// `attn/sdpa_paged_mma.wgsl` is Metal's entrypoint names over a scalar body
/// with the tiled shader's exact buffers and scalars.
///
/// Both times the gap outlived the commit that opened it. `kernels-metal`
/// stated the operands, this table went on stating none, and nothing said so
/// until THIS file was next run — which is a parity gate that lives in one of
/// three crates being only as good as the habit of running that crate.
/// EMPTY, and the six entries that stood here did not go because they were
/// settled.
///
/// They recorded operand 13 of the six `sdpa_paged_*` rows: metal reads the
/// TEXT's scalar there and this backend reads `Source::AttentionMaskStride`,
/// the pitch the driver actually staged. Then metal finished Stage 4, its
/// `KERNELS` emptied, and the scrape above had no rows to read at all — so
/// this comparison was repointed at `kernels-vulkan`, which still has 94, and
/// wgpu and vulkan AGREE at operand 13. The exceptions became unobservable
/// rather than resolved.
///
/// The record did not move with them, and that is the point of writing this
/// down: `kernels/tests/shader_backends_agree.rs`'s `DRIFTED` still carries
/// all six with the sentence explaining which backend is wrong, and
/// `kernels`'s `the_two_settled_drifts_are_still_true_of_the_drivers_they_name`
/// still reports `AttentionMaskStride` in ZERO places in `driver-metal`. The
/// defect is live; only this crate's view of it is gone.
const DELIBERATE: &[(&str, usize)] = &[];

/// The two runs of the launch ABI never collide, on any row.
///
/// A row's storage bindings and its uniform fields are two independent
/// numberings, and the property worth asserting over all 100 rows is the one a
/// single hand-picked example cannot give: that every operand lands somewhere,
/// exactly once, and that the two runs together account for the row.
#[test]
fn every_row_places_every_operand_exactly_once() {
    for sig in kernels_wgpu::KERNELS {
        let bindings = kernels_wgpu::bindings(sig);
        assert_eq!(
            bindings.len(),
            sig.operands.len(),
            "`{}` has {} operands and {} placements",
            sig.symbol,
            sig.operands.len(),
            bindings.len(),
        );

        let mut storages = BTreeSet::new();
        let mut uniforms = BTreeSet::new();
        for binding in &bindings {
            match binding {
                kernels_wgpu::Binding::Storage(n) => assert!(
                    storages.insert(*n),
                    "`{}` binds storage {n} twice",
                    sig.symbol,
                ),
                kernels_wgpu::Binding::Uniform(n) => assert!(
                    uniforms.insert(*n),
                    "`{}` fills uniform field {n} twice",
                    sig.symbol,
                ),
                kernels_wgpu::Binding::Packed => {}
            }
        }

        // Contiguous from zero, both runs. A hole in the ROW's numbering would
        // be a descriptor the shell never writes and the shader may still read.
        // (A hole in the SHADER's declarations is a different and legitimate
        // thing: a variant that never reads a buffer has it eliminated.)
        for (at, n) in storages.iter().enumerate() {
            assert_eq!(*n as usize, at, "`{}`'s storage run has a hole", sig.symbol);
        }
        for (at, n) in uniforms.iter().enumerate() {
            assert_eq!(*n as usize, at, "`{}`'s uniform run has a hole", sig.symbol);
        }

        assert_eq!(storages.len() as u32, kernels_wgpu::storage_count(sig));
        assert_eq!(uniforms.len(), kernels_wgpu::uniform_layout(sig).len());
    }
}

/// Every uniform field is aligned the way WGSL's uniform address space wants.
///
/// Checked over all 100 rows rather than on the one example the unit test picks,
/// because the failure — a shell packing by concatenation where the language
/// pads — is per-row and silent. A four-byte scalar before an eight-byte one is
/// the only shape that shows it, and only some rows have it.
#[test]
fn every_uniform_block_is_laid_out_the_way_wgsl_reads_it() {
    for sig in kernels_wgpu::KERNELS {
        let fields = kernels_wgpu::uniform_layout(sig);
        let mut at = 0u32;
        for field in &fields {
            assert_eq!(
                field.offset % field.size,
                0,
                "`{}`'s `{}` sits at {} with a size of {}",
                sig.symbol,
                field.name,
                field.offset,
                field.size,
            );
            assert!(
                field.offset >= at,
                "`{}`'s `{}` overlaps the field before it",
                sig.symbol,
                field.name,
            );
            assert_eq!(
                field.split,
                field.size == 8,
                "`{}`'s `{}` is eight bytes iff it crosses as a vec2<u32>",
                sig.symbol,
                field.name,
            );
            at = field.offset + field.size;
        }

        let size = kernels_wgpu::uniform_size(sig);
        if fields.is_empty() {
            assert_eq!(
                size, 0,
                "`{}` states no scalars and asks for a block",
                sig.symbol
            );
        } else {
            assert_eq!(
                size % 16,
                0,
                "`{}`'s block is not a multiple of 16",
                sig.symbol
            );
            assert!(
                size >= at,
                "`{}`'s block does not cover its own fields",
                sig.symbol
            );
        }
    }
}

/// The parity scrape reads every field a row can carry.
///
/// `every_row_states_the_same_facts_the_sibling_table_does` compares a NAMED list
/// of fields, and a named list rots: a field added to `kernels::KernelSig`
/// upstream and not added to [`FIELDS`] is a field the parity check silently
/// stops comparing. That is the direction that matters, because the whole
/// purpose of that check is to notice that somebody edited one table only.
///
/// It is not hypothetical. `publishes_aux` was added to `KernelSig` while this
/// crate was being written, and the scrape went on passing without it.
/// (That field has since left `KernelSig` altogether — step 9 measured it
/// CUDA-only and consolidated it onto `kernels_cuda::x::Contract`. The
/// anecdote is history, and it is the reason this test exists; the check
/// below is what caught the departure too, from the other direction.)
///
/// So the list is held against the struct itself: `kernels/src/lib.rs` is read,
/// its `pub struct KernelSig` fields are counted, and [`FIELDS`] together with
/// [`UNCOMPARED`] must account for exactly all of them.
///
/// Reading a struct out of a sibling crate's source rather than deriving it is
/// coarse. It is also the only option — a field list is not reflectable in Rust
/// — and the alternative, which is nothing, is what let `publishes_aux`
/// through.
#[test]
fn the_parity_check_reads_every_field_a_row_can_carry() {
    let source = manifest().join("../kernels/src/lib.rs");
    let text = std::fs::read_to_string(&source)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", source.display()));

    let open = text
        .find("pub struct KernelSig {")
        .expect("`kernels` declares `KernelSig`");
    let body = &text[open..];
    let close = body.find("\n}").expect("the struct is closed");

    let mut declared: BTreeSet<&str> = BTreeSet::new();
    for line in body[..close].lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("pub ") else {
            continue;
        };
        let Some((field, _)) = rest.split_once(':') else {
            continue;
        };
        let field = field.trim();
        if !field.is_empty() && field.chars().all(|c| c.is_alphanumeric() || c == '_') {
            declared.insert(field);
        }
    }

    assert!(
        declared.len() > 10,
        "the scrape of `KernelSig` read {} fields, which means it stopped \
         reading rather than that the struct shrank",
        declared.len(),
    );

    for (excluded, why) in UNCOMPARED {
        assert!(
            declared.remove(excluded),
            "`{excluded}` is excluded from the parity comparison ({why}) but is \
             no longer a field of `KernelSig`; the exclusion now hides nothing \
             and should go",
        );
    }

    let compared: BTreeSet<&str> = FIELDS.into_iter().collect();

    let unchecked: Vec<&&str> = declared.iter().filter(|f| !compared.contains(*f)).collect();
    assert!(
        unchecked.is_empty(),
        "`KernelSig` carries {unchecked:?}, which \
         `every_row_states_the_same_facts_the_sibling_table_does` does not compare. \
         Add them to `FIELDS`, or to `UNCOMPARED` with a reason. A field nobody \
         compares is a field the two tables may already disagree about.",
    );

    let phantom: Vec<&&str> = compared.iter().filter(|f| !declared.contains(*f)).collect();
    assert!(
        phantom.is_empty(),
        "the parity check looks for {phantom:?}, which `KernelSig` no longer \
         has. A field that cannot appear is a comparison that always agrees.",
    );
}

/// Every scalar `KernelSig` field the parity check compares.
///
/// Held against the struct itself by
/// [`the_parity_check_reads_every_field_a_row_can_carry`], because a named list
/// rots: a field added upstream and not added here is a field the comparison
/// silently stops making. `publishes_aux` arrived exactly that way, and left
/// the same list in step 9 — the `phantom` half of that test is what made the
/// departure visible.
const FIELDS: [&str; 11] = [
    "grid_param",
    "rows_param",
    "head_param",
    "heads_param",
    "launch",
    "lacks",
    "sink",
    "in_place",
    "whole",
    "depth_prefix_plan",
    "axes",
];

/// The `KernelSig` fields the parity check deliberately does NOT compare.
///
/// Each for its own reason, and each asserted still to BE a field — an
/// exclusion that outlives its subject hides nothing and should go.
const UNCOMPARED: [(&str, &str); 5] = [
    ("name", "the key this map is built on"),
    ("symbol", "the key this map is built on"),
    (
        "args",
        "DERIVED from a routine's `fn` signature, so no table states it in \
         source and a scrape of these files would find nothing to compare; \
         empty in all three of these tables, and only CUDA's fills it",
    ),
    (
        "file",
        "the one thing the three tables are SUPPOSED to disagree about: \
         `.metal` against `.comp` against `.wgsl`",
    ),
    (
        "operands",
        "compared per index by \
         `every_row_asks_for_the_same_operands_kernels_metal_does`, which is a \
         stronger check than a line-fragment comparison could be",
    ),
];

/// `name -> [field = value]` for every scalar field, scraped from `src/*.rs`.
///
/// The macro writes `kernel!(name "symbol"` to open a row, so the row a field
/// belongs to is the last `kernel!` seen — and a field can sit on its own line
/// or inline in the `kernel!` call, so both spellings are read. `file` is
/// deliberately NOT among the fields: it is the one thing the three tables are
/// supposed to disagree about (`.metal` against `.comp` against `.wgsl`).
///
/// The read is a line at a time, so a field whose value spans lines is compared
/// only as far as its first line. That is weaker than it looks for
/// `sdpa_paged_decode`'s multi-line `axes`, and it is still a real check: the
/// fragment is compared on both sides, and the entrypoint-set test above covers
/// what an `axes` list actually produces.
fn scrape_fields(dir: &Path) -> BTreeMap<String, Vec<String>> {
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for path in table_sources(dir) {
        let text = std::fs::read_to_string(&path).expect("a readable source");
        let mut row: Option<String> = None;
        // ONLY INSIDE THE TABLE.
        //
        // This walked the whole file and kept "the last `kernel!` seen", which
        // was harmless while a family's source was nothing but rows. It is not
        // any more: `norm.rs` now carries twelve routine BODIES after its
        // table, and `per_axis`'s `let axes = (width.unsigned_abs() / ...)`
        // was read as a field of `vnorm_single_row`, the last row above it --
        // so the two tables "disagreed" about a kernel over a local variable.
        //
        // Every table opens `pub static KERNELS: &[KernelSig] = &[` and closes
        // with a `];` in the first column, which is the bound.
        let mut in_table = false;
        for line in text.lines() {
            if line.starts_with("pub static KERNELS") {
                in_table = true;
                continue;
            }
            if in_table && line.starts_with("];") {
                in_table = false;
                row = None;
                continue;
            }
            if !in_table {
                continue;
            }
            let t = line.trim();
            if let Some(rest) = t.strip_prefix("kernel!(") {
                let name = leading_ident(rest);
                if !name.is_empty() {
                    out.entry(name.clone()).or_default();
                    row = Some(name);
                }
            }
            let Some(row) = &row else { continue };
            // Prose is not a field. A `//` line can legitimately contain
            // `field = value` while quoting code, and both tables' comments do.
            if t.starts_with("//") {
                continue;
            }
            for field in FIELDS {
                let Some(at) = t.find(&format!("{field} = ")) else {
                    continue;
                };
                // The preceding character must be a delimiter, so `axes = `
                // counts and `its axes = ` inside prose does not.
                if at > 0 && !matches!(t.as_bytes()[at - 1], b' ' | b'(' | b',') {
                    continue;
                }
                let value = t[at + field.len() + 3..]
                    .trim()
                    .trim_end_matches(',')
                    .to_owned();
                out.entry(row.clone())
                    .or_default()
                    .push(format!("{field} = {value}"));
            }
        }
    }

    for fields in out.values_mut() {
        fields.sort();
    }
    out
}

/// `name -> ["Ty <- Source", ...]`, scraped from an `operands![...]` block.
///
/// One operand per line in every one of the three crates, spelled
/// `name: Ty <- Source::X(n),`. The NAME is dropped on purpose (see the
/// caller); everything after the colon is kept verbatim, so a `| null` marker
/// is compared too.
fn scrape_operands(dir: &Path) -> BTreeMap<String, Vec<String>> {
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for path in table_sources(dir) {
        let text = std::fs::read_to_string(&path).expect("a readable source");
        let mut row: Option<String> = None;
        let mut inside = false;
        for line in text.lines() {
            let t = line.trim();
            if let Some(rest) = t.strip_prefix("kernel!(") {
                let name = leading_ident(rest);
                if !name.is_empty() {
                    out.entry(name.clone()).or_default();
                    row = Some(name);
                    inside = false;
                }
            }
            if t.starts_with("//") {
                continue;
            }
            if t.contains("operands![") {
                inside = true;
                continue;
            }
            if inside && t.starts_with(']') {
                inside = false;
                continue;
            }
            if !inside {
                continue;
            }
            let Some(row) = &row else { continue };
            let Some((_, rest)) = t.split_once(':') else {
                continue;
            };
            let operand = rest.trim().trim_end_matches(',').trim().to_owned();
            if !operand.is_empty() {
                out.entry(row.clone()).or_default().push(operand);
            }
        }
    }
    out
}

/// The family modules of a `kernels-*` crate's `src/`, sorted.
///
/// `lib.rs` is dropped because it holds the fold and not the rows, and any
/// non-table module (`axes`, `capability`, `preproc`, `source`) states no
/// `kernel!` so it contributes nothing either way.
fn table_sources(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .map(|e| e.expect("a readable entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "rs"))
        .filter(|p| p.file_name().is_some_and(|n| n != "lib.rs"))
        .collect();
    files.sort();
    files
}

/// The identifier a `kernel!(` call opens with.
fn leading_ident(rest: &str) -> String {
    rest.chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

/// The tile in an entrypoint's NAME is the tile its shader was compiled with.
///
/// 350 of this tree's instantiations state their GEMM tile twice — once as a
/// `_bm_N_bn_M` suffix on the entrypoint name, and once as `PIE_BM=N
/// PIE_BN=M` on the same `pie:instantiate` line — and until this test nothing
/// compared them.
///
/// # Why it matters, from the backend that paid for it
///
/// `driver-wgpu` reads the tile off the NAME (`geometry::Tile::from_entrypoint`)
/// and the workgroup off the shader's own `@workgroup_size`, and its own
/// comment says why: *"both numbers come from the thing being launched … so
/// there is no table for either to drift from."* That is the right design and
/// it leaves exactly one seam — the name and the defines, on one line.
///
/// `driver-metal` had the other design, and it cost a correctness defect of
/// the first order: its `Rule::Qmm` derived the tile a second time from the
/// fire's geometry, and at a 512-row prefill of a 2048-wide projection the two
/// sides said `(64, 64)` and `(32, 32)`. The grid is workgroups times the
/// COMPILED tile, so it launched a quarter of the output it needed and left
/// the other three quarters as arena residue — through sixteen layers, with no
/// error from any of them.
///
/// This backend cannot make that mistake the same way. It can make it on this
/// one line, and one line is what this reads.
#[test]
fn the_tile_in_an_entrypoints_name_is_the_tile_its_defines_compile() {
    /// The decimal literal after `key` in `text`, as the driver parses it.
    fn after(text: &str, key: &str) -> Option<u32> {
        let rest = &text[text.find(key)? + key.len()..];
        rest.chars()
            .take_while(char::is_ascii_digit)
            .collect::<String>()
            .parse()
            .ok()
    }

    fn wgsl(dir: &std::path::Path, into: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let path = e.path();
            if path.is_dir() {
                wgsl(&path, into);
            } else if path.extension().is_some_and(|x| x == "wgsl") {
                into.push(path);
            }
        }
    }

    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels");
    let mut files = Vec::new();
    wgsl(&root, &mut files);
    assert!(
        files.len() > 20,
        "{} shader files under {} is not this tree",
        files.len(),
        root.display()
    );

    let mut tiled = 0usize;
    for file in &files {
        let text = std::fs::read_to_string(file).expect("a readable shader");
        for line in text.lines() {
            let Some(rest) = line.trim().strip_prefix("// pie:instantiate ") else {
                continue;
            };
            let mut words = rest.split_whitespace();
            let name = words.next().expect("an instantiation names an entrypoint");
            let defines: Vec<&str> = words.collect();
            let joined = defines.join(" ");

            let named = (after(name, "_bm_"), after(name, "_bn_"));
            let defined = (after(&joined, "PIE_BM="), after(&joined, "PIE_BN="));

            // Both directions. A name carrying a tile the defines do not set
            // compiles a DIFFERENT kernel from the one the driver grids for;
            // defines without the suffix leave `Tile::from_entrypoint` with
            // `None`, and the row is refused as `Ungeometric::Untiled` — a
            // launch that never happens rather than one that writes a quarter,
            // but still not what anyone wrote.
            assert_eq!(
                named,
                defined,
                "`{}` in {}: the name says bm/bn {named:?} and its defines say \
                 {defined:?}. The driver grids from the NAME and the shader is \
                 compiled from the DEFINES, so these two disagreeing is the \
                 same defect that made `driver-metal` write the top-left \
                 quarter of every long prefill.",
                name,
                file.strip_prefix(&root).unwrap_or(file).display()
            );
            if named.0.is_some() {
                tiled += 1;
            }
        }
    }
    assert_eq!(
        tiled, 350,
        "{tiled} tiled instantiations, and this tree had 350. If a family \
         gained or lost one, say so here — the number is what makes this a \
         census rather than a spot check."
    );
}

/// A flat grid's shader does not read the row off the y axis.
///
/// `LaunchRule::Elementwise` is `[width * rows, 1, 1]`. The y extent is ONE,
/// so `gid.y` and `workgroup_id.y` are structurally zero for every invocation
/// — and a body that takes its row from either writes row 0 and returns for
/// the rest, while the dispatch succeeds.
///
/// This tree has paid for it once. `geglu_tanh_strided` was `@workgroup_size(16,
/// 16)` reading `gid.y` as the row, and its own header now records the
/// measurement: at 21 rows on a 4090 row 16 came back holding the sentinel it
/// was born with, so any gemma prefill longer than sixteen tokens was silently
/// dropping most of its per-layer embeddings. `driver-metal` found the same
/// defect in the same kernel independently, which is what makes it a class
/// rather than an incident.
///
/// So it is asked of every entrypoint of every flat row, over the EXPANDED
/// source — the variant the driver will actually build, with its `//#if` arms
/// resolved — because the y read can live under a define that only one point
/// takes.
///
/// Only `Elementwise` is checked. Other rules put real extents on y and z, and
/// a rule-by-rule table of which axes are live would be the `LaunchRule` enum
/// written down twice; the flat one is the case where the answer is the same
/// for every row that states it.
#[test]
fn a_flat_rows_shader_does_not_read_its_row_off_the_y_axis() {
    /// The parameter names bound to the two id builtins in `source`.
    fn id_names(source: &str) -> Vec<String> {
        let mut out = Vec::new();
        for builtin in ["global_invocation_id", "workgroup_id"] {
            let needle = format!("@builtin({builtin})");
            for (at, _) in source.match_indices(&needle) {
                let rest = &source[at + needle.len()..];
                let name: String = rest
                    .trim_start()
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                if !name.is_empty() {
                    out.push(name);
                }
            }
        }
        out
    }

    /// The source with its line comments removed.
    ///
    /// Necessary and not fastidious: `gated.wgsl`'s header EXPLAINS this very
    /// defect, in prose, naming `gid.y` twice. A scan over raw text fires on
    /// the comment that records the fix, which is the most misleading possible
    /// false positive -- it points at the one body that has already been made
    /// right.
    fn code(source: &str) -> String {
        source
            .lines()
            .map(|l| l.split_once("//").map_or(l, |(before, _)| before))
            .collect::<Vec<_>>()
            .join("\n")
    }

    let mut checked = 0usize;
    let mut flat = 0usize;
    for row in kernels_wgpu::KERNELS {
        if row.launch != kernels::LaunchRule::Elementwise {
            continue;
        }
        flat += 1;
        for entrypoint in row.entrypoints() {
            let Ok(source) =
                kernels_wgpu::entrypoint_source(&entrypoint, kernels_wgpu::Capability::Baseline)
            else {
                continue;
            };
            let body = code(&source);
            for name in id_names(&body) {
                assert!(
                    !body.contains(&format!("{name}.y")),
                    "`{entrypoint}` states `LaunchRule::Elementwise`, whose \
                     grid is `[width * rows, 1, 1]`, and its body reads \
                     `{name}.y`. That is structurally zero, so it writes row 0 \
                     and returns for every other row while the dispatch \
                     succeeds — the defect `geglu_tanh_strided`'s header \
                     records, measured at 21 rows."
                );
            }
            checked += 1;
        }
    }
    assert!(
        flat >= 10 && checked >= 20,
        "{flat} flat rows over {checked} entrypoints is not this table"
    );
}

/// The entrypoints that IGNORE the sliding window are exactly the ones that
/// say so in their name.
///
/// `attn/sdpa_paged.wgsl` compiles two starts:
///
/// ```text
/// //#if defined(PIE_FAST_FULL)
///     var start = 0;
/// //#else
///     var start = 0;
///     if (params.window > 0 && q_pos >= params.window) { start = q_pos - params.window + 1; }
/// //#endif
/// ```
///
/// So a variant built with `PIE_FAST_FULL` reads `params.window` nowhere: a
/// caller that asks for a window gets FULL attention, and the dispatch
/// succeeds. `kernels-vulkan` reached the same conclusion from the other side
/// while crossing `attn` — it declines to spell those names from a routine at
/// all, calling them "deliberately unreachable".
///
/// # This tree can reach them, and nothing stops it
///
/// `sdpa_paged_decode`'s row declares `_p32`, `_p32` at d_128 and `_p32_sg8`
/// as points of its own axis, so they are in `entrypoints()`, the driver
/// builds pipelines for them, and `sig_in` resolves a statement that names
/// one. `grep -rn 'FAST_FULL\|_p32' crates/driver-wgpu/src` finds NOTHING —
/// there is no refusal anywhere for naming a window-ignoring kernel while
/// stating a window.
///
/// Removing the three points is a three-backend decision, not this file's:
/// `refactor-bigplan.md` §1 measured the `axes` column identical in 100 of 100
/// rows, so dropping them here alone would be the first divergence in that
/// column. Until it is taken, this pins the set: the names that ignore the
/// window are exactly the instantiations that set the define, and a fourth
/// cannot arrive without this failing.
#[test]
fn the_entrypoints_that_ignore_the_window_are_exactly_the_ones_named_p32() {
    fn wgsl(dir: &std::path::Path, into: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let path = e.path();
            if path.is_dir() {
                wgsl(&path, into);
            } else if path.extension().is_some_and(|x| x == "wgsl") {
                into.push(path);
            }
        }
    }

    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels");
    let mut files = Vec::new();
    wgsl(&root, &mut files);

    let mut fast_full: Vec<String> = Vec::new();
    for file in &files {
        let text = std::fs::read_to_string(file).expect("a readable shader");
        for line in text.lines() {
            let Some(rest) = line.trim().strip_prefix("// pie:instantiate ") else {
                continue;
            };
            if rest.contains("PIE_FAST_FULL") {
                fast_full.push(
                    rest.split_whitespace()
                        .next()
                        .expect("an instantiation names an entrypoint")
                        .to_owned(),
                );
            }
        }
    }
    fast_full.sort();

    assert_eq!(
        fast_full,
        [
            "sdpa_paged_decode_bfloat16_d_128_p32",
            "sdpa_paged_decode_bfloat16_d_64_p32",
            "sdpa_paged_decode_bfloat16_d_64_p32_sg8",
        ],
        "the set of window-ignoring entrypoints has moved. Every one of these \
         returns FULL attention for a statement that asked for a window, and \
         succeeds while doing it -- so a new one is a new way to be silently \
         wrong, and one that stopped setting the define is a name that no \
         longer means what it says."
    );

    // And every one of them SAYS SO in its name, which is the only warning a
    // reader of a trace gets.
    for name in &fast_full {
        assert!(
            name.contains("_p32"),
            "`{name}` ignores the window and its name does not say so"
        );
    }
}

/// No shader in this tree declares a `read` storage binding, and that is a
/// PERFORMANCE decision with a measurement behind it.
///
/// # The rule
///
/// WebGPU's usage scope admits any number of INCLUSIVE usages or exactly one
/// EXCLUSIVE usage, never both (`wgpu-core-30.0.0/src/track/mod.rs:333`).
/// `STORAGE_READ_ONLY` is inclusive and `STORAGE_READ_WRITE` is exclusive, so
/// one buffer bound both ways in one dispatch is two bits with an exclusive
/// one among them, and the dispatch is refused. Buffers have no subresources,
/// so disjoint ranges do not help.
///
/// `binding::Arena` is ONE allocation holding every activation, so a launch's
/// input and its output are two ranges of it — and 451 of a 452-launch decode
/// hit exactly this. `driver-wgpu` answered by SHADOWING: copying each
/// read-only range into scratch and binding that instead. Correct, and it cost
/// a copy and a compute-pass boundary per rectangle.
///
/// # Two `read_write` bindings are ONE bit
///
/// `state.any_exclusive() && !state.bits().is_power_of_two()` — two exclusive
/// usages of the same buffer are the SAME BIT, so `is_power_of_two` holds and
/// the dispatch is legal. This is WebGPU's "usage scope storage exception",
/// and it means the whole workaround is avoidable by declaring the read side
/// `read_write` too. `driver-wgpu::device`'s
/// `two_read_write_bindings_into_one_buffer` is that, proven on a device.
///
/// Measured on an RTX 4090, qwen3-0.6B, median of 40 decodes, same probe:
///
/// | | shadowed | ms | tok/s |
/// | --- | --- | --- | --- |
/// | `var<storage, read>` | 451 | 25.1 | 39.8 |
/// | `var<storage, read_write>` | 0 | 11.2 | 89.3 |
///
/// # Why it is safe
///
/// A binding declared `read_write` that the body only reads is read-only in
/// fact; nothing writes more than it did. The hazard a `read` declaration
/// would have caught is two operands covering the same bytes partially, and
/// that is now a named refusal — `driver-wgpu`'s `Failed::Overlapping`, which
/// no real plan raises. `kernels-metal` and `kernels-vulkan` never had the
/// declaration to lose: both bind the arena both ways without comment.
///
/// # Why it is a test
///
/// One `var<storage, read>` in one new shader silently reinstates the shadow
/// for every launch that touches it, and the only symptom is that decoding
/// got slower.
#[test]
fn no_shader_declares_a_read_only_storage_binding() {
    fn wgsl(dir: &std::path::Path, into: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let path = e.path();
            if path.is_dir() {
                wgsl(&path, into);
            } else if path.extension().is_some_and(|x| x == "wgsl") {
                into.push(path);
            }
        }
    }

    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels");
    let mut files = Vec::new();
    wgsl(&root, &mut files);
    assert_eq!(
        files.len(),
        37,
        "the shader tree changed size; this scan reads every `.wgsl` under \
         `kernels/`, and a count that moved without anyone noticing is a \
         file it might not be reading"
    );

    let mut offenders: Vec<String> = Vec::new();
    let mut read_write = 0usize;
    for file in &files {
        let text = std::fs::read_to_string(file).expect("a readable shader");
        for (at, line) in text.lines().enumerate() {
            if line.contains("var<storage, read>") || line.contains("var<storage,read>") {
                offenders.push(format!(
                    "{}:{}: {}",
                    file.file_name().unwrap_or_default().to_string_lossy(),
                    at + 1,
                    line.trim()
                ));
            }
            if line.contains("var<storage, read_write>") {
                read_write += 1;
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "a `read` storage binding reinstates the per-rectangle shadow copy \
         for every launch that touches it, and the only symptom is that a \
         decode got twice as slow. Declare it `read_write`; see this test's \
         own docs. Found: {offenders:#?}"
    );
    assert!(
        read_write > 150,
        "only {read_write} `read_write` bindings were seen, which is too few \
         for this tree -- the scan is probably not reading what it thinks"
    );
}
