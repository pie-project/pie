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
use std::path::PathBuf;

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

/// Every `Fire`, with whether its `lanes` came from a FLAT `elementwise` call.
///
/// `elementwise` is `[width * rows, 1, 1]`; `elementwise_rows` is
/// `[width, rows, 1]`. The two differ by a suffix, so the match is on the
/// whole call and `elementwise_rows(` must not read as `elementwise(`.
/// One `Fire`: the line it names its entrypoint on, that name, and whether
/// its `lanes` came from a FLAT `elementwise` call.
type Fired = (usize, String, bool);

fn elementwise_fires(text: &str) -> Vec<Fired> {
    let mut out = Vec::new();
    /// Brace depth, the entrypoint seen so far, and whether a flat
    /// `elementwise` call was seen — the state of one open `Fire { .. }`.
    type Open = (i32, Option<(usize, String)>, bool);
    let mut depth: Option<Open> = None;
    for (n, line) in text.lines().enumerate() {
        let code = line.split_once("//").map_or(line, |(before, _)| before);
        if depth.is_none() && code.contains("Fire {") {
            depth = Some((0, None, false));
        }
        let Some((level, named, flat)) = depth.as_mut() else {
            continue;
        };
        if let Some((_, rest)) = code.split_once("entrypoint:") {
            *named = Some((n + 1, rest.trim().trim_end_matches(',').to_owned()));
        }
        if code.contains("elementwise(") {
            *flat = true;
        }
        *level += i32::try_from(code.matches('{').count()).expect("few braces");
        *level -= i32::try_from(code.matches('}').count()).expect("few braces");
        if *level <= 0 {
            if let Some((at, value)) = named.take() {
                out.push((at, value, *flat));
            }
            depth = None;
        }
    }
    out
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
/// `every_entrypoint_a_body_names_is_one_the_tree_carries` makes about bodies.
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
        ("mlp.rs", kernels_wgpu::mlp::ENTRYPOINTS),
        ("norm.rs", kernels_wgpu::norm::ENTRYPOINTS),
        ("layout.rs", kernels_wgpu::layout::ENTRYPOINTS),
        ("rope.rs", kernels_wgpu::rope::ENTRYPOINTS),
        ("quant.rs", kernels_wgpu::quant::ENTRYPOINTS),
        ("moe.rs", kernels_wgpu::moe::ENTRYPOINTS),
        ("ssm.rs", kernels_wgpu::ssm::ENTRYPOINTS),
        ("attn.rs", kernels_wgpu::attn::ENTRYPOINTS),
    ];

    // Against `retired()`, not a COUNT of families. The first version of this
    // compared `families.len()` to `retired_rows().len()` — a module count
    // against a ROW count — which held only while every retired family was one
    // kernel, and broke on `mlp`'s five. What has to hold is that the modules
    // listed here name every entrypoint the crate says is retired.
    let stated: BTreeSet<&str> = kernels_wgpu::retired().into_iter().collect();
    let listed: BTreeSet<&str> = families
        .iter()
        .flat_map(|(_, e)| e.iter().copied())
        .collect();
    assert_eq!(
        listed, stated,
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
        // AN IDENTITY NOW, and kept as one deliberately. It filtered out the
        // entrypoints a row still stated, because a family part way through
        // Stage 3 held rows for the kernels whose arms had not landed and
        // `entrypoints()` already reached those through the table -- listing
        // them here would have named them twice. Every family has crossed, so
        // `sig` answers `None` for all of them and nothing is dropped.
        //
        // Left in rather than deleted because it is the line that would fail
        // if a row came back: the comparison below would then see a name from
        // both directions and disagree with itself, which is a worse failure
        // to read than this filter quietly doing nothing.
        let fired: BTreeSet<String> = fired
            .into_iter()
            .filter(|e| kernels_wgpu::sig(e).is_none())
            .collect();
        let want: BTreeSet<String> = stated.iter().map(|s| (*s).to_owned()).collect();

        // Every entrypoint a body FIRES is stated. The reverse does not hold
        // and must not be forced to: a retired row's `axes` generated points
        // its body never selects, and `attn`'s are exactly the ones
        // `refactor-bigplan.md` §8b is about — `_p32` and `_sg8` compile the
        // sliding-window clamp out, and no body picks them. They stay in the
        // census because the shader tree has them and the device sweep builds
        // them; what would be wrong is a body firing something the census does
        // not name, which is the direction asserted here.
        let unfired: Vec<&String> = want.difference(&fired).collect();
        assert!(
            unfired
                .iter()
                .all(|e| e.contains("_p32") || e.contains("_sg8")),
            "`{module}` states entrypoints no body fires and that are not the \
             known window-eliding variants: {unfired:?}. A stated name nothing \
             fires is a name every sweep walks and nothing exercises",
        );
        let fired: BTreeSet<String> = fired.into_iter().collect();
        assert!(
            fired.is_subset(&want),
            "`{module}`'s bodies fire {:?}, which its `ENTRYPOINTS` does not \
             state, and `ENTRYPOINTS` is what every sweep keyed on \
             `entrypoints()` will walk",
            fired.difference(&want).collect::<Vec<_>>(),
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

// RETIRED: THE TABLE IS EMPTY, so the walk has no rows to find a clash among.
//
// It asserted that no two rows generate the same entrypoint name -- two rows
// whose `axes` cross to the same string, where whichever the tree walk reached
// second would silently win. The shader side of the same question is caught
// earlier and harder: a duplicate `pie:instantiate` is two variants under one
// name, and `build.rs` refuses it.
//
// It BECAME BLIND, and with no floor it could not say so: an empty iteration
// passes.
//
// The claim is LIVE on the routine plane and is asserted twice there, because
// a duplicate has no correct resolution:
// `driver-wgpu::lowering::arm::no_stem_is_registered_twice` refuses two
// `Crossed` rows with the same stem -- which had actually HAPPENED, for
// `affine_qmm_t_routed` and `affine_qmm_t_routed_fp16`, where which one
// answered was decided by `max_by_key`'s last-maximum tie-break rather than by
// anything the file said -- and
// `every_entrypoint_is_claimed_by_the_stem_that_owns_it` asserts that the stem
// each of the 481 resolves to is the longest one that names it.

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

// RETIRED: THE TABLE IS EMPTY, so every entrypoint is skipped before it is
// asked about.
//
// It walked all 481 entrypoints and required each to resolve to a row through
// `sig_in` -- exact match first, then axis-point matching, so
// `sdpa_paged_decode_bfloat16_d_128_p32` finds the `sdpa_paged_decode` row.
// A family that had crossed was skipped, on the correct argument that such an
// entrypoint resolves through `driver-wgpu`'s `arm_for` instead and what must
// hold is that it has SOME owner.
//
// It BECAME BLIND, not true, and it did so ONE FAMILY AT A TIME: each
// retirement moved names into `retired()` and out of the loop, so the check
// shrank silently until the skip covered everything. Nothing in it could have
// said so -- there is no floor, and an empty walk passes.
//
// The successor is the one that check was deferring to, and it is stronger:
// `driver-wgpu::lowering::arm::the_armed_stems_are_the_ones_registered_and
// _nothing_else` asserts that NO entrypoint is unclaimed, over the same 481,
// and it is not a courtesy -- with no table to fall back to, an entrypoint no
// stem claims cannot be planned by any path.

// RETIRED: `KernelSig::file` is gone from `kernels`, and there is no row left
// to have carried one.
//
// Two tests stood here and both read that column.
//
// `every_stated_file_exists` asserted that a row naming a shader named one
// `kernels_wgpu::source` could resolve. A path pointing at a file nobody
// wrote is a pipeline the shell asks for and cannot create, one layer away
// from the row that named it, and the runtime compiler is the only thing that
// would ever have said so.
//
// `every_stated_file_carries_the_rows_own_entrypoints` was the word the first
// one was short: a row may name a real shader that has nothing to do with it,
// and `moe`'s `qmv_routed` named `quant/qmv.wgsl`, which exists and contains
// the string `qmv_routed` exactly zero times. So it read the file's text and
// required a `pie:instantiate` line for every entrypoint the row generates.
// Its count was ASSERTED and not bounded -- 196, then 192, 185, 159, 154, 24
// as families came off -- and its last message said the quiet part: *"when
// the last row goes this test goes with it"*.
//
// Neither became TRUE. Both went BLIND, and each in two ways at once: the
// column they read was deleted, and the `for row in kernels_wgpu::KERNELS`
// they read it in walks an empty slice. A sweep with nothing to sweep reads
// exactly like a sweep that found nothing wrong, which is why the second one
// had already been retuned to `checked == 0` before the column went.
//
// The claim crossed with the column. A ROUTINE states its shader as the
// `module` of the `Fire` its body returns, and `kernels-wgpu`'s
// `every_entrypoint_a_body_names_is_one_the_tree_carries` resolves every
// entrypoint every body can name through `source::entrypoint_source`. That is
// the two questions answered as one and per ENTRYPOINT rather than per file:
// a name the tree has not got fails whether the module is missing or merely
// does not instantiate it, which is precisely the gap the second test existed
// to close. `driver-wgpu`'s
// `every_entrypoint_in_the_tree_builds_a_pipeline_on_this_adapter` asks it of
// a device.

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

// RETIRED: THE TABLE IS EMPTY, so `sig` answers `None` and every candidate is
// skipped.
//
// It read the SHADERS' prose. Seven of them justified their binding order with
// a sentence beginning "the row is UNSTATED in the table" -- meaning no row
// named the operands, so the order was taken from the sibling backends rather
// than derived -- and this test caught the case where such a sentence outlived
// its premise, because the table had since stated that kernel.
//
// It BECAME BLIND. `sig` cannot return a row, so the inner test never ran.
//
// AND ITS PREMISE OUTLIVED IT, which is the part worth recording: with the
// tables deleted, all seven sentences described a table that does not exist,
// and one of them named `kernels_wgpu::bindings` and
// `kernels_wgpu::uniform_layout`, which are deleted too. Every one has been
// rewritten to point at the ROUTINE whose signature states the order --
// the same order, differently written down. The word does not appear in
// `kernels/` any more except in one deliberate quotation of what it used to
// say, so there is nothing left for this test to find.

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

// RETIRED, both of them, and not because they were settled.
//
// Two tests stood here. One scraped a sibling crate's `kernel!` rows and
// compared them field for field; the other did the same for operand lists. They were the fleet's strongest parity
// gate — text against text, so a fact stated in one table and not the other
// failed here rather than in whichever driver read it.
//
// They compared against `kernels-metal` until `489a36031` emptied it, then
// against `kernels-vulkan`, which agreed everywhere but `route_gather` — the
// one entry in the fleet's `DRIFTED`. Then vulkan finished Stage 4 too, and
// **this crate is the last with rows**: `kernels-wgpu` 82, `kernels-vulkan`
// 1, `kernels-metal` 0. There is nothing left to compare against, and a
// scrape that reads one row is not a weaker check, it is no check.
//
// What replaces them, and it is not nothing:
// `kernels/tests/shader_backends_agree.rs` compares ROUTINE signatures
// across all three backends, which is the same claim about the same kernels
// with the tables taken out of the middle. That gate grows as this one
// shrinks, which is the trade Stage 3 was for.
//
// Do not restore these against a one-row table. If a sibling grows rows
// again that is a revert, not a reference.
//
// The operands this table deliberately asks for differently, and why.
//
// Not a skip list: the test asserts each entry still names a REAL difference
// (see the staleness check), so an exception that outlives its reason fails
// rather than accumulates.
//
// `sdpa_paged_decode{,_sink}` operand 13 is the mask PITCH. `kernels-metal`
// takes it from `Source::Param(3)` -- a launch parameter, which means from
// the model TEXT -- and no text can know it: the pitch is a property of the
// fire the driver is building, not of the program being lowered, and every
// text in the corpus states `0` there. Zero is the value the shader reads as
// "forbid every key of an enabled row", so a mask bound through that slot
// answers nothing rather than answering wrongly -- which is the safe
// direction and still not the mask the guest asked for.
//
// This table asks for `Source::AttentionMaskStride`, which the driver
// answers from the fire it is assembling. `tart-masked` runs on this backend
// because of it, and `driver-wgpu/tests/serving.rs` holds the numeric pair
// that says so: a causal mask gives bit-identical logits to no mask, and
// forbidding half the keys moves the answer.
// The tiled pair joined the list when upstream STATED their operands in
// `kernels-metal`: they carry the same seventeen the decode rows do, in the
// same order, so they carry the same divergence at the same index. The row
// count is an eighteenth operand and sits past it. The MMA pair joined the
// same way one rebase later, and for the same reason — this backend's
// `attn/sdpa_paged_mma.wgsl` is Metal's entrypoint names over a scalar body
// with the tiled shader's exact buffers and scalars.
//
// Both times the gap outlived the commit that opened it. `kernels-metal`
// stated the operands, this table went on stating none, and nothing said so
// until THIS file was next run — which is a parity gate that lives in one of
// three crates being only as good as the habit of running that crate.
// EMPTY, and the six entries that stood here are now settled -- which took
// three moves, only the last of which was a fix.
//
// They recorded operand 13 of the six `sdpa_paged_*` rows: metal read the
// TEXT's scalar there and this backend read the pitch the driver staged.
// Then metal finished Stage 4, its `KERNELS` emptied, this comparison was
// repointed at `kernels-vulkan`, and wgpu and vulkan AGREED at operand 13 --
// so the exceptions became unobservable rather than resolved, and this file
// said so rather than closing them.
//
// The fix came from the other side. All three planes now spell operand 13
// `Ask<keys::AttentionMaskStride, u32>` and each DRIVER answers it: wgpu and
// vulkan with the pitch of the rectangle they staged, metal with zero,
// because the mask metal stages is one enable word per token and no mask
// beside it. Metal's number is what it always was; the sentence is now true,
// and `DRIFTED` is empty.

// RETIRED: THE TABLE IS EMPTY, so neither sweep has a row to walk, and
// `bindings`, `storage_count`, `uniform_layout` and `uniform_size` -- every
// function the two were written in -- have gone with it.
//
// `every_row_places_every_operand_exactly_once` was the launch ABI's central
// claim, made over the whole table instead of over one example: a row's
// STORAGE bindings and its UNIFORM fields are two independent numberings, so
// every operand lands in exactly one of them, no slot and no field is taken
// twice, and both runs are contiguous from zero. A hole in either numbering
// is a descriptor the shell never writes and the shader may still read. It
// closed by holding the two runs it had counted against `storage_count` and
// `uniform_layout`, so the placements and the sizes could not disagree.
//
// `every_uniform_block_is_laid_out_the_way_wgsl_reads_it` was WGSL's uniform
// address space, asked of every row: a member is aligned to its own size, so
// a `Usize` after a `u32` starts at 8 and not at 4, an eight-byte field is
// exactly the one declared `vec2<u32>`, no field overlaps the one before it,
// and the block rounds up to a multiple of 16 no smaller than the fields it
// holds. A shell packing by concatenation where the language pads writes
// every value after the first wide one four bytes low, and nothing at runtime
// reports it: a uniform buffer is bytes.
//
// Neither became TRUE. Both went BLIND -- `for sig in kernels_wgpu::KERNELS`
// over an empty slice asserts nothing at all, and an empty sweep is
// indistinguishable from a clean one in a test log.
//
// Both claims are alive on the other side of the crossing, and against the
// shader rather than against a description of it. `driver-wgpu`'s
// `reflect::Declared` is `naga`'s reading of the real module: `bindings` and
// `used` are the `@group(0)` run with its holes, `uniform_offsets` is where
// the module reads each member of its `@group(1)` block, and `uniform_bytes`
// is the span it needs. `driver-wgpu/tests/arena.rs`'s
// `every_launchs_scalars_land_where_its_module_reads_them` holds the offsets
// a body PACKS to against `Declared::uniform_offsets` field for field, over
// every rectangle of every real lowering -- two independent readings of one
// layout, neither of them a table. `tests/gpu.rs`'s `block_of_shader` applies
// the same alignment rule to the parsed struct when this suite fills a block
// by name, and `kernels-wgpu`'s
// `every_routine_binds_a_buffer_for_every_binding_its_module_declares` is the
// buffer run's half: a body passes one buffer per declared `@group(0)`
// binding, including the ones its entrypoint never reads.

// RETIRED: `the_parity_check_reads_every_field_a_row_can_carry`, with
// `FIELDS` and `UNCOMPARED`, because the comparison they were the manifest
// for has itself retired and `KernelSig` has since shed eight of the eleven
// columns they named.
//
// It read `kernels/src/lib.rs` as TEXT, scraped the field names out of `pub
// struct KernelSig`, and required `FIELDS` and `UNCOMPARED` together to
// account for exactly all of them -- in both directions. A field upstream
// that neither list named was a field the row-parity pair had silently
// stopped comparing; a name in either list that was no longer a field was a
// comparison that always agreed. `publishes_aux` is why it existed: it was
// added to `KernelSig` while this crate was being written and the parity
// scrape went on passing without it, and the `phantom` half is what noticed
// when it left again.
//
// `FIELDS` named `grid_param`, `rows_param`, `head_param`, `heads_param`,
// `launch`, `lacks`, `sink`, `in_place`, `whole`, `depth_prefix_plan` and
// `axes`; `UNCOMPARED` excused `name` and `symbol` as the key the comparison
// was built on, `args` as derived from a routine signature and stated by no
// table in source, `file` as the one column three backends are SUPPOSED to
// disagree about, and `operands` as compared per index by a stronger check.
// Seven of those eighteen are gone from the struct.
//
// It did not become TRUE and it did not go blind either -- it went RED, which
// is what it was built to do. `declared.len() > 10` is the floor it kept so a
// scrape that stopped READING could not read as a struct that had shrunk, and
// `KernelSig` is down to eight fields; the `UNCOMPARED` loop then refuses an
// exclusion whose field has left, and `phantom` refuses a compared name that
// cannot appear. Every one of those fires. What has no answer is the repair:
// the pair it was the manifest for compared this crate's rows against a
// sibling table's, and there are no rows on either side to compare.
//
// A routine states what these columns stated, in a form nothing has to scrape:
// `whole`, `depth_prefix_plan` and `in_place` are fields of
// `kernels::routine::Routine`, `args` is derived from the body's own `fn`
// signature, and the four `*_param` indices -- a row pointing into a
// statement's argument list -- are arguments the body simply takes, which
// `kernels-wgpu`'s `a_body_passes_the_arguments_its_signature_takes_in_order`
// holds to the module. Divergence between backends is asked where it can
// still be asked: `kernels/tests/shader_backends_agree.rs`.

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
/// A FLAT grid is `[width * rows, 1, 1]` — what a row said with
/// `LaunchRule::Elementwise` and what a body says by calling
/// `kernels::shader::elementwise`. The y extent is ONE, so `gid.y` and
/// `workgroup_id.y` are structurally zero for every invocation — and a body
/// that takes its row from either writes row 0 and returns for the rest,
/// while the dispatch succeeds.
///
/// This tree has paid for it once. `geglu_tanh_strided` was `@workgroup_size(16,
/// 16)` reading `gid.y` as the row, and its own header now records the
/// measurement: at 21 rows on a 4090 row 16 came back holding the sentinel it
/// was born with, so any gemma prefill longer than sixteen tokens was silently
/// dropping most of its per-layer embeddings. `driver-metal` found the same
/// defect in the same kernel independently, which is what makes it a class
/// rather than an incident.
///
/// So it is asked of every entrypoint of every flat FIRE, over the EXPANDED
/// source — the variant the driver will actually build, with its `//#if` arms
/// resolved — because the y read can live under a define that only one point
/// takes.
///
/// Only the flat grid is checked. `elementwise_rows` and the shaped rules put
/// real extents on y and z, and a rule-by-rule table of which axes are live
/// would be the `LaunchRule` enum written down twice; the flat one is the case
/// where the answer is the same for every kernel that asks for it.
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

    // The FLAT kernels, from the crossed bodies.
    //
    // RETIRED: the ROW half of this sweep -- `KernelSig::launch` is gone from
    // `kernels` and `KERNELS` is empty, so there is neither a column to read
    // nor a row to read it on.
    //
    // It counted every row whose `LaunchRule` was `Elementwise` and added
    // that row's entrypoints to the set the y-axis scan below runs over. It
    // did not become TRUE, it went BLIND -- and it had been contributing zero
    // for as long as the table has been empty, which is why the count below
    // does not move.
    //
    // A routine says FLAT by calling `kernels::shader::elementwise`, which
    // returns the same `[width * rows, 1, 1]` the rule did, so the property
    // moved from the row to the body as each family crossed and the body
    // half read below is now the whole of it. `mlp` was four of them, every
    // one an `elementwise` body, and `geglu_tanh_strided`'s header is where
    // this defect was measured at 21 rows in the first place.
    let mut flat_points: Vec<String> = Vec::new();
    let mut flat = 0usize;
    // The crossed bodies, whose rows are gone.
    //
    // Per FIRE rather than per module, because a module holds both kinds:
    // `norm/residual_add.wgsl` has a flat variant and a strided one, and the
    // strided one reads `gid.y` on purpose. Counting the module would have
    // flagged the correct kernel — which is precisely how `residual_add`'s
    // own defect was hidden in the first place, its body asking for
    // `elementwise_rows` against a shader that reads `gid.x` alone.
    let tables = entrypoint_tables();
    for module in routine_sources() {
        let text = std::fs::read_to_string(&module).expect("a readable module");
        for (line, value, flat_call) in elementwise_fires(&text) {
            if !flat_call {
                continue;
            }
            let names = resolve(&value, &tables).unwrap_or_else(|| {
                panic!(
                    "{}:{line}: `{value}` is not a readable entrypoint",
                    module.display()
                )
            });
            flat += 1;
            flat_points.extend(names);
        }
    }

    let mut checked = 0usize;
    {
        for entrypoint in flat_points {
            let Ok(source) =
                kernels_wgpu::entrypoint_source(&entrypoint, kernels_wgpu::Capability::Baseline)
            else {
                continue;
            };
            let body = code(&source);
            // Lines that RECONSTRUCT a flat index from a 2-D grid are not the
            // defect and are dropped first.
            //
            // `qmm_t.wgsl`'s cast is the shape: `idx = gid.x + gid.y *
            // groups.x * 32u` over a `@workgroup_size(32, 2, 2)`, which is a
            // flat walk that needs more than one dimension because a 1-D grid
            // of `count` would pass the per-dimension workgroup limit. It
            // reads `gid.y` and is right to.
            //
            // The tell is `num_workgroups`: a body that mistakes y for a ROW
            // index has no use for the grid's width, and a body that flattens
            // cannot work without it. Narrowed on that rather than by naming
            // the kernel, and the falsification below is that a sabotaged
            // `geglu_tanh_strided` — the defect this test was written for,
            // measured at 21 wrong rows — still fails.
            let body: String = body
                .lines()
                .filter(|l| !l.contains("groups.") && !l.contains("num_workgroups"))
                .collect::<Vec<_>>()
                .join("\n");
            for name in id_names(&body) {
                assert!(
                    !body.contains(&format!("{name}.y")),
                    "`{entrypoint}` is fired by a body calling `elementwise`, \
                     whose grid is `[width * rows, 1, 1]`, and its body reads \
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
        flat == 10 && checked >= 10,
        "{flat} flat kernels over {checked} entrypoints is not this tree. \
         Every one of them now states an `elementwise` body; a family \
         crossing used to move one from the row count to the body count and \
         must not drop it. It was 10 while this read only rows and `mlp`'s \
         bodies -- reading every crossed body found four more, in families \
         whose rows stated no launch rule at all and which this check had \
         therefore never covered. `layout` retiring moved three of its six \
         from the row side to the body side and dropped the three that are \
         `elementwise_rows`, which were never flat. The rows contributed zero \
         for as long as the table has been empty, so their sweep retiring \
         does not move this"
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
