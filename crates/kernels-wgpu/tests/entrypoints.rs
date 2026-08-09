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
//! `entrypoints.generated.txt` is still committed, and it is still checked. Not
//! because anything needs it to run, but because it is the artifact a human
//! diffs against `kernels-metal`'s and `kernels-vulkan`'s copies — a set
//! difference in a review, rather than a number in a test log.
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

fn census() -> PathBuf {
    manifest().join("entrypoints.generated.txt")
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

#[test]
fn the_table_names_exactly_what_the_shaders_instantiate() {
    let shaders = from_the_shaders();
    let table = from_the_table();

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

/// The committed census is the table's product, so a reviewer can diff it.
///
/// Regenerate with `cargo run -p kernels-wgpu --example write_census`. It is
/// checked rather than generated at build time because the point of a committed
/// artifact is that a CHANGE to it shows up in a diff — a file the build
/// rewrites silently is a file nobody reads.
#[test]
fn the_committed_census_is_what_the_table_produces() {
    let text = std::fs::read_to_string(census()).unwrap_or_else(|e| {
        panic!(
            "cannot read {}: {e}. Regenerate it with \
             `cargo run -p kernels-wgpu --example write_census`",
            census().display(),
        )
    });
    let filed: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let table = kernels_wgpu::entrypoints();

    assert_eq!(
        filed,
        table.iter().map(String::as_str).collect::<Vec<_>>(),
        "entrypoints.generated.txt has drifted from the table. Regenerate it \
         with `cargo run -p kernels-wgpu --example write_census`",
    );
}

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
    assert_eq!(kernels_wgpu::KERNELS.len(), 100);
    assert_eq!(kernels_wgpu::entrypoints().len(), 481);
}

/// The parity with `kernels-metal` above, actually compared rather than
/// asserted as a number.
///
/// A matching count is much weaker than it looks — two tables can agree on 480
/// while disagreeing about which 480, and that is the drift the claim exists to
/// catch. Both crates commit a generated entrypoint list, so the sets can be
/// diffed directly, and a dev-dependency is not needed (`kernels-metal` does
/// not build off macOS, which is precisely why the comparison has to go through
/// the file rather than the crate).
#[test]
fn every_entrypoint_is_one_kernels_metal_also_has() {
    let sibling = manifest().join("../kernels-metal/entrypoints.generated.txt");
    let text = std::fs::read_to_string(&sibling)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", sibling.display()));

    let metal: BTreeSet<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let ours = from_the_table();

    let extra: Vec<&String> = ours
        .iter()
        .filter(|n| !metal.contains(n.as_str()))
        .collect();
    let missing: Vec<&&str> = metal.iter().filter(|n| !ours.contains(**n)).collect();
    assert!(
        extra.is_empty() && missing.is_empty(),
        "the two tables have drifted apart\n  only in wgpu:  {extra:?}\n  \
         only in metal: {missing:?}",
    );
}

/// Every entrypoint resolves through the public lookup `model-compiler` uses,
/// at every point of every axis.
///
/// `sig_in` tries exact matches first and axis matches second, so a base that
/// shadows a sibling's point surfaces here and nowhere else.
#[test]
fn every_entrypoint_resolves_through_sig_in() {
    for name in kernels_wgpu::entrypoints() {
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
    let known = from_the_table();
    for (file, text) in kernels_wgpu::SOURCES {
        let variants = kernels_wgpu::instantiations(text)
            .unwrap_or_else(|why| panic!("kernels/{file}: {why}"));
        for variant in variants {
            assert!(
                known.contains(&variant.entrypoint),
                "kernels/{file}:{}: `{}` is instantiated but the table does not name it",
                variant.line,
                variant.entrypoint,
            );
        }
    }
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
fn every_row_states_the_same_facts_kernels_metal_does() {
    let ours = scrape_fields(&manifest().join("src"));
    let theirs = scrape_fields(&manifest().join("../kernels-metal/src"));

    assert_eq!(
        ours.len(),
        kernels_wgpu::KERNELS.len(),
        "the scrape read {} rows out of this crate's own {} — a text-based \
         check that silently reads nothing is a check that passes",
        ours.len(),
        kernels_wgpu::KERNELS.len(),
    );
    assert_eq!(
        theirs.len(),
        kernels_wgpu::KERNELS.len(),
        "the scrape read {} rows out of `kernels-metal`, which should have the \
         same {}",
        theirs.len(),
        kernels_wgpu::KERNELS.len(),
    );

    let mut wrong = Vec::new();
    for (name, want) in &theirs {
        let Some(got) = ours.get(name) else {
            wrong.push(format!(
                "`{name}`: kernels-metal has this row and we do not"
            ));
            continue;
        };
        if got != want {
            wrong.push(format!(
                "`{name}`:\n    metal: {want:?}\n    wgpu:  {got:?}"
            ));
        }
    }
    for name in ours.keys() {
        if !theirs.contains_key(name) {
            wrong.push(format!(
                "`{name}`: we have this row and kernels-metal does not"
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
fn every_row_asks_for_the_same_operands_kernels_metal_does() {
    let ours = scrape_operands(&manifest().join("src"));
    let theirs = scrape_operands(&manifest().join("../kernels-metal/src"));

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
    for (name, want) in &theirs {
        let Some(got) = ours.get(name) else { continue };
        if got.len() != want.len() {
            wrong.push(format!(
                "`{name}`: metal states {} operands, we state {}",
                want.len(),
                got.len(),
            ));
            continue;
        }
        for (at, (want, got)) in want.iter().zip(got).enumerate() {
            if want != got {
                wrong.push(format!(
                    "`{name}` operand {at}: metal `{want}`, wgpu `{got}`"
                ));
            }
        }
    }

    assert!(
        wrong.is_empty(),
        "the two tables ask for different operands:\n{}",
        wrong.join("\n"),
    );
}

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
/// `every_row_states_the_same_facts_kernels_metal_does` compares a NAMED list
/// of fields, and a named list rots: a field added to `kernels::KernelSig`
/// upstream and not added to [`FIELDS`] is a field the parity check silently
/// stops comparing. That is the direction that matters, because the whole
/// purpose of that check is to notice that somebody edited one table only.
///
/// It is not hypothetical. `publishes_aux` was added to `KernelSig` while this
/// crate was being written, and the scrape went on passing without it.
/// (That field has since left `KernelSig` altogether — step 9 measured it
/// CUDA-only and consolidated it onto `kernels_cuda_new::x::Contract`. The
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
         `every_row_states_the_same_facts_kernels_metal_does` does not compare. \
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
const FIELDS: [&str; 12] = [
    "grid_param",
    "rows_param",
    "head_param",
    "heads_param",
    "launch",
    "needs",
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
const UNCOMPARED: [(&str, &str); 4] = [
    ("name", "the key this map is built on"),
    ("symbol", "the key this map is built on"),
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
        for line in text.lines() {
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
