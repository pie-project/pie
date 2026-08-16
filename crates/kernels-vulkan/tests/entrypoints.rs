//! The table's product, against the shader tree.
//!
//! Invariant (1), which this backend inherits from its Metal sibling:
//!
//! > every entrypoint in `kernels/` resolves to exactly one (row, axis point),
//! > and every (row, axis point) to exactly one entrypoint
//!
//! Both halves are read here, in one hermetic test binary. The shader half used
//! to arrive as a committed `entrypoints.generated.txt` written by
//! `scripts/vulkan-kernel-audit.py`, on the reasoning that the census was the
//! toolchain's to produce — but it never was. A variant is DECLARED on a
//! `// pie:instantiate` line, so reading the set is a parse, which is what
//! `build.rs` already does and what [`from_the_shaders`] does below. Only
//! proving a declared variant COMPILES needs `slangc`, and that half stays in
//! the audit script where a box without a Vulkan toolchain never runs it.
//!
//! What the file bought, and what its removal costs, is the cross-backend
//! comparison: `kernels-metal` cannot expand its own census without a C
//! preprocessor, so parity with it was a diff of two committed artifacts and
//! there is no hermetic replacement for it here.

use std::collections::BTreeSet;
use std::path::PathBuf;

/// Every entrypoint the shader tree instantiates, from the directives.
///
/// A `@tier` variant is another compile of an entrypoint that already exists at
/// baseline — same name, different defines — so only the baseline lines name
/// the set. `every_tier_has_a_baseline_beneath_it` is what holds that claim up.
fn from_the_shaders() -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for (_, text) in shader_sources() {
        for line in text.lines() {
            let Some(rest) = line
                .trim_start()
                .strip_prefix("//")
                .map(str::trim_start)
                .and_then(|r| r.strip_prefix("pie:instantiate"))
            else {
                continue;
            };
            let mut words = rest.split_whitespace();
            let Some(name) = words.next() else { continue };
            match words.next().and_then(|w| w.strip_prefix('@')) {
                Some(tier) if tier != "baseline" => continue,
                _ => {
                    out.insert(name.to_string());
                }
            }
        }
    }
    out
}

#[test]
fn the_table_names_exactly_what_the_shaders_instantiate() {
    let shaders = from_the_shaders();
    let table: BTreeSet<String> = kernels_vulkan::entrypoints().into_iter().collect();

    let undeclared: Vec<_> = shaders.difference(&table).collect();
    assert!(
        undeclared.is_empty(),
        "{} entrypoints exist in kernels/ that no row declares. A new \
         instantiation needs a row, or a point on an existing row's axis:\n{:#?}",
        undeclared.len(),
        undeclared
    );

    let phantom: Vec<_> = table.difference(&shaders).collect();
    assert!(
        phantom.is_empty(),
        "{} entrypoints are declared that no shader instantiates. An axis whose \
         product over-generates is the usual cause — see `sdpa_paged_decode`, \
         which lists its tails for exactly this reason:\n{:#?}",
        phantom.len(),
        phantom
    );
}

/// Two rows claiming one entrypoint would make `sig_in` order-dependent, and
/// the set comparison above cannot see it: a duplicate is absorbed by the set.
///
/// The shader side of the same question is caught earlier and harder — a
/// duplicate `pie:instantiate` is two variants writing one `.spv`, so both the
/// audit and `build.rs` refuse it rather than let the second silently win.
#[test]
fn no_two_rows_claim_the_same_entrypoint() {
    let mut seen: std::collections::BTreeMap<String, &str> = Default::default();
    for row in kernels_vulkan::KERNELS {
        for name in row.entrypoints() {
            if let Some(other) = seen.insert(name.clone(), row.name) {
                panic!("`{name}` is claimed by both `{other}` and `{}`", row.name);
            }
        }
    }
}

/// The row count is `kernels-metal`'s, and that is the point rather than a
/// coincidence: this backend's coverage is defined as its sibling's, so the
/// two tables are comparable row for row and a divergence is a statement
/// somebody made rather than a drift nobody noticed.
///
/// Change it here when a kernel is added, deliberately — and when you do, say
/// in the same diff whether Metal grew one too, because a number that moves on
/// one side alone is exactly the fact this assertion exists to surface.
///
/// 99/480 became 100/481 with `add_bias`, and Metal grew the same row in the
/// same diff. It was written here first — the Qwen-2 biases are a Vulkan
/// wrong-answer this driver could measure against a CPU oracle — and closing
/// it on the Metal side too was the only honest option: the shared text can
/// name an op only if some Metal kernel implements it, so leaving Metal short
/// would have meant an exception list on the parity tests below, which is
/// precisely how the next real divergence gets waved through.
#[test]
fn the_table_is_one_hundred_kernels_over_four_hundred_and_eighty_one_entrypoints() {
    // Rows PLUS retired rows: `.wiki/kernel-x/refactor-bigplan.md` §7 empties
    // the table family by family, and coverage is what the two together name.
    // The hundred is the invariant; which side of the crossing a kernel sits
    // on is not.
    assert_eq!(
        kernels_vulkan::KERNELS.len() + kernels_vulkan::retired_rows().len(),
        100
    );
    assert_eq!(kernels_vulkan::entrypoints().len(), 481);
}

/// The parity with `kernels-metal` above was checked here, entrypoint for
/// entrypoint, by diffing this crate's committed census against that crate's.
///
/// Both artifacts are gone, and this comparison went with them. It cannot be
/// rebuilt in-process: `kernels-metal` does not build off macOS, so a
/// dev-dependency is not available, and its census cannot be parsed the way
/// this crate's now is — the Metal axis product is written as `instantiate_*`
/// macros that only a C preprocessor expands. The count above is what remains,
/// and it is strictly weaker: two tables can agree on 481 while disagreeing
/// about which 481.
/// Every entrypoint resolves through the public lookup `model-ir` uses,
/// at every point of every axis. `sig_in` tries exact matches first and axis
/// matches second, so a base that shadows a sibling's point surfaces here.
#[test]
fn every_entrypoint_resolves_through_sig_in() {
    let retired = kernels_vulkan::retired();
    for name in from_the_shaders() {
        // A crossed family resolves through `driver-vulkan/src/arm.rs`'s stem
        // lookup instead. What must hold is that the entrypoint has SOME
        // owner, not that the owner is a row.
        if retired.contains(&name.as_str()) {
            continue;
        }
        assert!(
            kernels::sig_in(kernels_vulkan::KERNELS, &name).is_some(),
            "`{name}` does not resolve"
        );
    }
}

/// Every row names a file, it exists, and the shader in it defines the row.
///
/// Metal can leave the first part to the runtime shader compiler, which fails
/// at model load with the path in hand. Vulkan cannot: the file is read at
/// BUILD time by `build.rs`, so a row pointing at a shader nobody wrote would
/// be a pipeline the shell asks for and cannot create, one layer away from the
/// row that named it.
///
/// The other two parts are why this is not just an existence check. Which
/// shader defines a row used to be recorded in a `//` comment beside it, on
/// 57 of the 100 rows, and a comment is not checked -- two of them were wrong
/// by the time anyone looked. `qmv_wide_strided` said `quant/qmm_t.slang` and
/// its instantiations are in `quant/qmv.slang`; `copy_logits_bf16` said
/// `ptir/ptir/logits_copy.slang`, a directory deep in a tree that has one.
/// Neither could fail anything, which is exactly the problem: a reader
/// following either lands in the wrong shader and the build says nothing.
///
/// So the comments became the `file` field on every row, and this test makes
/// the field load-bearing by reading the `pie:instantiate` directives out of
/// the stated shader and requiring the row's own host name among them. A row
/// that moves to another file now fails here rather than misleading whoever
/// reads it next. `lowering::dispatch` on the Metal side opens this same
/// field, which is the other reason a row without one is a row that cannot be
/// found.
#[test]
fn every_row_names_the_shader_that_defines_it() {
    let kernels = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels");
    for row in kernels_vulkan::KERNELS {
        let file = row.file.unwrap_or_else(|| {
            panic!(
                "`{}` states no file; every row names the shader it is \
                 stamped from",
                row.name
            )
        });
        let path = kernels.join(file);
        assert!(
            path.exists(),
            "`{}` states `{file}`, which does not exist",
            row.name
        );
        let text = std::fs::read_to_string(&path).expect("the shader reads");
        // The row's host name is a STEM: the axes stamp suffixes onto it, so
        // `affine_qmv_wide_strided` is declared as
        // `affine_qmv_wide_strided_bfloat16_gs_64_...`. An exact match is
        // right for an unaxised row and a prefix for the rest, and the
        // underscore keeps `gdn_core` from matching `gdn_core_recurrent`,
        // which is a different row in a different shader.
        let host = row.symbol;
        let found = text.lines().filter_map(directive).any(|name| {
            name == host || name.strip_prefix(host).is_some_and(|r| r.starts_with('_'))
        });
        assert!(
            found,
            "`{}` states `{file}`, which instantiates no `{host}`",
            row.name
        );
    }
}

/// The name a `// pie:instantiate` line declares, if this line is one.
///
/// The same anchoring `build.rs` uses, so a `pie:instantiate` mentioned in
/// prose is not read as a declaration here either.
fn directive(line: &str) -> Option<&str> {
    line.trim_start()
        .strip_prefix("//")?
        .trim_start()
        .strip_prefix("pie:instantiate")?
        .split_whitespace()
        .next()
}

/// Every `@tier` directive names an entrypoint that also has a baseline.
///
/// This is the whole of the backward-compatibility guarantee, and it is a test
/// rather than a convention because the failure it prevents is invisible until
/// a specific device runs a specific model: a tiered module with no baseline is
/// an entrypoint that resolves on the author's GPU and on no other.
///
/// `build.rs` asserts the same thing, but only under `--features native` — that
/// is, only on a machine with slangc. This runs everywhere.
#[test]
fn every_tier_has_a_baseline_beneath_it() {
    let mut baseline = BTreeSet::new();
    let mut tiered: Vec<(String, String, String)> = Vec::new();

    for (file, text) in shader_sources() {
        for line in text.lines() {
            let Some(rest) = line
                .trim_start()
                .strip_prefix("//")
                .map(str::trim_start)
                .and_then(|r| r.strip_prefix("pie:instantiate"))
            else {
                continue;
            };
            let mut words = rest.split_whitespace();
            let Some(name) = words.next() else { continue };
            match words.next().and_then(|w| w.strip_prefix('@')) {
                None => {
                    baseline.insert(name.to_string());
                }
                Some(tier) => {
                    assert!(
                        ["baseline", "fp16", "coopmat"].contains(&tier),
                        "{file}: `@{tier}` on `{name}` is not a capability tier",
                    );
                    if tier == "baseline" {
                        baseline.insert(name.to_string());
                    } else {
                        tiered.push((name.to_string(), tier.to_string(), file.clone()));
                    }
                }
            }
        }
    }

    for (name, tier, file) in &tiered {
        assert!(
            baseline.contains(name),
            "{file}: `{name}` is instantiated at tier `{tier}` with no baseline; \
             every entrypoint must resolve on a device with no optional features",
        );
    }
}

/// A tier never invents an entrypoint the table does not name.
#[test]
fn no_tier_names_an_unknown_entrypoint() {
    let known: BTreeSet<String> = kernels_vulkan::entrypoints().into_iter().collect();
    for (file, text) in shader_sources() {
        for line in text.lines() {
            let Some(rest) = line
                .trim_start()
                .strip_prefix("//")
                .map(str::trim_start)
                .and_then(|r| r.strip_prefix("pie:instantiate"))
            else {
                continue;
            };
            let mut words = rest.split_whitespace();
            let Some(name) = words.next() else { continue };
            assert!(
                known.contains(name),
                "{file}: `{name}` is instantiated but the table does not name it",
            );
        }
    }
}

/// Baseline is unsuffixed, so a driver that has never heard of a tier finds the
/// right module knowing only the entrypoint name.
#[test]
fn baseline_modules_are_unsuffixed() {
    use kernels_vulkan::Capability;
    assert_eq!(
        Capability::Baseline.module("rms_single_row_bfloat16"),
        "rms_single_row_bfloat16.spv"
    );
    assert_eq!(
        Capability::Coopmat.module("rms_single_row_bfloat16"),
        "rms_single_row_bfloat16.coopmat.spv"
    );
    // Best first: a driver takes the first tier its device supports.
    assert_eq!(
        *Capability::PREFERENCE.last().expect("non-empty"),
        Capability::Baseline
    );
    assert!(Capability::Baseline.requires().is_empty());
}

/// Every `.slang` under `kernels/`, as `(display path, contents)`.
fn shader_sources() -> Vec<(String, String)> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels");
    let mut out = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("a readable kernels/ directory") {
            let path = entry.expect("a readable entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .extension()
                .is_some_and(|e| e == "comp" || e == "slang")
            {
                let rel = path
                    .strip_prefix(&root)
                    .expect("under kernels/")
                    .display()
                    .to_string();
                out.push((
                    rel,
                    std::fs::read_to_string(&path).expect("a readable shader"),
                ));
            }
        }
    }
    out.sort();
    out
}

/// Every row states the same FACTS about its kernel as the `kernels-metal` row
/// of the same name -- the grid it runs under, and every other scalar the
/// macro can carry.
///
/// The entrypoint-set comparison above is about coverage -- which kernels
/// exist -- and it is blind to what a row SAYS about the ones it has. That gap
/// is not hypothetical: `kernels-metal` grew `KernelSig::heads_param` (and
/// `head_param` before it) to fix a gemma-4 miswrite, this backend's
/// `kv_append`/`kv_append_paged` shaders do the identical pool arithmetic, and
/// the port carried neither field across. The tables agreed on all 480 names
/// while disagreeing about the grid two of them run under, which is a wrong
/// answer rather than a missing one -- an undershot `PerHead` grid writes
/// nothing, so the gap reads back as the zeros the pool was born with and the
/// fire completes.
///
/// It began as a check on those three grid fields and found a third gap --
/// `vnorm_single_row`'s `grid_param` -- on its first run, which is the
/// argument for checking the CATEGORY rather than the fields that prompted it.
/// So it covers every scalar a row can state. Some of those are live on both
/// sides (`launch`, `lacks`, `axes`); `sink`, `in_place`, `whole`, `returns`
/// and `depth_prefix_plan` are stated by no row in either table today, and
/// including them is the cheap half -- they cost nothing until the day one
/// table grows one and the other does not.
///
/// The read is a line at a time, so a field whose value spans lines is
/// compared only as far as its first line. That is a weaker check than it
/// looks for `sdpa_paged_decode`'s multi-line `axes`, and it is still a real
/// one: the fragment is compared on both sides, and the entrypoint-set test
/// above covers what an `axes` list actually produces.
///
/// `kernels-metal` does not build off macOS, so this cannot compare the two
/// `KERNELS` arrays as values. It reads the sibling's SOURCE instead and
/// scrapes the field off the row -- coarse, but the fields are written by a
/// macro whose shape both crates share, and being coarse in the direction of
/// "diff the text" is the safe direction for a check whose whole job is to
/// notice that somebody edited one side only.
#[test]
fn every_row_states_the_same_facts_kernels_metal_does() {
    /// `name -> [(field, value)]` for every scalar field, scraped from a
    /// crate's `src/*.rs`. The macro writes `kernel!(name "symbol"` to open a
    /// row, so the row a field belongs to is the last `kernel!` seen -- and a
    /// field can sit on its own line or inline in the `kernel!` call, so both
    /// spellings are read.
    fn grid_fields(dir: &std::path::Path) -> std::collections::BTreeMap<String, Vec<String>> {
        const FIELDS: [&str; 12] = [
            "grid_param",
            "head_param",
            "heads_param",
            "launch",
            "needs",
            "lacks",
            "sink",
            "in_place",
            "whole",
            "returns",
            "depth_prefix_plan",
            "axes",
        ];
        let mut out: std::collections::BTreeMap<String, Vec<String>> = Default::default();
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
            .map(|e| e.expect("a readable entry").path())
            .filter(|p| p.extension().is_some_and(|e| e == "rs"))
            .filter(|p| p.file_name().is_some_and(|n| n != "lib.rs"))
            .collect();
        files.sort();

        for path in files {
            let text = std::fs::read_to_string(&path).expect("a readable source");
            let mut row: Option<String> = None;
            for line in text.lines() {
                let t = line.trim();
                if let Some(rest) = t.strip_prefix("kernel!(") {
                    let name: String = rest
                        .chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect();
                    if !name.is_empty() {
                        out.entry(name.clone()).or_default();
                        row = Some(name);
                    }
                }
                // The table ends here; see `row_calls`, which had the same
                // gap and was reading a crossed family's tests as operands.
                if t == "];" {
                    row = None;
                }
                let Some(row) = &row else { continue };
                // A field can open the line or sit inline in the `kernel!`
                // call, and both spellings appear in both crates. What is NOT
                // read is prose: the preceding character must be a delimiter,
                // so `axes = ` counts and `... its axes = ...` in a comment
                // does not. Doc lines are skipped outright, since a `///` can
                // legitimately contain `field = value` while quoting code.
                if t.starts_with("//") {
                    continue;
                }
                for field in FIELDS {
                    let Some(at) = t.find(&format!("{field} = ")) else {
                        continue;
                    };
                    if at > 0 {
                        let before = t.as_bytes()[at - 1];
                        if !matches!(before, b' ' | b'(' | b',') {
                            continue;
                        }
                    }
                    let value = &t[at + field.len() + 3..];
                    // An inline field is followed by the rest of the call, so
                    // the value ends at the first top-level comma. A field on
                    // its own line has no such comma and takes the whole tail.
                    let mut depth = 0i32;
                    let mut end = value.len();
                    for (i, c) in value.char_indices() {
                        match c {
                            '(' | '[' | '{' => depth += 1,
                            ')' | ']' | '}' => {
                                if depth == 0 {
                                    end = i;
                                    break;
                                }
                                depth -= 1;
                            }
                            ',' if depth == 0 => {
                                end = i;
                                break;
                            }
                            _ => {}
                        }
                    }
                    out.entry(row.clone())
                        .or_default()
                        .push(format!("{field} = {}", value[..end].trim()));
                }
            }
        }
        out
    }

    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let ours = grid_fields(&src);
    let mut theirs = grid_fields(&src.join("../../kernels-metal/src"));
    for gone in kernels_vulkan::retired_rows() {
        theirs.remove(*gone);
    }

    // The scrape has to find the whole table, and enough facts within it, or
    // this passes by reading nothing -- the exact failure a text-based check
    // invites. Every row states `axes`, so the field count is a lower bound
    // the macro's shape cannot drift under without this firing.
    assert_eq!(
        theirs.len(),
        kernels_vulkan::KERNELS.len(),
        "scraped {} rows from kernels-metal against a table of {}; the macro's \
         shape must have changed and this test is no longer reading it",
        theirs.len(),
        kernels_vulkan::KERNELS.len()
    );
    let facts: usize = theirs.values().map(Vec::len).sum();
    assert!(
        facts >= theirs.len(),
        "scraped {facts} facts over {} rows; every row states at least `axes`, \
         so the scrape has stopped reading some",
        theirs.len()
    );

    let mut wrong: Vec<String> = Vec::new();
    for (row, metal) in &theirs {
        let vulkan = ours.get(row);
        if vulkan != Some(metal) {
            wrong.push(format!(
                "  {row}: metal says {metal:?}, vulkan says {:?}",
                vulkan.map(Vec::as_slice).unwrap_or(&[])
            ));
        }
    }
    for (row, vulkan) in &ours {
        if !theirs.contains_key(row) {
            wrong.push(format!(
                "  {row}: vulkan says {vulkan:?}, metal says nothing"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} rows state a fact differently from the kernels-metal row of the \
         same name. The grid fields say which of the STATEMENT's params give \
         the head shape, and a row that omits one takes the fire's numbers \
         instead -- correct until a deployment states two head shapes, as \
         gemma-4 does. The rest decide aliasing, selection and which \
         entrypoints the row generates:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// Every row asks for its operands in the same order, from the same sources,
/// as the `kernels-metal` row of the same name.
///
/// A row is a claim about how a kernel is CALLED, and until this test the only
/// part of that claim compared across the two backends was the set of
/// entrypoint names. That gap was not theoretical: it hid a missing
/// `heads_param` on `kv_append_paged` and a missing `grid_param` on
/// `vnorm_single_row`, both of which
/// `every_row_decides_its_grid_the_way_kernels_metal_does` now covers. This is
/// the rest of the same claim -- the launch rule, and the operand list an
/// operand at a time.
///
/// An operand list is POSITIONAL: slot `i` of the row is buffer `i` of the
/// kernel. So a divergence here is not a style difference, it is one backend
/// handing a kernel the wrong pointer in a slot the other gets right, and the
/// most likely spellings of the mistake -- an inserted slot, a dropped one, a
/// swapped pair -- all keep the LENGTH the same. The comparison is therefore
/// per index, and it reports the first index that differs rather than the two
/// lists.
///
/// It passes with no exceptions, which is worth saying because it did not at
/// first: `mxfp4_qmv_routed_bias` was stated here and not on the Metal side.
/// That was a live gap in `kernels-metal` rather than a divergence to
/// tolerate -- `model-ir` names the symbol -- so it was closed there
/// instead of allow-listed here. An exception list on a test like this is how
/// the next real divergence gets waved through.
#[test]
fn every_row_asks_for_the_same_operands_kernels_metal_does() {
    /// `name -> (launch, [(kind, source)])` scraped from a crate's `src/*.rs`.
    ///
    /// `operands![]` writes one operand per line as `name: Kind <- source,`,
    /// and the NAME is deliberately dropped: the two crates are free to call a
    /// slot whatever reads best beside their own shader, and what has to agree
    /// is what the slot IS. A `Buf` with no `<-` is an unbound placeholder,
    /// which is a real fact about the row and compares as `None`.
    fn row_calls(
        dir: &std::path::Path,
    ) -> std::collections::BTreeMap<String, (Option<String>, Vec<String>)> {
        let mut out: std::collections::BTreeMap<String, (Option<String>, Vec<String>)> =
            Default::default();
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
            .map(|e| e.expect("a readable entry").path())
            .filter(|p| p.extension().is_some_and(|e| e == "rs"))
            .filter(|p| p.file_name().is_some_and(|n| n != "lib.rs"))
            .collect();
        files.sort();

        for path in files {
            let text = std::fs::read_to_string(&path).expect("a readable source");
            let mut row: Option<String> = None;
            for line in text.lines() {
                let t = line.trim();
                if let Some(rest) = t.strip_prefix("kernel!(") {
                    let name: String = rest
                        .chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect();
                    if !name.is_empty() {
                        out.entry(name.clone()).or_default();
                        row = Some(name);
                    }
                }
                // The table ENDS here, and everything after it is ordinary
                // Rust. Without this the last row of a file goes on claiming
                // every `name: value,` line beneath it, and once a family
                // crosses to a routine the file below the table is full of
                // them -- `sample.rs`'s tests state `handle: 10,` and this
                // read 10, 11, 12 and 13 as `argmax_logits`'s four operands.
                // A text scrape that cannot tell where it stopped reading is
                // the failure a text scrape invites, and it fails LOUDLY here
                // only because the whole point of this test is a comparison.
                if t == "];" {
                    row = None;
                }
                let Some(row) = &row else { continue };
                if let Some(rest) = t.strip_prefix("launch = kernels::") {
                    out.entry(row.clone()).or_default().0 =
                        Some(rest.trim_end_matches(',').to_string());
                }
                // `name: Kind,` or `name: Kind <- kernels::Source::X(n),`. The
                // leading `name:` is what distinguishes an operand line from
                // the prose around it, and the trailing comma from a `kernel!`
                // argument that happens to contain a colon.
                let Some(body) = t.strip_suffix(',') else {
                    continue;
                };
                let Some((lhs, rhs)) = body.split_once(": ") else {
                    continue;
                };
                if !lhs.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') || lhs.is_empty() {
                    continue;
                }
                let (kind, source) = match rhs.split_once(" <- kernels::") {
                    Some((k, s)) => (k, Some(s)),
                    None => (rhs, None),
                };
                if !kind.chars().all(|c| c.is_ascii_alphanumeric()) || kind.is_empty() {
                    continue;
                }
                out.entry(row.clone()).or_default().1.push(format!(
                    "{kind}{}",
                    source.map(|s| format!(" <- {s}")).unwrap_or_default()
                ));
            }
        }
        out
    }

    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let ours = row_calls(&src);
    let mut theirs = row_calls(&src.join("../../kernels-metal/src"));
    for gone in kernels_vulkan::retired_rows() {
        theirs.remove(*gone);
    }

    // The scrape has to find the whole table on both sides, or this passes by
    // comparing almost nothing -- the failure a text-based check invites.
    assert_eq!(
        ours.len(),
        kernels_vulkan::KERNELS.len(),
        "scraped {} rows from this crate but the table has {}; the macro's \
         shape must have changed and this test is no longer reading it",
        ours.len(),
        kernels_vulkan::KERNELS.len()
    );
    assert_eq!(
        theirs.len(),
        ours.len(),
        "scraped {} rows from kernels-metal against {} here",
        theirs.len(),
        ours.len()
    );

    // 56 of the 100 rows are unstated by design, so 44 is the whole of what
    // there is to compare and a threshold below it would let the scrape rot
    // silently. Asserting the exact number would just be a second copy of the
    // count `the_table_is_one_hundred_kernels...` already owns.
    //
    // Less one per row this crate has RETIRED, since those are dropped from
    // `theirs` above. That is the loosest the floor may be and still mean
    // something: it falls by exactly what the crossing removes and by nothing
    // else, so a scrape that stops reading the macro is still caught.
    let floor = 44 - kernels_vulkan::retired_rows().len();
    let stated = theirs.values().filter(|(_, ops)| !ops.is_empty()).count();
    assert!(
        stated >= floor,
        "only {stated} kernels-metal rows scraped with any operands at all; \
         {floor} rows state operands and are not retired here, so the scrape \
         has stopped reading some"
    );

    let mut wrong: Vec<String> = Vec::new();
    let mut allowed_seen: Vec<(String, usize)> = Vec::new();
    for (row, (metal_launch, metal_ops)) in &theirs {
        let Some((our_launch, our_ops)) = ours.get(row) else {
            wrong.push(format!("  {row}: kernels-metal has this row and we do not"));
            continue;
        };
        if our_launch != metal_launch {
            wrong.push(format!(
                "  {row}: launch is {our_launch:?} here and {metal_launch:?} in kernels-metal"
            ));
        }
        if our_ops == metal_ops {
            continue;
        }
        if our_ops.len() != metal_ops.len() {
            wrong.push(format!(
                "  {row}: {} operands here, {} in kernels-metal",
                our_ops.len(),
                metal_ops.len()
            ));
        }
        for i in 0..our_ops.len().max(metal_ops.len()) {
            let (a, b) = (our_ops.get(i), metal_ops.get(i));
            if a != b {
                if DELIBERATE.contains(&(row.as_str(), i)) {
                    allowed_seen.push((row.clone(), i));
                    continue;
                }
                wrong.push(format!(
                    "  {row}[{i}]: {} here, {} in kernels-metal",
                    a.map(String::as_str).unwrap_or("<nothing>"),
                    b.map(String::as_str).unwrap_or("<nothing>")
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} row facts differ from kernels-metal. An operand list is \
         POSITIONAL -- slot i of the row is buffer i of the kernel -- so a \
         divergence here is one backend handing a kernel the wrong pointer in \
         a slot the other gets right:\n{}",
        wrong.len(),
        wrong.join("\n")
    );

    // A stale exception is the failure mode an exception list has, so it is
    // checked in BOTH directions: an entry that no longer names a difference
    // is one the tables have since agreed on, and keeping it would hide the
    // next divergence at the same slot.
    let mut stale: Vec<String> = DELIBERATE
        .iter()
        .filter(|(row, at)| {
            !allowed_seen
                .iter()
                .any(|(seen, sat)| seen == row && sat == at)
        })
        .map(|(row, at)| format!("`{row}` operand {at}"))
        .collect();
    stale.sort();
    assert!(
        stale.is_empty(),
        "these rows agree with kernels-metal now, so the exception is stale:\n  {}",
        stale.join("\n  "),
    );
}

/// The operands this table deliberately asks for differently, and why.
///
/// Not a skip list: the test above asserts each entry still names a REAL
/// difference, so an exception that outlives its reason fails rather than
/// accumulates.
///
/// `sdpa_paged_decode{,_sink}` operand 13 is the mask PITCH. `kernels-metal`
/// takes it from `Source::Param(3)` -- a launch parameter, which means from
/// the model TEXT -- and no text can know it: the pitch is a property of the
/// fire the driver is assembling, not of the program being lowered, and every
/// text in the corpus states `0` there. Zero is what the shader reads as
/// "forbid every key of an enabled row", so a mask bound through that slot
/// answers nothing rather than answering wrongly -- the safe direction, and
/// still not the mask the guest asked for.
///
/// This table asks for `Source::AttentionMaskStride`, which the driver
/// answers from the fire it is assembling. `kernels-wgpu` states the same
/// exception for the same two rows.
/// `sdpa_paged_tiled{,_sink}` carry the same exception at the same slot for
/// the same reason, because they are the same operand list with `n_rows`
/// added to the end.
const DELIBERATE: &[(&str, usize)] = &[
    ("sdpa_paged_decode", 13),
    ("sdpa_paged_decode_sink", 13),
    ("sdpa_paged_tiled", 13),
    ("sdpa_paged_tiled_sink", 13),
    ("sdpa_paged_mma", 13),
    ("sdpa_paged_mma_sink", 13),
];
