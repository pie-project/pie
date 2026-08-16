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

// `no_two_rows_claim_the_same_entrypoint` STOOD HERE. Two rows claiming one
// entrypoint would have made `sig_in` order-dependent, and the set comparison
// above could not see it because a set absorbs a duplicate.
//
// There are no rows. The loop ran zero times and the test passed, which is the
// §7 shape: a check does not go false when its subject retires, it goes
// silent. The shader side of the same question is caught earlier and harder
// anyway -- a duplicate `pie:instantiate` is two variants writing one `.spv`,
// and both the audit and `build.rs` refuse it rather than let the second
// silently win. That is the check that is still real.

/// The row count is `kernels-metal`'s, and that is the point rather than a
/// coincidence: this backend's coverage is defined as its sibling's, so the
/// two tables are comparable row for row and a divergence is a statement
/// somebody made rather than a drift nobody noticed.
///
/// Change it here when a kernel is added, deliberately — and when you do, say
/// in the same diff whether Metal grew one too, because a number that moves on
/// one side alone is exactly the fact this assertion exists to surface.
///
/// 481 became 490 with the flash decode: four `sdpa_paged_decode_split`
/// widths, four `sdpa_paged_decode_combine` widths and one sinked fold. Metal
/// did NOT grow those nine, and that is the divergence this assertion is for:
/// they are a Vulkan occupancy fix -- a 128-SM card ran a 16-workgroup
/// dispatch -- and not an operation the shared text can name. No model asks
/// for a "split decode"; `sdpa_paged_decode` is still the op, and which of
/// the three modules serves it is a decision `attn::decode_splits` makes from
/// the history length. The parity tests below compare OPS, so they stay
/// green without an exception list.
///
/// 490 became 496 with `rms_rope`, the fused per-head norm and NEOX
/// rotation. Metal did NOT grow those six, and unlike the flash decode that
/// divergence is TEMPORARY rather than principled: the fusion is a real op
/// that a shared text will have to name, and until Metal carries it the
/// statement is gated so that no plan text can ask for it. Six and not one
/// because the family mirrors `neox`'s exactly -- plain, `freqs` and `prop`,
/// each in a decode and a multi-batch shape -- and only the plain multi-batch
/// arm has a routine today. The other five are compiled and unreachable, and
/// that is deliberate: they are what llama-3.1's frequency table and gemma-4's
/// partial rotation will need, and instantiating them one at a time is how a
/// family ends up with five subtly different bodies.
///
/// 99/480 became 100/481 with `add_bias`, and Metal grew the same row in the
/// same diff. It was written here first — the Qwen-2 biases are a Vulkan
/// wrong-answer this driver could measure against a CPU oracle — and closing
/// it on the Metal side too was the only honest option: the shared text can
/// name an op only if some Metal kernel implements it, so leaving Metal short
/// would have meant an exception list on the parity tests below, which is
/// precisely how the next real divergence gets waved through.
#[test]
fn the_table_is_one_hundred_and_one_kernels_over_four_hundred_and_ninety_six_entrypoints() {
    // Rows PLUS retired rows: `.wiki/kernel-x/refactor-bigplan.md` §7 empties
    // the table family by family, and coverage is what the two together name.
    // The hundred is the invariant; which side of the crossing a kernel sits
    // on is not.
    // `KERNELS.len() + retired_rows().len()` STOOD HERE, because §7 empties
    // the table family by family and coverage was what the two together
    // named. Every family has crossed, so the first term is 0 and the retired
    // list carries the whole hundred on its own.
    assert_eq!(kernels_vulkan::retired_rows().len(), 101);
    assert_eq!(kernels_vulkan::entrypoints().len(), 496);
}

// `every_entrypoint_resolves_through_sig_in` STOOD HERE. It walked the shader
// tree's `pie:instantiate` names and required each to resolve through
// `kernels::sig_in(KERNELS, ..)`, skipping the ones a crossed family had
// retired. Every family crossed, so every name hit the skip and the loop
// asserted nothing 481 times.
//
// What it was really asking -- does every entrypoint the shaders declare have
// an OWNER -- is asked, and asked better, by
// `driver-vulkan/src/arm.rs`'s `every_entrypoint_a_plan_can_name_finds_an_arm`:
// it sweeps `kernels_vulkan::entrypoints()` against the stem registry that
// actually serves a launch, so a miss is a symbol a plan can name and no code
// can run. That sweep found 363 unreachable entrypoints on its first run,
// which is the answer this one had stopped being able to give.

// `every_row_names_the_shader_that_defines_it` STOOD HERE, and it is the one
// worth reading the loss of carefully.
//
// Which shader defines a kernel used to be a `//` comment beside its row, on
// 57 of the 100, and two were WRONG by the time anyone looked --
// `qmv_wide_strided` said `quant/qmm_t.slang` when its instantiations are in
// `quant/qmv.slang`, and `copy_logits_bf16` named a directory that does not
// exist. Neither could fail anything. So the comments became the row's `file`
// field and this test made the field load-bearing: open the stated shader,
// read its `pie:instantiate` directives, and require the row's own host name
// among them.
//
// There are no rows and no `file` field. `routine::Routine` states a `file`
// too, and holding it to the shader the same way is a real check somebody
// should write -- it is not written, and this comment is here so that the gap
// is a known one rather than a discovered one. `build.rs` still refuses a
// `pie:instantiate` it cannot compile, so a shader nobody wrote is caught;
// what is NOT caught is a routine pointing at the wrong existing shader.

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

// `every_row_states_the_same_facts_kernels_metal_does` STOOD HERE, with
// `every_row_asks_for_the_same_operands_kernels_metal_does` after it. They
// scraped `kernels-metal/src`'s `kernel!` calls out of the SOURCE TEXT -- the
// sibling does not build off macOS, so its rows could not be read as values --
// and compared them field for field and operand for operand against this
// crate's table. They were the fleet's strongest parity gate: a fact stated on
// one side and not the other failed here, in text, rather than in whichever
// driver read it.
//
// Both sides are empty. `kernels-metal` finished Stage 4 first and
// `kernels-vulkan` has now, so the scrape reads zero rows from a table of
// zero and every guard the tests carried against reading nothing -- and they
// carried three -- is satisfied by nothing on both sides at once.
//
// What replaces them, and it is the same claim with the tables taken out of
// the middle: `kernels/tests/shader_backends_agree.rs` compares ROUTINE
// signatures across all three backends. That gate grows as this one shrinks,
// which is the trade Stage 3 was for. `kernels-wgpu/tests/entrypoints.rs`
// records the same retirement from the last crate that still has rows.
//
// The one divergence these carried IS resolved now, and it is worth saying
// which way. It was operand 13 of the six `sdpa_paged_*` rows: metal read the
// text's scalar, wgpu read the fire's. All three planes now say
// `Ask<keys::AttentionMaskStride, u32>` and each driver answers for its own
// fire -- this one and wgpu with the pitch of the mask they staged, metal
// with zero, because metal stages an enable word per token and no mask. The
// entry in `kernels`'s `DRIFTED` went when the sentence became true, and the
// gate that read `driver-metal`'s sources from the outside went with it.
