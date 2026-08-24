//! Do the two ports refuse the same blocks?
//!
//! `kernels::LaunchRule` is the whole fleet's vocabulary: thirty-seven
//! variants, most of them CUDA's. `kernels-wgpu`'s table and
//! `kernels-vulkan`'s are both `kernels-metal`'s coverage, row for row — that
//! is the premise both ports were written under, and it is stated in both
//! crates' module docs.
//!
//! If the premise holds, the two backends must be unable to run **exactly the
//! same set of blocks**. So `driver-wgpu`'s `Unruled` list and
//! `driver-vulkan`'s must be the same twenty-four names, and their served lists
//! the same seventeen. Both numbers are asserted below, which is the only
//! reason they are written here at all: the served one said "fifteen" while
//! the assertion beneath it was moved to sixteen and then seventeen, and
//! `geometry.rs` carried a third copy that still said twenty-one. A count
//! repeated away from the assertion that owns it is a count nobody updates.
//! A divergence is not a difference of opinion about a grid;
//! it means one of the two ports quietly grew or lost a row, and the useful
//! moment to learn that is here rather than when a model runs on one backend
//! and is declined by the other.
//!
//! # Why this reads the sibling's SOURCE
//!
//! Because it must not depend on it. `driver-wgpu` takes no dependency on
//! `driver-vulkan` and should not: they are peers, `tests/pure.rs` measures
//! this crate's closure, and an edge added to satisfy a test would be an edge
//! shipped to every user. Reading the text is the cheap way to compare two
//! ledgers that are, by design, written down rather than derived.
//!
//! It is a text comparison and so it is exact about what it claims: it says
//! the two files NAME the same rules in their `Unruled` arm and their `SERVED`
//! list. It does not claim the two compute the same grids — they do not, and
//! `geometry.rs` says where they differ and why (a bf16 lane owns a pair here,
//! so decode attention halves).
//!
//! # It skips rather than fails when the sibling is absent
//!
//! A checkout without `crates/driver-vulkan` is a legitimate state — a
//! published crate, a sparse checkout, a tree where the Vulkan shell was
//! retired. A test that failed there would be a test that gets deleted. It
//! prints what it skipped and why, which is the difference between a skip and
//! an omission.

use std::collections::BTreeSet;
use std::path::PathBuf;

/// Where the sibling's geometry lives, if it is checked out.
fn sibling() -> Option<String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../driver-vulkan/src/geometry.rs")
        .canonicalize()
        .ok()?;
    std::fs::read_to_string(path).ok()
}

/// This crate's own geometry, read the same way, so the two are parsed by one
/// function and cannot disagree because the parser was written twice.
fn own() -> String {
    std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/geometry.rs"))
        .expect("a crate can read its own source")
}

/// The rules named in the one match arm that ends in `Unruled(rule)`.
///
/// Anchored on `RecurrentScan` because it is the first name in both files'
/// arm, and terminated by the `=> return` that closes it. A parse that drifted
/// would come back empty or absurdly large, and both are asserted against
/// below rather than trusted.
fn unruled(source: &str) -> BTreeSet<String> {
    let from = source
        .find("Rule::RecurrentScan")
        .expect("both files refuse the mamba scan by name");
    let to = source[from..]
        .find("=> return Err(Ungeometric::Unruled(rule)),")
        .expect("the arm ends in the refusal it is for")
        + from;
    names(&source[from..to])
}

/// The rules listed in the `SERVED` const.
fn served(source: &str) -> BTreeSet<String> {
    let from = source
        .find("const SERVED: &[Rule] = &[")
        .expect("both files keep a ledger of what they serve");
    let to = source[from..].find("];").expect("the list is closed") + from;
    names(&source[from..to])
}

/// Every `Rule::Name` in a span.
fn names(span: &str) -> BTreeSet<String> {
    span.split("Rule::")
        .skip(1)
        .filter_map(|rest| {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            (!name.is_empty()).then_some(name)
        })
        .collect()
}

/// The two ports refuse the same rules and serve the same rules.
#[test]
fn the_two_ports_of_metals_table_decline_the_same_blocks() {
    let Some(vulkan) = sibling() else {
        println!(
            "SKIPPED: crates/driver-vulkan is not in this checkout, so there is \
             no second ledger to compare against. This test claims nothing when \
             it cannot read one."
        );
        return;
    };
    let wgpu = own();

    let (mine, theirs) = (unruled(&wgpu), unruled(&vulkan));
    // A parse that came back empty would make every claim below vacuous, and a
    // parse that swallowed the whole file would make them meaningless the
    // other way. Both are ruled out by a number before anything is compared.
    assert_eq!(
        mine.len(),
        24,
        "this crate's `Unruled` arm parsed as {mine:?}, which is not the \
         twenty-four rules it names"
    );
    assert_eq!(
        theirs.len(),
        24,
        "`driver-vulkan`'s `Unruled` arm parsed as {theirs:?}"
    );
    assert_eq!(
        mine,
        theirs,
        "the two ports of `kernels-metal`'s table refuse different blocks. \
         Only in wgpu: {:?}; only in vulkan: {:?}. One of the two tables has \
         grown or lost a row, and the tables are supposed to be the same one.",
        mine.difference(&theirs).collect::<Vec<_>>(),
        theirs.difference(&mine).collect::<Vec<_>>()
    );

    // The two ledgers PARTITION the rule space, which is the arithmetic the
    // prose in `geometry.rs` narrates and nothing checked. The exhaustive
    // match makes a NEW rule a compile error -- that is how `SdpaTiled` and
    // `SdpaMma` were caught -- but it cannot see a rule that is in the
    // `SERVED` const AND refused by an arm, because the const is a ledger
    // rather than the code. Served plus refused must be every real rule, once
    // each.
    //
    // `Unstated` is the one that is neither: it means a row that exists and
    // has not said how to launch, which is a different sentence from a block
    // that was never ported.
    let real = kernels::LaunchRule::ALL.len() - 1;
    assert_eq!(
        served(&wgpu).len() + mine.len(),
        real,
        "this port serves {} rules and refuses {}, which is not the {real} \
         real ones `LaunchRule::ALL` has. A rule in both ledgers, or in \
         neither, is one the arms and the const disagree about",
        served(&wgpu).len(),
        mine.len(),
    );

    let (mine, theirs) = (served(&wgpu), served(&vulkan));
    assert_eq!(mine.len(), 17, "this crate's `SERVED` parsed as {mine:?}");
    assert_eq!(
        theirs.len(),
        17,
        "`driver-vulkan`'s `SERVED` parsed as {theirs:?}"
    );

    // The rules this port serves and `driver-vulkan` does not. There are
    // none, and the list stays because an EMPTY difference is a claim while a
    // missing list is a silence.
    //
    // It has held two and shed both, each time by failing rather than by
    // anyone noticing. `LaunchRule::SdpaTiled` arrived upstream with a stated
    // `sdpa_paged_tiled` in `kernels-metal` and this backend served it first;
    // `LaunchRule::SdpaMma` arrived a rebase later and this backend served
    // that first too. Both times the sibling caught up within a day, and both
    // times the way this file learned was the assertion below.
    //
    // The two rules are ONE arm here. Metal separates them because a
    // matrix-unit threadgroup is 128 threads where the scalar one is 1024;
    // WGSL has no matrix unit, so `attn/sdpa_paged_mma.wgsl` is a scalar body
    // wearing Metal's entrypoint names and takes the tiled grid exactly.
    const AHEAD: &[&str] = &[];
    let only_mine: std::collections::BTreeSet<&str> =
        mine.difference(&theirs).map(String::as_str).collect();
    assert_eq!(
        only_mine,
        AHEAD
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>(),
        "the rules this port serves and `driver-vulkan` does not are no longer \
         the ones `AHEAD` names. If vulkan has caught up, empty it; if this \
         port grew another rule alone, add it and say why."
    );
    assert!(
        theirs.difference(&mine).next().is_none(),
        "`driver-vulkan` serves a rule this port does not: {:?}",
        theirs.difference(&mine).collect::<Vec<_>>()
    );

    // And the two halves partition the fleet, which is the claim that makes
    // the two comparisons above worth making: sixteen served plus the refused
    // ones plus `Unstated` is every rule there is, so agreeing on both lists
    // is agreeing on all of it.
    assert_eq!(
        served(&wgpu).len() + unruled(&wgpu).len() + 1,
        kernels::LaunchRule::ALL.len(),
        "the ledgers no longer cover the vocabulary"
    );
}

// RETIRED: THE TABLE IS EMPTY, and no routine names a `LaunchRule` at all.
//
// It asked the TABLE what the `Unruled` list above asks the file: a row that
// stated `RecurrentScan` would compile, plan, and then refuse at every fire —
// a worse failure than not compiling, and a worse one to diagnose, because the
// row looks fine. It lived here beside the cross-check because it was the same
// question asked of the other side of the same seam.
//
// It BECAME BLIND, not true: zero rows name zero rules, so nothing was
// established about any of them. Its own floor (`named.len() > 5`) is what
// said so out loud rather than letting an empty iteration pass.
//
// There is no routine-plane counterpart to write, and that is the point rather
// than a gap. A `LaunchRule` was a ROW's way of describing a grid to a driver
// that had to build one on its behalf. A routine states its grid directly —
// `kernels::shader::{elementwise, elementwise_rows, ...}` or a helper of its
// own — so there is no name to disagree with `geometry.rs` about, and the
// class of failure this guarded against cannot be spelled any more.
//
// What replaces it is stronger and is already running: `driver-wgpu`'s
// `no_module_reads_a_grid_axis_its_rule_leaves_flat` compares the grid a body
// actually asks for against the axes its MODULE actually reads, over every
// entrypoint a corpus of real fires reaches. That is a claim about the shader
// rather than about a rule's name, and it caught two things this never could:
// `residual_add` asking a 2-D grid of a `gid.x`-only shader, and
// `gdn_prep`'s unread `global_invocation_id` making reflection report three
// axes it never touches.

// `the_sibling_gates_this_port_has_no_twin_for_are_named` STOOD HERE, and its
// SUBJECT is what left rather than its premise.
//
// It scanned `tests/gpu/tests` for `vulkan_*` and `wgpu_*` gate files, took the
// sibling gates with no wgpu twin, and required that set to equal a `NO_TWIN`
// table naming what covered each one here. Two self-cleaning rules: a twin that
// gets written leaves the list, a gate the sibling adds joins it.
//
// THERE ARE NO SUCH FILES ANY MORE, on either side. R3 (`6393b8ddb`) deleted
// the vulkan gates along with `model-legacy`, and this port's three went with
// them; `tests/gpu/tests` holds `cuda_*` and nothing else. So the test read a
// directory it correctly identified, found zero of each, and tripped its own
// "a port with none of either is a scan that matched nothing" guard -- which
// was the right guard and the right answer: the comparison has no subject.
//
// It has been in that state since R3 and nobody saw it, because this crate was
// quarantined out of the workspace at the same commit and its suite has not run
// since. Rejoining the workspace at P5b is what surfaced it, which is the
// argument for rejoining.
//
// It is deleted rather than skipped because a skip would keep a `NO_TWIN` table
// of prose about nine files that do not exist, and the one finding in it worth
// carrying is recorded where it belongs: `common::wgpu_standalone_toml`
// hard-coded `[model] name = "qwen3"`, so no wgpu gate could serve a second
// architecture at all -- one of the nine was unwritable rather than unwritten.
// If `tests/gpu` grows shader-plane gates again, this test is worth rewriting
// against them from scratch.

// `the_first_ported_routine_asks_for_the_grid_its_row_asked_for` STOOD HERE.
//
// It held `kernels_wgpu::layout::ple_combine` against `driver_wgpu::geometry`:
// it read the REAL shader's `@workgroup_size` -- not a transcription of it --
// asked `geometry::groups` what `LaunchRule::Elementwise` wants over four
// rectangles, ran the routine body against a probe `Encode` that recorded only
// the LANES it asked for, and required the two equal after the driver's own
// `div_ceil`. The danger it was written for is silent: a body that computes a
// DIFFERENT grid still dispatches, still returns `Ok`, and produces wrong
// numbers rather than a refusal.
//
// It was deliberately in this crate rather than in `kernels-wgpu`, which
// cannot depend on it: that crate's own half compared against a transcribed
// `width * rows`, and this one compared against the function that decides it.
//
// `kernels_wgpu::layout` is a `#[claims] impl` now and states no
// `ple_combine`, so there is no body left to ask. The claim it made is the one
// a claim body's grid still owes, and whoever fires one on this plane owes the
// same comparison against `geometry::groups`.
