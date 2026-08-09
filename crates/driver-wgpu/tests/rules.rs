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
//! the same fifteen. A divergence is not a difference of opinion about a grid;
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

    let (mine, theirs) = (served(&wgpu), served(&vulkan));
    assert_eq!(mine.len(), 15, "this crate's `SERVED` parsed as {mine:?}");
    assert_eq!(
        theirs.len(),
        15,
        "`driver-vulkan`'s `SERVED` parsed as {theirs:?}"
    );
    assert_eq!(
        mine,
        theirs,
        "the two ports serve different blocks. Only in wgpu: {:?}; only in \
         vulkan: {:?}.",
        mine.difference(&theirs).collect::<Vec<_>>(),
        theirs.difference(&mine).collect::<Vec<_>>()
    );

    // And the two halves partition the fleet, which is the claim that makes
    // the two comparisons above worth making: fifteen served plus twenty-four
    // refused plus `Unstated` is every rule there is, so agreeing on both
    // lists is agreeing on all of it.
    assert_eq!(
        served(&wgpu).len() + unruled(&wgpu).len() + 1,
        kernels::LaunchRule::ALL.len(),
        "the ledgers no longer cover the vocabulary"
    );
}

/// No row of this backend's table names a rule this backend refuses.
///
/// The `Unruled` list is only correct if the table agrees with it. This asks
/// the TABLE rather than the file: a row that stated `RecurrentScan` would
/// compile, plan, and then refuse at every fire, which is a worse failure than
/// not compiling — and a worse one to diagnose, because the row looks fine.
///
/// It lives here beside the cross-check rather than in `geometry.rs` because
/// it is the same question asked of the other side of the same seam.
#[test]
fn no_row_of_the_wgpu_table_names_a_rule_this_backend_refuses() {
    let refused = unruled(&own());
    let mut named = BTreeSet::new();
    for sig in kernels_wgpu::KERNELS {
        let rule = format!("{:?}", sig.launch);
        assert!(
            !refused.contains(&rule),
            "`{}` states {rule}, which `geometry.rs` refuses as unruled",
            sig.symbol
        );
        named.insert(rule);
    }
    // The table is 99 rows and states a handful of distinct rules; a run that
    // saw one would mean the iteration is wrong rather than the table narrow.
    assert!(
        named.len() > 5,
        "the table's rows name only {named:?}, so this checked almost nothing"
    );
}
