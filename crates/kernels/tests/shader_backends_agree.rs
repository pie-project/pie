//! The three shader backends' tables, compared.
//!
//! `.wiki/kernel-x/refactor-bigplan.md` §0: the three are **one table written
//! down three times**. Measured over all 300 statements: the 100 names are the
//! same set in all three, `launch` is identical in 100 of 100 rows, `axes` in
//! 100 of 100, `operands` agrees on 94 of 100 — and the statements that differ
//! are drift rather than difference.
//!
//! **Nothing in this workspace compared them**, which is why the drift was
//! found by three greps in an afternoon rather than by a test, and why it had
//! been there long enough that nobody could say which side was right. This is
//! the missing comparison, and it is the same defect class
//! `refactor-plan-followup.md` §0 named once and generally: *a hand-transcribed
//! fact outlived the test that checked it.*
//!
//! # `every_plane_is_answered` STOOD BESIDE THIS AND IS DELETED
//!
//! It held the fourth plane, which this file cannot: the kernel sets are
//! almost disjoint, so CUDA could not be compared by signature and was
//! compared by MACHINERY instead. Every plane's bodies reached for facts with
//! `ctx.ask::<C, keys::K>()`, and a body that asked for one its driver did not
//! answer returned `Refusal::Unstated` and could not fire at all, so the scan
//! held each plane's asks against its driver's answers.
//!
//! There are no asks. `9c1ed0e6e` deleted `keys.rs`, `Asks::ask`,
//! `Source::Named`, the `Answer` enum, `Holds::fact` and `Derivation::ASKED`
//! — *"Zero ctx.ask sites remain in the tree"* — and every fact a body used to
//! reach for is a parameter now. The question the test asked cannot be put:
//! a missing parameter is an arity refusal the binder sees, not an unanswered
//! question the body discovers.
//!
//! What it found before it went is worth keeping, because it is the argument
//! FOR the deletion rather than against it: `driver-cuda` answered NOT ONE of
//! the 59 keys `keys.rs` §M and §L declared, 47 of which were declared and
//! first asked for in a single commit whose subject was about the three shader
//! planes and which carried the CUDA half as a rider. A channel where the
//! asking half can land without the answering half, and stay that way
//! unnoticed across 51 live call sites, is the defect the no-ask series
//! removed at the root.
//!
//! It also left one lesson, and the shape of the successor is the lesson's
//! answer. A scan of source TEXT counts comments: that test believed
//! `driver-cuda` answered `MoeMaxBlocks` because the crate named it once, in a
//! sentence reading *"the signature takes as `Const` because no driver answers
//! `keys::MoeMaxBlocks`"*. It read a denial as a confirmation, and stripping
//! comments moved the count 50 → 51 on that one line alone.
//!
//! `driver-cuda/tests/every_runtime_name_is_answered.rs` is the same question
//! reborn on the operand channel, and it greps nothing. It holds
//! `plan.runtime` — every name a catalogued text mints — against
//! `bind::views::ANSWERED`, two lists of objects the compiler maintains. The
//! original could be fooled by a sentence because it was reading prose; its
//! heir cannot be, because there is no prose in what it reads. That is the
//! upgrade, and it is why this file records the death rather than the test
//! being kept on life support.
//!
//! # Why it is here and not later
//!
//! The bigplan's §3 gate compares `ROUTINES` — derived argument lists — and
//! cannot exist until the ports do. This one compares the tables that exist
//! today, and it is Stage 0's verification: *"settle the seven drifted
//! statements and make the three tables agree, so what gets frozen into three
//! sets of bodies is one decision rather than an accident."* A cleanup that
//! lands without it blesses whatever it finds.
//!
//! When a backend crosses to routines its rows leave `KERNELS`, and this test
//! compares what is left. It goes quiet family by family and the §3 gate takes
//! over; that hand-off is deliberate and is why both exist.
//!
//! # What is compared, and what is deliberately not
//!
//! Compared: the name set, and per name `launch`, `axes`, the operand list
//! (type, nullability and source, in order), the four `*_param` columns,
//! `whole`, `in_place` and `depth_prefix_plan`. These are facts about the
//! KERNEL.
//!
//! Not compared: `file` (three shader languages, three extensions), `symbol`
//! spellings beyond the row name, and anything about a device. Those are
//! properly per-backend, and a gate that failed on them would be a gate people
//! learn to edit.

use std::collections::{BTreeMap, BTreeSet};

use kernels::KernelSig;

/// How many kernels the three backends declare between them.
///
/// It was written as `100` in two assertions and as "the hundred" in five
/// sentences around them, which meant ADDING a kernel turned a gate that
/// checks agreement into a gate that checks a number somebody last edited.
/// `qmv_fast_rmsnorm` is the kernel that made the point, and it made it
/// twice. It arrived correct on two planes and failed here three times
/// over, none of them about the kernel; then a measurement refuted the
/// fusion and it left again, and the departure was one edit here instead of
/// seven. A gate that costs seven edits to tell the truth gets told to be
/// quiet instead.
///
/// Raise it when a kernel is added to all the planes that want it. Lowering
/// it was what `retired_rows` was for, until all three planes deleted theirs;
/// a departing kernel now leaves nothing behind but this number, so lower it
/// and say in the commit what left -- which is also what to do for a kernel
/// that never shipped, that never being a retirement in the first place.
///
/// # It was 100, and 100 was measured over a merged list
///
/// The three planes each declared their routine slice as
/// `#[distributed_slice] static ROUTINES`, and `linkme` keys a slice on the
/// STATIC's identifier rather than on the crate — so the three sections were
/// ONE section, and `kernels_vulkan::declared()` returned vulkan's rows plus
/// wgpu's plus metal's. Every plane answered the same list, so the union was
/// that list and the three could not be compared at all: this gate was reading
/// one table three times and finding it equal to itself.
///
/// The slices carry per-plane names now and the union is honest.
///
/// It was 101 with `rms_rope` -- the fused norm+rope -- crossed by vulkan
/// alone. Metal has crossed it since, so NO kernel in this tree is now
/// exclusive to one plane, and the union held at 101 while [`COMPARED`] rose
/// by one. That is the "a backend has GAINED a family" case the entrypoint
/// census below states in `EXCLUSIVE`, arriving from the direction this list
/// wants: two of the 101 are crossed by exactly two planes -- `rms_rope`
/// (vulkan, metal) and `silu_mul_strided` (wgpu, vulkan) -- and the other 99
/// by all three.
const CENSUS: usize = 101;

/// How many of [`CENSUS`] are crossed by more than one backend, and so are
/// actually compared below.
///
/// The difference from `CENSUS` is the kernels only one plane has. This
/// number moving DOWN while `CENSUS` holds still is the failure worth
/// naming: it means two spellings drifted apart, so the pair silently
/// stopped being a pair and the comparison it used to make disappeared
/// without any test going red. Moving down WITH `CENSUS` is just a kernel
/// leaving.
///
/// It was 199, when `rms_rope` was vulkan's alone and contributed nothing.
/// Metal crossed it, so the pair contributes one comparison and this rose --
/// with `CENSUS` holding at 101, which is the shape a GAIN has and a drift
/// does not.
///
/// It rose again, to 201, WHEN WGPU CROSSED `rms_rope` as well -- *"wgpu fuses
/// the per-head qk norm into its rope"*. A third plane on a kernel already
/// crossed by two adds no name to `CENSUS` and one comparison here, which is
/// the same shape a second time, and the signatures agreed with no edit. The
/// entrypoint census states the other half of the same crossing, as
/// `("wgpu", "rms_rope_bfloat16")` in `EXCLUSIVE`.
const COMPARED: usize = 201;

/// A kernel a backend does not have AT ALL, and the sentence saying why.
///
/// Different from a retirement, which is a name a backend gave up when it
/// crossed: this is a name it never had. [`CENSUS`] counts the union of the
/// three, so a kernel written for two planes leaves the third one short by
/// exactly the names listed here.
///
/// It is a list rather than a subtraction so that the third plane's absence
/// stays a QUESTION. A count that quietly allowed any shortfall would let a
/// port lose a kernel and a port never write one look identical, which is
/// the whole reason this file exists.
const UNCROSSED: &[(&str, &str, &str)] = &[
    // THE FUSED NORM+ROPE, WHICH ONLY VULKAN HAS. `kernels-vulkan/src/norm.rs`
    // has declared `rms_rope` since before the `Env` -> `Const` migration; it
    // normalises and rotates in one dispatch, which is a fusion the other two
    // planes have never written a module for.
    //
    // It became VISIBLE with that migration and not because of it. The three
    // planes each spelled their routine slice `#[distributed_slice] static
    // ROUTINES`, and `linkme` keys a slice on the STATIC's identifier rather
    // than on the crate — so the three sections were one section, every
    // plane's `declared()` answered the same merged list, and no shortfall
    // could show. The slices carry per-plane names now.
    (
        "wgpu",
        "rms_rope",
        "the fused norm+rope is vulkan's alone; wgpu has no module for it",
    ),
    (
        "metal",
        "rms_rope",
        "the fused norm+rope is vulkan's alone; metal has no module for it",
    ),
];

/// One backend's table, under the name this file reports it by.
struct Table {
    what: &'static str,
    rows: &'static [KernelSig],
    /// The names this backend's table used to hold, and that no accessor of
    /// its own reports any more. Stated here because this file was the only
    /// reader `retired_rows` ever had, so deleting it moved the record
    /// rather than ending it.
    retired: &'static [&'static str],
}

fn tables() -> Vec<Table> {
    vec![
        Table {
            what: "metal",
            // METAL PUBLISHES NO TABLE AT ALL NOW. Its `KERNELS` was an empty
            // slice kept for this file to read; every family crossed, so an
            // absent table and an empty one say the same thing.
            rows: &[],
            // METAL NO LONGER PUBLISHES ITS RETIREMENTS. The hundred names it
            // kept are inside `kernels-metal`'s own `kernel_of`, which was the
            // only thing that read them, and its routines answer for all but
            // one of them. `silu_mul_strided` is the one: DARK on this plane,
            // with a shader the tree still stamps and no routine that fires
            // it, so nothing else carries the name.
            retired: &["silu_mul_strided"],
        },
        // NEITHER SIBLING PUBLISHES A TABLE OR A RETIREMENT LIST NOW.
        // Both crossed every family, so their `KERNELS` was an empty slice
        // and `retired_rows` a hundred names no code read but this file.
        // What each still has to account for is the kernel it retired and
        // no routine of its own answers for.
        Table {
            what: "vulkan",
            rows: &[],
            retired: &[],
        },
        Table {
            what: "wgpu",
            rows: &[],
            retired: &["rms_rope"],
        },
    ]
}

/// A statement the three tables are ALLOWED to disagree about, and why.
///
/// Every entry needs a sentence. A bare name would make this list the place
/// disagreements go to stop being questions, which is the failure mode of
/// every exceptions list — so the test requires the sentence to be non-empty
/// and the entry to still be disagreeing.
///
/// These are the seven `refactor-bigplan.md` §1.1 found. **Neither looks like
/// a backend difference**; both look like a transcription made once and not
/// made again, and Stage 0 is where they get settled with the code in front of
/// whoever settles them. Until then they are written down rather than
/// tolerated silently.
const DRIFTED: &[(&str, &str)] = &[
    // EMPTY, and the six that stood here left the way their own message
    // instructed. All of them said the same thing: the mask's pitch is a FIRE
    // fact, and metal read `Slot(Param, 3)` -- the text's scalar, a literal
    // zero -- while wgpu and vulkan read the fire.
    //
    // The fix was not to pick a winner. All three rows now `Ask` the fire the
    // same question and each driver answers its own: wgpu and vulkan return
    // the pitch of the rectangle they staged, and metal returns ZERO because
    // the mask it stages is one enable word per token and no mask beside it.
    // Metal's number did not change -- it arrived by the literal before and
    // by its binder now -- but the SENTENCE is true, and the day metal learns
    // to stage a user mask is one line in `driver-metal`'s `named()` rather
    // than six signature rows in `kernels-metal`.
    //
    // `route_gather` left earlier the same way. An empty table here is the
    // state this file is for, not an absence of vigilance: the gate above
    // fails the moment two planes disagree again.
];

/// Every column of a row that two tables could disagree about.
///
/// Was `launch`, the four `*_param` indices, `axes`, `whole`, `in_place`,
/// `depth_prefix_plan` and the whole operand list. `refactor-bigplan.md` §7
/// Stage 5 deleted the positional half of `KernelSig` -- `launch`, `file`,
/// `lacks`, `sink` and the four indices -- once no backend read them, so what
/// is left to compare is the half a ROUTINE also states.
///
/// `operands` stood here too, and its comment said it survived "only because
/// `driver-cuda::bind` still walks it for `Source::Aux`, and this compares it
/// while it does". That walk turned out to have ended already:
/// `kernels_cuda::sigs()` fills `operands` from a base whose value is `&[]`,
/// so the filter crossed two hundred and five rows and matched none. Both the
/// walk and the column are deleted, and the list below is what a row still
/// says that a routine also says.
fn kernel_facts(sig: &KernelSig) -> String {
    format!(
        "axes={:?} whole={} in_place={:?} depth_prefix_plan={}",
        sig.axes.iter().map(|a| a.points).collect::<Vec<_>>(),
        sig.whole,
        // ALIASING, OFF THE SOURCE COLUMN. `KernelSig` never had an
        // `in_place()` of its own -- `Declared` does -- and the pairs come off
        // the same `Source::Alias` the marks derive, so there is one statement
        // of them and this reads it.
        kernels::routine::aliased(sig.sources),
        sig.depth_prefix_plan,
    )
}

/// The three tables name the same kernels.
///
/// Not a coincidence and not an aspiration: `kernels-wgpu`'s own table test
/// says the row count *"is `kernels-metal`'s, and that is the point rather
/// than a coincidence: this backend's coverage is DEFINED as its sibling's"*.
/// Each of the three asserts its own count against 100 and none of them
/// compares the NAMES, so three tables could hold a hundred rows each and
/// disagree about which hundred.
#[test]
fn the_three_shader_backends_name_the_same_kernels() {
    let tables = tables();
    let sets: Vec<(&str, BTreeSet<&str>)> = tables
        .iter()
        .map(|t| (t.what, t.rows.iter().map(|r| r.name).collect()))
        .collect();

    // Ported rows LEAVE `KERNELS`, so a backend part-way through the crossing
    // is a subset rather than a mismatch. The union is what must still be the
    // hundred, and a name in one table and absent from another is only news
    // when the second table has not started crossing.
    //
    // The union is over rows, ROUTINES and RETIREMENTS, and it needed all
    // three in that order. Rows alone read 99 once all three backends had
    // retired `sample`: `argmax_logits` was in no table and the invariant had
    // not been broken, the question had. Adding routines fixed that, because
    // a kernel is a row until its family crosses and a routine afterwards.
    // Then metal crossed all ten families, and `silu_mul_strided` -- DARK,
    // with a row's name and no routine anywhere -- fell out of both planes at
    // once. `retired_rows` is each backend's own record of what it let go,
    // which is what keeps this a claim about the SHADER TREE rather than
    // about three lists that shrink.
    let crossed = [
        kernels_wgpu::declared(),
        kernels_vulkan::declared(),
        kernels_metal::declared(),
    ];
    let union: BTreeSet<&str> = sets
        .iter()
        .flat_map(|(_, s)| s.iter().copied())
        .chain(crossed.iter().flat_map(|d| d.iter().map(|x| x.name)))
        .chain(tables.iter().flat_map(|t| t.retired.iter().copied()))
        .collect();
    assert_eq!(
        union.len(),
        CENSUS,
        "the union of the three tables is {} kernels, not the {CENSUS} all \
         three declare",
        union.len()
    );

    for (what, set) in &sets {
        if set.len() == CENSUS {
            continue;
        }
        // A shrinking table is a port in progress; say so rather than fail,
        // and name what it has given up so the countdown is readable.
        let gone: Vec<&str> = union.difference(set).copied().collect();
        println!("{what} has crossed {} rows: {gone:?}", gone.len());
    }
}

/// Where the three tables disagree, the disagreement is written down.
///
/// This is `refactor-bigplan.md` §1.1 as a test. It compares only the columns
/// that are facts about the kernel — see the module docs for what is left out
/// and why — and it fails on a disagreement that is not in [`DRIFTED`], and on
/// a [`DRIFTED`] entry that has stopped disagreeing.
#[test]
fn the_three_tables_disagree_only_where_it_is_written_down() {
    let tables = tables();
    let mut by_name: BTreeMap<&str, Vec<(&str, String)>> = BTreeMap::new();
    for t in &tables {
        for row in t.rows {
            by_name
                .entry(row.name)
                .or_default()
                .push((t.what, kernel_facts(row)));
        }
    }

    let excused: BTreeMap<&str, &str> = DRIFTED.iter().copied().collect();
    assert_eq!(
        excused.len(),
        DRIFTED.len(),
        "a kernel is named twice in DRIFTED"
    );
    for (name, why) in DRIFTED {
        assert!(
            why.len() > 30,
            "`{name}` is excused without a sentence saying which backend is \
             different and why, which makes this list the place questions go \
             to stop being questions"
        );
    }

    let mut disagreeing = BTreeSet::new();
    let mut report = Vec::new();
    for (name, answers) in &by_name {
        // One backend holding the row is not a disagreement: the others may
        // have crossed to routines.
        if answers.len() < 2 {
            continue;
        }
        let first = &answers[0].1;
        if answers.iter().all(|(_, facts)| facts == first) {
            continue;
        }
        disagreeing.insert(*name);
        if !excused.contains_key(name) {
            report.push(format!(
                "`{name}`:\n{}",
                answers
                    .iter()
                    .map(|(what, facts)| format!("    {what:8} {facts}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }
    }

    assert!(
        report.is_empty(),
        "the three shader tables disagree about {} statement(s) that nothing \
         explains. Each is a fact about the KERNEL, so at most one of the \
         answers is right — settle it, or add it to DRIFTED with a sentence \
         saying which backend is different and why:\n\n{}",
        report.len(),
        report.join("\n\n")
    );

    // STALE means SETTLED, not UNOBSERVABLE.
    //
    // An entry stops being checkable the moment a backend retires the row it
    // was about, and Stage 4 retires all of them. Metal finished first, and
    // the six `sdpa_paged_*` entries — every one of which says *"metal is the
    // wrong one"* — immediately looked settled: the two tables still holding a
    // row are wgpu's and vulkan's, and those two always agreed. Deleting them
    // would have recorded progress that did not happen and erased a defect
    // that is still there.
    //
    // So an entry is stale only when EVERY table still states the name and
    // they agree. Once one has retired it, the drift is carried by
    // `the_two_settled_drifts_are_still_true_of_the_drivers_they_name`, which
    // reads the DRIVERS and does not care what any table says — and which
    // still reports `AttentionMaskStride` in zero places in `driver-metal`.
    let all = tables.len();
    let settled: Vec<&str> = excused
        .keys()
        .copied()
        .filter(|n| !disagreeing.contains(n) && by_name.get(n).is_some_and(|a| a.len() == all))
        .collect();
    assert!(
        settled.is_empty(),
        "{settled:?} no longer disagree, so their DRIFTED entries are stale. \
         Delete them — that edit is what records the progress."
    );
}

/// A kernel is a ROW or a ROUTINE, and during a family's port it is both.
///
/// `refactor-bigplan.md` §8 asks for the union of `KERNELS` and `ROUTINES` to
/// still be the hundred names per backend, and for the two to be **pairwise
/// disjoint**. Disjoint is the end state: §7 Stage 3 says a family's bodies,
/// its driver arms and its `kernel!` rows land in ONE commit, so both planes
/// exist only *within* a family and never across one.
///
/// This tree does not meet that today, and what is written down here is the
/// exact list rather than the claim -- or even a count of it.
///
/// A family's rows cannot come off when its bodies land, because deleting a
/// row means the driver can reach the routine instead, and NOTHING in this
/// workspace can yet. A row states which trace operand fills each shader
/// binding (`In(0), Weight(0), Out(0)`); a `Routine`'s signature states the
/// types and their [`Provenance`], which is `Trace` or `Env` and not WHICH
/// trace operand. That mapping lives only in the row. The arm that would
/// carry it is `refactor-bigplan.md` §7's "driver arms", and grepping the
/// workspace for a consumer of `Provenance` finds tests and nothing else --
/// including in `kernels-cuda-new`, the port every other backend is copying.
/// So Stage 3's one-commit rule is aspirational for all three of us until
/// somebody builds that seam, and until then every crossed family is stated
/// twice.
///
/// A bare `<= N` would not tell a family crossing from a family being
/// UN-crossed while another crosses, which is why this was a written-out list
/// of names for a long time. It is a per-backend equality now, for the reason
/// `WINDOW` gives: the names are derivable and the list charged every agent
/// for every other backend's work. Equality keeps the ratchet -- the number
/// may only fall, and the fall is the record -- and the failure message
/// prints the names, so a reader still gets what the list gave them.
#[test]
fn the_kernels_stated_twice_are_the_ones_written_down() {
    /// Names held by BOTH a table and a routine set, per backend.
    ///
    /// Sorted, because they are compared as sets and a set has one spelling.
    ///
    /// wgpu's six are its `layout` family, crossed first because the
    /// axis-suffix fork of `wgpu-refactor.md` §4 needed a family with two
    /// instantiation axes to settle at all. vulkan's are `sample`, the
    /// one-kernel family that exercises the whole surface, and then `mlp` and
    /// `layout` and `rope` and `ssm` and `norm` and `moe` and `attn` and
    /// `quant` and `ptir`, in `refactor-bigplan.md` §7's size order.
    ///
    /// **metal's is empty.** Not because it never crossed but because it has
    /// finished: a name is stated twice only while a family's rows and its
    /// routines both exist, and metal has no rows. That is what this list
    /// measures, and reaching zero is what it was counting toward.
    /// The most a backend may still state twice, per backend. It may only fall.
    ///
    /// This WAS the list itself — all 93 names, per backend, sorted. The list
    /// answered "which kernels live in two planes", which is what a reader of
    /// a failure needs, and the edit that deleted a family's names was the
    /// record of its retirement.
    ///
    /// It cost more than it paid. The set is DERIVABLE — rows ∩ routines —
    /// so writing it down asserts only that someone typed it, and the someone
    /// was whichever agent noticed: a sibling backend crossing a family left
    /// `origin/rewrite` failing here until an unrelated branch edited a list
    /// about a crate it does not touch. That happened four times in a day,
    /// twice within an hour.
    ///
    /// What the list actually protected is below and is not derivable: the
    /// union is still the hundred (a port that LOSES a name loses it
    /// silently), and the window only ever narrows. The ceiling is one number
    /// per backend and lowering it is still the record — but only the backend
    /// that crossed the family has to make that edit, which is the property
    /// the list did not have.
    const WINDOW: &[(&str, usize)] = &[("wgpu", 0), ("vulkan", 0), ("metal", 0)];

    // JOINING THIS GATE, for the next backend to port a family: expose
    //
    //     pub fn declared() -> Vec<kernels::routine::Declared>
    //
    // over your `ROUTINES` -- `Routine::declared()` is the per-row view -- and
    // add one line below. That is the whole of it, and from then on your
    // routines are compared against your own rows while both exist, and
    // against the other backends' once two of you have ported the same
    // family.
    //
    // `kernels-metal` has crossed ALL TEN. It started with `sample` and
    // `ptir` — its two one-kernel families, both dark, so neither crossing
    // could change what a model computes — and finished the remaining eight
    // once `driver-metal` stopped resolving symbols through the table at all.
    // Its `KERNELS` is an empty slice, which is what makes the union
    // assertion below load-bearing rather than arithmetic: it is the only
    // line left that proves the hundred are all still stated somewhere.
    let crossed: Vec<(&str, Vec<&str>, BTreeSet<&str>)> = vec![
        (
            "wgpu",
            kernels_wgpu::declared().iter().map(|d| d.name).collect(),
            BTreeSet::new(),
        ),
        (
            "vulkan",
            kernels_vulkan::declared().iter().map(|d| d.name).collect(),
            BTreeSet::new(),
        ),
        (
            "metal",
            kernels_metal::declared().iter().map(|d| d.name).collect(),
            BTreeSet::new(),
        ),
    ];

    for (what, routines, rows) in &crossed {
        let names: BTreeSet<&str> = routines.iter().copied().collect();
        assert_eq!(
            names.len(),
            routines.len(),
            "{what} declares a routine name twice"
        );

        // Rows, routines AND the names the rows gave up. A kernel a backend
        // has fully crossed is in neither plane -- metal's `silu_mul_strided`
        // is DARK, so it has a row's name and no routine at all, and once its
        // row retired nothing else would have carried the name. `retired_rows`
        // is each backend's own statement of what it let go, so this stays a
        // claim about the hundred rather than about two lists that shrink.
        let retired: BTreeSet<&str> = tables()
            .iter()
            .find(|t| t.what == *what)
            .map_or(BTreeSet::new(), |t| t.retired.iter().copied().collect());
        let uncrossed: BTreeSet<&str> = UNCROSSED
            .iter()
            .filter(|(b, ..)| b == what)
            .map(|(_, name, _)| *name)
            .collect();
        let union: BTreeSet<&str> = names
            .union(rows)
            .copied()
            .chain(retired.iter().copied())
            .chain(uncrossed.iter().copied())
            .collect();
        assert_eq!(
            union.len(),
            CENSUS,
            "{what}'s rows, routines and retirements together are {} kernels, \
             not the {CENSUS}. A port that loses a name loses it silently: \
             the row is gone, no routine answers for it, and nothing recorded \
             it.",
            union.len()
        );

        let both: Vec<&str> = names.intersection(rows).copied().collect();
        let ceiling = WINDOW
            .iter()
            .find(|(b, _)| b == what)
            .unwrap_or_else(|| panic!("`{what}` has no entry in WINDOW"))
            .1;
        assert!(
            both.len() <= ceiling,
            "{what} states {} kernels in both a table and a routine set and \
             WINDOW allows {ceiling}. The window only NARROWS: a family that \
             gained a routine without losing its rows is a real event, and it \
             is one this test refuses rather than records. {both:?}",
            both.len(),
        );
        assert_eq!(
            both.len(),
            ceiling,
            "{what} states {} kernels in both planes, down from {ceiling}. \
             Lower the entry -- that edit is the record of the retirement, \
             and it is the crossing backend's own edit to make.",
            both.len(),
        );
    }

    let crossed_backends: BTreeSet<&str> = crossed.iter().map(|(w, _, _)| *w).collect();
    for (what, _) in WINDOW {
        assert!(
            crossed_backends.contains(what),
            "WINDOW writes down `{what}`, which is not in the list of backends \
             this gate reads. A backend cannot be checked by being mentioned."
        );
    }
}

// THREE TESTS STOOD HERE, and each of them said in its own failure message
// exactly when to delete it. All three now say it.
//
// * `a_ported_routine_takes_exactly_the_operands_its_row_states` compared a
//   crossed routine's signature against the ROW of the same name on the same
//   backend -- the strongest check in the fleet while a family was mid-
//   crossing, because it held the new plane to the old one, on the same
//   kernel, in the same crate. Its message: *"no kernel is stated twice, so
//   this compared nothing."*
// * `a_kernel_crossed_on_one_backend_is_compared_against_the_row_on_another`
//   did the same across backends, so a family that crossed on vulkan could
//   still be checked against metal's row. Its message: *"nothing was
//   compared."*
// * `the_crossed_routines_with_no_row_to_check_are_counted` existed to notice
//   the other two going quiet, and it is the one that fired first: *"0 of 99
//   crossed have a row to check"*, on every backend.
//
// ALL THREE now hold an empty `KERNELS`: `kernels-wgpu` was the last, and its
// final row was `silu_mul_strided` -- a kernel every backend had recorded as
// unable to take a positional argument list, which was true of MSL's flat
// argument table and false of `gated.wgsl`. There is no row left to hold a
// routine to, anywhere.
//
// What carries the signatures from here is
// `two_backends_that_crossed_the_same_kernel_agree_on_its_signature`, which is
// the same claim between two ROUTINE planes with the tables taken out of the
// middle -- and which grew as these shrank, which is the trade §3 was for.
// `the_two_settled_drifts_are_still_true_of_the_drivers_they_name` STOOD HERE
// and is gone because both drifts it watched are settled.
//
// It asked a question about two OTHER crates by reading their sources: does
// `driver-vulkan` resolve the padded extent, does `driver-metal` resolve
// `AttentionMaskStride`. Both now do, and the second one is why: it failed
// with "check whether its six rows now read `AttentionMaskStride`, and if
// they do, delete the six entries." They did, so they were.
//
// A gate that reads another crate's TEXT is a second spelling of what that
// crate does, and the shared binder made the first spelling askable: three
// planes now state the same source and each driver answers it. There is
// nothing left here to read from the outside.

/// A retired row's ENTRYPOINTS are still in the backend's census.
///
/// The one thing `RETIRED` exists for, asked directly. A row's `axes`
/// GENERATED its entrypoints, so `entrypoints()` used to mean both *"what the
/// table says"* and *"what this backend can do"*; deleting a row separates
/// them, and every sweep keyed on `entrypoints()` follows the table. On wgpu
/// that silently stopped compiling `argmax_logits_bfloat16` on a real adapter
/// while passing, and the loss compounds to a sweep that builds nothing.
///
/// Each crate answered it by folding a `RETIRED` list back in. None does
/// now: `930fee2cb` deleted `retired_rows` on all three planes along with
/// the empty `KERNELS` slices, and each backend's `entrypoints()` walks its
/// own routine registry instead. That is a better answer -- the census is
/// derived from what the crate can actually fire rather than from a ledger
/// somebody has to remember to write -- and it has one new way to be short,
/// which this test found within hours of the change. `930fee2cb` also put
/// the quantised forms on STAMPS, and a stamped point is declared by no
/// file, so a registry walk that reads only the file's declarations misses
/// 216 points that a fire can still spell. Metal read 265 against wgpu's
/// 481 until `75e4d9699` taught it to count the host's points too.
///
/// This does not care how any of them answers. It asks whether the three
/// censuses are still the SAME 481, which is derivable from the crates as
/// they are and needs no accessor any of them might not have. A backend
/// that stops naming what it can fire fails here, whichever goes first.
#[test]
fn retiring_a_row_does_not_shrink_a_backends_census() {
    // Entrypoints ONE backend has and the others do not, named individually.
    //
    // The assertion below is an equality between three censuses, and its
    // message is written for the case that made it: a backend that deleted
    // rows without stating what they named. That is a SHRINK. A backend that
    // gains a family the others have not got is the opposite, and reading it
    // as the same failure would leave only two ways out -- teach the other
    // backends a kernel they do not implement, or stop comparing -- and the
    // second is how a census quietly stops covering anything.
    //
    // So an addition is allowed, and the price of allowing it is naming every
    // entrypoint of it here. The list is subtracted from each census before
    // the comparison, so the SHARED census is still held to being identical
    // and to its exact size, and a deletion anywhere in it still fails.
    //
    // These nine are the flash decode: `decode_split` computes an
    // unnormalised partial over a slice of the key range and `decode_combine`
    // folds the slices, which is how a decode gets more than
    // `q_heads * rows` workgroups onto a card with 128 SMs. Vulkan-only
    // because it is a two-pass decomposition of a kernel the other backends
    // still fire in one pass, not a shader any of them is short of. Only
    // `d_64` has a sink form, because the sink texts are the ones with that
    // head width.
    const EXCLUSIVE: &[(&str, &str)] = &[
        ("vulkan", "sdpa_paged_decode_combine_bfloat16_d_64"),
        ("vulkan", "sdpa_paged_decode_combine_bfloat16_d_128"),
        ("vulkan", "sdpa_paged_decode_combine_bfloat16_d_256"),
        ("vulkan", "sdpa_paged_decode_combine_bfloat16_d_512"),
        ("vulkan", "sdpa_paged_decode_combine_sink_bfloat16_d_64"),
        ("vulkan", "sdpa_paged_decode_split_bfloat16_d_64"),
        ("vulkan", "sdpa_paged_decode_split_bfloat16_d_128"),
        ("vulkan", "sdpa_paged_decode_split_bfloat16_d_256"),
        ("vulkan", "sdpa_paged_decode_split_bfloat16_d_512"),
        // AND WGPU HAS CROSSED THE SAME DECOMPOSITION, which is why the four
        // `_split_` names appear twice: this list is keyed on (backend,
        // name), and a name two backends have needs an entry for each or the
        // one that is missing carries it into the shared census alone.
        //
        // `kernels-wgpu/src/attn.rs` fires them for real -- `PIE_SPLIT_BELOW`
        // decides, at 128 workgroups, when one workgroup per (row, query
        // head) is too few to fill a card, and `PIE_SPLITS` cuts the key
        // range eight ways. That is the same argument vulkan's note makes
        // above, arrived at independently on a backend whose smallest target
        // is a twenty-core GPU.
        //
        // THE SECOND PASS HAS TWO NAMES. Vulkan calls it `_combine_` and wgpu
        // calls it `_merge_`, and they are the same kernel: fold the
        // unnormalised partial softmax states one slice each wrote. Nothing
        // here can make them agree -- a rename crosses two shader trees and
        // every table that spells them -- so what this file can do is write
        // the divergence down where the next reader of either list will find
        // it, rather than leave two spellings looking like two families.
        ("wgpu", "sdpa_paged_decode_split_bfloat16_d_64"),
        ("wgpu", "sdpa_paged_decode_split_bfloat16_d_128"),
        ("wgpu", "sdpa_paged_decode_split_bfloat16_d_256"),
        ("wgpu", "sdpa_paged_decode_split_bfloat16_d_512"),
        ("wgpu", "sdpa_paged_decode_merge_bfloat16_d_64"),
        ("wgpu", "sdpa_paged_decode_merge_bfloat16_d_128"),
        ("wgpu", "sdpa_paged_decode_merge_bfloat16_d_256"),
        ("wgpu", "sdpa_paged_decode_merge_bfloat16_d_512"),
        // THE FUSED NORM+MATVEC, which metal and vulkan both have and
        // wgpu has not crossed. Two backends gaining a family reads
        // oddly in a list called EXCLUSIVE, but the question the list
        // answers is the right one either way: what to leave out of a
        // comparison of what the backends SHARE.
        //
        // THE FUSED NORM+ROPE, WHICH IS VULKAN'S ALONE. Six entrypoints of
        // one family `kernels-vulkan/src/norm.rs` has declared since before
        // the `Env` -> `Const` migration and neither other plane has a
        // module for -- `UNCROSSED` records the kernel, and these are the
        // entrypoints that kernel spells. It became visible with the
        // migration and not because of it: the three planes shared one
        // `linkme` section until their slices were given per-plane names,
        // so no shortfall between them could show.
        //
        // **Five of the six, since `rms_rope_bfloat16` crossed.** `wgpu fuses
        // the per-head qk norm into its rope` gave `kernels-wgpu` the base
        // form of this family -- one entrypoint of the six, not the decode,
        // freqs or prop variants -- so the name needs an entry per backend,
        // for the reason the four `_split_` names above have two: this list
        // is keyed on (backend, name), and a name two backends have that is
        // subtracted from only one of them walks into the shared census
        // alone.
        //
        // METAL declares `norm::rms_rope` too, in `ELSEWHERE`, and still does
        // not appear here: its entrypoint is in `DECLARED_ELSEWHERE`, so the
        // shader is not in metal's stamped table and `entrypoints()` does not
        // enumerate it. Two of the three planes carry the name in a census
        // and the third carries the kernel without one, which is exactly the
        // case this list is a subtraction rather than an exemption for.
        ("vulkan", "rms_rope_bfloat16"),
        ("wgpu", "rms_rope_bfloat16"),
        ("vulkan", "rms_rope_decode_bfloat16"),
        ("vulkan", "rms_rope_freqs_bfloat16"),
        ("vulkan", "rms_rope_freqs_decode_bfloat16"),
        ("vulkan", "rms_rope_prop_bfloat16"),
        ("vulkan", "rms_rope_prop_decode_bfloat16"),
    ];

    let shared = |what: &str, census: &[String]| -> Vec<String> {
        census
            .iter()
            .filter(|e| {
                !EXCLUSIVE
                    .iter()
                    .any(|(b, n)| b == &what && n == &e.as_str())
            })
            .cloned()
            .collect()
    };

    let censuses = [
        ("wgpu", kernels_wgpu::entrypoints()),
        ("vulkan", kernels_vulkan::entrypoints()),
        ("metal", kernels_metal::entrypoints()),
    ];
    // Every name in the list must actually be there. Otherwise an allowance
    // outlives the thing it allowed and starts hiding a deletion, which is
    // the failure this test exists for.
    for (backend, name) in EXCLUSIVE {
        let census = &censuses
            .iter()
            .find(|(w, _)| w == backend)
            .expect("EXCLUSIVE names a backend this test does not compare")
            .1;
        assert!(
            census.iter().any(|e| e == name),
            "`kernels-{backend}` does not have `{name}`, which is listed as \
             exclusive to it. An allowance that names nothing hides a deletion."
        );
    }
    let (first, reference) = (censuses[0].0, shared(censuses[0].0, &censuses[0].1));
    for (what, census) in &censuses[1..] {
        assert_eq!(
            shared(what, census),
            reference,
            "`kernels-{what}`'s census is {} entrypoints and `kernels-{first}`'s \
             is {}, counting only what they share. The crossing moves who \
             NAMES an entrypoint, never whether it exists, so a backend \
             part-way through Stage 3 must still name every one of them. \
             Rows and `RETIRED` lists were how; neither exists on any plane \
             now, and `entrypoints()` walks the routine registry instead -- \
             so a short side is a registry that has stopped enumerating \
             something it can still fire. The way that has actually happened \
             is a point no file DECLARES: a stamped form exists only when a \
             fire asks for it, and a census that reads the shader text alone \
             will not see it. Every sweep keyed on `entrypoints()` there has \
             stopped covering the difference in silence. A backend that \
             has GAINED a family the others lack states it in `EXCLUSIVE` \
             above.",
            shared(what, census).len(),
            reference.len(),
        );
    }
    assert_eq!(reference.len(), 481, "the shared shader census");
}

/// The countdown: `REMAINING` rows to 0.
///
/// `refactor-bigplan.md` §8. `KERNELS.len()` summed across the three, against
/// a constant that only ever goes DOWN. It makes the dual-maintenance window
/// visible and makes *"we will finish this later"* a number rather than a
/// sentence — which is what stops Stage 5, the only stage that pays for the
/// other four, from being permanently one backend away.
#[test]
fn the_three_tables_only_ever_lose_rows() {
    /// Lower this when a family crosses. It may not be raised.
    ///
    /// 300 → 192. vulkan retired `sample`, `ptir` and four more families;
    /// wgpu retired `sample` and `ptir`, one arm at a time; **metal retired
    /// ALL TEN**, which is `refactor-bigplan.md` §7 Stage 4 finished on one
    /// backend: `kernels_metal::KERNELS` is an empty slice, and the driver
    /// that used to read it resolves every symbol through the stem its
    /// routine registry states.
    ///
    /// A hundred rows is what the stage costs to reach, and what it buys is
    /// that no reader of an `operands`, `launch` or `*_param` column is left
    /// on this backend. Stage 5 deletes those columns, and it can only run
    /// once the LAST backend is here.
    ///
    /// 19 → 0. THE COUNTDOWN IS OVER. wgpu retired its remaining rows — it was
    /// the last backend holding any — and vulkan's last row went in the same
    /// window, so all three tables are empty slices and no reader of an
    /// `operands`, `launch` or `*_param` column is left anywhere in the tree.
    ///
    /// Stage 5, the only stage that pays for the other four, is unblocked:
    /// those columns and then the `kernel!` macro itself can go. This test
    /// asserts EQUALITY at zero rather than being deleted, so that a row
    /// re-appearing in any of the three is a failure and not a silence.
    ///
    /// The countdown's floor assertion went with it. `total <= REMAINING` was
    /// the progress bar's own guard while `REMAINING` was falling; at zero it
    /// compares a `usize` against the minimum value of its type, so it is
    /// true whatever the tables hold and checks nothing. Clippy denies it
    /// (`absurd_extreme_comparisons`), and it was the reason this crate --
    /// which IS in the workspace clippy gate -- failed that step. The
    /// equality below is what was doing the work, and it now says what a
    /// non-zero total means at the end of a countdown, which is not "lower
    /// the constant".
    const REMAINING: usize = 0;

    let total: usize = tables().iter().map(|t| t.rows.len()).sum();
    assert_eq!(
        total, REMAINING,
        "the three tables hold {total} rows and the countdown ended at \
         {REMAINING}. A row came BACK to a table that was emptied, and the \
         constant may not be raised to accommodate it — Stage 5 deletes the \
         `operands`, `launch` and `*_param` columns, so a row that states \
         them has nowhere to be read."
    );
}

/// Two backends that crossed the same kernel state the same signature.
///
/// This is `refactor-bigplan.md` §3's gate proper, and until now it had
/// nothing to run on: it needs one kernel ported by two backends, and `layout`
/// is the first -- `kernels-wgpu` crossed it to settle the axis-suffix
/// question and `kernels-vulkan` crossed it as §7's second real family.
///
/// What it compares is everything a `Declared` carries, which is everything
/// that is NOT device-shaped: the argument types and their provenance, whether
/// the statement consumes its whole operand, whether it joins the depth-prefix
/// plan, and which of its operands must be given the same address. Grids,
/// tiers, workgroup sizes and entrypoint spellings are properly per-backend --
/// §2 is the argument for why the bodies are not shared at all -- and none of
/// them are here.
///
/// The reason this is worth a test rather than a convention: §1 measured the
/// three `kernel!` tables and found them to be ONE table written three times.
/// A hundred identical names, `launch` identical 100/100, `axes` identical
/// 100/100, wgpu and vulkan's `operands` identical 100/100, and seven
/// statements differing workspace-wide -- every one of which is written down
/// in `DRIFTED` above. The refactor moves that table into three crates' worth
/// of separate `fn` signatures, which is exactly the move that lets them drift
/// silently for the first time. So the agreement stops being a measurement
/// somebody once took and becomes a thing that fails.
///
/// A real divergence is not impossible, and it is not this test's business to
/// forbid one. It is this test's business to make it an EDIT: a backend that
/// genuinely needs a different operand list adds itself here the way `DRIFTED`
/// records the seven, with the reason written next to it.
/// A crossed kernel two backends declare differently, and why.
///
/// The routine-level twin of [`DRIFTED`], and it works the same way: an entry
/// needs a sentence, and it may only be deleted — when the two agree, the
/// entry is stale and this test says so.
const DIVERGED: &[(&str, &str)] = &[
    // The flash decode's two extra operands, which vulkan's decode signature
    // carries and no other backend's does.
    //
    // `sdpa_paged_decode` on vulkan is two entrypoints behind one routine: a
    // SPLIT that walks a slice of the key range into an unnormalised partial,
    // and a COMBINE that folds the slices. That is what gets a decode more
    // than `q_heads * rows` workgroups onto a 128-SM card -- at a 384-key
    // history it is 67.8 us against 8.2. The two operands are the partials
    // buffer the split writes and the fold reads (`F32sMut`, and `f32` rather
    // than the activation type because a partial is an unnormalised weighted
    // sum whose scale is `exp(score - split_max)`), and the split count the
    // fold has to be told, since its grid is `(head, row)` with no third axis
    // to read it off.
    //
    // A real divergence and not a mistyped port: the other backends fire this
    // kernel in ONE pass, and a one-pass decode has no partial to store and
    // no splits to count. The routine takes the same rectangle and answers
    // the same thing; only vulkan's needs somewhere to put the middle of it.
    // The day another backend splits its decode, its signature grows these
    // two and this entry is deleted -- which the test enforces, since an
    // entry whose backends have stopped disagreeing fails as stale.
    // THE THREE TRANSCODE KERNELS WERE HERE, AND THEY ARE SETTLED.
    //
    // `encode_u4_bf16`, `encode_u4_f32` and `mxfp4_dequant_bf16` diverged on
    // where one pair of numbers travelled: `{groups, group_size}`
    // (`{blocks, block_size}` for mxfp4) was a params BUFFER on metal and
    // vulkan, minted with `o.params_block()` and passed positionally, while
    // `transcode.wgsl` had nowhere to put one -- its `@group(0)` bindings are
    // dense data buffers and the pair is two `u32` fields of a `@group(1)`
    // uniform filled from the scalars a body forwards. Same two numbers, two
    // carriers.
    //
    // All three backends now declare `[.., I32]` and no params buffer, so the
    // entries went STALE and this gate said so rather than letting them sit.
    // That is the whole design of this list: an excuse is deleted by the edit
    // that settles it, and the deletion is the record.
    //
    // Worth keeping the history because of what the divergence was hiding.
    // These bodies once took a `_params: Buf` they could not use and forwarded
    // NO scalars at all, so the uniform arrived empty and the shader read zero
    // groups -- a loop that runs no iterations and reports success. Nothing
    // caught it because all three rows were UNSTATED and no model fires them:
    // `encode_u4` appears in no lowering in this tree.
    // SEVENTEEN quant kernels, all the same shape, all metal's `pad: Buf`.
    //
    // `0fc54bedb` ("Seventeen quant kernels were reading whatever the last
    // step left") added a padding buffer to metal's signatures. It is not a
    // fix wgpu needs and copying it would BREAK this backend.
    //
    // Metal has one flat, POSITIONAL `[[buffer(n)]]` table shared by every
    // variant of a shader, so a variant that does not read a slot still has to
    // reserve it or the numbering shifts under the ones that do — `pad` is
    // that reservation. WGSL is preprocessed per variant here and numbered
    // densely from the row, and
    // `every_routine_binds_a_buffer_for_every_binding_its_module_declares`
    // machine-checks the density against the parsed module. Adding a pad would
    // make every one of these seventeen bind one buffer more than its shader
    // declares.
    //
    // Written down as seventeen entries rather than one, because a rule with
    // an exception list is a rule and a waiver is not: if one of these ever
    // stops diverging, that entry goes stale and this gate says so.
    // Vulkan's descriptor set holds buffers and its push block holds scalars,
    // and they are two namespaces: `sdpa_sliding.slang` gives `sinks` binding
    // 4, right behind `out_`, because a buffer cannot sit past a push
    // constant. Metal has one flat argument table, and the same shader there
    // declares `sinks [[buffer(14)]]` -- after `window` and both row pitches,
    // so the sinked and unsinked forms share a prefix and the sink is
    // appended. Both are the right order for their own ABI and neither can
    // take the other's without renumbering a shader that is already correct.
    (
        "sdpa_vector_decode_sink",
        "vulkan must put every buffer ahead of its push constants; Metal's \
         flat argument table appends the sink plane past the scalars instead",
    ),
    // `ssm/gdn_prep.slang` declares `struct Push { int row_pitch; int n_scan; }`
    // under `PIE_SCAN` and binds it as a push constant; Metal's counterpart
    // reads the same two numbers out of the params block it already binds, so
    // there they are grid-only. Metal's scan also binds a leading `pad`
    // buffer the Slang module has no binding for at all. Both are in the
    // shaders, and neither is expressible as one signature.
    // `gdn_prep_prefill` STOOD HERE AND IS SETTLED. Its divergence was one
    // parameter: vulkan's signature carried a bare `Env<i32>` for `rows` --
    // a wrapper claiming NO source at all, so the routine's column resolved
    // nothing and the row could never be bound -- where metal asked for the
    // fire's token count. The `Env` -> `ask` move left the wrapper with
    // nothing to be, and the parameter turned out to be `keys::Rows`, which
    // is what its own sibling `gdn_prep_slotted` had been asking for beside
    // it all along. Two backends, one signature.
    // The norm family's four strided/gated forms. Metal sizes the
    // THREADGROUP on the axis -- `grid::rms` has always been `axis / 4`
    // threads, capped at 1024 -- while every Slang module here is compiled at
    // a fixed 256 and walks the axis in a loop. So metal needs the axis (or
    // the head width) as a grid fact and vulkan does not, in exactly the four
    // places where the axis is not ALSO needed to count the norms in a row.
    //
    // The four PACKED forms are not here, and that is the useful half of this
    // finding: there the axis divides the row into `width / axis` reductions,
    // so both backends need both numbers, vulkan was launching one workgroup
    // per row, and it took metal's signature rather than an excuse.
    // ── FIVE ENTRIES THE `Env` -> `Const`/`ask` MIGRATION SETTLED ────────────
    //
    // `gdn_prep_prefill`, `rms_strided_row`, `rms_strided_head_row`,
    // `sdpa_paged_decode` and `sdpa_paged_decode_sink` stood in this list and no
    // longer do. Not one of them was a device difference; every one was a
    // PARAMETER LIST difference, and each is a finding `.wiki/migration.md`
    // predicted by name:
    //
    //   * the two `rms_strided_*` forms differed over `axis` -- §9.1's *"one key
    //     under two parameter names"*, `keys::Width` twice, at fifteen sites. With
    //     the rectangle off the mark it is `x.width` on both planes, which is a
    //     grid fact and not a parameter, so the signatures are one signature.
    //   * `gdn_prep_prefill` differed over a bare `Env<i32>` that claimed no
    //     source at all, so vulkan's row could never be bound. It is the fire's
    //     token count, which its own sibling was already asking for beside it.
    //   * the two `sdpa_paged_decode` forms differed over which of twenty facts
    //     each plane spelled as a parameter -- §5.6's case exactly. Twelve of them
    //     are the page tables, the KV pool, the staged mask and the partials
    //     buffer, and they leave the signature entirely; six are the checkpoint's
    //     and stay as `Const`. What is left agrees.
    //
    // The list is shorter because the signatures are shorter, and that is the
    // point of the change rather than a side effect of it.

    // `rms_strided_head_row` STOOD HERE AND IS SETTLED. Its divergence was
    // `axis`: metal declared it and vulkan did not, because metal sizes the
    // threadgroup on it and vulkan compiles a fixed 256-wide workgroup that
    // walks it. Both spelled the number `keys::Width` -- §9.1's finding, one
    // key under two parameter names at fifteen sites -- and with the rectangle
    // off the mark it is `x.width` on both planes, which is a grid fact and
    // not a parameter. The two signatures were the same signature all along;
    // only the parameter list said otherwise.
    // The moe family's six, in two groups of three, and both groups are a
    // shape this list already carries.
    //
    // The three routed GEMMs take a `pad`. Their entrypoints declare buffers
    // 0..=6 and then `tile_expert` at TWELVE, so one argument-table ordinal
    // can serve both the routed GEMM and the routed matvec -- and a Metal
    // argument table is a contiguous run, so the holes must still hold an
    // address. `gdn_core_recurrent_prefill` above is excused for the same
    // reason and says so at more length. The MXFP4 one pads six slots rather
    // than five: it declares nothing at 2 either, because the codec has no
    // zero point to bind where affine puts `biases`.
    (
        "mxfp4_qmm_t_routed_bias",
        "metal binds a pad at slot 2 and at 8..=11: tile_expert is at buffer \
         12 and MXFP4 has no zero point to bind where the affine codec puts \
         its biases, so six slots are holes rather than five",
    ),
    // The three routing kernels size their threadgroup on the EXPERT COUNT --
    // `route.metal`'s top-k reduces across simdgroups, so the group is the
    // expert count rounded to a whole simdgroup and clamped at 1024. Every
    // Slang module here is compiled at a flat 1024 and strides. So the expert
    // count is a grid fact on one backend and a shader constant on the other,
    // which is the `norm` family's divergence in a second family.
    // =================================================================
    // THE PARAMS BLOCK, SETTLED. Twenty-five entries stood here and are
    // gone. The way they went is worth the paragraphs they cost.
    //
    // Each was ONE operand: metal declared `params: Env<Buf>` where wgpu
    // and vulkan declared `params: Buf`. Same position, same type, every
    // other argument agreeing -- the two backends disagreed about
    // nothing except WHO SUPPLIES IT. `Provenance`'s own definition
    // settled it, and it is a definition and not a convention: *"The
    // distinction is not 'who owns the memory' but 'who can be asked for
    // it at trace time'"*. A params block is the STATEMENT'S OWN SCALAR
    // RUN, packed; `driver-metal`'s `Handles::params_block`
    // (`lowering/arm.rs:356`) opens by saying so and names the column it
    // comes from, `Source::Slot(Kind::Param, 0)`, *"meaning the whole
    // run"*. What metal had wrapped was the STAGING and not the value --
    // the encoder mints one region per fire -- and that sentence refuses
    // exactly that reading in advance: not who owns the memory.
    //
    // `argmax_logits` was the one that could not be argued at all: its
    // arm binds `params` from `o.input(1)?`
    // (`driver-metal/src/lowering/arm.rs:487`), which is not the staged
    // block but the STATEMENT'S SECOND INPUT OPERAND, and
    // `kernels-metal/src/sample.rs:47` called it `Env<Buf>`.
    //
    // And `kernels-metal` had been saying so itself the whole time,
    // which is the part to remember. Its own `sample.rs` asserts
    // `(Ty::Buf, Supplier::Trace)` in that slot under the heading *"The
    // derived row says what the signature says"* and reads *"Four
    // buffers the trace supplies and one extent the environment does"*;
    // `ptir.rs` asserts the same for `copy_logits_bf16`; and every
    // in-crate call site passed a bare `Buf` for the parameter. The
    // `Env<>` was at the declaration line and NOWHERE ELSE -- not in a
    // call, not in a body, not in a driver that branched on it. A gate
    // between crates found what the crate's own tests could not, because
    // the tests were right and the signature they derive from was not.
    //
    // So the thirty-four wrappers were dropped and the twenty-five
    // entries were deleted by hand, along with the seventeen in
    // `UNRECONCILED` that were the same finding seen from the row plane.
    // THE NINE MASKED SITES REMAIN EXCUSED ABOVE, for their own and
    // unrelated reasons: `gated_rms`, `gated_rms_strided`,
    // `rms_strided_row` and `rms_strided_head_row` for threadgroup
    // width, `router_topk`, `router_topk_scaled` and `route_sort` for
    // the same, `gdn_prep_prefill` and `gdn_core_recurrent_prefill` for
    // push constants. None of those entries ever named the params block
    // and none of them changed.
    // -----------------------------------------------------------------
    // THE FOUR EMBEDDING GATHERS, and here metal is the RIGHT one.
    //
    // The operand is `id`, the token-id vector, and it is the opposite
    // shape from the params block settled above: metal declares
    // `Env<I32s>` (`kernels-metal/src/layout.rs:245` and its three
    // siblings) and wgpu and vulkan declare a bare `I32s`. Nothing else
    // differs.
    //
    // Both crates already AGREE about the fact and disagree only about
    // its spelling. metal's parameter carries a shouted comment -- *"THE
    // TOKEN IDS ARE THE FIRE'S. An embedding gather at the top of a
    // graph has no traced operand -- the ids come off the fire's
    // frame"* -- and its arm binds `o.table(FireTable::TokenIds)`. The
    // wgpu ROW for the same kernel says it in nearly the same words:
    // *"The token IDS: the FIRE's, not the statement's. A text cannot
    // state them"*, and sources the operand
    // `Source::Named(<keys::TokenIds as keys::Fact>::KEY)`.
    //
    // A fact reached by NAME off the fire is the textbook `Env`. The
    // `Provenance` doc's own examples are *"a position vector, a plan, a
    // workspace"*; `KernelSig::args` (`kernels/src/lib.rs`) names
    // the failure mode exactly, complaining that with no `Env` wrappers
    // every position derived as `Trace` *"including the head widths,
    // position vectors and cache descriptors that driver-cuda's arms
    // read off the fire because no statement carries them"*; and
    // `kernels-cuda`, the port the others are copying, writes
    // `token_ids: Env<keys::TokenIds>` (`layout.rs:616`) and
    // `positions: Env<keys::Positions>` (`rope.rs:1478`). Metal is
    // right.
    //
    // SETTLED, AND THE FOUR ENTRIES ARE GONE. It was excused because wgpu
    // could not take metal's spelling without failing
    // `a_ported_routine_takes_exactly_the_operands_its_row_states`, which
    // compared a routine's TRACE arguments against the ROW's whole operand
    // list -- `Named`-sourced operands included -- and had no way to express
    // a row operand that is an `Env` fact. That test is deleted; the note
    // above this section records why, and with no rows left there is nothing
    // for an `Env` argument to fail against. The second half of the
    // settlement, `Env` everywhere, then landed: wgpu and vulkan now spell
    // `id` the way metal and cuda always did.
    //
    // Rope's `position` went the same way and for the same reason, though
    // nothing here would have caught it -- all three planes agreed on the
    // wrong spelling, which is what made it invisible. §6.2's arity rule on
    // the metal plane is what found it: a statement places what a routine
    // READS, and a rotation reading a fire table it never named made the
    // count come out one short.
    // -----------------------------------------------------------------
    // The twenty-five params-block entries stood here, below the four
    // above and grouped by family. They are gone; the note at the head
    // of this section says why, and that is now the whole of the record.
];

// The other six `gdn_*` kernels were parked here as UNRESOLVED and are not
// parked here any more, because both halves of that finding were settled
// rather than excused and both crates moved.
//
// (a) ELEMENT TYPES: metal now says `F32s`/`F32sMut` in the four positions
// where `gdn_core.metal` and `gdn_prep.metal` declare `device float*`. The
// typed spelling was not vulkan's preference, it was what all three trees
// declare, so metal took it.
//
// (b) ARITY: metal's extra trailing `Env<i32>` was the better statement and
// not an extra. Its grid takes `rows` and `v_heads` where vulkan took their
// PRODUCT, and the shader takes the product apart again -- `hv = z % Hv`,
// `row = z / Hv` -- so a body handed the product cannot tell one
// factorisation from another and two different routings with the same
// product address different state. Vulkan took metal's.
//
// What is left above is the part that really is two different shaders.

#[test]
fn two_backends_that_crossed_the_same_kernel_agree_on_its_signature() {
    let backends: Vec<(&str, Vec<kernels::routine::Declared>)> = vec![
        ("wgpu", kernels_wgpu::declared()),
        ("vulkan", kernels_vulkan::declared()),
        ("metal", kernels_metal::declared()),
    ];

    // name -> the backends that have crossed it, with what they declared.
    let mut by_name: BTreeMap<&str, Vec<(&str, kernels::routine::Declared)>> = BTreeMap::new();
    for (what, declared) in &backends {
        for d in declared {
            by_name.entry(d.name).or_default().push((what, *d));
        }
    }

    let excused: BTreeMap<&str, &str> = DIVERGED.iter().copied().collect();
    assert_eq!(
        excused.len(),
        DIVERGED.len(),
        "a kernel is named twice in DIVERGED"
    );
    for (name, why) in DIVERGED {
        assert!(
            why.len() > 40,
            "`{name}` is excused without a sentence saying what differs and \
             why, which makes this list the place questions go to stop being \
             questions"
        );
    }

    let mut compared = 0usize;
    let mut report: Vec<String> = Vec::new();
    for (name, ports) in &by_name {
        let Some(((first_what, first), rest)) = ports.split_first() else {
            continue;
        };
        if excused.contains_key(name) {
            // Still compared, so a DIVERGED entry that has stopped diverging
            // fails below rather than sitting here forever.
            if rest.iter().all(|(_, other)| other.args == first.args) {
                panic!(
                    "`{name}` is in DIVERGED and the backends now agree. \
                     Delete the entry -- that edit is what records it being \
                     settled."
                );
            }
            compared += rest.len();
            continue;
        }
        for (what, other) in rest {
            if other.args != first.args {
                report.push(format!(
                    "`{name}`: {what} takes {:?}, {first_what} takes {:?}",
                    other.args, first.args
                ));
            }
            assert_eq!(
                (other.whole, other.depth_prefix_plan, other.in_place()),
                (first.whole, first.depth_prefix_plan, first.in_place()),
                "`{name}` is stated differently in {what} and {first_what}. \
                 These three are facts about how a TRACE may use the kernel -- \
                 whether it consumes its whole operand, whether it joins the \
                 depth-prefix plan, which operands must be aliased -- and a \
                 trace does not know which backend will run it."
            );
            // WHICH OPERAND, which `args` cannot see.
            //
            // A position wrapper is transparent for both columns above it:
            // `InSlot<0, Buf>` reports `Buf`'s `Ty` and `Buf`'s `Provenance`,
            // so a plane that states a slot and a plane that does not compare
            // EQUAL on `args` and the drift goes unreported. That is not a
            // hypothetical -- it is the shape the shader planes are in right
            // now, mid-migration, and the whole point of stating the slot is
            // that something reads it.
            //
            // It belongs beside the three above rather than in `report`,
            // which collects `args` mismatches for one message: a slot is a
            // fact about the STATEMENT, exactly as `in_place` is, and the
            // same sentence covers it -- a trace does not know which backend
            // will run it, so it cannot place an operand at one index here
            // and another there.
            // `sides` STOOD HERE AND IS DELETED, but the claim it made is
            // not: a slot is a fact about the STATEMENT, so a trace that does
            // not know which backend will run it cannot place an operand at
            // one index here and another there. `Source` carries the slot now
            // -- `Slot(Kind::In, n)`, `Slot(Kind::Out, n)`, `Alias(i, o)` --
            // and it is compared just below, which is where the assertion
            // moved rather than where it went.
            // WHICH OF THE ENVIRONMENT'S QUESTIONS, which `args` cannot see
            // either, and for the same reason: `Ask<keys::TokenIds, I32s>`
            // and `Env<I32s>` are the same `Ty` and the same `Provenance`.
            //
            // The asymmetry worth naming is that this column is `None` until
            // a signature is taught, so a plane that has been and a plane
            // that has not DIVERGE here while both are correct. That is the
            // intended reading -- it is the same asymmetry `sides` has, and
            // the mirror is applied to all three planes at once precisely so
            // this never has to be excused.
            assert_eq!(
                other.sources, first.sources,
                "`{name}` asks the binder different questions in {what} and \
                 {first_what}. Which of the fire's planes an argument is, is \
                 the STATEMENT's business: a trace does not know which \
                 backend will run it."
            );
            compared += 1;
        }
    }

    assert!(
        report.is_empty(),
        "{} crossed kernel(s) are declared differently by two backends. A \
         signature is now the only statement of an operand list, so this is \
         either a port that mistyped one or a real divergence nobody wrote \
         down -- record it in DIVERGED with the reason:\n  {}",
        report.len(),
        report.join("\n  ")
    );

    // Thirty-four, which is a count of COMPARISONS and not of kernels: a name
    // two backends carry is one, a name three carry is two. The six of
    // `layout`, crossed by all three, are therefore twelve of it; plus
    // `argmax_logits`, which metal and vulkan both crossed and which was the
    // first kernel this gate compared across two backends that arrived at it
    // independently; plus `mlp`'s four, the first LIVE ones -- every gemma and
    // every gpt-oss layer names one, so unlike `argmax_logits` these are
    // signatures a model's output depends on; plus `ssm`'s eight, which is
    // where the gate first paid for itself. Two of those eight are in DIVERGED
    // and are counted here anyway: an excused divergence is a comparison whose
    // answer is written down, not one that stopped happening. The other six were
    // settled rather than excused, and settling them moved both crates --
    // metal's four float planes stopped being spelled `Buf` when the MSL says
    // `device float*`, and vulkan stopped folding `rows * v_heads` into one
    // number the shader takes apart again. The thirty-fifth is
    // `copy_logits_bf16`, which wgpu and vulkan crossed against shader text
    // that does NOT agree -- wgpu's writes two vocabulary entries per lane and
    // vulkan's writes one -- and the signatures match anyway, because a grid
    // is not an operand. That is the gate's shape working as intended: it
    // compares what a caller must pass, and leaves what a lane does to the
    // family tests that can see the shader.
    //
    // Plus `norm`'s twelve, and they paid the same way: metal's crossing
    // showed that the four PACKED reductions need `width` as well as `axis`,
    // because their base is `group * axis_size` and a row holds `width / axis`
    // of them. Vulkan was launching one workgroup per ROW, which normalizes
    // the first head of a q/k norm and leaves the other thirty-one as the
    // projection wrote them. It took metal's signature. Four of the twelve are
    // in DIVERGED for a difference that really is two shaders -- metal sizes
    // its threadgroup on the axis, vulkan compiles a fixed 256 and loops.
    //
    // Plus `moe`'s thirteen. Six of those are excused just above, in two
    // groups of three, and the other seven matched on the first attempt --
    // including all three routed matvecs, whose twelve arguments are the
    // longest signature either backend has agreed on without an edit.
    //
    // Plus `attn`'s sixteen, fifteen of which matched at once. That includes
    // `kv_append_paged`'s sixteen slots, six of them a ring ABI neither
    // backend reads and both name one by one.
    //
    // Plus `quant`'s thirty-one -- the largest family and the one that needed
    // the least argument, because both backends were ported from the same MLX
    // bodies and the 303 instantiation names are spelled identically. Every
    // one of them agreed.
    assert_eq!(
        compared, COMPARED,
        "{compared} kernels are crossed by more than one backend and compared \
         here, and this test expects {COMPARED}. A fall is only ever right \
         alongside a fall in `CENSUS`: if the union did not shrink, then a \
         name stopped matching its counterpart, which is how a comparison \
         stops happening without anybody deleting it."
    );
}

// RETIRED, exactly as its own failure message instructed.
//
// It compared `kernels-metal::kernel_of`'s retired-row fallback against
// `kernels::sig_in` over `kernels-wgpu`'s rows — available only while a
// backend was BEHIND metal's Stage 4, and it caught the defect it was written
// for: the fallback matched a row name as a PREFIX, and 363 of 479
// entrypoints resolved to `None`, which refused every metal text at load
// time.
//
// `kernels-wgpu` is down to 19 rows over 31 entrypoints, so the comparison
// reads a fourteenth of the census. The assertion said to delete it rather
// than let it pass on a nearly empty loop, and this is that edit.
//
// What holds the rule now: `kernel_of`'s own `at_word_boundary`, which is
// unit-tested in `kernels-metal`, and `model-ir`'s load-time check, which
// refuses any text naming a symbol no backend resolves. Both are on the
// side that has the rows.
