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

/// The gates the sibling runs and this port does not, named.
///
/// `tests/gpu` is the harness where a driver is booted for real and asked for
/// tokens through the client edge. `driver-vulkan` has twelve files there;
/// this backend had three, and nothing anywhere said which nine were missing.
/// A gap nobody names is a gap nobody closes — and one of the nine turned out
/// to be unwritable rather than unwritten, because `common::wgpu_standalone_
/// toml` hard-coded `[model] name = "qwen3"` and so no wgpu gate could serve a
/// second architecture at all.
///
/// So the difference is written down, with what covers each here. Two rules,
/// both self-cleaning: a twin that gets written must leave this list, and a
/// gate the sibling adds must join it.
///
/// It skips when `tests/gpu` is absent, for the reason this file's header
/// gives about `crates/driver-vulkan`.
#[test]
fn the_sibling_gates_this_port_has_no_twin_for_are_named() {
    /// A sibling gate with no wgpu twin, and what stands in for it here.
    ///
    /// "Covered" means something in THIS tree fails if the path breaks —
    /// not that the subject is unimportant. Where nothing covers it, the
    /// entry says so, because an excuse and a gap should not read alike.
    ///
    /// One entry said exactly that and is gone: `shared_prefix` had no cover,
    /// so `wgpu_shared_prefix` was written and this list shed it — which is
    /// what a list like this is for.
    const NO_TWIN: &[(&str, &str)] = &[
        (
            "add_program",
            "not a gap: that gate is a DIAGNOSTIC for the client edge and its \
             own header says it does not care about the driver — it picked \
             the sibling only because that is the cheapest boot on a machine \
             with no CUDA. A wgpu twin would boot a second driver to measure \
             the same chunked upload through the same gateway, and every wgpu \
             gate already carries a program over that edge to get started",
        ),
        (
            "boot_smoke",
            "covered: all four wgpu gates boot the same standalone through \
             `common::boot_wgpu`, so a boot that stopped working fails every \
             one of them rather than none",
        ),
        (
            "chat_completion_e2e",
            "covered: `wgpu_second_model` runs that exact inferlet — the same \
             `chat-completion` wasm, the same greedy `Paris` — through the \
             whole stack, on a model this driver was not written against",
        ),
        (
            "two_conversations",
            "covered: `wgpu_many_conversations` is the same proof at EIGHT, \
             and its own header says why two is the weaker number — a \
             two-request frame's first request starts at row zero, so an \
             off-by-a-base and a correct answer are the same table",
        ),
        (
            "grammar_constrained",
            "covered: `tests/inferlets`' `asap-grammar-aligned-decoding` \
             passes on this driver, which is a host-held grammar steering a \
             decode token by token through the client edge",
        ),
        (
            "programmable_sampler",
            "covered: the curated suite's guest-written samplers — \
             `gumbel-watermark`, `synthid-tournament-sampling`, `xtc-sampling` \
             and a dozen more — are PTIR programs the guest wrote, and they \
             pass here on two architectures",
        ),
        (
            "sampled_completion",
            "covered: every curated sampler test samples rather than takes an \
             argmax, and `greedy-decoding-is-the-same-alone-and-in-a-crowd` is \
             the one that pins the argmax against them",
        ),
        (
            "sampling_primitives",
            "covered at the kernel rather than the gate: `kernels-wgpu`'s \
             `tests/gpu.rs` computes the sampler op set on the device and \
             compares it against host references, and refuses a perturbed one",
        ),
    ];

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/gpu/tests");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        println!(
            "SKIP: no {}, so the gate difference is unmeasured",
            dir.display()
        );
        return;
    };

    let (mut vulkan, mut wgpu) = (BTreeSet::new(), BTreeSet::new());
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let Some(stem) = name.strip_suffix(".rs") else {
            continue;
        };
        if let Some(rest) = stem.strip_prefix("vulkan_") {
            vulkan.insert(rest.to_string());
        } else if let Some(rest) = stem.strip_prefix("wgpu_") {
            wgpu.insert(rest.to_string());
        }
    }
    // The directory is identified by something that IDENTIFIES it, not by a
    // count of files in it. This read `vulkan.len() >= 10` and went red the
    // moment upstream deleted three vulkan gates (`244df6054`, "Census 34 ->
    // 21") -- a hand-kept number about another port's files, which is a number
    // that drifts every time that port adds or removes one, and whose drift
    // says nothing about whether this scan found the right directory.
    //
    // `common` is what makes it the right directory: every gate in the tree
    // includes it, and no other directory has it.
    assert!(
        dir.join("common").is_dir(),
        "no `common` module here, so this is not the gate directory: {}",
        dir.display()
    );
    assert!(
        !vulkan.is_empty() && !wgpu.is_empty(),
        "found {} vulkan and {} wgpu gates in {}; a port with none of either \
         is a scan that matched nothing",
        vulkan.len(),
        wgpu.len(),
        dir.display()
    );

    let listed: BTreeSet<&str> = NO_TWIN.iter().map(|(g, _)| *g).collect();
    let missing: BTreeSet<&str> = vulkan
        .iter()
        .map(String::as_str)
        .filter(|g| !wgpu.contains(*g))
        .collect();
    assert_eq!(
        missing, listed,
        "the sibling gates without a wgpu twin are no longer the ones NO_TWIN \
         names. If a twin was written, drop its entry; if the sibling added a \
         gate, add it with what covers it here — or with the fact that nothing \
         does, which is a finding and not a failure."
    );
}

/// The first ported routine asks for the grid its row asked for.
///
/// `.wiki/kernel-x/wgpu-refactor.md` is the plan: each `kernel!` row becomes a
/// `fn` whose body states the entrypoint and the lanes, and `LaunchRule` and
/// `geometry.rs` dissolve into those bodies. The danger of that migration is
/// silent: a body that computes a *different* grid still dispatches, still
/// returns `Ok`, and produces wrong numbers rather than a refusal.
///
/// So the two are compared for as long as both exist. This reads the REAL
/// shader's `@workgroup_size` — not a transcription of it — asks
/// `geometry::groups` what the row's `LaunchRule::Elementwise` wants, asks the
/// routine what it wants, and requires them equal after the driver's own
/// `div_ceil`.
///
/// It is deliberately in `driver-wgpu`: `kernels-wgpu` cannot depend on this
/// crate, so its own half of this check (`the_body_asks_for_the_elementwise_grid`)
/// compares against a transcribed `width * rows`. This one compares against
/// the function that decides it.
///
/// RETIRED WITH `kernels-wgpu`'s TEST TREE. That name is a record of a
/// measurement now, not a live proof: the crate lost `tests/` and every
/// in-file `mod tests` when the three shader planes moved their numbers to
/// the fire that reads them, and nothing in this workspace re-runs it. What
/// it reported is still why the sentence above says what it says; what is
/// gone is the thing that would notice if it stopped being true.
#[test]
fn the_first_ported_routine_asks_for_the_grid_its_row_asked_for() {
    use driver_wgpu::geometry::{Dims, Module, Rule, groups};
    use kernels_wgpu::routine::{ArgValue, Encode, Fire, Tensor};

    /// The lanes the body asked for, and the row count it will be told.
    ///
    /// IT RECORDS ONE THING NOW. `ple_combine` used to reach its row count
    /// through `ctx.ask::<i32, keys::Rows>()`, and this probe answered on
    /// that channel; the no-ask sweep retired `Source::Named` and the whole
    /// key vocabulary, so the routine takes rows as a `Const<i32>` on the
    /// signature and `resolve` has nothing left to answer. It still exists —
    /// the `Encode` trait declares it — and refuses everything, because
    /// this test is about the LANES the body asks for and not a scalar it
    /// no longer reads through the resolver.
    #[derive(Default)]
    struct Lanes {
        seen: std::cell::RefCell<Option<[u32; 3]>>,
    }
    impl Encode for Lanes {
        fn fire(&self, fire: Fire, _args: &[ArgValue]) -> Result<(), kernels::routine::Refusal> {
            *self.seen.borrow_mut() = Some(fire.lanes);
            Ok(())
        }

        fn resolve(
            &self,
            _ty: kernels::Ty,
            _source: kernels::Source,
        ) -> Result<ArgValue, kernels::routine::Refusal> {
            // No routine on this backend reaches `ctx.resolve` any more —
            // every fact a body used to ask for is a mark on the signature
            // now. A refusal here catches a routine that regresses to the
            // ask channel loudly, at the first argument it does.
            Err(kernels::routine::Refusal::Unstated {
                what: "a fact this probe does not answer",
            })
        }
    }

    // The module's own divisor, read off the shader this routine names.
    let source =
        kernels_wgpu::entrypoint_source("ple_combine_bfloat16", kernels_wgpu::Capability::Baseline)
            .expect("the tree carries the entrypoint the routine names");
    let declared = driver_wgpu::reflect::declared(&source).expect("it reflects");
    let module = Module::new(declared.local);

    // The RULE, stated rather than read off a row: `layout` has retired and
    // `ple_combine` has none. It is `Elementwise` — `[width * rows, 1, 1]` —
    // which is what `every_launchs_scalars_land_where_its_module_reads_them`
    // compared this body against, on every rectangle of every text, in the
    // commit that armed it. What this test still adds is the DRIVER's
    // `groups`: the body states lanes and this is the function that turns
    // them into workgroups, so a `div_ceil` that disagreed with the body's
    // extent would show here and nowhere else.
    let rule = Rule::Elementwise;

    for (rows, width) in [(1_u32, 64_u32), (7, 64), (3, 4096), (1, 1)] {
        let dims = Dims {
            rows,
            width,
            in_width: width,
            ..Dims::default()
        };
        let want = groups(rule, dims, module).expect("the rule answers");

        let to = Lanes::default();
        // THE WIDTH RIDES THE OPERAND. `In<Tensor<_>>` carries the
        // rectangle the statement placed, so the body reads `proj.width`
        // off its own argument where it used to take a separate `Env`.
        let w = i32::try_from(width).expect("fits");
        let r = i32::try_from(rows).expect("fits");
        kernels_wgpu::layout::ple_combine(
            &to,
            kernels::routine::In {
                ptr: Tensor::new(0),
                rows: 1,
                width: w,
            },
            kernels::routine::In {
                ptr: Tensor::new(1),
                rows: 1,
                width: w,
            },
            kernels::routine::Out {
                ptr: Tensor::new(2),
                rows: 1,
                width: w,
            },
            // THE SCALE IS A MARK NOW. It reaches no grid -- this test asserts
            // the lane rule and nothing else -- so the value is the honest one
            // rather than a placeholder: `layout/ple_combine.wgsl` divides the
            // projection by root two, and `kernels-vulkan`'s twin of this test
            // states the same constant.
            kernels::Const::new(core::f32::consts::FRAC_1_SQRT_2),
            // AND SO IS ROWS. The no-ask sweep took `ctx.ask::<_, keys::Rows>()`
            // off this body and put a `Const<i32>` on its signature in its
            // place, so the row count reaches the lanes computation as an
            // argument the caller states -- which is what a sweep of
            // rectangles is about to do, one call per point.
            kernels::Const::new(r),
        )
        .expect("the body dispatches");
        let lanes = to.seen.borrow().expect("it dispatched once");

        let got = [
            lanes[0].div_ceil(declared.local[0]),
            lanes[1].div_ceil(declared.local[1]),
            lanes[2].div_ceil(declared.local[2]),
        ];
        assert_eq!(
            got, want,
            "at rows={rows} width={width}, the ported body and \
             `LaunchRule::Elementwise` disagree about the grid"
        );
    }
}
