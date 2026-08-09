//! Which catalog rows this shell can SERVE, stated as a set.
//!
//! §3.3 of the plan reads: "the executor has arms for more than it can
//! load — nemotron_h, gemma3n, gpt-oss/mixtral, gemma-2 all dispatch but
//! cannot be opened, because facts derivation and the weight binder are
//! per-family and were only written three times."
//!
//! That sentence was true and nothing enforced it, which is the problem:
//! the gap was a paragraph in a document rather than a list a build could
//! check. Before there was a registry it could not even be STATED,
//! because the shell decided the family by SNIFFING weight names — there
//! was no set to enumerate, and a model type nobody had thought about was
//! answered by whichever sniff happened to match rather than refused.
//!
//! # HALF THIS FILE'S JOB IS DONE BY THE TYPE NOW
//!
//! It used to reconcile TWO sets: `deployment_cuda::openable_model_types`
//! (what a derivation existed for) against `contract::HF_ROWS` (what an
//! author existed for), both keyed on a `config.json` `model_type`
//! string. A family in one and not the other was a checkpoint that would
//! load and never fire, or fire and never load.
//!
//! Those sets cannot differ any more. `catalog::Variant` requires
//! `author` and `deployment` and `trace` and `chat` of EVERY row, with
//! no default bodies, so a row that can be authored is a row that can be
//! deployed by construction — there is no second table to fall out of.
//!
//! What is left is the part a type still cannot hold: a row may compile
//! and still REFUSE at `deployment()`, because this build has no KV pool
//! or no forward text for its shape. That refusal is a real gap and it
//! is stated below, in the same closed-set idiom, so a row that gains a
//! path leaves the list in a commit rather than being discovered by a
//! checkpoint that will not boot.

#![cfg(all(feature = "_cuda", feature = "abi"))]

use std::collections::BTreeSet;

use model::catalog::{self, Deployed};

/// Catalog rows this build cannot project a `Deployment` for.
///
/// Every line is a model whose row EXISTS and whose author is written —
/// what is missing is a driver-side path for the shape it projects.
/// Grouped by what has to be written, because that is how a reader
/// decides whether a line is theirs.
const NOT_YET_SERVABLE: &[&str] = &[
    // Not a decode backbone at all: CSM is a codec stack, so it has no
    // fire class this shell serves.
    "csm-1b",
    // Gemma-4's routed sibling: `attention_k_eq_v` reads V out of the K
    // projection and the block is routed over 128 experts. This build
    // traces neither leg, and the row says so at the door rather than
    // deploying a stack whose first fire would find no kernel.
    "gemma-4-26b-a4b",
    // ── THE MLA LINEAGE ─────────────────────────────────────────────
    //
    // All four state `KvStyle::Mla` or `KvStyle::CompressedPlane`, and this build
    // provisions neither store: a compressed KV plane and a positional
    // one do not fit the k/v pair the pager allocates.
    //
    // These four ARE the defect `KvStyle` was made an enum to end. The
    // doc on it says so: the MLA lineage "registered in `FACTS_ROWS`,
    // answered `facts_from_hf` happily, and had no forward path at all"
    // — a registry hit, a successful load, and a death at the first
    // fire. A refusal at deployment is that same fact told early, and
    // this list is what keeps it from being told silently.
    "deepseek-v4",
    "glm-5-106b-a12b",
    "kimi-k2",
    "kimi-k3",
];

/// Every catalog row either projects a `Deployment` or is on the list.
#[test]
fn every_row_deploys_or_is_stated_unservable() {
    let stated: BTreeSet<&str> = NOT_YET_SERVABLE.iter().copied().collect();
    let mut refused: BTreeSet<&str> = BTreeSet::new();
    for row in catalog::catalog() {
        if row.deployment(Deployed::single()).is_err() {
            refused.insert(row.id());
        }
    }

    // A row on the list that now deploys: delete its line. A row that
    // refuses and is not on the list: a checkpoint would match it,
    // load, and die at the door.
    let unstated: Vec<&str> = refused.difference(&stated).copied().collect();
    assert!(
        unstated.is_empty(),
        "these rows refuse to deploy and are not stated as unservable, so a \
         checkpoint matching one loads and then fails at the door: \
         {unstated:?}"
    );
    let stale: Vec<&str> = stated.difference(&refused).copied().collect();
    assert!(
        stale.is_empty(),
        "these rows are listed as unservable but deploy fine; delete their \
         lines: {stale:?}"
    );
}

/// A row named in the list is a row that exists.
///
/// Separate from the reconciliation above so a typo in the list reports
/// as a typo rather than as a stale line.
#[test]
fn the_unservable_list_names_only_real_rows() {
    let missing: Vec<&str> = NOT_YET_SERVABLE
        .iter()
        .copied()
        .filter(|id| catalog::find(id).is_none())
        .collect();
    assert!(
        missing.is_empty(),
        "NOT_YET_SERVABLE names no such row: {missing:?}"
    );

    let unique: BTreeSet<&str> = NOT_YET_SERVABLE.iter().copied().collect();
    assert_eq!(unique.len(), NOT_YET_SERVABLE.len(), "a line is duplicated");
}

/// The deployments the A/Bs actually run stay servable.
///
/// If one of these ever stops deploying, `real_prefill`, `real_hybrid`
/// and `real_gemma4` lose the ability to open their checkpoints — and
/// they would fail with a status code rather than a message naming this
/// file.
#[test]
fn the_three_live_families_are_servable() {
    for want in ["qwen3-0.6b", "gemma-4-e4b-it", "qwen3.5-4b"] {
        let row = catalog::find(want).unwrap_or_else(|| panic!("{want} must stay in the catalog"));
        assert!(
            row.deployment(Deployed::single()).is_ok(),
            "{want} must stay servable"
        );
    }
}

/// THE CLASS IS ON THE WIRE, and these are the four readings of it.
///
/// The shell used to answer "which fire class" from the fire's SHAPE
/// alone — one row per request is a decode, anything else is prefill —
/// which cannot see a service pass at all, because a service pass is not
/// a shape. It is what the pass is FOR, and the recurrent-state flags
/// have said so since ABI v23.
///
/// This is the same derivation the C++ composer ran
/// (`pipeline/batch_compose.hpp`, `RsExecutionMode`), held against its
/// four outcomes. The mixed case is refused for the composer's reason: a
/// replay row gathers activations out of its slabs and a computing row
/// does not, so no single op list serves both.
mod fire_class {
    use driver_api::local::{PIE_RS_FLAG_BUFFER_WRITE, PIE_RS_FLAG_FOLD};
    use driver_api::{LaunchPlan, StepSubmission};
    use driver_cuda::serve::fire_class_of;
    use model_compiler::trace::FireClass;

    /// A step carrying `requests` rows of recurrent state, with the given
    /// flags and buffer CSR. Everything else is a fire's ordinary shape.
    ///
    /// It returns an OWNED step now. The `PieStepDesc` version returned a
    /// struct of pointers into the caller's argument slices, which the C
    /// shape had no way to tie a lifetime to.
    fn step(flags: &[u8], buf_indptr: &[u32], slots: &[u32], sampling: &[u32]) -> StepSubmission {
        StepSubmission {
            plan: LaunchPlan {
                rs_slot_ids: slots.to_vec(),
                rs_slot_flags: flags.to_vec(),
                rs_buffer_slot_indptr: buf_indptr.to_vec(),
                sampling_indices: sampling.to_vec(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn a_fire_with_no_recurrent_state_is_read_off_its_shape() {
        let s = step(&[], &[], &[], &[0]);
        assert_eq!(fire_class_of(&s, 4, 4), Ok(FireClass::Decode));
        assert_eq!(fire_class_of(&s, 9, 4), Ok(FireClass::Prefill));
    }

    /// THE SERVICE CLASSES ARE GONE, and this is what replaces four
    /// tests that pinned them (`.wiki/driver/graph.md` §4.2).
    ///
    /// `CommitAdvance`, `FrozenVerify` and `StateOnly` were derived from
    /// the recurrent-state flags. The driver executes those flags now
    /// rather than classifying on them: a speculative decode buffers its
    /// tokens and folds only the accepted prefix, so a rejected token is
    /// never folded and there is no repair pass to select. The class is a
    /// shape question again, and the flags no longer change the answer.
    #[test]
    fn the_recurrent_state_flags_no_longer_pick_a_class() {
        let fold = step(&[PIE_RS_FLAG_FOLD; 2], &[0, 3, 6], &[0, 1], &[0]);
        assert_eq!(fire_class_of(&fold, 9, 2), Ok(FireClass::Prefill));
        assert_eq!(fire_class_of(&fold, 2, 2), Ok(FireClass::Decode));

        let write = step(
            &[PIE_RS_FLAG_FOLD | PIE_RS_FLAG_BUFFER_WRITE; 2],
            &[0, 3, 6],
            &[0, 1],
            &[0],
        );
        assert_eq!(fire_class_of(&write, 9, 2), Ok(FireClass::Prefill));

        // The mixed fire the composer used to refuse: a replaying row
        // beside a computing one. There is no op-list difference between
        // them any more, so there is nothing to refuse.
        let mixed = step(&[PIE_RS_FLAG_FOLD, 0], &[0, 3, 3], &[0, 1], &[0]);
        assert_eq!(fire_class_of(&mixed, 9, 2), Ok(FireClass::Prefill));
    }
}

/// The GQA group sizes this build's decode instantiates, and the rows
/// they exclude.
///
/// A SECOND kind of unservable, and a later one: these rows project a
/// `Deployment` fine — the shape is ordinary, the KV is paged, the
/// author is written — and `serve::load` refuses them anyway because
/// FlashInfer's decode was never instantiated at their ratio.
///
/// It is stated here rather than left to be discovered because the
/// alternative is worse than a refusal. `refuse_unservable_gqa` asked
/// this at HEAD and the catalog refactor dropped it on the floor: the
/// check moved to `Deployment::servable_by` and, for one commit, nobody
/// called it. An uninstantiated ratio reaching the decode is not a
/// refusal but a THROW crossing the C ABI, which is undefined
/// behaviour — so the cost of forgetting this is not a bad error
/// message.
///
/// A line leaves by someone instantiating the group size in the kernel
/// build, at which point `DECODE_GQA_GROUPS` grows and this test fails
/// pointing at the line to delete.
const UNSERVABLE_GQA: &[&str] = &[
    // 40 over 8 is a group of five.
    "olmo-3-32b",
    "qwen2.5-14b",
    "qwen2.5-32b",
    "qwen3-14b",
    // 14 over 2 and 28 over 4 are both seven.
    "qwen2.5-0.5b",
    "qwen2.5-7b",
    // Twelve over two, which the `servable_by` doc names.
    "qwen2.5-1.5b",
    // 24 over 4 is six — the doc's OTHER named example, and the one
    // that proves this is not the llama lineage's business: qwen3.6-27b
    // is a GDN hybrid, reaching the same decode from a different
    // generation with the same unservable ratio.
    "qwen3.6-27b",
    // 64 over 4 is sixteen.
    "qwen3-235b-a22b",
];

#[test]
fn every_deployable_row_is_servable_by_this_builds_decode_or_is_stated() {
    // MOVED. The list was `model`'s while the servable set was a fact about
    // the model text; it is `driver_cuda::serve`'s now, because what a driver
    // can serve is a fact about the driver. `project.rs:734` says so in the
    // comment it left behind.
    use driver_cuda::serve::DECODE_GQA_GROUPS;

    let stated: BTreeSet<&str> = UNSERVABLE_GQA.iter().copied().collect();
    let mut refused: BTreeSet<&str> = BTreeSet::new();
    let mut checked = 0usize;
    for row in catalog::catalog() {
        // Only rows that GET to the question. A row refused for its KV
        // shape never reaches `servable_by`, and listing it here would
        // record one gap as two.
        let Ok(d) = row.deployment(Deployed::single()) else {
            continue;
        };
        checked += 1;
        if d.servable_by(DECODE_GQA_GROUPS).is_err() {
            refused.insert(row.id());
        }
    }

    assert!(
        checked > 20,
        "only {checked} rows projected a deployment, so this audit is \
         asking almost nothing"
    );

    let unstated: Vec<&str> = refused.difference(&stated).copied().collect();
    assert!(
        unstated.is_empty(),
        "these rows deploy but this build's decode has no kernel at their \
         GQA ratio, and nothing says so: a checkpoint matching one loads \
         through every other check and is refused at the last door. State \
         them or instantiate the group size: {unstated:?}"
    );

    let stale: Vec<&str> = stated.difference(&refused).copied().collect();
    assert!(
        stale.is_empty(),
        "these rows are listed as unservable at their GQA ratio but this \
         build serves them; delete their lines: {stale:?}"
    );
}

/// The set this build states must be the set the refusal uses.
///
/// Pinned because `DECODE_GQA_GROUPS` is what makes the list above
/// meaningful: a set that quietly grew to include everything would empty
/// the list and pass, having stopped asking.
#[test]
fn the_instantiated_set_is_the_one_the_kernels_were_built_for() {
    // MOVED. The list was `model`'s while the servable set was a fact about
    // the model text; it is `driver_cuda::serve`'s now, because what a driver
    // can serve is a fact about the driver. `project.rs:734` says so in the
    // comment it left behind.
    use driver_cuda::serve::DECODE_GQA_GROUPS;
    assert_eq!(
        DECODE_GQA_GROUPS,
        &[1, 2, 3, 4, 8],
        "FlashInfer's decode instantiations; changing this without changing \
         the kernel build is how a throw reaches the C ABI"
    );
}

/// The rows whose norm gain is stored as an OFFSET FROM ONE, so that
/// firing the norm means `(1 + w) * x`.
///
/// Stated as a set for the same reason `NOT_YET_SERVABLE` is: the fact
/// used to be READ OFF the norm placement, and that reading is wrong for
/// exactly one published family. gemma-1, -2, -3 and -3n pair the
/// sandwich with the offset; gemma-4 publishes the sandwich and stores a
/// plain multiplier. A derivation cannot tell those apart, so every row
/// answers and this list is the answer collected.
///
/// It did not fail loudly when it was derived, which is why a list is
/// worth its maintenance. `(1 + w)/w` is 1.002 where `w` is 444 and 1.38
/// where `w` is 2.6, so the largest gains agreed to three digits while
/// the ordinary ones were off by a third — finite, plausible, wrong.
const FOLDS_UNIT_OFFSET: &[&str] = &[
    "gemma-2-2b",
    "gemma-2-9b",
    "gemma-2-27b",
    "gemma-3-1b",
    "gemma-3-4b",
    "gemma-3-12b",
    "gemma-3-27b",
    "gemma-3n-e2b",
    "gemma-3n-e4b",
];

/// Every deployable row's fold is the one this list states.
///
/// A new generation cannot join the catalog without landing on one side
/// of this: either its id appears here or it is asserted not to fold.
/// That is the whole point — the previous arrangement let a family
/// inherit an answer nobody wrote down.
#[test]
fn the_norm_fold_is_stated_by_every_row_and_derived_by_none() {
    let stated: BTreeSet<&str> = FOLDS_UNIT_OFFSET.iter().copied().collect();
    let mut folds: BTreeSet<&str> = BTreeSet::new();
    for row in catalog::catalog() {
        if let Ok(d) = row.deployment(Deployed::single()) {
            if d.norm_unit_offset {
                folds.insert(row.id());
            }
        }
    }

    let unstated: Vec<&str> = folds.difference(&stated).copied().collect();
    assert!(
        unstated.is_empty(),
        "these rows fold `(1 + w)` and are not stated to: {unstated:?}"
    );
    let stale: Vec<&str> = stated.difference(&folds).copied().collect();
    assert!(
        stale.is_empty(),
        "these rows are stated to fold and do not: {stale:?}"
    );
}

/// gemma-4 is the exception, and it is the reason the field exists.
///
/// Kept separate from the set above so that the ONE row that breaks the
/// placement inference is named in a test of its own. A regression here
/// is a whole generation served with a fold its checkpoint never asked
/// for, and the failure is silent.
#[test]
fn gemma_4_sandwiches_its_norms_without_folding_them() {
    let mut checked = 0;
    for row in catalog::catalog() {
        if !row.id().starts_with("gemma-4") {
            continue;
        }
        let Ok(d) = row.deployment(Deployed::single()) else {
            continue;
        };
        assert!(
            !d.norm_unit_offset,
            "`{}` must store a plain multiplier: `gemma_4/forward/mod.rs` \
             fires `NormVariant::Plain` at all fourteen of its norm sites",
            row.id()
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "the gemma-4 rows are gone; this test is now vacuous"
    );
}
