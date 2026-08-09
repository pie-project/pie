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
    // All four state `KvStyle::Mla` or `KvStyle::Dsv4`, and this build
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
    let missing: Vec<&str> =
        NOT_YET_SERVABLE.iter().copied().filter(|id| catalog::find(id).is_none()).collect();
    assert!(missing.is_empty(), "NOT_YET_SERVABLE names no such row: {missing:?}");

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
    use driver_api::local::{
        PIE_RS_FLAG_BUFFER_WRITE, PIE_RS_FLAG_FOLD, PieStepDesc, PieU8Slice, PieU32Slice,
    };
    use driver_cuda::serve::fire_class_of;
    use model_compiler::trace::FireClass;

    fn u32s(v: &[u32]) -> PieU32Slice {
        PieU32Slice { ptr: v.as_ptr(), len: v.len() }
    }

    /// A step carrying `requests` rows of recurrent state, with the given
    /// flags and buffer CSR. Everything else is a fire's ordinary shape.
    fn step(flags: &[u8], buf_indptr: &[u32], slots: &[u32], sampling: &[u32]) -> PieStepDesc {
        PieStepDesc {
            rs_slot_ids: u32s(slots),
            rs_slot_flags: PieU8Slice { ptr: flags.as_ptr(), len: flags.len() },
            rs_buffer_slot_indptr: u32s(buf_indptr),
            sampling_indices: u32s(sampling),
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
