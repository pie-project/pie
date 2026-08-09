//! Which `model_type` this shell can OPEN, held against what the loader
//! can author.
//!
//! §3.3 of the plan reads: "the executor has arms for more than it can
//! load — nemotron_h, gemma3n, gpt-oss/mixtral, gemma-2 all dispatch but
//! cannot be opened, because facts derivation and the weight binder are
//! per-family and were only written three times."
//!
//! That sentence was true and nothing enforced it, which is the problem:
//! the gap was a paragraph in a document rather than a list a build could
//! check. Before the registry it could not even be stated, because the
//! shell decided the family by SNIFFING weight names — there was no set to
//! enumerate, and a model type nobody had thought about was answered by
//! whichever sniff happened to match rather than refused.
//!
//! So this file closes it the way `executor_bind.rs` closes the unarmed
//! symbols: as a set that must be exactly right. A family that gains a
//! derivation leaves `NOT_YET_OPENABLE`; a family that appears in the
//! loader and nowhere else joins it deliberately, in a commit, rather than
//! being discovered by a checkpoint that will not boot.

#![cfg(all(feature = "_cuda", feature = "abi"))]

use std::collections::BTreeSet;

use driver_cuda_new::abi_shell::openable_model_types;

/// Model types the loader can author but this shell cannot yet open.
///
/// Every line is a family whose forward IS declared and whose arms DO
/// dispatch — what is missing is only the derivation from the checkpoint's
/// own config, and in some cases the weight binder. Grouped by what has to
/// be written, because that is how a reader decides whether a line is
/// theirs.
const NOT_YET_OPENABLE: &[&str] = &[
    // Not a decode backbone at all: CSM is a codec stack, so it has no
    // fire class this shell serves and no facts to derive.
    "csm",
];

#[test]
fn the_openable_set_and_the_authorable_set_account_for_each_other() {
    let openable: BTreeSet<&str> = openable_model_types().into_iter().collect();
    let authorable: BTreeSet<&str> =
        model::contract::HF_ROWS.iter().map(|(k, _)| *k).collect();

    // Nothing may be openable that the loader cannot author: the facts
    // would have no weights to bind.
    let orphans: Vec<&str> = openable.difference(&authorable).copied().collect();
    assert!(
        orphans.is_empty(),
        "these model types have a facts derivation but no author, so a \
         checkpoint could never reach them: {orphans:?}"
    );

    // And the complement is the family-coverage gap, which must be
    // exactly the stated list.
    let gap: BTreeSet<&str> = authorable.difference(&openable).copied().collect();
    let stated: BTreeSet<&str> = NOT_YET_OPENABLE.iter().copied().collect();
    assert_eq!(
        gap, stated,
        "the set of families the loader can author but this shell cannot \
         open has changed. If a derivation landed, delete its line; if a \
         model type was added to the loader, add a line — do not leave it \
         to be discovered by a checkpoint that will not boot."
    );
}

#[test]
fn no_model_type_is_listed_twice() {
    let rows = openable_model_types();
    let unique: BTreeSet<&str> = rows.iter().copied().collect();
    assert_eq!(rows.len(), unique.len(), "a model type has two rows");

    let unique_gap: BTreeSet<&str> = NOT_YET_OPENABLE.iter().copied().collect();
    assert_eq!(
        NOT_YET_OPENABLE.len(),
        unique_gap.len(),
        "a line in NOT_YET_OPENABLE is duplicated"
    );
}

#[test]
fn the_three_live_families_are_openable() {
    // The deployments the A/Bs actually run. If one of these ever leaves
    // the table, `real_prefill`, `real_hybrid` and `real_gemma4` stop
    // being able to open their checkpoints — and they would fail with a
    // status code rather than a message naming this table.
    let openable: BTreeSet<&str> = openable_model_types().into_iter().collect();
    for want in ["qwen3", "gemma4", "qwen3_5"] {
        assert!(openable.contains(want), "{want} must stay openable");
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
    use driver_abi::local::{
        PIE_RS_FLAG_BUFFER_WRITE, PIE_RS_FLAG_FOLD, PieStepDesc, PieU8Slice, PieU32Slice,
    };
    use driver_cuda_new::abi_shell::fire_class_of;
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

    #[test]
    fn every_row_replaying_buffered_tokens_is_the_commit_pass() {
        // FOLD without BUFFER_WRITE, over a non-empty buffer span: the
        // row's tokens come out of the slabs rather than out of a GEMM.
        let s = step(&[PIE_RS_FLAG_FOLD; 2], &[0, 3, 6], &[0, 1], &[0]);
        assert_eq!(fire_class_of(&s, 9, 2), Ok(FireClass::CommitAdvance));
    }

    #[test]
    fn a_row_that_writes_its_slabs_is_the_frozen_verify_pass() {
        let flags = [PIE_RS_FLAG_FOLD | PIE_RS_FLAG_BUFFER_WRITE; 2];
        let s = step(&flags, &[0, 3, 6], &[0, 1], &[0]);
        assert_eq!(fire_class_of(&s, 9, 2), Ok(FireClass::FrozenVerify));
    }

    /// `StateOnly` is NOT read here, and this pins the reason.
    ///
    /// Its signal is an empty `sampling_indices`, and the shell does not
    /// read that field: it marks every row `samples: true`. So the field
    /// is empty on every fire, and keying the class on it made every
    /// hybrid prefill a service pass with no epilogue — the logits simply
    /// stopped arriving. When readout rows reach `Row::samples`, this
    /// test is the one that flips.
    #[test]
    fn recurrent_state_alone_is_not_yet_a_service_pass() {
        let s = step(&[0, 0], &[], &[0, 1], &[]);
        assert_eq!(fire_class_of(&s, 9, 2), Ok(FireClass::Prefill));
        assert_eq!(fire_class_of(&s, 2, 2), Ok(FireClass::Decode));
    }

    #[test]
    fn a_replay_row_may_not_share_a_fire_with_a_computing_one() {
        // Row 0 replays, row 1 owns no slabs and computes.
        let s = step(&[PIE_RS_FLAG_FOLD, 0], &[0, 3, 3], &[0, 1], &[0]);
        assert!(fire_class_of(&s, 9, 2).is_err(), "the mixed fire is refused");
    }
}
