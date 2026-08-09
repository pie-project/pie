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
    // ── MLA + latent cache: 11 unarmed symbols as well (executor_bind).
    "deepseek_v2",
    "deepseek_v3",
    "glm_moe_dsa",
    "kimi_k2",
    // ── kimi_k3's KDA: 5 unarmed symbols.
    "kimi_k3",
    // ── deepseek_v4's DSA indexer + hyper-connections: 10 unarmed.
    "deepseek_v4",
    // ── Armed and dispatchable; ONLY the facts derivation is missing.
    //    These are the cheapest ones to take off this list.
    "gemma2",
    "gemma3",
    "gemma3_text",
    "gemma3n",
    "gemma3n_text",
    "gpt_oss",
    "mixtral",
    "nemotron_h",
    "qwen3_moe",
    // ── Not a decode backbone at all: CSM is a codec stack.
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
