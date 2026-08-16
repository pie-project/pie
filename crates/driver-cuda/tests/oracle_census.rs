//! Which oracles can still be run, and which are descriptions of a run that
//! can never happen again.
//!
//! **Sixteen of the eighteen `run.sh` cannot run** — not "fail for want of a
//! GPU" but cannot run anywhere, because the sources they copy were deleted.
//! That is worse than a deleted test, because they read as live
//! infrastructure.
//!
//! This asserts **path existence** and nothing else, so it is self-retiring in
//! both directions; see [`ALIVE`], [`RETIRED`] and [`every_oracle_is_classified`].

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    // `CARGO_MANIFEST_DIR` is `crates/driver-cuda`.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/driver-cuda has two ancestors")
        .to_path_buf()
}

fn oracle_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle")
}

/// Why an oracle's `run.sh` cannot run, and the input that proves it.
///
/// `at` is the phase the script dies in, and the two values distinguish a
/// script that never reaches a compiler from one that does. `store` and
/// `weight_view` copy only files that exist and then hand the compiler a path
/// that does not, so a check reading `cp` arguments alone called them alive.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Dies {
    /// Dies at a `cp` under `set -euo pipefail`, before any compiler runs.
    Cp,
    /// Dies at the compiler, on a positional input that is not there.
    Compile,
    /// Does NOT die. The tree path is gone -- so the row still asserts that,
    /// and a restore into the tree still has to be noticed -- but the script
    /// fetches the source from git and runs to completion.
    ///
    /// This is not a softer `Cp`. It is the opposite finding, and it costs
    /// something to claim: [`a_restoring_oracle_can_actually_restore`] makes
    /// every row using it name a revision whose blobs git can still produce.
    /// Without that, the variant would be a way to describe an oracle as
    /// alive without ever checking, which is the failure `Cp` rows caused in
    /// the other direction.
    GitRestores,
}

/// The sixteen, each with the first input its `run.sh` cannot find.
///
/// The path is repo-relative and is asserted MISSING. Only the first absent
/// input is listed, because it is the one `set -e` stops on and listing the
/// rest would make this table decay every time an unrelated file moved.
const DEAD: &[(&str, Dies, &str, &str)] = &[
    // ── Cause A: `crates/driver-cuda/csrc` deleted wholesale.
    //
    // Fourteen of the sixteen. These compiled the driver's own C++ host layer
    // — workspace planner, KV cache, stage hooks — which is now Rust.
    ("attn_score", Dies::Cp, "crates/driver-cuda/csrc/src/model/attn_score.cu",
     "the attention-score observation hook"),
    ("attn_ws", Dies::Cp, "crates/driver-cuda/csrc/src/attention_workspace.cpp",
     "the attention workspace's size and offset arithmetic"),
    ("caches", Dies::Cp, "crates/driver-cuda/csrc/src/store/mla_cache.cpp",
     "MLA, DSv4-compress and swap-pool cache geometry"),
    ("kv_cache", Dies::Cp, "crates/driver-cuda/csrc/src/store/kv_cache.cpp",
     "paged KV cache layout"),
    ("kv_cache_live", Dies::Cp, "crates/driver-cuda/csrc/src/store/kv_cache.cpp",
     "KV cache behaviour under live page allocation"),
    ("lora_stage", Dies::Cp, "crates/driver-cuda/csrc/src/model/llama_like/llama_like.cpp",
     "LoRA staging order"),
    // `lora_stage/run.sh` names TWO missing inputs now, and this row asserts
    // only the first because the first is where `set -e` stops. The second is
    // `../llama_like_prepare/stub/`, deleted with that oracle — see
    // [`RETIRED`]. It matters when the self-retiring direction fires: restore
    // `llama_like.cpp` and this row reports the oracle may be alive again,
    // and it will NOT be, because the shared stub tree is still gone. The row
    // is right that its reason expired; whoever reads the failure needs this
    // sentence to avoid concluding the script runs.
    // This row is STILL TRUE and its conclusion is STILL WRONG, which is the
    // exact inverse of the `lora_stage` note above and the reason both are
    // written down. The path really is gone from the tree, so the assertion
    // holds -- but this oracle is not dead. `memory_planner/run.sh` now
    // restores the `.cpp`/`.hpp` from `e7cd33cf1` and builds them against the
    // surviving `stub/` tree with plain `g++` and no CUDA, and it reproduces
    // `GOLDEN_FNV1A64` exactly. Nothing about it needed the deleted tree.
    //
    // `memory_planner_parity.rs` spent months red on a real divergence while
    // saying in its own failure message that the C++ could not be re-run to
    // diff against. It could. That claim is what made the divergence look
    // undiagnosable, so the cost of a row like this is not the row -- it is
    // the reader who believes it and stops.
    //
    // Before adding a `Dies::Cp` row, check whether the input is one
    // `git show` away, as this one was.
    ("memory_planner", Dies::GitRestores, "crates/driver-cuda/csrc/src/store/memory_planner.cpp",
     "the memory planner's allocation decisions"),
    ("page_mask", Dies::Cp, "crates/driver-cuda/csrc/src/model/attn_page_mask.cu",
     "page-mask construction"),
    ("profile_cache", Dies::Cp, "crates/driver-cuda/csrc/src/store/planner_profile_cache.cpp",
     "the planner's profile cache"),
    ("recurrent", Dies::Cp, "crates/driver-cuda/csrc/src/store/recurrent_state_cache.cpp",
     "recurrent state cache geometry"),

    ("sideband_arena", Dies::Cp, "crates/driver-cuda/csrc/src/model/hook_sideband_arena.cpp",
     "the sideband arena's bump allocation"),
    ("stage_hooks", Dies::Cp, "crates/driver-cuda/csrc/src/model/stage_hooks.hpp",
     "stage hook firing order"),
    ("workspace", Dies::Cp, "crates/driver-cuda/csrc/src/model/workspace.cpp",
     "the model workspace's section offsets"),
    // The fourteenth of cause A, and the one that hides: its `cp` list is
    // clean and it dies at `g++`, so it is reached only by looking at
    // compiler inputs.
    ("store", Dies::Compile, "crates/driver-cuda/csrc/src/store/kv_cache_format.cpp",
     "the on-disk KV cache format"),
    // ── Cause B: individual archive files retired by this migration.
    //
    // Two, and the kind that will keep happening: an oracle whose subject is a
    // HOST source in the kernels tree dies the day that source is ported.
    //
    // Their path strings are the ARCHIVE crate's — `kernels-cuda` when it was
    // a CMake+nvcc crate — and are kept verbatim because they are what each
    // `run.sh` reaches for. Nothing in this tree holds either path now.
    ("cublas_handle", Dies::Cp, "crates/kernels-cuda/csrc/src/gemm/gemm.cpp",
     "cuBLAS handle and workspace lifetime"),
    ("weight_view", Dies::Compile, "crates/kernels-cuda/csrc/src/tensor.cpp",
     "weight view strides over a quantised tensor"),
];

/// The three that were DELETED rather than kept as descriptions, and why the
/// "read but not re-derived" policy did not reach them.
///
/// # The policy needs a golden and there was not one
///
/// **These three produced none** — no `GOLDEN_*` constant, no captured file,
/// and no `.rs` in the workspace mentioning them except this census. There is
/// no record for a description to describe.
///
/// # And there was nothing inside them either
///
/// **None of the three reimplements anything.** Each `#include`s and compiles
/// the REAL driver source and drives it, replacing only the flashinfer
/// plan-cache entry points or `allocate_device_memory` with recorders, so the
/// arithmetic lived in the subject rather than the `oracle.cpp`. Deleting
/// `crates/driver-cuda/csrc` took the derivation with it.
///
/// # No absence gate, deliberately
///
/// An absence assertion over a deleted path is a green check that can never
/// fail again. [`every_oracle_is_classified`]'s WALK covers them instead:
/// recreate any of the three with a `run.sh` and it is unclassified. What the
/// walk does not see is a directory with no `run.sh` — a bare
/// `llama_like_prepare/stub/`, which `lora_stage/run.sh` still copies and
/// which is recorded at the `lora_stage` row above.
const RETIRED: &[(&str, &str)] = &[
    ("llama_like_cfg", "no golden; drove the deleted llama_like.cpp"),
    ("llama_like_prepare", "no golden; drove the deleted llama_like.cpp"),
    ("qwen35_la_ws", "no golden; drove the deleted qwen3_5_forward.cpp"),
];

/// The two that can still be run, and neither can be run *here*.
///
/// They fall outside the "read but not re-derived" policy in opposite
/// directions, which is why the policy covers sixteen and not eighteen:
///
/// - `dtoa` is **runnable anywhere**: it compiles against a vendored
///   `nlohmann/json.hpp` and touches no repo C++, so its golden is
///   re-derivable and its parity test's instruction to run it is true.
/// - `gemm_service` is **runnable only on a CUDA machine** (`$CUDA_HOME/bin/
///   nvcc`), and is the only oracle holding captured output on disk rather
///   than a transcribed constant, which makes that re-derivation optional.
const ALIVE: &[(&str, &str)] = &[
    ("dtoa", "vendored `nlohmann/json.hpp` only — reaches no repo C++"),
    ("gemm_service", "needs `$CUDA_HOME/bin/nvcc`; a CUDA host, not this one"),
];

/// Directories with no `run.sh`, which are therefore not oracles at all.
///
/// `launch_abi/` holds headers `tests/launch_abi.rs` reads from disk and
/// `real_decode/` holds captured JSON four `real_*.rs` tests read. Neither has
/// ever had a script, so neither can have stopped working; listing them stops
/// a future census from "discovering" them.
const NO_SCRIPT: &[&str] = &["launch_abi", "real_decode"];

fn dirs_with_script() -> Vec<String> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(oracle_dir()).expect("tests/oracle is readable") {
        let entry = entry.expect("a readable dir entry");
        if !entry.file_type().expect("a stat-able entry").is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if entry.path().join("run.sh").is_file() {
            out.push(name);
        }
    }
    out.sort();
    out
}

/// Every oracle directory is in exactly one of the three tables.
///
/// This is the half that catches an oracle ARRIVING: a new directory with a
/// `run.sh` is unclassified until someone says whether it runs.
#[test]
fn every_oracle_is_classified() {
    let on_disk = dirs_with_script();

    let mut classified: Vec<String> = DEAD
        .iter()
        .map(|(n, ..)| (*n).to_owned())
        .chain(ALIVE.iter().map(|(n, _)| (*n).to_owned()))
        .collect();
    classified.sort();

    let mut dup = classified.clone();
    dup.dedup();
    assert_eq!(
        dup.len(),
        classified.len(),
        "an oracle is listed twice across DEAD and ALIVE: {classified:?}"
    );

    let unclassified: Vec<&String> =
        on_disk.iter().filter(|n| !classified.contains(n)).collect();
    assert!(
        unclassified.is_empty(),
        "tests/oracle/{unclassified:?} has a `run.sh` and no row here.\n  \
         Read the script and say whether it can run: if its `cp` and compiler \
         inputs all exist it goes in ALIVE, and if any is gone it goes in DEAD \
         with that path. An oracle nobody classified reads as live \
         infrastructure, which is the whole failure this file exists for."
    );

    let vanished: Vec<&String> =
        classified.iter().filter(|n| !on_disk.contains(n)).collect();
    assert!(
        vanished.is_empty(),
        "{vanished:?} is classified here and has no `tests/oracle/<name>/run.sh`.\n  \
         Either the directory was deleted — drop the row, its parity test's \
         status line is now the only record and should say so — or the script \
         was renamed, in which case this row is pointing at nothing."
    );

    for name in NO_SCRIPT {
        let dir = oracle_dir().join(name);
        assert!(
            dir.is_dir(),
            "tests/oracle/{name}/ is listed as a data directory and is gone"
        );
        assert!(
            !dir.join("run.sh").is_file(),
            "tests/oracle/{name}/ has grown a `run.sh`.\n  \
             It was listed as a data directory precisely because it had none. \
             Classify it in DEAD or ALIVE and remove it from NO_SCRIPT."
        );
    }

    assert_eq!(
        DEAD.len(),
        16,
        "the count in this file's header is part of the finding; update both"
    );
    assert_eq!(on_disk.len(), DEAD.len() + ALIVE.len());

    // A RETIRED oracle may be classified again only if its directory is back.
    // `vanished` above catches a row without a directory; this catches the
    // same thing one step earlier and names the specific reason.
    for (name, why) in RETIRED {
        assert!(
            !classified.iter().any(|c| c.as_str() == *name)
                || on_disk.iter().any(|d| d.as_str() == *name),
            "`{name}` is classified here and its directory is gone: it was \
             RETIRED — {why}.\n  If it is genuinely back, restore the \
             directory and drop it from RETIRED. If the row was added from \
             this file's history, the oracle it describes does not exist."
        );
    }
}

/// Every dead oracle's named input is still missing.
///
/// This is the half that retires the row. The claim is not "this script
/// fails" — that would need a shell — but "the input it dies on is not
/// there". Restore the file and this fires, which is correct.
#[test]
fn the_dead_oracles_inputs_are_still_missing() {
    let root = repo_root();
    for (name, dies, missing, subject) in DEAD {
        let path = root.join(missing);
        let phase = match dies {
            Dies::Cp => "the `cp` that `set -euo pipefail` stops on",
            Dies::Compile => "the compiler, which is handed it as a positional input",
            // Restoring rows assert the same absence for the same reason: a
            // file reappearing in the tree changes which source the oracle
            // builds, and that has to be seen. They just do not die of it.
            Dies::GitRestores => "nothing -- the script restores it from git",
        };
        assert!(
            !path.exists(),
            "tests/oracle/{name}/run.sh is recorded dead because `{missing}` is \
             gone, and it is BACK.\n  It dies at {phase}. If the file was \
             restored deliberately, re-read the script — the rest of its \
             inputs may still be missing, in which case this row wants the new \
             first one, and if they are all present the oracle covering \
             {subject} is alive again and belongs in ALIVE."
        );
    }
}

/// Every live oracle's `run.sh` still reaches only files that exist.
///
/// The cheap, exact version: neither mentions `driver-cuda/csrc`, the tree
/// that was deleted wholesale. Not a proof that they run — a proof that they
/// have not silently joined cause A.
#[test]
fn the_alive_oracles_still_have_their_inputs() {
    for (name, why) in ALIVE {
        let script = oracle_dir().join(name).join("run.sh");
        let text = std::fs::read_to_string(&script)
            .unwrap_or_else(|e| panic!("tests/oracle/{name}/run.sh: {e}"));
        assert!(
            !text.contains("driver-cuda/csrc"),
            "tests/oracle/{name}/run.sh now reaches into `driver-cuda/csrc`, \
             which `4569b9e4b` deleted.\n  It is listed ALIVE because {why}. \
             Either the reach is new and the script is broken, or this row is \
             wrong; either way it is not alive as described."
        );
    }
}

/// No dead oracle's parity test tells its reader to run it.
///
/// Twenty did, in two places, and the worse of the two is inside an
/// `assert_eq!` message: a header is read once, but a panic message is read at
/// the exact moment someone has found a mismatch, is establishing which side
/// moved, and is least inclined to doubt the infrastructure. The corrected
/// messages say the C++ side cannot be re-run, so the golden is the only
/// record of it and a divergence is this crate changing.
///
/// One-directional on purpose: it forbids the imperative in a dead oracle's
/// test and requires no particular replacement wording, because prose a test
/// dictates is prose nobody edits. `dtoa` keeps its instruction and must.
#[test]
fn no_dead_oracle_is_advertised_as_runnable() {
    let tests = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let sources: Vec<(String, String)> = std::fs::read_dir(&tests)
        .expect("tests/ is readable")
        .filter_map(|e| {
            let path = e.ok()?.path();
            if path.extension()? != "rs" || path.file_name()? == "oracle_census.rs" {
                return None;
            }
            let name = path.file_name()?.to_string_lossy().into_owned();
            Some((name, std::fs::read_to_string(&path).ok()?))
        })
        .collect();
    assert!(sources.len() > 20, "tests/ scan found only {} files", sources.len());

    for (name, dies, ..) in DEAD {
        // A restoring oracle IS runnable, so telling a reader to run it is
        // true and this ban does not apply. The variant is what earns the
        // exemption, and it is checked -- see the enum.
        if *dies == Dies::GitRestores {
            continue;
        }
        // The imperative in every form it was found in. Matching the path
        // alone would forbid naming the script at all, and naming it is the
        // point — it is kept as the description of how the golden was taken.
        for verb in ["Run `tests/oracle/", "run tests/oracle/", "Regenerate with `tests/oracle/"] {
            let needle = format!("{verb}{name}/run.sh");
            for (file, text) in &sources {
                assert!(
                    !text.contains(&needle),
                    "{file} says \"{needle}\", and that script has not run \
                     since its inputs were deleted.\n  Say what the golden is \
                     instead: a record read but not re-derived, with the \
                     script kept as the description of how it was taken. \
                     `cublas_handle_parity.rs` is the wording, and if this is \
                     an `assert_eq!` message say that a divergence is this \
                     crate changing — there is no other side left to blame."
                );
            }
        }
    }
}

/// The revision a [`Dies::GitRestores`] script fetches its source from.
///
/// One constant because one oracle restores today. It is here rather than
/// only in the script so the check below and the script cannot disagree
/// silently -- the check reads the script and requires this string in it.
const RESTORE_REV: &str = "e7cd33cf1";

/// A restoring oracle's source is really still fetchable.
///
/// The point of the variant is that "deleted" and "unavailable" are not the
/// same word, and the whole reason it exists is that this repo spent months
/// treating them as one. So the claim has to be checked, or it is just a
/// nicer-sounding version of the assertion it replaced.
///
/// Two things are required, because either alone is satisfiable while the
/// oracle stays broken: the script must actually contain the restore, and git
/// must actually still have the blob. The second is the one that decays --
/// history is rewritten, branches are pruned -- and it decays silently.
#[test]
fn a_restoring_oracle_can_actually_restore() {
    let root = repo_root();
    let restoring: Vec<_> = DEAD
        .iter()
        .filter(|(_, d, ..)| *d == Dies::GitRestores)
        .collect();
    assert!(
        !restoring.is_empty(),
        "no row uses Dies::GitRestores, so this test measures nothing. If the \
         last restoring oracle was reclassified, delete this test and \
         RESTORE_REV with it rather than leaving a green check over an empty \
         loop."
    );

    for (name, _, path, subject) in restoring {
        let script = root.join(format!("crates/driver-cuda/tests/oracle/{name}/run.sh"));
        let text = std::fs::read_to_string(&script)
            .unwrap_or_else(|e| panic!("read {}: {e}", script.display()));
        assert!(
            text.contains(RESTORE_REV),
            "tests/oracle/{name}/run.sh is recorded as restoring its source \
             from git, and it does not name {RESTORE_REV}. Either the script \
             pins a different revision -- in which case this constant is \
             stale and the check below is testing the wrong blob -- or it no \
             longer restores anything and the row belongs back in Dies::Cp."
        );

        let object = format!("{RESTORE_REV}:{path}");
        let found = std::process::Command::new("git")
            .arg("-C")
            .arg(&root)
            .args(["cat-file", "-e"])
            .arg(&object)
            .status();
        match found {
            Ok(status) => assert!(
                status.success(),
                "git cannot produce `{object}`, so tests/oracle/{name}/run.sh \
                 cannot rebuild the C++ and the oracle covering {subject} is \
                 dead after all -- not by anyone's decision, which is why \
                 this is checked.\n  If history was rewritten, find the \
                 1,221-line memory_planner.cpp on the new history and repoint \
                 RESTORE_REV and the script together. If it is unrecoverable, \
                 move the row to Dies::Cp and say so in \
                 memory_planner_parity.rs, which currently tells its reader \
                 to go re-run this."
            ),
            // A machine without git is not a machine that has this repo, but
            // failing to LAUNCH git is not evidence about the blob and must
            // not be reported as if it were.
            Err(e) => panic!("could not run git to check `{object}`: {e}"),
        }
    }
}
