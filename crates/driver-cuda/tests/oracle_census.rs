//! Which oracles can still be run, and which are descriptions of a run that
//! can never happen again.
//!
//! # The failure this exists for
//!
//! `tests/oracle/` holds twenty directories. Each one is a C++ program
//! that was compiled against the real driver sources, run once, and its output
//! transcribed into a `GOLDEN_FNV1A64` in the matching `*_parity.rs`. The
//! goldens are not files in these directories — they are constants in the Rust
//! tests, and only `gemm_service/` and `real_decode/` keep captured data on
//! disk at all. What the directories hold is the *instrument*: `oracle.cpp`
//! and a `run.sh` that copies the driver sources next to it and compiles them.
//!
//! **Sixteen of the eighteen `run.sh` cannot run.** Not "fail on this
//! machine for want of a GPU" — cannot run anywhere, because the sources they
//! copy were deleted. `4569b9e4b` removed `crates/driver-cuda/csrc` wholesale
//! and this migration removed individual archive files, and every one of those
//! scripts opens `set -euo pipefail` and then `cp` a path that is gone. They
//! die on line ten of forty, before the compiler is ever reached.
//!
//! That is worse than a deleted test. A deleted test is absent; these read as
//! live infrastructure. Twelve of the parity tests carried the instruction
//! *"Run `tests/oracle/X/run.sh` to regenerate the golden"* — an instruction
//! that has been false since a commit nobody connected to them, addressed to
//! whoever next needs the golden and finds out the hard way. This file is the
//! census that makes the state checkable, and the corrected wording in those
//! twelve is what it enforces.
//!
//! # What this asserts, and what it deliberately does not
//!
//! It asserts **path existence**, nothing else. For each dead oracle it names
//! the first input its `run.sh` reaches for and asserts that input is *still
//! missing*. It does not parse shell, does not model `set -e`, does not try to
//! predict whether a compiler would succeed given its inputs. A gate whose
//! subject is approximately right reports failures that are approximately
//! real; this one's subject is a filename.
//!
//! It is therefore **self-retiring in both directions**. Restore
//! `driver-cuda/csrc/src/model/workspace.cpp` and the `workspace` entry fails,
//! saying the oracle may be alive again and the row is stale. Delete an input
//! an ALIVE oracle needs and [`the_alive_oracles_still_have_their_inputs`]
//! fails. Add a twenty-first oracle directory and
//! [`every_oracle_is_classified`] fails until it is classified. None of the
//! three can pass by neglect.
//!
//! # The policy these sixteen fall under
//!
//! `cublas_handle_parity.rs` states it and it was written for exactly this:
//! a golden is "a permanent record of behaviour that can be **read but not
//! re-derived**", and `run.sh` is kept "as the description of how it was taken
//! rather than as a command anyone can issue". The sixteen are all covered by
//! it, because in every case the *behaviour* the golden records was the
//! archive's and the archive is what left. Nothing was lost by the oracle
//! dying that was not already lost by its subject dying.
//!
//! **The policy's precondition is a golden, and three oracles had none.**
//! `llama_like_cfg`, `llama_like_prepare` and `qwen35_la_ws` were deleted
//! rather than left as descriptions, and the argument is in [`RETIRED`]. It
//! is the reason the count moved from nineteen to sixteen and from
//! twenty-three directories to twenty.
//!
//! The two that sit outside the policy are the two ALIVE ones, and they sit
//! outside it in opposite directions — see [`ALIVE`].

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
/// `at` is the phase the script dies in. The two values are not decoration:
/// they distinguish a script that never reaches a compiler from one that does,
/// and the second kind is the one a naive census misses. `store` and
/// `weight_view` copy only files that still exist and then hand the compiler a
/// path that does not, so a check that looked at `cp` arguments alone called
/// them alive. They are not.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Dies {
    /// Dies at a `cp` under `set -euo pipefail`, before any compiler runs.
    Cp,
    /// Dies at the compiler, on a positional input that is not there.
    Compile,
}

/// The sixteen, each with the first input its `run.sh` cannot find.
///
/// The path is repo-relative and is asserted MISSING. Where a script reaches
/// for several absent inputs only the first is listed, because the first is
/// the one `set -e` stops on and listing the rest would make this table decay
/// every time an unrelated file moved.
const DEAD: &[(&str, Dies, &str, &str)] = &[
    // ── Cause A: `crates/driver-cuda/csrc` deleted wholesale by `4569b9e4b`.
    //
    // Fourteen of the sixteen. These oracles compiled the driver's own C++
    // host layer — the workspace planner, the KV cache, the stage hooks — and
    // that layer is now Rust. The golden each one produced is a record of what
    // the C++ did, kept so the Rust can be compared against it by reading.
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
    ("memory_planner", Dies::Cp, "crates/driver-cuda/csrc/src/store/memory_planner.cpp",
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
    // The fourteenth of cause A, and the one that hides. Its `cp` list is
    // clean — every file it copies is present — and it dies at `g++`, which is
    // why it must be reached by looking at compiler inputs and not just `cp`.
    ("store", Dies::Compile, "crates/driver-cuda/csrc/src/store/kv_cache_format.cpp",
     "the on-disk KV cache format"),
    // ── Cause B: individual archive files retired by this migration.
    //
    // Two, and they are the ones that will keep happening: an oracle whose
    // subject is a HOST source in the kernels tree dies the day that source
    // is ported. Both of these were live until this migration reached their
    // file.
    //
    // Their two path strings are the ARCHIVE crate's — `kernels-cuda` when it
    // was a CMake+nvcc crate whose `csrc/` held `.cu` and host `.hpp`/`.cpp`,
    // deleted whole at `85c6c674b`. The strings are kept verbatim because
    // they are what each `run.sh` reaches for. Nothing in this tree holds
    // either path now and nothing is meant to, so the rows still assert what
    // they were written to assert; a reader must not take the prefix for a
    // crate that is alive.
    ("cublas_handle", Dies::Cp, "crates/kernels-cuda/csrc/src/gemm/gemm.cpp",
     "cuBLAS handle and workspace lifetime"),
    ("weight_view", Dies::Compile, "crates/kernels-cuda/csrc/src/tensor.cpp",
     "weight view strides over a quantised tensor"),
];

/// The three that were DELETED rather than kept as descriptions, and why the
/// "read but not re-derived" policy did not reach them.
///
/// ```text
///   llama_like_cfg        model/llama_like/llama_like.cpp     no golden
///   llama_like_prepare    model/llama_like/llama_like.cpp     no golden
///   qwen35_la_ws          model/qwen3_5/qwen3_5_forward.cpp   no golden
/// ```
///
/// # The policy needs a golden and there was not one
///
/// `cublas_handle_parity.rs` states it: a golden is "a permanent record of
/// behaviour that can be **read but not re-derived**", and the `run.sh` is
/// kept "as the description of how it was taken". Both halves name the
/// golden. **These three produced none** — no `GOLDEN_*` constant in any
/// `.rs`, no captured file on disk, and no `.rs` in the workspace mentioned
/// them at all except this census. The instrument was built, and either it
/// was never run or its reading was never written down. There is no record
/// for a description to describe.
///
/// # And there was nothing inside them either
///
/// The reason that is decisive rather than merely tidy: **none of the three
/// reimplements anything.** Each one `#include`s and compiles the REAL
/// driver source — `llama_like.cpp` at 3.2k lines, `qwen3_5_forward.cpp` at
/// 2.4k — and drives it, replacing only the flashinfer plan-cache entry
/// points or `allocate_device_memory` with recorders. The arithmetic they
/// were pointed at lived in the subject, not in the `oracle.cpp`. So when
/// `4569b9e4b` deleted `crates/driver-cuda/csrc` wholesale it took the
/// derivation with it, and what remained was 41 KB of driver code for a
/// subject that is gone, holding no formula of its own.
///
/// An instrument whose subject is deleted and whose reading was never
/// recorded holds nothing. That is the discriminator, and it is why these
/// three separate cleanly from the sixteen in [`DEAD`] — every one of those
/// has a golden that is still read.
///
/// # No absence gate, deliberately
///
/// There is no test here asserting these three directories stay gone. An
/// absence assertion over a deleted path is a green check that can never
/// fail again. What covers them is [`every_oracle_is_classified`]'s WALK:
/// recreate any of the three with a `run.sh` and it is unclassified, and the
/// walk sees directories that do not exist yet, which an absence assertion
/// cannot.
///
/// The one thing the walk does not see is a directory with no `run.sh` — a
/// bare `llama_like_prepare/stub/` recreated because `lora_stage/run.sh:29`
/// still copies it. That is recorded at the `lora_stage` row above and in
/// the script itself, which is where the person who would recreate it looks.
const RETIRED: &[(&str, &str)] = &[
    ("llama_like_cfg", "no golden; drove the deleted llama_like.cpp"),
    ("llama_like_prepare", "no golden; drove the deleted llama_like.cpp"),
    ("qwen35_la_ws", "no golden; drove the deleted qwen3_5_forward.cpp"),
];

/// The two that can still be run, and neither can be run *here*.
///
/// They fall outside the "read but not re-derived" policy in opposite
/// directions, which is why the policy sentence covers the sixteen and not
/// the eighteen:
///
/// - `dtoa` is **runnable anywhere**. It compiles `oracle.cpp` against a
///   vendored `nlohmann/json.hpp` and touches no repo C++ at all, so nothing
///   this migration does can break it. Its golden is re-derivable and its
///   parity test's instruction to run it is true — the one of the thirteen
///   that was left alone.
/// - `gemm_service` is **runnable only on a CUDA machine**: its `run.sh`
///   invokes `$CUDA_HOME/bin/nvcc` twice. Its golden is re-derivable in
///   principle and not on this host, and it is the only oracle holding
///   captured output on disk (`golden.txt`, `bias_fold.txt`) rather than a
///   transcribed constant — which is what makes that re-derivation optional.
const ALIVE: &[(&str, &str)] = &[
    ("dtoa", "vendored `nlohmann/json.hpp` only — reaches no repo C++"),
    ("gemm_service", "needs `$CUDA_HOME/bin/nvcc`; a CUDA host, not this one"),
];

/// Directories with no `run.sh`, which are therefore not oracles in this
/// sense at all.
///
/// `launch_abi/` holds headers that `tests/launch_abi.rs` reads from disk, and
/// `real_decode/` holds captured JSON that four `real_*.rs` tests read. Both
/// are data directories that happen to live under `tests/oracle/`. Neither has
/// ever had a script, so neither can have stopped working, and listing them
/// here is what stops a future census from "discovering" them.
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
/// This is the half that catches an oracle ARRIVING. A new directory with a
/// `run.sh` is unclassified until someone says whether it runs, and an entry
/// whose directory was deleted is a row describing nothing.
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
    // A row without a directory is what `vanished` above catches; this catches
    // the same thing one step earlier and says the specific reason the name is
    // absent, which `vanished`'s "renamed or deleted" wording cannot.
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
/// there", which is a filename and is exactly as true as the claim it stands
/// for. Restore the file and this fires, which is correct: the row's reason
/// expired and the oracle may be runnable again.
#[test]
fn the_dead_oracles_inputs_are_still_missing() {
    let root = repo_root();
    for (name, dies, missing, subject) in DEAD {
        let path = root.join(missing);
        let phase = match dies {
            Dies::Cp => "the `cp` that `set -euo pipefail` stops on",
            Dies::Compile => "the compiler, which is handed it as a positional input",
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
/// The cheap, exact version of that claim: neither of the two mentions
/// `driver-cuda/csrc`, the tree that was deleted wholesale. This is not a
/// proof that they run — `gemm_service` needs a GPU and this asserts nothing
/// about one — it is a proof that they have not silently joined cause A.
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
/// Twenty did, in two places. Twelve said *"Run `tests/oracle/X/run.sh` to
/// regenerate"* in the module header, and eight more said *"run
/// tests/oracle/X/run.sh with X_ORACLE_OUT set to diff them"* inside the
/// `assert_eq!` message — and the second location is much the worse of the
/// two. A header is read once; a panic message is read at the exact moment
/// someone has found a mismatch, is trying to establish which side moved, and
/// is least inclined to doubt the infrastructure. Sending that reader to a
/// script that dies at its first `cp` costs them the hour before they think to
/// check whether the oracle still exists.
///
/// The corrected messages say the thing the reader actually needs: the C++
/// side cannot be re-run, so the golden is the only record of it, and a
/// divergence is therefore this crate changing.
///
/// The rule is one-directional on purpose. It forbids the imperative in a dead
/// oracle's test; it does not require any particular replacement wording,
/// because prose that a test dictates is prose nobody edits. `dtoa` keeps its
/// instruction and must — it is true.
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

    for (name, ..) in DEAD {
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
