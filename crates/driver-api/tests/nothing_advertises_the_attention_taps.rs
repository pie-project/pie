//! No driver in this repository advertises the attention taps, and the Python
//! harness is written on that assumption.
//!
//! `PtirCaps` (`src/capabilities.rs`) declares three fields for the
//! observe-and-evict family:
//!
//! * `has_kv_envelopes` -- per-page KV key envelopes and the `envelope_dot`
//!   second-party kernel (Quest).
//! * `has_attn_score` -- the `AttnScore` intrinsic at an `OnAttn` tap (H2O,
//!   TOVA, SnapKV).
//! * `has_attn_page_mask` -- honouring an `attn_page_mask` sink, which is
//!   ENFORCEMENT and independent of observing.
//!
//! Every driver that fills a `PtirCaps` in states all three as a literal
//! `false`, with no condition on the checkpoint, the device or the
//! environment. There are five such sites and they are listed in `SITES`.
//!
//! # Why a test says so, instead of a comment
//!
//! Eight scripts in `tests/inferlets` exist for these three capabilities:
//! `test_axis_smoke.py`, `test_chunked_prefill.py`, `test_h2o.py`,
//! `test_mask_enforced.py`, `test_quest_pages.py`, `test_snapkv.py`,
//! `bench_quest.py` and `bench_trackb.py`. Measured on a 4090 against
//! qwen-3-0.6b on `cuda_native`, the six test scripts scored 0 of 18 -- every
//! case refused at bind, none of them reached a kernel, and each spent a model
//! boot to say so.
//!
//! They now skip up front, on a `requires=` declaration read against
//! `conftest.UNADVERTISED`. That declaration is a claim about THIS side of the
//! repository, made in a language that cannot see it, and a skip that outlives
//! its reason is the worst outcome available: the day someone finishes the
//! feature, eight suites would quietly go on not running, and the regression
//! floor they were written to be would be gone without a single test turning
//! red.
//!
//! So this gate holds the other end. Flip any one of the fifteen literals and
//! it fails, naming the file that has to change. It is the only way the Python
//! finds out.
//!
//! # This is not an argument for keeping them false
//!
//! Half the floor is built: envelope MAINTENANCE runs in
//! `kernels-cuda/src/attn/mod.rs`, where the write-KV kernels call
//! `envelope_merge_written` and `envelope_update_appended` behind the
//! `KvHasEnvelopes` ask, and `envelope_dot` itself is a second-party region the
//! tensor compiler generates rather than a hand-written kernel. What is missing
//! is the rest of it and the honest bit at the door. Finish the work, flip the
//! literal, delete the `requires=` this gate points at, and delete the entry
//! here. That order.

use std::path::{Path, PathBuf};

/// Where a `PtirCaps` is built, and the three fields each one must still be
/// stating as `false`.
///
/// Paths are relative to the repository root. Every one was verified by reading
/// the file rather than by matching a directory: the two engine-side adapters
/// are the in-process shells for `vulkan` and `wgpu`, while the three
/// `driver-*` entries are what the out-of-process ABI reports.
const SITES: &[&str] = &[
    "crates/driver-cuda/src/serve/load.rs",
    "crates/driver-vulkan/src/shell.rs",
    "crates/driver-metal/src/serve/load.rs",
    "crates/engine/src/driver/backend/vulkan.rs",
    "crates/engine/src/driver/backend/wgpu.rs",
];

/// The three fields, and the harness name each one gates.
const FIELDS: &[(&str, &str)] = &[
    ("has_kv_envelopes", "envelope_dot"),
    ("has_attn_score", "attn_score"),
    ("has_attn_page_mask", "attn_page_mask"),
];

/// The repository root: this crate's manifest directory is `crates/driver-api`.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/driver-api has two ancestors")
        .to_path_buf()
}

/// Every site still states every field as a literal `false`.
#[test]
fn nothing_advertises_the_attention_taps() {
    let root = repo_root();
    let mut advertised = Vec::new();
    for site in SITES {
        let path = root.join(site);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "{site}: {e}. This gate names its sites by path, so a moved \
                 file has to be renamed here -- silently scanning fewer files \
                 would let it pass while advertising nothing"
            )
        });
        for (field, name) in FIELDS {
            if !text.contains(&format!("{field}: false")) {
                advertised.push(format!("{site}: `{field}` (gates `{name}`)"));
            }
        }
    }
    advertised.sort();
    assert!(
        advertised.is_empty(),
        "these no longer state `false`:\n  {}\n\nIf the capability is real \
         now, that is the point of the work and this gate is doing its only \
         job: `tests/inferlets/conftest.py` skips eight suites on \
         `UNADVERTISED`, and they will go on skipping until the entry is \
         removed. Delete the matching key there and the `requires=` argument \
         from each script that names it, then delete the field from `FIELDS` \
         here. If instead the field was merely renamed or reformatted, fix \
         this gate -- it matches source text, which is the only way to ask \
         five crates a question none of them export.",
        advertised.join("\n  ")
    );
}

/// The harness still declares what this gate is holding open for it.
///
/// Without this, deleting a `requires=` would leave the suite booting a model
/// to fail every case at bind again -- the state this replaced -- and the gate
/// above would keep passing, because it only ever looks at Rust.
#[test]
fn the_harness_still_declares_what_it_needs() {
    let root = repo_root();
    let conftest = root.join("tests/inferlets/conftest.py");
    let text = std::fs::read_to_string(&conftest)
        .unwrap_or_else(|e| panic!("tests/inferlets/conftest.py: {e}"));
    for (field, name) in FIELDS {
        assert!(
            text.contains(&format!("\"{name}\": \"{field}\"")),
            "conftest.py's UNADVERTISED has no `{name}` -> `{field}` entry, \
             but every driver still states `{field}: false`. Either the entry \
             was deleted ahead of the capability -- in which case the suites \
             that bind `{name}` are booting a model to fail at bind again -- \
             or the mapping was reformatted and this gate has to follow it."
        );
    }
    let suites = [
        "test_axis_smoke.py",
        "test_chunked_prefill.py",
        "test_h2o.py",
        "test_mask_enforced.py",
        "test_quest_pages.py",
        "test_snapkv.py",
        "bench_quest.py",
        "bench_trackb.py",
    ];
    let mut undeclared = Vec::new();
    for suite in suites {
        let path = root.join("tests/inferlets").join(suite);
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        if !text.contains("requires=") {
            undeclared.push(suite);
        }
    }
    assert!(
        undeclared.is_empty(),
        "these suites bind an unadvertised capability and no longer declare \
         it: {undeclared:?}. Each measured 0 passing on `cuda_native`, every \
         case refused at bind; without the declaration they spend a model boot \
         to reprint that."
    );
}
