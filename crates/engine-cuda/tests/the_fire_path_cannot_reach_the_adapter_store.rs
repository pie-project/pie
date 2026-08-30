//! **A-5'S STRUCTURAL HALF**: a fire cannot reach the shared-adapter store
//! (alto adapter §6.1).
//!
//! # What A-5 asks and what this can answer
//!
//! The campaign's A-5 gate is "a fire with no adapter lane is byte- and
//! launch-count-identical to pre-campaign; the sink's channels are never read
//! at fire time". The byte identity is a DEVICE claim and belongs in the GPU
//! session. This is the other half, and it is the half that catches the
//! regression early: the store is unreachable from the fire path by
//! CONSTRUCTION, so no future edit can quietly put a lookup on it.
//!
//! §6.1's ruling is exactly this shape. The reason the blob store was
//! promoted out of wave 3 is that per-fire channel materialization re-paid a
//! 12 MiB payload every token — `CHAN_READ` into per-lane scratch,
//! `pull_validate` dragging the mirror over PCIe. The fix is that the bytes
//! land ONCE at bind. A grep is what says the fix is still in place, for the
//! same reason `no_shell_reads_the_environment.rs` is a grep: the failure is
//! about PROVENANCE — which call site reached which state — and provenance is
//! what a scan can see.
//!
//! ```text
//! cargo test -p engine-cuda --test the_fire_path_cannot_reach_the_adapter_store
//! ```

use std::path::{Path, PathBuf};

/// The modules a fire is made of — everything the walk, the record, the
/// windows, the pools and the guest plane run through. **NONE of them may
/// name the store.**
const FIRE_PATH: [&str; 13] = [
    "run.rs",
    "record.rs",
    "window.rs",
    "inputs.rs",
    "mask.rs",
    "dispatch.rs",
    "store.rs",
    "arena.rs",
    "exports.rs",
    "settle.rs",
    "experts.rs",
    "program.rs",
    "weights.rs",
];

/// The files that may reach into the store at all, `blob.rs` aside: the shell
/// that owns the verbs, and nothing else. (`lib.rs` declares the module and
/// `api.rs` states the mount through `Shell::mount_adapters`; neither reaches
/// a type inside it.)
///
/// **THIS LIST IS THE CLAIM.** A file joining it is a file that can reach
/// adapter residency, and the question to ask of it is when it runs — between
/// fires, or during one.
const NAMERS: [&str; 1] = ["serve.rs"];

/// The only methods of `Shell` that may touch the store. Every one runs
/// BETWEEN fires, on the host, exactly as `register_adapter` does.
const VERBS: [&str; 4] = [
    "mount_adapters",
    "bind_adapter",
    "release_adapter",
    "adapters",
];

/// **CODE, NOT PROSE.** Every module in this crate explains itself at length
/// and several of them name [`crate::blob`] in a doc link — `weights.rs` says
/// what the resolver needs from a bank, which is exactly the sentence a
/// reader wants there. A gate that counted those would be a gate against
/// documentation. So the scan drops line comments, which is every comment
/// this crate writes.
fn code(text: &str) -> impl Iterator<Item = (usize, &str)> {
    text.lines()
        .enumerate()
        .filter(|(_, said)| !said.trim_start().starts_with("//"))
}

fn src() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Every `.rs` under a directory, recursively.
fn rust_files(at: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![at.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// **(1)** No module a fire runs through names the store.
#[test]
fn no_module_the_fire_runs_through_names_the_store() {
    let src = src();
    for module in FIRE_PATH {
        let at = src.join(module);
        assert!(at.is_file(), "{at:?} is not a file; this gate scans it");
        let text = std::fs::read_to_string(&at).expect("a readable module");
        for (line, said) in code(&text) {
            assert!(
                !said.contains("crate::blob") && !said.contains("self.adapters"),
                "{module}:{} reaches the shared-adapter store, and a fire runs \
                 through this module — alto adapter §6.1 lands an adapter's bytes \
                 ONCE at bind, and a lookup here is the per-fire cost the ruling \
                 exists to refuse: {said}",
                line + 1
            );
        }
    }
    // The submodule directories too — `store/`, `dispatch/`, `program/`,
    // `weights/`: the fire path is not only the files that name it.
    for dir in ["store", "dispatch", "program", "weights", "device"] {
        let at = src.join(dir);
        if !at.is_dir() {
            continue;
        }
        for file in rust_files(&at) {
            let text = std::fs::read_to_string(&file).expect("a readable module");
            assert!(
                !code(&text).any(|(_, said)| said.contains("crate::blob")),
                "{} reaches the shared-adapter store from inside the fire path",
                file.display()
            );
        }
    }
}

/// **(2)** Exactly one file besides `blob.rs` reaches into the store.
#[test]
fn the_store_is_reached_from_one_file_and_it_is_not_the_fire() {
    let src = src();
    let mut namers: Vec<String> = rust_files(&src)
        .into_iter()
        .filter(|file| file.file_name().is_some_and(|name| name != "blob.rs"))
        .filter(|file| {
            std::fs::read_to_string(file)
                .map(|text| code(&text).any(|(_, said)| said.contains("crate::blob")))
                .unwrap_or(false)
        })
        .map(|file| {
            file.strip_prefix(&src)
                .unwrap_or(&file)
                .display()
                .to_string()
        })
        .collect();
    namers.sort();
    assert_eq!(
        namers,
        NAMERS.map(str::to_string).to_vec(),
        "the shared-adapter store's reach is a list, and this is it — a file \
         joining it must be a file that runs BETWEEN fires"
    );
}

/// **(3)** Inside the shell, only the four host verbs touch the store.
///
/// The scan tracks the enclosing `fn` and reports the one it found, which is
/// the diagnostic that matters: it is never news that `serve.rs` names the
/// store — it is news WHERE.
#[test]
fn only_the_four_host_verbs_touch_the_store() {
    let text = std::fs::read_to_string(src().join("serve.rs")).expect("the shell");
    let mut enclosing = String::from("<no function>");
    let mut touched: Vec<String> = Vec::new();
    for (_, said) in code(&text) {
        let trimmed = said.trim_start();
        if let Some(rest) = trimmed
            .strip_prefix("pub fn ")
            .or_else(|| trimmed.strip_prefix("fn "))
            .or_else(|| trimmed.strip_prefix("pub(crate) fn "))
        {
            enclosing = rest
                .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                .next()
                .unwrap_or("")
                .to_string();
        }
        if said.contains("self.adapters") && !touched.contains(&enclosing) {
            touched.push(enclosing.clone());
        }
    }
    touched.sort();
    let mut want = VERBS.map(str::to_string).to_vec();
    want.sort();
    assert_eq!(
        touched, want,
        "the store is reached from these functions of `Shell`; every one of \
         them must run between fires, the way `register_adapter` does"
    );
    // And the field is really there — a scan that found nothing would pass
    // every assertion above for the wrong reason.
    assert!(
        text.contains("adapters: crate::blob::Adapters"),
        "the shell holds the store; this gate is about where it is READ"
    );
}
