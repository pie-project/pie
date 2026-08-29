//! **ARTICLE 9'S GATE**: a shell reads no environment (alto design §1).
//!
//! > *Shells read no environment: every knob is typed in `Boot`/`Budget`/
//! > `Profile` and portable across backends. Enforcement: a grep gate
//! > (`env::var` count in shells = 0).*
//!
//! This is that grep, written down. It is a plain test and not a build script
//! on purpose — a build script would run at compile time on a machine that may
//! not have the sibling crate, and would report through a channel nobody reads
//! on a green run. A test names the file and the line.
//!
//! # Why a grep and not a type
//!
//! Everything else in this constitution is enforced by a type or a submit-time
//! gate, because the failure has a shape the compiler can see. This one does
//! not: `std::env::var` is legal Rust everywhere, and the bug is not that the
//! call is wrong — it is that the ANSWER did not come from the boot document.
//! A knob a shell reads out of its own process environment is a knob that is
//! not in the config, does not travel to the other shell, cannot be diffed
//! against what a deployment asked for, and is invisible to every reader. That
//! is a fact about provenance, and provenance is what a grep can see.
//!
//! # What died to make this pass
//!
//! Nine `PIE_CUDA_*` words in `serve.rs` — `GRAPHS`, `STREAMS`, `GROUPED`,
//! `BUCKETS`, `PAD`, `FOLD`, `PIPELINE`, `FOLD_DISABLE`, `FALLBACK_COPY` —
//! and, in `program/compile.rs`, the three-step `PIE_HOME`/`XDG_CACHE_HOME`/
//! `HOME` walk that found the cubin cache. They are `Boot` fields and
//! `[engine]`/`[cache]` keys now (alto wave P). The Metal shell never had any,
//! and this gate is what keeps that true of both.

use std::path::{Path, PathBuf};

/// The two shells, as directories relative to this crate's manifest.
///
/// `engine-metal` is a SIBLING and is read on purpose: the constitution's
/// count is over shells, not over one shell, and "Metal has zero env vars"
/// (survey §2 debt 6) is a property that has to keep being true. A gate that
/// policed only the crate it lives in would let the next one land next door.
const SHELLS: [&str; 2] = ["../engine-cuda/src", "../engine-metal/src"];

/// What a shell may not do. `env::var` matches `var`, `var_os` and any
/// `var_*` a future standard library grows, which is the whole family: the
/// spelling is not the point, the ANSWER's provenance is.
const FORBIDDEN: &str = "env::var";

#[test]
fn no_shell_reads_the_environment() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut offences: Vec<String> = Vec::new();
    let mut scanned = 0usize;
    for shell in SHELLS {
        let directory = root.join(shell);
        assert!(
            directory.is_dir(),
            "{directory:?} is not a directory; this gate reads both shells' \
             sources and cannot answer for one it cannot find"
        );
        for file in rust_files(&directory) {
            let text = std::fs::read_to_string(&file)
                .unwrap_or_else(|error| panic!("read {file:?}: {error}"));
            scanned += 1;
            for (at, line) in text.lines().enumerate() {
                if line.contains(FORBIDDEN) && !line.trim_start().starts_with("//") {
                    offences.push(format!(
                        "{}:{}: {}",
                        file.display(),
                        at + 1,
                        line.trim()
                    ));
                }
            }
        }
    }
    assert!(
        scanned > 20,
        "only {scanned} source files were scanned; the gate is looking in the \
         wrong place and would pass on an empty tree"
    );
    assert!(
        offences.is_empty(),
        "article 9: shells read no environment, and {} read(s) are compiled \
         into one. Every knob is typed on `Boot`/`Budget`/`Profile` and \
         arrives through the boot document:\n{}",
        offences.len(),
        offences.join("\n")
    );
}

/// Every `.rs` file under `directory`, depth-first.
fn rust_files(directory: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![directory.to_path_buf()];
    while let Some(at) = stack.pop() {
        let entries = std::fs::read_dir(&at)
            .unwrap_or_else(|error| panic!("read the directory {at:?}: {error}"));
        for entry in entries {
            let path = entry.expect("a directory entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|kind| kind == "rs") {
                out.push(path);
            }
        }
    }
    out
}
