//! Grep gate: no shell (`engine-cuda` or `engine-metal`) may call
//! `env::var` — every knob must be typed in `Boot`/`Budget`/`Profile`.

use std::path::{Path, PathBuf};

/// Both shells: the constitution's count is over shells, not one, so this
/// also polices `engine-metal`.
const SHELLS: [&str; 2] = ["../engine-cuda/src", "../engine-metal/src"];

/// Matches `var`, `var_os`, and any future `var_*` — the spelling doesn't
/// matter, the provenance does.
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
