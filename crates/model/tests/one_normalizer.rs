//! NO normalization of `config.json`, and a test that says so.
//!
//! `catalog` states what a model is made of as a `const`, so the question is
//! not asked at run time at all. The property this file guards is not "only
//! one reader parses this file" but **nobody does**.
//!
//! That is the half a type cannot hold: nothing stops a future reader in
//! `model` or `engine` from opening `config.json` again, and `serde_json` is
//! still in scope — `encoding.rs` needs it for the one question a `const`
//! cannot answer.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/model has two ancestors")
        .to_path_buf()
}


/// The **runtime** does not read `config.json` either.
///
/// A Rust-side guard where a type could not be one. A second reader
/// disagreeing with the driver surfaces as a vocab-padding device fault
/// during decode rather than as a load error, so it has to be caught by a
/// grep and not by review.
///
/// `crates/model/src` is walked WITH NO EXCEPTION, and so are `engine` and
/// both drivers. The single reader that is left,
/// [`model::encoding::Encoding::from_config_json`], takes the BYTES — it is
/// handed a document it never located, so it cannot read the wrong one. The
/// stronger, family-level guard each driver needs is its own
/// `no_family_names.rs`.
#[test]
fn the_runtime_does_not_read_config_json() {
    let root = repo_root();
    let mut found = Vec::new();
    for rel in [
        "crates/model/src",
        "crates/engine/src",
        "crates/driver-metal/src",
        "crates/driver-cuda/src",
    ] {
        let dir = root.join(rel);
        let mut stack = vec![dir];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir)
                .unwrap_or_else(|err| panic!("read {}: {err}", dir.display()))
                .filter_map(Result::ok)
            {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path.extension().is_some_and(|ext| ext == "rs") {
                    let text = std::fs::read_to_string(&path).unwrap();
                    for (i, line) in text.lines().enumerate() {
                        let code = line.split("//").next().unwrap_or("");
                        if code.contains("\"config.json\"") {
                            found.push(format!("{}:{}: {}", path.display(), i + 1, line.trim()));
                        }
                    }
                }
            }
        }
    }
    assert!(
        found.is_empty(),
        "the runtime reads `config.json` again. What a model is made of is a \
         `catalog` row, stated as a `const` and matched to a checkpoint by \
         its TENSORS; the file itself answers exactly one question, the \
         declared encoding, and `model::encoding` is handed the bytes rather \
         than the path so that it cannot be the second reader:\n{}",
        found.join("\n")
    );
}
