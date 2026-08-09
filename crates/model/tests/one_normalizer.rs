//! NO normalization of `config.json`, and a test that says so.
//!
//! It was named for a weaker claim: `pie.model/1` existed to make "what is
//! this model made of" a question answered ONCE, in Rust, at import. One
//! normalizer instead of four.
//!
//! There is no normalizer. `catalog` states the answer as a `const`, so the
//! question is not asked at run time at all, and the 845 lines that used to
//! answer it are deleted along with the descriptor they produced. What this
//! file still guards is the same perimeter for a stronger property: not
//! "only one reader parses this file" but **nobody does**.
//!
//! That property was worth a source GREP for as long as the second normalizer
//! could be new C++, in another language, behind an FFI boundary this crate
//! cannot see. It had already grown three times —
//! `crates/driver-cuda/csrc/src/model/config.cpp` (855 lines, 25 `model_type`
//! conditionals), `crates/driver-metal/csrc/src/model_facts.cpp`'s
//! `read_model_facts`, and `model::config` — and the three agreed only by
//! coincidence and a differential test.
//!
//! **Both of those trees are deleted.** Both drivers are Rust and read the
//! descriptor they are handed, so a fourth normalizer can no longer appear
//! where this file could not see it. The two greps that watched them are
//! retired below, each in the idiom its own guard demanded rather than left
//! to pass vacuously.
//!
//! What is left is the half a type still cannot hold: nothing stops a future
//! reader in `model` or `engine` from opening `config.json` again, and
//! `serde_json` is still in scope — `encoding.rs` needs it for the one
//! question a `const` cannot answer.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/model has two ancestors")
        .to_path_buf()
}



// THE CUDA CLAIM IS STRUCTURAL NOW. `the_cuda_driver_has_no_config_json_parser`
// walked `crates/driver-cuda/csrc/src` asking whether the C++ shell had
// grown its own `config.json` parser. That tree is deleted, so the answer
// is yes in the strongest available way and the test could only pass
// vacuously — which its own guard refused to let it do.
//
// AND NOW THE METAL CLAIM IS STRUCTURAL TOO.
//
// `the_metal_driver_boot_has_no_config_json_parser` walked
// `crates/driver-metal/csrc/src` asking whether the Metal C++ shell called
// `read_model_facts` on a boot path. That tree is deleted -- both drivers are
// Rust -- so the answer is yes in the strongest available way, and the test
// could only pass vacuously, which its own guard refused to let it do.
//
// `the_grep_finds_what_it_is_looking_for` went with it, and it named this
// exact ending: "Either it moved into the test tree -- in which case delete
// the Metal guard above, the duplication is gone for good -- or it was
// renamed". It was neither: the tree went.
//
// What remains of this file is the RUST-side half below, which is the half a
// type still cannot hold: nothing stops a future reader from opening
// `config.json` again, and `serde_json` is still in scope.

/// The **runtime** does not read `config.json` either.
///
/// `model`'s model service used to, when it was handed no artifact: two
/// probes reading `vocab_size` and `num_hidden_layers` straight off the file,
/// each walking `text_config` and key alternatives in its own order. They were
/// the last two of the four normalizations, and the least visible — a
/// disagreement with the driver showed up as a vocab-padding device fault
/// during decode, not as a load error.
///
/// This is a Rust-side guard where a type could not be one: nothing stops a
/// future reader from opening the file again, and `serde_json` is already in
/// scope for the descriptor.
///
/// `crates/model/src` is walked WITH NO EXCEPTION, which it could not be
/// before: it held the normalizer, and the normalizer was excused because
/// `normalize` took a `&serde_json::Value` and a path it only quoted in
/// errors. There is nothing to excuse now. The single reader that is left,
/// [`model::encoding::Encoding::from_config_json`], takes the BYTES — it is
/// handed a document it never located, so it cannot read the wrong one.
///
/// # The two drivers are walked here now
///
/// They were not, and three documents said they were. This file used to hold
/// a metal half and a cuda half, each walking a `csrc/src` tree; both trees
/// are deleted, both halves went with them, and what stayed was a Rust walk
/// over `model` and `engine` only. So the citations naming this test as the
/// guard on either driver had been pointing at a file that did not look at
/// them — which is `north-star.md` rule 4's failure mode, in the very test
/// written to name it.
///
/// Both driver trees are Rust now, so the same walk covers them: neither
/// opens `config.json` today, and adding them costs one line each and turns
/// "they do not" from a claim into a check. The stronger, family-level guard
/// each driver needs is its own `no_family_names.rs`.
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

