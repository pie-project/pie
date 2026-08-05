//! One normalization of `config.json`, and a test that says so.
//!
//! `pie.model/1` exists to make "what is this model made of" a question that
//! is answered once, in Rust, at import — or, for a plain HF snapshot, by
//! `worker/src/weights.rs` before any driver is created. What a driver reads
//! is the answer, never the question.
//!
//! That property is worth a test because it is not one a type can hold: the
//! second normalizer would be new C++, in another language, behind an FFI
//! boundary this crate cannot see. It had already grown three times —
//! `driver/cuda/src/model/config.cpp` (855 lines, 25 `model_type`
//! conditionals), `driver/metal/src/model_facts.cpp`'s `read_model_facts`,
//! and this crate — and the three agreed only by coincidence and a
//! differential test.
//!
//! So this is a source grep, in the idiom of `model/tests/registry_agreement.rs`
//! and `loader/tests/standalone.rs`: the property is about what another
//! language declares, and no amount of Rust-side assertion can reach it.
//!
//! What it does **not** check: that the drivers read the descriptor
//! *correctly*. That is `driver/cuda/tests/hf_config_dump/check_descriptor.sh`,
//! which ran the whole round trip — `config.json` → Rust normalize →
//! `pie.model/1` → C++ read — against `parse_hf_config` on all 55 corpus
//! configs and got the same answer. This test is what keeps the thing that
//! check retired from coming back.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("model/config has two ancestors")
        .to_path_buf()
}

/// Every `.cpp`/`.hpp`/`.mm`/`.cu` file under `rel`, recursively.
fn sources(rel: &str) -> Vec<PathBuf> {
    let root = repo_root().join(rel);
    let mut out = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir)
            .unwrap_or_else(|err| panic!("read {}: {err}", dir.display()));
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .extension()
                .is_some_and(|ext| matches!(ext.to_str(), Some("cpp" | "hpp" | "mm" | "cu")))
            {
                out.push(path);
            }
        }
    }
    assert!(
        !out.is_empty(),
        "found no sources under {rel}; the guard below would pass vacuously"
    );
    out.sort();
    out
}

/// Lines that call `needle`, ignoring comments and the function's own
/// definition or declaration.
///
/// Deliberately blunt. A cleverer filter would be a second thing to keep in
/// step with the C++; what this has to catch is a *call*, and a call is
/// `name(` in code that is neither commented out nor the signature itself.
///
/// Comments are cut at `//` rather than only skipped when they start the
/// line: prose that names the function it is about — a trailing
/// `// read_model_facts() arch` — is documentation, not a caller, and the
/// first version of this test failed on exactly that.
fn callers(path: &Path, needle: &str) -> Vec<String> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    text.lines()
        .enumerate()
        .filter(|(_, line)| {
            let code = line.split("//").next().unwrap_or("");
            let trimmed = code.trim_start();
            if trimmed.starts_with('*') {
                return false;
            }
            if !trimmed.contains(&format!("{needle}(")) {
                return false;
            }
            // The definition (`ModelFacts read_model_facts(const std::string&`)
            // and the declaration end in `{` or `;` with the return type on the
            // same line. A call site has the name preceded by `=`, `(`, or the
            // start of a statement.
            let before = trimmed.split(needle).next().unwrap_or("");
            !before.ends_with("ModelFacts ") && !before.ends_with("HfConfig ")
        })
        .map(|(i, line)| format!("{}:{}: {}", path.display(), i + 1, line.trim()))
        .collect()
}

/// The CUDA driver does not parse `config.json`.
///
/// `parse_hf_config` and its 855-line implementation were deleted outright, so
/// this guards against the declaration coming back as much as against a call.
#[test]
fn the_cuda_driver_has_no_config_json_parser() {
    let found: Vec<String> = sources("driver/cuda/src")
        .iter()
        .flat_map(|path| callers(path, "parse_hf_config"))
        .collect();
    assert!(
        found.is_empty(),
        "`parse_hf_config` is back in the CUDA driver. `config.json` is \
         normalized once, in this crate, and the driver reads the \
         `pie.model/1` descriptor (`driver/cuda/src/model/descriptor.hpp`):\n{}",
        found.join("\n")
    );
}

/// The Metal driver does not parse `config.json` either.
///
/// `read_model_facts` still *exists* — two Metal test binaries build fixtures
/// from a snapshot directory and link the driver lib to get it — but nothing
/// the driver runs may call it. That is the line this holds.
#[test]
fn the_metal_driver_boot_has_no_config_json_parser() {
    let found: Vec<String> = sources("driver/metal/src")
        .iter()
        .flat_map(|path| callers(path, "read_model_facts"))
        .filter(|line| !line.contains("read_model_facts_from_descriptor"))
        .collect();
    assert!(
        found.is_empty(),
        "the Metal driver calls `read_model_facts` again. It is test-only: \
         the boot reads the `pie.model/1` descriptor it is handed, and a \
         snapshot is normalized by `worker/src/weights.rs` before the driver \
         exists:\n{}",
        found.join("\n")
    );
}

/// The **runtime** does not read `config.json` either.
///
/// `pie-model`'s model service used to, when it was handed no artifact: two
/// probes reading `vocab_size` and `num_hidden_layers` straight off the file,
/// each walking `text_config` and key alternatives in its own order. They were
/// the last two of the four normalizations, and the least visible — a
/// disagreement with the driver showed up as a vocab-padding device fault
/// during decode, not as a load error.
///
/// This is a Rust-side guard where a type could not be one: nothing stops a
/// future reader from opening the file again, and `serde_json` is already in
/// scope for the descriptor.
#[test]
fn the_runtime_does_not_read_config_json() {
    let root = repo_root();
    let mut found = Vec::new();
    for rel in ["model/src", "runtime/engine/src"] {
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
        "the runtime reads `config.json` again. Model facts come from the \
         `pie.model/1` descriptor the worker hands it — normalized by this \
         crate for a snapshot, carried compiled by a `.zt`:\n{}",
        found.join("\n")
    );
}

/// The guard is not vacuous: the needle it greps for is one the file it was
/// deleted from would still match.
///
/// Without this, a rename on the C++ side (or a typo here) would turn both
/// tests above into assertions about a string nothing contains — passing
/// forever while the duplication grows back under a new name.
#[test]
fn the_grep_finds_what_it_is_looking_for() {
    let metal = sources("driver/metal/src");
    let declared = metal
        .iter()
        .any(|path| std::fs::read_to_string(path).unwrap().contains("read_model_facts"));
    assert!(
        declared,
        "no file under driver/metal/src mentions `read_model_facts`. Either it \
         moved into the test tree — in which case delete the Metal guard above, \
         the duplication is gone for good — or it was renamed, and the guard is \
         now watching a name nobody uses."
    );
}
