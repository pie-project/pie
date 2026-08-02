//! The architecture, as a test.
//!
//! `loader/architecture.md` §12 row 12 states one property: the loader is a
//! standalone compile library, so
//!
//! ```text
//! plan = compile(source_facts, program, target)
//! ```
//!
//! takes exactly three values, none of which is a model's name and none of
//! which is a path. Every other file in this directory tests what the compiler
//! *computes*; this one tests what it is *allowed to know*, which no amount of
//! plan-level assertion can catch — a `File::open` buried three modules down
//! produces perfectly correct plans right up until the caller has no directory.
//!
//! Two source greps and one compile. They are cheap, and they are the only
//! thing standing between the current design and a slow drift back.

use std::path::{Path, PathBuf};

fn loader_src() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Every `.rs` file under `loader/src`, with its path relative to `src`.
fn sources() -> Vec<(String, String)> {
    fn walk(dir: &Path, root: &Path, out: &mut Vec<(String, String)>) {
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .unwrap_or_else(|err| panic!("read {}: {err}", dir.display()))
            .map(|entry| entry.expect("dir entry").path())
            .collect();
        entries.sort();
        for path in entries {
            if path.is_dir() {
                walk(&path, root, out);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let rel = path
                    .strip_prefix(root)
                    .expect("under root")
                    .to_string_lossy()
                    .replace('\\', "/");
                out.push((rel, std::fs::read_to_string(&path).expect("read source")));
            }
        }
    }
    let root = loader_src();
    let mut out = Vec::new();
    walk(&root, &root, &mut out);
    assert!(out.len() > 20, "the walk found almost nothing: {out:?}");
    out
}

/// Lines outside `#[cfg(test)]`, approximated by cutting each file at its test
/// module. Tests legitimately write fixtures to disk and name real models.
fn production_lines(body: &str) -> impl Iterator<Item = (usize, &str)> {
    let end = body
        .find("\n#[cfg(test)]\n")
        .map_or(body.len(), |at| at + 1);
    body[..end]
        .lines()
        .enumerate()
        .map(|(index, line)| (index + 1, line))
        .filter(|(_, line)| !line.trim_start().starts_with("//"))
}

/// The compiler opens no files.
///
/// Five places are allowed to, and each is allowed for a reason that is about
/// *bytes* or about the *world*, never about a decision:
///
/// * `checkpoint/` — the reader. Turning a directory into a
///   `CheckpointMetadata` is precisely the step this test exists to keep
///   separate, so it has to live somewhere, and it lives in one module.
/// * `executor/host.rs` — it runs a finished plan, which means copying
///   weight bytes; that is its whole job. (It lived under `testkit/` until
///   `pie model convert` made it a production path.)
/// * `cache_key.rs` — the on-disk plan cache. It stats and writes files that are
///   outputs of compilation, never inputs to it.
/// * `weight_store.rs` — the materialized-weight artifact, which is an output
///   for the same reason: it is written *after* a plan has run, from device
///   memory the driver already holds, and read back only to skip doing that
///   again. Nothing the compiler decides depends on whether one exists.
/// * `verify.rs` — staleness. `verify` is deliberately *not* `compile`: its
///   whole point is to compare a plan against the world it was compiled for, so
///   asking whether the files still have the recorded sizes is the check, not a
///   leak. A `compile` that could do this would be a different function.
/// * `main.rs` — the CLI, which is a caller, not the library.
/// * `testkit.rs` — feature-gated test support (the fixture writer). The
///   build this property protects — the worker's — compiles with `testkit`
///   off, so the gate enforces for production what this exemption relaxes
///   for tests.
#[test]
fn nothing_below_the_reader_opens_a_file() {
    const ALLOWED: &[&str] = &[
        "executor/host.rs",
        "cache_key.rs",
        "verify.rs",
        "weight_store.rs",
        "main.rs",
        "testkit.rs",
    ];
    // `Path`/`PathBuf` are values and may be passed around freely; what must not
    // appear is anything that *touches* the filesystem.
    const FORBIDDEN: &[&str] = &[
        "std::fs",
        "File::open",
        "File::create",
        "read_to_string",
        "read_dir",
        "memmap",
        "Mmap",
    ];

    let mut offences = Vec::new();
    for (rel, body) in sources() {
        // Test-only files are exempt wholesale: a fixture has to be written
        // somewhere, and a golden has to be read back.
        if ALLOWED.contains(&rel.as_str())
            || rel.starts_with("checkpoint/")
            || rel.ends_with("tests.rs")
        {
            continue;
        }
        for (line_no, line) in production_lines(&body) {
            for needle in FORBIDDEN {
                if line.contains(needle) {
                    offences.push(format!("{rel}:{line_no}: {needle} in `{}`", line.trim()));
                }
            }
        }
    }
    assert!(
        offences.is_empty(),
        "the compiler must not touch the filesystem; move the read into \
         `checkpoint/read.rs` and pass a value:\n{}",
        offences.join("\n")
    );
}

/// A plan compiles with no checkpoint on disk.
///
/// The negative test above says no one *calls* the filesystem. This one says
/// the interface does not *need* it: the fourteen golden plans in
/// `golden_plans.rs` are all built from synthetic `CheckpointMetadata`, and this
/// asserts the property directly, with a path that provably does not exist.
#[test]
fn a_plan_compiles_from_a_value_with_no_checkpoint_anywhere() {
    use pie_loader::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
    use pie_loader::contract::{Expr, ModelContract, TensorContract};
    use pie_loader::plan::StorageTarget;
    use pie_loader::plan::compile as compile_load_plan;
    use pie_loader::types::{CheckpointFormat, DType, Encoding, FileId, TensorId};

    let nowhere = "/nonexistent/there-is-no-checkpoint-here/model.safetensors";
    assert!(
        !Path::new(nowhere).exists(),
        "the fixture path must not exist for this test to mean anything"
    );

    let metadata = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: nowhere.to_string(),
            size_bytes: 4096,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![RawTensor {
            id: TensorId(0),
            name: "w".into(),
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: 16,
            shape: vec![2, 2],
            encoding: Encoding::Raw(DType::F32),
        }],
    };
    let contract = ModelContract {
        alignment: 256,
        tensors: vec![TensorContract::new(
            "w",
            Expr::src("w"),
            vec![2, 2],
            Encoding::Raw(DType::F32),
        )],
        groups: Vec::new(),
    };

    let plan = compile_load_plan(&metadata, &contract, StorageTarget::default())
        .expect("compiling must not require the checkpoint to exist");
    assert_eq!(plan.tensors.len(), 1);
    assert!(!plan.instrs.is_empty());
}

/// The compiler names no model and no device.
///
/// The second half of the standalone property. A library that opens no files
/// but still contains `if model_type == "llama"` is not standalone: it has a
/// finite list of things it can load, and the list is in the wrong repository.
/// Same for a device constant — the moment `128` means "the row granularity of
/// this GPU's block scales" and lives here, the loader has an opinion about
/// hardware it cannot see.
///
/// There is no exemption. `arch.rs`, `arch/` and `config.rs` used to be exempt
/// *by name*, which is what made deleting them turn this test from a lint into
/// a proof: the moment the exemption had nothing left to cover, the property
/// held over the whole crate rather than over the part of it nobody had
/// finished migrating.
#[test]
fn nothing_in_the_compiler_knows_what_a_model_is() {
    // Real families the loader currently dispatches on, plus the vendor
    // vocabularies that are the same mistake in a different spelling.
    const MODEL_NAMES: &[&str] = &[
        "llama", "qwen", "gpt_oss", "gpt-oss", "nemotron", "phi3", "gemma", "mistral", "deepseek",
        "mixtral", "granite",
    ];
    let mut offences = Vec::new();
    for (rel, body) in sources() {
        if rel.ends_with("tests.rs") {
            continue;
        }
        for (line_no, line) in production_lines(&body) {
            // Only string literals count. A local named `llama_rows` would be a
            // style problem; `"llama"` is a dispatch decision.
            let lowered = line.to_ascii_lowercase();
            for name in MODEL_NAMES {
                if lowered.contains(&format!("\"{name}")) {
                    offences.push(format!("{rel}:{line_no}: \"{name}\" in `{}`", line.trim()));
                }
            }
        }
    }
    assert!(
        offences.is_empty(),
        "the compiler must not know what a model is; the driver states it in a \
         contract:\n{}",
        offences.join("\n")
    );
}

/// The backend lowering states no device constant of its own.
///
/// `plan/passes/tile.rs` decides how a transform is *expressed* for a device;
/// what the device can *do* arrives in `StorageTarget`. The distinction is easy
/// to lose one number at a time — a fallback tile size, a block-scale
/// granularity, a scratch dtype — and each one makes the plan depend on
/// something no caller stated and no artifact key covers.
#[test]
fn the_backend_lowering_reads_its_numbers_off_the_target() {
    const TILE: &str = "plan/passes/tile.rs";
    let sources = sources();
    let (_, tile) = sources
        .iter()
        .find(|(rel, _)| rel == TILE)
        .expect("the backend lowering must exist for this test to mean anything");

    let mut offences = Vec::new();
    for (line_no, line) in production_lines(tile) {
        let trimmed = line.trim();
        // A `const` here is a number the driver never stated. The tile-map masks
        // are the one exception and are not capability claims: they are the set
        // of transforms the *loader* knows how to emit, which the driver
        // intersects with its own.
        if trimmed.starts_with("const ") || trimmed.starts_with("pub const ") {
            if trimmed.contains("TILE_MAP_MASK") {
                continue;
            }
            offences.push(format!("{TILE}:{line_no}: {trimmed}"));
        }
    }
    assert!(
        offences.is_empty(),
        "a device constant belongs in `StorageTarget`, stated by the driver \
         that measured it:\n{}",
        offences.join("\n")
    );
}
