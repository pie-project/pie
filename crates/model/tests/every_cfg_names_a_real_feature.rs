//! No `cfg(feature = "...")` names a feature its crate does not declare.
//!
//! # The slip this catches
//!
//! `#[cfg(feature = "forward")]` on a crate that has no `forward`
//! feature is not an error. It is not a warning. It compiles, and the
//! item behind it silently does not exist. Cargo's `unexpected_cfgs`
//! lint was built for exactly this and is off by default for `feature`
//! values on a crate that declares any features at all, because a
//! feature can legitimately be enabled by a dependency's `dep:`
//! forwarding.
//!
//! On an item, the cost is a function that quietly is not there. On a
//! test file, it is worse, because `#![cfg(feature = "x")]` at the top
//! of an integration test compiles the WHOLE FILE away and the runner
//! prints
//!
//! ```text
//! test result: ok. 0 passed; 0 failed
//! ```
//!
//! which reads exactly like a passing suite. A test that does not exist
//! cannot fail, so every negative control run against it comes back
//! green and the file looks unusually well-behaved.
//!
//! That is not hypothetical: `advertised_matches_what_is_shipped.rs`
//! was written with `#![cfg(feature = "forward")]` at the top, naming a
//! feature this crate's own refactor had deleted, and it reported
//! success over zero tests until someone read the count.
//!
//! # Why the whole workspace
//!
//! The same refactor deleted `config` too, and features are deleted by
//! whoever is removing the code behind them — who greps their own crate.
//! A `cfg` in a sibling crate naming a feature that just vanished has no
//! one looking at it. Scanning one crate would have caught the file
//! above and nothing else.
//!
//! # Not the same question as `declared_features.rs`
//!
//! That sibling asks whether a CONSUMER crate declares the features it
//! needs from a dependency -- the trap where dev-dependency feature
//! unification makes a library that cannot build alone pass every command
//! anyone runs. This asks whether a `cfg` names a feature its OWN crate
//! declares. Neither catches the other's case: the sibling reads
//! dependency edges and would not have looked at an inner attribute on an
//! integration test naming this crate's own deleted feature.
//!
//! # What it does NOT flag
//!
//! A feature enabled only by a dependency (`dep:serde`, or an optional
//! dependency's implicit feature) is declared as far as this is
//! concerned — both forms are collected below. Anything in a comment or
//! a doc string is skipped, because two files in this crate DESCRIBE the
//! deleted `forward` gate in prose and describing a mistake is how it
//! stays fixed.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The workspace root: `crates/` has a parent, and the parent has
/// sources of its own.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/model has a parent")
        .parent()
        .expect("crates/ has a parent")
        .to_path_buf()
}

/// Every directory holding a `Cargo.toml` that is not `target/`.
fn manifests(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    if dir.join("Cargo.toml").is_file() {
        out.push(dir.join("Cargo.toml"));
    }
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir()
            && !path
                .file_name()
                .is_some_and(|n| n == "target" || n == ".git")
        {
            manifests(&path, out);
        }
    }
}

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // A nested crate answers to its OWN manifest, and is scanned
            // when that manifest's turn comes.
            if !path.join("Cargo.toml").is_file()
                && !path
                    .file_name()
                    .is_some_and(|n| n == "target" || n == ".git")
            {
                rust_sources(&path, out);
            }
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The feature names a manifest declares, by either route.
///
/// `[features]` keys are the explicit ones. An optional dependency
/// creates a feature of its own name without appearing there at all,
/// which is why the dependency tables are read too — missing that would
/// make this file fail on correct code, and a gate that cries wolf gets
/// deleted.
fn declared_features(manifest: &str) -> BTreeSet<String> {
    let mut features = BTreeSet::new();
    let mut section = String::new();
    let mut table_dep: Option<String> = None;

    for line in manifest.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            section = line.trim_matches(['[', ']']).to_string();
            // `[dependencies.foo]` declares `foo` across the lines below.
            table_dep = section
                .strip_prefix("dependencies.")
                .or_else(|| section.strip_prefix("dev-dependencies."))
                .or_else(|| section.strip_prefix("build-dependencies."))
                .map(str::to_string);
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim().trim_matches('"');

        if section == "features" {
            features.insert(key.to_string());
            // `foo = ["dep:bar"]` and `foo = ["bar/baz"]` do not declare
            // anything further, but `dep:` names are worth having for the
            // reader.
            continue;
        }
        if section.ends_with("dependencies") && value.contains("optional") && value.contains("true")
        {
            features.insert(key.to_string());
        }
        if let Some(dep) = &table_dep
            && key == "optional"
            && value.contains("true")
        {
            features.insert(dep.clone());
        }
    }
    features
}

/// Every `feature = "..."` in real code, with its line number.
///
/// Occurrences after a `//` are skipped: this crate has three files that
/// explain the deleted `forward` gate in prose, and the explanation is
/// the reason it has not come back.
fn features_named(source: &str) -> Vec<(usize, String)> {
    let mut found = Vec::new();
    for (n, line) in source.lines().enumerate() {
        let code = match line.find("//") {
            Some(at) => &line[..at],
            None => line,
        };
        let mut rest = code;
        while let Some(at) = rest.find("feature = \"") {
            let after = &rest[at + "feature = \"".len()..];
            if let Some(end) = after.find('"') {
                found.push((n + 1, after[..end].to_string()));
                rest = &after[end..];
            } else {
                break;
            }
        }
    }
    found
}

/// Nothing in the workspace is gated on a feature that is not declared.
#[test]
fn every_cfg_names_a_feature_its_crate_declares() {
    let root = workspace_root();
    let mut found = Vec::new();
    manifests(&root, &mut found);
    assert!(
        found.len() >= 10,
        "found only {} manifests under {}, so this walk found nothing to \
         check",
        found.len(),
        root.display()
    );

    let mut ghosts: Vec<String> = Vec::new();
    let mut checked = 0usize;
    for manifest_path in &found {
        let Ok(manifest) = std::fs::read_to_string(manifest_path) else {
            continue;
        };
        let declared = declared_features(&manifest);
        let dir = manifest_path.parent().expect("a manifest has a directory");

        let mut sources = Vec::new();
        for sub in ["src", "tests", "benches", "examples"] {
            rust_sources(&dir.join(sub), &mut sources);
        }
        for source_path in sources {
            let Ok(source) = std::fs::read_to_string(&source_path) else {
                continue;
            };
            for (line, name) in features_named(&source) {
                checked += 1;
                if !declared.contains(&name) {
                    let shown = source_path
                        .strip_prefix(&root)
                        .unwrap_or(&source_path)
                        .display();
                    ghosts.push(format!(
                        "{shown}:{line} is gated on `{name}`, which its crate \
                         does not declare (it has: {declared:?})"
                    ));
                }
            }
        }
    }

    assert!(
        checked >= 50,
        "only {checked} `feature = \"...\"` sites were examined, which is \
         fewer than this workspace has -- the scan is not reaching the code"
    );
    assert!(
        ghosts.is_empty(),
        "{} cfg site(s) name a feature that does not exist. Each one \
         compiles its item away in silence, and on a test file that means \
         a green run over nothing:\n  {}",
        ghosts.len(),
        ghosts.join("\n  ")
    );
}

/// The scan can tell a declared feature from a deleted one.
///
/// Without this, every assertion above holds when `declared_features`
/// returns everything it sees, or when `features_named` returns nothing
/// at all -- the two ways this file could pass while looking at
/// nothing. It states the exact shape the mistake took, so a rewrite of
/// either helper that loses the ability is a failure here rather than a
/// silence in the test above.
#[test]
fn the_scan_can_see_the_mistake_it_was_written_for() {
    let manifest = "\
[package]
name = \"model\"

[features]
default = []
chat = []
contract = []

[dependencies]
serde = { version = \"1\", optional = true }

[dev-dependencies.criterion]
version = \"0.5\"
optional = true
";
    let declared = declared_features(manifest);
    assert!(declared.contains("chat"), "an explicit feature is declared");
    assert!(
        declared.contains("serde"),
        "an optional dependency declares a feature of its own name inline"
    );
    assert!(
        declared.contains("criterion"),
        "an optional dependency declares one in table form too"
    );
    assert!(
        !declared.contains("forward"),
        "a feature the refactor deleted is NOT declared -- if this ever \
         holds, the manifest reader is answering yes to everything"
    );

    let named = features_named(
        "\
#![cfg(feature = \"forward\")]
// a comment about cfg(feature = \"ghost\")
#[cfg(all(feature = \"chat\", feature = \"contract\"))]
fn f() {}
",
    );
    assert_eq!(
        named,
        vec![
            (1, "forward".to_string()),
            (3, "chat".to_string()),
            (3, "contract".to_string()),
        ],
        "the scan reads an inner attribute, reads both halves of one \
         line, and skips prose"
    );
}

// ---------------------------------------------------------------------------
// The same question, asked of the workflow instead of the source.
// ---------------------------------------------------------------------------

/// A package's own name, as its manifest gives it.
fn package_name(manifest: &str) -> Option<String> {
    let mut section = String::new();
    for line in manifest.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            section = line.trim_matches(['[', ']']).to_string();
            continue;
        }
        if section == "package"
            && let Some((key, value)) = line.split_once('=')
            && key.trim() == "name"
        {
            return Some(value.trim().trim_matches('"').to_string());
        }
    }
    None
}

/// Every `.yml`/`.yaml` under `.github/workflows`.
fn workflows(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(root.join(".github/workflows")) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "yml" || e == "yaml") {
            out.push(path);
        }
    }
    out.sort();
    out
}

/// Logical shell commands: `\`-continued lines joined, comments dropped.
///
/// A workflow's `run:` block is shell, and its cargo invocations wrap. A
/// scan that read physical lines would see `cargo test -p model \` and
/// `--features forward --test kernels_table` as two unrelated things and
/// attribute the feature to no package at all.
fn shell_commands(yaml: &str) -> Vec<(usize, String)> {
    let mut out: Vec<(usize, String)> = Vec::new();
    let mut pending: Option<(usize, String)> = None;
    for (n, raw) in yaml.lines().enumerate() {
        let line = raw.trim();
        if line.starts_with('#') {
            continue;
        }
        let (text, continues) = match line.strip_suffix('\\') {
            Some(head) => (head.trim_end(), true),
            None => (line, false),
        };
        match &mut pending {
            Some((_, acc)) => {
                acc.push(' ');
                acc.push_str(text);
            }
            None => pending = Some((n + 1, text.to_string())),
        }
        if !continues && let Some(done) = pending.take() {
            out.push(done);
        }
    }
    if let Some(done) = pending.take() {
        out.push(done);
    }
    out
}

/// The `-p`/`--package` names and the `--features` names in one command.
fn packages_and_features(command: &str) -> (Vec<String>, Vec<String>) {
    let mut packages = Vec::new();
    let mut features = Vec::new();
    let words: Vec<&str> = command.split_whitespace().collect();
    let mut i = 0;
    while i < words.len() {
        let (flag, inline) = match words[i].split_once('=') {
            Some((f, v)) => (f, Some(v)),
            None => (words[i], None),
        };
        let value = match inline {
            Some(v) => Some(v.to_string()),
            None => words.get(i + 1).map(|w| (*w).to_string()),
        };
        match flag {
            "-p" | "--package" | "--exclude" => {
                if let Some(v) = value
                    && !v.starts_with('-')
                {
                    packages.push(v);
                }
            }
            "-F" | "--features" => {
                if let Some(v) = value
                    && !v.starts_with('-')
                {
                    features.extend(v.split(',').map(str::to_string));
                }
            }
            _ => {}
        }
        i += 1;
    }
    (packages, features)
}

/// Every `features: x` value a matrix supplies, with its line number.
///
/// A build command reads `--features ${{ matrix.target.features }}`, and
/// no amount of shell parsing turns that into a feature name. The values
/// are in the same file, one `features:` key per matrix row, so a command
/// that asks for an expression is checked against all of them instead.
/// That is looser than checking the row that will actually run -- but a
/// renamed feature breaks EVERY row, and this is what stands between a
/// release build and finding out at tag time.
fn matrix_features(yaml: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in yaml.lines() {
        let line = line.trim();
        if line.starts_with('#') {
            continue;
        }
        let Some(value) = line.strip_prefix("features:") else {
            continue;
        };
        let value = value.trim().trim_matches(['"', '\'']);
        if value.is_empty() || value.contains("${{") {
            continue;
        }
        out.extend(value.split(',').map(|f| f.trim().to_string()));
    }
    out
}

/// No cargo command in a workflow asks for a feature nobody declares.
///
/// # Why this is not the sibling above
///
/// That one reads `cfg(feature = "...")` in Rust, where naming a deleted
/// feature compiles to nothing and says nothing. This reads the CI
/// workflow, where naming a deleted feature is a HARD error --
/// `the package 'model' does not contain this feature` -- and cargo
/// stops before it builds anything.
///
/// A hard error sounds like the safer of the two, and it is exactly as
/// invisible when nothing runs the workflow. `ci.yml` fires on pushes to
/// `main` and on pull requests; the branch this crate is being rewritten
/// on is neither, so a rewrite that deletes a feature leaves four cargo
/// commands naming it and every one of them looks fine until the merge.
///
/// That is not hypothetical either. The same refactor whose deleted
/// `forward` feature the sibling test above was written for left
/// `--features forward` in four places in `ci.yml`, and it sat there
/// because the only machine that would have objected never got a turn.
#[test]
fn every_workflow_feature_names_a_feature_its_package_declares() {
    let root = workspace_root();
    let mut manifest_paths = Vec::new();
    manifests(&root, &mut manifest_paths);

    let mut declared: std::collections::BTreeMap<String, BTreeSet<String>> =
        std::collections::BTreeMap::new();
    for path in &manifest_paths {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        if let Some(name) = package_name(&text) {
            declared.insert(name, declared_features(&text));
        }
    }
    assert!(
        declared.contains_key("model") && declared.len() >= 10,
        "read {} packages, which is not this workspace",
        declared.len()
    );

    let files = workflows(&root);
    assert!(
        !files.is_empty(),
        "no workflow found under {}/.github/workflows -- this test would \
         pass over nothing",
        root.display()
    );

    let mut ghosts = Vec::new();
    let mut checked = 0usize;
    for file in &files {
        let Ok(yaml) = std::fs::read_to_string(file) else {
            continue;
        };
        for (line, command) in shell_commands(&yaml) {
            if !command.contains("cargo ") {
                continue;
            }
            let (packages, features) = packages_and_features(&command);
            if features.is_empty() {
                continue;
            }
            // Attributed to a package or not checkable: `--workspace
            // --features x` unifies across everything and this cannot
            // say whose feature it is.
            let known: Vec<&BTreeSet<String>> =
                packages.iter().filter_map(|p| declared.get(p)).collect();
            if known.is_empty() {
                continue;
            }
            let matrix = matrix_features(&yaml);
            for feature in features {
                // `dep/feat` names a DEPENDENCY's feature, which this
                // package need not declare.
                if feature.contains('/') || feature.is_empty() {
                    continue;
                }
                // `--features ${{ matrix.target.features }}`: the name is
                // not here, so every value the matrix can supply is
                // checked in its place.
                let names: Vec<String> = if feature.contains("${{") {
                    matrix.clone()
                } else {
                    vec![feature]
                };
                for feature in names {
                    checked += 1;
                    if !known.iter().any(|d| d.contains(&feature)) {
                        let name = file
                            .file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        ghosts.push(format!(
                            "{name}:{line} asks {packages:?} for `--features \
                         {feature}`, which none of them declares"
                        ));
                    }
                }
            }
        }
    }

    assert!(
        checked >= 10,
        "only {checked} feature names were checked across {} workflow \
         files, so this scan is not reading the cargo commands it thinks \
         it is",
        files.len()
    );
    assert!(
        ghosts.is_empty(),
        "CI names features that do not exist, and cargo stops with `the \
         package '...' does not contain this feature` before it builds \
         anything:\n  {}",
        ghosts.join("\n  ")
    );
}

/// The workflow scan reads continuations, flags, and prose the way it claims.
#[test]
fn the_workflow_scan_reads_a_wrapped_cargo_line() {
    let sample = "\
      # cargo test -p model --features ghost
      - name: something
        run: |
          cargo test -p model --features forward \\
            --test kernels_table
          cargo clippy -p driver-metal --features=metal-4 --all-targets
";
    let commands = shell_commands(sample);
    let joined: Vec<&str> = commands.iter().map(|(_, c)| c.as_str()).collect();
    assert!(
        joined.contains(&"cargo test -p model --features forward --test kernels_table"),
        "the wrapped command was not rejoined: {joined:?}"
    );
    assert!(
        !joined.iter().any(|c| c.contains("ghost")),
        "a commented-out cargo line was read as a command: {joined:?}"
    );

    let (packages, features) =
        packages_and_features("cargo test -p model -p kernels --features contract,dep/x");
    assert_eq!(packages, vec!["model", "kernels"]);
    assert_eq!(features, vec!["contract", "dep/x"]);

    let (_, equals) = packages_and_features("cargo clippy -p driver-metal --features=metal-4");
    assert_eq!(equals, vec!["metal-4"], "`--features=x` was not read");

    let (_, none) = packages_and_features("cargo test -p model --features --test x");
    assert!(
        none.is_empty(),
        "a flag was read as a feature name: {none:?}"
    );
}
