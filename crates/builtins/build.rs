//!
//! The inferlets are a separate cargo workspace (they target wasm), so this
//! runs cargo for it, into that workspace's own target directory, on the
//! toolchain that workspace's `rust-toolchain.toml` pins (nightly, for
//! `wasm32-wasip3`); without it the build fails here with how to add it.
//!
//! `PIE_BUILTINS=skip` embeds nothing — for a host that cannot build wasm
//! at all. A pie built that way answers 404 on every compat route and
//! `pie doctor` says so.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

struct Builtin {
    name: String,
    dir: PathBuf,
}

fn main() {
    let root = manifest_dir()
        .join("../..")
        .canonicalize()
        .expect("repository root");
    let inferlets = manifest_dir().join("inferlets");
    let policies = manifest_dir().join("policies");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let target_dir = env::var_os("PIE_BUILTINS_TARGET_DIR").map(PathBuf::from);

    println!("cargo:rerun-if-env-changed=PIE_BUILTINS");
    println!("cargo:rerun-if-env-changed=PIE_BUILTINS_TARGET_DIR");
    for dep in ["compat", "Cargo.toml", "Cargo.lock"] {
        println!("cargo:rerun-if-changed={}", inferlets.join(dep).display());
    }
    for dep in ["Cargo.toml", "Cargo.lock"] {
        println!("cargo:rerun-if-changed={}", policies.join(dep).display());
    }
    for dep in [
        "crates/inferlet/src",
        "crates/inferlet/wit",
        "crates/client-api/src",
        "crates/inferlet-macros/src",
        "crates/eta-dsl/src",
        "crates/policy/src",
        "crates/policy/wit",
    ] {
        println!("cargo:rerun-if-changed={}", root.join(dep).display());
    }

    let builtins = find_builtins(&inferlets);
    let test_policies = env::var_os("CARGO_FEATURE_TEST_POLICIES").is_some();
    let policy_builtins: Vec<Builtin> = find_builtins(&policies)
        .into_iter()
        .filter(|policy| test_policies || !policy.name.starts_with("test-"))
        .collect();
    for builtin in builtins.iter().chain(&policy_builtins) {
        println!("cargo:rerun-if-changed={}", builtin.dir.display());
    }

    let skip = env::var("PIE_BUILTINS").is_ok_and(|v| v == "skip");
    if skip {
        println!("cargo:warning=PIE_BUILTINS=skip: this pie embeds no built-in inferlets");
    }
    // What the built-ins are built from, by content: every file of the
    // inputs above (the inferlets' and policies' workspaces, their manifests
    // and lockfiles, the guest crates they build on), each by its path
    // under the repository and its bytes, sorted. Two trees of the same
    // source give the same fingerprint, where the built bytes differ (cargo
    // keys a path crate's symbols by where it lies), so a key on a
    // built-in's behaviour (a committed host cover's) takes this, not the
    // component's hash.
    let mut inputs: Vec<PathBuf> = vec![inferlets.clone(), policies.clone()];
    for dep in [
        "crates/inferlet/src",
        "crates/inferlet/wit",
        "crates/client-api/src",
        "crates/inferlet-macros/src",
        "crates/eta-dsl/src",
        "crates/policy/src",
        "crates/policy/wit",
    ] {
        inputs.push(root.join(dep));
    }
    let source = source_fingerprint(&root, &inputs);
    let mut generated = format!("/// What the built-ins are built from, by content.\npub const SOURCE: &str = {source:?};\n");
    for (name, workspace, set) in [
        ("ALL", &inferlets, &builtins),
        ("POLICIES", &policies, &policy_builtins),
    ] {
        let staged_dir = out_dir.join(name.to_lowercase());
        fs::create_dir_all(&staged_dir).expect("create staging directory");
        generated.push_str(&format!("static {name}: &[Builtin] = &[\n"));
        if !skip {
            let target_dir = target_dir
                .clone()
                .unwrap_or_else(|| workspace.join("target"));
            build(workspace, &target_dir, set);
            for builtin in set {
                let wasm = target_dir
                    .join("wasm32-wasip3/release")
                    .join(format!("{}.wasm", builtin.name.replace('-', "_")));
                let staged = staged_dir.join(format!("{}.wasm", builtin.name));
                fs::copy(&wasm, &staged)
                    .unwrap_or_else(|e| panic!("{} was not built: {e}", wasm.display()));
                generated.push_str(&format!(
                    "    Builtin {{ name: {:?}, component: include_bytes!({:?}), source: SOURCE }},\n",
                    builtin.name,
                    staged.display().to_string(),
                ));
            }
        }
        generated.push_str("];\n");
    }
    fs::write(out_dir.join("builtins.rs"), generated).expect("write builtins.rs");
}

/// A fingerprint of every file under `inputs` (no `target` directory),
/// each by its path relative to `root` and its bytes, in sorted order.
fn source_fingerprint(root: &Path, inputs: &[PathBuf]) -> String {
    let mut files: Vec<PathBuf> = Vec::new();
    let mut walk: Vec<PathBuf> = inputs.to_vec();
    while let Some(at) = walk.pop() {
        let Ok(entries) = fs::read_dir(&at) else {
            if at.is_file() {
                files.push(at);
            }
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().is_some_and(|n| n == "target" || n == ".git") {
                    continue;
                }
                walk.push(path);
            } else {
                files.push(path);
            }
        }
    }
    let mut keyed: Vec<(String, PathBuf)> = files
        .into_iter()
        .map(|p| {
            let rel = p.strip_prefix(root).unwrap_or(&p).to_string_lossy().replace('\\', "/");
            (rel, p)
        })
        .collect();
    keyed.sort();
    keyed.dedup_by(|a, b| a.0 == b.0);
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for (rel, path) in &keyed {
        eat(rel.as_bytes());
        eat(&[0]);
        eat(&fs::read(path).unwrap_or_default());
        eat(&[0]);
    }
    format!("{hash:016x}")
}

fn find_builtins(workspace: &Path) -> Vec<Builtin> {
    let mut builtins = Vec::new();
    let entries =
        fs::read_dir(workspace).unwrap_or_else(|e| panic!("reading {}: {e}", workspace.display()));
    for entry in entries.flatten() {
        let cargo_toml = entry.path().join("Cargo.toml");
        let Ok(text) = fs::read_to_string(&cargo_toml) else {
            continue;
        };
        if !text.lines().any(|line| {
            line.split_once('=')
                .is_some_and(|(k, v)| k.trim() == "crate-type" && v.contains("cdylib"))
        }) {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        builtins.push(Builtin {
            name,
            dir: entry.path(),
        });
    }
    builtins.sort_by(|a, b| a.name.cmp(&b.name));
    assert!(
        !builtins.is_empty(),
        "no crate under {} builds a component",
        workspace.display()
    );
    builtins
}

fn build(workspace: &Path, target_dir: &Path, builtins: &[Builtin]) {
    // rustup's `cargo`, so the workspace's own rust-toolchain.toml (nightly,
    // for wasm32-wasip3) applies, not the host build's toolchain.
    let mut command = Command::new("cargo");
    command.current_dir(workspace);
    command
        .arg("build")
        .arg("--manifest-path")
        .arg(workspace.join("Cargo.toml"))
        .arg("--release")
        .arg("--target")
        .arg("wasm32-wasip3");
    for builtin in builtins {
        command.arg("-p").arg(&builtin.name);
    }
    // The outer build's toolchain and flags are for the host; the built-ins
    // are wasm.
    for var in [
        "RUSTUP_TOOLCHAIN",
        "RUSTC",
        "RUSTDOC",
        "CARGO_ENCODED_RUSTFLAGS",
        "CARGO_BUILD_RUSTFLAGS",
        "RUSTFLAGS",
        "CARGO_BUILD_TARGET",
        "CARGO_TARGET_DIR",
        "RUSTC_WORKSPACE_WRAPPER",
    ] {
        command.env_remove(var);
    }
    command.env("CARGO_TARGET_DIR", target_dir);

    let output = command
        .output()
        .unwrap_or_else(|e| panic!("running cargo for {}: {e}", workspace.display()));
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let hint = if stderr.contains("wasm32-wasip3")
            && stderr.contains("target may not be installed")
        {
            "\n\nthe wasm32-wasip3 target is missing: `rustup toolchain install` in the workspace adds it"
        } else {
            ""
        };
        panic!(
            "building the built-ins in {} for wasm32-wasip3 failed:\n{stderr}{hint}\n\n\
             PIE_BUILTINS=skip builds a pie with no built-in inferlets (every compat route then answers 404)",
            workspace.display()
        );
    }
}

/// This crate's directory, read when the build script runs: a path baked in
/// when the script was compiled would name another checkout's crate where
/// worktrees share a target directory.
fn manifest_dir() -> std::path::PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .expect("cargo sets CARGO_MANIFEST_DIR for a build script")
        .into()
}
