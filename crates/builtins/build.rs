//! Build every `inferlets/*` whose Pie.toml says `tier = "builtin"` for
//! wasm32-wasip2 and hand the components to `lib.rs` as `include_bytes!`.
//!
//! The inferlets are a separate cargo workspace (they target wasm), so this
//! runs cargo for it, into that workspace's own target directory. The
//! `wasm32-wasip2` target comes with the toolchain `rust-toolchain.toml`
//! pins; without it the build fails here with the command to add it.
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
    version: String,
    manifest: PathBuf,
}

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repository root");
    let inferlets = Path::new(env!("CARGO_MANIFEST_DIR")).join("inferlets");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let target_dir = match env::var_os("PIE_BUILTINS_TARGET_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => inferlets.join("target"),
    };

    println!("cargo:rerun-if-env-changed=PIE_BUILTINS");
    println!("cargo:rerun-if-env-changed=PIE_BUILTINS_TARGET_DIR");
    for dep in ["compat", "Cargo.toml", "Cargo.lock"] {
        println!("cargo:rerun-if-changed={}", inferlets.join(dep).display());
    }
    for dep in [
        "crates/inferlet/src",
        "crates/inferlet/wit",
        "crates/inferlet-macros/src",
        "crates/eta-dsl/src",
    ] {
        println!("cargo:rerun-if-changed={}", root.join(dep).display());
    }

    let builtins = find_builtins(&inferlets);
    for builtin in &builtins {
        println!(
            "cargo:rerun-if-changed={}",
            builtin.manifest.parent().unwrap().display()
        );
    }

    let mut generated = String::from("static ALL: &[Builtin] = &[\n");
    if env::var("PIE_BUILTINS").is_ok_and(|v| v == "skip") {
        println!("cargo:warning=PIE_BUILTINS=skip: this pie embeds no built-in inferlets");
    } else {
        build(&inferlets, &target_dir, &builtins);
        for builtin in &builtins {
            let wasm = target_dir
                .join("wasm32-wasip2/release")
                .join(format!("{}.wasm", builtin.name.replace('-', "_")));
            let staged = out_dir.join(format!("{}.wasm", builtin.name));
            fs::copy(&wasm, &staged)
                .unwrap_or_else(|e| panic!("{} was not built: {e}", wasm.display()));
            generated.push_str(&format!(
                "    Builtin {{ name: {:?}, version: {:?}, manifest: include_str!({:?}), component: include_bytes!({:?}) }},\n",
                builtin.name,
                builtin.version,
                builtin.manifest.display().to_string(),
                staged.display().to_string(),
            ));
        }
    }
    generated.push_str("];\n");
    fs::write(out_dir.join("builtins.rs"), generated).expect("write builtins.rs");
}

/// Every `<dir>/Pie.toml` under `inferlets` that says `tier = "builtin"`,
/// with its `version`.
fn find_builtins(inferlets: &Path) -> Vec<Builtin> {
    let mut builtins = Vec::new();
    let entries =
        fs::read_dir(inferlets).unwrap_or_else(|e| panic!("reading {}: {e}", inferlets.display()));
    for entry in entries.flatten() {
        let manifest = entry.path().join("Pie.toml");
        let Ok(text) = fs::read_to_string(&manifest) else {
            continue;
        };
        if !text
            .lines()
            .any(|line| string_value(line, "tier").as_deref() == Some("builtin"))
        {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        let version = text
            .lines()
            .find_map(|line| string_value(line, "version"))
            .unwrap_or_else(|| panic!("{}: no version", manifest.display()));
        builtins.push(Builtin {
            name,
            version,
            manifest,
        });
    }
    builtins.sort_by(|a, b| a.name.cmp(&b.name));
    assert!(
        !builtins.is_empty(),
        "no Pie.toml under {} says tier = \"builtin\"",
        inferlets.display()
    );
    builtins
}

/// The string a `key = "..."` line holds, for the few keys this needs
/// without a TOML parser.
fn string_value(line: &str, key: &str) -> Option<String> {
    let (k, v) = line.split_once('=')?;
    if k.trim() != key {
        return None;
    }
    let v = v.trim().split('#').next()?.trim();
    v.strip_prefix('"')?.strip_suffix('"').map(str::to_string)
}

fn build(inferlets: &Path, target_dir: &Path, builtins: &[Builtin]) {
    let cargo = env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let mut command = Command::new(cargo);
    command
        .arg("build")
        .arg("--manifest-path")
        .arg(inferlets.join("Cargo.toml"))
        .arg("--release")
        .arg("--target")
        .arg("wasm32-wasip2");
    for builtin in builtins {
        command.arg("-p").arg(&builtin.name);
    }
    // The outer build's flags are for the host; the inferlets are wasm.
    for var in [
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
        .unwrap_or_else(|e| panic!("running cargo for the built-in inferlets: {e}"));
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let hint =
            if stderr.contains("wasm32-wasip2") && stderr.contains("target may not be installed") {
                "\n\nthe wasm32-wasip2 target is missing: `rustup target add wasm32-wasip2` adds it"
            } else {
                ""
            };
        panic!(
            "building the built-in inferlets (crates/builtins/inferlets) for wasm32-wasip2 failed:\n{stderr}{hint}\n\n\
             PIE_BUILTINS=skip builds a pie with no built-in inferlets (every compat route then answers 404)"
        );
    }
}
