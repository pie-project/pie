//! Embeds every package under the repository's `models/`: each directory
//! with a `package.star`, and every `.star` file in it.

use std::fmt::Write as _;
use std::path::Path;

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    println!("cargo:rerun-if-changed={}", root.display());
    let mut packages: Vec<(String, Vec<(String, String)>)> = Vec::new();
    for dir in std::fs::read_dir(&root).expect("the repository's models/") {
        let dir = dir.expect("a models/ entry").path();
        if !dir.join("package.star").is_file() {
            continue;
        }
        println!("cargo:rerun-if-changed={}", dir.display());
        let name = dir.file_name().unwrap().to_str().unwrap().to_string();
        let mut files = Vec::new();
        for file in std::fs::read_dir(&dir).expect("a package directory") {
            let path = file.expect("a package file").path();
            let file = path.file_name().unwrap().to_str().unwrap().to_string();
            if file.ends_with(".star") {
                println!("cargo:rerun-if-changed={}", path.display());
                let path = path.canonicalize().expect("a package file's path");
                files.push((file, path.display().to_string()));
            }
        }
        files.sort();
        packages.push((name, files));
    }
    packages.sort();
    // The libraries every package may load, under `//lib/`.
    let mut libraries = Vec::new();
    let lib = root.join("lib");
    if lib.is_dir() {
        println!("cargo:rerun-if-changed={}", lib.display());
        walk(&lib, "//lib/", &mut libraries);
    }
    libraries.sort();
    for (_, files) in &mut packages {
        files.extend(libraries.iter().cloned());
    }
    let mut out = String::from("pub static PACKAGES: &[(&str, &[(&str, &str)])] = &[\n");
    for (name, files) in &packages {
        writeln!(out, "    ({name:?}, &[").unwrap();
        for (file, path) in files {
            writeln!(out, "        ({file:?}, include_str!({path:?})),").unwrap();
        }
        out.push_str("    ]),\n");
    }
    out.push_str("];\n");
    let dest = Path::new(&std::env::var("OUT_DIR").unwrap()).join("packages.rs");
    std::fs::write(dest, out).expect("the embedded packages' table");
}

fn walk(dir: &Path, prefix: &str, out: &mut Vec<(String, String)>) {
    for entry in std::fs::read_dir(dir).expect("a library directory") {
        let path = entry.expect("a library file").path();
        let file = path.file_name().unwrap().to_str().unwrap().to_string();
        if path.is_dir() {
            println!("cargo:rerun-if-changed={}", path.display());
            walk(&path, &format!("{prefix}{file}/"), out);
        } else if file.ends_with(".star") {
            println!("cargo:rerun-if-changed={}", path.display());
            let path = path.canonicalize().expect("a library file's path");
            out.push((format!("{prefix}{file}"), path.display().to_string()));
        }
    }
}
