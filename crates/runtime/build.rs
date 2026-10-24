//! Embeds the repository's `models/`: every package and the `lib/` libraries.

use std::fmt::Write as _;
use std::path::Path;

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    println!("cargo:rerun-if-changed={}", root.display());
    let mut packages: Vec<(String, Vec<(String, String)>)> = Vec::new();
    for dir in std::fs::read_dir(&root).expect("the repository's models/") {
        let dir = dir.expect("a models/ entry").path();
        // A package is a `.poem` file, or a directory of them, `lib/`
        // excepted.
        if dir.is_file() && dir.extension().is_some_and(|e| e == "poem") {
            println!("cargo:rerun-if-changed={}", dir.display());
            let name = dir.file_stem().unwrap().to_str().unwrap().to_string();
            let path = dir.canonicalize().expect("a package file's path");
            packages.push((name.clone(), vec![(format!("{name}.poem"), path.display().to_string())]));
            continue;
        }
        let poem = |entry: std::fs::DirEntry| entry.path().extension().is_some_and(|e| e == "poem");
        let is_package = dir.is_dir()
            && dir.file_name().is_some_and(|name| name != "lib")
            && std::fs::read_dir(&dir).is_ok_and(|entries| entries.flatten().any(poem));
        if !is_package {
            continue;
        }
        println!("cargo:rerun-if-changed={}", dir.display());
        let name = dir.file_name().unwrap().to_str().unwrap().to_string();
        let mut files = Vec::new();
        for file in std::fs::read_dir(&dir).expect("a package directory") {
            let path = file.expect("a package file").path();
            let file = path.file_name().unwrap().to_str().unwrap().to_string();
            if file.ends_with(".poem") {
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
    let mut out = String::from("pub static PACKAGES: &[(&str, &[(&str, &str)])] = &[\n");
    for (name, files) in &packages {
        writeln!(out, "    ({name:?}, &[").unwrap();
        for (file, path) in files {
            writeln!(out, "        ({file:?}, include_str!({path:?})),").unwrap();
        }
        out.push_str("    ]),\n");
    }
    out.push_str("];\n");
    out.push_str("pub static LIBRARY: &[(&str, &str)] = &[\n");
    for (file, path) in &libraries {
        writeln!(out, "    ({file:?}, include_str!({path:?})),").unwrap();
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
        } else if file.ends_with(".poem") {
            println!("cargo:rerun-if-changed={}", path.display());
            let path = path.canonicalize().expect("a library file's path");
            out.push((format!("{prefix}{file}"), path.display().to_string()));
        }
    }
}
