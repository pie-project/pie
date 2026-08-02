//! The include splicer against the shaders it actually ships with.
//!
//! `kernels-metal/kernels/` is the directory the driver hands to Metal's
//! runtime compiler, so it is the only input that matters. These run on any
//! host: splicing is text, and the .metal files are checked in.

use std::path::{Path, PathBuf};

fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/driver-metal-new has a parent")
        .join("kernels-metal/kernels")
}

fn shaders(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("kernels dir is readable")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "metal"))
        .collect();
    v.sort();
    v
}

#[test]
fn every_shipped_shader_splices() {
    let dir = kernels_dir();
    let files = shaders(&dir);
    assert!(!files.is_empty(), "no .metal found under {}", dir.display());

    for path in &files {
        let out = driver_metal_new::shader::read_source(path)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert!(
            !out.contains("#include \""),
            "{}: a quoted include survived splicing",
            path.display()
        );
    }
}

/// The splice has to actually bring the definitions in, not just delete the
/// directive. `rms_norm.metal` includes `rms_params.h`, so a symbol declared
/// only in the header must appear in the spliced text.
#[test]
fn splicing_brings_the_header_body_in() {
    let dir = kernels_dir();
    let root = dir.join("rms_norm.metal");
    if !root.exists() {
        return;
    }
    let header = std::fs::read_to_string(dir.join("rms_params.h")).expect("header is readable");
    let marker = header
        .lines()
        .find(|l| l.starts_with("struct ") || l.starts_with("typedef "))
        .map(str::trim);

    let out = driver_metal_new::shader::read_source(&root).expect("splices");
    if let Some(marker) = marker {
        assert!(out.contains(marker), "spliced text lost `{marker}`");
    }
    assert!(out.len() > header.len(), "spliced text is smaller than the header it pulled in");
}

/// Angle-bracket includes are Metal's own headers and must survive: dropping
/// `<metal_stdlib>` would compile to a wall of unknown-identifier errors.
#[test]
fn system_includes_survive() {
    for path in shaders(&kernels_dir()) {
        let raw = std::fs::read_to_string(&path).expect("readable");
        let want = raw.matches("#include <").count();
        if want == 0 {
            continue;
        }
        let out = driver_metal_new::shader::read_source(&path).expect("splices");
        assert_eq!(
            out.matches("#include <").count(),
            want,
            "{}: system includes changed count",
            path.display()
        );
    }
}
