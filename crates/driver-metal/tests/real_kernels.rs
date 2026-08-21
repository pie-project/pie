//! The include splicer against the shaders it actually ships with.
//!
//! `kernels-metal/kernels/` is the directory the driver hands to Metal's
//! runtime compiler, so it is the only input that matters. These run on any
//! host: splicing is text, and the .metal files are checked in.

use std::path::{Path, PathBuf};

fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/driver-metal has a parent")
        .join("kernels-metal/kernels")
}

/// Every `.metal` under `dir`, recursively.
///
/// Recursive because the kernels grew subject directories (`norm/`,
/// `sample/`, `ssm/`, ...) and a flat scan of the root silently became a
/// scan of nothing — these tests were red for "no .metal found" while
/// thirty shaders sat one level down.
fn shaders(dir: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("kernels dir is readable") {
            let path = entry.expect("kernels dir entry").path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "metal") {
                out.push(path);
            }
        }
    }
    let mut v = Vec::new();
    walk(dir, &mut v);
    v.sort();
    v
}

#[test]
fn every_shipped_shader_splices() {
    let dir = kernels_dir();
    let files = shaders(&dir);
    assert!(!files.is_empty(), "no .metal found under {}", dir.display());

    for path in &files {
        let out = driver_metal::layout::shader::read_source(path)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert!(
            !out.contains("#include \""),
            "{}: a quoted include survived splicing",
            path.display()
        );
    }
}

/// The splice has to actually bring the definitions in, not just delete the
/// directive. `norm/rms.metal` includes `rms_reduce.h`, so a symbol declared
/// only in the header must appear in the spliced text.
///
/// IT WAS `rms_params.h`, WHICH NO SHADER INCLUDES ANY MORE. The packed
/// parameter blocks were unwound -- every field is a `const constant uint&`
/// of its own now -- and `rms_params.h` went with them, except that the FILE
/// stayed on disk. So this read a header nothing splices, looked for its
/// `struct` in a text that never asked for it, and failed on a tree that is
/// correct. A splice test has to name a header something actually includes,
/// which is what makes `#include "` the thing it looks up rather than a
/// filename written down here.
///
/// The marker is no longer a `struct`: `rms_reduce.h` declares FUNCTIONS, and
/// a definition is a definition whichever keyword introduces it.
#[test]
fn splicing_brings_the_header_body_in() {
    let dir = kernels_dir().join("norm");
    let root = dir.join("rms.metal");
    assert!(
        root.exists(),
        "norm/rms.metal moved again; point this test at a shader with a quoted include"
    );
    let raw = std::fs::read_to_string(&root).expect("the shader is readable");
    assert!(
        raw.contains("#include \"rms_reduce.h\""),
        "norm/rms.metal stopped including rms_reduce.h; point this test at a shader that \
         still has a quoted include, and delete the header if nothing includes it"
    );
    let header = std::fs::read_to_string(dir.join("rms_reduce.h")).expect("header is readable");
    let marker = header
        .lines()
        .find(|l| {
            l.starts_with("struct ") || l.starts_with("typedef ") || l.contains("METAL_FUNC ")
        })
        .map(str::trim);
    assert!(
        marker.is_some(),
        "rms_reduce.h declares nothing this test can look for in the spliced text"
    );

    let out = driver_metal::layout::shader::read_source(&root).expect("splices");
    if let Some(marker) = marker {
        assert!(out.contains(marker), "spliced text lost `{marker}`");
    }
    assert!(
        out.len() > header.len(),
        "spliced text is smaller than the header it pulled in"
    );
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
        let out = driver_metal::layout::shader::read_source(&path).expect("splices");
        // At least: a spliced quoted header brings its own system includes
        // with it (the MLX steel fragments each pull <metal_stdlib>), so the
        // count may grow. What must never happen is a system include being
        // dropped.
        assert!(
            out.matches("#include <").count() >= want,
            "{}: a system include was dropped by splicing",
            path.display()
        );
    }
}
