//! Keeps the checked-in `ptir_abi.h` byte-identical to the generator (the op
//! table is the single source of truth; the header is a projection).
//! Regenerate with
//! `PTIR_REGEN=1 cargo test -p pie-compiler-tests --test ptir_header`.
//!
//! Two copies are checked in: the canonical one under `compiler/codegen/include`
//! and the mirror the native drivers actually `#include` (their include path is
//! rooted at `driver/common/include`). Both are written from the same string, so
//! the mirror cannot drift — it used to be maintained by hand and nothing
//! verified it. The mirror goes away when the drivers stop carrying their own
//! PTIR decode headers.

use std::path::{Path, PathBuf};

use pie_codegen::header::generate_c_header;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn check_or_regenerate(path: &Path, expected: &str) {
    if std::env::var("PTIR_REGEN").is_ok() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, expected).unwrap();
        return;
    }
    let on_disk = std::fs::read_to_string(path).unwrap_or_else(|error| {
        panic!(
            "{} missing ({error}) — run with PTIR_REGEN=1 to generate",
            path.display()
        )
    });
    assert_eq!(
        on_disk,
        expected,
        "{} is stale — regenerate with PTIR_REGEN=1 cargo test -p pie-compiler-tests --test ptir_header",
        path.display()
    );
}

#[test]
fn ptir_header_uptodate() {
    let root = repo_root();
    let expected = generate_c_header();
    check_or_regenerate(&root.join("compiler/codegen/include/ptir_abi.h"), &expected);
}
