//! Keeps the checked-in `ptir_abi.h` byte-identical to the generator (the op
//! table is the single source of truth; the header is a projection).
//! Regenerate with
//! `PTIR_REGEN=1 cargo test -p pie-compiler-tests --test ptir_header`.
//!
//! One copy is checked in, under `compiler/codegen/include`; the native drivers
//! put that directory on their include path directly. The hand-maintained
//! driver mirror is gone.

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

/// No C++ header may re-type a value the generated header already owns.
///
/// A re-typed tag is a whole bug class, not a one-off: `iota` written `0x31`
/// where the table says `0x64`, `const` `0x30` for `0x81`, `PTIR_INTR_LAYER`
/// `4` for `5`. Each compiles, each is a plausible-looking constant, and each
/// mis-dispatches only for the one op it names.
///
/// So the discipline needs its own gate: a driver header that spells a port,
/// stage or intrinsic tag as a bare
/// integer literal has forked from `pie_ir::registry` and nothing else will say so.
#[test]
fn drivers_do_not_retype_generated_tags() {
    let root = repo_root();
    // (file, constant-name fragment, the generated prefix it must derive from)
    let guarded = [
        (
            "driver/common/include/pie/driver/fire/descriptor.hpp",
            "kPort",
            "PTIR_PORT_",
        ),
        (
            "driver/common/include/pie/driver/launch/program.hpp",
            "StageKind",
            "PTIR_STAGE_",
        ),
    ];
    for (relative, fragment, prefix) in guarded {
        let path = root.join(relative);
        let Ok(text) = std::fs::read_to_string(&path) else {
            // Deleted by a later phase of the refactor; the gate retires with it.
            continue;
        };
        let offenders: Vec<&str> = text
            .lines()
            .filter(|line| line.contains(fragment))
            .filter(|line| {
                // A definition assigning a bare decimal literal, rather than
                // naming the generated constant.
                line.contains('=')
                    && !line.contains(prefix)
                    && line.split('=').nth(1).is_some_and(|rhs| {
                        rhs.trim_start().starts_with(|c: char| c.is_ascii_digit())
                    })
            })
            .collect();
        assert!(
            offenders.is_empty(),
            "{relative} re-types {prefix}* instead of including <ptir_abi.h>:\n{}",
            offenders.join("\n")
        );
    }
}
