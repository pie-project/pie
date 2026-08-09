//! Layers point down.
//!
//! # Why this exists
//!
//! `.wiki/driver/real-metal-north-star.md` §9 asks for a tree with a
//! direction, and the tree had two cycles instead:
//!
//! * `gpu/device/recording.rs` imported `lowering::dispatch::Dispatch` while
//!   `gpu/fire/run.rs` imported `Recordings`. The ICB path knew what a fire
//!   was, so recording could not be understood, tested or replaced without
//!   the model layer in scope.
//! * `gpu/device/ring.rs` imported `gpu/program/channel.rs` while ten files
//!   under `gpu/program/` imported `gpu/device/`. That one was worse for
//!   reading badly: `channel.rs` is `pub use driver::*` and nothing else, so
//!   the arrow pointed at an external crate's ABI and *looked* like a
//!   dependency on the shader compiler.
//!
//! Both are gone — `Command` for the first, moving `channel` to the crate
//! root for the second. This test is what keeps them gone. A direction that
//! is only written down is restored by whoever next needs one symbol from
//! one layer up, and the `use` line that does it is two words in a diff.
//!
//! # What it does not check
//!
//! Cycles in general. This names ONE edge — out of `gpu/device/`, upward —
//! because that is the edge the crate has actually paid for twice, and a
//! general cycle detector over `use` lines is a resolver, which is the
//! compiler's job. If a third layer starts costing something, add its rule
//! here rather than generalising this one into something nobody reads.
//!
//! # Why it is not gated
//!
//! It reads files. `gpu/device/*.rs` is on disk whether or not `metal-4`
//! compiles it, so this runs in the portable half too and the no-GPU CI job
//! catches a re-introduced cycle without a Mac. That is the same reason it
//! is absent from the `required-features` list in `Cargo.toml`: absence is
//! this crate's claim that a target needs no device, and this one does not.

use std::path::Path;

/// Layers above `gpu/device/`, spelled as they appear in a `use`.
///
/// `gpu/device/` is the bottom of the Metal half: a context, memory, an
/// encoder, a queue. Everything here is something built ON that.
const ABOVE: &[&str] = &[
    "crate::lowering",
    "crate::gpu::bind",
    "crate::gpu::fire",
    "crate::gpu::pools",
    "crate::gpu::program",
    "crate::gpu::serve",
    "crate::gpu::weights",
];

/// The one upward import that is allowed, and why.
///
/// `Keepalive` compiles a spin kernel. Concurrent Metal pipeline compilation
/// corrupts the process heap, so every compilation in this crate goes
/// through one process-wide mutex, and the only way in is `Compiler`. Doing
/// it any other way would be a second compiler racing a model load — the
/// dependency is not an accident of layout, it is the safety property.
///
/// The honest reading is that `keepalive` is a subsystem shelved under
/// `device/`, not a device primitive. Moving it is a separate change; until
/// then this is the exception, named, with the reason attached.
const ALLOWED: &[(&str, &str)] = &[("keepalive.rs", "crate::gpu::program::compile::Compiler")];

#[test]
fn gpu_device_does_not_import_the_layers_built_on_it() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/device");
    let mut scanned = 0usize;
    let mut violations = Vec::new();

    for entry in std::fs::read_dir(&dir).expect("gpu/device is a directory") {
        let path = entry.expect("a readable entry").path();
        if path.extension().is_none_or(|e| e != "rs") {
            continue;
        }
        let name = path
            .file_name()
            .expect("a file name")
            .to_string_lossy()
            .into_owned();
        let body = std::fs::read_to_string(&path).expect("a readable file");
        scanned += 1;

        // `mod.rs` re-exports the tree it is the root of; naming a sibling
        // there is the declaration, not a dependency.
        if name == "mod.rs" {
            continue;
        }

        for (number, line) in body.lines().enumerate() {
            let line = line.trim();
            if !line.starts_with("use ") {
                continue;
            }
            let Some(layer) = ABOVE.iter().find(|l| line.contains(**l)) else {
                continue;
            };
            if ALLOWED
                .iter()
                .any(|(file, import)| *file == name && line.contains(import))
            {
                continue;
            }
            violations.push(format!(
                "gpu/device/{name}:{} imports {layer}\n    {line}",
                number + 1
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "gpu/device/ is the bottom of the Metal half, and these point up:\n\n{}\n\n\
         Either the thing being reached for belongs lower (move it, as \
         `channel.rs` moved to the crate root), or what `device/` needs is a \
         narrower type it can own (as `recording` took `Command` instead of \
         `Dispatch`). Adding an entry to `ALLOWED` is the third option and \
         costs a written reason.",
        violations.join("\n")
    );

    // And the audit has to be able to fail. A scan that reads no files
    // passes, silently, forever -- the exact failure `mod_audit.rs` exists
    // for, one directory over.
    assert!(
        scanned > 15,
        "the scan read only {scanned} files under gpu/device/, which means it \
         is not scanning -- a broken audit passes"
    );
}

/// The crate root names the portable half, and `gpu` — nothing else.
///
/// # Why this is the same rule as the one above
///
/// A facade is only a facade if the alternative is not also public. Sixty-five
/// device types used to be re-exported flat from `lib.rs`, so `Stepper` had
/// two paths — `driver_metal::Stepper` and `driver_metal::gpu::Stepper` — and
/// the engine could reach past `gpu::serve::Shell` into any of them without
/// the compiler saying a word. That is §5's "one concept, two names" happening
/// inside a single crate, and §9's "everything else goes private" is the fix.
///
/// The list is short enough to read, which is the point. Adding to it is
/// allowed and costs a line in a diff that a reviewer will see, where adding
/// to a sixty-five-name flat list did not.
#[test]
fn the_crate_root_exposes_the_portable_half_and_one_door() {
    // What `lib.rs` may re-export at the root. Every one answers a question
    // no GPU changes, so it is the same on a machine with no Metal at all.
    const PORTABLE_ROOT: &[&str] = &[
        "error::{Error, Result}",
        "facts::{ModelFacts, ModelFamily}",
        "layout::{Batch, Region, Request}",
    ];

    let source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs"),
    )
    .expect("lib.rs is beside this test");

    let found: Vec<String> = source
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("pub use "))
        .map(|line| {
            line.trim_start_matches("pub use ")
                .trim_end_matches(';')
                .to_string()
        })
        .collect();

    let unexpected: Vec<&String> = found
        .iter()
        .filter(|item| !PORTABLE_ROOT.contains(&item.as_str()))
        .collect();

    assert!(
        unexpected.is_empty(),
        "the crate root re-exports {unexpected:?}, which is not part of the \
         portable half.\n\nThe device half is reached through `gpu::`, and \
         only through it: a second path to a type is a way around the facade \
         that the compiler cannot see. If this genuinely belongs at the root, \
         add it to PORTABLE_ROOT here — that is a line a reviewer reads, which \
         appending to a flat re-export list was not."
    );

    // The audit has to be able to fail, same as the one above: a `lib.rs`
    // that stopped matching would give an empty list and pass.
    assert_eq!(
        found.len(),
        PORTABLE_ROOT.len(),
        "expected {} root re-exports and read {} — the scan is not reading \
         lib.rs",
        PORTABLE_ROOT.len(),
        found.len()
    );
}
