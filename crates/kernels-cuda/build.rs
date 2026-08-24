//! The header set, walked instead of written.
//!
//! NVRTC does no path resolution: it matches `includeNames[]` against the
//! literal string in an `#include` directive, so what a compile needs is not a
//! list of FILES but a list of SPELLINGS. `include_str!` takes a literal path,
//! which is why the three lists in `src/source.rs` were 187 lines maintained
//! under a rule a person had to remember -- *"adding a file under `kernels/` or
//! `shim/` means adding a line to one of these"*.
//!
//! Both halves are on disk. This walks the two directories for the files and
//! scans their own `#include` directives for the spellings, and emits the same
//! three `const` slices into `OUT_DIR`.
//!
//! # Why a build script and not a proc macro
//!
//! `rerun-if-changed`. A proc macro can read a directory at expansion time and
//! nothing tells `cargo` to expand it again when a file APPEARS -- which is
//! precisely the omission the hand-written list existed to prevent, returned
//! silently one layer down. A build script declares the dependency.
//!
//! # What this catches that the list could not
//!
//! `src/source.rs`'s own doc admitted two gaps: *"a file nothing includes yet,
//! or one reached only by an angled spelling"*. The first is gone -- a file is
//! carried because it EXISTS, not because something reached it. The second is
//! unchanged and still belongs to `every_device_include_resolves`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// The upstream subtrees, which a `norm` compile must not carry: `nvrtcCreateProgram`
/// copies every byte it is given.
const UPSTREAM_ROOTS: [&str; 1] = ["flashinfer"];

/// Carried files are device text. A licence is not.
fn is_header(path: &Path) -> bool {
    !matches!(
        path.file_name().and_then(|n| n.to_str()),
        Some("LICENSE" | "MODIFICATIONS" | "README.md")
    )
}

/// Every file under `dir`, as paths relative to it.
fn walk(dir: &Path, base: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut sorted: Vec<_> = entries.filter_map(Result::ok).map(|e| e.path()).collect();
    sorted.sort();
    for path in sorted {
        if path.is_dir() {
            walk(&path, base, out);
        } else if is_header(&path) {
            out.push(
                path.strip_prefix(base)
                    .expect("walked from base")
                    .to_path_buf(),
            );
        }
    }
}

/// A path as an `#include` spells it: forward slashes, whatever the host is.
fn spell(rel: &Path) -> String {
    rel.components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

/// The quoted `#include`s of `source`, at column zero.
///
/// `src/source.rs::quoted_includes` is this same rule, and the two must agree:
/// that one validates the set at `cargo test`, this one builds it. Column zero
/// is what tells a directive from a string literal that contains one.
fn quoted_includes(source: &str) -> Vec<&str> {
    source
        .lines()
        .filter_map(|line| {
            let rest = line.strip_prefix("#include")?;
            let rest = rest.strip_prefix([' ', '\t'])?.trim_start();
            let rest = rest.strip_prefix('"')?;
            rest.split('"').next()
        })
        .collect()
}

/// `a/b/../c` -> `a/c`, without touching the filesystem.
///
/// Not `canonicalize`: the point is to answer what the SPELLING would resolve
/// to if NVRTC resolved anything, and a symlink would give a different answer
/// from the one the directive means.
fn normalise(base: &Path, spelling: &str) -> Option<PathBuf> {
    let mut out = base.to_path_buf();
    for part in spelling.split('/') {
        match part {
            "." | "" => {}
            ".." => {
                if !out.pop() {
                    return None;
                }
            }
            other => out.push(other),
        }
    }
    Some(out)
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-changed=shim");

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets this"));
    let (kernels, shim) = (manifest.join("kernels"), manifest.join("shim"));

    let mut kernel_files = Vec::new();
    walk(&kernels, &kernels, &mut kernel_files);
    let mut shim_files = Vec::new();
    walk(&shim, &shim, &mut shim_files);

    // WHICH LIST A FILE LANDS IN IS ITS DIRECTORY, which is the rule the three
    // hand-written lists already obeyed at every one of their 187 entries.
    let upstream = |rel: &Path| {
        rel.components()
            .next()
            .and_then(|c| c.as_os_str().to_str())
            .is_some_and(|first| UPSTREAM_ROOTS.contains(&first))
    };

    // The primary spelling of every carried file: its path minus the root it
    // was walked from. `name` -> (root, relative path).
    let mut named: BTreeMap<String, (&str, PathBuf)> = BTreeMap::new();
    for rel in &shim_files {
        named.insert(spell(rel), ("shim", rel.clone()));
    }
    for rel in &kernel_files {
        named.insert(spell(rel), ("kernels", rel.clone()));
    }

    // THE ALTERNATE SPELLINGS, which are the other half of what the lists said.
    // The upstream trees moved in INTACT -- that is why not one upstream byte
    // had to change -- so they still reach their siblings the way they always
    // did: `../cp_async.cuh` from `attention/decode.cuh`, and `cp_async.cuh`
    // bare from a file beside it. Neither is a typo, and NVRTC needs an entry
    // per spelling because it resolves nothing.
    //
    // An entry goes in the list of the file it NAMES, not of the file that
    // spelled it: that is what keeps a `norm` compile from carrying upstream.
    let mut extra: BTreeSet<(String, String)> = BTreeSet::new();
    for rel in &kernel_files {
        let text = std::fs::read_to_string(kernels.join(rel)).unwrap_or_default();
        let dir = rel.parent().unwrap_or(Path::new("")).to_path_buf();
        for spelling in quoted_includes(&text) {
            if named.contains_key(spelling) {
                continue;
            }
            let Some(target) = normalise(&dir, spelling) else {
                continue;
            };
            if kernels.join(&target).is_file() {
                extra.insert((spelling.to_owned(), spell(&target)));
            }
        }
    }

    let row = |name: &str, root: &str, rel: &str| {
        format!(
            "    Header {{ name: {name:?}, \
             text: include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/{root}/{rel}\")) }},\n"
        )
    };

    let (mut shim_rows, mut lib_rows, mut up_rows) = (String::new(), String::new(), String::new());
    for (name, (root, rel)) in &named {
        let line = row(name, root, &spell(rel));
        if *root == "shim" {
            shim_rows.push_str(&line);
        } else if upstream(rel) {
            up_rows.push_str(&line);
        } else {
            lib_rows.push_str(&line);
        }
    }
    for (spelling, target) in &extra {
        let line = row(spelling, "kernels", target);
        if upstream(Path::new(target)) {
            up_rows.push_str(&line);
        } else {
            lib_rows.push_str(&line);
        }
    }

    let out = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets this")).join("headers.rs");
    std::fs::write(
        &out,
        format!(
            "// GENERATED by build.rs from `kernels/` and `shim/`. Do not edit.\n\
             /// The impersonation layer: headers wearing NVIDIA's and the standard\n\
             /// library's filenames, carried because the source that reaches for them\n\
             /// is source we do not own and the spelling is the contract.\n\
             #[rustfmt::skip]\n\
             pub const SHIM: &[Header] = &[\n{shim_rows}];\n\n\
             /// This crate's own device text: every `__global__` template a unit\n\
             /// compiles and the prelude they are written over.\n\
             #[rustfmt::skip]\n\
             pub const LIBRARY: &[Header] = &[\n{lib_rows}];\n\n\
             /// The internalised FlashInfer and XQA closure, handed only to the units\n\
             /// that ask for it.\n\
             #[rustfmt::skip]\n\
             pub const UPSTREAM: &[Header] = &[\n{up_rows}];\n"
        ),
    )
    .expect("OUT_DIR is writable");
}
