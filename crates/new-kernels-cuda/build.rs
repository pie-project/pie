//! Generates the carried-header table into `OUT_DIR` from the `.cuh` and
//! shim trees. Generated rather than checked in because a stale checked-in
//! list still compiles; generated into `OUT_DIR`, it cannot be stale by
//! construction.
//!
//! MENLO-SEAM: the trees still live in the OLD crate (`crates/kernels-cuda/
//! {kernels,shim}`), because two copies of the device text are two contracts
//! waiting to disagree and the old plane is still firing them. When the old
//! crates are deleted (design "Porting order" step 6), the trees move here
//! and `TREES` drops its `../kernels-cuda` prefix — a one-line change.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const TREES: [(&str, &str); 2] = [
    ("kernels", "../kernels-cuda/kernels"),
    ("shim", "../kernels-cuda/shim"),
];

const UPSTREAM_ROOTS: [&str; 1] = ["flashinfer"];

fn is_header(path: &Path) -> bool {
    !matches!(
        path.file_name().and_then(|n| n.to_str()),
        Some("LICENSE" | "MODIFICATIONS" | "README.md")
    )
}

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

fn spell(rel: &Path) -> String {
    rel.components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

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
    for (_, tree) in TREES {
        println!("cargo:rerun-if-changed={tree}");
    }

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets this"));
    let [(_, kernels_tree), (_, shim_tree)] = TREES;
    let (kernels, shim) = (manifest.join(kernels_tree), manifest.join(shim_tree));

    let mut kernel_files = Vec::new();
    walk(&kernels, &kernels, &mut kernel_files);
    let mut shim_files = Vec::new();
    walk(&shim, &shim, &mut shim_files);

    let upstream = |rel: &Path| {
        rel.components()
            .next()
            .and_then(|c| c.as_os_str().to_str())
            .is_some_and(|first| UPSTREAM_ROOTS.contains(&first))
    };

    let mut named: BTreeMap<String, (&str, PathBuf)> = BTreeMap::new();
    for rel in &shim_files {
        named.insert(spell(rel), (shim_tree, rel.clone()));
    }
    for rel in &kernel_files {
        named.insert(spell(rel), (kernels_tree, rel.clone()));
    }

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
        if *root == shim_tree {
            shim_rows.push_str(&line);
        } else if upstream(rel) {
            up_rows.push_str(&line);
        } else {
            lib_rows.push_str(&line);
        }
    }
    for (spelling, target) in &extra {
        let line = row(spelling, kernels_tree, target);
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
            "// GENERATED by build.rs from the carried `.cuh` and shim trees. Do not edit.\n\
             /// The impersonation layer: headers wearing NVIDIA's and the standard\n\
             /// library's filenames, carried because the source that reaches for them\n\
             /// is source we do not own and the spelling is the contract.\n\
             #[rustfmt::skip]\n\
             pub const SHIM: &[Header] = &[\n{shim_rows}];\n\n\
             /// The plane's own device text: every `__global__` template a unit\n\
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
