//! Write `entrypoints.generated.txt` — the census a reviewer diffs.
//!
//! `kernels-vulkan` produces its copy with `scripts/vulkan-kernel-audit.py
//! --write`, because over there the census is a fact about the SHADER TREE that
//! only a preprocessor can read, and the Rust table is the thing being checked
//! against it.
//!
//! Here it is the other way round and simpler. The tree's directives are
//! parsed by a library function, so `tests/entrypoints.rs` already compares
//! tree and table directly and needs no file to do it. What the file is for is
//! REVIEW: three backends each commit one, and the useful question — "did this
//! change move the coverage, and away from which sibling?" — is a set
//! difference in a diff rather than a number in a test log.
//!
//! So this writes the TABLE's product, and the test checks the file against it.
//! Generating it in `build.rs` instead was considered and rejected: a file the
//! build rewrites silently is a file nobody reads, and its whole value is being
//! read.
//!
//! ```sh
//! cargo run -p kernels-wgpu --example write_census
//! ```

fn main() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("entrypoints.generated.txt");

    let mut text = kernels_wgpu::entrypoints().join("\n");
    text.push('\n');

    std::fs::write(&path, &text).unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));

    eprintln!(
        "wrote {} entrypoints over {} kernels to {}",
        kernels_wgpu::entrypoints().len(),
        kernels_wgpu::KERNELS.len(),
        path.display(),
    );
}
