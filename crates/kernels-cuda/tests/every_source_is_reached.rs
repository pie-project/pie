//! Every file this crate carries into the binary is reached by something.
//!
//! The third of the family, after `kernels-vulkan/tests/every_source_is_reached.rs`
//! and `kernels-wgpu`'s twin of it, and it exists for the same reason at a
//! larger scale: `build.rs` walks the whole `kernels/` tree and emits an
//! `include_str!` row for every file it finds, under every spelling anything
//! uses to name it. Nothing in that walk asks whether the file has a reader.
//! An orphan therefore costs one `Header` row, warns about nothing, and goes on
//! being cited by its neighbours as though it were live.
//!
//! # What it found
//!
//! `attn/pack_dense_mask.cuh` -- 241 lines holding two non-template
//! `__global__`s that packed the element attention bitmap on the device. The
//! driver packs that bitmap on the HOST instead, in
//! `driver-cuda/src/fire/page_mask.rs`, which owns `packed_len`, the
//! `mask[base + (index >> 3)] |= 1 << (index & 7)` write, and a test that reads
//! the result back through the same `bit >> 3` arithmetic the kernels use. So
//! nothing included the header and no `Fire::at` named it.
//!
//! The prose was the expensive part. The header's longest section argued that a
//! second Rust definition of a three-`u32` POD was acceptable *because* the
//! mirror was checked -- "`pack_dense_mask.cu` includes BOTH definitions and
//! `static_assert`s size, alignment and all three field offsets against each
//! other", against a `StructuredMaskParams` declared in `pack_dense_mask.hpp`.
//! Neither file was in the tree. The `.cu`/`.hpp` pair went with the archive
//! crate's ahead-of-time build when the tree moved to NVRTC, and what survived
//! was a paragraph describing a compile-time check that no compiler had run in
//! a long time, in a file no compiler read either.
//!
//! Deleting it took the whole chain with it: `kernels-cuda::attn::params` (the
//! mirrored struct, its `Abi` impl, five `const _: () = assert!` layout checks
//! and a `LAYOUTS` row pointing at a `nvrtc-probes/attn_structured_mask.py`
//! that is also not in the tree), `driver-cuda::bind::abi`'s re-export of it,
//! and `kernels::Ty::StructuredMasks` with its four match arms.
//!
//! # What "reached" means here
//!
//! Two ways in, and unlike the WGSL backends there is no in-file marker:
//!
//!  * another carried file `#include`s it, by a spelling that resolves either
//!    beside the including file or from the tree root -- both are real, because
//!    the upstream trees moved in intact and still say `../cp_async.cuh` from
//!    `attention/decode.cuh` and bare `cp_async.cuh` from a sibling, and
//!    `build.rs` carries a row per spelling precisely because NVRTC resolves
//!    nothing itself;
//!  * Rust names it as a compilation ROOT -- `Fire::at("attn/kv_paged.cuh")`,
//!    or a `CONFIGURED` row in `jit/root.rs`. Both are ordinary string literals
//!    under `src/`, which is what this walks for.
//!
//! # The three exemptions are not source
//!
//! `flashinfer/LICENSE`, `flashinfer/MODIFICATIONS` and `xqa/LICENSE` are
//! carried by the same indiscriminate walk and are unreachable by construction:
//! nothing may `#include` a licence. They are the reason the vendored trees can
//! be here at all, and the `MODIFICATIONS` note is the record of what was
//! changed in one -- deleting either to satisfy a reachability rule would be
//! exactly backwards. They are named individually rather than matched by
//! pattern so that a fourth one has to be argued for.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Carried, but not source, and not deletable. See the module doc.
const NOTICES: &[&str] =
    &["flashinfer/LICENSE", "flashinfer/MODIFICATIONS", "xqa/LICENSE"];

/// Quoted spellings that NVRTC itself supplies, so no `Header` row backs them.
///
/// A quoted include normally means "this tree's", and that is why the second
/// test exists at all. These three are the exception the tree already argues
/// for at length: `prelude/fp8.cuh` opens with a section titled *"why this file
/// is not just `#include \"cuda_fp8.h\"`"*, and
/// `attn/attention_naive_paged.cuh` names them as "three of the seven spellings
/// §15 makes resolve to NVIDIA's headers under nvcc and to this tree's own
/// under NVRTC". They are quoted rather than angled because that is how the
/// upstream sources that reach them were written, and rewriting an upstream
/// byte is the one thing the vendoring rule forbids.
const TOOLCHAIN: &[&str] = &["cuda_fp16.h", "cuda_fp8.h", "cuda_bf16.h"];

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every file under `dir`, as a `/`-spelled path relative to `base`.
fn walk(dir: &Path, base: &Path, out: &mut Vec<String>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("a directory entry").path();
        if path.is_dir() {
            walk(&path, base, out);
        } else {
            let rel = path.strip_prefix(base).expect("walked from base");
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
}

/// The `#include "…"` targets in one file, verbatim.
///
/// Angled includes are deliberately skipped: those are the toolchain's
/// (`<cuda_fp8.h>`), never this tree's, and the headers that document which
/// ones NVRTC resolves say so at length.
fn quoted_includes(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| {
            let rest = line.trim_start().strip_prefix('#')?.trim_start();
            let rest = rest.strip_prefix("include")?.trim_start();
            let rest = rest.strip_prefix('"')?;
            let (target, _) = rest.split_once('"')?;
            Some(target.to_string())
        })
        .collect()
}

/// `a/b/../c` -> `a/c`, lexically. Not `canonicalize`, which would need the
/// file to exist -- and whether it exists is the question.
fn normalise(spelling: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for part in spelling.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    parts.join("/")
}

/// Both resolutions of one include spelling: beside the includer, and from the
/// tree root.
fn candidates(from: &str, spelling: &str) -> [String; 2] {
    let dir = from.rsplit_once('/').map_or("", |(d, _)| d);
    let beside =
        if dir.is_empty() { spelling.to_string() } else { format!("{dir}/{spelling}") };
    [normalise(&beside), normalise(spelling)]
}

/// Every `"…"` literal under `src/` that names a file in the kernel tree.
///
/// A root is spelled as a plain string in exactly two places -- `Fire::at` and
/// the `CONFIGURED` table -- and both are string literals, so one sweep for
/// literals that happen to be carried paths finds them without this test having
/// to track which call sites exist.
fn named_from_rust(carried: &BTreeSet<String>) -> BTreeSet<String> {
    let src_root = crate_root().join("src");
    let mut src = Vec::new();
    walk(&src_root, &src_root, &mut src);

    let mut named = BTreeSet::new();
    for rel in src.iter().filter(|r| r.ends_with(".rs")) {
        let text = std::fs::read_to_string(src_root.join(rel)).expect("read a source file");
        for chunk in text.split('"').skip(1).step_by(2) {
            if carried.contains(chunk) {
                named.insert(chunk.to_string());
            }
        }
    }
    named
}

#[test]
fn no_file_is_carried_that_nothing_includes_and_nothing_fires() {
    let kernels = crate_root().join("kernels");
    let mut files = Vec::new();
    walk(&kernels, &kernels, &mut files);
    assert!(files.len() > 100, "the tree shrank to {} files -- is the walk right?", files.len());

    let carried: BTreeSet<String> = files.iter().cloned().collect();

    let mut reached = named_from_rust(&carried);
    for rel in &files {
        let Ok(text) = std::fs::read_to_string(kernels.join(rel)) else {
            continue;
        };
        for spelling in quoted_includes(&text) {
            for candidate in candidates(rel, &spelling) {
                if carried.contains(&candidate) {
                    reached.insert(candidate);
                }
            }
        }
    }

    let orphans: Vec<&String> = files
        .iter()
        .filter(|rel| !reached.contains(*rel) && !NOTICES.contains(&rel.as_str()))
        .collect();

    assert!(
        orphans.is_empty(),
        "carried but unreachable -- no file includes them and no Rust literal \
         names them as a compilation root. Delete them, or include them from \
         the header whose ABI they state: {orphans:?}"
    );
}

/// The exemptions still exist, so the list cannot quietly become fiction.
///
/// Without this, deleting `flashinfer/MODIFICATIONS` would leave a `NOTICES`
/// entry that reads as a live claim about the tree and excuses nothing -- the
/// same shape of stale prose the first test was written to catch.
#[test]
fn the_exempt_notices_are_still_there() {
    let kernels = crate_root().join("kernels");
    for notice in NOTICES {
        assert!(
            kernels.join(notice).is_file(),
            "{notice} is exempted from the reachability rule but is not in the \
             tree -- drop the exemption"
        );
    }
}

/// The other half of the rule: an `#include` naming a file the tree does not
/// carry, and that nothing outside it supplies either.
///
/// NVRTC resolves NOTHING. Every `"…"` include has to arrive as a named
/// `Header` row or the compile fails at run time, on a device, in whichever
/// arm first asks for that root -- which may be a tier this machine does not
/// have. This says it for every file at once with no device involved.
#[test]
fn every_quoted_include_resolves_inside_the_tree() {
    let kernels = crate_root().join("kernels");
    let mut files = Vec::new();
    walk(&kernels, &kernels, &mut files);
    let carried: BTreeSet<String> = files.iter().cloned().collect();

    let mut dangling = Vec::new();
    for rel in &files {
        // A notice is prose, and `flashinfer/MODIFICATIONS` quotes the diffs it
        // is describing -- including their `#include` lines. Reading those as
        // this tree's own includes would be reading a changelog as source.
        if NOTICES.contains(&rel.as_str()) {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(kernels.join(rel)) else {
            continue;
        };
        for spelling in quoted_includes(&text) {
            if TOOLCHAIN.contains(&spelling.as_str()) {
                continue;
            }
            if !candidates(rel, &spelling).iter().any(|c| carried.contains(c)) {
                dangling.push(format!("{rel} includes {spelling}"));
            }
        }
    }

    assert!(dangling.is_empty(), "includes that resolve to nothing carried: {dangling:?}");
}
