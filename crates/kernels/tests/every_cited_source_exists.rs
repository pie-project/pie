//! A path named in prose is a claim, and it can be checked.
//!
//! Nearly every explanation in this tree is anchored to a shader: a routine
//! says which entrypoint it fires, a driver says which header a struct came
//! out of, a kernel says which sibling on another plane carries the same
//! layout. `Fire::at` is checked -- a symbol that names no source refuses at
//! plan time, and the conformance tests hold each dispatch list against the
//! `.metal`/`.slang`/`.wgsl` it names. A COMMENT is checked by nobody.
//!
//! So when a shader is renamed or deleted, the code that fired it moves and
//! the prose that described it does not. `lowering/consts.rs` replicated
//! seven structs "EXACTLY from the kernel headers `norm/rms_params.h`,
//! `moe/params.h`, `ssm/gdn_params.h`" for a long time after all three were
//! deleted, and `model-dsl` explained a `GdnShape` by saying "the shader
//! reads it as one struct" while `ssm/gdn_core.metal` opened with "THE
//! GEOMETRY IS ELEVEN SCALARS, NOT ONE `constant GdnCoreParams&`". Both
//! texts were confident, specific and wrong, and nothing could tell.
//!
//! ## What is a citation
//!
//! A `dir/file.ext` whose `dir` is one the kernel trees actually use. That
//! set is derived here rather than listed, so a new kernel directory is
//! covered by existing and an upstream path (MLX's `kernels/rms_norm.metal`,
//! a CUTLASS include) is not mistaken for one of ours.
//!
//! ## Naming something the trees do not carry is allowed
//!
//! Three good reasons to: the file was DELETED and the comment narrates its
//! going, which is half the value in this tree's history; it belongs to
//! UPSTREAM and the comment says where a number came from; or it is real and
//! STAGED from elsewhere, so the path is a deployment's and not a source
//! tree's. [`UNCARRIED`] is where each earns the right to be named -- with
//! what it is, so that an entry is a statement rather than a silencer.
//!
//! What no list can catch is naming a deleted file in the PRESENT tense.
//! What this one catches is naming one by ACCIDENT, and the entries below
//! are then the finite set whose tense a reader can check.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Paths the text may name that no kernel tree carries, each with what it is.
///
/// An entry is not an excuse to write in the present tense. It is a note for
/// the reader who greps for the path and finds nothing, so they land
/// somewhere instead of concluding the comment is a lie.
const UNCARRIED: &[(&str, &str)] = &[
    (
        "norm/rms_params.h",
        "The RMS family's parameter blocks, copied into `rms_params.slang` \
         and `rms_params.inc.wgsl`. Every entrypoint takes its five scalars \
         one `const constant uint&` at a time now; see the head of \
         `kernels-metal/kernels/norm/rms.metal`.",
    ),
    (
        "norm/rms_params.slang",
        "The Slang copy of the above. It outlived its last includer by long \
         enough to go on describing a block layout none of them had; \
         `kernels-vulkan/tests/every_source_is_reached.rs` is what found it.",
    ),
    (
        "moe/params.h",
        "`RouterParams`, `ExpertCombineParams` and `MoeRouteParams`. \
         `kernels-metal/kernels/moe/route.metal` states its going.",
    ),
    (
        "ssm/gdn_params.h",
        "The eleven-field `GdnCoreParams`, called by its own header the \
         \"shared host/shader ABI for every GDN core variant\" and copied \
         onto two other planes. `kernels-metal/kernels/ssm/gdn_core.metal` \
         states its going.",
    ),
    (
        "attn/pack_dense_mask.cuh",
        "Two non-template `__global__`s that packed the element attention \
         bitmap on the device, plus a `StructuredMaskParams` mirror justified \
         by `static_assert`s in a `pack_dense_mask.cu` that had already gone. \
         The driver packs that bitmap on the host -- see `packed_len` and the \
         `mask[base + (index >> 3)] |= 1 << (index & 7)` write in \
         `driver-cuda/src/fire/page_mask.rs`. Nothing included it and no \
         `Fire::at` named it; `kernels-cuda/tests/every_source_is_reached.rs` \
         is what found it, and took `kernels::Ty::StructuredMasks` with it.",
    ),
    (
        "moe/params.inc.wgsl",
        "The WGSL copy of `moe/params.h`'s three routing blocks. The five MoE \
         routing arms take every one of those fields as a `Const<u32>` mark \
         now, packed into the `@group(1)` uniform by \
         `driver-wgpu::lowering::routine::bind`; \
         `kernels-wgpu/kernels/moe/route.wgsl` states its going.",
    ),
    (
        "attn/kv_paged.cu",
        "The paged-KV host program. Its launchers are `driver-cuda`'s \
         `fire/kv_paged.rs`; the device half is `attn/kv_paged.cuh`.",
    ),
    (
        "gemm/gemv.cu",
        "The matvec host program, deleted with `gemm/gemv.hpp`. The kernels \
         are `gemm/gemv.cuh` and the host side is `kernels-cuda`'s \
         `gemm/gemv.rs`.",
    ),
    (
        "moe/flashinfer_moe.cu",
        "The FlashInfer mixture host program.",
    ),
    (
        "attn/attention_xqa.cu",
        "The XQA host program; the `from:` provenance on `kernels-cuda`'s \
         `attn/xqa.rs` rows points into it, and the device half survives as \
         `attn/attention_xqa.cuh`.",
    ),
    (
        "attn/attention_xqa_gqa2.cu",
        "One of the six per-GQA-width XQA translation units.",
    ),
    (
        "attn/attention_xqa_gqa2_p16.cu",
        "One of the six per-GQA-width XQA translation units.",
    ),
    (
        "attn/attention_xqa_gqa4.cu",
        "One of the six per-GQA-width XQA translation units.",
    ),
    (
        "attn/attention_xqa_gqa8.cu",
        "One of the six per-GQA-width XQA translation units.",
    ),
    (
        "attn/attention_xqa_gqa8_sm90.cu",
        "One of the six per-GQA-width XQA translation units.",
    ),
    (
        "norm/scale.metal",
        "Never existed. `driver-metal`'s `lowering/routine.rs` fires it from \
         two test bodies on purpose -- what is under test is the seam \
         between a body's statement and a dispatch, and a body naming a real \
         entrypoint would invite the reader to check it against a shader, \
         which is what `tests/` is for once a family has crossed.",
    ),
    (
        "attn/kv_cache_view.hpp",
        "GONE. The paged-KV view types the C++ handed the device half; \
         `driver-cuda`'s `enum_mirrors.rs` is what holds their Rust mirrors \
         against the `.cuh` now.",
    ),
    (
        "gemm/gemv.hpp",
        "GONE, with `gemm/gemv.cu`.",
    ),
    (
        "xqa/mha.cu",
        "UPSTREAM: FlashInfer's, cited by `kernels-cuda`'s `attn/xqa.rs` for \
         where a constant came from. Our XQA sources are `attn/attention_xqa*`.",
    ),
    (
        "xqa/mha_sm90.cu",
        "UPSTREAM: FlashInfer's, as above.",
    ),
    (
        "xqa/tensorMap.h",
        "UPSTREAM: FlashInfer's, as above -- the `CUtensorMapDataType_enum` \
         spelling `attn/xqa.rs` mirrors.",
    ),
    (
        "ptir/ptir_rng.generated.metal",
        "STAGED. Real, and generated: it is written to \
         `crates/tensor-compiler/include/ptir_rng.generated.metal` by \
         `rng_contract.rs` (`PTIR_REGEN=1`), and `ptir/` is where a \
         deployment's kernels directory puts it -- which is the path \
         `driver-metal`'s emitted sources `#include`.",
    ),
];

/// The extensions a kernel source can have.
const EXTS: &[&str] = &["h", "hpp", "metal", "slang", "wgsl", "cu", "cuh"];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn walk(dir: &Path, keep: &mut dyn FnMut(&Path)) {
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            walk(&path, keep);
        } else {
            keep(&path);
        }
    }
}

/// Every kernel source the trees carry, keyed `dir/file.ext`.
///
/// The key is the tail after a crate's kernel root, which is the form every
/// citation uses and the form `Fire::at` takes.
fn carried() -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for crate_dir in std::fs::read_dir(repo_root()).into_iter().flatten().flatten() {
        let name = crate_dir.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with("kernels-") {
            continue;
        }
        for root in ["kernels", "src"] {
            let root = crate_dir.path().join(root);
            walk(&root, &mut |path| {
                if !path
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| EXTS.contains(&e))
                {
                    return;
                }
                let rel = path.strip_prefix(&root).unwrap_or(path);
                let mut parts: Vec<&str> = rel
                    .components()
                    .map(|c| c.as_os_str().to_str().unwrap_or_default())
                    .collect();
                // The tail is `dir/file.ext`; a source directly under the
                // root has no directory and is never cited with one.
                if parts.len() >= 2 {
                    parts.drain(..parts.len() - 2);
                    out.insert(parts.join("/"));
                }
            });
        }
    }
    out
}

/// The directories the kernel trees use, so an upstream path is not read as
/// a claim about this one.
fn kernel_dirs(carried: &BTreeSet<String>) -> BTreeSet<String> {
    let mut dirs: BTreeSet<String> = carried
        .iter()
        .filter_map(|s| s.split('/').next().map(str::to_owned))
        .collect();
    // A deleted source's directory may have gone with it.
    for (path, _) in UNCARRIED {
        if let Some(dir) = path.split('/').next() {
            dirs.insert(dir.to_owned());
        }
    }
    dirs
}

/// Every `dir/file.ext` named anywhere in the workspace's text, with where.
fn citations(dirs: &BTreeSet<String>) -> BTreeMap<String, BTreeSet<String>> {
    let mut found: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for crate_dir in std::fs::read_dir(repo_root()).into_iter().flatten().flatten() {
        walk(&crate_dir.path(), &mut |path| {
            if !path.extension().is_some_and(|e| e == "rs") {
                return;
            }
            // This file's own `UNCARRIED` names every entry once.
            if path.file_name().is_some_and(|n| n == "every_cited_source_exists.rs") {
                return;
            }
            let Ok(text) = std::fs::read_to_string(path) else {
                return;
            };
            let where_ = path
                .strip_prefix(repo_root())
                .unwrap_or(path)
                .display()
                .to_string();
            for cite in scan(&text, dirs) {
                found.entry(cite).or_default().insert(where_.clone());
            }
        });
    }
    found
}

/// The citations in one text.
///
/// A citation is `dir/file.ext` with `dir` one of the kernel trees', taken
/// from a run of path characters so that a longer upstream path -- which
/// has a segment before the directory -- is not clipped down to one of ours.
fn scan(text: &str, dirs: &BTreeSet<String>) -> BTreeSet<String> {
    let is_path = |c: char| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '/' | '-');
    let mut out = BTreeSet::new();
    for run in text.split(|c: char| !is_path(c)) {
        let parts: Vec<&str> = run.split('/').collect();
        if parts.len() != 2 {
            continue;
        }
        let (dir, file) = (parts[0], parts[1]);
        if !dirs.contains(dir) {
            continue;
        }
        // `foo.cuh:65-74` and `foo.metal::symbol` are citations of the file.
        let file = file.split(':').next().unwrap_or(file);
        let file = file.trim_end_matches(['.', ',', '`', ')']);
        if file
            .rsplit_once('.')
            .is_some_and(|(_, e)| EXTS.contains(&e))
        {
            out.insert(format!("{dir}/{file}"));
        }
    }
    out
}

/// Every kernel source named in prose either exists or is listed as gone.
#[test]
fn no_text_names_a_kernel_source_the_trees_do_not_carry() {
    let carried = carried();
    assert!(
        carried.len() > 100,
        "the walk found {} sources, which is too few to be the kernel trees",
        carried.len()
    );
    let dirs = kernel_dirs(&carried);
    let excused: BTreeSet<&str> = UNCARRIED.iter().map(|(p, _)| *p).collect();

    let mut dangling: Vec<String> = Vec::new();
    for (cite, wheres) in citations(&dirs) {
        if carried.contains(&cite) || excused.contains(cite.as_str()) {
            continue;
        }
        let wheres: Vec<&str> = wheres.iter().map(String::as_str).take(3).collect();
        dangling.push(format!("  {cite}  <- {}", wheres.join(", ")));
    }

    assert!(
        dangling.is_empty(),
        "these paths are named in the text and no kernel tree carries \
         them:\n{}\n\nEither the file was renamed and the text was not, or \
         it was deleted and the text still names it -- in which case put it \
         in `UNCARRIED` with what it was, so the next reader who greps for \
         it lands somewhere.",
        dangling.join("\n")
    );
}

/// A listed path is really not carried.
///
/// The list is for what the trees do not have, and an entry naming a source
/// that arrived (or came back) would quietly stop this gate from checking a
/// path it can check.
#[test]
fn nothing_listed_as_uncarried_is_carried_after_all() {
    let carried = carried();
    let resurrected: Vec<&str> = UNCARRIED
        .iter()
        .map(|(p, _)| *p)
        .filter(|p| carried.contains(*p))
        .collect();
    assert!(
        resurrected.is_empty(),
        "these are listed as gone and the trees carry them: \
         {resurrected:?} -- drop the entry, the real file is checked now."
    );
    let why_missing: Vec<&str> = UNCARRIED
        .iter()
        .filter(|(_, why)| why.trim().is_empty())
        .map(|(p, _)| *p)
        .collect();
    assert!(
        why_missing.is_empty(),
        "an entry without a reason is a silencer: {why_missing:?}"
    );
}
