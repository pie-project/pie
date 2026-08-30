//! The shader SOURCES, as static text — the one thing a Metal driver cannot
//! get from a [`Fire`](crate::Fire) alone.
//!
//! A `Fire` names its shader by path (`"attn/kv_write.metal"`) and its
//! entrypoint by name, and that is all a kernel entry knows. The driver has
//! to turn the pair into an `MTLComputePipelineState`, which means it has to
//! have the source text: `newLibraryWithSource:` is the only compile door a
//! machine with the Command Line Tools and no Xcode has (there is no
//! `xcrun metal` there, measured), so an offline `.metallib` is not a
//! portable option and the sources travel INSIDE the rlib.
//!
//! This module is glue, not a kernel: it adds no shader, states no geometry
//! and names no entrypoint. It is the file-path half of the `Fire` contract,
//! answered where the files live.
//!
//! **Includes are the driver's to flatten.** `newLibraryWithSource:` has no
//! header search path, so a `#include "../third_party/…"` inside a source
//! resolves to nothing. The paths are relative to the INCLUDING file's
//! directory; [`resolve`] is that arithmetic, published here so the two
//! shells that need it cannot spell it differently.

/// Where the sources live in the source tree, for `include_str!`.
macro_rules! source_root {
    () => {
        concat!(env!("CARGO_MANIFEST_DIR"), "/kernels")
    };
}
/// Every `.metal` file this crate ships, by the path a [`Fire`](crate::Fire)
/// names it with — third-party headers included, since they are what the
/// `#include` lines resolve to.
pub const SOURCES: &[(&str, &str)] = &[
    ("attn/attn_sink.metal", include_str!(concat!(source_root!(), "/attn/attn_sink.metal"))),
    ("attn/dense.metal", include_str!(concat!(source_root!(), "/attn/dense.metal"))),
    ("attn/kv_write.metal", include_str!(concat!(source_root!(), "/attn/kv_write.metal"))),
    ("attn/logit_softcap.metal", include_str!(concat!(source_root!(), "/attn/logit_softcap.metal"))),
    ("attn/merge_lse.metal", include_str!(concat!(source_root!(), "/attn/merge_lse.metal"))),
    ("attn/score.metal", include_str!(concat!(source_root!(), "/attn/score.metal"))),
    ("attn/sdpa_paged.metal", include_str!(concat!(source_root!(), "/attn/sdpa_paged.metal"))),
    ("attn/sdpa_paged_mma.metal", include_str!(concat!(source_root!(), "/attn/sdpa_paged_mma.metal"))),
    ("attn/sdpa_sliding.metal", include_str!(concat!(source_root!(), "/attn/sdpa_sliding.metal"))),
    ("attn/sdpa_vector.metal", include_str!(concat!(source_root!(), "/attn/sdpa_vector.metal"))),
    ("attn/split_qkv.metal", include_str!(concat!(source_root!(), "/attn/split_qkv.metal"))),
    ("attn/ssm_causal_conv1d.metal", include_str!(concat!(source_root!(), "/attn/ssm_causal_conv1d.metal"))),
    ("attn/ssm_gated_delta.metal", include_str!(concat!(source_root!(), "/attn/ssm_gated_delta.metal"))),
    ("attn/ssm_gdn_core.metal", include_str!(concat!(source_root!(), "/attn/ssm_gdn_core.metal"))),
    ("attn/ssm_gdn_prep.metal", include_str!(concat!(source_root!(), "/attn/ssm_gdn_prep.metal"))),
    ("attn/ssm_kda.metal", include_str!(concat!(source_root!(), "/attn/ssm_kda.metal"))),
    ("elemwise/gate.metal", include_str!(concat!(source_root!(), "/elemwise/gate.metal"))),
    ("elemwise/norm_add_bias.metal", include_str!(concat!(source_root!(), "/elemwise/norm_add_bias.metal"))),
    ("elemwise/norm_gated_rms.metal", include_str!(concat!(source_root!(), "/elemwise/norm_gated_rms.metal"))),
    ("elemwise/norm_layer_scalar.metal", include_str!(concat!(source_root!(), "/elemwise/norm_layer_scalar.metal"))),
    ("elemwise/norm_residual_add.metal", include_str!(concat!(source_root!(), "/elemwise/norm_residual_add.metal"))),
    ("elemwise/norm_rms.metal", include_str!(concat!(source_root!(), "/elemwise/norm_rms.metal"))),
    ("elemwise/norm_vector.metal", include_str!(concat!(source_root!(), "/elemwise/norm_vector.metal"))),
    ("elemwise/rope_mrope.metal", include_str!(concat!(source_root!(), "/elemwise/rope_mrope.metal"))),
    ("elemwise/rope_neox.metal", include_str!(concat!(source_root!(), "/elemwise/rope_neox.metal"))),
    ("icb/rebind.metal", include_str!(concat!(source_root!(), "/icb/rebind.metal"))),
    ("layout/blit.metal", include_str!(concat!(source_root!(), "/layout/blit.metal"))),
    ("layout/deinterleave.metal", include_str!(concat!(source_root!(), "/layout/deinterleave.metal"))),
    ("layout/embed.metal", include_str!(concat!(source_root!(), "/layout/embed.metal"))),
    ("layout/embed_gather.metal", include_str!(concat!(source_root!(), "/layout/embed_gather.metal"))),
    ("layout/ple_combine.metal", include_str!(concat!(source_root!(), "/layout/ple_combine.metal"))),
    ("layout/row_gather.metal", include_str!(concat!(source_root!(), "/layout/row_gather.metal"))),
    ("linear/gemm_dense.metal", include_str!(concat!(source_root!(), "/linear/gemm_dense.metal"))),
    ("linear/lora.metal", include_str!(concat!(source_root!(), "/linear/lora.metal"))),
    ("linear/mlp_gated.metal", include_str!(concat!(source_root!(), "/linear/mlp_gated.metal"))),
    ("linear/mlp_packed.metal", include_str!(concat!(source_root!(), "/linear/mlp_packed.metal"))),
    ("linear/moe_route.metal", include_str!(concat!(source_root!(), "/linear/moe_route.metal"))),
    ("linear/moe_select.metal", include_str!(concat!(source_root!(), "/linear/moe_select.metal"))),
    ("linear/quant_qmm_t.metal", include_str!(concat!(source_root!(), "/linear/quant_qmm_t.metal"))),
    ("linear/quant_qmv.metal", include_str!(concat!(source_root!(), "/linear/quant_qmv.metal"))),
    ("linear/quant_transcode.metal", include_str!(concat!(source_root!(), "/linear/quant_transcode.metal"))),
    ("ptir/logits_copy.metal", include_str!(concat!(source_root!(), "/ptir/logits_copy.metal"))),
    ("ptir/ptir_rng.generated.metal", include_str!(concat!(source_root!(), "/ptir/ptir_rng.generated.metal"))),
    ("sample/argmax.metal", include_str!(concat!(source_root!(), "/sample/argmax.metal"))),
    ("third_party/mlx_quantized_block.metal", include_str!(concat!(source_root!(), "/third_party/mlx_quantized_block.metal"))),
    ("third_party/mlx_steel_loader.metal", include_str!(concat!(source_root!(), "/third_party/mlx_steel_loader.metal"))),
    ("third_party/mlx_steel_mma.metal", include_str!(concat!(source_root!(), "/third_party/mlx_steel_mma.metal"))),
    ("third_party/mlx_steel_prelude.metal", include_str!(concat!(source_root!(), "/third_party/mlx_steel_prelude.metal"))),
    ("third_party/mlx_steel_transforms.metal", include_str!(concat!(source_root!(), "/third_party/mlx_steel_transforms.metal"))),
];

/// The text of one shader file, by the path a `Fire` names.
#[must_use]
pub fn source(file: &str) -> Option<&'static str> {
    SOURCES
        .iter()
        .find(|(name, _)| *name == file)
        .map(|(_, text)| *text)
}

/// Resolve `file`'s `#include "…"` lines against [`SOURCES`], recursively,
/// and return one flat translation unit.
///
/// Angle-bracket includes (`<metal_stdlib>`) are the toolchain's and pass
/// through untouched. A quoted include is resolved relative to the including
/// file's directory, once: a header pulled in twice is emitted once, which
/// is what the sources' own `#pragma once`-less style requires.
///
/// # Errors
///
/// The name of the first file — the root or an include — this crate does not
/// ship.
pub fn resolve(file: &str) -> Result<String, String> {
    let mut out = String::new();
    let mut seen = Vec::new();
    expand(file, &mut out, &mut seen)?;
    Ok(out)
}

fn expand(file: &str, out: &mut String, seen: &mut Vec<String>) -> Result<(), String> {
    if seen.iter().any(|s| s == file) {
        return Ok(());
    }
    seen.push(file.to_string());
    let text = source(file).ok_or_else(|| file.to_string())?;
    let dir = file.rsplit_once('/').map_or("", |(d, _)| d);
    for line in text.lines() {
        match quoted_include(line) {
            Some(target) => expand(&join(dir, target), out, seen)?,
            None => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    Ok(())
}

/// The target of a `#include "…"` line, or `None` for anything else.
fn quoted_include(line: &str) -> Option<&str> {
    let rest = line.trim_start().strip_prefix("#include")?.trim_start();
    let rest = rest.strip_prefix('"')?;
    rest.split_once('"').map(|(target, _)| target)
}

/// `dir` joined with a possibly-`../`-prefixed relative path, normalized.
fn join(dir: &str, target: &str) -> String {
    let mut parts: Vec<&str> = if dir.is_empty() {
        Vec::new()
    } else {
        dir.split('/').collect()
    };
    for step in target.split('/') {
        match step {
            "." | "" => {}
            ".." => {
                parts.pop();
            }
            name => parts.push(name),
        }
    }
    parts.join("/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_shipped_source_is_named_by_its_path() {
        for (name, text) in SOURCES {
            assert!(name.ends_with(".metal"), "{name} is not a shader path");
            assert!(!text.is_empty(), "{name} is empty");
            assert_eq!(source(name), Some(*text));
        }
    }

    #[test]
    fn resolving_leaves_the_toolchain_headers_alone_and_pulls_the_local_ones_in() {
        // `gemm_dense` is the deepest include chain the tree has: four
        // third-party headers under `../third_party/`.
        let flat = resolve("linear/gemm_dense.metal").expect("gemm_dense resolves");
        assert!(
            !flat.contains("#include \""),
            "a quoted include survived the flattening"
        );
        assert!(
            flat.contains("#include <metal_stdlib>"),
            "an angle-bracket include was eaten"
        );
        // A symbol only the prelude defines, so the pull really happened.
        let prelude = source("third_party/mlx_steel_prelude.metal").expect("prelude ships");
        let marker = prelude
            .lines()
            .find(|l| l.starts_with("struct") || l.starts_with("template"))
            .expect("the prelude declares something");
        assert!(flat.contains(marker), "the prelude was not pulled in");
    }

    #[test]
    fn every_source_resolves() {
        for (name, _) in SOURCES {
            resolve(name).unwrap_or_else(|missing| panic!("{name} includes {missing}, unshipped"));
        }
    }

    #[test]
    fn a_relative_include_is_joined_against_the_including_files_directory() {
        assert_eq!(join("linear", "../third_party/x.metal"), "third_party/x.metal");
        assert_eq!(join("", "a.metal"), "a.metal");
        assert_eq!(join("a/b", "c.metal"), "a/b/c.metal");
    }
}
