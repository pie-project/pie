macro_rules! source_root {
    () => {
        concat!(env!("CARGO_MANIFEST_DIR"), "/kernels")
    };
}
pub const SOURCES: &[(&str, &str)] = &[
    (
        "attn/dense.metal",
        include_str!(concat!(source_root!(), "/attn/dense.metal")),
    ),
    (
        "attn/index.metal",
        include_str!(concat!(source_root!(), "/attn/index.metal")),
    ),
    (
        "attn/kv_codec.metal",
        include_str!(concat!(source_root!(), "/attn/kv_codec.metal")),
    ),
    (
        "attn/kv_write.metal",
        include_str!(concat!(source_root!(), "/attn/kv_write.metal")),
    ),
    (
        "attn/logit_softcap.metal",
        include_str!(concat!(source_root!(), "/attn/logit_softcap.metal")),
    ),
    (
        "attn/merge.metal",
        include_str!(concat!(source_root!(), "/attn/merge.metal")),
    ),
    (
        "attn/mla.metal",
        include_str!(concat!(source_root!(), "/attn/mla.metal")),
    ),
    (
        "attn/pool.metal",
        include_str!(concat!(source_root!(), "/attn/pool.metal")),
    ),
    (
        "attn/q8_decode.metal",
        include_str!(concat!(source_root!(), "/attn/q8_decode.metal")),
    ),
    (
        "attn/q8_mpp.metal",
        include_str!(concat!(source_root!(), "/attn/q8_mpp.metal")),
    ),
    (
        "attn/q8_split.metal",
        include_str!(concat!(source_root!(), "/attn/q8_split.metal")),
    ),
    (
        "attn/q8_write.metal",
        include_str!(concat!(source_root!(), "/attn/q8_write.metal")),
    ),
    (
        "attn/ragged.metal",
        include_str!(concat!(source_root!(), "/attn/ragged.metal")),
    ),
    (
        "attn/score.metal",
        include_str!(concat!(source_root!(), "/attn/score.metal")),
    ),
    (
        "attn/sdpa_paged.metal",
        include_str!(concat!(source_root!(), "/attn/sdpa_paged.metal")),
    ),
    (
        "attn/sdpa_paged_mma.metal",
        include_str!(concat!(source_root!(), "/attn/sdpa_paged_mma.metal")),
    ),
    (
        "attn/sink.metal",
        include_str!(concat!(source_root!(), "/attn/sink.metal")),
    ),
    (
        "attn/voxel.metal",
        include_str!(concat!(source_root!(), "/attn/voxel.metal")),
    ),
    (
        "conv/conv3d.metal",
        include_str!(concat!(source_root!(), "/conv/conv3d.metal")),
    ),
    (
        "conv/grid.h",
        include_str!(concat!(source_root!(), "/conv/grid.h")),
    ),
    (
        "conv/group_norm.metal",
        include_str!(concat!(source_root!(), "/conv/group_norm.metal")),
    ),
    (
        "conv/resample.metal",
        include_str!(concat!(source_root!(), "/conv/resample.metal")),
    ),
    (
        "conv/rule.metal",
        include_str!(concat!(source_root!(), "/conv/rule.metal")),
    ),
    (
        "elemwise/gate.metal",
        include_str!(concat!(source_root!(), "/elemwise/gate.metal")),
    ),
    (
        "elemwise/hc.metal",
        include_str!(concat!(source_root!(), "/elemwise/hc.metal")),
    ),
    (
        "elemwise/modulate.metal",
        include_str!(concat!(source_root!(), "/elemwise/modulate.metal")),
    ),
    (
        "elemwise/norm_add_bias.metal",
        include_str!(concat!(source_root!(), "/elemwise/norm_add_bias.metal")),
    ),
    (
        "elemwise/norm_gated_rms.metal",
        include_str!(concat!(source_root!(), "/elemwise/norm_gated_rms.metal")),
    ),
    (
        "elemwise/norm_layer_scalar.metal",
        include_str!(concat!(source_root!(), "/elemwise/norm_layer_scalar.metal")),
    ),
    (
        "elemwise/norm_layernorm.metal",
        include_str!(concat!(source_root!(), "/elemwise/norm_layernorm.metal")),
    ),
    (
        "elemwise/norm_residual_add.metal",
        include_str!(concat!(source_root!(), "/elemwise/norm_residual_add.metal")),
    ),
    (
        "elemwise/norm_rms.metal",
        include_str!(concat!(source_root!(), "/elemwise/norm_rms.metal")),
    ),
    (
        "elemwise/norm_standardize.metal",
        include_str!(concat!(source_root!(), "/elemwise/norm_standardize.metal")),
    ),
    (
        "elemwise/norm_vector.metal",
        include_str!(concat!(source_root!(), "/elemwise/norm_vector.metal")),
    ),
    (
        "elemwise/pointwise.metal",
        include_str!(concat!(source_root!(), "/elemwise/pointwise.metal")),
    ),
    (
        "elemwise/res_blend.metal",
        include_str!(concat!(source_root!(), "/elemwise/res_blend.metal")),
    ),
    (
        "elemwise/rope_axes.metal",
        include_str!(concat!(source_root!(), "/elemwise/rope_axes.metal")),
    ),
    (
        "elemwise/rope_mrope.metal",
        include_str!(concat!(source_root!(), "/elemwise/rope_mrope.metal")),
    ),
    (
        "elemwise/rope_neox.metal",
        include_str!(concat!(source_root!(), "/elemwise/rope_neox.metal")),
    ),
    (
        "elemwise/sinusoid.metal",
        include_str!(concat!(source_root!(), "/elemwise/sinusoid.metal")),
    ),
    (
        "gather/deinterleave.metal",
        include_str!(concat!(source_root!(), "/gather/deinterleave.metal")),
    ),
    (
        "gather/embed.metal",
        include_str!(concat!(source_root!(), "/gather/embed.metal")),
    ),
    (
        "gather/embed_gather.metal",
        include_str!(concat!(source_root!(), "/gather/embed_gather.metal")),
    ),
    (
        "gather/embed_weighted.metal",
        include_str!(concat!(source_root!(), "/gather/embed_weighted.metal")),
    ),
    (
        "gather/fold.metal",
        include_str!(concat!(source_root!(), "/gather/fold.metal")),
    ),
    (
        "gather/grid_taps.metal",
        include_str!(concat!(source_root!(), "/gather/grid_taps.metal")),
    ),
    (
        "gather/ngram.metal",
        include_str!(concat!(source_root!(), "/gather/ngram.metal")),
    ),
    (
        "gather/pixels.metal",
        include_str!(concat!(source_root!(), "/gather/pixels.metal")),
    ),
    (
        "gather/rows.metal",
        include_str!(concat!(source_root!(), "/gather/rows.metal")),
    ),
    (
        "gather/split_qkv.metal",
        include_str!(concat!(source_root!(), "/gather/split_qkv.metal")),
    ),
    (
        "linear/ane.metal",
        include_str!(concat!(source_root!(), "/linear/ane.metal")),
    ),
    (
        "linear/gemm.metal",
        include_str!(concat!(source_root!(), "/linear/gemm.metal")),
    ),
    (
        "linear/glu_packed.metal",
        include_str!(concat!(source_root!(), "/linear/glu_packed.metal")),
    ),
    (
        "linear/glu_split.metal",
        include_str!(concat!(source_root!(), "/linear/glu_split.metal")),
    ),
    (
        "linear/lane_gemm.metal",
        include_str!(concat!(source_root!(), "/linear/lane_gemm.metal")),
    ),
    (
        "linear/lora.metal",
        include_str!(concat!(source_root!(), "/linear/lora.metal")),
    ),
    (
        "linear/moe/route.metal",
        include_str!(concat!(source_root!(), "/linear/moe/route.metal")),
    ),
    (
        "linear/moe/select.metal",
        include_str!(concat!(source_root!(), "/linear/moe/select.metal")),
    ),
    (
        "linear/quant_block.h",
        include_str!(concat!(source_root!(), "/linear/quant_block.h")),
    ),
    (
        "linear/quant_ptq1_0.metal",
        include_str!(concat!(source_root!(), "/linear/quant_ptq1_0.metal")),
    ),
    (
        "linear/quant_qmm_mpp.metal",
        include_str!(concat!(source_root!(), "/linear/quant_qmm_mpp.metal")),
    ),
    (
        "linear/quant_qmm_sg.metal",
        include_str!(concat!(source_root!(), "/linear/quant_qmm_sg.metal")),
    ),
    (
        "linear/quant_qmm_staged.metal",
        include_str!(concat!(source_root!(), "/linear/quant_qmm_staged.metal")),
    ),
    (
        "linear/quant_qmm_t.metal",
        include_str!(concat!(source_root!(), "/linear/quant_qmm_t.metal")),
    ),
    (
        "linear/quant_qmv.metal",
        include_str!(concat!(source_root!(), "/linear/quant_qmv.metal")),
    ),
    (
        "linear/quant_qmv_rows.metal",
        include_str!(concat!(source_root!(), "/linear/quant_qmv_rows.metal")),
    ),
    (
        "prim/gemm_loader.h",
        include_str!(concat!(source_root!(), "/prim/gemm_loader.h")),
    ),
    (
        "prim/gemm_mma.h",
        include_str!(concat!(source_root!(), "/prim/gemm_mma.h")),
    ),
    (
        "prim/gemm_traits.h",
        include_str!(concat!(source_root!(), "/prim/gemm_traits.h")),
    ),
    (
        "prim/gemm_transforms.h",
        include_str!(concat!(source_root!(), "/prim/gemm_transforms.h")),
    ),
    (
        "prim/mpp.h",
        include_str!(concat!(source_root!(), "/prim/mpp.h")),
    ),
    (
        "prim/register_table.h",
        include_str!(concat!(source_root!(), "/prim/register_table.h")),
    ),
    (
        "runtime/blit.metal",
        include_str!(concat!(source_root!(), "/runtime/blit.metal")),
    ),
    (
        "runtime/keepalive.metal",
        include_str!(concat!(source_root!(), "/runtime/keepalive.metal")),
    ),
    (
        "runtime/nan_check.metal",
        include_str!(concat!(source_root!(), "/runtime/nan_check.metal")),
    ),
    (
        "runtime/rebind.metal",
        include_str!(concat!(source_root!(), "/runtime/rebind.metal")),
    ),
    (
        "sampling/argmax_rows.metal",
        include_str!(concat!(source_root!(), "/sampling/argmax_rows.metal")),
    ),
    (
        "sampling/selector_walk.metal",
        include_str!(concat!(source_root!(), "/sampling/selector_walk.metal")),
    ),
    (
        "sampling/topk.metal",
        include_str!(concat!(source_root!(), "/sampling/topk.metal")),
    ),
    (
        "ssm/conv1d.metal",
        include_str!(concat!(source_root!(), "/ssm/conv1d.metal")),
    ),
    (
        "ssm/dyn_conv.metal",
        include_str!(concat!(source_root!(), "/ssm/dyn_conv.metal")),
    ),
    (
        "ssm/gated_delta.metal",
        include_str!(concat!(source_root!(), "/ssm/gated_delta.metal")),
    ),
    (
        "ssm/gdn_prep.metal",
        include_str!(concat!(source_root!(), "/ssm/gdn_prep.metal")),
    ),
    (
        "ssm/gdn_scan.metal",
        include_str!(concat!(source_root!(), "/ssm/gdn_scan.metal")),
    ),
    (
        "ssm/gdn_staged.metal",
        include_str!(concat!(source_root!(), "/ssm/gdn_staged.metal")),
    ),
    (
        "ssm/kda.metal",
        include_str!(concat!(source_root!(), "/ssm/kda.metal")),
    ),
];

#[must_use]
pub fn source(file: &str) -> Option<&'static str> {
    SOURCES
        .iter()
        .find(|(name, _)| *name == file)
        .map(|(_, text)| *text)
}

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

fn quoted_include(line: &str) -> Option<&str> {
    let rest = line.trim_start().strip_prefix("#include")?.trim_start();
    let rest = rest.strip_prefix('"')?;
    rest.split_once('"').map(|(target, _)| target)
}

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
