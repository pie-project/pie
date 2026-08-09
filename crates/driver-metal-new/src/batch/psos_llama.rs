//! The llama families' PSO plan and their one piece of new arithmetic.
//!
//! The C++ file this ports (`llama/kernels.cpp`) is short, and that is
//! the finding rather than an omission: the shared decode table already
//! carries every dispatch a dense llama layer makes, because the shared
//! `Kernel` enum's common prefix IS a llama decoder. What the family
//! adds is exactly three things the shared table has no reason to
//! carry: attention at its OWN head width, the freq-table rope for
//! llama 3.1's schedule, and the routed FFN. Notably NOT borrowed:
//! `gptoss_swiglu` — that kernel bakes in gpt-oss's asymmetric clamp,
//! its alpha and its `(up + 1)`; Qwen3-MoE uses plain SwiGLU, so the
//! routed path runs the ordinary `silu_mul` over the sorted stack.

use super::abi::Kernel;
use super::geometry::AffineFormat;
use super::llama::LlamaGeometry;
use super::psos::{DecodePsoPlan, EntryNames, Features, PsoRequest, plan_decode_psos};

/// The shared table's entry names, spelled from this checkpoint's affine
/// format rather than the shipped g64/b4 constant.
#[must_use]
pub fn llama_entry_names(quant: AffineFormat) -> EntryNames {
    let s = quant.kernel_suffix();
    EntryNames {
        embed_gather: format!("embed_gather_4bit{s}"),
        qmv_fast: format!("affine_qmv_fast{s}"),
        qmv_residual: format!("affine_qmv_fast_residual{s}"),
        qmv_routed: format!("affine_qmv_routed{s}"),
    }
}

/// The whole M=1 compile list: the shared plan under this family's
/// feature set, plus the two claims only the geometry can spell.
///
/// The attention width is the geometry's, not a literal: these entries
/// were once named `_d128` outright, which is the width llama, mistral,
/// qwen2, qwen3 and the Qwen MoEs all happen to use — and Llama-3.2-1B
/// is 32 heads of 64, which a d=128 pipeline does not fail on: it
/// strides past the end of every head and writes zeros. Spelled from
/// `head_dim`, an uninstantiated width fails to build BY NAME at load.
///
/// The freq-table rope is claimed AFTER the base rope, so `source_of`
/// answers with it — the base geometric series stays compiled and
/// unused, which is the cheap direction of that trade.
#[must_use]
pub fn llama_step_plan(g: &LlamaGeometry) -> DecodePsoPlan {
    let names = llama_entry_names(g.quant);
    let features = Features {
        argmax: true,
        routed: g.is_moe(),
        untied: !g.tied_embeddings,
        ..Features::default()
    };
    let mut plan = plan_decode_psos(&names, features);
    plan.requests.push(PsoRequest {
        file: "attn/sdpa_vector.metal",
        entry: format!("sdpa_vector_decode_bfloat16_d_{}", g.head_dim),
        kinds: vec![Kernel::Sdpa],
    });
    if g.rope_freq_table {
        plan.requests.push(PsoRequest {
            file: "rope/neox.metal",
            entry: "neox_freqs_decode_bfloat16".to_owned(),
            kinds: vec![Kernel::Rope, Kernel::RopeK],
        });
    }
    plan
}

/// The M>1 compile list: the M=1 plan with the batched forms claimed
/// AFTER their base entries, so `source_of` answers with them. The MB
/// embed reads a row per token off `grid.y`; the rope reads
/// `position[row]`; the attention and the append walk pages. The paged
/// attention's name carries the page shape too — `_p32` is a separate
/// instantiation, and a 32-page pool under the generic entry is correct
/// but leaves the specialised loads on the table. The `_sg8` rung
/// (d=64/p32) is deferred: its threadgroup is 256 where the generic one
/// is 1024, and a pipeline choice that does not move the launch with it
/// is the grid-against-wrong-pipeline mismatch again.
#[must_use]
pub fn llama_mb_plan(g: &LlamaGeometry) -> DecodePsoPlan {
    let names = llama_entry_names(g.quant);
    let features = Features {
        argmax: true,
        routed: g.is_moe(),
        untied: !g.tied_embeddings,
        ..Features::default()
    };
    let mut plan = plan_decode_psos(&names, features);
    let s = g.quant.kernel_suffix();
    let mut want = |file: &'static str, entry: String, kinds: &[Kernel]| {
        plan.requests.push(PsoRequest {
            file,
            entry,
            kinds: kinds.to_vec(),
        });
    };
    want(
        "layout/embed_gather.metal",
        format!("embed_gather_mb_4bit{s}"),
        &[Kernel::EmbedGather, Kernel::EmbedUntied],
    );
    want(
        "rope/neox.metal",
        if g.rope_freq_table {
            "neox_freqs_mb_bfloat16".to_owned()
        } else {
            "neox_mb_bfloat16".to_owned()
        },
        &[Kernel::Rope, Kernel::RopeK],
    );
    want(
        "attn/kv_write.metal",
        "kv_append_paged_bfloat16".to_owned(),
        &[Kernel::KvAppendPaged],
    );
    want(
        "attn/sdpa_paged.metal",
        format!(
            "sdpa_paged_decode_bfloat16_d_{}{}",
            g.head_dim,
            if g.kv_page_size == 32 { "_p32" } else { "" }
        ),
        &[Kernel::SdpaPaged],
    );
    want(
        "layout/row_gather.metal",
        "row_gather_bfloat16".to_owned(),
        &[Kernel::G4RowGather],
    );
    plan
}

/// Llama 3.1's rotary frequencies, ported from
/// `mlx_lm/models/rope_utils.py::Llama3RoPE`.
///
/// A rotary dimension's WAVELENGTH decides what happens to it.
/// Dimensions whose wavelength is longer than the original context
/// could hold (they turn fewer than `low_freq_factor` times in it) are
/// interpolated by the full factor; those short enough to turn more
/// than `high_freq_factor` times are left alone, because extrapolating
/// them is safe; between the two the schedule ramps smoothly. A closed
/// form over `rotary_dims/2` values with no dependence on position, so
/// the host computes it once at setup — exactly what
/// `rope_neox_freqs_*` was built to consume.
///
/// Returns `inv_freq`, the RECIPROCAL of mlx's `_freqs`: `mx.fast.rope`
/// divides the position by its table and this kernel multiplies by its
/// own.
#[must_use]
pub fn llama3_inv_freq(g: &LlamaGeometry) -> Vec<f32> {
    let dims = g.rotary_dims();
    let half = (dims / 2).max(1) as usize;
    if dims < 2 {
        return vec![0.0; half];
    }
    let base = g.rope_theta;
    let factor = if g.rope_scaling_factor > 0.0 {
        g.rope_scaling_factor
    } else {
        1.0
    };
    let lo = g.rope_low_freq_factor;
    let hi = g.rope_high_freq_factor;
    let orig = g.rope_original_max_position as f32;
    let low_wavelen = orig / lo;
    let high_wavelen = orig / hi;

    (0..half)
        .map(|i| {
            // mlx's `_freqs` is base^(2i/dims) — a WAVELENGTH-like
            // quantity, the reciprocal of the usual inv_freq. The
            // schedule is expressed on it, so it is computed on it and
            // inverted once at the end.
            let freq = base.powf(2.0 * i as f32 / dims as f32);
            let wavelen = 2.0 * std::f32::consts::PI * freq;
            let scaled = if wavelen > low_wavelen {
                // Turns too slowly to extrapolate: interpolate by the
                // whole factor.
                freq * factor
            } else if wavelen > high_wavelen {
                // The ramp. Below `high_wavelen` the dimension is left
                // alone — the untouched arm below.
                let smooth = (orig / wavelen - lo) / (hi - lo);
                freq / ((1.0 - smooth) / factor + smooth)
            } else {
                freq
            };
            if scaled != 0.0 { 1.0 / scaled } else { 0.0 }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::build_llama_dag;
    use crate::tuning::Tuning;

    fn assert_close(actual: f32, expected: f32, what: &str) {
        let rel = (actual - expected).abs() / expected.abs().max(f32::MIN_POSITIVE);
        assert!(rel < 1e-5, "{what}: {actual} vs {expected}");
    }

    #[test]
    fn the_31_schedule_matches_mlx_at_both_ends_and_on_the_ramp() {
        // Llama-3.1's own numbers: theta 500k, factor 8, ramp 1..4,
        // original context 8192, 128-wide heads. The pinned values are
        // mlx_lm's Llama3RoPE, checked byte-for-byte in float32.
        let g = LlamaGeometry {
            rope_freq_table: true,
            rope_scaling_factor: 8.0,
            ..LlamaGeometry::default()
        };
        let inv = llama3_inv_freq(&g);
        assert_eq!(inv.len(), 64);
        // The fast dimensions turn more than 4 times in 8192 positions:
        // extrapolation is safe and the schedule must not touch them.
        assert_eq!(
            inv[0], 1.0,
            "an altered fast dimension breaks short context"
        );
        assert_close(inv[40], 3.428_102_2e-5, "the ramp");
        assert_close(inv[50], 4.411_534_7e-6, "the ramp's slow end");
        // The slowest dimensions interpolate by the whole factor.
        assert_close(inv[63], 3.068_926e-7, "the interpolated tail");
        let plain = LlamaGeometry::default();
        let unscaled = llama3_inv_freq(&plain);
        assert_close(
            inv[63] * 8.0,
            unscaled[63],
            "the tail is the geometric series divided by exactly the factor",
        );
    }

    #[test]
    fn the_plan_claims_every_kind_each_shape_dispatches() {
        let table: std::collections::HashSet<String> =
            kernels_metal::entrypoints().into_iter().collect();
        let tuning = Tuning::default();
        let shapes = [
            LlamaGeometry::default(), // dense, untied, plain rope
            LlamaGeometry {
                qk_norm: true,
                tied_embeddings: true,
                rope_freq_table: true,
                rope_scaling_factor: 8.0,
                ..LlamaGeometry::default()
            },
            LlamaGeometry {
                qk_norm: true,
                n_experts: 128,
                experts_per_token: 8,
                moe_intermediate: 768,
                ..LlamaGeometry::default()
            },
        ];
        for g in shapes {
            let plan = llama_step_plan(&g);
            let dag = build_llama_dag(&g, &tuning, true);
            for d in &dag {
                assert!(
                    plan.source_of(d.kind).is_some(),
                    "{:?} has no compiled pipeline — the fire would refuse at prepare",
                    d.kind
                );
            }
            for request in &plan.requests {
                assert!(table.contains(&request.entry), "{}", request.entry);
            }
        }
    }

    #[test]
    fn the_attention_width_is_the_geometrys_and_the_table_rope_wins() {
        let entry_of = |plan: &DecodePsoPlan, kind: Kernel| {
            let i = plan.source_of(kind).expect("claimed");
            plan.requests[i].entry.clone()
        };
        let g = LlamaGeometry::default();
        assert!(entry_of(&llama_step_plan(&g), Kernel::Sdpa).ends_with("_d_128"));
        // Llama-3.2-1B: 64-wide heads. The old literal `_d128` handed
        // them a 128-wide pipeline that strode past the end of every
        // head and wrote zeros; spelled from the geometry, the name IS
        // the width.
        let narrow = LlamaGeometry {
            head_dim: 64,
            ..LlamaGeometry::default()
        };
        assert!(entry_of(&llama_step_plan(&narrow), Kernel::Sdpa).ends_with("_d_64"));
        // The plain series serves the base rope; the 3.1 table claims
        // Rope AFTER it and source_of answers with the later claim.
        assert_eq!(
            entry_of(&llama_step_plan(&g), Kernel::Rope),
            "neox_decode_bfloat16"
        );
        let table = LlamaGeometry {
            rope_freq_table: true,
            rope_scaling_factor: 8.0,
            ..LlamaGeometry::default()
        };
        assert_eq!(
            entry_of(&llama_step_plan(&table), Kernel::Rope),
            "neox_freqs_decode_bfloat16"
        );
    }
}
