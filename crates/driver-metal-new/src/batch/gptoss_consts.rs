//! GPT-OSS's per-dispatch constants: the portable arithmetic.
//!
//! Unlike gemma4's, none of these depend on the LAYER: every layer has the
//! same head width, the same rope, the same MLP shape. What varies is the
//! attention type, and that is one value (the window).
//!
//! Two are the family's own and both are easy to get silently wrong: the
//! attention scale is `1/sqrt(head_dim)` applied to q — not gemma4's 1.0,
//! which is what a family that folds the scale into a q-norm uses, and
//! gpt-oss has no q-norm — and the routed matvecs bind the SAME K and N as
//! an unrouted one: the expert axis is not a shape the kernel is told, it
//! derives every stride from K and N, so a routed projection's constants
//! are indistinguishable from a dense one's. Deliberate — one fewer thing
//! to disagree.

use super::abi::Kernel;
use super::consts::KN;
use super::gptoss::GptOssGeometry;

/// `KN` per matvec kind, in this family's geometry.
#[must_use]
pub fn gptoss_qmv_kn(kind: Kernel, g: &GptOssGeometry) -> KN {
    let h = g.hidden;
    let kn = |k, n| KN { k, n };
    match kind {
        Kernel::GoQmvQ => kn(h, g.q_dim()),
        Kernel::GoQmvK | Kernel::GoQmvV => kn(h, g.kv_dim()),
        Kernel::GoQmvO => kn(g.q_dim(), h),
        Kernel::GoRouter => kn(h, g.n_experts),
        // The routed three: same shape as a dense projection — the expert
        // axis is a base offset the kernel computes, not a dimension it is
        // told.
        Kernel::GoExpertGate | Kernel::GoExpertUp => kn(h, g.intermediate),
        Kernel::GoExpertDown => kn(g.intermediate, h),
        Kernel::LmHeadUntied => kn(h, g.vocab),
        _ => kn(0, 0),
    }
}

/// `gptoss_swiglu.metal`'s params: the clamped, biased SwiGLU.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwiGluParams {
    /// Elements the dispatch covers.
    pub count: u32,
    /// The clamp on both operands.
    pub limit: f32,
    /// The sigmoid's gain.
    pub alpha: f32,
}
const _: () = assert!(size_of::<SwiGluParams>() == 12);

/// `row_gather.metal`'s params: the sampled-row compaction.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowGatherParams {
    /// The row width.
    pub width: u32,
    /// Rows the tail runs on.
    pub rows: u32,
}
const _: () = assert!(size_of::<RowGatherParams>() == 8);

/// YaRN's frequency table, ported from
/// `mlx_lm/models/rope_utils.py::YarnRoPE`.
///
/// Two frequencies exist for every dimension: the one the original context
/// implies and the one the extended context implies (`theta * factor`).
/// YaRN interpolates by a ramp over the dimensions where the ORIGINAL
/// window held between `beta_slow` and `beta_fast` full rotations — low
/// dimensions rotate fast enough that extrapolation is safe, high ones do
/// not. Position-independent arithmetic over `head_dim / 2` values, so it
/// happens once here rather than in every head every token; the same
/// buffer serves every rope dispatch in the model.
#[must_use]
pub fn yarn_inv_freq(g: &GptOssGeometry) -> Vec<f32> {
    let half = (g.head_dim / 2) as usize;
    if half < 1 {
        return vec![0.0];
    }
    let base = g.rope_theta;
    let factor = if g.rope_factor > 0.0 {
        g.rope_factor
    } else {
        1.0
    };
    let dim = g.head_dim as f32;
    let orig = g.rope_original_max_position as f32;

    // The dimension at which `rotations` full turns fit the original window.
    let find_dim = |rotations: f32| {
        dim * (orig / (rotations * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln())
    };
    let mut low = find_dim(g.rope_beta_fast).floor().max(0.0);
    let mut high = find_dim(g.rope_beta_slow).ceil().min(dim - 1.0);
    // mlx nudges the singular case rather than dividing by zero; matched so
    // the ramp is the same function on both sides.
    if low == high {
        high += 0.001;
    }
    let _ = &mut low;

    (0..half)
        .map(|i| {
            let d = (2 * i) as f32;
            let freq_extra = 1.0 / base.powf(d / dim);
            let freq_inter = freq_extra / factor;
            let ramp = if high - low <= 0.0 {
                0.0
            } else {
                ((i as f32 - low) / (high - low)).clamp(0.0, 1.0)
            };
            let mask = 1.0 - ramp;
            freq_inter * (1.0 - mask) + freq_extra * mask
        })
        .collect()
}

/// YaRN's `mscale`, which scales q and k: `0.1 * ln(factor) + 1` — 1.3466
/// at gpt-oss's factor of 32. mlx computes it as a ratio of two
/// `yarn_get_mscale` calls whose denominator is 1 for this config; stated
/// directly, with the shape kept so a checkpoint that did carry
/// `mscale_all_dim` would be visibly unhandled rather than silently wrong.
#[must_use]
pub fn yarn_mscale(g: &GptOssGeometry) -> f32 {
    if g.rope_factor <= 1.0 {
        return 1.0;
    }
    0.1 * g.rope_factor.ln() + 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shapes_are_this_familys_and_routed_looks_dense() {
        let g = GptOssGeometry::default();
        assert_eq!(gptoss_qmv_kn(Kernel::GoQmvQ, &g), KN { k: 2880, n: 4096 });
        assert_eq!(gptoss_qmv_kn(Kernel::GoRouter, &g), KN { k: 2880, n: 32 });
        // The routed projection's constants are indistinguishable from a
        // dense one's — deliberately.
        assert_eq!(
            gptoss_qmv_kn(Kernel::GoExpertGate, &g),
            KN { k: 2880, n: 2880 }
        );
        assert_eq!(
            gptoss_qmv_kn(Kernel::LmHeadUntied, &g),
            KN {
                k: 2880,
                n: 201_088
            }
        );
        assert_eq!(
            gptoss_qmv_kn(Kernel::QmvQ, &g).n,
            0,
            "a qwen kind is not this family's"
        );
    }

    #[test]
    fn yarn_ramps_between_extrapolation_and_interpolation() {
        let g = GptOssGeometry::default();
        let freqs = yarn_inv_freq(&g);
        assert_eq!(freqs.len(), 32);
        // Dimension 0 rotates fastest: safe to extrapolate, so the ORIGINAL
        // frequency survives (1.0 at d = 0).
        assert!((freqs[0] - 1.0).abs() < 1e-6);
        // The last dimension is interpolated: the extended-context
        // frequency, `factor` times slower than the original's.
        let dim = g.head_dim as f32;
        let last_extra = 1.0 / g.rope_theta.powf((2.0 * 31.0) / dim);
        assert!((freqs[31] - last_extra / g.rope_factor).abs() / freqs[31] < 1e-4);
        // Monotone decreasing, as frequencies are.
        assert!(freqs.windows(2).all(|w| w[0] > w[1]));
        // The temperature: 0.1 ln 32 + 1.
        assert!((yarn_mscale(&g) - 1.346_573_6).abs() < 1e-5);
        // No scaling, no correction.
        let plain = GptOssGeometry {
            rope_factor: 1.0,
            ..GptOssGeometry::default()
        };
        assert!((yarn_mscale(&plain) - 1.0).abs() < 1e-9);
    }
}
