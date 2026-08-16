//! The rotary frequency table, when a base cannot express one.
//!
//! A plain rope is a geometric ladder in `theta` and the shader raises two to
//! a base to get it. A deployment that RESCALES that ladder -- llama-3, YaRN
//! -- is not a ladder in any base, so there is nothing for a text to state and
//! nothing for the shader to derive. `rope_neox_freqs_decode` takes the
//! frequencies as a buffer for exactly this reason and this is where the
//! buffer comes from.
//!
//! Both kinds are derived here now. YaRN was named in this paragraph from the
//! day it was written and implemented in neither this module nor the
//! geometry, which declined a YaRN row rather than serve it an unrescaled
//! ladder -- an honest refusal, and one that meant gpt-oss reached a Metal
//! text, lowered, staged its weights, and was then turned away by the last
//! thing that reads its config. The refusal named exactly one missing thing
//! and [`Rescale::Yarn`] is it.
//!
//! Derived at LOAD from the checkpoint's config, which makes it the driver's
//! answer and not the text's -- the same argument
//! [`kernels::Source::Named(<kernels::keys::KvHeadStride as kernels::keys::Fact>::KEY)`] makes for the pool's strides. A model
//! states an architecture; a config states a deployment.

/// The rescaling a config asks for, or none.
///
/// One variant per `rope_type` the catalog states, because the two are not
/// the same recipe with different constants: llama-3 is piecewise in
/// WAVELENGTH and YaRN is a ramp in ROTATIONS-per-token, and the channels
/// each leaves alone are found by different arithmetic. This was one struct
/// with llama-3's four fields, which is why a YaRN row had nowhere to land
/// and [`super::super::batch::geometry`] declined it.
///
/// YaRN's fifth number, `attention_factor`, is deliberately NOT here. It
/// scales the attention LOGITS, not the ladder -- HF multiplies `cos` and
/// `sin` by it, which is the same as multiplying the rotated `q` and `k`, so
/// what reaches the softmax is the dot product times its SQUARE. A frequency
/// table cannot carry an amplitude, so it travels as
/// [`model::shared::llama_like::forward::LlamaLikeMetalFacts::attn_scale`]
/// instead and this module would silently drop it if it took it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rescale {
    /// llama-3's piecewise-by-wavelength rescaling (`rope_type: "llama3"`).
    Piecewise {
        /// How much the low-frequency end is stretched.
        factor: f32,
        /// The wavelength bounds, as divisors of `original_max`.
        low: f32,
        /// See [`Self::Piecewise::low`].
        high: f32,
        /// The context the ladder was trained at.
        original_max: f32,
    },
    /// YaRN's NTK-by-parts interpolation (`rope_type: "yarn"`).
    Yarn {
        /// The context-extension ratio; the interpolated end divides by it.
        factor: f32,
        /// The high-rotation end of the ramp, in rotations per trained
        /// context. Channels turning FASTER than this are extrapolated,
        /// which is to say left alone.
        beta_fast: f32,
        /// The low-rotation end. Channels turning slower than this are
        /// interpolated, which is to say divided by `factor`.
        beta_slow: f32,
        /// The context the ladder was trained at.
        original_max: f32,
        /// Whether the ramp's ends are snapped to whole channels.
        ///
        /// A config's own field, not a convention: HF defaults it to true
        /// and gpt-oss writes `false`, which moves the ramp's ends from
        /// `(8, 18)` to `(8.09, 17.40)` and changes ten of thirty-two
        /// channels. A driver that assumed either one would serve the other
        /// family with a ladder nobody trained.
        truncate: bool,
    },
}

/// The ladder a GEOMETRY asks for.
///
/// One call, so the derivation has one site. What stood here was five lines
/// spelled out at the load and hand-copied into ten places in
/// `device_real_weights.rs` -- and a hand-copy of the driver's derivation is
/// the one thing a numeric gate must not hold, because the copies go stale
/// silently and the test then compares the model against the ladder the
/// driver USED TO build. That is a gate which cannot fail for the reason it
/// exists.
#[must_use]
pub fn table(geometry: &crate::batch::DecodeGeometry) -> Vec<f32> {
    frequencies(
        geometry.head_dim,
        geometry.rope_theta,
        geometry.rope_rescale,
    )
}

/// `[rotary_dims/2]` inverse frequencies for `head_dim` at `theta`.
///
/// Without a rescale this is the same ladder the shader would have raised
/// itself, which is the point: one code path, and a deployment that rescales
/// differs only in the table it is handed.
///
/// llama-3's rescaling is piecewise in WAVELENGTH. Frequencies whose
/// wavelength runs past `original_max / low` are divided by the factor --
/// stretched, so a long context does not run off the end of what was trained.
/// Those under `original_max / high` are left exactly alone. Between them the
/// two are blended, which is what keeps the ladder continuous; a hard
/// switch there puts a discontinuity in the middle of the channels.
#[must_use]
pub fn frequencies(head_dim: u32, theta: f32, rescale: Option<Rescale>) -> Vec<f32> {
    let half = (head_dim / 2).max(1);
    (0..half)
        .map(|i| {
            let freq = theta.powf(-(2.0 * i as f32) / head_dim as f32);
            match rescale {
                None => freq,
                Some(Rescale::Piecewise {
                    factor,
                    low,
                    high,
                    original_max,
                }) => {
                    if factor <= 0.0 || low <= 0.0 || high <= low {
                        return freq;
                    }
                    // Wavelength, which is what the bounds are stated in.
                    let wl = 2.0 * std::f32::consts::PI / freq;
                    let (lo_wl, hi_wl) = (original_max / low, original_max / high);
                    if wl > lo_wl {
                        freq / factor
                    } else if wl < hi_wl {
                        freq
                    } else {
                        let smooth = (original_max / wl - low) / (high - low);
                        (1.0 - smooth) * freq / factor + smooth * freq
                    }
                }
                Some(Rescale::Yarn {
                    factor,
                    beta_fast,
                    beta_slow,
                    original_max,
                    truncate,
                }) => {
                    if factor <= 0.0 {
                        return freq;
                    }
                    let (lo, hi) = ramp_ends(
                        head_dim,
                        theta,
                        original_max,
                        beta_fast,
                        beta_slow,
                        truncate,
                    );
                    // Zero at the fast end and one at the slow end, so the
                    // ladder walks from untouched to fully divided across
                    // the band the two betas bound.
                    let ramp = ((i as f32 - lo) / (hi - lo)).clamp(0.0, 1.0);
                    ramp * (freq / factor) + (1.0 - ramp) * freq
                }
            }
        })
        .collect()
}

/// The channel indices YaRN's ramp starts and ends at.
///
/// A channel completes `original_max / wavelength` rotations over the trained
/// context, and the betas are stated in exactly those rotations -- so the
/// ends are found by inverting `wavelength(i)` rather than by walking the
/// table. Solving `2pi * theta^(2i/head_dim) = original_max / rotations` for
/// `i` gives the expression below; it is HF's `find_correction_dim` and
/// spelled to match it, because a ladder that differs from the publisher's in
/// the tenth channel is the kind of wrong this driver cannot see.
fn ramp_ends(
    head_dim: u32,
    theta: f32,
    original_max: f32,
    beta_fast: f32,
    beta_slow: f32,
    truncate: bool,
) -> (f32, f32) {
    let dim = head_dim as f32;
    let at = |rotations: f32| {
        dim * (original_max / (rotations * 2.0 * std::f32::consts::PI)).ln() / (2.0 * theta.ln())
    };
    let (mut lo, mut hi) = (at(beta_fast), at(beta_slow));
    if truncate {
        lo = lo.floor();
        hi = hi.ceil();
    }
    // HF clamps against the FULL head_dim, not the half the table has, and
    // the two differ for a short head with wide betas. Matched deliberately.
    let (lo, mut hi) = (lo.max(0.0), hi.min(dim - 1.0));
    if hi <= lo {
        // A degenerate band is a step, not a division by zero.
        hi = lo + 0.001;
    }
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_rescale_is_the_ladder_the_shader_would_have_raised() {
        let f = frequencies(64, 500_000.0, None);
        assert_eq!(f.len(), 32);
        assert!((f[0] - 1.0).abs() < 1e-6, "the first channel is unscaled");
        // Each channel is `theta^(-2i/head_dim)`, so they fall monotonically.
        assert!(f.windows(2).all(|w| w[0] > w[1]));
    }

    /// gpt-oss's own numbers, against the publisher's arithmetic.
    ///
    /// The four oracle values were computed from HuggingFace's
    /// `_compute_yarn_parameters` in f64 and written down, so this compares
    /// two independent evaluations of one formula rather than this file
    /// against itself. `head_dim` 64, `theta` 150000, factor 32 over a
    /// trained 4096, betas 32 and 1, `truncate` false -- every one off
    /// `config.json`.
    #[test]
    fn the_yarn_ladder_is_the_one_the_publisher_computes() {
        let f = frequencies(64, 150_000.0, Some(gpt_oss()));
        assert_eq!(f.len(), 32);
        for (i, want) in [
            (0usize, 1.0f32),
            (8, 0.050_813_275),
            (12, 0.006_794_959_5),
            (31, 3.023_511_4e-7),
        ] {
            let rel = (f[i] - want).abs() / want;
            assert!(rel < 1e-5, "channel {i}: {} is not {want} ({rel})", f[i]);
        }
    }

    /// The fast end is untouched and the slow end divides by the whole
    /// factor, which is the claim `Rescale::Yarn`'s doc makes in words.
    #[test]
    fn yarn_extrapolates_the_fast_channels_and_interpolates_the_slow_ones() {
        let plain = frequencies(64, 150_000.0, None);
        let scaled = frequencies(64, 150_000.0, Some(gpt_oss()));
        assert!(
            (plain[0] - scaled[0]).abs() < 1e-9,
            "channel 0 turns fastest"
        );
        let last = plain.len() - 1;
        let rel = (scaled[last] - plain[last] / 32.0).abs() / (plain[last] / 32.0);
        assert!(rel < 1e-5, "the slow end divides by 32: {rel}");
        // And no channel escapes the interval the two ends bound.
        assert!(
            plain
                .iter()
                .zip(&scaled)
                .all(|(p, s)| *s <= *p * (1.0 + 1e-6) && *s >= p / 32.0 * (1.0 - 1e-6))
        );
    }

    /// WHY `truncate` is a field and not a convention.
    ///
    /// HF defaults it to true; gpt-oss's config writes false. If this driver
    /// picked either one it would serve the other family a ladder nobody
    /// trained, and the difference is not in the last decimal: it moves the
    /// ramp's ends from `(8, 18)` to `(8.09, 17.40)` and channel 12 by three
    /// percent. A rope error of that size does not fail -- it attends with
    /// the wrong wavelengths and degrades fluently, which is the whole
    /// reason the geometry refused YaRN rather than zeroing it.
    #[test]
    fn truncating_the_ramp_moves_the_channels_between_its_ends() {
        let Rescale::Yarn {
            factor,
            beta_fast,
            beta_slow,
            original_max,
            ..
        } = gpt_oss()
        else {
            unreachable!()
        };
        let snapped = Rescale::Yarn {
            factor,
            beta_fast,
            beta_slow,
            original_max,
            truncate: true,
        };
        let a = frequencies(64, 150_000.0, Some(gpt_oss()));
        let b = frequencies(64, 150_000.0, Some(snapped));
        let rel = (b[12] - a[12]).abs() / a[12];
        assert!(
            (rel - 0.032_487_967).abs() < 1e-4,
            "channel 12 differs by {rel}, not the 3.2% the two ramps imply"
        );
        // Outside the band the two agree exactly, which is what makes the
        // difference above a RAMP difference and not a scale one.
        assert!((a[0] - b[0]).abs() < 1e-9 && (a[31] - b[31]).abs() < 1e-12);
    }

    fn gpt_oss() -> Rescale {
        Rescale::Yarn {
            factor: 32.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max: 4096.0,
            truncate: false,
        }
    }

    #[test]
    fn llama3_stretches_the_long_wavelengths_and_leaves_the_short_ones() {
        let r = Rescale::Piecewise {
            factor: 32.0,
            low: 1.0,
            high: 4.0,
            original_max: 8192.0,
        };
        let plain = frequencies(64, 500_000.0, None);
        let scaled = frequencies(64, 500_000.0, Some(r));
        // The FIRST channel is the shortest wavelength and is left alone.
        assert!((plain[0] - scaled[0]).abs() < 1e-6);
        // The LAST is the longest and is stretched by the whole factor.
        let last = plain.len() - 1;
        assert!(
            (scaled[last] - plain[last] / 32.0).abs() < 1e-9,
            "the low-frequency end divides by the factor: {} vs {}",
            scaled[last],
            plain[last] / 32.0
        );
        // And nothing in between escapes the interval the two ends bound.
        assert!(
            plain
                .iter()
                .zip(&scaled)
                .all(|(p, s)| *s <= *p + 1e-9 && *s >= p / 32.0 - 1e-9)
        );
    }
}
