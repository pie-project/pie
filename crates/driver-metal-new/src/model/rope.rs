//! The rotary frequency table, when a base cannot express one.
//!
//! A plain rope is a geometric ladder in `theta` and the shader raises two to
//! a base to get it. A deployment that RESCALES that ladder -- llama-3, YaRN
//! -- is not a ladder in any base, so there is nothing for a text to state and
//! nothing for the shader to derive. `rope_neox_freqs_decode` takes the
//! frequencies as a buffer for exactly this reason and this is where the
//! buffer comes from.
//!
//! Derived at LOAD from the checkpoint's config, which makes it the driver's
//! answer and not the text's -- the same argument
//! [`kernels::Source::KvHeadStride`] makes for the pool's strides. A model
//! states an architecture; a config states a deployment.

/// The rescaling a config asks for, or none.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rescale {
    /// How much the low-frequency end is stretched.
    pub factor: f32,
    /// The wavelength bounds, as divisors of `original_max`.
    pub low: f32,
    /// See [`Self::low`].
    pub high: f32,
    /// The context the ladder was trained at.
    pub original_max: f32,
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
            let Some(r) = rescale else { return freq };
            if r.factor <= 0.0 || r.low <= 0.0 || r.high <= r.low {
                return freq;
            }
            // Wavelength, which is what the bounds are stated in.
            let wl = 2.0 * std::f32::consts::PI / freq;
            let (lo_wl, hi_wl) = (r.original_max / r.low, r.original_max / r.high);
            if wl > lo_wl {
                freq / r.factor
            } else if wl < hi_wl {
                freq
            } else {
                let smooth = (r.original_max / wl - r.low) / (r.high - r.low);
                (1.0 - smooth) * freq / r.factor + smooth * freq
            }
        })
        .collect()
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

    #[test]
    fn llama3_stretches_the_long_wavelengths_and_leaves_the_short_ones() {
        let r = Rescale {
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
