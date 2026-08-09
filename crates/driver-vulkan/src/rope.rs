//! The rotary frequency table, for the deployments that cannot state one.
//!
//! A plain rope is a geometric ladder and `rope/neox.comp` raises it itself:
//! `theta = scale * position * exp2(-(i / pair_half) * base)`, where the
//! statement carries `base`. Measured against three real texts, that `base` is
//! `log2(rope_theta)` -- `2^19.931568` is 1_000_000 for qwen3 and `2^17.194603`
//! is 150_000 for gpt-oss -- so a ladder and a base are the same thing said
//! two ways.
//!
//! A deployment that RESCALES the ladder is where they stop being the same
//! thing. llama-3's rescaling is piecewise in wavelength and YaRN's is not a
//! ladder in any base, so there is nothing for a text to state and nothing for
//! a shader to raise. `neox_freqs_mb` and `neox_freqs_decode` take the
//! frequencies as a BUFFER for that reason, and this is where the buffer comes
//! from.
//!
//! Derived from the deployment's config rather than read off the plan, for the
//! same reason the pool's page size is: a model states an architecture and a
//! config states a deployment. This is the driver's answer.
//!
//! Ported from `driver-metal`'s `model::rope`, and the port is checked against
//! the shader rather than against the original -- see
//! `the_ladder_this_driver_builds_is_the_one_the_shader_raises`.

/// The rescaling a config asks for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rescale {
    /// How much the low-frequency end is stretched.
    pub factor: f32,
    /// The wavelength bounds, as divisors of [`Self::original_max`].
    pub low: f32,
    /// See [`Self::low`].
    pub high: f32,
    /// The context the ladder was trained at.
    pub original_max: f32,
}

/// `[rotary_dims / 2]` inverse frequencies for `head_dim` at `theta`.
///
/// Without a rescale this is exactly the ladder the shader would have raised,
/// which is the point: one code path, and a deployment that rescales differs
/// only in the table it is handed.
///
/// llama-3's rescaling is piecewise in WAVELENGTH. Frequencies whose
/// wavelength runs past `original_max / low` are divided by the factor --
/// stretched, so a long context does not run off the end of what was trained.
/// Those under `original_max / high` are left exactly alone. Between them the
/// two blend, which is what keeps the ladder continuous; a hard switch puts a
/// discontinuity in the middle of the channels.
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

/// The ladder as the words a fire table carries.
///
/// `FireTable::RopeFrequencies` is a buffer of `float`, and a table is stated
/// in `u32`. The bits, not a cast: `4.0f32 as u32` is 4 and this has to be
/// `0x4080_0000`.
#[must_use]
pub fn words(head_dim: u32, theta: f32, rescale: Option<Rescale>) -> Vec<u32> {
    frequencies(head_dim, theta, rescale)
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{Rescale, frequencies, words};

    /// A plain ladder is what the base form raises, channel for channel.
    ///
    /// The shader computes `exp2(-(i / pair_half) * base)` and a real plan
    /// states `base = log2(rope_theta)`, so this is the arithmetic identity
    /// the two forms rest on. Checked here over every channel; the card
    /// checks it through two compiled shaders.
    #[test]
    fn the_ladder_is_what_the_base_form_raises() {
        let (head_dim, theta) = (128u32, 1_000_000.0f32);
        let base = theta.log2();
        let pair_half = head_dim / 2;
        let f = frequencies(head_dim, theta, None);
        assert_eq!(f.len(), pair_half as usize);
        for (i, got) in f.iter().enumerate() {
            let want = (-(i as f32 / pair_half as f32) * base).exp2();
            assert!(
                (got - want).abs() <= 1e-6 * want.abs().max(1e-6),
                "channel {i}: the ladder says {got} and the shader raises {want}"
            );
        }
        // The first channel is 1 and they fall from there. A ladder that came
        // out constant would satisfy a comparison against itself.
        assert!((f[0] - 1.0).abs() < 1e-6);
        assert!(f.windows(2).all(|w| w[0] > w[1]), "the ladder is monotone");
        assert!(
            f[f.len() - 1] < 1e-5,
            "the last channel is {} and the ladder barely falls",
            f[f.len() - 1]
        );
    }

    /// llama-3 stretches the long wavelengths and leaves the short ones.
    #[test]
    fn a_rescale_moves_the_low_end_and_nothing_else() {
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
        // Nothing in between escapes the interval the two ends bound.
        assert!(
            plain
                .iter()
                .zip(&scaled)
                .all(|(p, s)| *s <= *p + 1e-9 && *s >= p / 32.0 - 1e-9)
        );
        let ratios: Vec<f32> = plain.iter().zip(&scaled).map(|(p, s)| s / p).collect();
        // The rescale did something. A factor that landed nowhere would pass
        // every bound above.
        assert!(
            ratios.iter().any(|r| *r < 0.5),
            "no channel was stretched, so the bounds prove nothing"
        );
        // The blend is not a hard switch. Stated as the difference from one,
        // rather than as a bound on how fast the ratios may fall: the band
        // spans four channels of a 64-wide head, so a continuous blend is
        // necessarily steep there and a bound on steepness would be measuring
        // the channel count.
        //
        // A hard switch is the same piecewise rule with the middle branch
        // replaced by either of its neighbours, so this is what the smoothing
        // adds and nothing else.
        let (lo_wl, hi_wl) = (r.original_max / r.low, r.original_max / r.high);
        let switched: Vec<f32> = plain
            .iter()
            .map(|f| {
                let wl = 2.0 * std::f32::consts::PI / f;
                if wl > lo_wl { f / r.factor } else { *f }
            })
            .collect();
        let band = plain
            .iter()
            .filter(|f| {
                let wl = 2.0 * std::f32::consts::PI / **f;
                wl <= lo_wl && wl >= hi_wl
            })
            .count();
        assert!(band >= 2, "only {band} channels are in the band");
        assert!(
            scaled
                .iter()
                .zip(&switched)
                .any(|(s, h)| (s - h).abs() > 1e-9 * h.abs().max(1e-9)),
            "the smoothed ladder is the hard-switched one"
        );
        // And they agree outside it, so the difference is the band's.
        for (i, (s, h)) in scaled.iter().zip(&switched).enumerate() {
            let wl = 2.0 * std::f32::consts::PI / plain[i];
            if wl > lo_wl || wl < hi_wl {
                assert!(
                    (s - h).abs() <= 1e-9 * h.abs().max(1e-9),
                    "channel {i} is outside the band and the two rules disagree"
                );
            }
        }
    }

    /// A degenerate rescale is the plain ladder rather than a NaN.
    ///
    /// A config with a zero factor or an inverted band is a config, not a
    /// crash, and a NaN in the table makes every rotation a NaN -- which
    /// reaches the whole rest of the plan.
    #[test]
    fn a_rescale_that_says_nothing_leaves_the_ladder_alone() {
        let plain = frequencies(64, 500_000.0, None);
        for bad in [
            Rescale {
                factor: 0.0,
                low: 1.0,
                high: 4.0,
                original_max: 8192.0,
            },
            Rescale {
                factor: 32.0,
                low: 4.0,
                high: 1.0,
                original_max: 8192.0,
            },
            Rescale {
                factor: 32.0,
                low: 0.0,
                high: 4.0,
                original_max: 8192.0,
            },
        ] {
            assert_eq!(frequencies(64, 500_000.0, Some(bad)), plain, "{bad:?}");
        }
        assert!(plain.iter().all(|f| f.is_finite()));
    }

    /// The table is the BITS of the floats, not the floats cast to integers.
    #[test]
    fn the_words_a_table_carries_are_the_bit_patterns() {
        let w = words(128, 1_000_000.0, None);
        let f = frequencies(128, 1_000_000.0, None);
        assert_eq!(w.len(), f.len());
        assert_eq!(w[0], 1.0f32.to_bits(), "the first channel is 1.0");
        assert_ne!(w[0], 1, "a cast rather than the bits");
        assert!(
            w.iter().zip(&f).all(|(a, b)| f32::from_bits(*a) == *b),
            "the words do not read back as the ladder"
        );
    }

    /// A head dim of zero asks for a ladder of one rather than of none.
    ///
    /// An empty table binds a zero-length range, and a shader indexing it
    /// reads whatever the driver decides that means.
    #[test]
    fn a_degenerate_head_dim_still_states_a_channel() {
        assert_eq!(frequencies(0, 1_000_000.0, None).len(), 1);
        assert_eq!(frequencies(1, 1_000_000.0, None).len(), 1);
    }
}
