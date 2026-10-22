//! A checkpoint's `rope_scaling` (or `rope_parameters`), as `config.json`
//! states it, turned into the frequencies a rotation turns at: each pair's
//! inverse frequency and the scale its cos and sin carry. The formulas are
//! vLLM's (`vllm/model_executor/layers/rotary_embedding`), which are
//! transformers' with every length-dependent choice made once, at the
//! longest context served.

use poem_ir::RopeFreqs;

/// The rope scalings a checkpoint may state.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScaling {
    /// Every position read `factor` times closer (`"linear"`).
    Linear { factor: f64 },
    /// NTK-aware base growth (`"dynamic"`), fixed once as vLLM fixes it:
    /// `theta * (factor * max / trained - (factor - 1)) ^ (d / (d - 2))`,
    /// `trained` the dict's `max_trained_positions`, else `max` (no growth).
    Dynamic {
        factor: f64,
        max_position: Option<u32>,
        max_trained: Option<u32>,
    },
    /// YaRN (`"yarn"`): pairs below `beta_fast` rotations at the original
    /// length keep their frequency, pairs above `beta_slow` are interpolated
    /// `factor` times, a ramp between; cos and sin scaled by the attention
    /// factor.
    Yarn {
        factor: f64,
        original_max_position: u32,
        beta_fast: f64,
        beta_slow: f64,
        mscale: Option<f64>,
        mscale_all_dim: Option<f64>,
        attention_factor: Option<f64>,
        truncate: bool,
    },
    /// Llama 3's (`"llama3"`): wavelengths shorter than the original length
    /// over `high_freq_factor` kept, longer than it over `low_freq_factor`
    /// interpolated `factor` times, a smooth blend between.
    Llama3 {
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        original_max_position: u32,
    },
    /// Phi's LongRoPE (`"longrope"`): each pair's frequency divided by its
    /// own factor. The long factors turn every position when the context
    /// served passes the original one (vLLM's choice), the short ones
    /// otherwise.
    LongRope {
        short_factor: Vec<f64>,
        long_factor: Vec<f64>,
        original_max_position: u32,
        max_position: u32,
        short_mscale: Option<f64>,
        long_mscale: Option<f64>,
        long: bool,
    },
}

/// What a scaling's dict holds, read by key.
pub trait Entries {
    fn number(&self, key: &str) -> anyhow::Result<Option<f64>>;
    fn numbers(&self, key: &str) -> anyhow::Result<Option<Vec<f64>>>;
    fn text(&self, key: &str) -> anyhow::Result<Option<String>>;
    fn flag(&self, key: &str) -> anyhow::Result<Option<bool>>;
}

fn whole(key: &str, n: f64) -> anyhow::Result<u32> {
    if n.fract() != 0.0 || n <= 0.0 || n > f64::from(u32::MAX) {
        anyhow::bail!("rope scaling's `{key}` is a positive whole length, not {n}");
    }
    Ok(n as u32)
}

impl RopeScaling {
    /// The scaling a `rope_scaling` dict states, `max_position` the
    /// checkpoint's `max_position_embeddings` (what dynamic NTK and LongRoPE
    /// read beside the dict); `None` for a rotation that turns as stated
    /// (`"default"`, Gemma's `"proportional"`, a dict without a type).
    pub fn of(d: &impl Entries, max_position: Option<u32>) -> anyhow::Result<Option<Self>> {
        let kind = match d.text("rope_type")? {
            Some(kind) => kind,
            None => match d.text("type")? {
                Some(kind) => kind,
                None => return Ok(None),
            },
        };
        let need = |key: &str| -> anyhow::Result<f64> {
            d.number(key)?
                .ok_or_else(|| anyhow::anyhow!("a {kind} rope scaling states `{key}`"))
        };
        let original = |key: &str| -> anyhow::Result<u32> { whole(key, need(key)?) };
        Ok(Some(match kind.as_str() {
            "default" | "proportional" => return Ok(None),
            "linear" => Self::Linear {
                factor: need("factor")?,
            },
            "dynamic" => Self::Dynamic {
                factor: need("factor")?,
                max_position,
                max_trained: d
                    .number("max_trained_positions")?
                    .map(|n| whole("max_trained_positions", n))
                    .transpose()?,
            },
            "yarn" => Self::Yarn {
                factor: need("factor")?,
                original_max_position: original("original_max_position_embeddings")?,
                beta_fast: d.number("beta_fast")?.unwrap_or(32.0),
                beta_slow: d.number("beta_slow")?.unwrap_or(1.0),
                mscale: d.number("mscale")?,
                mscale_all_dim: d.number("mscale_all_dim")?,
                attention_factor: d.number("attention_factor")?,
                truncate: d.flag("truncate")?.unwrap_or(true),
            },
            "llama3" => Self::Llama3 {
                factor: need("factor")?,
                low_freq_factor: need("low_freq_factor")?,
                high_freq_factor: need("high_freq_factor")?,
                original_max_position: original("original_max_position_embeddings")?,
            },
            "longrope" | "su" => {
                let original_max_position = original("original_max_position_embeddings")?;
                let max_position = max_position.ok_or_else(|| {
                    anyhow::anyhow!(
                        "a longrope scaling reads the checkpoint's max_position_embeddings"
                    )
                })?;
                let factors = |key: &str| -> anyhow::Result<Vec<f64>> {
                    d.numbers(key)?
                        .ok_or_else(|| anyhow::anyhow!("a longrope scaling states `{key}`"))
                };
                Self::LongRope {
                    short_factor: factors("short_factor")?,
                    long_factor: factors("long_factor")?,
                    original_max_position,
                    max_position,
                    short_mscale: d.number("short_mscale")?,
                    long_mscale: d.number("long_mscale")?,
                    long: max_position > original_max_position,
                }
            }
            other => anyhow::bail!(
                "a rope scaling of type `{other}` is not one pie turns (linear, dynamic, yarn, \
                 llama3, longrope)"
            ),
        }))
    }

    /// The context this scaling stretches the model to, where it says.
    #[must_use]
    pub fn context(&self) -> Option<u32> {
        match self {
            Self::Yarn {
                factor,
                original_max_position,
                ..
            } => Some((f64::from(*original_max_position) * factor) as u32),
            Self::LongRope { max_position, .. } => Some(*max_position),
            _ => None,
        }
    }

    /// The frequencies of a rotation of `pairs` pairs, pair `i` turning at
    /// `theta^(-i/span)` unscaled (`span` is half the rotary width for a
    /// plain rotation, the dims the scaling's formulas are stated over).
    pub fn freqs(&self, theta: f32, pairs: u32, span: u32) -> anyhow::Result<RopeFreqs> {
        if pairs == 0 || span == 0 || pairs > span {
            anyhow::bail!("a rotation of {pairs} pairs {span} apart has no frequencies to scale");
        }
        let theta = f64::from(theta);
        let dim = 2.0 * f64::from(span);
        let plain = |base: f64| -> Vec<f64> {
            (0..pairs)
                .map(|i| base.powf(-2.0 * f64::from(i) / dim))
                .collect()
        };
        let (inv, mscale) = match self {
            Self::Linear { factor } => {
                (plain(theta).into_iter().map(|f| f / factor).collect(), 1.0)
            }
            Self::Dynamic {
                factor,
                max_position,
                max_trained,
            } => {
                let max = f64::from(max_position.unwrap_or(1));
                let trained = max_trained.map_or(max, f64::from);
                let grown =
                    theta * (factor * max / trained - (factor - 1.0)).powf(dim / (dim - 2.0));
                (plain(grown), 1.0)
            }
            Self::Yarn {
                factor,
                original_max_position,
                beta_fast,
                beta_slow,
                mscale,
                mscale_all_dim,
                attention_factor,
                truncate,
            } => {
                let correction = |rotations: f64| {
                    dim * (f64::from(*original_max_position) / (rotations * core::f64::consts::TAU))
                        .ln()
                        / (2.0 * theta.ln())
                };
                let (mut low, mut high) = (correction(*beta_fast), correction(*beta_slow));
                if *truncate {
                    low = low.floor();
                    high = high.ceil();
                }
                let (low, mut high) = (low.max(0.0), high.min(dim - 1.0));
                if low == high {
                    high += 0.001;
                }
                let inv = plain(theta)
                    .into_iter()
                    .enumerate()
                    .map(|(i, extrapolated)| {
                        let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
                        extrapolated / factor * ramp + extrapolated * (1.0 - ramp)
                    })
                    .collect();
                let get = |m: f64| {
                    if *factor <= 1.0 {
                        1.0
                    } else {
                        0.1 * m * factor.ln() + 1.0
                    }
                };
                let mscale = match (attention_factor, mscale, mscale_all_dim) {
                    (Some(a), _, _) => *a,
                    (None, Some(m), Some(all)) if *m != 0.0 && *all != 0.0 => get(*m) / get(*all),
                    _ => get(1.0),
                };
                (inv, mscale)
            }
            Self::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position,
            } => {
                let original = f64::from(*original_max_position);
                let (low_wavelen, high_wavelen) =
                    (original / low_freq_factor, original / high_freq_factor);
                let inv = plain(theta)
                    .into_iter()
                    .map(|f| {
                        let wavelen = core::f64::consts::TAU / f;
                        if wavelen < high_wavelen {
                            f
                        } else if wavelen > low_wavelen {
                            f / factor
                        } else {
                            let smooth = if low_freq_factor == high_freq_factor {
                                0.0
                            } else {
                                (original / wavelen - low_freq_factor)
                                    / (high_freq_factor - low_freq_factor)
                            };
                            (1.0 - smooth) * f / factor + smooth * f
                        }
                    })
                    .collect();
                (inv, 1.0)
            }
            Self::LongRope {
                short_factor,
                long_factor,
                original_max_position,
                max_position,
                short_mscale,
                long_mscale,
                long,
            } => {
                let factors = if *long { long_factor } else { short_factor };
                if factors.len() != pairs as usize {
                    anyhow::bail!(
                        "a longrope scaling states {} factors for a rotation of {pairs} pairs",
                        factors.len()
                    );
                }
                let scale = f64::from(*max_position) / f64::from(*original_max_position);
                let default = if scale <= 1.0 {
                    1.0
                } else {
                    (1.0 + scale.ln() / f64::from(*original_max_position).ln()).sqrt()
                };
                let mscale = if *long { long_mscale } else { short_mscale }.unwrap_or(default);
                let inv = plain(theta)
                    .into_iter()
                    .zip(factors)
                    .map(|(f, by)| f / by)
                    .collect();
                (inv, mscale)
            }
        };
        let inv_freq: Vec<f32> = inv.into_iter().map(|f: f64| f as f32).collect();
        Ok(RopeFreqs {
            inv_freq,
            mscale: mscale as f32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Dict(Vec<(&'static str, f64)>, &'static str);

    impl Entries for Dict {
        fn number(&self, key: &str) -> anyhow::Result<Option<f64>> {
            Ok(self.0.iter().find(|(k, _)| *k == key).map(|(_, v)| *v))
        }
        fn numbers(&self, _: &str) -> anyhow::Result<Option<Vec<f64>>> {
            Ok(None)
        }
        fn text(&self, key: &str) -> anyhow::Result<Option<String>> {
            Ok((key == "rope_type").then(|| self.1.to_string()))
        }
        fn flag(&self, _: &str) -> anyhow::Result<Option<bool>> {
            Ok(None)
        }
    }

    fn close(a: f32, b: f64) {
        assert!(
            (f64::from(a) - b).abs() <= 1e-6 * b.abs().max(1e-30),
            "{a} is not {b}"
        );
    }

    /// Llama 3.1's table, against vLLM's `Llama3RotaryEmbedding` at head 128.
    #[test]
    fn llama3_keeps_the_short_waves_and_stretches_the_long() {
        let d = Dict(
            vec![
                ("factor", 8.0),
                ("low_freq_factor", 1.0),
                ("high_freq_factor", 4.0),
                ("original_max_position_embeddings", 8192.0),
            ],
            "llama3",
        );
        let s = RopeScaling::of(&d, None).unwrap().unwrap();
        let f = s.freqs(500_000.0, 64, 64).unwrap();
        assert_eq!(f.mscale, 1.0);
        close(f.inv_freq[0], 1.0);
        let plain = |i: u32| 500_000f64.powf(-f64::from(i) / 64.0);
        close(f.inv_freq[10], plain(10));
        close(f.inv_freq[63], plain(63) / 8.0);
    }

    /// Qwen's YaRN at factor 4 over 32768, head 128: the attention factor
    /// is 0.1 ln 4 + 1, the fastest pair kept, the slowest interpolated.
    #[test]
    fn yarn_ramps_between_kept_and_interpolated() {
        let d = Dict(
            vec![
                ("factor", 4.0),
                ("original_max_position_embeddings", 32768.0),
            ],
            "yarn",
        );
        let s = RopeScaling::of(&d, None).unwrap().unwrap();
        let f = s.freqs(1_000_000.0, 64, 64).unwrap();
        close(f.mscale, 0.1 * 4f64.ln() + 1.0);
        close(f.inv_freq[0], 1.0);
        close(f.inv_freq[63], 1_000_000f64.powf(-63.0 / 64.0) / 4.0);
        assert_eq!(s.context(), Some(131_072));
    }

    #[test]
    fn linear_and_dynamic_move_every_pair() {
        let lin = RopeScaling::of(&Dict(vec![("factor", 2.0)], "linear"), None)
            .unwrap()
            .unwrap();
        let f = lin.freqs(10_000.0, 4, 4).unwrap();
        close(f.inv_freq[1], 10_000f64.powf(-0.25) / 2.0);
        // vLLM's dynamic NTK grows the base only past a stated trained
        // length: at the checkpoint's own it turns as stated.
        let dynamic = RopeScaling::of(&Dict(vec![("factor", 2.0)], "dynamic"), Some(4096))
            .unwrap()
            .unwrap();
        let f = dynamic.freqs(10_000.0, 4, 4).unwrap();
        close(f.inv_freq[1], 10_000f64.powf(-0.25));
        let trained = Dict(
            vec![("factor", 2.0), ("max_trained_positions", 2048.0)],
            "dynamic",
        );
        let dynamic = RopeScaling::of(&trained, Some(4096)).unwrap().unwrap();
        let f = dynamic.freqs(10_000.0, 4, 4).unwrap();
        let grown = 10_000.0 * 3f64.powf(8.0 / 6.0);
        close(f.inv_freq[1], grown.powf(-0.25));
    }

    #[test]
    fn a_default_rotation_is_not_scaled() {
        assert!(
            RopeScaling::of(&Dict(vec![], "default"), None)
                .unwrap()
                .is_none()
        );
        assert!(RopeScaling::of(&Dict(vec![], "su2"), None).is_err());
    }
}
