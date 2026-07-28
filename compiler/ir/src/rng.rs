//! The canonical PTIR RNG contract.
//!
//! The deterministic CUDA/C++ and MSL projections of this formula are emitted
//! by `pie-codegen`; this module is the formula itself plus the host
//! implementation the reference interpreter runs.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitMix64Round {
    pub xor_shift: u32,
    pub multiplier: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RngFormula {
    pub splitmix64_rounds: [SplitMix64Round; 3],
    pub lane_stride: u64,
    pub lane_index_bias: u64,
    pub uniform_mantissa_shift: u32,
    pub uniform_mantissa_bits: u32,
    pub uniform_midpoint: f32,
    pub ambient_seed_xor: u64,
    pub keyed_word_bits: u32,
}

pub const RNG_FORMULA: RngFormula = RngFormula {
    splitmix64_rounds: [
        SplitMix64Round {
            xor_shift: 27,
            multiplier: Some(0x3C79_AC49_2BA7_B653),
        },
        SplitMix64Round {
            xor_shift: 33,
            multiplier: Some(0x1C69_B3F7_4AC4_AE35),
        },
        SplitMix64Round {
            xor_shift: 27,
            multiplier: None,
        },
    ],
    lane_stride: 0x9E37_79B9_7F4A_7C15,
    lane_index_bias: 1,
    uniform_mantissa_shift: 40,
    uniform_mantissa_bits: 24,
    uniform_midpoint: 0.5,
    ambient_seed_xor: 0xA5A5_A5A5,
    keyed_word_bits: 32,
};

#[inline]
pub fn splitmix64(mut value: u64) -> u64 {
    for round in RNG_FORMULA.splitmix64_rounds {
        value ^= value >> round.xor_shift;
        if let Some(multiplier) = round.multiplier {
            value = value.wrapping_mul(multiplier);
        }
    }
    value
}

#[inline]
pub fn seed_eff(seed: u32) -> u64 {
    seed as u64 ^ RNG_FORMULA.ambient_seed_xor
}

#[inline]
pub fn stream_salt(stream: u32) -> u64 {
    splitmix64((stream as u64).wrapping_mul(RNG_FORMULA.lane_stride))
}

#[inline]
pub fn seed_eff_stream(seed: u32, stream: u32) -> u64 {
    seed_eff(seed) ^ stream_salt(stream)
}

#[inline]
pub fn keyed_seed(key: u32, counter: u32) -> u64 {
    splitmix64(((key as u64) << RNG_FORMULA.keyed_word_bits) | counter as u64)
}

/// The largest `f32` strictly below `1.0`.
///
/// `(bits + 0.5) / 2^24` is mathematically in `(0, 1)`, but for the single top
/// mantissa value `bits = 2^24 - 1` the quotient is `1 - 2^-25`, which sits
/// exactly halfway between `0x1.fffffep-1` and `1.0` and so rounds to **`1.0`
/// exactly** in `f32`. That one draw in `2^24` breaks every consumer that
/// assumes a half-open range: `gumbel = -log(-log(u))` evaluates to `+inf` at
/// `u = 1`, and `+inf` unconditionally wins `argmax(logits + gumbel)`, so the
/// sampler returns a uniformly random token. At `vocab = 262144` that is
/// `262144 / 2^24 ≈ 1.6 %` of decode steps.
pub const UNIFORM_MAX: f32 = 1.0 - f32::EPSILON / 2.0;

#[inline]
pub fn hash_uniform(seed_eff: u64, index: u32) -> f32 {
    let x = seed_eff.wrapping_add(
        RNG_FORMULA
            .lane_stride
            .wrapping_mul(index as u64 + RNG_FORMULA.lane_index_bias),
    );
    let bits = (splitmix64(x) >> RNG_FORMULA.uniform_mantissa_shift) as u32;
    let denominator = (1u32 << RNG_FORMULA.uniform_mantissa_bits) as f32;
    let raw = (bits as f32 + RNG_FORMULA.uniform_midpoint) * (1.0 / denominator);
    if raw < UNIFORM_MAX { raw } else { UNIFORM_MAX }
}
