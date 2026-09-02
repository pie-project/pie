//! The canonical ETA RNG contract.
//!
//! The deterministic CUDA/C++ and MSL projections of this formula are emitted
//! by `eta-compiler`; this module is the formula itself plus the host
//! implementation the reference interpreter runs.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// One `value ^= value >> xor_shift; value *= multiplier` step of
/// [`splitmix64`].
pub struct SplitMix64Round {
    /// Right-shift distance of the xor-fold.
    pub xor_shift: u32,
    /// Odd multiplier applied after the fold, or `None` for a final fold
    /// with no multiply.
    pub multiplier: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Every constant the ETA RNG is made of, in one value. These numbers are ABI: a backend that reproduces the ops but not the constants produces a different token stream from the same seed. Emitters must project this struct rather than transcribe the literals.
pub struct RngFormula {
    /// The three mixing rounds, applied in order.
    pub splitmix64_rounds: [SplitMix64Round; 3],
    /// Multiplied by the lane index to decorrelate neighbouring lanes.
    pub lane_stride: u64,
    /// Added to the lane index before scaling, so lane `0` is not the
    /// unmixed seed.
    pub lane_index_bias: u64,
    /// How far to shift a mixed word down before taking mantissa bits.
    pub uniform_mantissa_shift: u32,
    /// How many bits of the mixed word become the uniform draw.
    pub uniform_mantissa_bits: u32,
    /// Added to the integer draw before scaling, centring it in its bucket
    /// so neither `0.0` nor `1.0` can be produced.
    pub uniform_midpoint: f32,
    /// Folded into an ambient seed so seed `0` is not the unmixed state.
    pub ambient_seed_xor: u64,
    /// How far the key is shifted above the counter when the two pack into
    /// one keyed seed word.
    pub keyed_word_bits: u32,
}

/// The one instance of [`RngFormula`] that defines ETA's RNG.
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
/// Mixes `value` through [`RNG_FORMULA`]'s rounds.
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
/// The effective 64-bit seed for an ambient draw from `seed`.
pub fn seed_eff(seed: u32) -> u64 {
    seed as u64 ^ RNG_FORMULA.ambient_seed_xor
}

#[inline]
/// The per-stream offset that keeps two [`Op::Rng`](crate::op::Op::Rng) ops
/// in one fire from drawing the same numbers.
pub fn stream_salt(stream: u32) -> u64 {
    splitmix64((stream as u64).wrapping_mul(RNG_FORMULA.lane_stride))
}

#[inline]
/// The effective seed for stream `stream` of ambient seed `seed`.
pub fn seed_eff_stream(seed: u32, stream: u32) -> u64 {
    seed_eff(seed) ^ stream_salt(stream)
}

#[inline]
/// The effective seed for a keyed draw from an `[key, ctr]` state tensor.
///
/// Nothing ambient enters here: the draw is a pure function of the state,
/// which is what makes a replay of the same program bit-identical.
pub fn keyed_seed(key: u32, counter: u32) -> u64 {
    splitmix64(((key as u64) << RNG_FORMULA.keyed_word_bits) | counter as u64)
}

/// The largest `f32` strictly below `1.0`.
///
/// For the top mantissa value, `(bits + 0.5) / 2^24` rounds to `1.0` exactly in `f32`, which breaks any consumer assuming a half-open range: `gumbel = -log(-log(u))` evaluates to `+inf` at `u = 1`, and `+inf` unconditionally wins `argmax(logits + gumbel)`.
pub const UNIFORM_MAX: f32 = 1.0 - f32::EPSILON / 2.0;

#[inline]
/// The uniform draw for lane `index` under effective seed `seed_eff`.
///
/// The result is in `(0, 1)` — never `0.0`, and never above
/// [`UNIFORM_MAX`].
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
