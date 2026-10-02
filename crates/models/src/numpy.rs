//! `numpy.random.default_rng(seed).integers(0, high, size)` — bit for bit.
//!
//! DeepSeek's Engram derives its per-module n-gram hash multipliers from
//! NumPy's default generator seeded by the module's layer id, so a checkpoint
//! can only be hashed the way it was trained by reproducing NumPy's
//! `SeedSequence` entropy mixing, the PCG64 (XSL-RR 128/64) stream, and the
//! generator's Lemire bounded draw. Pinned against NumPy in the tests.

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;
const POOL_SIZE: usize = 4;

const PCG_MULTIPLIER: u128 =
    ((2_549_297_995_355_413_924u128) << 64) | 4_865_540_595_714_422_341u128;

/// `numpy.random.SeedSequence(entropy)` for an integer entropy.
pub struct SeedSequence {
    pool: [u32; POOL_SIZE],
}

impl SeedSequence {
    #[must_use]
    pub fn new(entropy: u128) -> SeedSequence {
        // `_coerce_to_uint32_array`: little-endian 32-bit words, `[0]` for zero.
        let mut words = Vec::new();
        let mut e = entropy;
        if e == 0 {
            words.push(0u32);
        }
        while e != 0 {
            words.push((e & 0xFFFF_FFFF) as u32);
            e >>= 32;
        }
        let mut hash_const = INIT_A;
        let mut hashmix = |mut value: u32| -> u32 {
            value ^= hash_const;
            hash_const = hash_const.wrapping_mul(MULT_A);
            value = value.wrapping_mul(hash_const);
            value ^ (value >> XSHIFT)
        };
        let mix = |x: u32, y: u32| -> u32 {
            let result = MIX_MULT_L
                .wrapping_mul(x)
                .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
            result ^ (result >> XSHIFT)
        };
        let mut pool = [0u32; POOL_SIZE];
        for (i, slot) in pool.iter_mut().enumerate() {
            *slot = hashmix(words.get(i).copied().unwrap_or(0));
        }
        for i_src in 0..POOL_SIZE {
            for i_dst in 0..POOL_SIZE {
                if i_src != i_dst {
                    let h = hashmix(pool[i_src]);
                    pool[i_dst] = mix(pool[i_dst], h);
                }
            }
        }
        for &word in words.iter().skip(POOL_SIZE) {
            for slot in pool.iter_mut() {
                let h = hashmix(word);
                *slot = mix(*slot, h);
            }
        }
        SeedSequence { pool }
    }

    /// `generate_state(n_words, dtype=np.uint64)`.
    #[must_use]
    pub fn generate_state_u64(&self, n_words: usize) -> Vec<u64> {
        let mut hash_const = INIT_B;
        let mut words = Vec::with_capacity(2 * n_words);
        for i_dst in 0..2 * n_words {
            let mut data_val = self.pool[i_dst % POOL_SIZE];
            data_val ^= hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            data_val = data_val.wrapping_mul(hash_const);
            data_val ^= data_val >> XSHIFT;
            words.push(data_val);
        }
        words
            .chunks(2)
            .map(|pair| u64::from(pair[0]) | (u64::from(pair[1]) << 32))
            .collect()
    }
}

/// NumPy's `PCG64` bit generator: PCG XSL-RR 128/64 with the setseq stream.
pub struct Pcg64 {
    state: u128,
    inc: u128,
}

impl Pcg64 {
    #[must_use]
    pub fn from_seed_sequence(seq: &SeedSequence) -> Pcg64 {
        let s = seq.generate_state_u64(4);
        let initstate = (u128::from(s[0]) << 64) | u128::from(s[1]);
        let initseq = (u128::from(s[2]) << 64) | u128::from(s[3]);
        let mut rng = Pcg64 {
            state: 0,
            inc: (initseq << 1) | 1,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(initstate);
        rng.step();
        rng
    }

    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(PCG_MULTIPLIER)
            .wrapping_add(self.inc);
    }

    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let hi = (self.state >> 64) as u64;
        let lo = self.state as u64;
        (hi ^ lo).rotate_right((self.state >> 122) as u32)
    }
}

/// `numpy.random.Generator` over PCG64, as `default_rng(seed)` builds it.
pub struct Generator {
    bits: Pcg64,
}

impl Generator {
    #[must_use]
    pub fn seeded(seed: u128) -> Generator {
        Generator {
            bits: Pcg64::from_seed_sequence(&SeedSequence::new(seed)),
        }
    }

    /// `integers(low=0, high, size=count, dtype=np.int64)`: `count` draws
    /// uniform on `0..high` by the generator's Lemire rejection sampler.
    /// `high` must exceed 2^32 (the range the 64-bit sampler serves; NumPy
    /// buffers 32-bit draws below it) — Engram's multiplier bound does.
    pub fn integers(&mut self, high: u64, count: usize) -> Vec<u64> {
        assert!(high > 0, "an empty range has no draws");
        let rng = high - 1;
        assert!(
            rng > 0xFFFF_FFFF,
            "a range of {high} is served by NumPy's buffered 32-bit sampler, which this port does not carry"
        );
        (0..count).map(|_| self.bounded_lemire_u64(rng)).collect()
    }

    fn bounded_lemire_u64(&mut self, rng: u64) -> u64 {
        if rng == u64::MAX {
            return self.bits.next_u64();
        }
        let rng_excl = u128::from(rng) + 1;
        let mut m = u128::from(self.bits.next_u64()) * rng_excl;
        let mut leftover = m & u128::from(u64::MAX);
        if leftover < rng_excl {
            let threshold = u128::from(u64::MAX - rng) % rng_excl;
            while leftover < threshold {
                m = u128::from(self.bits.next_u64()) * rng_excl;
                leftover = m & u128::from(u64::MAX);
            }
        }
        (m >> 64) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `numpy.random.default_rng(10007 * layer).integers(0, bound, size=4)`
    /// for DeepSeek-V4.1-Flash's Engram (`bound = (2**63 - 1) // 99092 // 2`),
    /// printed by NumPy 2.4.4.
    #[test]
    fn the_engram_multiplier_draws_are_numpys() {
        let bound = ((i64::MAX as u64) / 99_092 / 2).max(1);
        for (layer, want) in [(1u128, PINNED[0]), (14u128, PINNED[1])] {
            let mut rng = Generator::seeded(10_007 * layer);
            assert_eq!(rng.integers(bound, 4), want, "layer {layer}");
        }
    }

    #[test]
    fn the_first_raw_draws_are_numpys() {
        // numpy.random.PCG64(0).random_raw(3)
        let mut bits = Pcg64::from_seed_sequence(&SeedSequence::new(0));
        assert_eq!(
            [bits.next_u64(), bits.next_u64(), bits.next_u64()],
            RAW_SEED_0
        );
    }

    const PINNED: [[u64; 4]; 2] = [PINNED_L1, PINNED_L14];
    const PINNED_L1: [u64; 4] = [
        38316048023122,
        2419938046656,
        17979836159674,
        36993668729195,
    ];
    const PINNED_L14: [u64; 4] = [
        33858405369630,
        25755403400457,
        15460673601360,
        41309613242795,
    ];
    const RAW_SEED_0: [u64; 3] = [
        11749869230777074271,
        4976686463289251617,
        755828109848996024,
    ];
}
