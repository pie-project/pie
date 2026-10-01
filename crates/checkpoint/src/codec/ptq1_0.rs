//! Host (CPU) decoder for the PTQ1_0 ternary codec (`Dtype::Ptq1_0`,
//! spelled `g128_t3_f16_n`) used by `prism-ml/Ternary-Bonsai-2-27B`.
//!
//! A block is 28 bytes for 128 weights (1.75 bpw): `qs[24]` (5 base-3 trits per
//! byte) + `qh[2]` (4 trits per byte) + an fp16 scale `d` at the END. A weight is
//! `value = (trit - 1) * d`, trit in {0,1,2} -> {-1,0,+1}.
//!
//! The subtle part is the *staging*: which byte and trit-slot stores each element.
//! The fork walks stages `{32,16,8}`; for a 24-byte `qs` the 32-stage is skipped,
//! leaving a 16-byte chunk (block elements 0..79) then an 8-byte chunk (80..119),
//! then `qh` (120..127). This mirrors `dequantize_row_ptq1_0` (PrismML-Eng
//! llama.cpp, branch `prism`, `ggml/src/ggml-quants.c`) exactly, and — like that
//! function — emits the 128 values in NATURAL element order 0..127; the staging
//! only permutes which byte/slot each value is read from. A naive "5 sequential
//! trits per byte" decode lands values in the wrong lanes. The bit-exact test
//! below pins this against the fork's own dequant.

/// Weights per block.
pub const QK_PTQ1_0: usize = 128;
/// Bytes per block: `qs[24]` + `qh[2]` + fp16 `d` = 28.
pub const BLOCK_BYTES: usize = 28;

const QS_LEN: usize = 24;
const QH_LEN: usize = 2;
/// Base-3 place weights `3^n` (as used to shift each trit into the byte MSBs).
/// `qs` uses slots 0..5, `qh` uses slots 0..4 (its first trit is pushed to the
/// MSB at encode time so only four trits are read back).
const POW3: [u16; 5] = [1, 3, 9, 27, 81];
/// Trit-staging chunk sizes. The 32-chunk never fits a 24-byte `qs`, so decoding
/// reduces to a 16-byte chunk then an 8-byte chunk, then the `qh` tail.
const STAGES: [usize; 3] = [32, 16, 8];

/// Extract one base-3 trit from a packed byte at place `n`, exactly as the fork:
/// `q = (byte * 3^n) mod 256; trit = (q * 3) >> 8`, yielding 0, 1 or 2.
#[inline]
fn trit(byte: u8, n: usize) -> i16 {
    let q = u16::from(byte).wrapping_mul(POW3[n]) & 0xff;
    ((q * 3) >> 8) as i16
}

/// Decode one 28-byte `PTQ1_0` block into its 128 f32 weights in natural order.
///
/// Reproduces the fork's `dequantize_row_ptq1_0` byte-for-byte, including the
/// `{16,8}` `qs` staging and the 4-trit `qh` tail. The fp16 scale is converted
/// losslessly (the scale is always finite), matching `GGML_FP16_TO_FP32`.
pub fn decode_block(block: &[u8]) -> [f32; QK_PTQ1_0] {
    assert_eq!(block.len(), BLOCK_BYTES, "a PTQ1_0 block is 28 bytes");
    let qs = &block[..QS_LEN];
    let qh = &block[QS_LEN..QS_LEN + QH_LEN];
    let d = half::f16::from_bits(u16::from_le_bytes([block[26], block[27]])).to_f32();

    let mut out = [0.0f32; QK_PTQ1_0];
    let mut idx = 0usize;

    // qs: walk the {32,16,8} stages. In a `c`-wide chunk starting at byte `j`,
    // trit-slot `n` of byte `j+m` holds one element; the loop order (n outer,
    // m inner) is what emits natural element order.
    let mut j = 0usize;
    for &c in &STAGES {
        while j + c <= QS_LEN {
            for n in 0..5 {
                for m in 0..c {
                    out[idx] = f32::from(trit(qs[j + m], n) - 1) * d;
                    idx += 1;
                }
            }
            j += c;
        }
    }

    // qh tail: elements 120..127, 4 trits per byte.
    for n in 0..4 {
        for &byte in qh {
            out[idx] = f32::from(trit(byte, n) - 1) * d;
            idx += 1;
        }
    }

    debug_assert_eq!(idx, QK_PTQ1_0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Real 28-byte blocks from Ternary-Bonsai-2-27B-PTQ1_0.gguf plus synthetic
    // edge cases, each with its expected f32 produced by the fork's OWN verbatim
    // `dequantize_row_ptq1_0` (and, for the synthetic ones, the fork's verbatim
    // `quantize_row_ptq1_0_ref`). See the module doc for the oracle provenance.
    include!("ptq1_0_fixture.rs");

    /// The whole point of M1a: pie's host decoder must reproduce the fork's own
    /// dequant BIT-EXACT on every real block and every synthetic edge case.
    #[test]
    fn host_decoder_is_bit_exact_against_the_fork_dequant() {
        assert!(
            FORK_BLOCKS.len() >= 20,
            "expected the frozen real+synthetic fixture set"
        );
        let mut real = 0usize;
        let mut synthetic = 0usize;
        for fb in FORK_BLOCKS {
            let got = decode_block(&fb.bytes);
            for (i, (g, want)) in got.iter().zip(fb.expect_bits.iter()).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    *want,
                    "{} block {} element {}: pie {:#010x} ({}) != fork {:#010x}",
                    fb.tensor,
                    fb.block_index,
                    i,
                    g.to_bits(),
                    g,
                    want,
                );
            }
            if fb.tensor.starts_with("synthetic:") {
                synthetic += 1;
            } else {
                real += 1;
            }
        }
        assert!(real >= 12, "want a spread of real blocks, got {real}");
        assert!(synthetic >= 6, "want the synthetic edges, got {synthetic}");
    }

    /// Every decoded weight is exactly one of {-d, 0, +d}: the fork maps trit
    /// {0,1,2} -> {-1,0,+1}. Confirms the value semantics beyond raw bit match.
    #[test]
    fn every_weight_is_minus_d_zero_or_plus_d() {
        for fb in FORK_BLOCKS {
            let d = half::f16::from_bits(u16::from_le_bytes([fb.bytes[26], fb.bytes[27]])).to_f32();
            for v in decode_block(&fb.bytes) {
                assert!(
                    v == 0.0 || v == d || v == -d,
                    "{} block {}: value {v} is not in {{-{d}, 0, {d}}}",
                    fb.tensor,
                    fb.block_index,
                );
            }
        }
    }

    /// The `mod3_lanes` synthetic block was encoded from `x[i] = (i % 3) - 1` in
    /// natural order. If the staging permutation were wrong, this ordered pattern
    /// would be scrambled — so it is a direct check that the decoder emits natural
    /// element order across the 16-byte chunk, the 8-byte chunk, and the qh tail.
    #[test]
    fn natural_order_survives_the_staging() {
        let fb = FORK_BLOCKS
            .iter()
            .find(|b| b.tensor == "synthetic:mod3_lanes")
            .expect("mod3_lanes fixture present");
        let d = half::f16::from_bits(u16::from_le_bytes([fb.bytes[26], fb.bytes[27]])).to_f32();
        let got = decode_block(&fb.bytes);
        for (i, v) in got.iter().enumerate() {
            let want = (i % 3) as f32 - 1.0;
            assert_eq!(*v, want * d, "element {i} broke natural order");
        }
    }

    /// The qh tail (elements 120..127) must be decoded independently of `qs`.
    /// The `qh_boundary` block is all -1 in 0..120 and all +1 in 120..128.
    #[test]
    fn qh_tail_maps_to_elements_120_through_127() {
        let fb = FORK_BLOCKS
            .iter()
            .find(|b| b.tensor == "synthetic:qh_boundary")
            .expect("qh_boundary fixture present");
        let d = half::f16::from_bits(u16::from_le_bytes([fb.bytes[26], fb.bytes[27]])).to_f32();
        let got = decode_block(&fb.bytes);
        for (i, v) in got.iter().enumerate() {
            let want = if i >= 120 { d } else { -d };
            assert_eq!(*v, want, "qh boundary wrong at element {i}");
        }
    }

    /// The block boundary: `two_blocks_boundary` is 256 elements = two blocks with
    /// different sign patterns. Decoding each 28-byte block independently must not
    /// bleed across the 128-element boundary.
    #[test]
    fn two_adjacent_blocks_decode_independently() {
        let blocks: Vec<_> = FORK_BLOCKS
            .iter()
            .filter(|b| b.tensor == "synthetic:two_blocks_boundary")
            .collect();
        assert_eq!(blocks.len(), 2, "two_blocks_boundary should be two blocks");
        assert_eq!(blocks[0].block_index, 0);
        assert_eq!(blocks[1].block_index, 1);
        for fb in blocks {
            let got = decode_block(&fb.bytes);
            for (g, want) in got.iter().zip(fb.expect_bits.iter()) {
                assert_eq!(g.to_bits(), *want);
            }
        }
    }
}
