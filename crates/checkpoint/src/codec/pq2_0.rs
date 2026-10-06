//! Host (CPU) decoder for the PQ2_0 2-bit codec (`Dtype::Pq2_0`, spelled
//! `g128_u2_f16_n`) used by `prism-ml/Ternary-Bonsai-2-27B` (the 2.125-bpw /
//! 7.21 GB sibling of the 1.75-bpw PTQ1_0 file).
//!
//! A block is 34 bytes for 128 weights (2.125 bpw): an fp16 scale `d` at the
//! START (bytes 0-1) + `qs[32]` holding positional 2-bit codes, 4 per byte. A
//! weight is `value = (code - 1) * d`, code in {0,1,2,3} -> {-1,0,+1,+2} (an
//! asymmetric quaternary with a `+2` outlier code, no zero-point).
//!
//! Unlike PTQ1_0 there is NO base-3 staging: element `j` lives in byte `j / 4`
//! at bit offset `(j % 4) * 2`, so the layout is purely positional. This mirrors
//! `dequantize_row_pq2_0` (PrismML-Eng llama.cpp, branch `prism`,
//! `ggml/src/ggml-quants.c`) exactly; the bit-exact test below pins it against
//! the fork's own dequant on real GGUF blocks.

/// Weights per block.
pub const QK_PQ2_0: usize = 128;
/// Bytes per block: fp16 `d` + `qs[32]` = 34.
pub const BLOCK_BYTES: usize = 34;

const QS_LEN: usize = 32;

/// Decode one 34-byte `PQ2_0` block into its 128 f32 weights in natural order.
///
/// Reproduces the fork's `dequantize_row_pq2_0` byte-for-byte: the fp16 scale is
/// read from the first two bytes (converted losslessly, matching
/// `GGML_FP16_TO_FP32`), then each weight `j` is the 2-bit code at byte `j / 4`,
/// bit offset `(j % 4) * 2`, mapped by `(code - 1) * d`.
pub fn decode_block(block: &[u8]) -> [f32; QK_PQ2_0] {
    assert_eq!(block.len(), BLOCK_BYTES, "a PQ2_0 block is 34 bytes");
    let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
    let qs = &block[2..2 + QS_LEN];

    let mut out = [0.0f32; QK_PQ2_0];
    for (j, slot) in out.iter_mut().enumerate() {
        let byte = qs[j / 4];
        let code = (byte >> ((j % 4) * 2)) & 0x03;
        *slot = (f32::from(code) - 1.0) * d;
    }
    out
}

// Real 34-byte blocks from Ternary-Bonsai-2-27B-PQ2_0.gguf plus synthetic edge
// cases, each with its expected f32 produced by the fork's OWN verbatim
// `dequantize_row_pq2_0` (and, for the synthetic ones, the fork's verbatim
// `quantize_row_pq2_0_ref`). See the module doc for the oracle provenance. The
// file is a sibling of this one, so `#[path]` points `mod` at it directly — a
// real declaration the mod-reachability audit can follow (an `include!` is not).
#[cfg(test)]
#[path = "pq2_0_fixture.rs"]
mod pq2_0_fixture;

#[cfg(test)]
mod tests {
    use super::pq2_0_fixture::*;
    use super::*;

    /// The whole point of M2a: pie's host decoder must reproduce the fork's own
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

    /// Every decoded weight is exactly one of {-d, 0, +d, +2d}: the fork maps the
    /// 2-bit code {0,1,2,3} -> {-1,0,+1,+2}. Confirms the asymmetric value
    /// semantics (including the `+2` outlier code) beyond a raw bit match.
    #[test]
    fn every_weight_is_minus_d_zero_plus_d_or_plus_two_d() {
        for fb in FORK_BLOCKS {
            let d = half::f16::from_bits(u16::from_le_bytes([fb.bytes[0], fb.bytes[1]])).to_f32();
            for v in decode_block(&fb.bytes) {
                assert!(
                    v == -d || v == 0.0 || v == d || v == 2.0 * d,
                    "{} block {}: value {v} is not in {{-{d}, 0, {d}, {}}}",
                    fb.tensor,
                    fb.block_index,
                    2.0 * d,
                );
            }
        }
    }

    /// The `positional_lanes` synthetic block was encoded from `code(j) = j % 4`
    /// in natural order. Decoding it must recover the ordered `{-1,0,+1,+2}`
    /// pattern lane for lane — a direct check that element `j` maps to byte `j/4`,
    /// bit `(j%4)*2`, with no staging permutation.
    #[test]
    fn natural_order_is_positional() {
        let fb = FORK_BLOCKS
            .iter()
            .find(|b| b.tensor == "synthetic:positional_lanes")
            .expect("positional_lanes fixture present");
        let d = half::f16::from_bits(u16::from_le_bytes([fb.bytes[0], fb.bytes[1]])).to_f32();
        let got = decode_block(&fb.bytes);
        for (j, v) in got.iter().enumerate() {
            let want = (j % 4) as f32 - 1.0;
            assert_eq!(*v, want * d, "element {j} broke the positional mapping");
        }
    }

    /// The `+2` outlier code (11) must decode to `+2d`, not saturate to `+d`. The
    /// `all_code3` block is every code = 3.
    #[test]
    fn the_plus_two_outlier_code_decodes_to_two_d() {
        let fb = FORK_BLOCKS
            .iter()
            .find(|b| b.tensor == "synthetic:all_code3")
            .expect("all_code3 fixture present");
        let d = half::f16::from_bits(u16::from_le_bytes([fb.bytes[0], fb.bytes[1]])).to_f32();
        for v in decode_block(&fb.bytes) {
            assert_eq!(v, 2.0 * d, "code 3 must map to +2d");
        }
    }

    /// The block boundary: `two_blocks_boundary` is 256 elements = two blocks with
    /// different code patterns. Decoding each 34-byte block independently must not
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
