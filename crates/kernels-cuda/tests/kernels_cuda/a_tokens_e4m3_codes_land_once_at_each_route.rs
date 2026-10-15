#![cfg(feature = "cuda")]

//! e4m3 activations quantized for a token's routes (`div` rows reading each
//! token, landed at `pos` rows of a grouped layout): each token's blocks are
//! quantized once and landed at each of its routes' rows, the same codes and
//! scales as quantizing each route's row on its own (the token's row
//! repeated `div` times, `div` one).

use crate::common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::linear::act_quant::{Format, Source, quantize};
use kernels_cuda::linear::tc5::atom_bytes;
use kernels_cuda::tensor::Tensor;

const OP: &str = "linear.act_quant";

#[test]
fn a_tokens_e4m3_codes_land_once_at_each_route() {
    for (tokens, div, k, gran) in [
        (37usize, 8usize, 512usize, 128u32),
        (5, 4, 256, 32),
        (130, 6, 384, 128),
    ] {
        let mut gpu = Gpu::open();
        let mut lcg = Lcg::seeded(0xe4 ^ tokens as u64);
        let (raw, _) = lcg.row(tokens * k);
        let rows = tokens * div;
        // Each route's row: a permutation of the rows, every seventh route
        // dropped.
        let pos: Vec<i32> = (0..rows)
            .map(|r| {
                if r % 7 == 3 {
                    -1
                } else {
                    ((r * 101 + 11) % rows) as i32
                }
            })
            .collect();
        let repeated: Vec<u16> = (0..rows)
            .flat_map(|r| raw[r / div * k..(r / div + 1) * k].iter().copied())
            .collect();
        let x = Tensor::new(gpu.up(&raw), tokens as u32, k as u32, Dtype::Bf16);
        let xr = Tensor::new(gpu.up(&repeated), rows as u32, k as u32, Dtype::Bf16);
        let pos_at = gpu.up(&pos);
        let bytes = atom_bytes(rows, k, gran as usize);
        let (fan_codes, fan_atoms) = (gpu.zeros(rows * k), gpu.zeros(bytes));
        let (one_codes, one_atoms) = (gpu.zeros(rows * k), gpu.zeros(bytes));
        let ctx = gpu.ctx();
        let fmt = Format::E4m3 { gran };
        let src = |x: Tensor, div: usize| Source {
            x,
            rows: rows as u32,
            div: div as u32,
            routes: None,
            input_scales: None,
            pos: pos_at,
            m_max: 0,
        };
        quantize(&ctx, OP, fmt, src(x, div), fan_codes, fan_atoms, bytes).expect("fans out");
        quantize(&ctx, OP, fmt, src(xr, 1), one_codes, one_atoms, bytes).expect("fires");
        gpu.sync();
        assert_eq!(
            gpu.down::<u8>(fan_codes, rows * k),
            gpu.down::<u8>(one_codes, rows * k),
            "{tokens} tokens of {div} routes: the codes"
        );
        assert_eq!(
            gpu.down::<u8>(fan_atoms, bytes),
            gpu.down::<u8>(one_atoms, bytes),
            "{tokens} tokens of {div} routes: the scales"
        );
    }
}
