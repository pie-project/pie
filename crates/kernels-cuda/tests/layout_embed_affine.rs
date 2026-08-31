//! **THE AFFINE TOKEN EMBEDDING, ON A REAL DEVICE** (qwen4 stored-form
//! wave): `layout::embed_mlx_affine` — the plain `layout.embed` read over a
//! table the store seats as an MLX triplet — held against a host dequant of
//! the rows the ids touch, at both code widths, out-of-vocab ids landing
//! zero.

#![cfg(feature = "_cuda")]

mod common;

use common::{Gpu, Lcg, close, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::layout_embed_concat::embed_mlx_affine;
use kernels_cuda::linear::moe::GroupSeat;
use kernels_cuda::tensor::Tensor;

const VOCAB: u32 = 40;
const WIDTH: u32 = 128;
const GROUP: usize = 64;

fn held(bits: u32) {
    let mut gpu = Gpu::open();
    let mut rng = Lcg::seeded(0xe4bed + u64::from(bits));
    let groups = WIDTH as usize / GROUP;

    let mut codes = Vec::new();
    let mut byte = 0u8;
    for at in 0..(VOCAB as usize * WIDTH as usize) {
        let code = (rng.unit().abs() * 255.0) as u32;
        match bits {
            8 => codes.push(code as u8),
            4 => {
                let nibble = (code & 0xF) as u8;
                if at % 2 == 0 {
                    byte = nibble;
                } else {
                    codes.push(byte | (nibble << 4));
                }
            }
            _ => unreachable!(),
        }
    }
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for _ in 0..(VOCAB as usize * groups) {
        scales.push(to_bf16(rng.unit() * 0.05));
        biases.push(to_bf16(rng.unit()));
    }

    let ids: Vec<i32> = vec![3, 0, 39, -1, 40, 17];
    let rows = ids.len() as u32;

    let ids_t = Tensor::new(gpu.up(&ids), rows, 1, Dtype::I32);
    let codes_width = if bits == 8 { WIDTH } else { WIDTH / 2 };
    let codes_t = Tensor::new(gpu.up(&codes), VOCAB, codes_width, Dtype::U8);
    let scales_t = Tensor::new(gpu.up(&scales), VOCAB, groups as u32 * 2, Dtype::U8);
    let biases_t = Tensor::new(gpu.up(&biases), VOCAB, groups as u32 * 2, Dtype::U8);
    let y_at = gpu.zeros(rows as usize * WIDTH as usize * 2);
    let mut y = Tensor::new(y_at, rows, WIDTH, Dtype::Bf16);

    embed_mlx_affine(
        &gpu.ctx(),
        ids_t,
        codes_t,
        scales_t,
        Some(biases_t),
        VOCAB,
        GroupSeat::RESIDENT,
        &mut y,
    )
    .expect("the affine gather fires");
    gpu.sync();

    let got: Vec<u16> = gpu.down(y_at, rows as usize * WIDTH as usize);
    for (r, id) in ids.iter().enumerate() {
        for w in 0..WIDTH as usize {
            let want = if *id < 0 || *id >= VOCAB as i32 {
                0.0
            } else {
                let at = *id as usize * WIDTH as usize + w;
                let code = match bits {
                    8 => f32::from(codes[at]),
                    4 => {
                        let b = codes[at / 2];
                        f32::from(if at % 2 == 0 { b & 0xF } else { b >> 4 })
                    }
                    _ => unreachable!(),
                };
                let fx = *id as usize * groups + w / GROUP;
                code * from_bf16(scales[fx]) + from_bf16(biases[fx])
            };
            let g = from_bf16(got[r * WIDTH as usize + w]);
            assert!(
                close(g, want),
                "y[{r}][{w}] answered {g} where the host dequant says {want} (bits {bits})"
            );
        }
    }
}

#[test]
fn the_eight_bit_gather_matches_the_host_dequant() {
    held(8);
}

#[test]
fn the_four_bit_gather_matches_the_host_dequant() {
    held(4);
}
