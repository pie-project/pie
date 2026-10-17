#![cfg(feature = "cuda")]

use crate::common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::layout_embed_e4m3::{embed_concat_e4m3, embed_concat_mxfp8};
use kernels_cuda::tensor::Tensor;

fn e4m3(b: u8) -> f32 {
    let e = i32::from((b >> 3) & 0xF);
    let m = f32::from(b & 7);
    let v = if e == 0 {
        m / 512.0
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    };
    if b & 0x80 != 0 { -v } else { v }
}

#[test]
fn the_e4m3_table_gather_widens_only_its_rows() {
    // 16-byte runs a thread (160), and a code a thread (24).
    gather_at(160);
    gather_at(24);
    // DeepSeek-V4.1's Engram rows: 256 codes under eight exponents.
    mxfp8_gather_at(256);
    mxfp8_gather_at(64);
}

/// An MXFP8 table: each value its code times two to its 32-block's
/// exponent, exact in bf16, an id outside the table zeros.
fn mxfp8_gather_at(width: usize) {
    const VOCAB: usize = 777;
    const HEADS: usize = 24;
    const ROWS: usize = 5;
    let mut lcg = Lcg::seeded(0x8f8 + width as u64);
    let mut byte = || {
        let b = ((lcg.unit() + 1.0) * 127.5) as u8;
        if b & 0x7F == 0x7F { b & 0xFE } else { b }
    };
    let table: Vec<u8> = (0..VOCAB * width).map(|_| byte()).collect();
    let exps: Vec<u8> = (0..VOCAB * width / 32)
        .map(|i| 112 + (i * 13 % 24) as u8)
        .collect();
    let ids: Vec<i32> = (0..ROWS * HEADS)
        .map(|i| match i % 13 {
            4 => -1,
            9 => VOCAB as i32,
            _ => ((i * 89 + 7) % VOCAB) as i32,
        })
        .collect();
    let mut gpu = Gpu::open();
    let ids_at = gpu.up(&ids);
    let table_at = gpu.up(&table);
    let exps_at = gpu.up(&exps);
    let y_at = gpu.zeros(ROWS * HEADS * width * 2);
    let ctx = gpu.ctx();
    let mut y = Tensor::new(y_at, ROWS as u32, (HEADS * width) as u32, Dtype::Bf16);
    embed_concat_mxfp8(
        &ctx,
        Tensor::new(ids_at, ROWS as u32, HEADS as u32, Dtype::I32),
        Tensor::new(table_at, VOCAB as u32, width as u32, Dtype::E4m3),
        Tensor::new(exps_at, VOCAB as u32, (width / 32) as u32, Dtype::E8m0),
        VOCAB as u32,
        &mut y,
    )
    .expect("the gather fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(y_at, ROWS * HEADS * width);
    for (slot, &id) in ids.iter().enumerate() {
        for w in 0..width {
            let want = if id < 0 || id as usize >= VOCAB {
                0.0
            } else {
                let at = id as usize * width + w;
                e4m3(table[at]) * 2f32.powi(i32::from(exps[at / 32]) - 127)
            };
            assert_eq!(
                from_bf16(got[slot * width + w]),
                want,
                "mxfp8 width {width}: slot {slot} (id {id}) column {w}"
            );
        }
    }
}

#[allow(non_snake_case)]
fn gather_at(WIDTH: usize) {
    const VOCAB: usize = 1000;
    const HEADS: usize = 16;
    const ROWS: usize = 7;
    let mut lcg = Lcg::seeded(0x81);
    let mut byte = || {
        let b = ((lcg.unit() + 1.0) * 127.5) as u8;
        if b & 0x7F == 0x7F { b & 0xFE } else { b }
    };
    let table: Vec<u8> = (0..VOCAB * WIDTH).map(|_| byte()).collect();
    let scale = 0.0123f32;
    let ids: Vec<i32> = (0..ROWS * HEADS)
        .map(|i| match i % 11 {
            5 => -1,
            7 => VOCAB as i32,
            _ => ((i * 97 + 13) % VOCAB) as i32,
        })
        .collect();

    let mut gpu = Gpu::open();
    let ids_at = gpu.up(&ids);
    let table_at = gpu.up(&table);
    let scale_at = gpu.up(&[scale]);
    let y_at = gpu.zeros(ROWS * HEADS * WIDTH * 2);
    let ctx = gpu.ctx();
    let mut y = Tensor::new(y_at, ROWS as u32, (HEADS * WIDTH) as u32, Dtype::Bf16);
    embed_concat_e4m3(
        &ctx,
        Tensor::new(ids_at, ROWS as u32, HEADS as u32, Dtype::I32),
        Tensor::new(table_at, VOCAB as u32, WIDTH as u32, Dtype::E4m3tensor),
        Tensor::new(scale_at, 1, 4, Dtype::U8),
        VOCAB as u32,
        &mut y,
    )
    .expect("the gather fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(y_at, ROWS * HEADS * WIDTH);
    for (slot, &id) in ids.iter().enumerate() {
        for w in 0..WIDTH {
            let want = if id < 0 || id as usize >= VOCAB {
                0.0
            } else {
                from_bf16(to_bf16(e4m3(table[id as usize * WIDTH + w]) * scale))
            };
            assert_eq!(
                from_bf16(got[slot * WIDTH + w]),
                want,
                "width {WIDTH}: slot {slot} (id {id}) column {w}"
            );
        }
    }
}
