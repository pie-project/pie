#![cfg(feature = "cuda")]

use crate::common;

use common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::gather;
use kernels_cuda::tensor::Tensor;

const TALL: u32 = 65_536;

#[test]
fn a_qkv_split_reaches_every_row_of_a_tall_fire() {
    // Element by element (a cut not a multiple of 8) and in 16-byte runs.
    split_every_row(8, 4);
    split_every_row(64, 16);
}

fn split_every_row(q_width: u32, kv_width: u32) {
    common::arm_cache();
    let stride = (q_width + 2 * kv_width) as usize;

    let mut lcg = Lcg::seeded(0x51);
    let (packed, _) = lcg.row(TALL as usize * stride);

    let mut gpu = Gpu::open();
    let packed_at = gpu.up(&packed);
    let q_at = gpu.zeros(TALL as usize * q_width as usize * 2);
    let k_at = gpu.zeros(TALL as usize * kv_width as usize * 2);
    let v_at = gpu.zeros(TALL as usize * kv_width as usize * 2);

    let src = Tensor::new(packed_at, TALL, stride as u32, Dtype::Bf16);
    let mut q = Tensor::new(q_at, TALL, q_width, Dtype::Bf16);
    let mut k = Tensor::new(k_at, TALL, kv_width, Dtype::Bf16);
    let mut v = Tensor::new(v_at, TALL, kv_width, Dtype::Bf16);
    gather::split_qkv(&gpu.ctx(), src, q_width, kv_width, &mut q, &mut k, &mut v)
        .expect("a tall split is a split");
    gpu.sync();

    let q_out: Vec<u16> = gpu.down(q_at, TALL as usize * q_width as usize);
    let k_out: Vec<u16> = gpu.down(k_at, TALL as usize * kv_width as usize);
    let v_out: Vec<u16> = gpu.down(v_at, TALL as usize * kv_width as usize);

    let mut first_wrong = None;
    for n in 0..TALL as usize {
        let row = &packed[n * stride..(n + 1) * stride];
        let q_ok =
            q_out[n * q_width as usize..(n + 1) * q_width as usize] == row[..q_width as usize];
        let k_ok = k_out[n * kv_width as usize..(n + 1) * kv_width as usize]
            == row[q_width as usize..(q_width + kv_width) as usize];
        let v_ok = v_out[n * kv_width as usize..(n + 1) * kv_width as usize]
            == row[(q_width + kv_width) as usize..];
        if !(q_ok && k_ok && v_ok) {
            first_wrong = Some(n);
            break;
        }
    }
    assert_eq!(
        first_wrong, None,
        "the {q_width}/{kv_width} split stopped at row {:?} of {TALL}; `gridDim.y` caps at \
         65535, so a split whose rows ride `y` silently drops everything past it",
        first_wrong
    );
}

/// Heads narrower than the cut's (MiMo's 192-wide q and k, 128-wide v)
/// land 256 wide, zero past their widths, on every row.
#[test]
fn a_padded_qkv_split_lays_each_head_out() {
    let (rows, d) = (TALL as usize, 256usize);
    let stride = 2 * 192 + 192 + 128;
    let (packed, _) = Lcg::seeded(0x52).row(rows * stride);
    let mut gpu = Gpu::open();
    let packed_at = gpu.up(&packed);
    let outs = [2usize, 1, 1].map(|h| gpu.up(&vec![0xffffu16; rows * h * d]));
    let t = |at: u64, w: usize| Tensor::new(at, rows as u32, w as u32, Dtype::Bf16);
    let (mut q, mut k, mut v) = (t(outs[0], 2 * d), t(outs[1], d), t(outs[2], d));
    gather::split_qkv_padded(
        &gpu.ctx(),
        t(packed_at, stride),
        [2, 1],
        [192, 128],
        256,
        &mut q,
        &mut k,
        &mut v,
    )
    .expect("the padded split fires");
    gpu.sync();
    // Each output head: (plane, heads in it, head, source offset, width).
    let at = [
        (0, 2, 0, 0, 192),
        (0, 2, 1, 192, 192),
        (1, 1, 0, 384, 192),
        (2, 1, 0, 576, 128),
    ];
    let got: Vec<Vec<u16>> = at
        .iter()
        .map(|&(p, h, ..)| gpu.down(outs[p], rows * h * d))
        .collect();
    for n in 0..rows {
        for (i, &(_, hs, h, from, w)) in at.iter().enumerate() {
            let head = &got[i][(n * hs + h) * d..(n * hs + h + 1) * d];
            let src = &packed[n * stride + from..n * stride + from + w];
            assert!(
                head[..w] == *src && head[w..].iter().all(|&x| x == 0),
                "row {n} head {i}"
            );
        }
    }
}
